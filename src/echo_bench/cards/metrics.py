"""Visual metric extraction and card assembly for ECHO-Bench (Task A-006).

Phase 2 REAL metrics over a 64x64 single-channel ``uint8`` raster produced by
the real per-basis renderers (:mod:`echo_bench.basis.renderers`). Every metric
is a deterministic, pure function of the raster array (plus, where noted, the
bounded params) — no semantic labels, no external data, no user/persona/emotion
field. Identical rasters always yield identical metrics.

``visualMetrics`` carries real image statistics (mean / std intensity, edge
density via gradient magnitude, spatial frequency via a 2D-FFT energy ratio,
occupancy, dynamic range, saturation fraction). ``coordinateContribution`` is a
fixed-length (4) low-dimensional, normalized [0, 1] projection of the raster.
``complexityScore`` and ``salienceScore`` are normalized scalars; the band is
derived from ``complexityScore`` via the documented :data:`COMPLEXITY_BANDS`
thresholds.

For backward compatibility, :func:`extract` also accepts a flat ``bytes`` raster
of length ``64*64`` and reshapes it.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np

from echo_bench.basis.renderers import get_renderer
from echo_bench.basis.schema import BasisSpec, sample_params
from echo_bench.cards.schema import Card
from echo_bench.utils.hash import canonical_hash, raster_hash

# Documented complexity-band thresholds over complexityScore in [0, 1].
# Bands are half-open on the upper side except the last: [lo, hi). Band names are
# unchanged from Phase 1 (low / mid / high).
COMPLEXITY_BANDS = (
    ("low", 0.0, 0.34),
    ("mid", 0.34, 0.67),
    ("high", 0.67, 1.0001),
)

# Number of coordinate-contribution channels emitted. These feed the coverage
# metrics in D-001; they are plain structural projections, not semantic axes.
COORDINATE_DIMS = 4

# Expected raster geometry (single channel, 64x64).
RASTER_SIDE = 64
RASTER_SIZE = RASTER_SIDE * RASTER_SIDE

_ROUND = 12


def complexity_band(score: float) -> str:
    """Map a ``complexityScore`` in [0, 1] to a documented band label."""
    for name, lo, hi in COMPLEXITY_BANDS:
        if lo <= score < hi:
            return name
    return COMPLEXITY_BANDS[0][0] if score < 0 else COMPLEXITY_BANDS[-1][0]


def _as_array(raster: Any) -> np.ndarray:
    """Coerce ``raster`` to a 2D ``uint8`` numpy array.

    Accepts a 2D uint8 array directly, or a flat ``bytes``/``bytearray`` of
    length ``RASTER_SIZE`` (reshaped to ``RASTER_SIDE x RASTER_SIDE``), or a 1D
    array of length ``RASTER_SIZE``.
    """
    if isinstance(raster, np.ndarray):
        arr = raster
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 1:
            if arr.size != RASTER_SIZE:
                raise ValueError(
                    f"extract: 1D raster length {arr.size} != {RASTER_SIZE}"
                )
            arr = arr.reshape(RASTER_SIDE, RASTER_SIDE)
        elif arr.ndim != 2:
            raise ValueError(f"extract: raster must be 1D/2D, got ndim={arr.ndim}")
        return arr
    if isinstance(raster, (bytes, bytearray)):
        data = bytes(raster)
        if len(data) == 0:
            raise ValueError("extract: empty raster")
        if len(data) != RASTER_SIZE:
            raise ValueError(
                f"extract: byte raster length {len(data)} != {RASTER_SIZE}"
            )
        return np.frombuffer(data, dtype=np.uint8).reshape(RASTER_SIDE, RASTER_SIDE)
    raise ValueError("extract: raster must be a numpy uint8 array or bytes")


def _spatial_frequency(norm: np.ndarray) -> float:
    """High-frequency energy ratio via the 2D FFT magnitude spectrum.

    Returns the fraction of total spectral energy (excluding the DC term) that
    lies in the outer/high-frequency half of the spectrum. Deterministic and in
    [0, 1].
    """
    spec = np.abs(np.fft.fftshift(np.fft.fft2(norm))) ** 2
    cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
    spec[cy, cx] = 0.0  # drop DC
    total = float(spec.sum())
    if total <= 0.0:
        return 0.0
    yy, xx = np.indices(spec.shape)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(radius.max())
    if rmax <= 0.0:
        return 0.0
    high = float(spec[radius > rmax * 0.5].sum())
    return max(0.0, min(1.0, high / total))


def extract(raster: Any, basis: str,
            params: Mapping[str, float]) -> dict[str, Any]:
    """Extract deterministic visual metrics from a raster.

    Args:
        raster: A 64x64 ``uint8`` numpy array (single channel) or, for
            back-compat, a flat ``bytes`` raster of length ``64*64``.
        basis: Basis id (recorded only; not used to fabricate semantics).
        params: Bounded params used to render (recorded context only; the
            metrics themselves are pure functions of the raster).

    Returns:
        A dict with keys ``visualMetrics`` (dict[str, float]),
        ``coordinateContribution`` (tuple of :data:`COORDINATE_DIMS` floats in
        [0, 1]), ``complexityScore`` (float in [0, 1]), ``complexityBand``
        (str), ``salienceScore`` (float in [0, 1]). No semantic labels.
    """
    arr = _as_array(raster)
    f = arr.astype(np.float64)
    norm = f / 255.0

    mean = float(norm.mean())
    std = float(norm.std())
    minv = float(norm.min())
    maxv = float(norm.max())
    dynamic_range = maxv - minv

    # Gradient magnitude (edge density), normalized by the max possible step.
    gy, gx = np.gradient(norm)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float(min(1.0, grad_mag.mean() * 2.0))

    # Occupancy: fraction of pixels meaningfully above the floor.
    occupancy = float((f > 8.0).mean())

    # Saturation: fraction pinned to the extremes (blank / saturated indicator).
    saturation = float(((f <= 1.0) | (f >= 254.0)).mean())

    # Normalized Shannon entropy over a 32-bin intensity histogram.
    hist, _ = np.histogram(f, bins=32, range=(0.0, 255.0))
    total = float(hist.sum())
    if total > 0:
        p = hist[hist > 0] / total
        entropy = float(-(p * np.log2(p)).sum() / np.log2(32))
    else:
        entropy = 0.0

    spatial_freq = _spatial_frequency(norm)

    visual_metrics: dict[str, float] = {
        "meanIntensity": round(mean, _ROUND),
        "stdIntensity": round(std, _ROUND),
        "dynamicRange": round(dynamic_range, _ROUND),
        "edgeDensity": round(edge_density, _ROUND),
        "spatialFrequency": round(spatial_freq, _ROUND),
        "occupancy": round(occupancy, _ROUND),
        "saturationFraction": round(saturation, _ROUND),
        "entropy": round(entropy, _ROUND),
    }

    # complexityScore: blend of entropy, edge activity, and spatial frequency.
    complexity = 0.45 * entropy + 0.35 * edge_density + 0.20 * spatial_freq
    complexity = max(0.0, min(1.0, complexity))
    band = complexity_band(complexity)

    # salienceScore: peak-region contrast — how much the brightest region stands
    # out from the global mean — combined with std, penalized by saturation.
    thresh = mean + std
    peak_mask = norm >= thresh
    if peak_mask.any():
        peak_contrast = float(norm[peak_mask].mean() - mean)
    else:
        peak_contrast = 0.0
    salience = 0.6 * std + 0.4 * peak_contrast
    salience *= (1.0 - 0.5 * saturation)
    salience = max(0.0, min(1.0, salience))

    # coordinateContribution: deterministic low-dim projection — the four
    # quadrant mean intensities, each normalized to [0, 1]. Purely positional.
    h, w = arr.shape
    hy, hx = h // 2, w // 2
    quadrants = (
        norm[:hy, :hx],
        norm[:hy, hx:],
        norm[hy:, :hx],
        norm[hy:, hx:],
    )
    coords = [
        round(float(q.mean()) if q.size else 0.0, _ROUND) for q in quadrants
    ]

    return {
        "visualMetrics": visual_metrics,
        "coordinateContribution": tuple(coords),
        "complexityScore": round(complexity, _ROUND),
        "complexityBand": band,
        "salienceScore": round(salience, _ROUND),
    }


def _as_rng(rng_or_seed: Any) -> random.Random:
    """Coerce a seed-or-RNG argument into a :class:`random.Random`."""
    if isinstance(rng_or_seed, random.Random):
        return rng_or_seed
    if isinstance(rng_or_seed, int) and not isinstance(rng_or_seed, bool):
        return random.Random(rng_or_seed)
    if hasattr(rng_or_seed, "uniform"):
        return rng_or_seed  # numpy Generator or compatible
    raise ValueError("build_card: rng_or_seed must be a random.Random or int seed")


def build_card(spec: BasisSpec, basis: str, seed: int,
               rng_or_seed: Any) -> Card:
    """Assemble a full :class:`Card` from a basis spec and seed.

    Pipeline: sample params -> render (real per-basis renderer) -> raster_hash
    -> extract metrics -> assemble Card. ``cardId`` is the :func:`canonical_hash`
    of the card's identifying fields. The 11-field Card schema is unchanged.

    Args:
        spec: The :class:`BasisSpec` for ``basis``.
        basis: Basis id; must equal ``spec.basis``.
        seed: Card seed (int).
        rng_or_seed: A seeded :class:`random.Random`, a numpy ``Generator``,
            or an int seed used to draw the params.

    Returns:
        A frozen :class:`Card` with renderHash and rendererVersion populated.
    """
    if basis != spec.basis:
        raise ValueError(
            f"build_card: basis {basis!r} != spec.basis {spec.basis!r}"
        )
    rng = _as_rng(rng_or_seed)
    params = sample_params(spec, rng)
    renderer_version = spec.rendererVersion

    renderer = get_renderer(basis)
    arr = renderer(int(seed), params, renderer_version)
    render_hash = raster_hash(arr)
    metrics = extract(arr, basis, params)

    identifying = {
        "basis": basis,
        "seed": int(seed),
        "params": params,
        "renderHash": render_hash,
        "rendererVersion": renderer_version,
    }
    card_id = canonical_hash(identifying)

    return Card(
        cardId=card_id,
        basis=basis,
        seed=int(seed),
        params=params,
        visualMetrics=metrics["visualMetrics"],
        coordinateContribution=metrics["coordinateContribution"],
        complexityScore=metrics["complexityScore"],
        complexityBand=metrics["complexityBand"],
        salienceScore=metrics["salienceScore"],
        renderHash=render_hash,
        rendererVersion=renderer_version,
    )
