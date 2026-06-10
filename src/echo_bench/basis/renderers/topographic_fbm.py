"""B3 Topographic fBM renderer for ECHO-Bench (Task A-004).

Produces a deterministic 64x64 single-channel ``uint8`` raster via fractional
Brownian motion built from seeded value-noise octaves. The result is a pure
function of ``(seed, params, rendererVersion)``:

- A local :class:`numpy.random.Generator` (seeded from a stable
  :func:`canonical_hash`) seeds the value-noise lattice. The global RNG is never
  touched; no wall-clock / process state enters the output.
- Octave count, lacunarity, gain, and scale are bounded params.
- Identical inputs yield a byte-identical array (identical ``raster_hash``).

Bounded params consumed (B3): ``octaves``, ``lacunarity``, ``gain``, ``scale``.

No semantic / user / persona field is produced anywhere.
All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from echo_bench.utils.hash import canonical_hash

RASTER_SIZE = 64
MAX_OCTAVES = 8


def _derive_seed_int(seed: int, params: Mapping[str, float],
                     renderer_version: str) -> int:
    """Derive a stable 63-bit integer seed for the local numpy Generator."""
    digest = canonical_hash(
        {
            "basis": "B3",
            "seed": int(seed),
            "params": dict(params),
            "rendererVersion": renderer_version,
        }
    )
    return int(digest, 16) % (2 ** 63)


def _value_noise(rng: np.random.Generator, n: int, freq: float) -> np.ndarray:
    """Bilinearly-interpolated value noise on an ``n x n`` grid.

    A lattice of random values at integer grid points (cells = ``freq``) is
    sampled and bilinearly interpolated with a smoothstep fade. Deterministic
    given ``rng``; no global state.
    """
    cells = max(1, int(round(freq)))
    lattice = rng.random((cells + 2, cells + 2))

    coords = np.linspace(0.0, cells, n, endpoint=False)
    gx = coords[None, :].repeat(n, axis=0)
    gy = coords[:, None].repeat(n, axis=1)

    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    tx = gx - x0
    ty = gy - y0
    # Smoothstep fade for C1 continuity.
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)

    v00 = lattice[y0, x0]
    v10 = lattice[y0, x1]
    v01 = lattice[y1, x0]
    v11 = lattice[y1, x1]
    top = v00 * (1.0 - sx) + v10 * sx
    bot = v01 * (1.0 - sx) + v11 * sx
    return top * (1.0 - sy) + bot * sy


def render(seed: int, params: Mapping[str, float],
           rendererVersion: str) -> np.ndarray:
    """Render the B3 topographic fBM raster as a 64x64 ``uint8`` array.

    Args:
        seed: Card seed (int).
        params: Bounded B3 params (``octaves``, ``lacunarity``, ``gain``,
            ``scale``).
        rendererVersion: Renderer version string; changing it changes output.

    Returns:
        ``np.ndarray`` of shape ``(64, 64)`` and dtype ``uint8``.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("render: seed must be an int")
    if not isinstance(rendererVersion, str) or not rendererVersion:
        raise ValueError("render: rendererVersion must be a non-empty str")

    rng = np.random.default_rng(_derive_seed_int(seed, params, rendererVersion))
    n = RASTER_SIZE

    octaves = int(round(float(params.get("octaves", 5))))
    octaves = max(1, min(MAX_OCTAVES, octaves))
    lacunarity = float(params.get("lacunarity", 2.0))
    lacunarity = max(1.1, min(3.5, lacunarity))
    gain = float(params.get("gain", 0.5))
    gain = max(0.2, min(0.8, gain))
    scale = float(params.get("scale", 4.0))
    scale = max(1.0, min(8.0, scale))

    total = np.zeros((n, n), dtype=np.float64)
    amplitude = 1.0
    freq = scale
    amp_sum = 0.0
    for _ in range(octaves):
        total += amplitude * _value_noise(rng, n, freq)
        amp_sum += amplitude
        amplitude *= gain
        freq *= lacunarity

    if amp_sum > 0:
        total /= amp_sum
    tmin, tmax = float(total.min()), float(total.max())
    if tmax - tmin > 1e-9:
        total = (total - tmin) / (tmax - tmin)
    img = (total * 255.0).clip(0, 255).astype(np.uint8)
    return img
