"""B2 Reaction-Diffusion renderer for ECHO-Bench (Task A-003).

Produces a deterministic 64x64 single-channel ``uint8`` raster by running a
Gray-Scott reaction-diffusion simulation. The result is a pure function of
``(seed, params, rendererVersion)``:

- A local :class:`numpy.random.Generator` (seeded from a stable
  :func:`canonical_hash`) seeds the initial perturbation only. The global RNG is
  never touched; no wall-clock / process state enters the output.
- Iteration count and diffusion / feed / kill rates come from bounded params.
- Iterations are capped to keep the smoke run CPU-fast.
- Identical inputs yield a byte-identical array (identical ``raster_hash``).

Bounded params consumed (B2): ``feed_rate``, ``kill_rate``, ``diffusion_u``,
``iterations``.

No semantic / user / persona field is produced anywhere.
All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from echo_bench.utils.hash import canonical_hash

RASTER_SIZE = 64
# Hard cap on simulation steps; keeps the CPU cost bounded regardless of params.
MAX_ITERATIONS = 400


def _derive_seed_int(seed: int, params: Mapping[str, float],
                     renderer_version: str) -> int:
    """Derive a stable 63-bit integer seed for the local numpy Generator."""
    digest = canonical_hash(
        {
            "basis": "B2",
            "seed": int(seed),
            "params": dict(params),
            "rendererVersion": renderer_version,
        }
    )
    return int(digest, 16) % (2 ** 63)


def _laplacian(z: np.ndarray) -> np.ndarray:
    """5-point Laplacian with toroidal (wrap) boundaries — fully vectorized."""
    return (
        np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
        - 4.0 * z
    )


def render(seed: int, params: Mapping[str, float],
           rendererVersion: str) -> np.ndarray:
    """Render the B2 reaction-diffusion raster as a 64x64 ``uint8`` array.

    Args:
        seed: Card seed (int).
        params: Bounded B2 params (``feed_rate``, ``kill_rate``,
            ``diffusion_u``, ``iterations``).
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

    feed = float(params.get("feed_rate", 0.045))
    kill = float(params.get("kill_rate", 0.062))
    du = float(params.get("diffusion_u", 0.16))
    dv = du * 0.5  # v diffuses slower (standard Gray-Scott ratio)
    iters = int(round(float(params.get("iterations", 200))))
    iters = max(20, min(MAX_ITERATIONS, iters))

    u = np.ones((n, n), dtype=np.float64)
    v = np.zeros((n, n), dtype=np.float64)

    # Seed a few deterministic square perturbations of v (the activator).
    n_seeds = 4
    for _ in range(n_seeds):
        cy = int(rng.integers(8, n - 8))
        cx = int(rng.integers(8, n - 8))
        r = int(rng.integers(2, 5))
        u[cy - r:cy + r, cx - r:cx + r] = 0.50
        v[cy - r:cy + r, cx - r:cx + r] = 0.25

    for _ in range(iters):
        lu = _laplacian(u)
        lv = _laplacian(v)
        uvv = u * v * v
        u += du * lu - uvv + feed * (1.0 - u)
        v += dv * lv + uvv - (kill + feed) * v
        np.clip(u, 0.0, 1.0, out=u)
        np.clip(v, 0.0, 1.0, out=v)

    # Map the activator concentration to grayscale, normalized for contrast.
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin > 1e-9:
        norm = (v - vmin) / (vmax - vmin)
    else:
        norm = v
    img = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return img
