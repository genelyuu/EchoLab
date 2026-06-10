"""B4 Flow-Attractor renderer for ECHO-Bench (Task A-005).

Produces a deterministic 64x64 single-channel ``uint8`` raster by integrating a
2D flow field / de Jong-style attractor with a fixed-step deterministic
integrator and accumulating visited cells. The result is a pure function of
``(seed, params, rendererVersion)``:

- A local :class:`numpy.random.Generator` (seeded from a stable
  :func:`canonical_hash`) only chooses deterministic start offsets. The global
  RNG is never touched; no wall-clock / process state enters the output.
- Attractor coefficients, step count, and step size come from bounded params.
- Identical inputs yield a byte-identical array (identical ``raster_hash``).

Bounded params consumed (B4): ``coeff_a``, ``coeff_b``, ``steps``, ``step_size``.

No semantic / user / persona field is produced anywhere.
All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from echo_bench.utils.hash import canonical_hash

RASTER_SIZE = 64
MAX_STEPS = 1200


def _derive_seed_int(seed: int, params: Mapping[str, float],
                     renderer_version: str) -> int:
    """Derive a stable 63-bit integer seed for the local numpy Generator."""
    digest = canonical_hash(
        {
            "basis": "B4",
            "seed": int(seed),
            "params": dict(params),
            "rendererVersion": renderer_version,
        }
    )
    return int(digest, 16) % (2 ** 63)


def render(seed: int, params: Mapping[str, float],
           rendererVersion: str) -> np.ndarray:
    """Render the B4 flow-attractor raster as a 64x64 ``uint8`` array.

    Args:
        seed: Card seed (int).
        params: Bounded B4 params (``coeff_a``, ``coeff_b``, ``steps``,
            ``step_size``).
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

    a = float(params.get("coeff_a", 1.4))
    b = float(params.get("coeff_b", -1.0))
    steps = int(round(float(params.get("steps", 600))))
    steps = max(100, min(MAX_STEPS, steps))
    step_size = float(params.get("step_size", 0.02))
    step_size = max(0.005, min(0.05, step_size))

    # Two extra coefficients are derived deterministically from the bounded
    # params (a, b, step_size) so the de Jong flow map is fully determined by
    # the config — no hidden free constants. step_size widens the coefficient
    # spread, acting as the integrator's fixed step scale.
    spread = 1.0 + step_size * 20.0  # in [1.1, 2.0]
    c = (a + math.sin(b * 1.7)) * spread
    d = (b + math.cos(a * 1.3)) * spread

    acc = np.zeros((n, n), dtype=np.float64)

    # A few deterministic trajectories with seeded start offsets. Each is a
    # fixed-step iteration of the de Jong attractor flow map.
    n_traj = 6
    for _ in range(n_traj):
        x = float(rng.uniform(-0.5, 0.5))
        y = float(rng.uniform(-0.5, 0.5))
        for _ in range(steps):
            xn = math.sin(a * y) - math.cos(b * x)
            yn = math.sin(c * x) - math.cos(d * y)
            x, y = xn, yn
            # Map attractor space (~[-2, 2]) into the raster.
            px = int((x + 2.0) / 4.0 * (n - 1))
            py = int((y + 2.0) / 4.0 * (n - 1))
            if 0 <= px < n and 0 <= py < n:
                acc[py, px] += 1.0

    # Log-compress density to expose fine structure, then normalize.
    acc = np.log1p(acc)
    amax = float(acc.max())
    if amax > 1e-9:
        acc = acc / amax
    img = (acc * 255.0).clip(0, 255).astype(np.uint8)
    return img
