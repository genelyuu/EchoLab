"""B1 Branching renderer for ECHO-Bench (Task A-002).

Produces a deterministic 64x64 single-channel ``uint8`` raster by rasterizing a
recursive, L-system-like set of branching strokes. The structure is a pure
function of ``(seed, params, rendererVersion)``:

- The local RNG is a :class:`numpy.random.Generator` seeded from a stable
  :func:`canonical_hash` of the inputs. The global RNG is never touched and no
  wall-clock / process state enters the output.
- Identical inputs yield a byte-identical array (hence identical ``raster_hash``).

Bounded params consumed (from ``configs/basis/bases.yaml`` B1):
    ``depth`` (recursion depth), ``branch_angle`` (degrees), ``branch_ratio``
    (child/parent length ratio), ``jitter`` (angular noise amount).

No semantic / user / persona field is produced anywhere.
All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from echo_bench.utils.hash import canonical_hash

# Fixed raster geometry. Single channel grayscale, one uint8 per pixel.
RASTER_SIZE = 64


def _derive_seed_int(seed: int, params: Mapping[str, float],
                     renderer_version: str) -> int:
    """Derive a stable 63-bit integer seed for the local numpy Generator."""
    digest = canonical_hash(
        {
            "basis": "B1",
            "seed": int(seed),
            "params": dict(params),
            "rendererVersion": renderer_version,
        }
    )
    return int(digest, 16) % (2 ** 63)


def _draw_line(acc: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               weight: float) -> None:
    """Accumulate a thin antialiased-ish line into ``acc`` (float buffer).

    Deterministic integer-step DDA; clamps to bounds. No RNG, no I/O.
    """
    n = RASTER_SIZE
    steps = int(max(abs(x1 - x0), abs(y1 - y0)) * 2.0) + 1
    for s in range(steps + 1):
        t = s / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < n and 0 <= yi < n:
            acc[yi, xi] += weight
            # Light cross spread for stroke thickness (still deterministic).
            if xi + 1 < n:
                acc[yi, xi + 1] += weight * 0.4
            if yi + 1 < n:
                acc[yi + 1, xi] += weight * 0.4


def render(seed: int, params: Mapping[str, float],
           rendererVersion: str) -> np.ndarray:
    """Render the B1 branching raster as a 64x64 ``uint8`` numpy array.

    Args:
        seed: Card seed (int).
        params: Bounded B1 params (``depth``, ``branch_angle``,
            ``branch_ratio``, ``jitter``).
        rendererVersion: Renderer version string; changing it changes output.

    Returns:
        ``np.ndarray`` of shape ``(64, 64)`` and dtype ``uint8``. Same inputs
        always return a byte-identical array.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("render: seed must be an int")
    if not isinstance(rendererVersion, str) or not rendererVersion:
        raise ValueError("render: rendererVersion must be a non-empty str")

    rng = np.random.default_rng(_derive_seed_int(seed, params, rendererVersion))
    n = RASTER_SIZE

    depth = int(round(float(params.get("depth", 5))))
    depth = max(2, min(10, depth))
    angle = math.radians(float(params.get("branch_angle", 30.0)))
    ratio = float(params.get("branch_ratio", 0.6))
    ratio = min(0.85, max(0.4, ratio))
    jitter = float(params.get("jitter", 0.2))

    acc = np.zeros((n, n), dtype=np.float64)

    # Iterative branch expansion via an explicit stack: (x, y, heading, length,
    # depth_left). Deterministic order; RNG only perturbs angle by `jitter`.
    init_len = n * 0.30
    stack = [(n / 2.0, n - 2.0, -math.pi / 2.0, init_len, depth)]
    while stack:
        x, y, heading, length, d = stack.pop()
        if d <= 0 or length < 1.0:
            continue
        x2 = x + math.cos(heading) * length
        y2 = y + math.sin(heading) * length
        _draw_line(acc, x, y, x2, y2, weight=0.5 + 0.5 * (d / depth))
        jl = float(rng.uniform(-jitter, jitter))
        jr = float(rng.uniform(-jitter, jitter))
        stack.append((x2, y2, heading - angle + jl, length * ratio, d - 1))
        stack.append((x2, y2, heading + angle + jr, length * ratio, d - 1))

    if acc.max() > 0:
        acc = acc / acc.max()
    img = (acc * 255.0).clip(0, 255).astype(np.uint8)
    return img
