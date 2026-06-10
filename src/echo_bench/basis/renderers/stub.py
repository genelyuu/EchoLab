"""Phase 1 STUB renderer for ECHO-Bench.

Phase 1 STUB renderer; real B1-B4 renderers land in Phase 2 (A-002..A-005).

This module produces a deterministic small raster (a flat ``bytes`` buffer)
purely from ``(basis, seed, params, rendererVersion)``. It exists so the rest
of the Phase 1 pipeline (metrics, filter, archive, hashing) can run end-to-end
and be tested for reproducibility before the real procedural renderers exist.

Determinism rules (mirror the renderer acceptance criteria in tasks/A_*):
- Output is a pure function of ``(basis, seed, params, rendererVersion)``.
- A SEEDED LOCAL :class:`random.Random` instance drives every draw; the global
  RNG is never touched and no wall-clock / process state enters the output.
- Identical inputs yield byte-identical output (hence identical ``renderHash``).

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from typing import Mapping

from echo_bench.utils.hash import canonical_hash

# Fixed raster geometry for the stub. Small enough to keep smoke runs CPU-fast
# while still giving the metric extractor a non-trivial byte distribution.
RASTER_WIDTH = 32
RASTER_HEIGHT = 32
RASTER_SIZE = RASTER_WIDTH * RASTER_HEIGHT  # one byte (0..255) per pixel


def _derive_seed(basis: str, seed: int, params: Mapping[str, float],
                 renderer_version: str) -> int:
    """Derive a deterministic integer RNG seed from the inputs.

    Uses :func:`canonical_hash` so the seed is stable across processes and
    platforms and changes whenever any input (including ``rendererVersion``)
    changes.
    """
    digest = canonical_hash(
        {
            "basis": basis,
            "seed": int(seed),
            "params": dict(params),
            "rendererVersion": renderer_version,
        }
    )
    # Fold the hex digest into a 64-bit int seed for random.Random.
    return int(digest[:16], 16)


def render(basis: str, seed: int, params: Mapping[str, float],
           rendererVersion: str) -> bytes:
    """Render a deterministic stub raster as a flat ``bytes`` buffer.

    Phase 1 STUB renderer; real B1-B4 renderers land in Phase 2 (A-002..A-005).

    Args:
        basis: Basis id (``B1``..``B4``). Mixed into the derived seed so
            different bases produce different rasters.
        seed: Card seed (int).
        params: Sampled bounded params for the basis. Different params produce
            different rasters.
        rendererVersion: Renderer version string; changing it changes output.

    Returns:
        A ``bytes`` object of length :data:`RASTER_SIZE`, one grayscale byte per
        pixel. Same inputs always return identical bytes.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("render: seed must be an int")
    if not isinstance(rendererVersion, str) or not rendererVersion:
        raise ValueError("render: rendererVersion must be a non-empty str")

    local_seed = _derive_seed(basis, seed, params, rendererVersion)
    rng = random.Random(local_seed)  # SEEDED LOCAL RNG — never the global RNG

    # Build a deterministic per-pixel byte stream. A slowly varying base level
    # derived from params plus seeded noise gives a non-degenerate distribution
    # for the metric extractor while staying fully reproducible.
    if params:
        param_mean = sum(float(v) for v in params.values()) / len(params)
    else:
        param_mean = 0.0
    base_level = int(abs(param_mean) * 7.0 + (local_seed % 97)) % 256

    buf = bytearray(RASTER_SIZE)
    level = base_level
    for i in range(RASTER_SIZE):
        # Deterministic random-walk over the byte range, seeded locally.
        level = (level + rng.randint(-12, 12)) % 256
        buf[i] = level
    return bytes(buf)
