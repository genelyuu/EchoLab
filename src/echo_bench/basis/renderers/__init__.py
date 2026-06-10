"""Renderer subpackage for ECHO-Bench bases.

Phase 2 ships four real deterministic CPU renderers, one per basis:

- ``B1`` -> :mod:`echo_bench.basis.renderers.branching`
- ``B2`` -> :mod:`echo_bench.basis.renderers.reaction_diffusion`
- ``B3`` -> :mod:`echo_bench.basis.renderers.topographic_fbm`
- ``B4`` -> :mod:`echo_bench.basis.renderers.flow_attractor`

Each real renderer exposes ``render(seed, params, rendererVersion) -> np.ndarray``
returning a 64x64 single-channel ``uint8`` array, fully determined by its inputs.

The Phase 1 STUB renderer (:mod:`echo_bench.basis.renderers.stub`) remains as a
fail-closed fallback for unknown basis ids. Its native signature is
``render(basis, seed, params, rendererVersion)``; the registry adapts it to the
uniform ``(seed, params, rendererVersion)`` calling convention.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

from typing import Callable

from echo_bench.logging import get_logger, log_ko

from . import (
    branching,
    flow_attractor,
    reaction_diffusion,
    stub,
    topographic_fbm,
)

_logger = get_logger(__name__)

# Registry of real renderers keyed by basis id. Each value is a callable
# ``(seed, params, rendererVersion) -> np.ndarray(uint8, 64x64)``.
RENDERERS: dict[str, Callable] = {
    "B1": branching.render,
    "B2": reaction_diffusion.render,
    "B3": topographic_fbm.render,
    "B4": flow_attractor.render,
}


def _stub_adapter(basis: str) -> Callable:
    """Adapt the Phase-1 stub to the uniform (seed, params, version) signature."""

    def _render(seed, params, rendererVersion):
        return stub.render(basis, seed, params, rendererVersion)

    return _render


def get_renderer(basis: str) -> Callable:
    """Return the real renderer for ``basis``.

    Unknown basis ids fall back to the Phase-1 stub renderer (adapted to the
    uniform signature) and emit a Korean warning. This fails closed rather than
    raising, so the pipeline degrades gracefully on an unexpected basis id.

    Args:
        basis: Basis id (``B1``..``B4``).

    Returns:
        A callable ``(seed, params, rendererVersion) -> np.ndarray``.
    """
    renderer = RENDERERS.get(basis)
    if renderer is None:
        log_ko(
            _logger,
            f"알 수 없는 basis id={basis!r} — stub 렌더러로 대체합니다.",
        )
        return _stub_adapter(basis)
    return renderer


__all__ = ["RENDERERS", "get_renderer", "stub"]
