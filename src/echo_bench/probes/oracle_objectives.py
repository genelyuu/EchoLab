"""Objective-specific oracle scorers for ECHO-Bench (Task C-010).

GUARDRAIL / FRAMING — READ FIRST
================================
The C-007 ``ORACLE_STRATEGY`` is **objective-specific, NOT a universal utility
upper bound**. It ranks candidate cards by ONE documented system-level objective
and is the regret reference for *that* objective only. This module defines the
objective scorers the oracle may optimize, kept **separate from the B-004
strategy ``PROBES``** registry so adding an oracle objective never perturbs the
``strategy_sensitivity`` probe family.

Each objective is a pure, deterministic ``score(card, trace) -> float`` over
*observable* card/trace fields only (``coordinateContribution``,
``complexityBand``). Higher = more preferred. They mirror the per-round achievers
in :mod:`echo_bench.metrics.utility` so the oracle's selection and the
``regret_to_oracle_for`` reference are measured in identical units:

- ``COVERAGE_GAIN`` — ``1.0`` if the card's coordinate-grid cell is unvisited in
  the trace so far, else ``0.0`` (marginal contribution to coordinate coverage).
- ``DIVERSITY_GAIN`` — ``1.0 / (1.0 + prior_count)`` of the card's
  ``complexityBand`` over prior rounds (rewards under-represented bands).

These objectives make no claim about users, preference, emotion, wellbeing,
privacy, or any real-world effect. Identifiers stay English; logs are Korean.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from echo_bench.metrics.utility import coordinate_cell

__all__ = [
    "coverage_gain_score",
    "diversity_gain_score",
    "ORACLE_OBJECTIVES",
    "get_oracle_objective",
]

# Ordinal ranking for the observable complexity band (unused directly here but
# kept for parity with the achievers / probes that bin bands).


def _visited_cells(trace: Any) -> set:
    """Set of coordinate-grid cells visited by the observable trace so far."""
    cells: set = set()
    rounds_fn = getattr(trace, "rounds", None)
    if not callable(rounds_fn):
        return cells
    for record in rounds_fn():
        cc = record.get("coordinateContribution")
        if cc:
            cells.add(coordinate_cell(cc))
    return cells


def coverage_gain_score(card: Mapping[str, Any], trace: Any) -> float:
    """Score a card by its marginal coordinate-coverage gain (Task C-010).

    ``1.0`` when the card's coordinate-grid cell has not been visited by any
    prior round in ``trace``, else ``0.0``. Matches
    :func:`echo_bench.metrics.utility._achieved_coverage_gain` evaluated on the
    selected card against the trace prefix. Reads only ``coordinateContribution``.
    """
    cc = card.get("coordinateContribution")
    if not cc:
        return 0.0
    return 0.0 if coordinate_cell(cc) in _visited_cells(trace) else 1.0


def _band_counts(trace: Any) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    rounds_fn = getattr(trace, "rounds", None)
    if callable(rounds_fn):
        for record in rounds_fn():
            band = record.get("complexityBand")
            counts[band] = counts.get(band, 0) + 1
    return counts


def diversity_gain_score(card: Mapping[str, Any], trace: Any) -> float:
    """Score a card by its complexity-band diversity gain (Task C-010).

    ``1.0 / (1.0 + prior_count)`` where ``prior_count`` is how many prior rounds
    used the card's ``complexityBand`` — a fresh/rare band scores high. Matches
    :func:`echo_bench.metrics.utility._achieved_diversity_gain`. Reads only
    ``complexityBand``.
    """
    band = card.get("complexityBand")
    prior = _band_counts(trace).get(band, 0)
    return 1.0 / (1.0 + prior)


# Registry of oracle objective scorers, keyed by the OBJECTIVE name. Kept
# SEPARATE from echo_bench.probes.strategy_probes.PROBES on purpose.
ORACLE_OBJECTIVES: Dict[str, Callable[[Mapping[str, Any], Any], float]] = {
    "COVERAGE_GAIN": coverage_gain_score,
    "DIVERSITY_GAIN": diversity_gain_score,
}


def get_oracle_objective(name: str) -> Callable[[Mapping[str, Any], Any], float]:
    """Return the oracle objective scorer registered under ``name``.

    :raises KeyError: (Korean message) if no objective is registered under
        ``name``.
    """
    if name not in ORACLE_OBJECTIVES:
        raise KeyError(
            f"등록되지 않은 oracle objective 이름입니다: {name!r} "
            f"(사용 가능: {sorted(ORACLE_OBJECTIVES)})"
        )
    return ORACLE_OBJECTIVES[name]
