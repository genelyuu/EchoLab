"""Degenerate-candidate filter for ECHO-Bench (Task A-007).

Rejects blank, saturated, or out-of-band cards before they enter the archive.
The filter is a pure function of a card's metrics plus configured thresholds:
identical inputs always yield the same accept/reject decision. Rejected cards
get a machine-readable reason code (no user / semantic references).

Thresholds live in ``configs/archive/filter.yaml`` and are passed in by the
caller (the archive builder) — nothing is hard-coded here.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from echo_bench.cards.schema import Card

# Machine-readable reason codes (stable strings; no user/semantic content).
REASON_BLANK_LOW_STD = "DEGENERATE_BLANK_LOW_STD"
REASON_BLANK_LOW_RANGE = "DEGENERATE_BLANK_LOW_DYNAMIC_RANGE"
REASON_SATURATED = "DEGENERATE_SATURATED"
REASON_COMPLEXITY_LOW = "DEGENERATE_COMPLEXITY_BELOW_BAND"
REASON_COMPLEXITY_HIGH = "DEGENERATE_COMPLEXITY_ABOVE_BAND"
REASON_SALIENCE_LOW = "DEGENERATE_SALIENCE_BELOW_FLOOR"


def is_degenerate(
    card: Card, thresholds: Mapping[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Decide whether ``card`` is degenerate under ``thresholds``.

    Args:
        card: The card to evaluate.
        thresholds: Mapping with the keys defined in
            ``configs/archive/filter.yaml`` (``min_std_intensity``, ...).

    Returns:
        ``(True, reason_code)`` if degenerate, else ``(False, None)``. The
        checks run in a fixed order, so the returned reason is deterministic.
    """
    vm = card.visualMetrics

    std = float(vm.get("stdIntensity", 0.0))
    dyn = float(vm.get("dynamicRange", 0.0))
    sat = float(vm.get("saturationFraction", 0.0))
    complexity = float(card.complexityScore)
    salience = float(card.salienceScore)

    if std < float(thresholds["min_std_intensity"]):
        return True, REASON_BLANK_LOW_STD
    if dyn < float(thresholds["min_dynamic_range"]):
        return True, REASON_BLANK_LOW_RANGE
    if sat > float(thresholds["max_saturation_fraction"]):
        return True, REASON_SATURATED
    if complexity < float(thresholds["min_complexity_score"]):
        return True, REASON_COMPLEXITY_LOW
    if complexity > float(thresholds["max_complexity_score"]):
        return True, REASON_COMPLEXITY_HIGH
    if salience < float(thresholds["min_salience_score"]):
        return True, REASON_SALIENCE_LOW

    return False, None
