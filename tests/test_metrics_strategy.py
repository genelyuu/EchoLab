"""Tests for echo_bench.metrics.utility.strategy_sensitivity (Task D-001, Phase 2).

``strategy_sensitivity`` measures how much a policy's observable trace changes
across the *controlled strategy probes* (B-004). It takes a dict mapping
probeName (str) -> TraceState and returns a normalized cross-probe divergence in
``[0, 1]``: identical traces under every probe -> 0.0, divergent traces -> > 0,
fewer than two probes -> 0.0. It is deterministic.

Also asserts that ``regret_to_oracle`` remains deferred (Phase 3, needs C-007).
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.utility import (
    compute_all_with_strategy,
    regret_to_oracle,
    strategy_sensitivity,
)


def _record(
    selected: str,
    coord: list,
    band: str,
    salience: float = 0.5,
) -> dict:
    """Build one valid caller-supplied round record (no ``roundHash``)."""
    return {
        "candidatePoolHash": "poolhash-abc",
        "slate": ["c1", "c2", "c3", "c4"],
        "selectedCardId": selected,
        "coordinateContribution": coord,
        "complexityBand": band,
        "salienceScore": salience,
        "slotPermutation": [0, 1, 2, 3],
    }


def _build_trace(records: list) -> TraceState:
    ts = TraceState()
    for rec in records:
        ts.append_round(dict(rec))
    return ts


def _low_trace() -> TraceState:
    """A trace whose selections sit in the low-complexity / low-coord region."""
    return _build_trace(
        [
            _record("c1", [0.05, 0.05, 0.05, 0.05], "low"),
            _record("c2", [0.10, 0.10, 0.10, 0.10], "low"),
            _record("c3", [0.05, 0.10, 0.05, 0.10], "low"),
        ]
    )


def _high_trace() -> TraceState:
    """A trace whose selections sit in the high-complexity / high-coord region."""
    return _build_trace(
        [
            _record("c4", [0.95, 0.95, 0.95, 0.95], "high"),
            _record("c5", [0.90, 0.90, 0.90, 0.90], "high"),
            _record("c6", [0.95, 0.90, 0.95, 0.90], "high"),
        ]
    )


def _mid_trace() -> TraceState:
    """A trace whose selections sit in the mid-complexity / mid-coord region."""
    return _build_trace(
        [
            _record("c7", [0.50, 0.50, 0.50, 0.50], "mid"),
            _record("c8", [0.45, 0.55, 0.45, 0.55], "mid"),
            _record("c9", [0.50, 0.45, 0.50, 0.55], "mid"),
        ]
    )


# --------------------------------------------------------------------------- #
# Type / bounds.
# --------------------------------------------------------------------------- #
def test_returns_float_in_unit_interval():
    traces = {
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
        "PREFER_COORD_NOVELTY": _mid_trace(),
    }
    value = strategy_sensitivity(traces)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# No sensitivity when all probes yield identical traces.
# --------------------------------------------------------------------------- #
def test_identical_traces_under_all_probes_is_zero():
    recs = [
        _record("c1", [0.2, 0.2, 0.2, 0.2], "mid"),
        _record("c2", [0.3, 0.3, 0.3, 0.3], "mid"),
    ]
    traces = {
        "PREFER_HIGH_COMPLEXITY": _build_trace(recs),
        "PREFER_LOW_SALIENCE": _build_trace(recs),
        "PREFER_COORD_NOVELTY": _build_trace(recs),
    }
    # Sanity: the traces are genuinely identical observable histories.
    hashes = {t.trace_hash() for t in traces.values()}
    assert len(hashes) == 1
    assert strategy_sensitivity(traces) == 0.0


# --------------------------------------------------------------------------- #
# Clearly divergent traces -> strictly positive sensitivity.
# --------------------------------------------------------------------------- #
def test_divergent_traces_are_positive():
    traces = {
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
    }
    assert strategy_sensitivity(traces) > 0.0


def test_more_divergence_means_higher_sensitivity():
    # low-vs-high spans the full band/coord range; low-vs-mid is a smaller gap.
    far = strategy_sensitivity(
        {"a": _low_trace(), "b": _high_trace()}
    )
    near = strategy_sensitivity(
        {"a": _low_trace(), "b": _mid_trace()}
    )
    assert far > near > 0.0


# --------------------------------------------------------------------------- #
# Determinism: same dict -> same value, independent of insertion order.
# --------------------------------------------------------------------------- #
def test_deterministic_same_value():
    traces1 = {
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
        "PREFER_COORD_NOVELTY": _mid_trace(),
    }
    traces2 = {
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
        "PREFER_COORD_NOVELTY": _mid_trace(),
    }
    assert strategy_sensitivity(traces1) == strategy_sensitivity(traces2)


def test_insertion_order_independent():
    a = {
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
        "PREFER_COORD_NOVELTY": _mid_trace(),
    }
    b = {
        "PREFER_COORD_NOVELTY": _mid_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
    }
    assert strategy_sensitivity(a) == strategy_sensitivity(b)


# --------------------------------------------------------------------------- #
# Degenerate inputs.
# --------------------------------------------------------------------------- #
def test_single_probe_is_zero():
    assert strategy_sensitivity({"only": _high_trace()}) == 0.0


def test_empty_dict_is_zero():
    assert strategy_sensitivity({}) == 0.0


def test_empty_traces_under_all_probes_is_zero():
    traces = {"a": TraceState(), "b": TraceState(), "c": TraceState()}
    assert strategy_sensitivity(traces) == 0.0


# --------------------------------------------------------------------------- #
# compute_all_with_strategy folds in the metric.
# --------------------------------------------------------------------------- #
def test_compute_all_with_strategy_keys_and_values():
    trace = _high_trace()
    traces = {
        "PREFER_HIGH_COMPLEXITY": _high_trace(),
        "PREFER_LOW_SALIENCE": _low_trace(),
    }
    result = compute_all_with_strategy(trace, traces)
    expected = {
        "traceHash",
        "coordinate_coverage",
        "artifact_diversity",
        "redundancy_rate",
        "round_coherence",
        "strategy_sensitivity",
    }
    assert set(result.keys()) == expected
    assert result["traceHash"] == trace.trace_hash()
    assert result["strategy_sensitivity"] == strategy_sensitivity(traces)
    # regret_to_oracle stays excluded.
    assert "regret_to_oracle" not in result


def test_compute_all_with_strategy_is_deterministic():
    trace = _high_trace()
    traces = {"a": _high_trace(), "b": _low_trace()}
    r1 = compute_all_with_strategy(trace, traces)
    r2 = compute_all_with_strategy(_high_trace(), {"a": _high_trace(), "b": _low_trace()})
    assert r1 == r2


# --------------------------------------------------------------------------- #
# regret_to_oracle is now implemented (Phase 3, against the C-007 oracle ref).
# NOTE: updates the previous assertion that it raised NotImplementedError. It now
# returns a float in [0, 1] given a per-round oracle objective sequence aligned
# with the trace; None / length mismatch raises ValueError. Full coverage lives
# in tests/test_metrics_regret.py; compute_all_with_strategy still excludes it.
# --------------------------------------------------------------------------- #
def test_regret_to_oracle_requires_oracle_ref():
    with pytest.raises(ValueError):
        regret_to_oracle(_high_trace(), oracle_ref=None)


def test_regret_to_oracle_length_mismatch_raises():
    with pytest.raises(ValueError):
        regret_to_oracle(_high_trace(), oracle_ref=[1.0])
