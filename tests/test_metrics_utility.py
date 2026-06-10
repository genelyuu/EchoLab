"""Tests for echo_bench.metrics.utility (Task D-001, Phase 1 subset).

Covers the original four trace-only utility metrics, their determinism and
bounds, the ``compute_all`` contract (correct ``traceHash`` + exactly the
``METRIC_KEYS`` set, no invocation of deferred metrics), and that the two
deferred metrics raise ``NotImplementedError``. The D-010 (TRD alias D-009)
coverage-distribution metrics are covered in tests/test_metrics_distribution.py.
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics import utility
from echo_bench.metrics.utility import (
    METRIC_KEYS,
    artifact_diversity,
    compute_all,
    coordinate_coverage,
    redundancy_rate,
    regret_to_oracle,
    round_coherence,
    strategy_sensitivity,
)

_METRIC_FUNCS = (
    coordinate_coverage,
    artifact_diversity,
    redundancy_rate,
    round_coherence,
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


def _synthetic_records() -> list:
    """A few synthetic round records spanning bands/coords/selections."""
    return [
        _record("c1", [0.05, 0.10, 0.20, 0.30], "low"),
        _record("c2", [0.40, 0.45, 0.50, 0.55], "mid"),
        _record("c3", [0.80, 0.85, 0.90, 0.95], "high"),
        _record("c1", [0.05, 0.10, 0.20, 0.30], "low"),  # repeats round 0
    ]


def _build_trace(records=None) -> TraceState:
    ts = TraceState()
    for rec in (records if records is not None else _synthetic_records()):
        ts.append_round(dict(rec))
    return ts


# --------------------------------------------------------------------------- #
# Bounds: each metric returns a float in [0, 1].
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", _METRIC_FUNCS)
def test_metric_returns_float_in_unit_interval(metric):
    trace = _build_trace()
    value = metric(trace)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("metric", _METRIC_FUNCS)
def test_metric_bounds_on_empty_trace(metric):
    trace = TraceState()
    value = metric(trace)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("metric", _METRIC_FUNCS)
def test_metric_bounds_on_single_round(metric):
    trace = _build_trace([_record("c1", [0.1, 0.2, 0.3, 0.4], "mid")])
    value = metric(trace)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# Determinism: identical traces -> identical metric values.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", _METRIC_FUNCS)
def test_metric_is_deterministic(metric):
    a = _build_trace()
    b = _build_trace()
    assert a.trace_hash() == b.trace_hash()
    assert metric(a) == metric(b)


def test_compute_all_is_deterministic():
    assert compute_all(_build_trace()) == compute_all(_build_trace())


# --------------------------------------------------------------------------- #
# Sanity on the discriminating behaviour of each metric.
# --------------------------------------------------------------------------- #
def test_redundancy_zero_when_all_distinct():
    recs = [
        _record("c1", [0.05, 0.1, 0.2, 0.3], "low"),
        _record("c2", [0.4, 0.45, 0.5, 0.55], "mid"),
        _record("c3", [0.8, 0.85, 0.9, 0.95], "high"),
    ]
    assert redundancy_rate(_build_trace(recs)) == 0.0


def test_redundancy_positive_when_repeated():
    assert redundancy_rate(_build_trace()) > 0.0


def test_artifact_diversity_zero_for_single_band():
    recs = [
        _record("c1", [0.1, 0.2, 0.3, 0.4], "mid"),
        _record("c2", [0.5, 0.6, 0.7, 0.8], "mid"),
    ]
    assert artifact_diversity(_build_trace(recs)) == 0.0


def test_artifact_diversity_one_for_uniform_bands():
    recs = [
        _record("c1", [0.1, 0.2, 0.3, 0.4], "low"),
        _record("c2", [0.5, 0.6, 0.7, 0.8], "high"),
    ]
    assert artifact_diversity(_build_trace(recs)) == pytest.approx(1.0)


def test_round_coherence_one_for_flat_progression():
    recs = [
        _record("c1", [0.2, 0.2, 0.2, 0.2], "mid"),
        _record("c2", [0.2, 0.2, 0.2, 0.2], "mid"),
        _record("c3", [0.2, 0.2, 0.2, 0.2], "mid"),
    ]
    assert round_coherence(_build_trace(recs)) == pytest.approx(1.0)


def test_round_coherence_single_round_is_one():
    recs = [_record("c1", [0.1, 0.2, 0.3, 0.4], "mid")]
    assert round_coherence(_build_trace(recs)) == 1.0


def test_coordinate_coverage_full_when_all_distinct_cells():
    recs = [
        _record("c1", [0.0, 0.0, 0.0, 0.0], "low"),
        _record("c2", [0.5, 0.5, 0.5, 0.5], "mid"),
        _record("c3", [0.99, 0.99, 0.99, 0.99], "high"),
    ]
    assert coordinate_coverage(_build_trace(recs)) == pytest.approx(1.0)


def test_coordinate_coverage_low_when_collapsed():
    recs = [
        _record("c1", [0.1, 0.1, 0.1, 0.1], "low"),
        _record("c2", [0.1, 0.1, 0.1, 0.1], "low"),
        _record("c3", [0.1, 0.1, 0.1, 0.1], "low"),
    ]
    # All three land in the same grid cell -> 1 distinct / 3 rounds.
    assert coordinate_coverage(_build_trace(recs)) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# compute_all contract.
# --------------------------------------------------------------------------- #
def test_compute_all_keys_are_exactly_metrics_plus_trace_hash():
    result = compute_all(_build_trace())
    expected_keys = {"traceHash", *METRIC_KEYS}
    assert set(result.keys()) == expected_keys
    # D-010 (TRD alias D-009) appended three coverage-distribution metrics to
    # the original four trace-only metrics (additive; original order kept).
    assert len(METRIC_KEYS) == 7
    assert METRIC_KEYS[:4] == (
        "coordinate_coverage",
        "artifact_diversity",
        "redundancy_rate",
        "round_coherence",
    )


def test_compute_all_carries_correct_trace_hash():
    trace = _build_trace()
    result = compute_all(trace)
    assert result["traceHash"] == trace.trace_hash()


def test_compute_all_metric_values_match_individual_functions():
    trace = _build_trace()
    result = compute_all(trace)
    assert result["coordinate_coverage"] == coordinate_coverage(trace)
    assert result["artifact_diversity"] == artifact_diversity(trace)
    assert result["redundancy_rate"] == redundancy_rate(trace)
    assert result["round_coherence"] == round_coherence(trace)


def test_compute_all_does_not_invoke_deferred_metrics(monkeypatch):
    calls = {"strategy": 0, "regret": 0}

    def _boom_strategy(*args, **kwargs):
        calls["strategy"] += 1
        raise AssertionError("strategy_sensitivity must not be called")

    def _boom_regret(*args, **kwargs):
        calls["regret"] += 1
        raise AssertionError("regret_to_oracle must not be called")

    monkeypatch.setattr(utility, "strategy_sensitivity", _boom_strategy)
    monkeypatch.setattr(utility, "regret_to_oracle", _boom_regret)

    compute_all(_build_trace())
    assert calls == {"strategy": 0, "regret": 0}


# --------------------------------------------------------------------------- #
# strategy_sensitivity is now implemented (Phase 2, over a probe-keyed dict).
# NOTE: this updates the previous Phase-1 assertion that it raised
# NotImplementedError. It now takes a dict mapping probeName -> TraceState and
# returns a float in [0, 1]; a single-probe (or empty) dict yields 0.0. Full
# coverage lives in tests/test_metrics_strategy.py. regret_to_oracle stays
# deferred.
# --------------------------------------------------------------------------- #
def test_strategy_sensitivity_single_probe_is_zero():
    value = strategy_sensitivity({"PREFER_HIGH_COMPLEXITY": _build_trace()})
    assert isinstance(value, float)
    assert value == 0.0


# --------------------------------------------------------------------------- #
# regret_to_oracle is now implemented (Phase 3, against the C-007 oracle ref).
# NOTE: this updates the previous Phase-1 assertion that it raised
# NotImplementedError. It now takes a per-round oracle objective sequence aligned
# with the trace and returns a float in [0, 1]; None / length mismatch raises
# ValueError. Full coverage lives in tests/test_metrics_regret.py.
# --------------------------------------------------------------------------- #
def test_regret_to_oracle_requires_oracle_ref():
    with pytest.raises(ValueError):
        regret_to_oracle(_build_trace(), oracle_ref=None)


def test_regret_to_oracle_zero_when_ref_matches_achieved():
    trace = _build_trace()
    achieved = utility._achieved_coordinate_novelty(trace)
    value = regret_to_oracle(trace, oracle_ref=achieved)
    assert value == 0.0
