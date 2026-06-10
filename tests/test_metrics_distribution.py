"""Tests for the D-010 coverage-distribution metrics (TRD alias D-009).

Covers ``coordinate_entropy``, ``cell_visit_gini``, and ``time_to_saturation``
in ``echo_bench.metrics.utility``: hand-computed expectations on tiny synthetic
traces, edge cases (empty trace, single round, coordinate-free rounds), bounds,
determinism, the extended ``compute_all`` / ``METRIC_KEYS`` contract, and an
integration check that small E1 / E2 / S1 runs report the three new keys (with
E2 additionally folding them into its ``comparisons`` block).

All three metrics are computed from the SAME observable cell-binning that
``coordinate_coverage`` uses (``coordinate_cell`` over the selected cards'
``coordinateContribution``) — no new trace fields, no user/persona data.
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics import utility
from echo_bench.metrics.utility import (
    COORDINATE_GRID_BINS,
    METRIC_KEYS,
    cell_visit_gini,
    compute_all,
    coordinate_coverage,
    coordinate_entropy,
    time_to_saturation,
)

_NEW_KEYS = ("coordinate_entropy", "cell_visit_gini", "time_to_saturation")

_NEW_FUNCS = (coordinate_entropy, cell_visit_gini, time_to_saturation)


def _record(selected: str, coord: list, band: str = "mid") -> dict:
    """Build one valid caller-supplied round record (no ``roundHash``)."""
    return {
        "candidatePoolHash": "poolhash-abc",
        "slate": ["c1", "c2", "c3", "c4"],
        "selectedCardId": selected,
        "coordinateContribution": coord,
        "complexityBand": band,
        "salienceScore": 0.5,
        "slotPermutation": [0, 1, 2, 3],
    }


def _build_trace(records) -> TraceState:
    ts = TraceState()
    for rec in records:
        ts.append_round(dict(rec))
    return ts


# 1-D coordinates make the grid exactly COORDINATE_GRID_BINS = 8 cells, so the
# hand computations below stay small. Cell index of [x] is int(clamp(x) * 8).
_BINS = COORDINATE_GRID_BINS


def _coord_for_cell(idx: int) -> list:
    """A 1-D coordinate vector landing in grid cell ``idx`` (0-based)."""
    return [(idx + 0.5) / _BINS]


# --------------------------------------------------------------------------- #
# Hand-computed values: all visits in ONE cell.
# --------------------------------------------------------------------------- #
def test_single_cell_trace_entropy_zero_gini_high():
    # 3 rounds, 4-D coords, all in the same cell -> C = 8**4 = 4096 cells.
    recs = [_record(f"c{i}", [0.1, 0.1, 0.1, 0.1]) for i in range(3)]
    trace = _build_trace(recs)

    assert coordinate_entropy(trace) == 0.0
    # All n visits in 1 of C cells -> Gini = (C - 1) / C.
    c = _BINS ** 4
    assert cell_visit_gini(trace) == pytest.approx((c - 1) / c)
    # The final distinct-cell set ({1 cell}) is complete after round 1 of 3.
    assert time_to_saturation(trace) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Hand-computed values: perfectly uniform visits over ALL grid cells.
# --------------------------------------------------------------------------- #
def test_uniform_trace_entropy_one_gini_zero():
    # 8 rounds with 1-D coords, one visit in each of the 8 cells.
    recs = [_record(f"c{i}", _coord_for_cell(i)) for i in range(_BINS)]
    trace = _build_trace(recs)

    assert coordinate_entropy(trace) == pytest.approx(1.0)
    assert cell_visit_gini(trace) == pytest.approx(0.0)
    # The last new cell arrives in the final round -> saturation at H/H = 1.0.
    assert time_to_saturation(trace) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Hand-computed intermediate values.
# --------------------------------------------------------------------------- #
def test_entropy_hand_computed_intermediate():
    # 1-D, C = 8 cells; visits: cell0 x2, cell1 x1, cell2 x1 (n = 4).
    # H = -(1/2 ln 1/2 + 2 * 1/4 ln 1/4) = 1.5 ln 2; normalized by ln 8 = 3 ln 2.
    recs = [
        _record("c1", _coord_for_cell(0)),
        _record("c2", _coord_for_cell(0)),
        _record("c3", _coord_for_cell(1)),
        _record("c4", _coord_for_cell(2)),
    ]
    trace = _build_trace(recs)
    assert coordinate_entropy(trace) == pytest.approx(0.5)


def test_gini_hand_computed_intermediate():
    # Same trace: per-cell counts over all 8 cells = [2,1,1,0,0,0,0,0], n = 4.
    # Sorted ascending: [0,0,0,0,0,1,1,2]; sum(i * x_i) (1-based) = 6+7+16 = 29.
    # G = 2*29/(8*4) - (8+1)/8 = 1.8125 - 1.125 = 0.6875.
    recs = [
        _record("c1", _coord_for_cell(0)),
        _record("c2", _coord_for_cell(0)),
        _record("c3", _coord_for_cell(1)),
        _record("c4", _coord_for_cell(2)),
    ]
    trace = _build_trace(recs)
    assert cell_visit_gini(trace) == pytest.approx(0.6875)


def test_time_to_saturation_known_round_by_construction():
    # Rounds visit cells 0, 1, 2, then REPEAT cell 0: the final distinct-cell
    # set {0,1,2} is complete at round 3 of 4 -> 3/4.
    recs = [
        _record("c1", _coord_for_cell(0)),
        _record("c2", _coord_for_cell(1)),
        _record("c3", _coord_for_cell(2)),
        _record("c4", _coord_for_cell(0)),
    ]
    trace = _build_trace(recs)
    assert time_to_saturation(trace) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", _NEW_FUNCS)
def test_empty_trace_returns_zero(metric):
    assert metric(TraceState()) == 0.0


def test_single_round_values():
    trace = _build_trace([_record("c1", [0.1, 0.2, 0.3, 0.4])])
    assert coordinate_entropy(trace) == 0.0  # one cell -> no spread
    c = _BINS ** 4
    assert cell_visit_gini(trace) == pytest.approx((c - 1) / c)
    # Single-round session with a selection -> saturates at round 1 of 1.
    assert time_to_saturation(trace) == 1.0


def test_no_coordinate_rounds_return_zero():
    # Rounds exist but carry empty coordinateContribution vectors: no cell is
    # ever visited, so all three distribution metrics fail closed to 0.0.
    recs = [_record("c1", []), _record("c2", [])]
    trace = _build_trace(recs)
    for metric in _NEW_FUNCS:
        assert metric(trace) == 0.0


# --------------------------------------------------------------------------- #
# Bounds and determinism.
# --------------------------------------------------------------------------- #
_SYNTH = [
    _record("c1", [0.05, 0.10, 0.20, 0.30], "low"),
    _record("c2", [0.40, 0.45, 0.50, 0.55], "mid"),
    _record("c3", [0.80, 0.85, 0.90, 0.95], "high"),
    _record("c1", [0.05, 0.10, 0.20, 0.30], "low"),
]


@pytest.mark.parametrize("metric", _NEW_FUNCS)
def test_metric_returns_float_in_unit_interval(metric):
    value = metric(_build_trace(_SYNTH))
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("metric", _NEW_FUNCS)
def test_metric_is_deterministic(metric):
    a = _build_trace(_SYNTH)
    b = _build_trace(_SYNTH)
    assert a.trace_hash() == b.trace_hash()
    assert metric(a) == metric(b)


def test_metrics_stay_informative_when_coverage_ties():
    # TRD V-003 motivation: when raw coordinate_coverage TIES across traces,
    # the distribution metrics must still separate differently-shaped visit
    # distributions. Both traces visit 3 distinct cells over 5 rounds
    # (coverage = 0.6) but A concentrates visits (3,1,1) while B spreads them
    # (2,2,1): B has strictly higher entropy, A strictly higher Gini.
    cells_a = (0, 0, 0, 1, 2)
    cells_b = (0, 0, 1, 1, 2)
    trace_a = _build_trace(
        [_record(f"a{i}", _coord_for_cell(c)) for i, c in enumerate(cells_a)]
    )
    trace_b = _build_trace(
        [_record(f"b{i}", _coord_for_cell(c)) for i, c in enumerate(cells_b)]
    )
    assert coordinate_coverage(trace_a) == coordinate_coverage(trace_b)
    assert coordinate_entropy(trace_b) > coordinate_entropy(trace_a)
    assert cell_visit_gini(trace_a) > cell_visit_gini(trace_b)


# --------------------------------------------------------------------------- #
# compute_all / METRIC_KEYS contract (additive extension).
# --------------------------------------------------------------------------- #
def test_metric_keys_extended_additively():
    # The original four keys keep their order; the three new keys are appended.
    assert METRIC_KEYS[:4] == (
        "coordinate_coverage",
        "artifact_diversity",
        "redundancy_rate",
        "round_coherence",
    )
    assert METRIC_KEYS[4:] == _NEW_KEYS


def test_compute_all_returns_new_keys_with_matching_values():
    trace = _build_trace(_SYNTH)
    result = compute_all(trace)
    assert set(result.keys()) == {"traceHash", *METRIC_KEYS}
    assert result["coordinate_entropy"] == coordinate_entropy(trace)
    assert result["cell_visit_gini"] == cell_visit_gini(trace)
    assert result["time_to_saturation"] == time_to_saturation(trace)


def test_existing_metric_values_unchanged():
    # Purely additive: the original four metrics keep their exact values.
    trace = _build_trace(_SYNTH)
    result = compute_all(trace)
    assert result["coordinate_coverage"] == utility.coordinate_coverage(trace)
    assert result["artifact_diversity"] == utility.artifact_diversity(trace)
    assert result["redundancy_rate"] == utility.redundancy_rate(trace)
    assert result["round_coherence"] == utility.round_coherence(trace)


def test_shared_cell_binning_with_coverage():
    # The new metrics must share coordinate_cell with coordinate_coverage: a
    # trace whose rounds all land in one cell has coverage 1/n AND entropy 0.
    recs = [_record(f"c{i}", [0.11, 0.11, 0.11, 0.11]) for i in range(4)]
    trace = _build_trace(recs)
    assert coordinate_coverage(trace) == pytest.approx(0.25)
    assert coordinate_entropy(trace) == 0.0
    assert time_to_saturation(trace) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Integration: small E1 / E2 / S1 runs report the three new keys.
# --------------------------------------------------------------------------- #
def _assert_rows_carry_new_keys(table):
    assert table, "report table must not be empty"
    for row in table:
        for key in _NEW_KEYS:
            assert isinstance(row[key], float), key
            assert 0.0 <= row[key] <= 1.0, key


def test_e1_report_contains_distribution_metrics():
    from echo_bench.experiments.e1_horizon import run_e1_horizon

    report = run_e1_horizon(base_seed=42, n=2, k=4, pool_size=16, dry_run=False)
    assert set(_NEW_KEYS) <= set(report["metricKeys"])
    _assert_rows_carry_new_keys(report["table"])


def test_s1_report_contains_distribution_metrics():
    from echo_bench.experiments.s1_k_sensitivity import run_s1_k_sensitivity

    report = run_s1_k_sensitivity(base_seed=42, n=2, H=4, pool_size=16, dry_run=False)
    assert set(_NEW_KEYS) <= set(report["metricKeys"])
    _assert_rows_carry_new_keys(report["table"])


def test_e2_report_and_comparisons_contain_distribution_metrics():
    from echo_bench.experiments.e2_policy import run_e2_policy

    report = run_e2_policy(base_seed=42, H=4, k=4, pool_size=16, n=2, dry_run=False)
    assert set(_NEW_KEYS) <= set(report["metricKeys"])
    _assert_rows_carry_new_keys(report["table"])
    # The D-007 comparisons block iterates the metric keys, so the new
    # distribution metrics must each get a per-metric comparison family.
    comparisons = report["comparisons"]
    assert comparisons is not None
    for key in _NEW_KEYS:
        assert key in comparisons["byMetric"], key
        assert comparisons["byMetric"][key]["comparisons"], key
