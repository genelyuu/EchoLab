"""Tests for echo_bench.metrics.aggregate rank stability (Task D-013).

Covers the unit-level ``rank_stability`` (hand-computed 3-policies x 4-units
case with a known winner pattern, tie handling via average ranks + strict-best
top-rank counting, direction handling for lower-is-better metrics, error cases,
determinism) plus the ``METRIC_DIRECTIONS`` map and the E2 report integration
(``rankStability`` block). All unit-level test data is hand-crafted; there is
no RNG anywhere in the ranking, so every expectation is exact.

NOTE: this is the D-013 *unit-level* rank stability (one value per resampling
unit per policy), distinct from the D-007 sub-batch diagnostic of the same name
in ``echo_bench.metrics.compare``.
"""

from __future__ import annotations

import math

import pytest

from echo_bench.experiments.e2_policy import E2_METRIC_KEYS, run_e2_policy
from echo_bench.metrics.aggregate import (
    METRIC_DIRECTIONS,
    rank_stability,
    rank_stability_by_metric,
)

# Hand-computed case: 3 policies x 4 units (higher is better).
# Unit-wise winners: A, A, C, A -> top probs A=0.75, B=0.0, C=0.25.
# Ranks per unit:  A: 1,1,3,1   B: 2,2,2,2   C: 3,3,1,3
_KNOWN = {
    "A": [3.0, 3.0, 1.0, 3.0],
    "B": [2.0, 2.0, 2.0, 2.0],
    "C": [1.0, 1.0, 3.0, 1.0],
}


def test_known_winner_pattern_exact():
    out = rank_stability(_KNOWN)
    per = out["per_policy"]
    assert out["n_units"] == 4
    assert out["higher_is_better"] is True

    assert per["A"]["top_rank_probability"] == 0.75
    assert per["B"]["top_rank_probability"] == 0.0
    assert per["C"]["top_rank_probability"] == 0.25

    assert per["A"]["mean_rank"] == 1.5
    assert per["B"]["mean_rank"] == 2.0
    assert per["C"]["mean_rank"] == 2.5

    assert per["A"]["rank_distribution"] == {"1": 3, "3": 1}
    assert per["B"]["rank_distribution"] == {"2": 4}
    assert per["C"]["rank_distribution"] == {"1": 1, "3": 3}

    for block in per.values():
        assert block["n_units"] == 4


def test_tie_handling_average_rank_and_strict_top():
    # Unit 1: A and B tie at 2.0 -> both get average rank 1.5, NEITHER is
    # strictly best. Unit 2: B wins outright.
    out = rank_stability({"A": [2.0, 1.0], "B": [2.0, 3.0]})
    per = out["per_policy"]
    assert out["tie_handling"] == "average_rank"

    assert per["A"]["top_rank_probability"] == 0.0
    assert per["B"]["top_rank_probability"] == 0.5
    # Strict ties make the top-prob sum < 1.0 (documented).
    total = sum(p["top_rank_probability"] for p in per.values())
    assert total == 0.5

    assert per["A"]["mean_rank"] == pytest.approx(1.75)
    assert per["B"]["mean_rank"] == pytest.approx(1.25)
    assert per["A"]["rank_distribution"] == {"1.5": 1, "2": 1}
    assert per["B"]["rank_distribution"] == {"1": 1, "1.5": 1}


def test_all_tied_units_have_no_strict_winner():
    out = rank_stability({"A": [1.0, 1.0], "B": [1.0, 1.0], "C": [1.0, 1.0]})
    for block in out["per_policy"].values():
        assert block["top_rank_probability"] == 0.0
        # Three-way tie over ranks 1..3 -> average rank 2.0 in every unit.
        assert block["mean_rank"] == 2.0
        assert block["rank_distribution"] == {"2": 2}


def test_lower_is_better_inverts_ranking():
    vals = {"LOW": [0.1, 0.2, 0.1], "HIGH": [0.9, 0.8, 0.9]}
    up = rank_stability(vals, higher_is_better=True)
    down = rank_stability(vals, higher_is_better=False)

    assert up["per_policy"]["HIGH"]["top_rank_probability"] == 1.0
    assert up["per_policy"]["LOW"]["mean_rank"] == 2.0

    assert down["higher_is_better"] is False
    assert down["per_policy"]["LOW"]["top_rank_probability"] == 1.0
    assert down["per_policy"]["LOW"]["mean_rank"] == 1.0
    assert down["per_policy"]["HIGH"]["mean_rank"] == 2.0


def test_single_unit_probabilities_are_zero_or_one():
    out = rank_stability({"A": [1.0], "B": [2.0]})
    assert out["n_units"] == 1
    assert out["per_policy"]["B"]["top_rank_probability"] == 1.0
    assert out["per_policy"]["A"]["top_rank_probability"] == 0.0
    assert out["per_policy"]["B"]["rank_distribution"] == {"1": 1}


def test_single_policy_is_always_top():
    out = rank_stability({"ONLY": [0.3, 0.1, 0.7]})
    block = out["per_policy"]["ONLY"]
    assert block["top_rank_probability"] == 1.0
    assert block["mean_rank"] == 1.0
    assert block["rank_distribution"] == {"1": 3}


def test_empty_input_raises_korean_value_error():
    with pytest.raises(ValueError, match="values_by_policy 가 비어 있습니다"):
        rank_stability({})


def test_empty_unit_lists_raise():
    with pytest.raises(ValueError, match="단위 값 리스트가 비어 있습니다"):
        rank_stability({"A": [], "B": []})


def test_unequal_lengths_raise():
    with pytest.raises(ValueError, match="정책별 단위 값 길이가 서로 다릅니다"):
        rank_stability({"A": [1.0, 2.0], "B": [1.0]})


def test_deterministic_same_input_same_output():
    a = rank_stability(_KNOWN)
    b = rank_stability(_KNOWN)
    assert a == b
    # Insertion order of the input dict must not matter.
    reordered = {"C": _KNOWN["C"], "A": _KNOWN["A"], "B": _KNOWN["B"]}
    assert rank_stability(reordered) == a


def test_metric_directions_cover_report_metrics():
    # Every E2-reported metric has an explicit ranking direction.
    for key in E2_METRIC_KEYS:
        assert key in METRIC_DIRECTIONS, key
    # Documented directions (True = higher is better for ranking).
    assert METRIC_DIRECTIONS["coordinate_coverage"] is True
    assert METRIC_DIRECTIONS["artifact_diversity"] is True
    assert METRIC_DIRECTIONS["redundancy_rate"] is False
    assert METRIC_DIRECTIONS["round_coherence"] is True
    assert METRIC_DIRECTIONS["coordinate_entropy"] is True
    assert METRIC_DIRECTIONS["cell_visit_gini"] is False
    assert METRIC_DIRECTIONS["time_to_saturation"] is False
    assert METRIC_DIRECTIONS["strategy_sensitivity"] is True
    assert METRIC_DIRECTIONS["regret_to_oracle"] is False


def test_rank_stability_by_metric_applies_directions():
    vals = {"LOW": [0.1, 0.2], "HIGH": [0.9, 0.8]}
    out = rank_stability_by_metric(
        {"redundancy_rate": vals, "coordinate_coverage": vals}
    )
    # redundancy_rate: lower is better -> LOW wins every unit.
    red = out["redundancy_rate"]
    assert red["higher_is_better"] is False
    assert red["per_policy"]["LOW"]["top_rank_probability"] == 1.0
    # coordinate_coverage: higher is better -> HIGH wins every unit.
    cov = out["coordinate_coverage"]
    assert cov["higher_is_better"] is True
    assert cov["per_policy"]["HIGH"]["top_rank_probability"] == 1.0
    # The wrapper labels each block with its metric + direction.
    assert red["metric"] == "redundancy_rate"
    assert red["direction"] == "lower_is_better"
    assert cov["direction"] == "higher_is_better"


def test_rank_stability_by_metric_unknown_metric_defaults_higher():
    out = rank_stability_by_metric({"unmapped_metric": {"A": [1.0], "B": [2.0]}})
    block = out["unmapped_metric"]
    assert block["higher_is_better"] is True
    assert block["per_policy"]["B"]["top_rank_probability"] == 1.0


# ---------------------------------------------------------------------------
# E2 integration: the report carries a top-level rankStability block.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e2_report():
    return run_e2_policy(
        base_seed=7, H=4, k=4, pool_size=16, n=3, replay_validate=False
    )


def test_e2_report_has_rank_stability_block(e2_report):
    assert "rankStability" in e2_report
    block = e2_report["rankStability"]
    assert set(block) == set(E2_METRIC_KEYS)

    policies = {row["policy"] for row in e2_report["table"]}
    for key in E2_METRIC_KEYS:
        per = block[key]["per_policy"]
        assert set(per) == policies
        assert block[key]["n_units"] == 3
        total_top = 0.0
        for p_block in per.values():
            prob = p_block["top_rank_probability"]
            assert 0.0 <= prob <= 1.0
            assert 1.0 <= p_block["mean_rank"] <= len(policies)
            assert sum(p_block["rank_distribution"].values()) == 3
            assert math.isfinite(p_block["mean_rank"])
            total_top += prob
        # Strict ties count as nobody-on-top, so the sum can be < 1.0 but
        # never exceeds it.
        assert total_top <= 1.0 + 1e-9


def test_e2_rank_stability_uses_documented_directions(e2_report):
    block = e2_report["rankStability"]
    for key in E2_METRIC_KEYS:
        assert block[key]["higher_is_better"] == METRIC_DIRECTIONS[key]
        assert block[key]["metric"] == key
