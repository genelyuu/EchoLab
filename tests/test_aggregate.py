"""Tests for D-006 seed-batch aggregation statistics."""
from echo_bench.metrics.aggregate import (
    AGG_FIELDS, MIN_SUFFICIENT_N, aggregate_values, aggregate_metric_dicts,
)


def test_fields_and_bounds():
    agg = aggregate_values([0.2, 0.4, 0.6, 0.8], "coordinate_coverage")
    assert set(agg.keys()) == set(AGG_FIELDS)
    assert agg["n"] == 4
    assert abs(agg["mean"] - 0.5) < 1e-9
    assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]
    assert agg["sufficient_n"] is True
    assert agg["ci_method"] == "bootstrap"


def test_determinism():
    vals = [0.1, 0.3, 0.35, 0.9, 0.5]
    a = aggregate_values(vals, "redundancy_rate")
    b = aggregate_values(vals, "redundancy_rate")
    assert a == b  # identical inputs -> byte-identical aggregate


def test_n1_is_degenerate():
    agg = aggregate_values([0.7], "round_coherence")
    assert agg["n"] == 1
    assert agg["sufficient_n"] is False
    assert agg["ci_method"] == "degenerate"
    assert agg["ci_low"] == agg["mean"] == agg["ci_high"] == 0.7


def test_empty_is_zero():
    agg = aggregate_values([], "artifact_diversity")
    assert agg["n"] == 0
    assert agg["mean"] == 0.0
    assert agg["sufficient_n"] is False


def test_constant_values():
    agg = aggregate_values([0.5, 0.5, 0.5], "x")
    assert agg["ci_low"] == agg["mean"] == agg["ci_high"] == 0.5
    assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]


def test_aggregate_metric_dicts():
    per_seed = [
        {"coordinate_coverage": 0.4, "redundancy_rate": 0.1},
        {"coordinate_coverage": 0.6, "redundancy_rate": 0.3},
        {"coordinate_coverage": 0.5, "redundancy_rate": 0.2},
    ]
    out = aggregate_metric_dicts(per_seed, ("coordinate_coverage", "redundancy_rate"))
    assert set(out.keys()) == {"coordinate_coverage", "redundancy_rate"}
    # mean of [0.4, 0.6, 0.5] == 0.5
    assert abs(out["coordinate_coverage"]["mean"] - 0.5) < 1e-9
    assert out["coordinate_coverage"]["n"] == 3
    assert out["coordinate_coverage"]["sufficient_n"] is True
    assert out["coordinate_coverage"]["ci_method"] == "bootstrap"
