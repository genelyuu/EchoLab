"""Tests for E-008: E2 reports n-seed aggregates with confidence intervals."""
from echo_bench.experiments.e2_policy import E2_METRIC_KEYS, run_e2_policy


def test_e2_rows_carry_aggregates():
    rep = run_e2_policy(base_seed=7, H=4, k=4, pool_size=16, n=3)
    assert rep["config"]["n"] == 3
    assert rep["table"], "expected at least one policy row"
    row = rep["table"][0]
    for key in E2_METRIC_KEYS:
        agg = row["stats"][key]
        assert set(agg) == {
            "mean", "std", "n", "ci_low", "ci_high", "ci_method", "sufficient_n",
        }
        assert agg["n"] == 3
        assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]


def test_e2_replay_stable():
    a = run_e2_policy(base_seed=7, H=4, k=4, pool_size=16, n=3)
    b = run_e2_policy(base_seed=7, H=4, k=4, pool_size=16, n=3)
    assert a["reportHash"] == b["reportHash"]
