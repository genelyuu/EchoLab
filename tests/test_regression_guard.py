"""Tests for F-006 reproducibility/guardrail regression guard."""
from echo_bench.tools.regression_guard import run_regression_guard


def test_all_checks_pass_on_clean_tree():
    result = run_regression_guard()
    assert result["ok"] is True
    names = {c["name"] for c in result["checks"]}
    assert {"smoke_replay", "e2_replay", "claim_scan", "forbidden_fields"} <= names
    assert all(c["passed"] for c in result["checks"])
