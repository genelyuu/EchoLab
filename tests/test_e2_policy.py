"""Tests for the E2 policy-utility runner (Task E-002)."""

from __future__ import annotations

import json

import pytest

from echo_bench.experiments.e2_policy import (
    CONTRAST_BASELINE_POLICY,
    E2_METRIC_KEYS,
    E2_POLICIES,
    ORACLE_POLICY,
    _REPORTS_DIR,
    run_e2_policy,
)

# Small but valid parameters: full policy x probe grid at a short horizon and a
# 16-card pool (matches the smoke runner). H=4 is in the horizon allowed set.
_KW = dict(base_seed=42, H=4, k=4, pool_size=16)

_REQUIRED_HASHES = (
    "configHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

# Forbidden user-model / persona / preference *field-key* tokens that must never
# appear as a key anywhere in the report data (the user-model leak set).
_FORBIDDEN_KEY_TOKENS = (
    "user_id",
    "persona",
    "emotion",
    "preference",
    "user_model",
)

# Forbidden *claim* phrases that must not appear in report string values. The
# documentary ``phaseNote`` field is exempt: it legitimately names the deferred
# Phase-3 policies (PSEUDO_USER_MODEL, ORACLE_STRATEGY) and the deferred
# regret_to_oracle metric to self-document scope.
_FORBIDDEN_CLAIM_PHRASES = (
    "wellbeing",
    "gdpr",
    "users prefer",
    "user satisfaction",
    "emotion",
    "personality",
    "diagnosis",
)


def _iter_keys(obj):
    """Yield every dict key (recursively) inside ``obj``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_keys(item)


def _iter_string_values(obj, skip_keys=()):
    """Yield every string value (recursively), skipping the given keys' subtrees."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            yield from _iter_string_values(v, skip_keys)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_string_values(item, skip_keys)
    elif isinstance(obj, str):
        yield obj


def test_e2_dry_run_returns_hashes_writes_nothing():
    before = set(_REPORTS_DIR.glob("e2_policy_*.json")) if _REPORTS_DIR.exists() else set()
    result = run_e2_policy(dry_run=True, **_KW)

    assert result["dryRun"] is True
    for key in ("configHash", "archiveHash", "poolHash"):
        assert isinstance(result[key], str) and result[key]
    after = set(_REPORTS_DIR.glob("e2_policy_*.json")) if _REPORTS_DIR.exists() else set()
    assert after == before


def test_e2_real_run_writes_report_with_hashes():
    report = run_e2_policy(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report

    out_path = _REPORTS_DIR / f"e2_policy_{report['seedBatchId'][:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


_EXPECTED_POLICIES = {
    "RANDOM",
    "FIXED_LOW_TO_HIGH",
    "FIXED_BALANCED",
    "TRACE_GREEDY",
    "TRACE_LIN_UCB",
    "PSEUDO_USER_MODEL",
    # C-008 strengthened contrast-baseline variants (E-013).
    "PSEUDO_USER_MODEL_DIVERSITY_REG",
    "PSEUDO_USER_MODEL_SESSION_EMBEDDING",
    "ORACLE_STRATEGY",
    # C-010 objective-specific oracle references (E-013).
    "ORACLE_COVERAGE",
    "ORACLE_DIVERSITY",
}


def test_e2_all_policies_present_with_metrics():
    report = run_e2_policy(dry_run=False, **_KW)
    table = report["table"]

    # Every policy present, exactly once each (now 11 after E-013).
    assert {row["policy"] for row in table} == set(E2_POLICIES)
    assert set(E2_POLICIES) == _EXPECTED_POLICIES
    assert len(table) == len(E2_POLICIES) == 11

    for row in table:
        # Every policy row carries the trace-only metrics +
        # strategy_sensitivity + regret_to_oracle.
        for key in E2_METRIC_KEYS:
            assert isinstance(row[key], float)
            assert 0.0 <= row[key] <= 1.0
        assert "strategy_sensitivity" in row
        assert "regret_to_oracle" in row


def test_e2_strengthened_baselines_are_not_straw_men():
    # E-013 acceptance: the strengthened contrast variants keep artifact_diversity
    # strictly positive AND at least as high as the BASIC contrast baseline (which
    # collapses toward 0 on the full pool) — so the contrast is fair, not a straw
    # man. All three pseudo rows stay flagged as contrast baselines.
    report = run_e2_policy(dry_run=False, **_KW)
    rows = {r["policy"]: r for r in report["table"]}
    basic_div = rows["PSEUDO_USER_MODEL"]["artifact_diversity"]

    for variant in (
        "PSEUDO_USER_MODEL_DIVERSITY_REG",
        "PSEUDO_USER_MODEL_SESSION_EMBEDDING",
    ):
        assert rows[variant]["artifact_diversity"] > 0.0, variant
        assert rows[variant]["artifact_diversity"] >= basic_div - 1e-9, variant
        assert rows[variant]["isContrastBaseline"] is True
        assert rows[variant]["traceOnly"] is False

    # Objective-specific oracle rows are present, flagged as oracles, and the
    # comparison block scores TRACE_GREEDY against the strengthened baselines.
    for oracle in ("ORACLE_COVERAGE", "ORACLE_DIVERSITY"):
        assert rows[oracle]["isOracle"] is True
    others = set(report["comparisons"]["others"])
    assert {"PSEUDO_USER_MODEL_DIVERSITY_REG", "ORACLE_COVERAGE"} <= others


def test_e2_regret_to_oracle_present_and_bounded():
    report = run_e2_policy(dry_run=False, **_KW)

    assert "regret_to_oracle" in set(report["metricKeys"])
    for row in report["table"]:
        regret = row["regret_to_oracle"]
        assert isinstance(regret, float)
        assert 0.0 <= regret <= 1.0


def test_e2_oracle_self_regret_near_zero():
    report = run_e2_policy(dry_run=False, **_KW)
    by_policy = {row["policy"]: row for row in report["table"]}

    oracle_row = by_policy[ORACLE_POLICY]
    # The oracle's reference is its own achieved coordinate-novelty objective, so
    # its self-regret is (numerically) ~0.
    assert oracle_row["regret_to_oracle"] == pytest.approx(0.0, abs=1e-9)
    assert oracle_row["isOracle"] is True


def test_e2_contrast_baseline_isolated():
    report = run_e2_policy(dry_run=False, **_KW)
    by_policy = {row["policy"]: row for row in report["table"]}

    baseline_row = by_policy[CONTRAST_BASELINE_POLICY]
    # The contrast baseline is present but flagged isolated and NOT trace-only.
    assert baseline_row["isContrastBaseline"] is True
    assert baseline_row["traceOnly"] is False
    assert report["contrastBaselinePolicy"] == CONTRAST_BASELINE_POLICY

    # Every contrast-baseline-family / oracle-family row is flagged non-trace-only;
    # every other policy stays trace-only and not a contrast baseline.
    from echo_bench.experiments.e2_policy import (
        CONTRAST_BASELINE_POLICIES,
        ORACLE_POLICIES,
    )

    for name, row in by_policy.items():
        if name in CONTRAST_BASELINE_POLICIES or name in ORACLE_POLICIES:
            assert row["traceOnly"] is False, name
            continue
        assert row["traceOnly"] is True, name
        assert row["isContrastBaseline"] is False, name


def test_e2_strategy_sensitivity_still_present():
    report = run_e2_policy(dry_run=False, **_KW)
    assert "strategy_sensitivity" in set(report["metricKeys"])
    for row in report["table"]:
        assert isinstance(row["strategy_sensitivity"], float)
        assert 0.0 <= row["strategy_sensitivity"] <= 1.0


def test_e2_replay_identical_report_hash():
    r1 = run_e2_policy(dry_run=False, **_KW)
    r2 = run_e2_policy(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]


def test_e2_no_forbidden_fields_in_report():
    report = run_e2_policy(dry_run=False, **_KW)

    # No user-model / persona / preference DATA field key anywhere in the report.
    # The PSEUDO_USER_MODEL policy *name* is a documented identifier (it appears
    # as a policy-row value and as a perPolicySeedBatchIds key), not a leaked
    # user-model data field, so policy-name keys are excluded from this scan.
    policy_name_keys = {name.lower() for name in E2_POLICIES}
    keys_lower = [
        str(k).lower()
        for k in _iter_keys(report)
        if str(k).lower() not in policy_name_keys
    ]
    for token in _FORBIDDEN_KEY_TOKENS:
        for key in keys_lower:
            assert token not in key, f"forbidden field key leaked: {key} ({token})"

    # No forbidden claim phrase in any report string value (phaseNote exempt:
    # it documents the Phase-3 scope and the isolated contrast baseline on purpose).
    values_lower = [v.lower() for v in _iter_string_values(report, skip_keys=("phaseNote",))]
    for token in _FORBIDDEN_CLAIM_PHRASES:
        for value in values_lower:
            assert token not in value, f"forbidden claim phrase leaked: {value!r} ({token})"

    # The contrast baseline IS present now (Phase 3) but must be explicitly
    # flagged isolated and never marked trace-only.
    by_policy = {row["policy"]: row for row in report["table"]}
    assert CONTRAST_BASELINE_POLICY in by_policy
    assert by_policy[CONTRAST_BASELINE_POLICY]["isContrastBaseline"] is True
    assert by_policy[CONTRAST_BASELINE_POLICY]["traceOnly"] is False
    # regret_to_oracle is now a reported metric (Phase 3 complete).
    assert "regret_to_oracle" in set(report["metricKeys"])
