"""Tests for the S4 salience-audit runner (Task E-007)."""

from __future__ import annotations

import json

from echo_bench.experiments.s4_salience_audit import (
    S4_METRIC_KEYS,
    S4_POLICIES,
    _REPORTS_DIR,
    run_s4_salience_audit,
)

# Small but valid parameters: full policy set, n=2, short H, 16-card pool.
_KW = dict(base_seed=42, n=2, H=4, k=4, pool_size=16)

_REQUIRED_HASHES = (
    "configHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

# Note: "emotion" / "preference" appear in S4's guardrail phaseNote (it disclaims
# them), so phaseNote is exempt from the claim-phrase scan, like the E2 test.
_FORBIDDEN_KEY_TOKENS = (
    "user_id",
    "persona",
    "user_model",
)

_FORBIDDEN_CLAIM_PHRASES = (
    "wellbeing",
    "gdpr",
    "users prefer",
    "user satisfaction",
    "personality",
    "diagnosis",
)


def _iter_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_keys(item)


def _iter_string_values(obj, skip_keys=()):
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


def test_s4_dry_run_returns_hashes_writes_nothing():
    before = (
        set(_REPORTS_DIR.glob("s4_salience_audit_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    result = run_s4_salience_audit(dry_run=True, **_KW)

    assert result["dryRun"] is True
    for key in ("configHash", "archiveHash", "poolHash"):
        assert isinstance(result[key], str) and result[key]
    after = (
        set(_REPORTS_DIR.glob("s4_salience_audit_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    assert after == before


def test_s4_real_run_writes_report_with_hashes():
    report = run_s4_salience_audit(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report

    out_path = _REPORTS_DIR / f"s4_salience_audit_{report['seedBatchId'][:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


def test_s4_salience_metrics_present_per_policy_and_bounded():
    report = run_s4_salience_audit(dry_run=False, **_KW)
    table = report["table"]

    # Every audited policy has a row.
    assert {row["policy"] for row in table} == set(S4_POLICIES)
    assert len(table) == len(S4_POLICIES)

    # Each row carries both salience metrics, bounded [0, 1], plus the
    # over-concentration flag.
    for row in table:
        for key in S4_METRIC_KEYS:
            assert key in row
            assert isinstance(row[key], float)
            assert 0.0 <= row[key] <= 1.0
        assert isinstance(row["overConcentratesOnHighSalience"], bool)
        # The flag is consistent with the configured threshold.
        expected = row["salience_outlier_rate"] > report["outlier_threshold"]
        assert row["overConcentratesOnHighSalience"] is expected

    # The metric key set is exactly the two salience-audit metrics.
    assert set(report["metricKeys"]) == set(S4_METRIC_KEYS)


def test_s4_replay_identical_report_hash():
    r1 = run_s4_salience_audit(dry_run=False, **_KW)
    r2 = run_s4_salience_audit(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]


def test_s4_no_forbidden_fields_in_report():
    report = run_s4_salience_audit(dry_run=False, **_KW)

    keys_lower = [str(k).lower() for k in _iter_keys(report)]
    for token in _FORBIDDEN_KEY_TOKENS:
        for key in keys_lower:
            assert token not in key, f"forbidden field key leaked: {key} ({token})"

    values_lower = [
        v.lower() for v in _iter_string_values(report, skip_keys=("phaseNote",))
    ]
    for token in _FORBIDDEN_CLAIM_PHRASES:
        for value in values_lower:
            assert token not in value, f"forbidden claim phrase: {value!r} ({token})"

    assert "PSEUDO_USER_MODEL" not in {row["policy"] for row in report["table"]}
