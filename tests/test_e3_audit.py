"""Tests for the E3 leakage/robustness/replay audit runner (Task E-003)."""

from __future__ import annotations

import json

from echo_bench.experiments.e3_audit import (
    E3_LEAKAGE_POLICIES,
    _REPORTS_DIR,
    run_e3_audit,
)
from echo_bench.metrics.leakage import PROXY_DISCLAIMER
from echo_bench.metrics.robustness import FAULTS
from echo_bench.metrics.utility import CORE_METRIC_KEYS

# Small but valid parameters: full leakage policy x probe grid + fault grid at a
# short horizon and a 16-card pool. H=4 is in the horizon allowed set.
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
# appear as a key in the report data (policy-name keys are excluded below).
_FORBIDDEN_KEY_TOKENS = (
    "user_id",
    "persona",
    "emotion",
    "preference",
    "user_model",
)

# Forbidden *claim* phrases that must not appear in report string values. The
# documentary ``phaseNote``/``disclaimer``/``note`` fields are exempt: they
# legitimately restate that leakage is a proxy (not privacy/legal) and that the
# faults are controlled.
_FORBIDDEN_CLAIM_PHRASES = (
    "wellbeing",
    "gdpr",
    "users prefer",
    "user satisfaction",
    "anonymity guarantee",
    "privacy guarantee",
    "legal compliance",
    "personality",
    "diagnosis",
)

_EXEMPT_VALUE_KEYS = ("phaseNote", "disclaimer", "note")


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


def test_e3_dry_run_returns_hashes_writes_nothing():
    before = (
        set(_REPORTS_DIR.glob("e3_audit_*.json")) if _REPORTS_DIR.exists() else set()
    )
    result = run_e3_audit(dry_run=True, **_KW)

    assert result["dryRun"] is True
    for key in ("configHash", "archiveHash", "poolHash"):
        assert isinstance(result[key], str) and result[key]
    assert result["sections"] == ["leakage", "robustness", "replayAudit"]
    after = (
        set(_REPORTS_DIR.glob("e3_audit_*.json")) if _REPORTS_DIR.exists() else set()
    )
    assert after == before


def test_e3_real_run_writes_report_with_hash_chain():
    report = run_e3_audit(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report

    out_path = _REPORTS_DIR / f"e3_audit_{report['seedBatchId'][:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


def test_e3_has_all_three_sections():
    report = run_e3_audit(dry_run=False, **_KW)
    assert "leakage" in report
    assert "robustness" in report
    assert "replayAudit" in report


def test_e3_leakage_section_is_proxy_with_disclaimer():
    report = run_e3_audit(dry_run=False, **_KW)
    leakage = report["leakage"]

    # The section carries the explicit proxy flag and the proxy disclaimer.
    assert leakage["isProxy"] is True
    assert leakage["disclaimer"] == PROXY_DISCLAIMER
    assert "proxy" in leakage["disclaimer"].lower()
    assert "not a privacy guarantee" in leakage["disclaimer"].lower()

    # Every leakage policy is present with a bounded proxy value flagged isProxy.
    policies = {row["policy"] for row in leakage["table"]}
    assert policies == set(E3_LEAKAGE_POLICIES)
    for row in leakage["table"]:
        assert row["isProxy"] is True
        assert isinstance(row["leakage_proxy"], float)
        assert 0.0 <= row["leakage_proxy"] <= 1.0


def test_e3_robustness_section_per_fault():
    report = run_e3_audit(dry_run=False, **_KW)
    robustness = report["robustness"]

    faults = {row["fault"] for row in robustness["table"]}
    assert faults == set(FAULTS)
    for row in robustness["table"]:
        assert isinstance(row["robustness_score"], float)
        assert 0.0 <= row["robustness_score"] <= 1.0
        assert isinstance(row["baselineTraceHash"], str) and row["baselineTraceHash"]
        assert isinstance(row["faultedTraceHash"], str) and row["faultedTraceHash"]
        # Each row records which keys were used — self-describing report (D-010 review).
        assert row["metricKeys"] == list(CORE_METRIC_KEYS), (
            f"E3 robustness row for fault={row['fault']} must record "
            f"metricKeys pinned to CORE_METRIC_KEYS, got {row['metricKeys']!r}"
        )
    # Documented as controlled faults, not real-world.
    assert "controlled" in robustness["note"].lower()
    # The robustness section itself records which keys were pinned (D-010 review).
    assert robustness["metricKeys"] == list(CORE_METRIC_KEYS), (
        f"E3 robustness section must record metricKeys={list(CORE_METRIC_KEYS)!r}, "
        f"got {robustness.get('metricKeys')!r}"
    )


def test_e3_replay_audit_reports_replayable_true():
    report = run_e3_audit(dry_run=False, **_KW)
    replay = report["replayAudit"]

    assert replay["replayable"] is True
    assert replay["first_divergent"] is None
    assert replay["runFn"] == "echo_bench.experiments.smoke.run_smoke"


def test_e3_replay_identical_report_hash():
    r1 = run_e3_audit(dry_run=False, **_KW)
    r2 = run_e3_audit(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]


def test_e3_no_forbidden_fields_in_report():
    report = run_e3_audit(dry_run=False, **_KW)

    # No forbidden DATA field key anywhere. Policy-name keys (which include
    # PSEUDO_USER_MODEL-style identifiers) are documented identifiers, excluded.
    policy_name_keys = {name.lower() for name in E3_LEAKAGE_POLICIES}
    keys_lower = [
        str(k).lower()
        for k in _iter_keys(report)
        if str(k).lower() not in policy_name_keys
    ]
    for token in _FORBIDDEN_KEY_TOKENS:
        for key in keys_lower:
            assert token not in key, f"forbidden field key leaked: {key} ({token})"

    # No forbidden claim phrase in any report string value (documentary fields
    # exempt: they restate the proxy/controlled-fault disclaimers on purpose).
    values_lower = [
        v.lower() for v in _iter_string_values(report, skip_keys=_EXEMPT_VALUE_KEYS)
    ]
    for token in _FORBIDDEN_CLAIM_PHRASES:
        for value in values_lower:
            assert token not in value, f"forbidden claim phrase leaked: {value!r} ({token})"
