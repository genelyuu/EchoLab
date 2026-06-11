"""Tests for the E3 leakage/robustness/replay audit runner (Tasks E-003, D-011, D-012, G-020)."""

from __future__ import annotations

import json
import math

import pytest

from echo_bench.experiments.e3_audit import (
    E3_LEAKAGE_DELTA_REFERENCE,
    E3_LEAKAGE_POLICIES,
    _REPORTS_DIR,
    run_e3_audit,
)
from echo_bench.metrics.leakage import (
    DEFAULT_NULL_PERMUTATIONS,
    LEAKAGE_RATIO_FLOOR,
    PROXY_DISCLAIMER,
    SATURATION_UNIQUE_RATE_THRESHOLD,
)
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


def test_e3_leakage_section_primary_label_and_legacy_alias_g020():
    """G-020: the leakage section's primary report label is
    probe_separability_proxy with leakage_proxy as the legacy alias (D-012
    legacyAlias precedent); table-row MACHINE KEYS stay leakage_proxy."""
    report = run_e3_audit(dry_run=False, **_KW)
    leakage = report["leakage"]

    assert leakage["metric"] == "probe_separability_proxy", (
        f"E3 leakage section primary metric label must be "
        f"'probe_separability_proxy', got {leakage.get('metric')!r}"
    )
    assert leakage["legacyAlias"] == "leakage_proxy", (
        f"E3 leakage section legacyAlias must be 'leakage_proxy', "
        f"got {leakage.get('legacyAlias')!r}"
    )
    # Machine row keys are UNCHANGED: every row still carries the
    # leakage_proxy value under its legacy key (replay compatibility).
    for row in leakage["table"]:
        assert "leakage_proxy" in row
        assert isinstance(row["leakage_proxy"], float)


def test_e3_leakage_delta_and_ratio_fields():
    """D-011 (TRD alias D-010): comparison-ready leakage fields in the report."""
    report = run_e3_audit(dry_run=False, **_KW)
    leakage = report["leakage"]

    # Section-level self-describing fields: the RANDOM reference, the documented
    # ratio floor, the utility metric used, and the explicit no-CI statement.
    assert leakage["deltaReference"] == E3_LEAKAGE_DELTA_REFERENCE == "RANDOM"
    assert leakage["ratioFloor"] == LEAKAGE_RATIO_FLOOR
    assert leakage["ratioUtilityMetric"] == "coordinate_coverage"
    assert leakage["ciAvailable"] is False
    reason = leakage["ciUnavailableReason"]
    assert isinstance(reason, str) and "scalar" in reason.lower()

    rows = {row["policy"]: row for row in leakage["table"]}
    assert E3_LEAKAGE_DELTA_REFERENCE in rows
    random_leak = rows[E3_LEAKAGE_DELTA_REFERENCE]["leakage_proxy"]

    for name, row in rows.items():
        # Delta = leakage(policy) - leakage(RANDOM), bounded to [-1, 1].
        assert row["leakage_delta_vs_random"] == pytest.approx(
            row["leakage_proxy"] - random_leak
        )
        assert -1.0 <= row["leakage_delta_vs_random"] <= 1.0
        # Mean coordinate_coverage over the policy's E3-aligned per-probe traces.
        assert 0.0 <= row["mean_coordinate_coverage"] <= 1.0
        # Ratio = mean coverage / max(leakage, floor); self-describing via the
        # section-level ratioFloor.
        expected_ratio = row["mean_coordinate_coverage"] / max(
            row["leakage_proxy"], LEAKAGE_RATIO_FLOOR
        )
        assert row["utility_per_leakage"] == pytest.approx(expected_ratio)

    # The RANDOM reference's own delta is 0.0 by definition (exact).
    assert rows[E3_LEAKAGE_DELTA_REFERENCE]["leakage_delta_vs_random"] == 0.0


def test_e3_leakage_rows_carry_null_corrected_fields():
    """D-015: every leakage-style report row carries observed/null/excess
    together — the absolute NMI is never reported alone."""
    report = run_e3_audit(dry_run=False, **_KW)
    leakage = report["leakage"]

    # Section-level: the permutation count is self-describing.
    assert leakage["nullPermutations"] == DEFAULT_NULL_PERMUTATIONS

    for row in leakage["table"]:
        for field in (
            "observed_nmi",
            "null_mean",
            "null_std",
            "excess_nmi",
            "excess_z",
        ):
            assert field in row, f"leakage row missing D-015 field: {field}"
            assert math.isfinite(row[field])
        # observed_nmi IS the legacy pooled NMI (same statistic, same value).
        assert row["observed_nmi"] == row["leakage_proxy"]
        assert row["excess_nmi"] == pytest.approx(
            row["observed_nmi"] - row["null_mean"]
        )
        assert 0.0 <= row["null_mean"] <= 1.0
        assert row["null_std"] >= 0.0


def test_e3_leakage_rows_carry_channel_separated_excess_fields():
    """D-016: every leakage row additionally carries the channel-separated
    excess trio (slate / selection / combined), additively — no legacy key is
    removed or renamed, and the combined channel reproduces the legacy D-015
    excess_nmi exactly (same signature bytes, same permutation seed)."""
    report = run_e3_audit(dry_run=False, **_KW)
    leakage = report["leakage"]

    for row in leakage["table"]:
        for field in (
            "slate_excess_nmi",
            "selection_excess_nmi",
            "combined_excess_nmi",
        ):
            assert field in row, f"leakage row missing D-016 field: {field}"
            assert isinstance(row[field], float)
            assert math.isfinite(row[field])
        # Combined channel IS the legacy null-corrected statistic: exact match.
        assert row["combined_excess_nmi"] == row["excess_nmi"]
        # D-015 legacy fields must still be present alongside (additive-only).
        for legacy_field in (
            "leakage_proxy",
            "observed_nmi",
            "null_mean",
            "null_std",
            "excess_nmi",
            "excess_z",
        ):
            assert legacy_field in row


def test_e3_leakage_rows_carry_saturation_flags():
    """D-017: every leakage row carries the signature-saturation diagnostics
    flags, additively. The exact key ``saturation_flag`` is promised by
    docs/12_CLAIM_LADDER.md Section 5 (Track L re-enable condition 2:
    ``saturation_flag = False``); it is the headline gate and equals the
    combined (legacy-signature) channel's flag. Flags are diagnostics over
    the measurement, never claims."""
    report = run_e3_audit(dry_run=False, **_KW)
    leakage = report["leakage"]

    # Section-level: the threshold the flags were computed against is
    # self-describing.
    assert leakage["saturationThreshold"] == SATURATION_UNIQUE_RATE_THRESHOLD

    for row in leakage["table"]:
        for field in (
            "saturation_flag",
            "slate_saturation_flag",
            "selection_saturation_flag",
            "combined_saturation_flag",
        ):
            assert field in row, f"leakage row missing D-017 field: {field}"
            assert isinstance(row[field], bool)
        # The headline gate IS the combined-channel flag.
        assert row["saturation_flag"] == row["combined_saturation_flag"]
        # All legacy D-015/D-016 keys must still be present (additive-only).
        for legacy_field in (
            "leakage_proxy",
            "observed_nmi",
            "excess_nmi",
            "slate_excess_nmi",
            "selection_excess_nmi",
            "combined_excess_nmi",
        ):
            assert legacy_field in row


def test_e3_trace_greedy_delta_vs_random_is_negative():
    """Sanity (D-011 acceptance): TRACE_GREEDY leaks LESS than RANDOM here.

    Consistent with the known n=10 result (TRACE_GREEDY 0.737 < RANDOM 0.947);
    at the deterministic small test config (seed=42, H=4, k=4, pool=16) the
    delta is negative (sign asserted; exact value may vary with pool/H).
    """
    report = run_e3_audit(dry_run=False, **_KW)
    rows = {row["policy"]: row for row in report["leakage"]["table"]}
    assert rows["TRACE_GREEDY"]["leakage_delta_vs_random"] < 0.0


def test_e3_robustness_section_per_fault():
    report = run_e3_audit(dry_run=False, **_KW)
    robustness = report["robustness"]

    faults = {row["fault"] for row in robustness["table"]}
    assert faults == set(FAULTS)
    for row in robustness["table"]:
        # D-012: primary label is sensitivity_score; legacy robustness_score kept.
        assert isinstance(row["sensitivity_score"], float)
        assert 0.0 <= row["sensitivity_score"] <= 1.0
        # Legacy field must still be present and equal.
        assert isinstance(row["robustness_score"], float)
        assert row["robustness_score"] == row["sensitivity_score"]
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


def test_e3_robustness_sensitivity_naming_fields():
    """D-012: robustness section carries sensitivity_score primary label + legacy alias.

    The exact phrase '0.0 = max robustness' must be present in the direction
    field (machine-readable claim text for the report).
    """
    report = run_e3_audit(dry_run=False, **_KW)
    robustness = report["robustness"]

    # Section-level: primary label, legacy alias, direction phrase.
    assert robustness["metric"] == "sensitivity_score", (
        f"E3 robustness section metric must be 'sensitivity_score', "
        f"got {robustness.get('metric')!r}"
    )
    assert robustness["legacyAlias"] == "robustness_score", (
        f"E3 robustness section legacyAlias must be 'robustness_score', "
        f"got {robustness.get('legacyAlias')!r}"
    )
    assert "0.0 = max robustness" in robustness["direction"], (
        f"E3 robustness direction must contain '0.0 = max robustness', "
        f"got {robustness.get('direction')!r}"
    )

    # Per-row: same fields.
    for row in robustness["table"]:
        assert row["legacyAlias"] == "robustness_score", (
            f"E3 robustness row legacyAlias must be 'robustness_score', "
            f"got {row.get('legacyAlias')!r} (fault={row.get('fault')})"
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
