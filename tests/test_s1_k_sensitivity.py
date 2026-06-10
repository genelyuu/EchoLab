"""Tests for the S1 k-sensitivity runner (Task E-004)."""

from __future__ import annotations

import json

from echo_bench.experiments.s1_k_sensitivity import (
    S1_K_SWEEP,
    S1_METRIC_KEYS,
    S1_POLICIES,
    _REPORTS_DIR,
    run_s1_k_sensitivity,
)

# Small but valid parameters: full k x policy grid at a short horizon, n=2, and a
# 16-card pool so the suite stays CPU-fast. H=4 is in the horizon allowed set.
_KW = dict(base_seed=42, n=2, H=4, pool_size=16)

_REQUIRED_HASHES = (
    "configHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

_FORBIDDEN_KEY_TOKENS = (
    "user_id",
    "persona",
    "emotion",
    "preference",
    "user_model",
)

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


def test_s1_dry_run_returns_hashes_writes_nothing():
    before = (
        set(_REPORTS_DIR.glob("s1_k_sensitivity_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    result = run_s1_k_sensitivity(dry_run=True, **_KW)

    assert result["dryRun"] is True
    for key in ("configHash", "archiveHash", "poolHash"):
        assert isinstance(result[key], str) and result[key]
    after = (
        set(_REPORTS_DIR.glob("s1_k_sensitivity_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    assert after == before


def test_s1_real_run_writes_report_with_hashes():
    report = run_s1_k_sensitivity(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report

    out_path = _REPORTS_DIR / f"s1_k_sensitivity_{report['seedBatchId'][:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


def test_s1_all_three_k_values_present():
    report = run_s1_k_sensitivity(dry_run=False, **_KW)
    table = report["table"]

    # Every k in the sweep is present, paired with every policy.
    assert set(S1_K_SWEEP) == {2, 4, 6}
    ks_present = {row["k"] for row in table}
    assert ks_present == set(S1_K_SWEEP)
    assert len(table) == len(S1_K_SWEEP) * len(S1_POLICIES)

    # Each (k, policy) cell carries the four trace-only metrics, bounded [0, 1],
    # and the documented per-k basis-diversity requirement.
    for row in table:
        for key in S1_METRIC_KEYS:
            assert isinstance(row[key], float)
            assert 0.0 <= row[key] <= 1.0
        if row["k"] == 2:
            assert row["requiredDistinctBases"] == 2
        else:  # k=4 / k=6 require >= 3 bases
            assert row["requiredDistinctBases"] == 3


def test_s1_replay_identical_report_hash():
    r1 = run_s1_k_sensitivity(dry_run=False, **_KW)
    r2 = run_s1_k_sensitivity(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]


def test_s1_no_forbidden_fields_in_report():
    report = run_s1_k_sensitivity(dry_run=False, **_KW)

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
    assert "regret_to_oracle" not in set(report["metricKeys"])
