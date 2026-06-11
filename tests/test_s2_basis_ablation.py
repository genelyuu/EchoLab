"""Tests for the S2 basis-ablation runner (Task E-005)."""

from __future__ import annotations

import json

from echo_bench.experiments.s2_basis_ablation import (
    ALL_BASES,
    S2_METRIC_KEYS,
    _REPORTS_DIR,
    run_s2_basis_ablation,
)

# Small but valid parameters: k=4 (>= 3 bases), n=2, short H, 16-card pool.
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


def test_s2_dry_run_returns_hashes_writes_nothing():
    before = (
        set(_REPORTS_DIR.glob("s2_basis_ablation_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    result = run_s2_basis_ablation(dry_run=True, **_KW)

    assert result["dryRun"] is True
    assert isinstance(result["configHash"], str) and result["configHash"]
    # Each planned arm carries its own fresh archive/pool hash.
    for arm in result["arms"]:
        assert arm["archiveHash"] and arm["poolHash"]
    after = (
        set(_REPORTS_DIR.glob("s2_basis_ablation_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    assert after == before


def test_s2_real_run_writes_report_with_hashes():
    report = run_s2_basis_ablation(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report

    out_path = _REPORTS_DIR / f"s2_basis_ablation_{report['seedBatchId'][:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


def test_s2_ablations_present_and_fresh_archives():
    report = run_s2_basis_ablation(dry_run=False, **_KW)
    table = report["table"]

    arms = {row["arm"] for row in table}
    # The full arm plus every drop-one 3-subset must be present at k=4.
    expected = {"FULL"} | {f"DROP_{b}" for b in ALL_BASES}
    assert arms == expected

    # Each arm rebuilt a fresh, independently-hashed archive (no stale reuse):
    # the four drop-one arms drop different bases, so their archive hashes differ.
    drop_hashes = {
        row["arm"]: row["archiveHash"]
        for row in table
        if row["arm"].startswith("DROP_")
    }
    assert len(set(drop_hashes.values())) == len(drop_hashes)

    # Each arm carries coverage/diversity + the trace-only metrics, bounded.
    for row in table:
        assert "cellOccupancy" in row and "perBasisCounts" in row
        # A drop-one arm must not contain the dropped basis in its archive.
        if row["arm"].startswith("DROP_"):
            dropped = row["arm"].split("_", 1)[1]
            assert row["perBasisCounts"].get(dropped, 0) == 0
        for key in S2_METRIC_KEYS:
            assert isinstance(row[key], float)
            assert 0.0 <= row[key] <= 1.0


def test_s2_two_base_ablations_skipped_and_logged_for_k4():
    report = run_s2_basis_ablation(dry_run=False, **_KW)

    # At k=4 (>= 3 bases required), no arm may use fewer than 3 bases.
    for row in report["table"]:
        assert len(row["bases"]) >= 3

    # 2-base ablations are enumerated candidates that get skipped explicitly at
    # k=4 (not silently capped): the report surfaces every skipped 2-base arm.
    assert "skippedAblations" in report
    assert len(report["skippedAblations"]) > 0  # the drop-two 2-base arms
    for skip in report["skippedAblations"]:
        assert len(skip["bases"]) < 3
        assert skip["reason"] == "insufficient_bases_for_k"
        assert skip["requiredDistinctBases"] == 3

    # The arm-feasibility predicate at k=2 (which DOES allow 2-base subsets)
    # confirms the skip path is real: at k=2 a 2-base subset is NOT skipped.
    report_k2 = run_s2_basis_ablation(
        base_seed=42, n=2, H=4, k=2, pool_size=16, dry_run=True
    )
    arm_sizes = [len(a["bases"]) for a in report_k2["arms"]]
    # k=2 needs only 2 bases, so 2-base subsets are now feasible arms.
    assert any(size == 2 for size in arm_sizes)


def test_s2_replay_identical_report_hash():
    r1 = run_s2_basis_ablation(dry_run=False, **_KW)
    r2 = run_s2_basis_ablation(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]


def test_s2_no_forbidden_fields_in_report():
    report = run_s2_basis_ablation(dry_run=False, **_KW)

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
