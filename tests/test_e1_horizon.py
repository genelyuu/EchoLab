"""Tests for the E1 horizon sweep runner (Task E-001)."""

from __future__ import annotations

import json

import pytest

from echo_bench.experiments.e1_horizon import (
    E1_METRIC_KEYS,
    E1_POLICIES,
    _REPORTS_DIR,
    run_e1_horizon,
)

# Small but valid parameters keep the test fast while still exercising the full
# H sweep x 3-policy grid. pool_size=16 matches the smoke runner; n=2 keeps the
# episode count low.
_KW = dict(base_seed=42, n=2, k=4, pool_size=16)

_EXPECTED_H = {4, 6, 8, 12, 16, 20}
_REQUIRED_HASHES = (
    "configHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)


def test_e1_dry_run_returns_hashes_writes_nothing():
    before = set(_REPORTS_DIR.glob("e1_horizon_*.json")) if _REPORTS_DIR.exists() else set()
    result = run_e1_horizon(dry_run=True, **_KW)

    assert result["dryRun"] is True
    for key in ("configHash", "archiveHash", "poolHash"):
        assert isinstance(result[key], str) and result[key]
    # 6 horizons x 3 policies = 18 planned cells.
    assert len(result["cells"]) == 6 * len(E1_POLICIES)
    # No report file appeared.
    after = set(_REPORTS_DIR.glob("e1_horizon_*.json")) if _REPORTS_DIR.exists() else set()
    assert after == before


def test_e1_real_run_writes_report_with_hashes():
    report = run_e1_horizon(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report

    out_path = _REPORTS_DIR / f"e1_horizon_{report['seedBatchId'][:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


def test_e1_sweeps_all_h_and_policies():
    report = run_e1_horizon(dry_run=False, **_KW)
    table = report["table"]

    assert len(table) == 6 * len(E1_POLICIES)
    assert {row["H"] for row in table} == _EXPECTED_H
    assert {row["policy"] for row in table} == set(E1_POLICIES)

    # Every cell carries the trace-only metrics + a compute_cost proxy.
    for row in table:
        for key in E1_METRIC_KEYS:
            assert isinstance(row[key], float)
            assert 0.0 <= row[key] <= 1.0
        assert row["compute_cost"] == row["H"] * _KW["n"]


def test_e1_replay_identical_report_hash():
    r1 = run_e1_horizon(dry_run=False, **_KW)
    r2 = run_e1_horizon(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]
