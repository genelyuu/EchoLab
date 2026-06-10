# tests/test_run_all.py
"""Tests for E-009 consolidated benchmark driver + index."""
from echo_bench.experiments.run_all import run_all


def test_dry_run_plans_without_writing(tmp_path, monkeypatch):
    import echo_bench.experiments.run_all as ra
    monkeypatch.setattr(ra, "_REPORTS_DIR", tmp_path)
    plan = run_all(base_seed=11, n=3, dry_run=True)
    assert plan["dryRun"] is True
    names = {e["experiment"] for e in plan["experiments"]}
    assert {"E1_HORIZON_SWEEP", "E2_POLICY_UTILITY", "E3_AUDIT"} <= names
    assert not any(tmp_path.iterdir())


def test_index_is_hashed_and_replay_stable():
    a = run_all(base_seed=11, n=3, dry_run=False)
    b = run_all(base_seed=11, n=3, dry_run=False)
    assert a["indexHash"] == b["indexHash"]
    for entry in a["experiments"]:
        assert "reportHash" in entry and "seedBatchId" in entry
        assert "designVersion" in entry
        assert "replayable" in entry  # key present on all entries (value may be None for E1/E2)
