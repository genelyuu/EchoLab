"""Tests for the Phase 1 RANDOM smoke runner (Task E, Wave 5).

Covers: dry-run (no file, hashes present), a real run (report carries all
required hashes and the written file round-trips), the core replay invariant
(two runs => identical traceHash AND reportHash), and the guardrail that no
forbidden user-model field appears anywhere in the serialized report.
"""

from __future__ import annotations

import json
from pathlib import Path

from echo_bench.experiments import smoke
from echo_bench.experiments.smoke import run_smoke

# Forbidden user-model / persona / emotion / preference fields. None of these
# may appear anywhere in the serialized report (guardrail: no forbidden fields).
FORBIDDEN_FIELDS = {"user_id", "persona", "emotion", "preference", "user_model"}

# Small, fast smoke parameters shared by the tests.
_ARGS = {"base_seed": 7, "H": 6, "k": 4, "pool_size": 12}


def _report_path(seed_batch_id: str) -> Path:
    return smoke._REPORTS_DIR / f"smoke_{seed_batch_id[:12]}.json"


def _assert_no_forbidden(blob: str) -> None:
    low = blob.lower()
    for field in FORBIDDEN_FIELDS:
        assert field not in low, f"forbidden field {field!r} leaked into report"


def test_dry_run_writes_no_file_and_has_hashes():
    result = run_smoke(dry_run=True, **_ARGS)

    assert result["dryRun"] is True
    for key in ("archiveHash", "poolHash", "configHash", "seedBatchId"):
        assert isinstance(result[key], str) and result[key], f"missing {key}"

    # Dry run must not write the report file.
    assert not _report_path(result["seedBatchId"]).exists()


def test_real_run_report_hashes_and_file_roundtrip():
    out_path = None
    try:
        report = run_smoke(**_ARGS)

        for key in (
            "poolHash",
            "slateHash",
            "traceHash",
            "reportHash",
            "archiveHash",
            "seedBatchId",
        ):
            assert isinstance(report[key], str) and report[key], f"missing {key}"

        out_path = _report_path(report["seedBatchId"])
        assert out_path.exists(), "report file was not written"

        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["reportHash"] == report["reportHash"]
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


def test_replay_identical_trace_and_report_hash():
    out_path = None
    try:
        first = run_smoke(**_ARGS)
        second = run_smoke(**_ARGS)

        assert first["traceHash"] == second["traceHash"]
        assert first["reportHash"] == second["reportHash"]

        out_path = _report_path(first["seedBatchId"])
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


def test_no_forbidden_fields_in_report():
    out_path = None
    try:
        report = run_smoke(**_ARGS)
        serialized = json.dumps(report, ensure_ascii=True)
        _assert_no_forbidden(serialized)

        out_path = _report_path(report["seedBatchId"])
        # Also check the file content on disk.
        _assert_no_forbidden(out_path.read_text(encoding="utf-8"))
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()
