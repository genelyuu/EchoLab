"""Tests for the replay validator (Task F-005).

Covers:
- A deterministic run (the smoke runner) validates as replayable with no
  divergence.
- A non-deterministic run is flagged non-replayable at the first divergent hash
  (``traceHash``), and ``assert_replayable`` raises ``ReplayError``.
- ``validate_report_file`` passes for a faithfully stored smoke report and fails
  (raises ``ReplayError``) when a stored hash is tampered.
- The validator itself is deterministic.
- The comparison is EXACT (no tolerance).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo_bench.experiments import smoke
from echo_bench.experiments.smoke import run_smoke
from echo_bench.logging.replay_validator import (
    CHAIN_HASH_KEYS,
    ReplayError,
    assert_replayable,
    extract_chain,
    validate_replay,
    validate_report_file,
)

# Small, fast, deterministic smoke parameters.
_ARGS = {"base_seed": 42, "H": 6, "k": 4, "pool_size": 16}


def _report_path(seed_batch_id: str) -> Path:
    return smoke._REPORTS_DIR / f"smoke_{seed_batch_id[:12]}.json"


# --------------------------------------------------------------------------- #
# Non-deterministic run fixtures
# --------------------------------------------------------------------------- #


def _make_nondeterministic_run():
    """Return a run_fn whose ``traceHash`` changes every call (a counter).

    Everything earlier in the chain (archive/pool/slate) is stable, so the
    *first* divergence is ``traceHash`` — exactly what the validator must report.
    """
    counter = {"n": 0}

    def run_fn(**kwargs):
        counter["n"] += 1
        return {
            "archiveHash": "arc-stable",
            "poolHash": "pool-stable",
            "slateHash": "slate-stable",
            "traceHash": f"trace-{counter['n']}",  # changes each call
            "outputHash": "out-stable",
            "reportHash": "report-stable",
            "seedBatchId": "seed-nondet",
        }

    return run_fn


# --------------------------------------------------------------------------- #
# Deterministic smoke run
# --------------------------------------------------------------------------- #


def test_smoke_run_is_replayable():
    out_path = None
    try:
        result = validate_replay(run_smoke, _ARGS)
        assert result["replayable"] is True
        assert result["first_divergent"] is None
        # The reproduced chain carries the full set of hash keys.
        for key in CHAIN_HASH_KEYS:
            assert key in result["chain"], f"missing {key} in chain"
        assert result["seedBatchId"]

        out_path = _report_path(result["seedBatchId"])
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


def test_assert_replayable_passes_for_smoke():
    out_path = None
    try:
        result = assert_replayable(run_smoke, _ARGS)
        assert result["replayable"] is True
        out_path = _report_path(result["seedBatchId"])
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


# --------------------------------------------------------------------------- #
# Non-deterministic run is flagged
# --------------------------------------------------------------------------- #


def test_nondeterministic_run_flagged_at_trace_hash():
    run_fn = _make_nondeterministic_run()
    result = validate_replay(run_fn, {})
    assert result["replayable"] is False
    assert result["first_divergent"] == "traceHash"


def test_assert_replayable_raises_for_nondeterministic_run():
    run_fn = _make_nondeterministic_run()
    with pytest.raises(ReplayError):
        assert_replayable(run_fn, {})


# --------------------------------------------------------------------------- #
# Stored-report gate
# --------------------------------------------------------------------------- #


def test_validate_report_file_passes_for_faithful_report():
    out_path = None
    try:
        report = run_smoke(**_ARGS)
        out_path = _report_path(report["seedBatchId"])
        assert out_path.exists()

        result = validate_report_file(str(out_path), run_smoke, _ARGS)
        assert result["replayable"] is True
        assert result["first_divergent"] is None
        # Recorded chain and re-run chain match across the full hash set.
        for key in CHAIN_HASH_KEYS:
            assert result["recordedChain"][key] == result["chain"][key]
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


def test_validate_report_file_fails_on_tampered_report_hash(tmp_path):
    out_path = None
    try:
        report = run_smoke(**_ARGS)
        out_path = _report_path(report["seedBatchId"])

        # Tamper the stored reportHash and write to an isolated path.
        tampered = json.loads(out_path.read_text(encoding="utf-8"))
        tampered["reportHash"] = "tampered-deadbeef"
        tampered_path = tmp_path / "tampered_smoke.json"
        tampered_path.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        with pytest.raises(ReplayError) as exc_info:
            validate_report_file(str(tampered_path), run_smoke, _ARGS)
        # The first (and only) divergence is the tampered reportHash.
        assert "reportHash" in str(exc_info.value)
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


# --------------------------------------------------------------------------- #
# Validator determinism + exact comparison
# --------------------------------------------------------------------------- #


def test_validator_is_deterministic():
    out_path = None
    try:
        r1 = validate_replay(run_smoke, _ARGS)
        r2 = validate_replay(run_smoke, _ARGS)
        assert r1["replayable"] == r2["replayable"] is True
        assert r1["first_divergent"] == r2["first_divergent"] is None
        assert r1["chain"] == r2["chain"]
        assert r1["seedBatchId"] == r2["seedBatchId"]

        out_path = _report_path(r1["seedBatchId"])
    finally:
        if out_path is not None and out_path.exists():
            out_path.unlink()


def test_extract_chain_is_exact_no_tolerance():
    """A single differing hash is a divergence — no fuzzy/tolerant matching."""
    report = {
        "archiveHash": "a",
        "poolHash": "b",
        "slateHash": "c",
        "traceHash": "d",
        "outputHash": "e",
        "reportHash": "f",
        "seedBatchId": "s",
    }
    chain = extract_chain(report)
    assert chain == report

    # A near-identical chain (one char differs) must NOT be treated as equal.
    near = _make_nondeterministic_run()
    res = validate_replay(near, {})
    assert res["replayable"] is False
