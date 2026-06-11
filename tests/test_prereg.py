"""Tests for echo_bench.logging.prereg (Task F-009).

TDD: written BEFORE the implementation. All tests are expected to fail
until the module and config files are created.

Coverage:
- prereg v1 JSON file loads and validates (load_prereg)
- missing-key fail-closed (load_prereg)
- v2 without supersedes/changeJustification fails (load_prereg)
- hash determinism including key-order independence (prereg_hash)
- stamp via injected git_runner (build_prereg_stamp)
- stamp fail-closed on empty git output (build_prereg_stamp)
- ledger append / no-op duplicate / corrupt file / family_entries
  (append_ledger_entry, load_ledger, family_entries)
- evaluationFamilies and pilotFamily disjoint in the committed v1 file
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from echo_bench.logging.prereg import (
    append_ledger_entry,
    build_prereg_stamp,
    entries_for_prereg,
    load_ledger,
    load_prereg,
    prereg_hash,
)
from echo_bench.utils.hash import canonical_hash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREREG_V1_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_LEDGER_PATH = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"

# The minimal set of required keys that load_prereg must validate.
_REQUIRED_KEYS = [
    "preregId",
    "version",
    "primaryEndpoint",
    "utilityGuard",
    "pairedUnit",
    "evaluationFamilies",
    "signRule",
    "ciRule",
    "tieBreakCaveatMarker",
    "experiments",
    "claimTransitions",
    "degenerateArmPolicy",
    "amendmentPolicy",
]


def _minimal_prereg_v1() -> Dict[str, Any]:
    """Return a minimal valid v1 prereg dict for unit tests (not the real file)."""
    return {
        "preregId": "test-prereg",
        "version": 1,
        "primaryEndpoint": {"metric": "slate_excess_nmi", "aggregation": "cross_family"},
        "utilityGuard": {"metric": "coordinate_coverage", "rule": "arm_mean >= RANDOM_mean"},
        "pairedUnit": "seed_family",
        "evaluationFamilies": ["42", "7", "101", "2025", "31337"],
        "signRule": {"minConsistentFamilies": 4, "totalFamilies": 5},
        "ciRule": {"method": "cross_family_seeded_bootstrap", "lowerBoundMustExceed": 0.0},
        "noPValues": True,
        "tieBreakCaveatMarker": "subject to a tie-breaking sensitivity caveat (AXS-010 soft_pass)",
        "experiments": {"AXS-003": {"arms": ["axs_ucb_default"]}},
        "claimTransitions": {"M2": {"requiresPass": ["AXS-003"]}},
        "degenerateArmPolicy": {"mustReport": True},
        "amendmentPolicy": {"supersedesRequired": True},
    }


def _minimal_ledger_entry() -> Dict[str, Any]:
    return {
        "reportId": "test-report-001",
        "experimentId": "AXS-003",
        "preregId": "axs-mechanism",
        "preregVersion": 1,
        "preregHash": "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
        "reportHash": "def456abc123def456abc123def456abc123def456abc123def456abc123def4",
        "reportPath": "outputs/reports/test_report.json",
        "runCommit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    }


# ---------------------------------------------------------------------------
# load_prereg: v1 file and unit-test dicts
# ---------------------------------------------------------------------------


class TestLoadPreregV1File:
    """The committed v1 file must load and validate cleanly."""

    def test_v1_file_exists(self):
        assert _PREREG_V1_PATH.exists(), (
            f"prereg v1 파일이 없습니다: {_PREREG_V1_PATH}"
        )

    def test_v1_file_loads(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        assert isinstance(prereg, dict)

    def test_v1_file_has_required_keys(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        for key in _REQUIRED_KEYS:
            assert key in prereg, f"필수 키 누락: {key}"

    def test_v1_prereg_id(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        assert prereg["preregId"] == "axs-mechanism"

    def test_v1_version_is_1(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        assert prereg["version"] == 1

    def test_v1_evaluation_families_count(self):
        """evaluationFamilies must have exactly 5 entries (DEFAULT_BASE_SEEDS)."""
        prereg = load_prereg(_PREREG_V1_PATH)
        assert len(prereg["evaluationFamilies"]) == 5

    def test_v1_evaluation_families_values(self):
        """Evaluation families must match DEFAULT_BASE_SEEDS family identifiers."""
        prereg = load_prereg(_PREREG_V1_PATH)
        expected = {"42", "7", "101", "2025", "31337"}
        assert set(str(f) for f in prereg["evaluationFamilies"]) == expected

    def test_v1_pilot_family_not_in_evaluation_families(self):
        """pilotFamily must be disjoint from evaluationFamilies."""
        prereg = load_prereg(_PREREG_V1_PATH)
        yoked = prereg.get("yokedSchedule", {})
        pilot = str(yoked.get("pilotFamily", ""))
        eval_families = set(str(f) for f in prereg["evaluationFamilies"])
        assert pilot not in eval_families, (
            f"pilotFamily={pilot!r} 는 evaluationFamilies={sorted(eval_families)}에 "
            "포함되면 안 됩니다."
        )

    def test_v1_pilot_family_nonempty(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        yoked = prereg.get("yokedSchedule", {})
        pilot = str(yoked.get("pilotFamily", ""))
        assert pilot, "pilotFamily 가 비어 있습니다."


class TestLoadPreregMissingKey:
    """Missing required key -> ValueError (fail-closed)."""

    @pytest.mark.parametrize("key", _REQUIRED_KEYS)
    def test_missing_key_raises(self, key, tmp_path):
        prereg = _minimal_prereg_v1()
        del prereg[key]
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        with pytest.raises(ValueError):
            load_prereg(p)


class TestLoadPreregV2Amendment:
    """v2 prereg without supersedes/changeJustification must fail."""

    def test_v2_without_supersedes_raises(self, tmp_path):
        prereg = _minimal_prereg_v1()
        prereg["version"] = 2
        # No supersedes key -> must fail
        p = tmp_path / "prereg_v2.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        with pytest.raises(ValueError):
            load_prereg(p)

    def test_v2_without_change_justification_raises(self, tmp_path):
        prereg = _minimal_prereg_v1()
        prereg["version"] = 2
        prereg["supersedes"] = "axs-mechanism-v1"
        # No changeJustification -> must fail
        p = tmp_path / "prereg_v2.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        with pytest.raises(ValueError):
            load_prereg(p)

    def test_v2_with_both_keys_passes(self, tmp_path):
        prereg = _minimal_prereg_v1()
        prereg["version"] = 2
        prereg["supersedes"] = "axs-mechanism-v1"
        prereg["changeJustification"] = "Added AXS-001 arm"
        p = tmp_path / "prereg_v2.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        result = load_prereg(p)
        assert result["version"] == 2


# ---------------------------------------------------------------------------
# prereg_hash
# ---------------------------------------------------------------------------


class TestPreregHash:
    """Hash must be deterministic and key-order-independent."""

    def test_deterministic(self):
        prereg = _minimal_prereg_v1()
        h1 = prereg_hash(prereg)
        h2 = prereg_hash(prereg)
        assert h1 == h2

    def test_key_order_independent(self):
        prereg = _minimal_prereg_v1()
        # Build a new dict with reversed key order
        keys = list(prereg.keys())
        reversed_prereg = {k: prereg[k] for k in reversed(keys)}
        h1 = prereg_hash(prereg)
        h2 = prereg_hash(reversed_prereg)
        assert h1 == h2

    def test_matches_canonical_hash(self):
        prereg = _minimal_prereg_v1()
        assert prereg_hash(prereg) == canonical_hash(prereg)

    def test_changes_with_content(self):
        prereg1 = _minimal_prereg_v1()
        prereg2 = dict(prereg1)
        prereg2["preregId"] = "different-id"
        assert prereg_hash(prereg1) != prereg_hash(prereg2)


# ---------------------------------------------------------------------------
# build_prereg_stamp
# ---------------------------------------------------------------------------


def _make_git_runner(prereg_commit: str = "aabbccdd" * 5, run_commit: str = "11223344" * 5):
    """Return an injectable git_runner that returns fake commits."""
    prereg_commit_full = prereg_commit if len(prereg_commit) == 40 else prereg_commit[:40].ljust(40, "0")
    run_commit_full = run_commit if len(run_commit) == 40 else run_commit[:40].ljust(40, "0")

    def git_runner(args):
        # git log -1 --format=%H -- <path>  -> preregCommit
        if "log" in args:
            return prereg_commit_full
        # git rev-parse HEAD -> runCommit
        if "rev-parse" in args:
            return run_commit_full
        return ""

    return git_runner


class TestBuildPreregStamp:
    """build_prereg_stamp must return correct fields via injected git_runner."""

    def test_stamp_fields(self, tmp_path):
        prereg = _minimal_prereg_v1()
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        runner = _make_git_runner()
        stamp = build_prereg_stamp(p, git_runner=runner)
        assert set(stamp.keys()) == {
            "preregId",
            "preregVersion",
            "preregPath",
            "preregHash",
            "preregCommit",
            "runCommit",
        }

    def test_stamp_values(self, tmp_path):
        prereg = _minimal_prereg_v1()
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        fake_prereg_commit = "aabbccdd" * 5
        fake_run_commit = "11223344" * 5
        runner = _make_git_runner(fake_prereg_commit, fake_run_commit)
        stamp = build_prereg_stamp(p, git_runner=runner)
        assert stamp["preregId"] == prereg["preregId"]
        assert stamp["preregVersion"] == prereg["version"]
        assert stamp["preregPath"] == str(p)
        assert stamp["preregHash"] == prereg_hash(prereg)
        assert stamp["preregCommit"] == fake_prereg_commit
        assert stamp["runCommit"] == fake_run_commit

    def test_stamp_fail_closed_on_empty_prereg_commit(self, tmp_path):
        """Empty preregCommit (uncommitted file) must raise ValueError."""
        prereg = _minimal_prereg_v1()
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")

        def runner(args):
            if "log" in args:
                return ""  # empty -> uncommitted
            return "a" * 40

        with pytest.raises(ValueError):
            build_prereg_stamp(p, git_runner=runner)

    def test_stamp_fail_closed_on_empty_run_commit(self, tmp_path):
        """Empty runCommit (git failure) must raise ValueError."""
        prereg = _minimal_prereg_v1()
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")

        def runner(args):
            if "log" in args:
                return "b" * 40
            return ""  # empty HEAD -> fail

        with pytest.raises(ValueError):
            build_prereg_stamp(p, git_runner=runner)


# ---------------------------------------------------------------------------
# append_ledger_entry / load_ledger / family_entries
# ---------------------------------------------------------------------------


class TestLedger:
    """append_ledger_entry must be append-only; duplicates are no-ops."""

    def _fresh_ledger(self, tmp_path) -> Path:
        p = tmp_path / "run_ledger.json"
        p.write_text(json.dumps({"ledgerVersion": 1, "entries": []}), encoding="utf-8")
        return p

    def test_append_entry(self, tmp_path):
        ledger_path = self._fresh_ledger(tmp_path)
        entry = _minimal_ledger_entry()
        append_ledger_entry(ledger_path, entry)
        ledger = load_ledger(ledger_path)
        assert len(ledger["entries"]) == 1
        assert ledger["entries"][0]["reportId"] == entry["reportId"]

    def test_preserves_existing_entries(self, tmp_path):
        ledger_path = self._fresh_ledger(tmp_path)
        e1 = _minimal_ledger_entry()
        e2 = dict(e1)
        e2["reportId"] = "test-report-002"
        e2["reportHash"] = "ff" * 32
        append_ledger_entry(ledger_path, e1)
        append_ledger_entry(ledger_path, e2)
        ledger = load_ledger(ledger_path)
        assert len(ledger["entries"]) == 2

    def test_duplicate_report_hash_is_noop(self, tmp_path):
        ledger_path = self._fresh_ledger(tmp_path)
        entry = _minimal_ledger_entry()
        append_ledger_entry(ledger_path, entry)
        # Appending same entry again (same reportHash) -> no-op
        append_ledger_entry(ledger_path, entry)
        ledger = load_ledger(ledger_path)
        assert len(ledger["entries"]) == 1

    def test_missing_required_key_raises(self, tmp_path):
        ledger_path = self._fresh_ledger(tmp_path)
        entry = _minimal_ledger_entry()
        del entry["reportHash"]
        with pytest.raises(ValueError):
            append_ledger_entry(ledger_path, entry)

    def test_corrupt_ledger_raises(self, tmp_path):
        p = tmp_path / "run_ledger.json"
        p.write_text("not valid json {{{{", encoding="utf-8")
        entry = _minimal_ledger_entry()
        with pytest.raises(ValueError):
            append_ledger_entry(p, entry)

    def test_ledger_missing_entries_list_raises(self, tmp_path):
        p = tmp_path / "run_ledger.json"
        p.write_text(json.dumps({"ledgerVersion": 1}), encoding="utf-8")
        entry = _minimal_ledger_entry()
        with pytest.raises(ValueError):
            append_ledger_entry(p, entry)


class TestLoadLedger:
    def test_load_returns_dict_with_entries(self, tmp_path):
        p = tmp_path / "run_ledger.json"
        p.write_text(json.dumps({"ledgerVersion": 1, "entries": []}), encoding="utf-8")
        ledger = load_ledger(p)
        assert "entries" in ledger
        assert ledger["entries"] == []

    def test_load_corrupt_raises(self, tmp_path):
        p = tmp_path / "run_ledger.json"
        p.write_text("{{not json}}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_ledger(p)

    def test_load_missing_entries_key_raises(self, tmp_path):
        p = tmp_path / "run_ledger.json"
        p.write_text(json.dumps({"ledgerVersion": 1}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_ledger(p)


class TestEntriesForPrereg:
    def _ledger_with_entries(self) -> dict:
        entries = [
            {**_minimal_ledger_entry(), "preregId": "axs-mechanism"},
            {**_minimal_ledger_entry(), "preregId": "axs-mechanism",
             "reportHash": "ee" * 32, "reportId": "rpt2"},
            {**_minimal_ledger_entry(), "preregId": "other-prereg",
             "reportHash": "ff" * 32, "reportId": "rpt3"},
        ]
        return {"ledgerVersion": 1, "entries": entries}

    def test_returns_matching_entries(self):
        ledger = self._ledger_with_entries()
        result = entries_for_prereg(ledger, "axs-mechanism")
        assert len(result) == 2

    def test_returns_empty_for_unknown_prereg(self):
        ledger = self._ledger_with_entries()
        result = entries_for_prereg(ledger, "nonexistent")
        assert result == []

    def test_filters_by_prereg_id(self):
        ledger = self._ledger_with_entries()
        result = entries_for_prereg(ledger, "other-prereg")
        assert len(result) == 1
        assert result[0]["preregId"] == "other-prereg"


# ---------------------------------------------------------------------------
# Committed v1 file — structural integrity
# ---------------------------------------------------------------------------


class TestPreregV1Structural:
    """Deep structural checks on the committed v1 file."""

    def test_committed_ledger_exists(self):
        assert _LEDGER_PATH.exists(), f"원장 파일이 없습니다: {_LEDGER_PATH}"

    def test_committed_ledger_is_valid(self):
        ledger = load_ledger(_LEDGER_PATH)
        assert "entries" in ledger
        assert isinstance(ledger["entries"], list)

    def test_committed_prereg_v1_all_required_keys(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        for key in _REQUIRED_KEYS:
            assert key in prereg

    def test_sign_rule_values(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        sr = prereg["signRule"]
        assert sr["minConsistentFamilies"] == 4
        assert sr["totalFamilies"] == 5

    def test_primary_endpoint(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        ep = prereg["primaryEndpoint"]
        assert ep["metric"] == "slate_excess_nmi"
        assert ep["aggregation"] == "cross_family"

    def test_experiments_block_has_five_entries(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        assert len(prereg["experiments"]) == 5

    def test_no_pvalues_true(self):
        prereg = load_prereg(_PREREG_V1_PATH)
        assert prereg.get("noPValues") is True


# ---------------------------------------------------------------------------
# Fix 1: Atomic ledger write — no tmp file left behind, content intact
# ---------------------------------------------------------------------------


class TestAtomicLedgerWrite:
    """append_ledger_entry must leave no .tmp file after a successful write."""

    def _fresh_ledger(self, tmp_path) -> Path:
        p = tmp_path / "run_ledger.json"
        p.write_text(json.dumps({"ledgerVersion": 1, "entries": []}), encoding="utf-8")
        return p

    def test_no_tmp_file_remains_after_append(self, tmp_path):
        ledger_path = self._fresh_ledger(tmp_path)
        entry = _minimal_ledger_entry()
        append_ledger_entry(ledger_path, entry)
        tmp_path_file = ledger_path.with_suffix(".json.tmp")
        assert not tmp_path_file.exists(), ".tmp ファイルが残っています"

    def test_content_intact_after_append(self, tmp_path):
        ledger_path = self._fresh_ledger(tmp_path)
        entry = _minimal_ledger_entry()
        append_ledger_entry(ledger_path, entry)
        ledger = load_ledger(ledger_path)
        assert len(ledger["entries"]) == 1
        assert ledger["entries"][0]["reportId"] == entry["reportId"]


# ---------------------------------------------------------------------------
# Fix 2: String-version bypass — version must be int >= 1, not bool/str
# ---------------------------------------------------------------------------


class TestVersionIntValidation:
    """load_prereg must reject non-int, bool, and out-of-range version values."""

    def _write_prereg(self, tmp_path, prereg: Dict[str, Any]) -> Path:
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps(prereg), encoding="utf-8")
        return p

    def test_string_version_raises(self, tmp_path):
        """'version': '2' (string) must raise ValueError, not silently skip amendment check."""
        prereg = _minimal_prereg_v1()
        prereg["version"] = "2"
        # No supersedes/changeJustification — would be bypass if not caught
        p = self._write_prereg(tmp_path, prereg)
        with pytest.raises(ValueError):
            load_prereg(p)

    def test_version_zero_raises(self, tmp_path):
        """version 0 is not a valid version number."""
        prereg = _minimal_prereg_v1()
        prereg["version"] = 0
        p = self._write_prereg(tmp_path, prereg)
        with pytest.raises(ValueError):
            load_prereg(p)

    def test_bool_true_version_raises(self, tmp_path):
        """True (bool) must not be accepted as version=1 (isinstance(True, int) is True in Python)."""
        prereg = _minimal_prereg_v1()
        prereg["version"] = True
        p = self._write_prereg(tmp_path, prereg)
        with pytest.raises(ValueError):
            load_prereg(p)

    def test_int_version_1_passes(self, tmp_path):
        prereg = _minimal_prereg_v1()
        prereg["version"] = 1
        p = self._write_prereg(tmp_path, prereg)
        result = load_prereg(p)
        assert result["version"] == 1

    def test_int_version_2_with_amendment_keys_passes(self, tmp_path):
        prereg = _minimal_prereg_v1()
        prereg["version"] = 2
        prereg["supersedes"] = "axs-mechanism-v1"
        prereg["changeJustification"] = "Added AXS-001 arm"
        p = self._write_prereg(tmp_path, prereg)
        result = load_prereg(p)
        assert result["version"] == 2


# ---------------------------------------------------------------------------
# Fix 3: Duplicate reportHash with conflicting metadata must raise ValueError
# ---------------------------------------------------------------------------


class TestDuplicateHashConflict:
    """Same reportHash + different fields must raise ValueError."""

    def _fresh_ledger(self, tmp_path) -> Path:
        p = tmp_path / "run_ledger.json"
        p.write_text(json.dumps({"ledgerVersion": 1, "entries": []}), encoding="utf-8")
        return p

    def test_identical_reappend_is_noop(self, tmp_path):
        """Exact same entry re-appended must be silently ignored (idempotent)."""
        ledger_path = self._fresh_ledger(tmp_path)
        entry = _minimal_ledger_entry()
        append_ledger_entry(ledger_path, entry)
        append_ledger_entry(ledger_path, entry)  # identical re-append
        ledger = load_ledger(ledger_path)
        assert len(ledger["entries"]) == 1

    def test_same_hash_different_report_id_raises(self, tmp_path):
        """Same reportHash but different reportId must raise ValueError (integrity anomaly)."""
        ledger_path = self._fresh_ledger(tmp_path)
        entry1 = _minimal_ledger_entry()
        append_ledger_entry(ledger_path, entry1)

        entry2 = dict(entry1)
        entry2["reportId"] = "conflicting-report-id"
        # same reportHash, different reportId -> integrity anomaly
        with pytest.raises(ValueError):
            append_ledger_entry(ledger_path, entry2)
