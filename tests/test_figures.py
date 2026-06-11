"""Tests for E-010 frontier figure data + optional rendering, and the
E-022 channel-resolved excess-separability frontier."""
import json

from echo_bench.experiments.figures import (
    SEPARABILITY_FRONTIER_SCHEMA,
    build_frontier_data,
    build_separability_frontier_data,
    render_figures,
    render_separability_figures,
    write_frontier_data,
    write_separability_frontier_data,
)


def _write_min_reports(reports_dir):
    e2 = {
        "experiment": "E2_POLICY_UTILITY", "reportHash": "e2hash",
        "table": [
            {"policy": "RANDOM", "coordinate_coverage": 0.4, "traceOnly": True},
            {"policy": "TRACE_GREEDY", "coordinate_coverage": 0.7, "traceOnly": True},
        ],
    }
    e3 = {
        "experiment": "E3_AUDIT", "reportHash": "e3hash",
        "leakage": {"table": [
            {"policy": "RANDOM", "leakage_proxy": 0.2},
            {"policy": "TRACE_GREEDY", "leakage_proxy": 0.5},
        ]},
    }
    (reports_dir / "e2_policy_min.json").write_text(json.dumps(e2))
    (reports_dir / "e3_audit_min.json").write_text(json.dumps(e3))


def test_frontier_data_deterministic(tmp_path):
    _write_min_reports(tmp_path)
    a = build_frontier_data(tmp_path)
    b = build_frontier_data(tmp_path)
    assert a["dataHash"] == b["dataHash"]
    policies = {p["policy"] for p in a["points"]}
    assert policies == {"RANDOM", "TRACE_GREEDY"}
    pt = next(p for p in a["points"] if p["policy"] == "TRACE_GREEDY")
    assert pt["utility"] == 0.7 and pt["leakage_proxy"] == 0.5
    assert set(a["sourceReportHashes"]) == {"e2hash", "e3hash"}


def test_render_is_graceful_without_matplotlib(tmp_path):
    _write_min_reports(tmp_path)
    data = build_frontier_data(tmp_path)
    out = render_figures(data, tmp_path)  # returns [] if matplotlib absent
    assert isinstance(out, list)


def test_empty_reports_dir_graceful(tmp_path):
    data = build_frontier_data(tmp_path)  # empty dir, no reports
    assert data["points"] == []
    assert isinstance(data["dataHash"], str) and data["dataHash"]


def test_policy_intersection(tmp_path):
    e2 = {
        "experiment": "E2_POLICY_UTILITY", "reportHash": "e2hash",
        "table": [
            {"policy": "RANDOM", "coordinate_coverage": 0.4, "traceOnly": True},
            {"policy": "TRACE_GREEDY", "coordinate_coverage": 0.7, "traceOnly": True},
        ],
    }
    e3 = {
        "experiment": "E3_AUDIT", "reportHash": "e3hash",
        "leakage": {"table": [
            {"policy": "RANDOM", "leakage_proxy": 0.2},
        ]},
    }
    (tmp_path / "e2_policy_min.json").write_text(json.dumps(e2))
    (tmp_path / "e3_audit_min.json").write_text(json.dumps(e3))
    data = build_frontier_data(tmp_path)
    assert {p["policy"] for p in data["points"]} == {"RANDOM"}


def test_note_labels_proxy(tmp_path):
    _write_min_reports(tmp_path)
    data = build_frontier_data(tmp_path)
    assert "PROXY" in data["note"]
    assert "No real-world generalization" in data["note"]


def test_write_frontier_data_roundtrip(tmp_path):
    _write_min_reports(tmp_path)
    data = build_frontier_data(tmp_path)
    path = write_frontier_data(data, reports_dir=tmp_path)
    with open(path, "r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    assert reloaded["dataHash"] == data["dataHash"]


def test_index_pins_current_run_over_lexicographic_latest(tmp_path):
    """benchmark_index.json must override the lexicographically-last report.

    Report filenames are hash-prefixed, so a stale report can sort last. The
    index pins the current run by seedBatchId; the frontier must follow it.
    """
    # Current run (pinned by index), seedBatchId starts with "aaaa..." (sorts FIRST).
    current = {
        "experiment": "E2_POLICY_UTILITY", "reportHash": "current",
        "table": [{"policy": "RANDOM", "coordinate_coverage": 0.30, "traceOnly": True}],
    }
    # Stale report, seedBatchId starts with "ffff..." (sorts LAST -> _latest would pick it).
    stale = {
        "experiment": "E2_POLICY_UTILITY", "reportHash": "stale",
        "table": [{"policy": "RANDOM", "coordinate_coverage": 0.99, "traceOnly": True}],
    }
    (tmp_path / "e2_policy_aaaaaaaaaaaa.json").write_text(json.dumps(current))
    (tmp_path / "e2_policy_ffffffffffff.json").write_text(json.dumps(stale))
    e3 = {
        "experiment": "E3_AUDIT", "reportHash": "e3",
        "leakage": {"table": [{"policy": "RANDOM", "leakage_proxy": 0.2}]},
    }
    (tmp_path / "e3_audit_aaaaaaaaaaaa.json").write_text(json.dumps(e3))
    index = {
        "experiments": [
            {"experiment": "E2_POLICY_UTILITY", "seedBatchId": "aaaaaaaaaaaa0000"},
            {"experiment": "E3_AUDIT", "seedBatchId": "aaaaaaaaaaaa0000"},
        ],
    }
    (tmp_path / "benchmark_index.json").write_text(json.dumps(index))

    data = build_frontier_data(tmp_path)
    pt = next(p for p in data["points"] if p["policy"] == "RANDOM")
    assert pt["utility"] == 0.30  # pinned current run, NOT the lexicographically-last 0.99
    assert "current" in data["sourceReportHashes"]
    assert "stale" not in data["sourceReportHashes"]


# ---------------------------------------------------------------------------
# E-022: channel-resolved excess-separability frontier
# ---------------------------------------------------------------------------

_CHANNELS = ("slate", "selection", "combined")


def _excess_block(channel, mean, ci_low, ci_high, *, sufficient_n=True,
                  sign_consistent=True, unsaturated=True):
    return {
        "channel": channel,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": "bootstrap",
        "n": 3,
        "std": 0.01,
        "perFamilyValues": [mean, mean, mean],
        "sufficient_n": sufficient_n,
        "signConsistent": sign_consistent,
        "allFamiliesUnsaturated": unsaturated,
    }


def _conditions_block(channel, hold):
    return {
        "channel": channel,
        "condition1_excess_positive_ci": hold,
        "condition2_not_saturated": hold,
        "condition3_channel_named": True,
        "condition4_cross_family_stable": hold,
        "allConditionsHold": hold,
    }


def _write_sep_reports(reports_dir, *, metric_label="leakage_proxy",
                       with_overlap_caveat=False):
    """Minimal synthetic E2 + leakage-diagnostic fixture pair (2 policies)."""
    e2 = {
        "experiment": "E2_POLICY_UTILITY", "reportHash": "e2hash",
        "table": [
            {"policy": "RANDOM", "coordinate_coverage": 0.4, "traceOnly": True},
            {"policy": "TRACE_LIN_UCB", "coordinate_coverage": 0.8, "traceOnly": True},
        ],
    }
    diag = {
        "experiment": "E_LEAKAGE_DIAGNOSTIC", "reportHash": "diaghash",
        "leakageMeta": {
            "isProxy": True,
            "disclaimer": "system-level proxy disclaimer text",
            "metric": metric_label,
        },
        "crossFamilyExcess": {
            "families": ["42", "7", "101"],
            "perPolicy": {
                # Track S candidate: positive excess everywhere, conditions hold.
                "TRACE_LIN_UCB": {
                    "slate": _excess_block("slate", 0.05, 0.04, 0.07),
                    "selection": _excess_block("selection", 0.20, 0.17, 0.22),
                    "combined": _excess_block("combined", 0.09, 0.07, 0.12),
                },
                # Track N candidate: negative excess, unsaturated.
                "RANDOM": {
                    "slate": _excess_block("slate", -0.02, -0.03, -0.01),
                    "selection": _excess_block("selection", -0.01, -0.02, -0.005),
                    "combined": _excess_block("combined", -0.02, -0.03, -0.01),
                },
            },
        },
        "trackLConditions": {
            "ladderRef": "docs/12_CLAIM_LADDER.md Section 5",
            "perPolicy": {
                "TRACE_LIN_UCB": {ch: _conditions_block(ch, True) for ch in _CHANNELS},
                "RANDOM": {ch: _conditions_block(ch, False) for ch in _CHANNELS},
            },
        },
    }
    if with_overlap_caveat:
        diag["overlapCaveat"] = {"note": "synthetic overlap caveat", "pairs": []}
    (reports_dir / "e2_policy_min.json").write_text(json.dumps(e2))
    (reports_dir / "leakage_diagnostic_min.json").write_text(json.dumps(diag))


def test_separability_join_and_carried_fields(tmp_path):
    _write_sep_reports(tmp_path)
    data = build_separability_frontier_data(tmp_path)
    assert data["schema"] == SEPARABILITY_FRONTIER_SCHEMA
    assert data["channels"] == ["slate", "selection", "combined"]
    assert data["families"] == ["42", "7", "101"]
    assert data["nFamilies"] == 3
    assert data["sourceReportHashes"] == {"e2": "e2hash", "leakageDiagnostic": "diaghash"}
    assert data["joinWarnings"] == []
    assert {p["policy"] for p in data["points"]} == {"RANDOM", "TRACE_LIN_UCB"}
    pt = next(p for p in data["points"] if p["policy"] == "TRACE_LIN_UCB")
    assert pt["utility"] == 0.8 and pt["traceOnly"] is True
    combined = pt["excess"]["combined"]
    assert combined == {
        "mean": 0.09, "ci_low": 0.07, "ci_high": 0.12,
        "sufficient_n": True, "signConsistent": True,
        "allFamiliesUnsaturated": True,
    }
    assert pt["trackSConditionsHold"] == {"slate": True, "selection": True, "combined": True}
    # leakageMeta echo leads with the G-020 primary label.
    echo = data["leakageMetaEcho"]
    assert echo["isProxy"] is True
    assert echo["primaryLabel"] == "probe_separability_proxy"
    assert echo["disclaimer"] == "system-level proxy disclaimer text"


def test_separability_track_assignment_s_and_n(tmp_path):
    _write_sep_reports(tmp_path)
    data = build_separability_frontier_data(tmp_path)
    by_policy = {p["policy"]: p["trackAssignment"] for p in data["points"]}
    assert by_policy["TRACE_LIN_UCB"] == "S"
    assert by_policy["RANDOM"] == "N"


def test_separability_track_assignment_indeterminate(tmp_path):
    """Positive CI excluding 0 but conditions NOT all holding -> indeterminate;
    saturated negative excess -> indeterminate (Track N needs no saturation)."""
    _write_sep_reports(tmp_path)
    diag = json.loads((tmp_path / "leakage_diagnostic_min.json").read_text())
    per_policy = diag["crossFamilyExcess"]["perPolicy"]
    # Positive combined excess, but Track S conditions fail (e.g. condition4).
    per_policy["POS_BUT_UNSTABLE"] = {
        ch: _excess_block(ch, 0.05, 0.02, 0.08, sign_consistent=False)
        for ch in _CHANNELS
    }
    diag["trackLConditions"]["perPolicy"]["POS_BUT_UNSTABLE"] = {
        ch: _conditions_block(ch, False) for ch in _CHANNELS
    }
    # Negative excess but saturated families: Track N requires no saturation.
    per_policy["NEG_BUT_SATURATED"] = {
        ch: _excess_block(ch, -0.05, -0.08, -0.02, unsaturated=False)
        for ch in _CHANNELS
    }
    diag["trackLConditions"]["perPolicy"]["NEG_BUT_SATURATED"] = {
        ch: _conditions_block(ch, False) for ch in _CHANNELS
    }
    (tmp_path / "leakage_diagnostic_min.json").write_text(json.dumps(diag))
    data = build_separability_frontier_data(tmp_path)
    by_policy = {p["policy"]: p["trackAssignment"] for p in data["points"]}
    assert by_policy["POS_BUT_UNSTABLE"] == "indeterminate"
    assert by_policy["NEG_BUT_SATURATED"] == "indeterminate"


def test_separability_join_warnings_on_one_sided_policies(tmp_path):
    _write_sep_reports(tmp_path)
    # Policy only in E2.
    e2 = json.loads((tmp_path / "e2_policy_min.json").read_text())
    e2["table"].append(
        {"policy": "ORACLE_STRATEGY", "coordinate_coverage": 0.9, "traceOnly": False})
    (tmp_path / "e2_policy_min.json").write_text(json.dumps(e2))
    # Policy only in the diagnostic.
    diag = json.loads((tmp_path / "leakage_diagnostic_min.json").read_text())
    diag["crossFamilyExcess"]["perPolicy"]["DIAG_ONLY"] = {
        ch: _excess_block(ch, -0.01, -0.02, -0.005) for ch in _CHANNELS}
    diag["trackLConditions"]["perPolicy"]["DIAG_ONLY"] = {
        ch: _conditions_block(ch, False) for ch in _CHANNELS}
    (tmp_path / "leakage_diagnostic_min.json").write_text(json.dumps(diag))

    data = build_separability_frontier_data(tmp_path)
    by_policy = {p["policy"]: p for p in data["points"]}
    # Never silently dropped: both one-sided policies are present with nulls.
    oracle = by_policy["ORACLE_STRATEGY"]
    assert oracle["excess"] == {ch: None for ch in _CHANNELS}
    assert oracle["trackSConditionsHold"] == {ch: None for ch in _CHANNELS}
    assert oracle["trackAssignment"] == "indeterminate"
    diag_only = by_policy["DIAG_ONLY"]
    assert diag_only["utility"] is None and diag_only["traceOnly"] is None
    assert diag_only["trackAssignment"] == "N"
    assert len(data["joinWarnings"]) == 2
    assert any("ORACLE_STRATEGY" in w for w in data["joinWarnings"])
    assert any("DIAG_ONLY" in w for w in data["joinWarnings"])


def test_separability_datahash_deterministic_and_roundtrip(tmp_path):
    _write_sep_reports(tmp_path)
    a = build_separability_frontier_data(tmp_path)
    b = build_separability_frontier_data(tmp_path)
    assert a["dataHash"] == b["dataHash"]
    # dataHash mirrors E-010: canonical hash over the dict minus dataHash.
    from echo_bench.utils.hash import canonical_hash
    body = {k: v for k, v in a.items() if k != "dataHash"}
    assert canonical_hash(body) == a["dataHash"]
    path = write_separability_frontier_data(a, reports_dir=tmp_path)
    assert path.endswith("frontier_separability_data.json")
    with open(path, "r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    assert reloaded["dataHash"] == a["dataHash"]
    assert canonical_hash({k: v for k, v in reloaded.items() if k != "dataHash"}) == a["dataHash"]


def test_separability_missing_reports_graceful(tmp_path):
    data = build_separability_frontier_data(tmp_path)  # empty reports dir
    assert data["points"] == []
    assert data["joinWarnings"] == []
    assert isinstance(data["dataHash"], str) and data["dataHash"]
    assert "missing" in data["note"].lower()


def test_separability_note_diagnostic_framing(tmp_path):
    _write_sep_reports(tmp_path)
    data = build_separability_frontier_data(tmp_path)
    note = data["note"]
    assert "probe_separability_proxy" in note
    assert "diagnostic" in note.lower()
    assert "PROXY" in note
    # PNG stays non-citable; hashed data JSON is the artifact.
    assert "PNG" in note


def test_separability_tolerates_g020_label_and_overlap_caveat(tmp_path):
    """Builder must accept reports with either leakageMeta.metric label and
    with/without the B-009 overlapCaveat block."""
    _write_sep_reports(tmp_path, metric_label="probe_separability_proxy",
                       with_overlap_caveat=True)
    data = build_separability_frontier_data(tmp_path)
    assert {p["policy"] for p in data["points"]} == {"RANDOM", "TRACE_LIN_UCB"}
    assert data["leakageMetaEcho"]["primaryLabel"] == "probe_separability_proxy"


def test_separability_render_graceful_without_matplotlib(tmp_path):
    _write_sep_reports(tmp_path)
    data = build_separability_frontier_data(tmp_path)
    out = render_separability_figures(data, tmp_path)  # [] if matplotlib absent
    assert isinstance(out, list)
