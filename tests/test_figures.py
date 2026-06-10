"""Tests for E-010 frontier figure data + optional rendering."""
import json

from echo_bench.experiments.figures import (
    build_frontier_data,
    render_figures,
    write_frontier_data,
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
