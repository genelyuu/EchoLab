"""Tests for echo_bench.archive.coverage (Task A-009).

Covers: deterministic report, per-basis counts sum to archive size, carried
archiveHash, stable reportHash, and the documented report fields.
"""

from __future__ import annotations

import json

import pytest

from echo_bench.archive.builder import build_archive
from echo_bench.archive.coverage import coverage_report, write_report
from echo_bench.basis.schema import load_bases

yaml = pytest.importorskip("yaml")

BASES_CFG = "configs/basis/bases.yaml"
ARCHIVE_CFG = "configs/archive/archive.yaml"


def _archive():
    bases = load_bases(BASES_CFG)
    with open(ARCHIVE_CFG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return build_archive(bases, cfg, 42)


def test_report_is_deterministic():
    arc = _archive()
    a = coverage_report(arc)
    b = coverage_report(arc)
    assert a == b
    assert a["reportHash"] == b["reportHash"]


def test_report_has_documented_fields():
    rep = coverage_report(_archive())
    for key in (
        "archiveHash",
        "archiveSize",
        "perBasisCounts",
        "complexityBandHistogram",
        "coordinateCoverage",
        "reportHash",
    ):
        assert key in rep


def test_per_basis_counts_sum_to_archive_size():
    arc = _archive()
    rep = coverage_report(arc)
    assert sum(rep["perBasisCounts"].values()) == rep["archiveSize"]
    assert rep["archiveSize"] == len(arc["cards"])


def test_band_histogram_sums_to_archive_size():
    rep = coverage_report(_archive())
    assert sum(rep["complexityBandHistogram"].values()) == rep["archiveSize"]


def test_carries_archive_hash():
    arc = _archive()
    rep = coverage_report(arc)
    assert rep["archiveHash"] == arc["archiveHash"]


def test_report_hash_stable_and_excludes_itself():
    from echo_bench.utils.hash import canonical_hash

    rep = coverage_report(_archive())
    body = {k: v for k, v in rep.items() if k != "reportHash"}
    assert canonical_hash(body) == rep["reportHash"]


def test_coordinate_coverage_shape():
    rep = coverage_report(_archive())
    cov = rep["coordinateCoverage"]
    assert cov["dims"] == 4
    assert len(cov["perChannel"]) == 4
    for ch in cov["perChannel"]:
        assert 0.0 <= ch["min"] <= 1.0
        assert 0.0 <= ch["max"] <= 1.0
        assert 0.0 <= ch["binOccupancy"] <= 1.0
    assert 0.0 <= cov["cellOccupancy"] <= 1.0


def test_write_report_roundtrip(tmp_path):
    arc = _archive()
    out = tmp_path / "archive_coverage.json"
    rep = write_report(arc, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == rep
    assert loaded["reportHash"] == rep["reportHash"]
