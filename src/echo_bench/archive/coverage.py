"""Candidate-archive coverage report for ECHO-Bench (Task A-009).

Produces a system-level coverage report over a built archive describing basis
distribution, coordinate coverage spread, and complexity-band spread. The report
is deterministic from a fixed archive, carries the archive's ``archiveHash``, and
is itself hashed (``reportHash``).

Claims are system-level only: counts, spreads, and histograms over the
controlled action space. No user-preference, population, emotion, wellbeing, or
real-world generalization claim appears anywhere.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

_logger = get_logger(__name__)

# Canonical basis order for stable, deterministic histogram keys.
BASIS_ORDER = ("B1", "B2", "B3", "B4")
# Canonical complexity-band order (matches echo_bench.cards.metrics bands).
BAND_ORDER = ("low", "mid", "high")
# Number of bins per coordinate channel for the coverage-spread grid.
COORD_BINS = 5

_ROUND = 12


def _per_basis_counts(cards: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count cards per basis id, in canonical order, including zero counts."""
    counts = {b: 0 for b in BASIS_ORDER}
    for rec in cards:
        b = str(rec.get("basis"))
        counts[b] = counts.get(b, 0) + 1
    # Deterministic ordering: canonical bases first, then any extras sorted.
    ordered = {b: counts[b] for b in BASIS_ORDER if b in counts}
    for b in sorted(k for k in counts if k not in BASIS_ORDER):
        ordered[b] = counts[b]
    return ordered


def _band_histogram(cards: list[Mapping[str, Any]]) -> dict[str, int]:
    """Histogram of complexityBand over the archive, in canonical band order."""
    hist = {band: 0 for band in BAND_ORDER}
    for rec in cards:
        band = str(rec.get("complexityBand"))
        hist[band] = hist.get(band, 0) + 1
    ordered = {band: hist[band] for band in BAND_ORDER if band in hist}
    for band in sorted(k for k in hist if k not in BAND_ORDER):
        ordered[band] = hist[band]
    return ordered


def _coordinate_coverage(cards: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage spread over the coordinateContribution channels.

    For each channel, reports the observed [min, max] range and the fraction of
    occupied bins on a fixed :data:`COORD_BINS`-bin grid over [0, 1]. Also
    reports the fraction of distinct multi-dimensional grid cells occupied — a
    system-level diversity measure of how the archive spreads in coordinate
    space.
    """
    coords = [rec.get("coordinateContribution") or [] for rec in cards]
    coords = [list(c) for c in coords if c]
    if not coords:
        return {
            "dims": 0,
            "perChannel": [],
            "occupiedCells": 0,
            "cellOccupancy": 0.0,
        }

    dims = min(len(c) for c in coords)
    per_channel: list[dict[str, float]] = []
    for d in range(dims):
        col = [float(c[d]) for c in coords]
        lo, hi = min(col), max(col)
        bins = set()
        for v in col:
            idx = int(min(COORD_BINS - 1, max(0, int(v * COORD_BINS))))
            bins.add(idx)
        per_channel.append(
            {
                "min": round(lo, _ROUND),
                "max": round(hi, _ROUND),
                "range": round(hi - lo, _ROUND),
                "occupiedBins": len(bins),
                "binOccupancy": round(len(bins) / COORD_BINS, _ROUND),
            }
        )

    # Multi-dim cell occupancy: distinct grid cells / theoretical max (capped by
    # the number of cards, since at most one cell per card can be added).
    cells = set()
    for c in coords:
        cell = tuple(
            int(min(COORD_BINS - 1, max(0, int(float(c[d]) * COORD_BINS))))
            for d in range(dims)
        )
        cells.add(cell)
    denom = min(len(coords), COORD_BINS ** dims) if dims else 1
    cell_occ = (len(cells) / denom) if denom else 0.0

    return {
        "dims": dims,
        "perChannel": per_channel,
        "occupiedCells": len(cells),
        "cellOccupancy": round(cell_occ, _ROUND),
    }


def coverage_report(archive_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic, system-level coverage report over an archive.

    Args:
        archive_dict: The output of
            :func:`echo_bench.archive.builder.build_archive`
            (``{"cards": [...], "archiveHash": <hex>}``).

    Returns:
        A report dict with ``archiveSize``, ``perBasisCounts``,
        ``complexityBandHistogram``, ``coordinateCoverage``, the carried
        ``archiveHash``, and a ``reportHash`` over the report minus that hash.
    """
    cards = list(archive_dict.get("cards", []))
    archive_hash = str(archive_dict.get("archiveHash", ""))

    report_body = {
        "archiveHash": archive_hash,
        "archiveSize": len(cards),
        "perBasisCounts": _per_basis_counts(cards),
        "complexityBandHistogram": _band_histogram(cards),
        "coordinateCoverage": _coordinate_coverage(cards),
    }

    report_hash = canonical_hash(report_body)
    report = dict(report_body)
    report["reportHash"] = report_hash

    log_ko(
        _logger,
        f"커버리지 리포트 생성 완료: archiveSize={len(cards)} "
        f"archiveHash={archive_hash[:12]} reportHash={report_hash[:12]}",
    )
    return report


def write_report(archive_dict: Mapping[str, Any],
                 path: str | Path = "outputs/reports/archive_coverage.json") -> dict[str, Any]:
    """Compute the coverage report and write it to ``path`` as JSON.

    Creates parent directories as needed. Returns the report dict. The written
    JSON is deterministic (sorted keys) and carries both ``archiveHash`` and
    ``reportHash`` — no unhashed output.

    Args:
        archive_dict: Built archive (see :func:`coverage_report`).
        path: Output path; defaults to
            ``outputs/reports/archive_coverage.json``.

    Returns:
        The report dict that was written.
    """
    report = coverage_report(archive_dict)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    log_ko(_logger, f"커버리지 리포트 기록 경로={out}")
    return report
