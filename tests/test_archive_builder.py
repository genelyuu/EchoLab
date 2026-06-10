"""Tests for echo_bench.archive.builder (Task A-008)."""

from __future__ import annotations

import pytest

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.cards.schema import CARD_FIELDS

yaml = pytest.importorskip("yaml")

BASES_CFG = "configs/basis/bases.yaml"
ARCHIVE_CFG = "configs/archive/archive.yaml"


def _archive_cfg():
    with open(ARCHIVE_CFG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_reproducible_archive_hash():
    bases = load_bases(BASES_CFG)
    cfg = _archive_cfg()
    a = build_archive(bases, cfg, 42)
    b = build_archive(bases, cfg, 42)
    assert a["archiveHash"] == b["archiveHash"]
    assert a["cards"] == b["cards"]


def test_different_seed_different_hash():
    bases = load_bases(BASES_CFG)
    cfg = _archive_cfg()
    a = build_archive(bases, cfg, 42)
    b = build_archive(bases, cfg, 7)
    assert a["archiveHash"] != b["archiveHash"]


def test_size_within_pool():
    bases = load_bases(BASES_CFG)
    cfg = _archive_cfg()
    a = build_archive(bases, cfg, 42)
    assert len(a["cards"]) <= int(cfg["pool_size"])
    assert len(a["cards"]) > 0


def test_archive_holds_only_card_records():
    bases = load_bases(BASES_CFG)
    cfg = _archive_cfg()
    a = build_archive(bases, cfg, 42)
    assert set(a.keys()) == {"cards", "archiveHash"}
    for rec in a["cards"]:
        assert set(rec.keys()) == set(CARD_FIELDS)


def test_basis_count_covered():
    bases = load_bases(BASES_CFG)
    cfg = _archive_cfg()
    a = build_archive(bases, cfg, 42)
    bases_seen = {rec["basis"] for rec in a["cards"]}
    assert bases_seen == set(bases.keys())
