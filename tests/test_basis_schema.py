"""Tests for echo_bench.basis.schema (Task A-001)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from echo_bench.basis.schema import (
    REQUIRED_BASIS_IDS,
    BasisSpec,
    load_bases,
    sample_params,
    validate_params,
)

CONFIG = "configs/basis/bases.yaml"


def test_four_bases_load():
    specs = load_bases(CONFIG)
    assert set(specs.keys()) == set(REQUIRED_BASIS_IDS)
    for bid, spec in specs.items():
        assert isinstance(spec, BasisSpec)
        assert spec.basis == bid
        assert spec.rendererVersion
        assert spec.param_ranges
        for name, (lo, hi) in spec.param_ranges.items():
            assert lo <= hi


def test_sampling_within_bounds():
    specs = load_bases(CONFIG)
    rng = random.Random(123)
    for spec in specs.values():
        params = sample_params(spec, rng)
        assert set(params) == set(spec.param_ranges)
        for name, value in params.items():
            lo, hi = spec.param_ranges[name]
            assert lo <= value <= hi


def test_sampling_deterministic_same_seed():
    specs = load_bases(CONFIG)
    spec = specs["B1"]
    a = sample_params(spec, random.Random(7))
    b = sample_params(spec, random.Random(7))
    assert a == b


def test_out_of_bounds_validate_raises():
    specs = load_bases(CONFIG)
    spec = specs["B1"]
    bad = {name: hi + 100.0 for name, (lo, hi) in spec.param_ranges.items()}
    with pytest.raises(ValueError):
        validate_params(spec, bad)


def test_config_has_no_personal_fields(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8")
    low = text.lower()
    for token in ("persona", "emotion", "preference", "user_id", "demographic"):
        # Token may appear only in comments; ensure not as a real key by
        # confirming load succeeds (load rejects forbidden keys).
        pass
    # Loading must succeed (no forbidden keys present as real config keys).
    load_bases(CONFIG)

    # A config with a forbidden key must be rejected.
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "bases:\n"
        "  B1:\n"
        "    rendererVersion: \"stub-r1\"\n"
        "    persona: oops\n"
        "    param_ranges:\n"
        "      depth: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_bases(str(bad))


def test_reject_wrong_basis_set(tmp_path):
    bad = tmp_path / "wrong.yaml"
    bad.write_text(
        "bases:\n"
        "  B1:\n"
        "    rendererVersion: \"stub-r1\"\n"
        "    param_ranges:\n"
        "      depth: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_bases(str(bad))


def test_reject_inverted_range(tmp_path):
    bad = tmp_path / "inv.yaml"
    body = "bases:\n"
    for bid in REQUIRED_BASIS_IDS:
        rng = "[5, 1]" if bid == "B1" else "[1, 5]"
        body += (
            f"  {bid}:\n"
            f"    rendererVersion: \"stub-r1\"\n"
            f"    param_ranges:\n"
            f"      depth: {rng}\n"
        )
    bad.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_bases(str(bad))
