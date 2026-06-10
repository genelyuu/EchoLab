"""Tests for echo_bench.env.horizon (Task B-005)."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo_bench.env.horizon import default_h, load_horizon, validate_h

_HORIZON_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "horizon.yaml"
)


@pytest.fixture(scope="module")
def cfg():
    return load_horizon(_HORIZON_PATH)


def test_load_allowed_set(cfg):
    assert cfg["H_allowed"] == [4, 6, 8, 12, 16, 20]
    assert cfg["H_default"] == 8


def test_default_in_allowed_set(cfg):
    d = default_h(cfg)
    assert d in cfg["H_allowed"]
    assert d == 8


@pytest.mark.parametrize("h", [4, 6, 8, 12, 16, 20])
def test_validate_allowed_values_pass(cfg, h):
    assert validate_h(h, cfg) == h


@pytest.mark.parametrize("h", [0, 1, 5, 7, 10, 24, -4, 100])
def test_validate_outside_set_raises(cfg, h):
    with pytest.raises(ValueError):
        validate_h(h, cfg)


def test_validate_rejects_bool(cfg):
    # True == 1 numerically but is not a valid horizon.
    with pytest.raises(ValueError):
        validate_h(True, cfg)


def test_round_trip_deterministic():
    a = load_horizon(_HORIZON_PATH)
    b = load_horizon(_HORIZON_PATH)
    assert a == b
    assert a is not b  # fresh dict each load


def test_selected_h_is_recordable(cfg):
    # The selected H must be a plain int so it can go straight into a manifest.
    h = validate_h(default_h(cfg), cfg)
    assert isinstance(h, int)
    assert not isinstance(h, bool)


def test_load_rejects_default_outside_allowed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("H_allowed: [4, 6, 8]\nH_default: 99\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_horizon(bad)


def test_load_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text("H_default: 8\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_horizon(bad)


def test_load_rejects_empty_allowed(tmp_path):
    bad = tmp_path / "bad3.yaml"
    bad.write_text("H_allowed: []\nH_default: 8\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_horizon(bad)
