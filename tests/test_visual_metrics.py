"""Tests for the Phase 2 real visual metric extractor (Task A-006).

Covers: documented keys, determinism for an identical raster, complexityBand
derived from thresholds, coordinateContribution fixed length & in [0, 1], and
back-compat acceptance of a flat bytes raster.
"""

from __future__ import annotations

import numpy as np
import pytest

from echo_bench.basis.renderers import get_renderer
from echo_bench.cards.metrics import (
    COMPLEXITY_BANDS,
    COORDINATE_DIMS,
    RASTER_SIZE,
    complexity_band,
    extract,
)

EXPECTED_VM_KEYS = {
    "meanIntensity",
    "stdIntensity",
    "dynamicRange",
    "edgeDensity",
    "spatialFrequency",
    "occupancy",
    "saturationFraction",
    "entropy",
}

PARAMS_B3 = {"octaves": 5.0, "lacunarity": 2.0, "gain": 0.5, "scale": 4.0}


def _sample_raster():
    return get_renderer("B3")(11, PARAMS_B3, "b3-r1")


def test_extract_returns_documented_keys():
    out = extract(_sample_raster(), "B3", PARAMS_B3)
    assert set(out.keys()) == {
        "visualMetrics",
        "coordinateContribution",
        "complexityScore",
        "complexityBand",
        "salienceScore",
    }
    assert set(out["visualMetrics"].keys()) == EXPECTED_VM_KEYS


def test_extract_deterministic_for_identical_raster():
    arr = _sample_raster()
    a = extract(arr.copy(), "B3", PARAMS_B3)
    b = extract(arr.copy(), "B3", PARAMS_B3)
    assert a == b


def test_filter_keys_present():
    # The degenerate filter (A-007) depends on these exact keys.
    vm = extract(_sample_raster(), "B3", PARAMS_B3)["visualMetrics"]
    for key in ("stdIntensity", "dynamicRange", "saturationFraction"):
        assert key in vm


def test_coordinate_contribution_fixed_length_and_unit_interval():
    out = extract(_sample_raster(), "B3", PARAMS_B3)
    coord = out["coordinateContribution"]
    assert isinstance(coord, tuple)
    assert len(coord) == COORDINATE_DIMS
    for v in coord:
        assert 0.0 <= float(v) <= 1.0


def test_scalars_in_unit_interval():
    out = extract(_sample_raster(), "B3", PARAMS_B3)
    assert 0.0 <= out["complexityScore"] <= 1.0
    assert 0.0 <= out["salienceScore"] <= 1.0
    for v in out["visualMetrics"].values():
        assert 0.0 <= float(v) <= 1.0


def test_complexity_band_from_thresholds():
    out = extract(_sample_raster(), "B3", PARAMS_B3)
    assert out["complexityBand"] == complexity_band(out["complexityScore"])
    assert out["complexityBand"] in {name for name, _, _ in COMPLEXITY_BANDS}


@pytest.mark.parametrize(
    "score,expected",
    [(0.0, "low"), (0.2, "low"), (0.34, "mid"), (0.5, "mid"),
     (0.67, "high"), (1.0, "high")],
)
def test_band_boundaries(score, expected):
    assert complexity_band(score) == expected


def test_accepts_bytes_raster_back_compat():
    arr = _sample_raster()
    flat = arr.tobytes()
    assert len(flat) == RASTER_SIZE
    from_bytes = extract(flat, "B3", PARAMS_B3)
    from_arr = extract(arr, "B3", PARAMS_B3)
    assert from_bytes == from_arr


def test_blank_raster_low_complexity_and_salience():
    blank = np.zeros((64, 64), dtype=np.uint8)
    out = extract(blank, "B3", PARAMS_B3)
    assert out["complexityScore"] == 0.0
    assert out["salienceScore"] == 0.0
    assert out["complexityBand"] == "low"
