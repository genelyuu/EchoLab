"""Tests for the Phase 2 real per-basis renderers (Tasks A-002..A-005).

Each of B1..B4 must:
- render byte-identically for identical (seed, params, version),
- yield identical raster_hash for identical inputs,
- produce different bytes for a different seed,
- return a uint8 64x64 single-channel array,
- and the four bases must produce DISTINCT rasters for the same seed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from echo_bench.basis.renderers import RENDERERS, get_renderer
from echo_bench.basis.renderers import branching, flow_attractor
from echo_bench.basis.renderers import reaction_diffusion, topographic_fbm
from echo_bench.utils.hash import raster_hash

PARAMS = {
    "B1": {"depth": 5.0, "branch_angle": 30.0, "branch_ratio": 0.6, "jitter": 0.2},
    "B2": {"feed_rate": 0.045, "kill_rate": 0.062, "diffusion_u": 0.16,
           "iterations": 200.0},
    "B3": {"octaves": 5.0, "lacunarity": 2.0, "gain": 0.5, "scale": 4.0},
    "B4": {"coeff_a": -2.0, "coeff_b": 2.0, "steps": 800.0, "step_size": 0.04},
}
VERSIONS = {"B1": "b1-r1", "B2": "b2-r1", "B3": "b3-r1", "B4": "b4-r1"}
ALL = ("B1", "B2", "B3", "B4")


@pytest.mark.parametrize("basis", ALL)
def test_same_inputs_identical_bytes_and_hash(basis):
    r = get_renderer(basis)
    a = r(11, PARAMS[basis], VERSIONS[basis])
    b = r(11, PARAMS[basis], VERSIONS[basis])
    assert np.array_equal(a, b)
    assert a.tobytes() == b.tobytes()
    assert raster_hash(a) == raster_hash(b)


@pytest.mark.parametrize("basis", ALL)
def test_output_is_uint8_64x64(basis):
    r = get_renderer(basis)
    a = r(11, PARAMS[basis], VERSIONS[basis])
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.uint8
    assert a.shape == (64, 64)


@pytest.mark.parametrize("basis", ALL)
def test_different_seed_different_bytes(basis):
    r = get_renderer(basis)
    a = r(11, PARAMS[basis], VERSIONS[basis])
    b = r(12, PARAMS[basis], VERSIONS[basis])
    assert a.tobytes() != b.tobytes()
    assert raster_hash(a) != raster_hash(b)


@pytest.mark.parametrize("basis", ALL)
def test_different_version_different_bytes(basis):
    r = get_renderer(basis)
    a = r(11, PARAMS[basis], VERSIONS[basis])
    b = r(11, PARAMS[basis], VERSIONS[basis] + "-x")
    assert a.tobytes() != b.tobytes()


def test_four_bases_produce_distinct_rasters_same_seed():
    hashes = {}
    for basis in ALL:
        arr = get_renderer(basis)(7, PARAMS[basis], VERSIONS[basis])
        hashes[basis] = raster_hash(arr)
    assert len(set(hashes.values())) == 4


@pytest.mark.parametrize("basis", ALL)
def test_no_global_rng_contamination(basis):
    r = get_renderer(basis)
    random.seed(0)
    np.random.seed(0)
    a = r(99, PARAMS[basis], VERSIONS[basis])
    random.seed(123456)
    np.random.seed(123456)
    [random.random() for _ in range(50)]
    _ = np.random.random(50)
    b = r(99, PARAMS[basis], VERSIONS[basis])
    assert np.array_equal(a, b)


def test_registry_maps_all_four_bases():
    assert set(RENDERERS.keys()) == set(ALL)
    assert RENDERERS["B1"] is branching.render
    assert RENDERERS["B2"] is reaction_diffusion.render
    assert RENDERERS["B3"] is topographic_fbm.render
    assert RENDERERS["B4"] is flow_attractor.render


def test_unknown_basis_falls_back_to_stub():
    r = get_renderer("B9")  # unknown -> stub fallback (adapted signature)
    out = r(3, {"x": 1.0}, "stub-r1")
    # Stub returns flat bytes; the adapter must not raise.
    assert out is not None
