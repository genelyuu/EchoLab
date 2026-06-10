"""Tests for echo_bench.basis.renderers.stub (Phase 1 STUB renderer)."""

from __future__ import annotations

import random

from echo_bench.basis.renderers import stub
from echo_bench.utils.hash import raster_hash

PARAMS = {"depth": 5.0, "branch_angle": 30.0, "branch_ratio": 0.6, "jitter": 0.2}


def test_same_inputs_identical_bytes_and_hash():
    a = stub.render("B1", 11, PARAMS, "stub-r1")
    b = stub.render("B1", 11, PARAMS, "stub-r1")
    assert a == b
    assert raster_hash(a) == raster_hash(b)
    assert len(a) == stub.RASTER_SIZE


def test_different_seed_different_bytes():
    a = stub.render("B1", 11, PARAMS, "stub-r1")
    b = stub.render("B1", 12, PARAMS, "stub-r1")
    assert a != b


def test_different_basis_different_bytes():
    a = stub.render("B1", 11, PARAMS, "stub-r1")
    b = stub.render("B2", 11, PARAMS, "stub-r1")
    assert a != b


def test_different_renderer_version_different_bytes():
    a = stub.render("B1", 11, PARAMS, "stub-r1")
    b = stub.render("B1", 11, PARAMS, "stub-r2")
    assert a != b


def test_no_global_rng_contamination():
    # Perturb the global RNG between renders; output must be unaffected.
    random.seed(0)
    a = stub.render("B1", 99, PARAMS, "stub-r1")
    random.seed(123456)
    [random.random() for _ in range(50)]
    b = stub.render("B1", 99, PARAMS, "stub-r1")
    assert a == b
