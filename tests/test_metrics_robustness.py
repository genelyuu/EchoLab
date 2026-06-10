"""Tests for echo_bench.metrics.robustness (Task D-003).

Covers: fault transforms are pure (no input mutation) and deterministic; they
are enumerated in FAULTS; robustness_score is bounded [0, 1] and deterministic;
metadata carries both traceHash references.
"""

from __future__ import annotations

import copy

from echo_bench.metrics.robustness import (
    FAULTS,
    SALIENCE_SCORE_MAX,
    SALIENCE_SCORE_MIN,
    basis_dropout,
    pool_shrink,
    robustness_score,
    robustness_score_with_metadata,
    salience_perturb,
)


def _pool() -> list:
    return [
        {"cardId": "c1", "basis": "B1", "salienceScore": 0.10},
        {"cardId": "c2", "basis": "B2", "salienceScore": 0.50},
        {"cardId": "c3", "basis": "B3", "salienceScore": 0.90},
        {"cardId": "c4", "basis": "B1", "salienceScore": 0.30},
        {"cardId": "c5", "basis": "B4", "salienceScore": 0.70},
        {"cardId": "c6", "basis": "B2", "salienceScore": 0.20},
    ]


def test_faults_registry_enumerates_transforms():
    assert set(FAULTS) == {"pool_shrink", "basis_dropout", "salience_perturb"}
    assert FAULTS["pool_shrink"] is pool_shrink
    assert FAULTS["basis_dropout"] is basis_dropout
    assert FAULTS["salience_perturb"] is salience_perturb


def test_pool_shrink_pure_and_deterministic():
    pool = _pool()
    before = copy.deepcopy(pool)
    out1 = pool_shrink(pool, frac=0.5, seed=7)
    out2 = pool_shrink(pool, frac=0.5, seed=7)
    # Pure: input unchanged.
    assert pool == before
    # Deterministic.
    assert out1 == out2
    # Drops half (6 -> 3).
    assert len(out1) == 3
    # Returned cards are independent copies.
    out1[0]["salienceScore"] = 999
    assert pool == before


def test_pool_shrink_bounds():
    pool = _pool()
    assert pool_shrink(pool, frac=0.0, seed=1) == pool
    assert pool_shrink(pool, frac=1.0, seed=1) == []
    # frac clamped.
    assert pool_shrink(pool, frac=2.0, seed=1) == []
    assert pool_shrink(pool, frac=-1.0, seed=1) == pool


def test_basis_dropout_pure_and_deterministic():
    pool = _pool()
    before = copy.deepcopy(pool)
    out1 = basis_dropout(pool, "B1")
    out2 = basis_dropout(pool, "B1")
    assert pool == before
    assert out1 == out2
    assert all(c["basis"] != "B1" for c in out1)
    assert len(out1) == 4  # two B1 cards removed
    # Multiple bases.
    out_multi = basis_dropout(pool, ["B1", "B2"])
    assert all(c["basis"] not in {"B1", "B2"} for c in out_multi)
    assert len(out_multi) == 2


def test_salience_perturb_pure_deterministic_and_bounded():
    pool = _pool()
    before = copy.deepcopy(pool)
    out1 = salience_perturb(pool, delta=0.2, seed=42)
    out2 = salience_perturb(pool, delta=0.2, seed=42)
    assert pool == before
    assert out1 == out2
    assert len(out1) == len(pool)
    for card in out1:
        assert SALIENCE_SCORE_MIN <= card["salienceScore"] <= SALIENCE_SCORE_MAX
    # delta=0 is the identity on salience.
    out0 = salience_perturb(pool, delta=0.0, seed=42)
    for orig, new in zip(pool, out0):
        assert new["salienceScore"] == orig["salienceScore"]


def test_salience_perturb_order_independent_per_card():
    pool = _pool()
    shuffled = list(reversed(pool))
    out_pool = {c["cardId"]: c["salienceScore"] for c in salience_perturb(pool, 0.2, 9)}
    out_shuf = {c["cardId"]: c["salienceScore"] for c in salience_perturb(shuffled, 0.2, 9)}
    assert out_pool == out_shuf


def test_robustness_score_bounded_and_zero_when_identical():
    baseline = {
        "traceHash": "ha",
        "coordinate_coverage": 0.5,
        "artifact_diversity": 0.4,
        "redundancy_rate": 0.1,
        "round_coherence": 0.8,
    }
    assert robustness_score(baseline, baseline) == 0.0
    faulted = dict(baseline)
    faulted["traceHash"] = "hb"
    faulted["coordinate_coverage"] = 0.9  # diff 0.4
    faulted["redundancy_rate"] = 0.3  # diff 0.2
    score = robustness_score(baseline, faulted)
    assert 0.0 <= score <= 1.0
    # mean of |.4|,0,|.2|,0 over 4 keys = 0.15
    assert abs(score - 0.15) < 1e-12


def test_robustness_score_deterministic():
    a = {"traceHash": "a", "m": 0.2, "n": 0.7}
    b = {"traceHash": "b", "m": 0.5, "n": 0.1}
    assert robustness_score(a, b) == robustness_score(a, b)


def test_robustness_score_no_shared_keys_is_zero():
    assert robustness_score({"traceHash": "a"}, {"traceHash": "b"}) == 0.0


def test_robustness_metadata_carries_both_trace_hashes():
    a = {"traceHash": "hash-a", "m": 0.2}
    b = {"traceHash": "hash-b", "m": 0.8}
    md = robustness_score_with_metadata(a, b)
    assert md["baselineTraceHash"] == "hash-a"
    assert md["faultedTraceHash"] == "hash-b"
    assert md["sharedKeys"] == ["m"]
    assert abs(md["value"] - 0.6) < 1e-12
