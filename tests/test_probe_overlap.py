"""Tests for echo_bench.probes.probe_overlap (Task B-007 / TRD B-008).

The overlap audit quantifies how redundant the registered strategy probes are
(pairwise_probe_overlap) and how varied each probe's selections are
(probe_entropy), flagging excessive-overlap pairs (high_overlap_pairs) at a
documented threshold. Probes remain controlled instrumented INPUT policies —
nothing here models people.
"""

from __future__ import annotations

import math

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.probes.probe_overlap import (
    PROBE_OVERLAP_THRESHOLD,
    probe_overlap_audit,
)
from echo_bench.probes.strategy_probes import PROBES, StrategyProbe


def _card(card_id, salience, edge, coord=(0.0, 0.0, 0.0), band="low"):
    return {
        "cardId": card_id,
        "basis": "B1",
        "complexityBand": band,
        "salienceScore": salience,
        "coordinateContribution": list(coord),
        "visualMetrics": {"edgeDensity": edge, "spatialFrequency": 0.5},
    }


class _MaxSalienceProbe(StrategyProbe):
    """Test-only controlled input rule: highest observable salienceScore."""

    name = "TEST_MAX_SALIENCE"
    version = "p1"

    def _score(self, card, trace):
        return float(card.get("salienceScore", 0.0))


class _MaxSalienceCloneProbe(_MaxSalienceProbe):
    """Identical scoring rule under a different name (redundant by design)."""

    name = "TEST_MAX_SALIENCE_CLONE"


class _MinSalienceProbe(StrategyProbe):
    """Test-only controlled input rule: lowest observable salienceScore."""

    name = "TEST_MIN_SALIENCE"
    version = "p1"

    def _score(self, card, trace):
        return -float(card.get("salienceScore", 0.0))


def _contexts_unique_argmax(n=6):
    # Each slate has a unique salience max and a unique salience min, so the
    # test probes select without any tie-break involvement.
    contexts = []
    for i in range(n):
        slate = [
            _card(f"s{i}-a", 0.9, 0.1),
            _card(f"s{i}-b", 0.5, 0.9),
            _card(f"s{i}-c", 0.1, 0.5),
        ]
        contexts.append((slate, None))
    return contexts


# --------------------------------------------------------------------------- #
# Pairwise overlap
# --------------------------------------------------------------------------- #
def test_identical_probes_overlap_one_and_flagged():
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        _MaxSalienceCloneProbe.name: _MaxSalienceCloneProbe(),
    }
    result = probe_overlap_audit(_contexts_unique_argmax(), probes, seed=1)
    pair_key = "TEST_MAX_SALIENCE|TEST_MAX_SALIENCE_CLONE"
    assert result["pairwise_probe_overlap"] == {pair_key: 1.0}
    assert result["high_overlap_pairs"] == [
        {
            "probe_a": "TEST_MAX_SALIENCE",
            "probe_b": "TEST_MAX_SALIENCE_CLONE",
            "overlap": 1.0,
        }
    ]


def test_orthogonal_probes_overlap_zero_and_not_flagged():
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        _MinSalienceProbe.name: _MinSalienceProbe(),
    }
    result = probe_overlap_audit(_contexts_unique_argmax(), probes, seed=1)
    pair_key = "TEST_MAX_SALIENCE|TEST_MIN_SALIENCE"
    assert result["pairwise_probe_overlap"][pair_key] == 0.0
    assert result["high_overlap_pairs"] == []


def test_partial_overlap_fraction():
    # MAX salience vs HIGH edge density: agree exactly when the highest-salience
    # card is also the highest-edge card. Build 4 agree + 6 disagree contexts.
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        "PREFER_HIGH_EDGE_DENSITY": PROBES["PREFER_HIGH_EDGE_DENSITY"],
    }
    agree = [
        (
            [_card(f"a{i}-x", 0.9, 0.9), _card(f"a{i}-y", 0.1, 0.1)],
            None,
        )
        for i in range(4)
    ]
    disagree = [
        (
            [_card(f"d{i}-x", 0.9, 0.1), _card(f"d{i}-y", 0.1, 0.9)],
            None,
        )
        for i in range(6)
    ]
    result = probe_overlap_audit(agree + disagree, probes, seed=1)
    pair_key = "PREFER_HIGH_EDGE_DENSITY|TEST_MAX_SALIENCE"
    assert result["pairwise_probe_overlap"][pair_key] == pytest.approx(0.4)
    assert result["high_overlap_pairs"] == []


# --------------------------------------------------------------------------- #
# Threshold boundary (inclusive: overlap >= threshold flags the pair)
# --------------------------------------------------------------------------- #
def _contexts_with_agreement(n_agree, n_disagree):
    agree = [
        ([_card(f"a{i}-x", 0.9, 0.9), _card(f"a{i}-y", 0.1, 0.1)], None)
        for i in range(n_agree)
    ]
    disagree = [
        ([_card(f"d{i}-x", 0.9, 0.1), _card(f"d{i}-y", 0.1, 0.9)], None)
        for i in range(n_disagree)
    ]
    return agree + disagree


def test_threshold_boundary_exact_overlap_is_flagged():
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        "PREFER_HIGH_EDGE_DENSITY": PROBES["PREFER_HIGH_EDGE_DENSITY"],
    }
    # 9/10 agreement = 0.9 = default threshold -> flagged (inclusive).
    result = probe_overlap_audit(_contexts_with_agreement(9, 1), probes, seed=1)
    assert result["threshold"] == PROBE_OVERLAP_THRESHOLD == 0.9
    assert len(result["high_overlap_pairs"]) == 1
    assert result["high_overlap_pairs"][0]["overlap"] == pytest.approx(0.9)

    # 8/10 agreement = 0.8 < threshold -> not flagged.
    result = probe_overlap_audit(_contexts_with_agreement(8, 2), probes, seed=1)
    assert result["high_overlap_pairs"] == []

    # Caller-supplied stricter threshold un-flags the 0.9 pair.
    result = probe_overlap_audit(
        _contexts_with_agreement(9, 1), probes, seed=1, threshold=0.95
    )
    assert result["high_overlap_pairs"] == []


# --------------------------------------------------------------------------- #
# probe_entropy
# --------------------------------------------------------------------------- #
def test_probe_entropy_constant_probe_is_zero():
    # Across contexts sharing the SAME slate, a deterministic probe always
    # selects the same cardId -> entropy 0.0 bits.
    slate = [_card("e-x", 0.9, 0.9), _card("e-y", 0.1, 0.1)]
    contexts = [(slate, None) for _ in range(5)]
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        _MinSalienceProbe.name: _MinSalienceProbe(),
    }
    result = probe_overlap_audit(contexts, probes, seed=1)
    assert result["probe_entropy"]["TEST_MAX_SALIENCE"] == 0.0
    assert result["probe_entropy"]["TEST_MIN_SALIENCE"] == 0.0


def test_probe_entropy_uniform_distinct_selections():
    # 4 contexts, each with a distinct unique-argmax card -> uniform over 4
    # distinct cardIds -> exactly log2(4) = 2.0 bits.
    contexts = _contexts_unique_argmax(4)
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        _MinSalienceProbe.name: _MinSalienceProbe(),
    }
    result = probe_overlap_audit(contexts, probes, seed=1)
    assert result["probe_entropy"]["TEST_MAX_SALIENCE"] == pytest.approx(
        math.log2(4)
    )


# --------------------------------------------------------------------------- #
# Determinism + full-registry audit
# --------------------------------------------------------------------------- #
def test_audit_is_deterministic():
    contexts = _contexts_unique_argmax(5)
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        _MinSalienceProbe.name: _MinSalienceProbe(),
    }
    first = probe_overlap_audit(contexts, probes, seed=42)
    for _ in range(3):
        assert probe_overlap_audit(contexts, probes, seed=42) == first


def test_full_registry_audit_covers_all_pairs():
    # Default probes argument = the full PROBES registry (7 probes, 21 pairs).
    trace = TraceState()
    trace.append_round(
        {
            "candidatePoolHash": "pool-1",
            "slate": ["a", "b"],
            "selectedCardId": "a",
            "coordinateContribution": [1.0, 1.0, 1.0],
            "complexityBand": "mid",
            "salienceScore": 0.3,
            "slotPermutation": [0, 1],
        }
    )
    slate = [
        _card("r-a", 0.9, 0.9, coord=(0.1, 0.1, 0.1), band="high"),
        _card("r-b", 0.1, 0.1, coord=(5.0, 5.0, 5.0), band="low"),
        _card("r-c", 0.5, 0.5, coord=(1.0, 1.0, 1.0), band="mid"),
    ]
    result = probe_overlap_audit([(slate, trace), (slate, None)], seed=7)
    n = len(PROBES)
    assert result["probes"] == sorted(PROBES)
    assert len(result["pairwise_probe_overlap"]) == n * (n - 1) // 2
    assert set(result["probe_entropy"]) == set(PROBES)
    assert result["n_contexts"] == 2
    assert result["probe_versions"] == {
        name: PROBES[name].probe_version() for name in PROBES
    }
    for overlap in result["pairwise_probe_overlap"].values():
        assert 0.0 <= overlap <= 1.0
    for pair in result["high_overlap_pairs"]:
        assert pair["overlap"] >= result["threshold"]


# --------------------------------------------------------------------------- #
# Validation (fail closed)
# --------------------------------------------------------------------------- #
def test_empty_contexts_raises():
    with pytest.raises(ValueError):
        probe_overlap_audit([], seed=1)


def test_fewer_than_two_probes_raises():
    with pytest.raises(ValueError):
        probe_overlap_audit(
            _contexts_unique_argmax(2),
            {_MaxSalienceProbe.name: _MaxSalienceProbe()},
            seed=1,
        )


def test_threshold_out_of_range_raises():
    probes = {
        _MaxSalienceProbe.name: _MaxSalienceProbe(),
        _MinSalienceProbe.name: _MinSalienceProbe(),
    }
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            probe_overlap_audit(
                _contexts_unique_argmax(2), probes, seed=1, threshold=bad
            )
