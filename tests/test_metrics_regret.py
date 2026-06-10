"""Tests for echo_bench.metrics.utility.regret_to_oracle (Task D-001, Phase 3).

``regret_to_oracle`` measures the normalized mean shortfall of a trace's achieved
system-level objective against the C-007 ``ORACLE_STRATEGY`` per-round reference.
The objective is the **coordinate-novelty of the selected card** — the L2 distance
between a round's ``coordinateContribution`` and the accumulated contributions of
all prior rounds — the SAME objective the oracle maximizes under its default
``PREFER_COORD_NOVELTY`` probe (oracle_strategy.py / PreferCoordNoveltyProbe).

Covers: regret == 0 when oracle == achieved; regret > 0 and bounded [0, 1] when
oracle strictly above achieved; ValueError on None / length mismatch; determinism;
the apples-to-apples match with the probe's own scoring; and the
compute_all_with_oracle wrapper.
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.utility import (
    REGRET_SCALE,
    compute_all_with_oracle,
    oracle_reference_from_objectives,
    regret_to_oracle,
)
from echo_bench.metrics.utility import _achieved_coordinate_novelty
from echo_bench.probes.strategy_probes import PreferCoordNoveltyProbe


def _record(
    selected: str,
    coord: list,
    band: str,
    salience: float = 0.5,
) -> dict:
    """Build one valid caller-supplied round record (no ``roundHash``)."""
    return {
        "candidatePoolHash": "poolhash-abc",
        "slate": ["c1", "c2", "c3", "c4"],
        "selectedCardId": selected,
        "coordinateContribution": coord,
        "complexityBand": band,
        "salienceScore": salience,
        "slotPermutation": [0, 1, 2, 3],
    }


def _build_trace(records=None) -> TraceState:
    """Build a small TraceState from records (a default 3-round trace)."""
    if records is None:
        records = [
            _record("c1", [0.10, 0.10, 0.10, 0.10], "low"),
            _record("c2", [0.40, 0.45, 0.50, 0.55], "mid"),
            _record("c3", [0.80, 0.85, 0.90, 0.95], "high"),
        ]
    trace = TraceState()
    for r in records:
        trace.append_round(r)
    return trace


# --------------------------------------------------------------------------- #
# Objective definition matches the oracle's probe (apples-to-apples).
# --------------------------------------------------------------------------- #
def test_achieved_objective_matches_coord_novelty_probe():
    """Per-round achieved objective == PreferCoordNoveltyProbe._score on prefix."""
    records = [
        _record("c1", [0.10, 0.10, 0.10, 0.10], "low"),
        _record("c2", [0.40, 0.45, 0.50, 0.55], "mid"),
        _record("c3", [0.80, 0.85, 0.90, 0.95], "high"),
    ]
    trace = _build_trace(records)
    achieved = _achieved_coordinate_novelty(trace)

    probe = PreferCoordNoveltyProbe()
    # Recompute the probe's score of each round's selected card against the
    # trace prefix of all strictly-earlier rounds.
    for i, rec in enumerate(records):
        prefix = _build_trace(records[:i])
        card = {"coordinateContribution": rec["coordinateContribution"]}
        expected = probe._score(card, prefix)
        assert achieved[i] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# oracle == achieved -> regret 0.0
# --------------------------------------------------------------------------- #
def test_regret_zero_when_oracle_equals_achieved():
    trace = _build_trace()
    achieved = _achieved_coordinate_novelty(trace)
    value = regret_to_oracle(trace, oracle_ref=achieved)
    assert value == 0.0


def test_regret_zero_when_oracle_below_achieved():
    """Shortfall is clipped at 0 per round; oracle below achieved -> 0.0."""
    trace = _build_trace()
    achieved = _achieved_coordinate_novelty(trace)
    below = [a - 0.5 for a in achieved]
    value = regret_to_oracle(trace, oracle_ref=below)
    assert value == 0.0


# --------------------------------------------------------------------------- #
# oracle strictly above achieved -> regret > 0 and bounded [0, 1]
# --------------------------------------------------------------------------- #
def test_regret_positive_and_bounded_when_oracle_above():
    trace = _build_trace()
    achieved = _achieved_coordinate_novelty(trace)
    above = [a + 0.2 for a in achieved]
    value = regret_to_oracle(trace, oracle_ref=above)
    assert value > 0.0
    assert 0.0 <= value <= 1.0
    # With a uniform +0.2 shortfall and REGRET_SCALE, the mean shortfall is 0.2.
    assert value == pytest.approx(0.2 / REGRET_SCALE)


def test_regret_saturates_to_one_when_oracle_far_above():
    trace = _build_trace()
    achieved = _achieved_coordinate_novelty(trace)
    huge = [a + 1000.0 for a in achieved]
    value = regret_to_oracle(trace, oracle_ref=huge)
    assert value == 1.0


def test_regret_always_in_unit_interval():
    trace = _build_trace()
    n = len(trace)
    for delta in (-10.0, 0.0, 0.01, 0.3, 5.0, 1e6):
        achieved = _achieved_coordinate_novelty(trace)
        ref = [a + delta for a in achieved]
        value = regret_to_oracle(trace, oracle_ref=ref)
        assert 0.0 <= value <= 1.0
    assert n == 3


# --------------------------------------------------------------------------- #
# None / length mismatch -> ValueError (Korean message), never silent padding.
# --------------------------------------------------------------------------- #
def test_none_oracle_ref_raises():
    trace = _build_trace()
    with pytest.raises(ValueError):
        regret_to_oracle(trace, oracle_ref=None)


def test_length_mismatch_too_short_raises():
    trace = _build_trace()  # 3 rounds
    with pytest.raises(ValueError):
        regret_to_oracle(trace, oracle_ref=[0.1, 0.2])


def test_length_mismatch_too_long_raises():
    trace = _build_trace()  # 3 rounds
    with pytest.raises(ValueError):
        regret_to_oracle(trace, oracle_ref=[0.1, 0.2, 0.3, 0.4])


def test_empty_trace_with_empty_ref_is_zero():
    trace = TraceState()
    value = regret_to_oracle(trace, oracle_ref=[])
    assert value == 0.0


def test_empty_trace_with_nonempty_ref_raises():
    trace = TraceState()
    with pytest.raises(ValueError):
        regret_to_oracle(trace, oracle_ref=[0.1])


# --------------------------------------------------------------------------- #
# Determinism: identical inputs -> identical value.
# --------------------------------------------------------------------------- #
def test_regret_is_deterministic():
    achieved_ref = _achieved_coordinate_novelty(_build_trace())
    above = [a + 0.15 for a in achieved_ref]
    v1 = regret_to_oracle(_build_trace(), oracle_ref=list(above))
    v2 = regret_to_oracle(_build_trace(), oracle_ref=list(above))
    assert v1 == v2


# --------------------------------------------------------------------------- #
# oracle_reference_from_objectives helper.
# --------------------------------------------------------------------------- #
def test_oracle_reference_from_objectives_coerces_floats():
    ref = oracle_reference_from_objectives([1, 2, 3])
    assert ref == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in ref)


def test_oracle_reference_from_objectives_rejects_non_numeric():
    with pytest.raises(ValueError):
        oracle_reference_from_objectives(["a", "b"])


def test_oracle_reference_from_objectives_rejects_none():
    with pytest.raises(ValueError):
        oracle_reference_from_objectives(None)


def test_oracle_reference_does_not_pad_or_truncate():
    """Helper preserves the count so length mismatch is surfaced downstream."""
    trace = _build_trace()  # 3 rounds
    ref = oracle_reference_from_objectives([0.1, 0.2])  # only 2 values
    assert len(ref) == 2
    with pytest.raises(ValueError):
        regret_to_oracle(trace, oracle_ref=ref)


# --------------------------------------------------------------------------- #
# compute_all_with_oracle wrapper.
# --------------------------------------------------------------------------- #
def test_compute_all_with_oracle_keys_and_values():
    trace = _build_trace()
    traces = {"a": _build_trace(), "b": _build_trace()}
    achieved = _achieved_coordinate_novelty(trace)
    oracle_ref = [a + 0.1 for a in achieved]

    result = compute_all_with_oracle(trace, traces, oracle_ref)
    expected_keys = {
        "traceHash",
        "coordinate_coverage",
        "artifact_diversity",
        "redundancy_rate",
        "round_coherence",
        "strategy_sensitivity",
        "regret_to_oracle",
    }
    assert set(result.keys()) == expected_keys
    assert result["traceHash"] == trace.trace_hash()
    assert result["regret_to_oracle"] == regret_to_oracle(trace, oracle_ref)
    assert 0.0 <= result["regret_to_oracle"] <= 1.0


def test_compute_all_with_oracle_is_deterministic():
    achieved = _achieved_coordinate_novelty(_build_trace())
    oracle_ref = [a + 0.1 for a in achieved]
    r1 = compute_all_with_oracle(
        _build_trace(), {"a": _build_trace(), "b": _build_trace()}, list(oracle_ref)
    )
    r2 = compute_all_with_oracle(
        _build_trace(), {"a": _build_trace(), "b": _build_trace()}, list(oracle_ref)
    )
    assert r1 == r2


def test_compute_all_with_oracle_propagates_length_error():
    trace = _build_trace()
    traces = {"a": _build_trace(), "b": _build_trace()}
    with pytest.raises(ValueError):
        compute_all_with_oracle(trace, traces, oracle_ref=[0.1])
