"""Tests for the FIXED_LOW_TO_HIGH policy (Task C-002).

Exercises the C-002 contract:

- slate has length ``k``, distinct cardIds, all from the pool;
- slate satisfies :func:`echo_bench.env.constraints.check_slate`;
- deterministic replay (same inputs -> identical slate);
- ``policyVersion`` changes with config; ``last_score_components`` populated;
- NON-ADAPTIVE: two DIFFERENT trace contents with the same round index & seed
  yield the same target band and the same slate (no trace-content adaptation).

No user/persona/emotion/preference field appears anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.fixed_low_to_high import (
    FixedLowToHighPolicy,
    band_index,
)


def _make_pool(n: int = 16) -> List[Dict[str, Any]]:
    """16 synthetic cards spanning B1..B4 with varied band/salience/coords."""
    bands = ("low", "mid", "high")
    return [
        {
            "cardId": f"c{i:02d}",
            "basis": f"B{(i % 4) + 1}",
            "complexityBand": bands[i % 3],
            "salienceScore": round(0.05 * (i % 11), 4),
            "coordinateContribution": [float(i), float(-i), float(i % 5), 0.25 * i],
        }
        for i in range(n)
    ]


def _round_record(card: Dict[str, Any]) -> dict:
    return {
        "candidatePoolHash": "ph",
        "slate": [card["cardId"]],
        "selectedCardId": card["cardId"],
        "coordinateContribution": card["coordinateContribution"],
        "complexityBand": card["complexityBand"],
        "salienceScore": card["salienceScore"],
        "slotPermutation": [0],
    }


def _trace_with(cards: List[Dict[str, Any]]) -> TraceState:
    trace = TraceState()
    for c in cards:
        trace.append_round(_round_record(c))
    return trace


def test_slate_length_distinct_and_from_pool() -> None:
    pool = _make_pool()
    policy = FixedLowToHighPolicy({"k": 4})
    slate = policy.select(pool, trace=TraceState(), seed=7)
    assert len(slate) == 4
    assert len(set(slate)) == 4
    assert all(cid in {c["cardId"] for c in pool} for cid in slate)


def test_slate_satisfies_constraints() -> None:
    pool = _make_pool()
    policy = FixedLowToHighPolicy({"k": 4})
    slate = policy.select(pool, trace=TraceState(), seed=7)
    by_id = {c["cardId"]: c for c in pool}
    ok, reason, perm = check_slate([by_id[c] for c in slate], 4, {}, 7)
    assert ok is True, reason
    assert sorted(perm) == [0, 1, 2, 3]


def test_deterministic_replay() -> None:
    pool = _make_pool()
    p1 = FixedLowToHighPolicy({"k": 4})
    p2 = FixedLowToHighPolicy({"k": 4})
    assert p1.select(pool, TraceState(), 7) == p2.select(pool, TraceState(), 7)


def test_policy_version_changes_with_config() -> None:
    a = FixedLowToHighPolicy({"k": 4, "horizon": 6})
    b = FixedLowToHighPolicy({"k": 4, "horizon": 8})
    assert a.policy_version() != b.policy_version()


def test_last_score_components_populated() -> None:
    pool = _make_pool()
    policy = FixedLowToHighPolicy({"k": 4})
    assert policy.last_score_components == {}
    slate = policy.select(pool, TraceState(), 7)
    assert set(policy.last_score_components) == set(slate)
    for comp in policy.last_score_components.values():
        assert "bandDistance" in comp and "targetBand" in comp


def test_target_band_increases_with_round_index() -> None:
    pool = _make_pool()
    policy = FixedLowToHighPolicy({"k": 4})
    pool_low = [c for c in pool if c["complexityBand"] == "low"]
    targets = []
    for r in range(6):
        # Build a trace of length r from arbitrary cards (content irrelevant).
        trace = _trace_with(pool_low[:1] * 0 + [pool[i % len(pool)] for i in range(r)])
        policy.select(pool, trace, seed=7)
        # Target band recorded in score components.
        any_comp = next(iter(policy.last_score_components.values()))
        targets.append(band_index(any_comp["targetBand"]))
    # Non-decreasing low -> high schedule.
    assert targets == sorted(targets)
    assert targets[0] <= targets[-1]


def test_non_adaptive_two_different_traces_same_round_index() -> None:
    """Same round index + seed but DIFFERENT trace content -> identical slate."""
    pool = _make_pool()
    policy = FixedLowToHighPolicy({"k": 4})

    # Two traces of equal length (3) but with different selected cards/coords.
    trace_a = _trace_with([pool[0], pool[1], pool[2]])
    trace_b = _trace_with([pool[5], pool[8], pool[11]])
    assert trace_a.trace_hash() != trace_b.trace_hash()  # genuinely different

    slate_a = policy.select(pool, trace_a, seed=7)
    target_a = next(iter(policy.last_score_components.values()))["targetBand"]
    slate_b = policy.select(pool, trace_b, seed=7)
    target_b = next(iter(policy.last_score_components.values()))["targetBand"]

    assert target_a == target_b  # schedule depends on round index only
    assert slate_a == slate_b    # no trace-content adaptation


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = FixedLowToHighPolicy({"k": 4})
    with pytest.raises(ValueError):
        policy.select(pool, TraceState(), 7)
