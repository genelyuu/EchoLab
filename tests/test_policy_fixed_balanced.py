"""Tests for the FIXED_BALANCED policy (Task C-003).

Exercises the C-003 contract:

- slate has length ``k``, distinct cardIds, all from the pool;
- slate satisfies :func:`echo_bench.env.constraints.check_slate` (incl. the
  >=3-distinct-bases rule for k=4);
- deterministic replay; ``policyVersion`` changes with config;
  ``last_score_components`` populated;
- NON-ADAPTIVE: two DIFFERENT traces (any content/length) with the same seed
  yield the same slate (balance is independent of the trace).

No user/persona/emotion/preference field appears anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.fixed_balanced import FixedBalancedPolicy


def _make_pool(n: int = 16) -> List[Dict[str, Any]]:
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
    policy = FixedBalancedPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    assert len(slate) == 4
    assert len(set(slate)) == 4
    assert all(cid in {c["cardId"] for c in pool} for cid in slate)


def test_slate_satisfies_constraints_and_basis_diversity() -> None:
    pool = _make_pool()
    policy = FixedBalancedPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    by_id = {c["cardId"]: c for c in pool}
    cards = [by_id[c] for c in slate]
    ok, reason, perm = check_slate(cards, 4, {}, 7)
    assert ok is True, reason
    assert sorted(perm) == [0, 1, 2, 3]
    # Balanced mix => for k=4 with 4 bases, all 4 distinct bases present.
    assert len({c["basis"] for c in cards}) >= 3


def test_deterministic_replay() -> None:
    pool = _make_pool()
    p1 = FixedBalancedPolicy({"k": 4})
    p2 = FixedBalancedPolicy({"k": 4})
    assert p1.select(pool, TraceState(), 7) == p2.select(pool, TraceState(), 7)


def test_policy_version_changes_with_config() -> None:
    a = FixedBalancedPolicy({"k": 4})
    b = FixedBalancedPolicy({"k": 4, "basis_targets": {"B1": 2.0, "B2": 1.0}})
    assert a.policy_version() != b.policy_version()


def test_last_score_components_populated() -> None:
    pool = _make_pool()
    policy = FixedBalancedPolicy({"k": 4})
    assert policy.last_score_components == {}
    slate = policy.select(pool, TraceState(), 7)
    assert set(policy.last_score_components) == set(slate)
    for comp in policy.last_score_components.values():
        assert "chosenBasis" in comp and "chosenBand" in comp


def test_non_adaptive_independent_of_trace() -> None:
    """DIFFERENT traces, same seed -> identical slate (trace-independent)."""
    pool = _make_pool()
    policy = FixedBalancedPolicy({"k": 4})

    trace_a = TraceState()  # empty
    trace_b = _trace_with([pool[3], pool[6], pool[9], pool[12]])  # non-empty
    assert trace_a.trace_hash() != trace_b.trace_hash()

    assert policy.select(pool, trace_a, 7) == policy.select(pool, trace_b, 7)


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = FixedBalancedPolicy({"k": 4})
    with pytest.raises(ValueError):
        policy.select(pool, TraceState(), 7)
