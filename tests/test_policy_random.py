"""Tests for the RANDOM policy (Task C-001, RANDOM portion).

Exercises the contract from C-001:

- the returned slate has length ``k``, all cardIds come from the pool and are
  distinct;
- the slate satisfies :func:`echo_bench.env.constraints.check_slate`;
- selection is a pure function of ``(pool, trace, seed, policyVersion)`` —
  identical inputs replay identically; different seeds (usually) differ;
- ``policy_version`` changes when config ``k`` changes;
- ``last_score_components`` is populated after :meth:`select`.

No user/persona/emotion/preference field appears anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.random import RandomPolicy


def _make_pool(n: int = 12) -> List[Dict[str, Any]]:
    """Build ~12 synthetic card dicts spanning bases B1..B4."""
    return [
        {
            "cardId": f"c{i:02d}",
            "basis": f"B{(i % 4) + 1}",
            "complexityBand": ("low", "mid", "high")[i % 3],
            "salienceScore": 0.1 * (i % 10),
            "coordinateContribution": [float(i), float(-i)],
        }
        for i in range(n)
    ]


def _fresh_trace() -> TraceState:
    return TraceState()


def test_slate_length_distinct_and_from_pool() -> None:
    pool = _make_pool(12)
    policy = RandomPolicy({"k": 4})
    slate = policy.select(pool, trace=_fresh_trace(), seed=7)

    assert len(slate) == 4
    assert len(set(slate)) == 4  # distinct
    pool_ids = {c["cardId"] for c in pool}
    assert all(cid in pool_ids for cid in slate)


def test_slate_satisfies_constraints() -> None:
    pool = _make_pool(12)
    policy = RandomPolicy({"k": 4})
    seed = 7
    slate = policy.select(pool, trace=_fresh_trace(), seed=seed)

    by_id = {c["cardId"]: c for c in pool}
    cards = [by_id[cid] for cid in slate]
    ok, reason, slot_permutation = check_slate(cards, 4, {}, seed)
    assert ok is True, reason
    assert sorted(slot_permutation) == [0, 1, 2, 3]


def test_deterministic_replay_same_inputs() -> None:
    pool = _make_pool(12)
    p1 = RandomPolicy({"k": 4})
    p2 = RandomPolicy({"k": 4})
    s1 = p1.select(pool, trace=_fresh_trace(), seed=7)
    s2 = p2.select(pool, trace=_fresh_trace(), seed=7)
    assert s1 == s2


def test_different_seed_usually_differs() -> None:
    pool = _make_pool(12)
    policy = RandomPolicy({"k": 4})
    slates = {
        tuple(policy.select(pool, trace=_fresh_trace(), seed=s))
        for s in range(8)
    }
    # With 8 distinct seeds over a 12-card pool, expect more than one outcome.
    assert len(slates) > 1


def test_policy_version_changes_with_k() -> None:
    p4 = RandomPolicy({"k": 4})
    p2 = RandomPolicy({"k": 2})
    assert p4.policy_version() != p2.policy_version()


def test_last_score_components_populated() -> None:
    pool = _make_pool(12)
    policy = RandomPolicy({"k": 4})
    assert policy.last_score_components == {}
    slate = policy.select(pool, trace=_fresh_trace(), seed=7)
    assert set(policy.last_score_components.keys()) == set(slate)
    for comp in policy.last_score_components.values():
        assert "randomScore" in comp


def test_config_override_at_call_time() -> None:
    pool = _make_pool(12)
    policy = RandomPolicy({"k": 4})
    slate = policy.select(pool, trace=_fresh_trace(), seed=7, config={"k": 2})
    assert len(slate) == 2
    by_id = {c["cardId"]: c for c in pool}
    cards = [by_id[cid] for cid in slate]
    ok, reason, _ = check_slate(cards, 2, {}, 7)
    assert ok is True, reason


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = RandomPolicy({"k": 4})
    with pytest.raises(ValueError):
        policy.select(pool, trace=_fresh_trace(), seed=7)


def test_raises_when_no_constraint_satisfying_slate() -> None:
    # All cards share a single basis -> k=4 needs >=3 distinct bases -> never ok.
    pool = [
        {
            "cardId": f"c{i:02d}",
            "basis": "B1",
            "complexityBand": "low",
            "salienceScore": 0.5,
            "coordinateContribution": [0.0, 0.0],
        }
        for i in range(8)
    ]
    policy = RandomPolicy({"k": 4, "max_attempts": 20})
    with pytest.raises(ValueError):
        policy.select(pool, trace=_fresh_trace(), seed=7)
