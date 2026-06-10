"""Tests for the TRACE_LIN_UCB policy (Task C-005).

Exercises the C-005 contract:

- slate has length ``k``, distinct cardIds, all from the pool;
- slate satisfies :func:`echo_bench.env.constraints.check_slate`;
- deterministic replay; ``policyVersion`` changes with config;
- ``scoreComponents`` log ``{"mean": .., "bonus": ..}`` per chosen card;
- TRACE-ONLY: no latent user vector exists; the module reads no latent field and
  does NOT import :mod:`echo_bench.policies.pseudo_user_model`.

No user/persona/emotion/preference field appears anywhere.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

import echo_bench.policies.trace_lin_ucb as lin_ucb_module
from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy


def _make_pool(n: int = 16) -> List[Dict[str, Any]]:
    bands = ("low", "mid", "high")
    return [
        {
            "cardId": f"c{i:02d}",
            "basis": f"B{(i % 4) + 1}",
            "complexityBand": bands[i % 3],
            "salienceScore": round(0.05 * (i % 11), 4),
            "coordinateContribution": [
                float(i), float(-i), float((i * 3) % 7), 0.3 * i
            ],
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
    policy = TraceLinUcbPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    assert len(slate) == 4
    assert len(set(slate)) == 4
    assert all(cid in {c["cardId"] for c in pool} for cid in slate)


def test_slate_satisfies_constraints() -> None:
    pool = _make_pool()
    policy = TraceLinUcbPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    by_id = {c["cardId"]: c for c in pool}
    ok, reason, perm = check_slate([by_id[c] for c in slate], 4, {}, 7)
    assert ok is True, reason
    assert sorted(perm) == [0, 1, 2, 3]


def test_deterministic_replay() -> None:
    pool = _make_pool()
    trace_cards = [pool[0], pool[5], pool[10]]
    p1 = TraceLinUcbPolicy({"k": 4})
    p2 = TraceLinUcbPolicy({"k": 4})
    s1 = p1.select(pool, _trace_with(trace_cards), 7)
    s2 = p2.select(pool, _trace_with(trace_cards), 7)
    assert s1 == s2


def test_policy_version_changes_with_alpha() -> None:
    a = TraceLinUcbPolicy({"k": 4, "alpha": 1.0})
    b = TraceLinUcbPolicy({"k": 4, "alpha": 2.0})
    assert a.policy_version() != b.policy_version()


def test_score_components_mean_and_bonus() -> None:
    pool = _make_pool()
    policy = TraceLinUcbPolicy({"k": 4})
    assert policy.last_score_components == {}
    slate = policy.select(pool, _trace_with([pool[0], pool[1]]), 7)
    assert set(policy.last_score_components) == set(slate)
    for comp in policy.last_score_components.values():
        assert set(comp) == {"mean", "bonus"}


def test_alpha_changes_exploration_outcome() -> None:
    """A large exploration coefficient can change the selected slate."""
    pool = _make_pool(24)
    trace = _trace_with([pool[0], pool[1], pool[2], pool[3]])
    low = TraceLinUcbPolicy({"k": 4}).select(pool, trace, 7, {"k": 4, "alpha": 0.0})
    high = TraceLinUcbPolicy({"k": 4}).select(pool, trace, 7, {"k": 4, "alpha": 50.0})
    # Both must still be valid; with this pool they differ.
    assert low != high


def test_no_latent_vector_and_reads_no_latent_field() -> None:
    """No latent user vector; injecting latent fields must not change outcome."""
    pool = _make_pool()
    poisoned = [
        {**c, "preference": 0.99, "persona": "X", "emotion": "joy", "user_id": 42}
        for c in pool
    ]
    policy = TraceLinUcbPolicy({"k": 4})
    clean = policy.select(pool, TraceState(), 7)
    dirty = policy.select(poisoned, TraceState(), 7)
    assert clean == dirty


def test_source_contains_no_latent_references() -> None:
    """Static guard: module source must not access latent fields as keys."""
    src = inspect.getsource(lin_ucb_module)
    for forbidden in ("user_id", "persona", "emotion", "preference", "demographic"):
        assert f'"{forbidden}"' not in src
        assert f"['{forbidden}']" not in src
        assert f'["{forbidden}"]' not in src


def test_does_not_import_pseudo_user_model() -> None:
    """TRACE-ONLY: this module must not import or reference the contrast baseline."""
    src = inspect.getsource(lin_ucb_module)
    assert "pseudo_user_model" not in src
    assert "PseudoUserModel" not in src


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = TraceLinUcbPolicy({"k": 4})
    with pytest.raises(ValueError):
        policy.select(pool, TraceState(), 7)
