"""Tests for the TRACE_GREEDY policy (Task C-004).

Exercises the C-004 contract:

- slate has length ``k``, distinct cardIds, all from the pool;
- slate satisfies :func:`echo_bench.env.constraints.check_slate`;
- deterministic replay; ``policyVersion`` changes with config;
  ``last_score_components`` populated with the weighted parts;
- TRACE-DRIVEN: two traces with DIFFERENT accumulated coordinates (same seed)
  produce different slates (it DOES adapt to observable trace content);
- the policy reads no latent field (only coordinateContribution / complexityBand
  / selectedCardId are read from the trace).

No user/persona/emotion/preference field appears anywhere.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

import echo_bench.policies.trace_greedy as trace_greedy_module
from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.trace_greedy import TraceGreedyPolicy


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
    policy = TraceGreedyPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    assert len(slate) == 4
    assert len(set(slate)) == 4
    assert all(cid in {c["cardId"] for c in pool} for cid in slate)


def test_slate_satisfies_constraints() -> None:
    pool = _make_pool()
    policy = TraceGreedyPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    by_id = {c["cardId"]: c for c in pool}
    ok, reason, perm = check_slate([by_id[c] for c in slate], 4, {}, 7)
    assert ok is True, reason
    assert sorted(perm) == [0, 1, 2, 3]


def test_deterministic_replay() -> None:
    pool = _make_pool()
    trace_cards = [pool[0], pool[1]]
    p1 = TraceGreedyPolicy({"k": 4})
    p2 = TraceGreedyPolicy({"k": 4})
    s1 = p1.select(pool, _trace_with(trace_cards), 7)
    s2 = p2.select(pool, _trace_with(trace_cards), 7)
    assert s1 == s2


def test_policy_version_changes_with_weights() -> None:
    a = TraceGreedyPolicy({"k": 4, "weights": {"novelty": 1.0}})
    b = TraceGreedyPolicy({"k": 4, "weights": {"novelty": 2.0}})
    assert a.policy_version() != b.policy_version()


def test_last_score_components_populated_with_weighted_parts() -> None:
    pool = _make_pool()
    policy = TraceGreedyPolicy({"k": 4})
    assert policy.last_score_components == {}
    slate = policy.select(pool, TraceState(), 7)
    assert set(policy.last_score_components) == set(slate)
    for comp in policy.last_score_components.values():
        assert {"novelty", "progression", "redundancy", "total"} <= set(comp)


def test_trace_driven_different_accumulated_coords_change_slate() -> None:
    """Different accumulated trace coordinates -> different slate (adaptive)."""
    pool = _make_pool()
    policy = TraceGreedyPolicy({"k": 4})

    # Two traces whose accumulated coordinate centroids differ markedly.
    low_coord_cards = [pool[0], pool[1], pool[2]]      # small indices -> small coords
    high_coord_cards = [pool[13], pool[14], pool[15]]  # large indices -> large coords
    trace_low = _trace_with(low_coord_cards)
    trace_high = _trace_with(high_coord_cards)

    slate_low = policy.select(pool, trace_low, seed=7)
    slate_high = policy.select(pool, trace_high, seed=7)

    assert slate_low != slate_high  # genuinely trace-driven


def test_reads_no_latent_field() -> None:
    """Injecting a latent field into pool cards must not change the outcome.

    The policy reads only coordinateContribution / complexityBand from the trace
    and the pool; a stray latent/persona/preference key must be ignored.
    """
    pool = _make_pool()
    poisoned = [
        {**c, "preference": 0.99, "persona": "X", "emotion": "joy", "user_id": 42}
        for c in pool
    ]
    policy = TraceGreedyPolicy({"k": 4})
    clean = policy.select(pool, TraceState(), 7)
    dirty = policy.select(poisoned, TraceState(), 7)
    assert clean == dirty


def test_source_contains_no_latent_references() -> None:
    """Static guard: the module source must not reference latent fields."""
    src = inspect.getsource(trace_greedy_module)
    for forbidden in ("user_id", "persona", "emotion", "preference", "demographic"):
        # Allowed only inside docstring disclaimers; assert not used as a key access.
        assert f'"{forbidden}"' not in src
        assert f"['{forbidden}']" not in src
        assert f'["{forbidden}"]' not in src


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = TraceGreedyPolicy({"k": 4})
    with pytest.raises(ValueError):
        policy.select(pool, TraceState(), 7)


# --- C-009: session-level objective redesign (no long-horizon collapse) ------

from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from echo_bench.archive.builder import build_archive  # noqa: E402
from echo_bench.basis.schema import load_bases  # noqa: E402
from echo_bench.env.round_runner import run_episode  # noqa: E402
from echo_bench.metrics.utility import (  # noqa: E402
    coordinate_coverage,
    redundancy_rate,
)

_REPO = Path(__file__).resolve().parents[1]


def _real_pool(n: int = 64):
    bases = load_bases(_REPO / "configs" / "basis" / "bases.yaml")
    with open(_REPO / "configs" / "archive" / "archive.yaml", encoding="utf-8") as fh:
        archive_cfg = yaml.safe_load(fh)
    archive = build_archive(bases, archive_cfg, 42)
    return archive["cards"][:n], bases


def test_components_include_session_terms():
    pool = _make_pool()
    policy = TraceGreedyPolicy({"k": 4})
    policy.select(pool, TraceState(), 7)
    for comp in policy.last_score_components.values():
        # The pre-existing keys remain, plus the new session-level terms.
        assert {"novelty", "progression", "redundancy", "total"} <= set(comp)
        assert {"session_coverage", "cell_repulsion"} <= set(comp)


def test_no_long_horizon_coverage_collapse():
    # The diagnostic showed the OLD objective collapsing at long horizons
    # (coverage 0.95 -> 0.33, redundancy 0.025 -> 0.665 over H 4 -> 20). The
    # redesigned session-level objective must hold coverage up and redundancy
    # down at H=20.
    pool, bases = _real_pool()
    trace = run_episode(pool, TraceGreedyPolicy({"k": 4}), 7, 20, 4, bases)
    cov = coordinate_coverage(trace)
    red = redundancy_rate(trace)
    assert cov > 0.5, f"coverage collapsed at H=20: {cov}"
    assert red < 0.5, f"redundancy too high at H=20: {red}"


def test_redesigned_policy_is_deterministic():
    pool, bases = _real_pool()
    a = run_episode(pool, TraceGreedyPolicy({"k": 4}), 7, 12, 4, bases)
    b = run_episode(pool, TraceGreedyPolicy({"k": 4}), 7, 12, 4, bases)
    assert a.trace_hash() == b.trace_hash()
