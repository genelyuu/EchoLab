"""Tests for echo_bench.env.round_runner (Task B-002)."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo_bench.basis.schema import load_bases
from echo_bench.env.round_runner import run_episode, run_round
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.base import Policy

_BASES_PATH = Path(__file__).resolve().parents[1] / "configs" / "basis" / "bases.yaml"

_FULL_KEYS = {
    "candidatePoolHash",
    "slate",
    "selectedCardId",
    "coordinateContribution",
    "complexityBand",
    "salienceScore",
    "slotPermutation",
    "roundHash",
}


@pytest.fixture(scope="module")
def bases_cfg():
    return load_bases(_BASES_PATH)


def _make_pool(n: int = 12) -> list[dict]:
    """Build a deterministic synthetic pool spanning 4 bases."""
    bases = ["B1", "B2", "B3", "B4"]
    bands = ["low", "mid", "high"]
    pool = []
    for i in range(n):
        pool.append(
            {
                "cardId": f"card-{i:02d}",
                "basis": bases[i % len(bases)],
                "complexityBand": bands[i % len(bands)],
                "salienceScore": round(0.1 * (i % 10), 3),
                "coordinateContribution": [float(i), float(i + 1), float(i + 2)],
            }
        )
    return pool


class FirstKPolicy(Policy):
    """Deterministic stub: return the first ``k`` cardIds whose bases are
    sufficiently diverse to satisfy the constraint (>=3 distinct for k=4)."""

    def select(self, pool, trace, seed, config):
        k = config["k"]
        chosen: list = []
        seen_bases: set = set()
        # First pass: greedily prefer new bases to maximise basis diversity.
        for card in pool:
            if len(chosen) >= k:
                break
            if card["basis"] not in seen_bases:
                chosen.append(card["cardId"])
                seen_bases.add(card["basis"])
        # Second pass: fill remaining slots in order.
        if len(chosen) < k:
            for card in pool:
                if len(chosen) >= k:
                    break
                if card["cardId"] not in chosen:
                    chosen.append(card["cardId"])
        return chosen[:k]


def test_run_round_appends_one_full_record(bases_cfg):
    pool = _make_pool()
    trace = TraceState()
    policy = FirstKPolicy()

    record = run_round(pool, policy, trace, seed=7, k=4, bases_cfg=bases_cfg)

    assert len(trace) == 1
    assert set(record.keys()) == _FULL_KEYS
    assert record["selectedCardId"] in record["slate"]
    assert len(record["slate"]) == 4


def test_run_round_deterministic(bases_cfg):
    pool = _make_pool()
    policy = FirstKPolicy()

    t1 = TraceState()
    r1 = run_round(pool, policy, t1, seed=7, k=4, bases_cfg=bases_cfg)
    t2 = TraceState()
    r2 = run_round(pool, policy, t2, seed=7, k=4, bases_cfg=bases_cfg)

    assert r1["roundHash"] == r2["roundHash"]
    assert t1.trace_hash() == t2.trace_hash()


def test_run_round_rejects_unknown_card_id(bases_cfg):
    pool = _make_pool()
    trace = TraceState()

    class BadPolicy(Policy):
        def select(self, pool, trace, seed, config):
            return ["card-00", "card-01", "card-02", "not-in-pool"]

    with pytest.raises(ValueError):
        run_round(pool, BadPolicy(), trace, seed=1, k=4, bases_cfg=bases_cfg)
    assert len(trace) == 0


def test_run_round_rejects_constraint_violation(bases_cfg):
    pool = _make_pool()
    trace = TraceState()

    class MonoBasisPolicy(Policy):
        # Returns 4 cards that all share basis B1 -> insufficient_bases.
        def select(self, pool, trace, seed, config):
            same = [c["cardId"] for c in pool if c["basis"] == "B1"]
            return same[:4]

    with pytest.raises(ValueError):
        run_round(pool, MonoBasisPolicy(), trace, seed=1, k=4, bases_cfg=bases_cfg)
    assert len(trace) == 0


def test_run_episode_appends_H_records(bases_cfg):
    pool = _make_pool()
    policy = FirstKPolicy()
    H = 6

    trace = run_episode(pool, policy, seed=42, H=H, k=4, bases_cfg=bases_cfg)
    assert len(trace) == H


def test_run_episode_deterministic_same_seed(bases_cfg):
    pool = _make_pool()
    policy = FirstKPolicy()

    t1 = run_episode(pool, policy, seed=42, H=6, k=4, bases_cfg=bases_cfg)
    t2 = run_episode(pool, policy, seed=42, H=6, k=4, bases_cfg=bases_cfg)
    assert t1.trace_hash() == t2.trace_hash()


def test_run_episode_different_seed_differs(bases_cfg):
    pool = _make_pool()
    policy = FirstKPolicy()

    t1 = run_episode(pool, policy, seed=42, H=6, k=4, bases_cfg=bases_cfg)
    t2 = run_episode(pool, policy, seed=99, H=6, k=4, bases_cfg=bases_cfg)
    assert t1.trace_hash() != t2.trace_hash()
