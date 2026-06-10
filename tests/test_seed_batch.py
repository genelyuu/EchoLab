"""Tests for echo_bench.env.seed_batch (Task B-006, minimal subset)."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo_bench.basis.schema import load_bases
from echo_bench.env.seed_batch import (
    derive_child_seeds,
    run_seed_batch,
    seed_batch_id,
)
from echo_bench.policies.base import Policy

_BASES_PATH = Path(__file__).resolve().parents[1] / "configs" / "basis" / "bases.yaml"


@pytest.fixture(scope="module")
def bases_cfg():
    return load_bases(_BASES_PATH)


def _make_pool(n: int = 12) -> list[dict]:
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
    def select(self, pool, trace, seed, config):
        k = config["k"]
        chosen: list = []
        seen_bases: set = set()
        for card in pool:
            if len(chosen) >= k:
                break
            if card["basis"] not in seen_bases:
                chosen.append(card["cardId"])
                seen_bases.add(card["basis"])
        if len(chosen) < k:
            for card in pool:
                if len(chosen) >= k:
                    break
                if card["cardId"] not in chosen:
                    chosen.append(card["cardId"])
        return chosen[:k]


def test_derive_child_seeds_deterministic():
    a = derive_child_seeds(123, 5)
    b = derive_child_seeds(123, 5)
    assert a == b
    assert len(a) == 5
    assert len(set(a)) == 5  # distinct per index
    # Different base seed -> different children.
    assert derive_child_seeds(124, 5) != a


def test_seed_batch_id_stable():
    sid1 = seed_batch_id(123, 5, "pv-abc", 6)
    sid2 = seed_batch_id(123, 5, "pv-abc", 6)
    assert sid1 == sid2
    assert seed_batch_id(123, 5, "pv-xyz", 6) != sid1
    assert seed_batch_id(123, 6, "pv-abc", 6) != sid1


def test_run_seed_batch_reproducible(bases_cfg):
    pool = _make_pool()
    policy = FirstKPolicy()

    out1 = run_seed_batch(pool, policy, base_seed=123, n=4, H=5, k=4, bases_cfg=bases_cfg)
    out2 = run_seed_batch(pool, policy, base_seed=123, n=4, H=5, k=4, bases_cfg=bases_cfg)

    assert out1["seedBatchId"] == out2["seedBatchId"]
    assert out1["traces"] == out2["traces"]
    assert out1["n"] == out2["n"] == 4
    assert len(out1["traces"]) == 4


def test_run_seed_batch_order_by_child_index(bases_cfg):
    pool = _make_pool()
    policy = FirstKPolicy()

    out = run_seed_batch(pool, policy, base_seed=123, n=4, H=5, k=4, bases_cfg=bases_cfg)
    child_seeds = derive_child_seeds(123, 4)

    # The i-th trace hash must equal an independent episode run on child seed i.
    from echo_bench.env.round_runner import run_episode

    for i, cs in enumerate(child_seeds):
        trace = run_episode(pool, policy, cs, H=5, k=4, bases_cfg=bases_cfg)
        assert out["traces"][i] == trace.trace_hash()
