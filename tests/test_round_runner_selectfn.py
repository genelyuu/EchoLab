"""Tests for the additive ``select_fn`` hook on the round runner (E-001/E-002).

Verifies that:
- ``select_fn=None`` reproduces the existing Phase-1 slot-0 selection exactly
  (byte-identical traceHash to a run made *without* the argument);
- a probe ``select_fn`` makes the round's ``selectedCardId`` match the probe's
  own choice from the slate;
- determinism holds for both paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from echo_bench.basis.schema import load_bases
from echo_bench.env.round_runner import run_episode, run_round
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.base import Policy
from echo_bench.probes.strategy_probes import PROBES

_BASES_PATH = Path(__file__).resolve().parents[1] / "configs" / "basis" / "bases.yaml"


@pytest.fixture(scope="module")
def bases_cfg():
    return load_bases(_BASES_PATH)


def _make_pool(n: int = 12) -> list[dict]:
    """Deterministic synthetic pool spanning 4 bases (mirrors test_round_runner)."""
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
    """Deterministic stub: first ``k`` cardIds with diverse-enough bases."""

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


def test_select_fn_none_round_byte_identical(bases_cfg):
    """run_round with select_fn=None == run_round without the arg (slot-0 rule)."""
    pool = _make_pool()

    t_no_arg = TraceState()
    r_no_arg = run_round(pool, FirstKPolicy(), t_no_arg, seed=7, k=4, bases_cfg=bases_cfg)

    t_none = TraceState()
    r_none = run_round(
        pool, FirstKPolicy(), t_none, seed=7, k=4, bases_cfg=bases_cfg, select_fn=None
    )

    assert r_no_arg == r_none
    assert r_no_arg["roundHash"] == r_none["roundHash"]
    assert t_no_arg.trace_hash() == t_none.trace_hash()


def test_select_fn_none_episode_byte_identical(bases_cfg):
    """run_episode with select_fn=None == run_episode without the arg."""
    pool = _make_pool()

    t_no_arg = run_episode(pool, FirstKPolicy(), seed=42, H=6, k=4, bases_cfg=bases_cfg)
    t_none = run_episode(
        pool, FirstKPolicy(), seed=42, H=6, k=4, bases_cfg=bases_cfg, select_fn=None
    )

    assert t_no_arg.trace_hash() == t_none.trace_hash()


def test_probe_select_fn_selects_probe_choice(bases_cfg):
    """With a probe select_fn the selectedCardId is the probe's own choice."""
    pool = _make_pool()
    probe = PROBES["PREFER_HIGH_COMPLEXITY"]

    # Run a single round with the probe as select_fn.
    trace = TraceState()
    record = run_round(
        pool,
        FirstKPolicy(),
        trace,
        seed=11,
        k=4,
        bases_cfg=bases_cfg,
        select_fn=lambda slate, tr, sd: probe.select(slate, tr, sd),
    )

    # Reconstruct the slate dicts (in the policy's returned order) and ask the
    # probe directly: the recorded selection must equal the probe's choice.
    by_id = {c["cardId"]: c for c in pool}
    slate_dicts = [by_id[cid] for cid in record["slate"]]
    expected = probe.select(slate_dicts, trace, 11)
    assert record["selectedCardId"] == expected
    assert record["selectedCardId"] in record["slate"]


def test_probe_select_fn_deterministic(bases_cfg):
    """A probe select_fn replays to an identical traceHash."""
    pool = _make_pool()
    probe = PROBES["PREFER_LOW_SALIENCE"]

    def make():
        return run_episode(
            pool,
            FirstKPolicy(),
            seed=5,
            H=5,
            k=4,
            bases_cfg=bases_cfg,
            select_fn=lambda slate, tr, sd: probe.select(slate, tr, sd),
        )

    assert make().trace_hash() == make().trace_hash()


def test_probe_select_fn_differs_from_default(bases_cfg):
    """A probe that does not pick slot-0 yields a different trace than default."""
    pool = _make_pool()
    probe = PROBES["PREFER_HIGH_COMPLEXITY"]

    default_trace = run_episode(pool, FirstKPolicy(), seed=5, H=5, k=4, bases_cfg=bases_cfg)
    probe_trace = run_episode(
        pool,
        FirstKPolicy(),
        seed=5,
        H=5,
        k=4,
        bases_cfg=bases_cfg,
        select_fn=lambda slate, tr, sd: probe.select(slate, tr, sd),
    )
    # The probe deliberately targets the highest-complexity card, which is not
    # generally slot-0, so the traces should diverge.
    assert default_trace.trace_hash() != probe_trace.trace_hash()
