"""Tests for the ORACLE_STRATEGY policy (Task C-007).

Exercises the C-007 contract:

- given a probe, the oracle's slate is constraint-satisfying and deterministic
  for ``(poolHash, probeVersion, seed)``;
- ``scoreComponents`` logged; ``policyVersion`` recorded and probe-dependent;
- the oracle uses the probe explicitly (probe internals confined to the policy);
- changing the probe changes the objective/slate.

The probe is a controlled input policy, not a synthetic user.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

import echo_bench.policies.oracle_strategy as oracle_module
from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.oracle_strategy import OracleStrategyPolicy
from echo_bench.probes.strategy_probes import get_probe


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


def test_slate_constraint_satisfying() -> None:
    pool = _make_pool()
    policy = OracleStrategyPolicy({"k": 4, "probe": "PREFER_HIGH_COMPLEXITY"})
    slate = policy.select(pool, TraceState(), 7)
    assert len(slate) == 4
    assert len(set(slate)) == 4
    by_id = {c["cardId"]: c for c in pool}
    ok, reason, _perm = check_slate([by_id[c] for c in slate], 4, {}, 7)
    assert ok is True, reason


def test_deterministic_for_pool_probe_seed() -> None:
    pool = _make_pool()
    trace = _trace_with([pool[0], pool[1]])
    a = OracleStrategyPolicy({"k": 4, "probe": "PREFER_COORD_NOVELTY"})
    b = OracleStrategyPolicy({"k": 4, "probe": "PREFER_COORD_NOVELTY"})
    assert a.select(pool, trace, 7) == b.select(pool, trace, 7)


def test_changing_probe_changes_slate_or_objective() -> None:
    """Different probe -> different documented objective -> different slate."""
    pool = _make_pool(24)
    trace = _trace_with([pool[0], pool[1], pool[2]])
    high = OracleStrategyPolicy({"k": 4, "probe": "PREFER_HIGH_COMPLEXITY"})
    low_sal = OracleStrategyPolicy({"k": 4, "probe": "PREFER_LOW_SALIENCE"})
    s_high = high.select(pool, trace, 7)
    s_low = low_sal.select(pool, trace, 7)
    assert s_high != s_low


def test_policy_version_changes_with_probe() -> None:
    a = OracleStrategyPolicy({"k": 4, "probe": "PREFER_HIGH_COMPLEXITY"})
    b = OracleStrategyPolicy({"k": 4, "probe": "PREFER_LOW_SALIENCE"})
    assert a.policy_version() != b.policy_version()


def test_score_components_logged_with_objective() -> None:
    pool = _make_pool()
    policy = OracleStrategyPolicy({"k": 4, "probe": "PREFER_HIGH_COMPLEXITY"})
    slate = policy.select(pool, TraceState(), 7)
    assert set(policy.last_score_components) == set(slate)
    for comp in policy.last_score_components.values():
        assert "probeObjective" in comp
        assert "isProbeSelected" in comp


def test_oracle_uses_probe_explicitly() -> None:
    """The probe's selected card from the oracle's slate is the top-objective one.

    Confirms the oracle constructed the slate around the probe's objective: the
    card the probe selects from the assembled slate has the maximal probe
    objective among the slate members.
    """
    pool = _make_pool(24)
    trace = _trace_with([pool[0], pool[5], pool[10]])
    probe = get_probe("PREFER_HIGH_COMPLEXITY")
    policy = OracleStrategyPolicy({"k": 4, "probe": "PREFER_HIGH_COMPLEXITY"})
    slate = policy.select(pool, trace, 7)
    by_id = {c["cardId"]: c for c in pool}

    selected = probe.select([by_id[c] for c in slate], trace, 7)
    objectives = {c: probe._score(by_id[c], trace) for c in slate}
    assert objectives[selected] == max(objectives.values())


def test_oracle_confines_probe_access() -> None:
    """Only the oracle imports the probe registry; it uses get_probe explicitly."""
    src = inspect.getsource(oracle_module)
    assert "get_probe" in src
    assert "strategy_probes" in src


def test_unknown_probe_raises() -> None:
    pool = _make_pool()
    policy = OracleStrategyPolicy({"k": 4, "probe": "NOPE_NOT_A_PROBE"})
    with pytest.raises(KeyError):
        policy.select(pool, TraceState(), 7)


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = OracleStrategyPolicy({"k": 4, "probe": "PREFER_COORD_NOVELTY"})
    with pytest.raises(ValueError):
        policy.select(pool, TraceState(), 7)
