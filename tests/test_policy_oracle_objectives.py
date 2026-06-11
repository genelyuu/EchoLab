"""Tests for the C-010 objective-specific ORACLE variants and regret machinery.

Covers:
- the oracle objective scorers (COVERAGE_GAIN / DIVERSITY_GAIN) are decoupled
  from the strategy PROBES registry and score observable fields only;
- ``regret_to_oracle_for`` matches ``regret_to_oracle`` for COORD_NOVELTY
  (back-compat) and dispatches to coverage/diversity achievers;
- an objective-specific oracle's self-regret on its own objective is ~0;
- the COVERAGE oracle attains coordinate coverage at least as high as RANDOM
  (sanity that it is a genuine upper reference for *that* objective);
- the objective oracle is deterministic and its policyVersion is objective-aware.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.round_runner import run_episode
from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.utility import (
    OBJECTIVE_ACHIEVERS,
    _achieved_coordinate_novelty,
    coordinate_coverage,
    oracle_reference_from_objectives,
    regret_to_oracle,
    regret_to_oracle_for,
)
from echo_bench.policies.oracle_strategy import OracleStrategyPolicy
from echo_bench.policies.random import RandomPolicy
from echo_bench.probes.oracle_objectives import (
    ORACLE_OBJECTIVES,
    coverage_gain_score,
    diversity_gain_score,
    get_oracle_objective,
)
from echo_bench.probes.strategy_probes import PROBES

_REPO = Path(__file__).resolve().parents[1]


def _real_pool(n: int = 64):
    bases = load_bases(_REPO / "configs" / "basis" / "bases.yaml")
    with open(_REPO / "configs" / "archive" / "archive.yaml", encoding="utf-8") as fh:
        archive_cfg = yaml.safe_load(fh)
    archive = build_archive(bases, archive_cfg, 42)
    return archive["cards"][:n], bases


def test_objectives_decoupled_from_strategy_probes():
    assert set(ORACLE_OBJECTIVES) == {"COVERAGE_GAIN", "DIVERSITY_GAIN"}
    # Oracle objectives must NOT leak into the strategy_sensitivity probe family.
    assert "COVERAGE_GAIN" not in PROBES
    assert "DIVERSITY_GAIN" not in PROBES


def test_coverage_gain_score_novel_vs_visited():
    trace = TraceState()
    card = {"coordinateContribution": [0.1, 0.1, 0.1, 0.1]}
    assert coverage_gain_score(card, trace) == 1.0  # empty trace -> novel
    trace.append_round(
        {
            "candidatePoolHash": "p",
            "slate": ["x"],
            "selectedCardId": "x",
            "coordinateContribution": [0.1, 0.1, 0.1, 0.1],
            "complexityBand": "low",
            "salienceScore": 0.0,
            "slotPermutation": [0],
        }
    )
    assert coverage_gain_score(card, trace) == 0.0  # same cell now visited


def test_diversity_gain_score_decays_with_repeats():
    trace = TraceState()
    card = {"complexityBand": "mid"}
    assert diversity_gain_score(card, trace) == 1.0
    for _ in range(2):
        trace.append_round(
            {
                "candidatePoolHash": "p",
                "slate": ["x"],
                "selectedCardId": "x",
                "coordinateContribution": [0.0, 0.0, 0.0, 0.0],
                "complexityBand": "mid",
                "salienceScore": 0.0,
                "slotPermutation": [0],
            }
        )
    # Two prior 'mid' rounds -> 1/(1+2).
    assert abs(diversity_gain_score(card, trace) - (1.0 / 3.0)) < 1e-12


def test_regret_for_matches_default_for_coord_novelty():
    pool, bases = _real_pool()
    oracle = OracleStrategyPolicy({"k": 4})  # default probe = coord novelty
    trace = run_episode(pool, oracle, 7, 8, 4, bases)
    ref = oracle_reference_from_objectives(_achieved_coordinate_novelty(trace))
    assert regret_to_oracle(trace, ref) == regret_to_oracle_for(
        trace, ref, "COORD_NOVELTY"
    )


def test_objective_oracle_self_regret_is_near_zero():
    pool, bases = _real_pool()
    for objective in ("COVERAGE_GAIN", "DIVERSITY_GAIN"):
        oracle = OracleStrategyPolicy({"k": 4, "objective": objective})
        trace = run_episode(pool, oracle, 7, 8, 4, bases)
        achieved = OBJECTIVE_ACHIEVERS[objective](trace)
        ref = oracle_reference_from_objectives(achieved)
        # The oracle's own achieved objective IS its reference -> regret 0.
        assert regret_to_oracle_for(trace, ref, objective) == 0.0


def test_coverage_oracle_covers_at_least_random():
    pool, bases = _real_pool()
    cov_oracle = OracleStrategyPolicy({"k": 4, "objective": "COVERAGE_GAIN"})
    rnd = RandomPolicy({"k": 4})
    cov = coordinate_coverage(run_episode(pool, cov_oracle, 7, 8, 4, bases))
    rnd_cov = coordinate_coverage(run_episode(pool, rnd, 7, 8, 4, bases))
    assert cov >= rnd_cov


def test_objective_oracle_deterministic_and_versioned():
    pool, bases = _real_pool()
    a = OracleStrategyPolicy({"k": 4, "objective": "COVERAGE_GAIN"})
    b = OracleStrategyPolicy({"k": 4, "objective": "COVERAGE_GAIN"})
    assert run_episode(pool, a, 7, 6, 4, bases).trace_hash() == run_episode(
        pool, b, 7, 6, 4, bases
    ).trace_hash()
    probe_oracle = OracleStrategyPolicy({"k": 4})
    assert a.policy_version() != probe_oracle.policy_version()


def test_unknown_objective_raises():
    import pytest

    pool, bases = _real_pool(8)
    with pytest.raises(KeyError):
        get_oracle_objective("NOPE")
    with pytest.raises(KeyError):
        OracleStrategyPolicy({"k": 4, "objective": "NOPE"}).select(
            pool, TraceState(), 7, {"k": 4}
        )
