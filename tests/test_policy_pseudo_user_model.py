"""Tests for the PSEUDO_USER_MODEL policy (Task C-006).

Exercises the C-006 contract and the CRITICAL isolation invariants:

- slate has length ``k``, distinct cardIds, all from the pool, satisfies
  :func:`echo_bench.env.constraints.check_slate`;
- deterministic replay; ``policyVersion`` changes with config; ``scoreComponents``
  logged per chosen card;
- (a) the synthetic latent vector NEVER appears in the round records / trace
  after a run;
- (b) NO trace-only policy module (trace_greedy.py, trace_lin_ucb.py) imports or
  references ``pseudo_user_model``.

The latent vector is a SYNTHETIC scoring construct, not a model of a person.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

import echo_bench.policies.pseudo_user_model as pum_module
import echo_bench.policies.trace_greedy as trace_greedy_module
import echo_bench.policies.trace_lin_ucb as lin_ucb_module
from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.pseudo_user_model import PseudoUserModelPolicy


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


def test_slate_length_distinct_and_from_pool() -> None:
    pool = _make_pool()
    policy = PseudoUserModelPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    assert len(slate) == 4
    assert len(set(slate)) == 4
    assert all(cid in {c["cardId"] for c in pool} for cid in slate)


def test_slate_satisfies_constraints() -> None:
    pool = _make_pool()
    policy = PseudoUserModelPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    by_id = {c["cardId"]: c for c in pool}
    ok, reason, perm = check_slate([by_id[c] for c in slate], 4, {}, 7)
    assert ok is True, reason


def test_deterministic_replay() -> None:
    pool = _make_pool()
    p1 = PseudoUserModelPolicy({"k": 4})
    p2 = PseudoUserModelPolicy({"k": 4})
    assert p1.select(pool, TraceState(), 7) == p2.select(pool, TraceState(), 7)


def test_policy_version_changes_with_latent_seed() -> None:
    a = PseudoUserModelPolicy({"k": 4, "latent_seed": 1})
    b = PseudoUserModelPolicy({"k": 4, "latent_seed": 2})
    assert a.policy_version() != b.policy_version()


def test_score_components_logged() -> None:
    pool = _make_pool()
    policy = PseudoUserModelPolicy({"k": 4})
    slate = policy.select(pool, TraceState(), 7)
    assert set(policy.last_score_components) == set(slate)
    for comp in policy.last_score_components.values():
        assert "latentScore" in comp


def test_different_latent_seed_changes_slate() -> None:
    pool = _make_pool(24)
    s1 = PseudoUserModelPolicy({"k": 4, "latent_seed": 1}).select(pool, TraceState(), 7)
    s2 = PseudoUserModelPolicy({"k": 4, "latent_seed": 999}).select(pool, TraceState(), 7)
    assert s1 != s2


def test_latent_vector_never_leaks_into_trace() -> None:
    """(a) The synthetic latent vector must not appear in round records / trace.

    Simulate a run by recording each selection into a TraceState exactly as the
    round runner would (only observable fields), then assert the latent vector
    and its config knobs do not appear anywhere in the recorded rounds, the
    scoreComponents, or the trace hash inputs.
    """
    pool = _make_pool()
    by_id = {c["cardId"]: c for c in pool}
    policy = PseudoUserModelPolicy({"k": 4, "latent_seed": 4242})

    # The actual latent vector (module-internal) for this config.
    latent = policy._latent_vector(7, 4242)
    latent_values = [round(float(v), 12) for v in latent]

    trace = TraceState()
    for _ in range(3):
        slate = policy.select(pool, trace, 7)
        ok, _reason, perm = check_slate([by_id[c] for c in slate], 4, {}, 7)
        assert ok
        selected = slate[0]
        sel_card = by_id[selected]
        trace.append_round(
            {
                "candidatePoolHash": "ph",
                "slate": slate,
                "selectedCardId": selected,
                "coordinateContribution": sel_card["coordinateContribution"],
                "complexityBand": sel_card["complexityBand"],
                "salienceScore": sel_card["salienceScore"],
                "slotPermutation": perm,
            }
        )

    rounds_repr = repr(trace.rounds())
    # No latent component value leaks into recorded rounds.
    for v in latent_values:
        assert str(v) not in rounds_repr
    # No latent-y key names appear in the trace records.
    for record in trace.rounds():
        for key in record:
            assert key not in {"latent", "latent_vector", "latentVector", "user_model"}
    # scoreComponents only carry the scalar score, never the vector itself.
    for comp in policy.last_score_components.values():
        assert set(comp) == {"latentScore"}


def test_trace_only_policies_do_not_import_pseudo_user_model() -> None:
    """(b) trace_greedy.py and trace_lin_ucb.py must not reference this module."""
    for mod in (trace_greedy_module, lin_ucb_module):
        src = inspect.getsource(mod)
        assert "pseudo_user_model" not in src, mod.__name__
        assert "PseudoUserModel" not in src, mod.__name__


def test_module_labelled_contrast_baseline() -> None:
    """The module docstring must label it a contrast / synthetic construct."""
    doc = (pum_module.__doc__ or "").lower()
    assert "contrast baseline" in doc
    assert "synthetic" in doc
    assert "not a model of a real person" in doc


def test_raises_when_pool_smaller_than_k() -> None:
    pool = _make_pool(3)
    policy = PseudoUserModelPolicy({"k": 4})
    with pytest.raises(ValueError):
        policy.select(pool, TraceState(), 7)


# --- C-008: stronger contrast-baseline variants -----------------------------

from echo_bench.archive.builder import build_archive  # noqa: E402
from echo_bench.basis.schema import load_bases  # noqa: E402
from echo_bench.env.round_runner import run_episode  # noqa: E402
from echo_bench.metrics.utility import artifact_diversity  # noqa: E402
from echo_bench.policies.pseudo_user_model import VARIANTS  # noqa: E402

import yaml  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _real_pool(n: int = 32):
    bases = load_bases(_REPO / "configs" / "basis" / "bases.yaml")
    with open(_REPO / "configs" / "archive" / "archive.yaml", encoding="utf-8") as fh:
        archive_cfg = yaml.safe_load(fh)
    archive = build_archive(bases, archive_cfg, 42)
    return archive["cards"][:n], bases


def test_variants_registered():
    assert set(VARIANTS) == {"BASIC", "DIVERSITY_REG", "SESSION_EMBEDDING"}


def test_basic_variant_matches_default():
    pool = _make_pool()
    default = PseudoUserModelPolicy({"k": 4}).select(pool, TraceState(), 7)
    basic = PseudoUserModelPolicy({"k": 4, "variant": "BASIC"}).select(
        pool, TraceState(), 7
    )
    assert default == basic


def test_basic_variant_collapses_but_reg_variants_do_not():
    # The BASIC static-latent baseline collapses artifact_diversity toward 0;
    # the regularized variants must keep it strictly positive (no straw man).
    pool, bases = _real_pool()
    basic = PseudoUserModelPolicy({"k": 4, "variant": "BASIC"})
    basic_div = artifact_diversity(run_episode(pool, basic, 7, 12, 4, bases))
    for variant in ("DIVERSITY_REG", "SESSION_EMBEDDING"):
        policy = PseudoUserModelPolicy({"k": 4, "variant": variant})
        div = artifact_diversity(run_episode(pool, policy, 7, 12, 4, bases))
        assert div > 0.0, variant
        assert div > basic_div, variant


def test_variants_are_deterministic():
    pool = _make_pool(24)
    for variant in VARIANTS:
        a = PseudoUserModelPolicy({"k": 4, "variant": variant}).select(
            pool, TraceState(), 7
        )
        b = PseudoUserModelPolicy({"k": 4, "variant": variant}).select(
            pool, TraceState(), 7
        )
        assert a == b, variant


def test_variant_changes_policy_version():
    base = PseudoUserModelPolicy({"k": 4, "variant": "BASIC"}).policy_version()
    reg = PseudoUserModelPolicy({"k": 4, "variant": "DIVERSITY_REG"}).policy_version()
    assert base != reg


def test_unknown_variant_raises():
    pool = _make_pool()
    with pytest.raises(ValueError):
        PseudoUserModelPolicy({"k": 4, "variant": "NOPE"}).select(pool, TraceState(), 7)
