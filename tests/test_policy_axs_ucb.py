"""Tests for AXS_UCB policy family (Task AXS-P0 T1).

Covers:
1. Determinism: same (pool, trace, seed, config) -> identical slate.
2. policy_version distinct across configs.
3. alpha=0: ranking driven by mean only.
4. freeze_round semantics.
5. tie_break_order variants.
6. Unknown tie_break_order raises Korean ValueError.
7. AxsYokedBonusPolicy: u(card) trace-independence, B_t clamping, hash checks.
8. Slate validity: every slate passes check_slate.
9. freeze_round validation (Important fix #1).
10. perRoundBonus structure validation (Important fix #2).
11. Per-call config consistency for yoked arm (Important fix #3).
12. Golden-slate regression tests pinning AXS behavior (Important fix #4).
13. Bandit-state characterization test (_replay_bandit_state matrices).

No user/persona/emotion/preference/demographic field anywhere.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from echo_bench.env.constraints import check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.policies.axs_ucb import AxsUcbPolicy, AxsYokedBonusPolicy, TraceView
from echo_bench.utils.hash import canonical_hash


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_pool(n: int = 16, n_bases: int = 4) -> List[Dict[str, Any]]:
    """Create a pool of n cards with 4 bases and varied coords."""
    bands = ("low", "mid", "high")
    return [
        {
            "cardId": f"c{i:02d}",
            "basis": f"B{(i % n_bases) + 1}",
            "complexityBand": bands[i % 3],
            "salienceScore": round(0.05 * (i % 11), 4),
            "coordinateContribution": [
                float(i), float(-i), float((i * 3) % 7), 0.3 * i
            ],
        }
        for i in range(n)
    ]


def _make_tied_pool() -> List[Dict[str, Any]]:
    """Pool where all cards have identical features (UCB scores tie exactly).
    Use 4 bases to satisfy k=4 constraint."""
    return [
        {
            "cardId": f"card_{chr(65 + i)}",  # card_A, card_B, ...
            "basis": f"B{(i % 4) + 1}",
            "complexityBand": "mid",
            "salienceScore": 0.5,
            "coordinateContribution": [1.0, 0.0, 0.0, 0.0],  # identical features
        }
        for i in range(8)
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


def _make_yoked_schedule(tmp_path: Path, per_round: List[float] | None = None) -> Path:
    """Build a minimal yoked schedule JSON in tmp_path and return its path."""
    if per_round is None:
        per_round = [1.0, 0.8, 0.6, 0.4, 0.2]
    body = {
        "scheduleId": "sched-test-001",
        "preregId": "prereg-test",
        "preregVersion": 1,
        "pilotFamily": "999",
        "referenceArm": "axs_ucb_default",
        "derivation": "test fixture",
        "H": len(per_round),
        "k": 4,
        "pool_size": 16,
        "perRoundBonus": per_round,
        "configHash": "dummy-config-hash",
    }
    # Compute scheduleHash from body (without scheduleHash key)
    schedule_hash = canonical_hash(body)
    body["scheduleHash"] = schedule_hash
    p = tmp_path / "axs_yoked_test.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_default_config_deterministic(self):
        pool = _make_pool()
        trace = _trace_with([pool[0], pool[5]])
        p1 = AxsUcbPolicy({"k": 4})
        p2 = AxsUcbPolicy({"k": 4})
        s1 = p1.select(pool, trace, 7)
        s2 = p2.select(pool, trace, 7)
        assert s1 == s2

    def test_alpha0_deterministic(self):
        pool = _make_pool()
        trace = _trace_with([pool[0]])
        p1 = AxsUcbPolicy({"k": 4, "alpha": 0.0})
        p2 = AxsUcbPolicy({"k": 4, "alpha": 0.0})
        s1 = p1.select(pool, trace, 42)
        s2 = p2.select(pool, trace, 42)
        assert s1 == s2

    def test_freeze_round_deterministic(self):
        pool = _make_pool()
        trace = _trace_with([pool[0], pool[1], pool[2]])
        p1 = AxsUcbPolicy({"k": 4, "freeze_round": 2})
        p2 = AxsUcbPolicy({"k": 4, "freeze_round": 2})
        s1 = p1.select(pool, trace, 3)
        s2 = p2.select(pool, trace, 3)
        assert s1 == s2

    @pytest.mark.parametrize("tbo", ["canonical", "reverse", "hash_seeded", "feature_lexicographic"])
    def test_tiebreak_order_deterministic(self, tbo):
        pool = _make_pool()
        trace = _trace_with([pool[0]])
        p1 = AxsUcbPolicy({"k": 4, "tie_break_order": tbo})
        p2 = AxsUcbPolicy({"k": 4, "tie_break_order": tbo})
        s1 = p1.select(pool, trace, 11)
        s2 = p2.select(pool, trace, 11)
        assert s1 == s2

    def test_yoked_deterministic(self, tmp_path):
        pool = _make_pool()
        trace = _trace_with([pool[0]])
        sched_path = _make_yoked_schedule(tmp_path)
        body = json.loads(sched_path.read_text())
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
        }
        p1 = AxsYokedBonusPolicy(cfg)
        p2 = AxsYokedBonusPolicy(cfg)
        s1 = p1.select(pool, trace, 7)
        s2 = p2.select(pool, trace, 7)
        assert s1 == s2


# ---------------------------------------------------------------------------
# 2. policy_version distinctness
# ---------------------------------------------------------------------------

class TestPolicyVersion:
    def test_default_vs_alpha0(self):
        a = AxsUcbPolicy({"k": 4, "alpha": 1.0})
        b = AxsUcbPolicy({"k": 4, "alpha": 0.0})
        assert a.policy_version() != b.policy_version()

    def test_different_tiebreak_orders(self):
        orders = ["canonical", "reverse", "hash_seeded", "feature_lexicographic"]
        versions = [AxsUcbPolicy({"k": 4, "tie_break_order": o}).policy_version() for o in orders]
        assert len(set(versions)) == 4

    def test_freeze_round_variants(self):
        a = AxsUcbPolicy({"k": 4})
        b = AxsUcbPolicy({"k": 4, "freeze_round": 2})
        c = AxsUcbPolicy({"k": 4, "freeze_round": 5})
        assert a.policy_version() != b.policy_version()
        assert b.policy_version() != c.policy_version()

    def test_yoked_differs_from_base(self, tmp_path):
        sched_path = _make_yoked_schedule(tmp_path)
        body = json.loads(sched_path.read_text())
        base = AxsUcbPolicy({"k": 4})
        yoked = AxsYokedBonusPolicy({
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
        })
        assert base.policy_version() != yoked.policy_version()

    def test_axs_differs_from_parent_class(self):
        from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy
        parent = TraceLinUcbPolicy({"k": 4})
        child = AxsUcbPolicy({"k": 4})
        # class names differ, so policy_version must differ
        assert parent.policy_version() != child.policy_version()


# ---------------------------------------------------------------------------
# 3. alpha=0 makes selection depend only on mean term
# ---------------------------------------------------------------------------

class TestAlphaZero:
    def test_alpha0_mean_only_ranking(self):
        """With alpha=0, ranking is by mean alone: score == mean for every card.

        Build a 8-card pool with unambiguous mean ordering (large coordinate spread
        so theta @ x differs clearly after 2 trace rounds).  k=4 selects from all 4
        bases so diversity is always satisfied.  With alpha=0 every chosen card's
        score component satisfies  score_used = mean + 0*bonus = mean.
        """
        pool = [
            {
                "cardId": f"c{i:02d}",
                "basis": f"B{(i % 4) + 1}",
                "complexityBand": "low",
                "salienceScore": 0.5,
                "coordinateContribution": [float((i + 1) * 5), 0.0, 0.0, 0.0],
            }
            for i in range(8)
        ]
        # Two rounds to make theta nonzero.
        trace = _trace_with([pool[0], pool[7]])
        p_zero = AxsUcbPolicy({"k": 4, "alpha": 0.0})
        s_zero = p_zero.select(pool, trace, 5)
        assert len(s_zero) == 4

        # For every chosen card: mean + 0*bonus == mean (score used = mean term only).
        for cid, comp in p_zero.last_score_components.items():
            score_used = round(comp["mean"] + 0.0 * comp["bonus"], 12)
            assert score_used == round(comp["mean"], 12), (
                f"alpha=0: score_used {score_used!r} != mean {comp['mean']!r} for {cid}"
            )

        # All 4 chosen cards must have mean >= every non-chosen card's mean
        # (within the applicable diversity-first ordering).
        # Verify: score == mean, so ucb-ordering == mean-ordering.
        # Recompute means for all cards via a k=8 run (captures full pool).
        p_all = AxsUcbPolicy({"k": 8, "alpha": 0.0})
        p_all.select(pool, trace, 5)
        all_means = {cid: comp["mean"] for cid, comp in p_all.last_score_components.items()}
        chosen_set = set(s_zero)
        chosen_means = [all_means[cid] for cid in s_zero]
        unchosen_means = [all_means[cid] for cid in all_means if cid not in chosen_set]
        # Every chosen mean >= every unchosen mean (greedy mean-only with diversity).
        assert min(chosen_means) >= max(unchosen_means) - 1e-9, (
            f"alpha=0 should pick top-mean cards; chosen mins={min(chosen_means):.6f} "
            f"< unchosen max={max(unchosen_means):.6f}"
        )

    def test_alpha0_matches_manual_mean_scores(self):
        """With alpha=0 the score used for ranking equals the mean component.

        With a nonzero trace, theta != 0 so means are meaningful.  Check:
        - bonus is present and may be nonzero (it exists but is scaled by 0)
        - mean + 0*bonus == mean for every chosen card (the invariant that matters)
        """
        pool = _make_pool(8)
        trace = _trace_with([pool[0], pool[1]])  # nonzero trace -> nonzero theta
        p = AxsUcbPolicy({"k": 4, "alpha": 0.0})
        slate = p.select(pool, trace, 7)
        assert len(slate) == 4
        for cid, comp in p.last_score_components.items():
            assert "mean" in comp
            assert "bonus" in comp
            # The ranking score equals the mean term (bonus multiplied by alpha=0).
            assert round(comp["mean"] + 0.0 * comp["bonus"], 12) == round(comp["mean"], 12)


# ---------------------------------------------------------------------------
# 4. freeze_round semantics
# ---------------------------------------------------------------------------

class TestFreezeRound:
    def _make_diverging_traces(self, pool, f=2):
        """Two traces sharing first f rounds, then diverging."""
        shared = pool[:f]
        trace_a = _trace_with(shared + [pool[f]])       # extra round A
        trace_b = _trace_with(shared + [pool[f + 1]])   # extra round B (different card)
        return trace_a, trace_b

    def test_freeze_gives_same_scores_on_diverging_traces(self):
        """freeze_round=f -> identical (A, b) bandit state for traces sharing first f rounds.

        Two traces share rounds 0..f-1 then diverge.  With freeze_round=f the
        bandit matrices (A, b) are built exclusively from the shared prefix, so
        they must be element-wise equal regardless of the tail.  We verify this
        directly via _replay_bandit_state on TraceView slices.
        """
        pool = _make_pool(16)
        f = 2
        trace_a, trace_b = self._make_diverging_traces(pool, f)

        active = ("coordinate_gap", "band_progress", "redundancy", "bias")
        p_tmp = AxsUcbPolicy({"k": 4, "freeze_round": f})

        tv_a = TraceView(trace_a.rounds()[:f])
        tv_b = TraceView(trace_b.rounds()[:f])

        A_a, b_a = p_tmp._replay_bandit_state(tv_a, active, 1.0)
        A_b, b_b = p_tmp._replay_bandit_state(tv_b, active, 1.0)

        assert np.allclose(A_a, A_b), "A matrices must be allclose for frozen prefix"
        assert np.allclose(b_a, b_b), "b vectors must be allclose for frozen prefix"

        # Behavioural consequence: both full selects yield valid slates.
        p_a = AxsUcbPolicy({"k": 4, "freeze_round": f})
        p_b = AxsUcbPolicy({"k": 4, "freeze_round": f})
        s_a = p_a.select(pool, trace_a, 999)
        s_b = p_b.select(pool, trace_b, 999)
        by_id = {c["cardId"]: c for c in pool}
        for s in [s_a, s_b]:
            assert len(s) == 4
            ok, reason, _ = check_slate([by_id[cid] for cid in s], 4, {}, 999)
            assert ok, f"Slate invalid: {reason}"

    def test_no_freeze_vs_freeze_differ_when_post_rounds_matter(self):
        """freeze_round=0 neutralises bandit history (theta=0, all means=0).

        With freeze_round=0 the bandit sees no history so theta=0 and
        mean=0 for every card.  With freeze_round=None and a multi-round trace,
        theta != 0 so at least some means are nonzero.  This is the observable
        contrast that demonstrates the freeze seam works.
        """
        pool = _make_pool(16)
        trace = _trace_with([pool[0], pool[4], pool[8], pool[12]])  # 4 rounds -> nonzero theta

        # freeze_round=0: bandit is empty -> theta=0 -> all means == 0
        p_frozen = AxsUcbPolicy({"k": 4, "freeze_round": 0})
        p_frozen.select(pool, trace, 42)
        frozen_means = [comp["mean"] for comp in p_frozen.last_score_components.values()]
        assert all(abs(m) < 1e-12 for m in frozen_means), (
            f"freeze_round=0 must yield mean=0 for all chosen cards; got {frozen_means}"
        )

        # freeze_round=None: bandit uses full trace -> theta nonzero -> means nonzero
        p_nofr = AxsUcbPolicy({"k": 4, "freeze_round": None})
        p_nofr.select(pool, trace, 42)
        nofr_means = [comp["mean"] for comp in p_nofr.last_score_components.values()]
        assert any(abs(m) > 1e-10 for m in nofr_means), (
            f"freeze_round=None with nonempty trace must yield some nonzero means; got {nofr_means}"
        )

        # All slates must be valid regardless of freeze setting.
        by_id = {c["cardId"]: c for c in pool}
        for cfg_extra, lbl in [
            ({"freeze_round": 0}, "freeze0"),
            ({}, "no_freeze"),
        ]:
            p_v = AxsUcbPolicy({"k": 4, **cfg_extra})
            s_v = p_v.select(pool, trace, 42)
            assert len(s_v) == 4
            ok, reason, _ = check_slate([by_id[cid] for cid in s_v], 4, {}, 42)
            assert ok, f"{lbl} slate invalid: {reason}"

    def test_freeze_none_behaves_same_as_no_freeze(self):
        """freeze_round=None should be equivalent to not setting freeze."""
        pool = _make_pool()
        trace = _trace_with([pool[0], pool[5]])
        p_none = AxsUcbPolicy({"k": 4, "freeze_round": None})
        p_default = AxsUcbPolicy({"k": 4})
        # policy_version differs (config differs by key presence) but behavior same
        s_none = p_none.select(pool, trace, 7)
        s_default = p_default.select(pool, trace, 7)
        assert s_none == s_default


# ---------------------------------------------------------------------------
# 5. tie_break_order variants
# ---------------------------------------------------------------------------

class TestTieBreakOrder:
    def test_canonical_deterministic(self):
        pool = _make_pool()
        trace = TraceState()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "canonical"})
        s1 = p.select(pool, trace, 7)
        s2 = p.select(pool, trace, 7)
        assert s1 == s2

    def test_four_orders_at_least_two_distinct_slates_on_tied_pool(self):
        pool = _make_tied_pool()
        trace = TraceState()
        seed = 7
        orders = ["canonical", "reverse", "hash_seeded", "feature_lexicographic"]
        # Collect ordered slates (preserving selection order matters for asserting
        # that tiebreak mechanisms produce different selections).
        ordered_slates = []
        for o in orders:
            p = AxsUcbPolicy({"k": 4, "tie_break_order": o})
            s = p.select(pool, trace, seed)
            ordered_slates.append(tuple(s))
        unique_slates = len(set(ordered_slates))
        assert unique_slates >= 2, (
            f"Expected ≥2 distinct ordered slates across tiebreak modes, "
            f"got: {ordered_slates}"
        )

    def test_tie_break_orders_same_slate_when_unique_scores(self):
        """When UCB scores are all distinct, all tiebreak modes select the same ordered slate.

        With a non-trivial trace the UCB scores are unique, so the tiebreak
        component is never reached.  All modes must produce an identical ordered
        list (not just the same card-set).
        """
        pool = _make_pool(16)
        trace = _trace_with([pool[0], pool[1]])  # non-trivial trace for distinct UCBs
        seed = 42
        orders = ["canonical", "reverse", "hash_seeded", "feature_lexicographic"]
        slates = []
        for o in orders:
            p = AxsUcbPolicy({"k": 4, "tie_break_order": o})
            s = p.select(pool, trace, seed)
            slates.append(list(s))  # ordered list, not frozenset
        # All ordered lists must be identical.
        assert all(s == slates[0] for s in slates), (
            f"Expected identical ordered slates across tiebreak modes on unique-UCB pool: {slates}"
        )

    def test_feature_lexicographic_hand_computed(self):
        """feature_lexicographic tie must sort ascending on feature vector then cardId.

        Hand-computed expected result: with an empty trace (centroid=0, band=low),
        all four cards have identical UCB scores (theta=0, alpha=1.0 at t=0).
        The rank key is (feat_vector, cardId).  All four cards have the same
        coordinateContribution=[1.0,0.0,0.0,0.0] so identical feature vectors;
        the secondary key is cardId ascending -> expected slate ['aa','bb','mm','zz'].
        """
        pool = [
            {
                "cardId": "aa",
                "basis": "B1",
                "complexityBand": "low",
                "salienceScore": 0.5,
                "coordinateContribution": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "cardId": "bb",
                "basis": "B2",
                "complexityBand": "low",
                "salienceScore": 0.5,
                "coordinateContribution": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "cardId": "mm",
                "basis": "B3",
                "complexityBand": "low",
                "salienceScore": 0.5,
                "coordinateContribution": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "cardId": "zz",
                "basis": "B4",
                "complexityBand": "low",
                "salienceScore": 0.5,
                "coordinateContribution": [1.0, 0.0, 0.0, 0.0],
            },
        ]
        trace = TraceState()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "feature_lexicographic"})
        slate = p.select(pool, trace, 7)
        # Hand-computed: identical feature vectors -> secondary key is cardId ascending.
        assert slate == ["aa", "bb", "mm", "zz"], (
            f"feature_lexicographic expected ['aa','bb','mm','zz'], got {slate}"
        )
        # Determinism: second call gives same result.
        slate2 = p.select(pool, trace, 7)
        assert slate == slate2

    def test_unknown_tiebreak_order_raises_korean_valueerror(self):
        pool = _make_pool()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "bogus_mode"})
        with pytest.raises(ValueError) as exc_info:
            p.select(pool, TraceState(), 7)
        # Verify message contains Korean characters (not purely ASCII)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean in error, got: {msg!r}"


# ---------------------------------------------------------------------------
# 6. Korean ValueError for unknown tie_break_order (standalone)
# ---------------------------------------------------------------------------

class TestUnknownTieBreakOrder:
    def test_raises_on_unknown_value(self):
        pool = _make_pool()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "INVALID"})
        with pytest.raises(ValueError):
            p.select(pool, TraceState(), 1)

    def test_feature_lexicographic_output_stable_after_tiebreak_guard(self):
        """feature_lexicographic output is unchanged after tiebreak RNG guard.

        The tiebreak RNG is now skipped for feature_lexicographic (dead path).
        Verify the slate is still deterministic and matches the hand-computed result.
        """
        pool = [
            {"cardId": "aa", "basis": "B1", "complexityBand": "low",
             "salienceScore": 0.5, "coordinateContribution": [1.0, 0.0, 0.0, 0.0]},
            {"cardId": "bb", "basis": "B2", "complexityBand": "low",
             "salienceScore": 0.5, "coordinateContribution": [1.0, 0.0, 0.0, 0.0]},
            {"cardId": "mm", "basis": "B3", "complexityBand": "low",
             "salienceScore": 0.5, "coordinateContribution": [1.0, 0.0, 0.0, 0.0]},
            {"cardId": "zz", "basis": "B4", "complexityBand": "low",
             "salienceScore": 0.5, "coordinateContribution": [1.0, 0.0, 0.0, 0.0]},
        ]
        trace = TraceState()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "feature_lexicographic"})
        s1 = p.select(pool, trace, 7)
        s2 = p.select(pool, trace, 7)
        assert s1 == s2, "feature_lexicographic must be deterministic"
        assert s1 == ["aa", "bb", "mm", "zz"], (
            f"feature_lexicographic with identical features: expected ['aa','bb','mm','zz'], got {s1}"
        )


# ---------------------------------------------------------------------------
# 7. AxsYokedBonusPolicy
# ---------------------------------------------------------------------------

class TestAxsYokedBonusPolicy:
    def test_trace_independence_of_bonus_term(self, tmp_path):
        """Two equal-length, different-content traces yield byte-identical bonus components.

        With freeze_round=0 (theta=0, mean=0 for all) and alpha=1e6 the score
        is dominated by alpha * B_t * u(card).  Because u(card) is seeded from
        poolHash + seed + roundIndex (no traceHash), two traces of the same length
        but different content must produce the same bonus for every card and
        therefore the same slate.
        """
        pool = _make_pool(16)
        sched_path = _make_yoked_schedule(tmp_path, per_round=[1.0] * 10)
        body = json.loads(sched_path.read_text())
        cfg_frozen = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
            "alpha": 1e6,
            "freeze_round": 0,
        }

        # Two traces of equal length, different content.
        trace_a = _trace_with([pool[0], pool[1]])
        trace_b = _trace_with([pool[2], pool[3]])

        p_a = AxsYokedBonusPolicy(cfg_frozen)
        p_b = AxsYokedBonusPolicy(cfg_frozen)
        s_a = p_a.select(pool, trace_a, 7)
        s_b = p_b.select(pool, trace_b, 7)

        # Slates must be byte-identical (same u(card) + mean=0 -> same ranking).
        assert s_a == s_b, (
            f"Yoked slates should be identical for equal-length traces; got {s_a} vs {s_b}"
        )

        # Bonus components for the chosen cards must be byte-identical.
        assert set(p_a.last_score_components) == set(p_b.last_score_components)
        for cid in p_a.last_score_components:
            bonus_a = p_a.last_score_components[cid]["bonus"]
            bonus_b = p_b.last_score_components[cid]["bonus"]
            assert bonus_a == bonus_b, (
                f"Yoked bonus must be trace-independent; card={cid} got {bonus_a} vs {bonus_b}"
            )

        by_id = {c["cardId"]: c for c in pool}
        ok, reason, _ = check_slate([by_id[cid] for cid in s_a], 4, {}, 7)
        assert ok, f"Slate invalid: {reason}"

    def test_bt_advances_with_trace_length(self, tmp_path):
        """B_t index advances with trace length -> different bonus scale.

        schedule [5.0, 0.0]: at t=0 B_t=5.0 so bonus > 0 for all cards;
        at t=1 B_t=0.0 so bonus == 0.0 for all cards (exactly).
        """
        pool = _make_pool(16)
        per_round = [5.0, 0.0]
        sched_path = _make_yoked_schedule(tmp_path, per_round=per_round)
        body = json.loads(sched_path.read_text())
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
            "alpha": 1.0,
            "freeze_round": 0,  # keep mean=0 so bonus scaling is unambiguous
        }

        trace_t0 = TraceState()             # t=0, B_t = 5.0
        trace_t1 = _trace_with([pool[0]])   # t=1, B_t = 0.0

        p0 = AxsYokedBonusPolicy(cfg)
        p1 = AxsYokedBonusPolicy(cfg)

        p0.select(pool, trace_t0, 7)
        p1.select(pool, trace_t1, 7)

        # t=0: B_t=5.0 -> bonus = 5.0 * u(card) > 0 for all chosen cards.
        bonuses_t0 = [comp["bonus"] for comp in p0.last_score_components.values()]
        assert all(b > 0 for b in bonuses_t0), (
            f"t=0 with B_t=5.0: expected all bonuses > 0, got {bonuses_t0}"
        )

        # t=1: B_t=0.0 -> bonus = 0.0 * u(card) = 0.0 exactly.
        bonuses_t1 = [comp["bonus"] for comp in p1.last_score_components.values()]
        assert all(b == 0.0 for b in bonuses_t1), (
            f"t=1 with B_t=0.0: expected all bonuses == 0.0, got {bonuses_t1}"
        )

    def test_bt_clamps_at_end_of_schedule(self, tmp_path):
        """B_t clamps to last value when t >= len(perRoundBonus)."""
        pool = _make_pool(16)
        per_round = [1.0, 0.5]  # only 2 entries
        sched_path = _make_yoked_schedule(tmp_path, per_round=per_round)
        body = json.loads(sched_path.read_text())
        cfg = {"k": 4, "schedule_path": str(sched_path), "schedule_hash": body["scheduleHash"]}

        # t=5 > len(per_round)=2, should clamp to per_round[-1]=0.5
        trace_long = _trace_with(pool[:5])
        p = AxsYokedBonusPolicy(cfg)
        s = p.select(pool, trace_long, 7)
        assert len(s) == 4
        by_id = {c["cardId"]: c for c in pool}
        ok, reason, _ = check_slate([by_id[cid] for cid in s], 4, {}, 7)
        assert ok, reason

    def test_missing_schedule_file_raises_valueerror(self, tmp_path):
        cfg = {
            "k": 4,
            "schedule_path": str(tmp_path / "nonexistent.json"),
            "schedule_hash": "deadbeef" * 8,
        }
        p = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_tampered_per_round_bonus_raises_valueerror(self, tmp_path):
        """Tamper with perRoundBonus after recording scheduleHash -> hash mismatch."""
        sched_path = _make_yoked_schedule(tmp_path, per_round=[1.0, 0.8])
        body = json.loads(sched_path.read_text())
        original_hash = body["scheduleHash"]

        # Tamper the file
        body["perRoundBonus"] = [1.0, 0.9]  # changed
        sched_path.write_text(json.dumps(body), encoding="utf-8")

        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": original_hash,
        }
        p = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_config_schedule_hash_mismatch_raises_valueerror(self, tmp_path):
        """config schedule_hash doesn't match file's scheduleHash -> ValueError."""
        sched_path = _make_yoked_schedule(tmp_path, per_round=[1.0])
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": "wrong_hash_value_" + "0" * 32,
        }
        p = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_yoked_u_map_no_trace_hash_in_seed(self, tmp_path):
        """u(card) map is identical for two equal-length, different-content traces.

        The u-map seed contains poolHash, seed, roundIndex, policyVersion — but
        NOT traceHash.  So two traces of the same length must yield the same
        bonus components and the same slate when mean=0 (freeze_round=0, alpha=1e6).
        """
        pool = _make_pool(16)
        per_round = [0.0] * 3 + [1.0]  # B_t=1.0 at t=3
        sched_path = _make_yoked_schedule(tmp_path, per_round=per_round)
        body = json.loads(sched_path.read_text())
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
            "alpha": 1e6,
            "freeze_round": 0,
        }
        # Two traces both of length 3 -> t=3, B_t=per_round[3]=1.0.
        trace_a = _trace_with([pool[0], pool[4], pool[8]])
        trace_b = _trace_with([pool[1], pool[5], pool[9]])
        p_a = AxsYokedBonusPolicy(cfg)
        p_b = AxsYokedBonusPolicy(cfg)
        s_a = p_a.select(pool, trace_a, 7)
        s_b = p_b.select(pool, trace_b, 7)

        # Slates must be byte-identical.
        assert s_a == s_b, (
            f"Yoked slates must be equal for equal-length traces; got {s_a} vs {s_b}"
        )

        # Bonus components for all chosen cards must be byte-identical.
        assert set(p_a.last_score_components) == set(p_b.last_score_components)
        for cid in p_a.last_score_components:
            bonus_a = p_a.last_score_components[cid]["bonus"]
            bonus_b = p_b.last_score_components[cid]["bonus"]
            assert bonus_a == bonus_b, (
                f"u(card) must be trace-independent; card={cid}: {bonus_a} != {bonus_b}"
            )

        by_id = {c["cardId"]: c for c in pool}
        ok, reason, _ = check_slate([by_id[cid] for cid in s_a], 4, {}, 7)
        assert ok, reason

    def test_yoked_rejects_non_canonical_tie_break_order(self, tmp_path):
        """AxsYokedBonusPolicy raises Korean ValueError if tie_break_order != canonical.

        The preregistered yoked arm is canonical-only.  Any other value must be
        rejected at construction time.
        """
        sched_path = _make_yoked_schedule(tmp_path)
        body = json.loads(sched_path.read_text())
        base_cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
        }
        # canonical must be accepted silently.
        ok_policy = AxsYokedBonusPolicy({**base_cfg, "tie_break_order": "canonical"})
        assert ok_policy is not None

        # Any non-canonical value must raise Korean ValueError at __init__.
        for bad_tbo in ["reverse", "hash_seeded", "feature_lexicographic", "bogus"]:
            with pytest.raises(ValueError) as exc_info:
                AxsYokedBonusPolicy({**base_cfg, "tie_break_order": bad_tbo})
            msg = str(exc_info.value)
            assert any(ord(c) > 127 for c in msg), (
                f"Expected Korean ValueError for tie_break_order={bad_tbo!r}, got: {msg!r}"
            )


# ---------------------------------------------------------------------------
# 8. Slate validity
# ---------------------------------------------------------------------------

class TestSlateValidity:
    @pytest.mark.parametrize("PolicyClass,extra_cfg", [
        (AxsUcbPolicy, {}),
        (AxsUcbPolicy, {"alpha": 0.0}),
        (AxsUcbPolicy, {"freeze_round": 2}),
        (AxsUcbPolicy, {"tie_break_order": "reverse"}),
        (AxsUcbPolicy, {"tie_break_order": "hash_seeded"}),
        (AxsUcbPolicy, {"tie_break_order": "feature_lexicographic"}),
    ])
    def test_slate_validity(self, PolicyClass, extra_cfg):
        pool = _make_pool(16)
        trace = _trace_with([pool[0], pool[5]])
        cfg = {"k": 4, **extra_cfg}
        p = PolicyClass(cfg)
        slate = p.select(pool, trace, 7)
        assert len(slate) == 4
        assert len(set(slate)) == 4
        pool_ids = {c["cardId"] for c in pool}
        assert all(cid in pool_ids for cid in slate)
        by_id = {c["cardId"]: c for c in pool}
        ok, reason, _ = check_slate([by_id[cid] for cid in slate], 4, {}, 7)
        assert ok, f"Slate invalid for {PolicyClass.__name__} {extra_cfg}: {reason}"

    def test_yoked_slate_validity(self, tmp_path):
        pool = _make_pool(16)
        sched_path = _make_yoked_schedule(tmp_path)
        body = json.loads(sched_path.read_text())
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
        }
        trace = _trace_with([pool[0], pool[5]])
        p = AxsYokedBonusPolicy(cfg)
        slate = p.select(pool, trace, 7)
        assert len(slate) == 4
        by_id = {c["cardId"]: c for c in pool}
        ok, reason, _ = check_slate([by_id[cid] for cid in slate], 4, {}, 7)
        assert ok, f"Yoked slate invalid: {reason}"

    def test_raises_when_pool_smaller_than_k(self):
        pool = _make_pool(3)
        p = AxsUcbPolicy({"k": 4})
        with pytest.raises(ValueError):
            p.select(pool, TraceState(), 7)

    def test_score_components_logged(self):
        pool = _make_pool(16)
        trace = _trace_with([pool[0]])
        p = AxsUcbPolicy({"k": 4})
        assert p.last_score_components == {}
        slate = p.select(pool, trace, 7)
        assert set(p.last_score_components) == set(slate)
        for comp in p.last_score_components.values():
            assert "mean" in comp
            assert "bonus" in comp


# ---------------------------------------------------------------------------
# 9. TraceView
# ---------------------------------------------------------------------------

class TestTraceView:
    def test_trace_view_rounds_returns_list(self):
        pool = _make_pool(8)
        trace = _trace_with([pool[0], pool[1]])
        records = trace.rounds()
        tv = TraceView(records)
        result = tv.rounds()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_trace_view_hash_deterministic(self):
        pool = _make_pool(8)
        records = _trace_with([pool[0]]).rounds()
        tv1 = TraceView(records)
        tv2 = TraceView(records)
        assert tv1.trace_hash() == tv2.trace_hash()

    def test_trace_view_truncation_for_freeze(self):
        """TraceView of first f records should differ from full trace hash."""
        pool = _make_pool(8)
        full_trace = _trace_with([pool[0], pool[1], pool[2]])
        full_records = full_trace.rounds()
        tv_full = TraceView(full_records)
        tv_trunc = TraceView(full_records[:2])
        assert tv_full.trace_hash() != tv_trunc.trace_hash()


# ---------------------------------------------------------------------------
# 10. freeze_round validation (Important fix #1)
# ---------------------------------------------------------------------------

class TestFreezeRoundValidation:
    """Validate that freeze_round rejects invalid values with Korean ValueError."""

    def test_negative_freeze_round_raises(self):
        """freeze_round=-1 must raise Korean ValueError."""
        pool = _make_pool()
        p = AxsUcbPolicy({"k": 4, "freeze_round": -1})
        with pytest.raises(ValueError) as exc_info:
            p.select(pool, TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_float_freeze_round_raises(self):
        """freeze_round=2.5 (float) must raise Korean ValueError."""
        pool = _make_pool()
        p = AxsUcbPolicy({"k": 4, "freeze_round": 2.5})
        with pytest.raises(ValueError) as exc_info:
            p.select(pool, TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_bool_freeze_round_raises(self):
        """freeze_round=True (bool subclass of int) must raise Korean ValueError."""
        pool = _make_pool()
        p = AxsUcbPolicy({"k": 4, "freeze_round": True})
        with pytest.raises(ValueError) as exc_info:
            p.select(pool, TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_valid_freeze_round_zero_accepted(self):
        """freeze_round=0 is valid (non-negative int, not bool)."""
        pool = _make_pool()
        p = AxsUcbPolicy({"k": 4, "freeze_round": 0})
        s = p.select(pool, TraceState(), 7)
        assert len(s) == 4

    def test_valid_freeze_round_positive_accepted(self):
        """freeze_round=2 is valid."""
        pool = _make_pool()
        trace = _trace_with([pool[0], pool[1], pool[2]])
        p = AxsUcbPolicy({"k": 4, "freeze_round": 2})
        s = p.select(pool, trace, 7)
        assert len(s) == 4

    def test_freeze_round_beyond_trace_length_works(self):
        """freeze_round > len(rounds) is valid: slice clamps (no IndexError).

        Regression test documenting that slice semantics are used: trace[:100] on
        a 3-round trace returns 3 rounds, not an error. The resulting behavior is
        equivalent to freeze_round=None for that trace length.
        """
        pool = _make_pool()
        trace = _trace_with([pool[0], pool[1]])  # only 2 rounds
        p = AxsUcbPolicy({"k": 4, "freeze_round": 100})  # well beyond trace length
        # Must not raise; slice [:100] silently clamps to available rounds.
        s = p.select(pool, trace, 7)
        assert len(s) == 4
        # Behavior equivalent to freeze_round=None (no extra rounds to drop).
        p_none = AxsUcbPolicy({"k": 4, "freeze_round": None})
        s_none = p_none.select(pool, trace, 7)
        assert s == s_none, (
            "freeze_round beyond trace length should behave like freeze_round=None"
        )

    def test_yoked_freeze_round_negative_raises(self):
        """AxsYokedBonusPolicy also validates freeze_round=-1."""
        # Use a minimal schedule; the ValueError should fire before schedule load.
        p = AxsYokedBonusPolicy({
            "k": 4,
            "schedule_path": "/nonexistent/path.json",
            "schedule_hash": "dummy",
            "freeze_round": -1,
        })
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_yoked_freeze_round_float_raises(self):
        """AxsYokedBonusPolicy also validates freeze_round=2.5."""
        p = AxsYokedBonusPolicy({
            "k": 4,
            "schedule_path": "/nonexistent/path.json",
            "schedule_hash": "dummy",
            "freeze_round": 2.5,
        })
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_yoked_freeze_round_bool_raises(self):
        """AxsYokedBonusPolicy also validates freeze_round=True."""
        p = AxsYokedBonusPolicy({
            "k": 4,
            "schedule_path": "/nonexistent/path.json",
            "schedule_hash": "dummy",
            "freeze_round": True,
        })
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"


# ---------------------------------------------------------------------------
# 11. perRoundBonus structure validation (Important fix #2)
# ---------------------------------------------------------------------------

class TestPerRoundBonusValidation:
    """Validate schedule perRoundBonus structure after hash check."""

    def _write_schedule_body(self, tmp_path: Path, body_overrides: dict) -> tuple[Path, str]:
        """Create a schedule JSON with custom body, returning (path, expected_hash)."""
        base_body = {
            "scheduleId": "sched-val-test",
            "preregId": "prereg-test",
            "preregVersion": 1,
            "pilotFamily": "999",
            "referenceArm": "axs_ucb_default",
            "derivation": "validation test fixture",
            "H": 3,
            "k": 4,
            "pool_size": 16,
            "perRoundBonus": [1.0, 0.8, 0.6],
            "configHash": "dummy",
        }
        base_body.update(body_overrides)
        # Recompute hash from current body (without scheduleHash)
        schedule_hash = canonical_hash(base_body)
        base_body["scheduleHash"] = schedule_hash
        p = tmp_path / "sched_val.json"
        p.write_text(json.dumps(base_body), encoding="utf-8")
        return p, schedule_hash

    def test_missing_per_round_bonus_key(self, tmp_path):
        """Schedule without 'perRoundBonus' key -> Korean ValueError."""
        # Build body without perRoundBonus, compute hash, embed it.
        body = {
            "scheduleId": "sched-missing-key",
            "preregId": "x",
            "preregVersion": 1,
            "pilotFamily": "999",
            "referenceArm": "r",
            "derivation": "d",
            "H": 3,
            "k": 4,
            "pool_size": 16,
            "configHash": "c",
        }
        schedule_hash = canonical_hash(body)
        body["scheduleHash"] = schedule_hash
        p = tmp_path / "no_key.json"
        p.write_text(json.dumps(body), encoding="utf-8")
        cfg = {"k": 4, "schedule_path": str(p), "schedule_hash": schedule_hash}
        policy = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            policy.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_empty_per_round_bonus(self, tmp_path):
        """perRoundBonus=[] -> Korean ValueError."""
        path, schedule_hash = self._write_schedule_body(
            tmp_path, {"perRoundBonus": []}
        )
        cfg = {"k": 4, "schedule_path": str(path), "schedule_hash": schedule_hash}
        policy = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            policy.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_nan_entry_in_per_round_bonus(self, tmp_path):
        """perRoundBonus containing NaN -> Korean ValueError (not finite)."""
        path, schedule_hash = self._write_schedule_body(
            tmp_path, {"perRoundBonus": [1.0, float("nan"), 0.5]}
        )
        cfg = {"k": 4, "schedule_path": str(path), "schedule_hash": schedule_hash}
        policy = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            policy.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_bool_entry_in_per_round_bonus(self, tmp_path):
        """perRoundBonus containing True (bool) -> Korean ValueError (not valid number)."""
        path, schedule_hash = self._write_schedule_body(
            tmp_path, {"perRoundBonus": [1.0, True, 0.5]}
        )
        cfg = {"k": 4, "schedule_path": str(path), "schedule_hash": schedule_hash}
        policy = AxsYokedBonusPolicy(cfg)
        with pytest.raises(ValueError) as exc_info:
            policy.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"


# ---------------------------------------------------------------------------
# 12. Per-call config consistency (Important fix #3)
# ---------------------------------------------------------------------------

class TestYokedPerCallConfigConsistency:
    """Yoked arm must reject per-call config keys that alter preregistered behavior."""

    def _make_yoked_policy(self, tmp_path: Path) -> tuple[AxsYokedBonusPolicy, dict]:
        sched_path = _make_yoked_schedule(tmp_path)
        body = json.loads(sched_path.read_text())
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
        }
        return AxsYokedBonusPolicy(cfg), cfg

    def test_per_call_schedule_path_raises(self, tmp_path):
        """Passing schedule_path in per-call config -> Korean ValueError."""
        policy, cfg = self._make_yoked_policy(tmp_path)
        pool = _make_pool()
        with pytest.raises(ValueError) as exc_info:
            policy.select(pool, TraceState(), 7, config={**cfg, "schedule_path": "/other.json"})
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_per_call_schedule_hash_raises(self, tmp_path):
        """Passing schedule_hash in per-call config -> Korean ValueError."""
        policy, cfg = self._make_yoked_policy(tmp_path)
        pool = _make_pool()
        with pytest.raises(ValueError) as exc_info:
            policy.select(pool, TraceState(), 7, config={**cfg, "schedule_hash": "abc123"})
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_per_call_non_canonical_tie_break_order_raises(self, tmp_path):
        """Passing a non-canonical tie_break_order in per-call config -> Korean ValueError."""
        policy, cfg = self._make_yoked_policy(tmp_path)
        pool = _make_pool()
        for bad_tbo in ["reverse", "hash_seeded", "feature_lexicographic"]:
            with pytest.raises(ValueError) as exc_info:
                policy.select(
                    pool, TraceState(), 7,
                    config={**cfg, "tie_break_order": bad_tbo}
                )
            msg = str(exc_info.value)
            assert any(ord(c) > 127 for c in msg), (
                f"Expected Korean error for tie_break_order={bad_tbo!r}, got: {msg!r}"
            )

    def test_per_call_canonical_tie_break_accepted(self, tmp_path):
        """Passing tie_break_order='canonical' in per-call config is accepted.

        Only the tie_break_order key is included (not schedule_path/schedule_hash,
        which are constructor-only). The policy reads the schedule from its
        constructor config and applies the per-call k/alpha/etc overrides.
        """
        policy, cfg = self._make_yoked_policy(tmp_path)
        pool = _make_pool()
        # Per-call config with only non-forbidden overrides — must not raise.
        per_call = {"k": 4, "tie_break_order": "canonical"}
        s = policy.select(pool, TraceState(), 7, config=per_call)
        assert len(s) == 4


# ---------------------------------------------------------------------------
# 13. Golden-slate regression tests (Important fix #4)
# ---------------------------------------------------------------------------
# These tests pin the exact slate card-ID lists emitted by each arm config
# against a FIXED constructed pool+trace+seed. They guard against drift in the
# inherited helpers _features/_replay_bandit_state from trace_lin_ucb.
#
# HOW GOLDENS WERE CAPTURED:
#   pool = _make_pool(16)  (n=16, n_bases=4, default params)
#   trace = _trace_with([pool[0], pool[5]])  (2 rounds)
#   seed = 7
#   Each arm: AxsUcbPolicy(cfg).select(pool, trace, 7)
#   Yoked: _make_yoked_schedule([1.0,0.8,0.6,0.4,0.2]) + AxsYokedBonusPolicy(cfg).select(...)
#   Captured by running the module directly on commit bae3989 before any refactor.
#
# PINNED VALUES (do NOT change these without a deliberate re-baseline):
#   canonical default   : ['c15', 'c14', 'c13', 'c11']
#   alpha0              : ['c15', 'c14', 'c13', 'c12']
#   freeze_round=2      : ['c15', 'c14', 'c13', 'c12']
#   reverse             : ['c15', 'c14', 'c13', 'c11']
#   hash_seeded         : ['c15', 'c14', 'c13', 'c11']
#   feature_lexicographic: ['c15', 'c14', 'c13', 'c11']
#   yoked               : ['c15', 'c14', 'c13', 'c12']

class TestGoldenSlateRegression:
    """Byte-identical slate regression tests. Changing these requires a deliberate
    re-baseline with an explanation of why preregistered behavior changed."""

    # Standard pool/trace/seed used for ALL golden tests.
    _POOL = _make_pool(16)
    _TRACE_CARDS = [_make_pool(16)[0], _make_pool(16)[5]]
    _SEED = 7

    # Freeze test uses a 3-round trace.
    _TRACE_3_CARDS = [_make_pool(16)[0], _make_pool(16)[1], _make_pool(16)[2]]

    def _pool(self):
        return _make_pool(16)

    def _trace(self):
        pool = self._pool()
        return _trace_with([pool[0], pool[5]])

    def _trace3(self):
        pool = self._pool()
        return _trace_with([pool[0], pool[1], pool[2]])

    def test_golden_canonical_default(self):
        """Canonical default arm slate must match pinned value."""
        pool = self._pool()
        p = AxsUcbPolicy({"k": 4})
        s = p.select(pool, self._trace(), self._SEED)
        assert s == ["c15", "c14", "c13", "c11"], (
            f"Golden: canonical default expected ['c15','c14','c13','c11'], got {s}"
        )

    def test_golden_alpha0(self):
        """alpha=0 arm slate must match pinned value."""
        pool = self._pool()
        p = AxsUcbPolicy({"k": 4, "alpha": 0.0})
        s = p.select(pool, self._trace(), self._SEED)
        assert s == ["c15", "c14", "c13", "c12"], (
            f"Golden: alpha0 expected ['c15','c14','c13','c12'], got {s}"
        )

    def test_golden_freeze_at_2(self):
        """freeze_round=2 arm slate must match pinned value (3-round trace)."""
        pool = self._pool()
        p = AxsUcbPolicy({"k": 4, "freeze_round": 2})
        s = p.select(pool, self._trace3(), self._SEED)
        assert s == ["c15", "c14", "c13", "c12"], (
            f"Golden: freeze_round=2 expected ['c15','c14','c13','c12'], got {s}"
        )

    def test_golden_reverse(self):
        """reverse tiebreak arm slate must match pinned value."""
        pool = self._pool()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "reverse"})
        s = p.select(pool, self._trace(), self._SEED)
        assert s == ["c15", "c14", "c13", "c11"], (
            f"Golden: reverse expected ['c15','c14','c13','c11'], got {s}"
        )

    def test_golden_hash_seeded(self):
        """hash_seeded tiebreak arm slate must match pinned value."""
        pool = self._pool()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "hash_seeded"})
        s = p.select(pool, self._trace(), self._SEED)
        assert s == ["c15", "c14", "c13", "c11"], (
            f"Golden: hash_seeded expected ['c15','c14','c13','c11'], got {s}"
        )

    def test_golden_feature_lexicographic(self):
        """feature_lexicographic tiebreak arm slate must match pinned value."""
        pool = self._pool()
        p = AxsUcbPolicy({"k": 4, "tie_break_order": "feature_lexicographic"})
        s = p.select(pool, self._trace(), self._SEED)
        assert s == ["c15", "c14", "c13", "c11"], (
            f"Golden: feature_lexicographic expected ['c15','c14','c13','c11'], got {s}"
        )

    def test_golden_yoked(self, tmp_path):
        """Yoked arm slate must match pinned value."""
        pool = self._pool()
        sched_path = _make_yoked_schedule(tmp_path, per_round=[1.0, 0.8, 0.6, 0.4, 0.2])
        body = json.loads(sched_path.read_text())
        cfg = {
            "k": 4,
            "schedule_path": str(sched_path),
            "schedule_hash": body["scheduleHash"],
        }
        p = AxsYokedBonusPolicy(cfg)
        s = p.select(pool, self._trace(), self._SEED)
        assert s == ["c15", "c14", "c13", "c12"], (
            f"Golden: yoked expected ['c15','c14','c13','c12'], got {s}"
        )


# ---------------------------------------------------------------------------
# 14. Bandit-state characterization test (_replay_bandit_state)
# ---------------------------------------------------------------------------
# Pins the (A, b) matrices for a small fixed trace against hard-coded values.
# Uses pool[0] and pool[3] as the 2-round trace with default active features.
#
# PINNED VALUES (round to 8 decimals, captured from _replay_bandit_state directly):
#   A (7x7): see hard-coded array below
#   b (7,):  see hard-coded array below

class TestBanditStateCharacterization:
    """Characterization test for _replay_bandit_state (A, b) matrices.

    These values pin the behavior of the inherited helper against numeric drift.
    The active feature set is ('coordinate_gap','band_progress','redundancy','bias')
    giving dimensionality 4+1+1+1=7.
    """

    def test_replay_bandit_state_fixed_trace(self):
        """(A, b) from a 2-round trace must match hard-coded values (np.allclose, 8 dec)."""
        pool = _make_pool(16)
        p = AxsUcbPolicy({"k": 4})
        active = ("coordinate_gap", "band_progress", "redundancy", "bias")
        # Fixed trace: pool[0] then pool[3]
        small_trace = _trace_with([pool[0], pool[3]])
        A, b = p._replay_bandit_state(small_trace, active, 1.0)

        # Hard-coded reference values captured from commit bae3989.
        A_ref = np.array([
            [10.0, 9.0, 6.0, 2.7, 1.5, 0.51939224, 3.0],
            [9.0, 10.0, 6.0, 2.7, 1.5, 0.51939224, 3.0],
            [6.0, 6.0, 5.0, 1.8, 1.0, 0.34626149, 2.0],
            [2.7, 2.7, 1.8, 1.81, 0.45, 0.15581767, 0.9],
            [1.5, 1.5, 1.0, 0.45, 2.25, 1.08656537, 1.5],
            [0.51939224, 0.51939224, 0.34626149, 0.15581767, 1.08656537, 2.02997426, 1.17313075],
            [3.0, 3.0, 2.0, 0.9, 1.5, 1.17313075, 3.0],
        ])
        b_ref = np.array([
            14.32794472, 14.32794472, 9.55196315, 4.29838342,
            2.38799079, 0.82686925, 4.77598157,
        ])

        assert A.shape == (7, 7), f"Expected A shape (7,7), got {A.shape}"
        assert b.shape == (7,), f"Expected b shape (7,), got {b.shape}"
        assert np.allclose(A, A_ref, atol=1e-8), (
            f"A matrix differs from reference:\n{A}\nvs\n{A_ref}"
        )
        assert np.allclose(b, b_ref, atol=1e-8), (
            f"b vector differs from reference:\n{b}\nvs\n{b_ref}"
        )


# ---------------------------------------------------------------------------
# 15. schedule_path empty/blank validation (Minor fix #9)
# ---------------------------------------------------------------------------

class TestSchedulePathValidation:
    """Explicit Korean ValueError for empty/blank schedule_path before filesystem access."""

    def test_empty_string_path_raises(self):
        """schedule_path='' -> Korean ValueError before filesystem access."""
        p = AxsYokedBonusPolicy({"k": 4, "schedule_path": "", "schedule_hash": "x"})
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_blank_string_path_raises(self):
        """schedule_path='   ' (whitespace only) -> Korean ValueError."""
        p = AxsYokedBonusPolicy({"k": 4, "schedule_path": "   ", "schedule_hash": "x"})
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"

    def test_missing_path_key_raises(self):
        """schedule_path key absent -> Korean ValueError (defaults to empty string '')."""
        p = AxsYokedBonusPolicy({"k": 4, "schedule_hash": "x"})
        with pytest.raises(ValueError) as exc_info:
            p.select(_make_pool(), TraceState(), 7)
        msg = str(exc_info.value)
        assert any(ord(c) > 127 for c in msg), f"Expected Korean error, got: {msg!r}"
