"""Tests for echo_bench.metrics.robustness (Task D-003).

Covers: fault transforms are pure (no input mutation) and deterministic; they
are enumerated in FAULTS; robustness_score is bounded [0, 1] and deterministic;
metadata carries both traceHash references.
"""

from __future__ import annotations

import copy

from echo_bench.metrics.robustness import (
    FAULTS,
    ROBUSTNESS_DIRECTION,
    SALIENCE_SCORE_MAX,
    SALIENCE_SCORE_MIN,
    basis_dropout,
    pool_shrink,
    robustness_score,
    robustness_score_with_metadata,
    salience_perturb,
    sensitivity_score,
)
from echo_bench.metrics.utility import CORE_METRIC_KEYS


def _pool() -> list:
    return [
        {"cardId": "c1", "basis": "B1", "salienceScore": 0.10},
        {"cardId": "c2", "basis": "B2", "salienceScore": 0.50},
        {"cardId": "c3", "basis": "B3", "salienceScore": 0.90},
        {"cardId": "c4", "basis": "B1", "salienceScore": 0.30},
        {"cardId": "c5", "basis": "B4", "salienceScore": 0.70},
        {"cardId": "c6", "basis": "B2", "salienceScore": 0.20},
    ]


def test_faults_registry_enumerates_transforms():
    assert set(FAULTS) == {"pool_shrink", "basis_dropout", "salience_perturb"}
    assert FAULTS["pool_shrink"] is pool_shrink
    assert FAULTS["basis_dropout"] is basis_dropout
    assert FAULTS["salience_perturb"] is salience_perturb


def test_pool_shrink_pure_and_deterministic():
    pool = _pool()
    before = copy.deepcopy(pool)
    out1 = pool_shrink(pool, frac=0.5, seed=7)
    out2 = pool_shrink(pool, frac=0.5, seed=7)
    # Pure: input unchanged.
    assert pool == before
    # Deterministic.
    assert out1 == out2
    # Drops half (6 -> 3).
    assert len(out1) == 3
    # Returned cards are independent copies.
    out1[0]["salienceScore"] = 999
    assert pool == before


def test_pool_shrink_bounds():
    pool = _pool()
    assert pool_shrink(pool, frac=0.0, seed=1) == pool
    assert pool_shrink(pool, frac=1.0, seed=1) == []
    # frac clamped.
    assert pool_shrink(pool, frac=2.0, seed=1) == []
    assert pool_shrink(pool, frac=-1.0, seed=1) == pool


def test_basis_dropout_pure_and_deterministic():
    pool = _pool()
    before = copy.deepcopy(pool)
    out1 = basis_dropout(pool, "B1")
    out2 = basis_dropout(pool, "B1")
    assert pool == before
    assert out1 == out2
    assert all(c["basis"] != "B1" for c in out1)
    assert len(out1) == 4  # two B1 cards removed
    # Multiple bases.
    out_multi = basis_dropout(pool, ["B1", "B2"])
    assert all(c["basis"] not in {"B1", "B2"} for c in out_multi)
    assert len(out_multi) == 2


def test_salience_perturb_pure_deterministic_and_bounded():
    pool = _pool()
    before = copy.deepcopy(pool)
    out1 = salience_perturb(pool, delta=0.2, seed=42)
    out2 = salience_perturb(pool, delta=0.2, seed=42)
    assert pool == before
    assert out1 == out2
    assert len(out1) == len(pool)
    for card in out1:
        assert SALIENCE_SCORE_MIN <= card["salienceScore"] <= SALIENCE_SCORE_MAX
    # delta=0 is the identity on salience.
    out0 = salience_perturb(pool, delta=0.0, seed=42)
    for orig, new in zip(pool, out0):
        assert new["salienceScore"] == orig["salienceScore"]


def test_salience_perturb_order_independent_per_card():
    pool = _pool()
    shuffled = list(reversed(pool))
    out_pool = {c["cardId"]: c["salienceScore"] for c in salience_perturb(pool, 0.2, 9)}
    out_shuf = {c["cardId"]: c["salienceScore"] for c in salience_perturb(shuffled, 0.2, 9)}
    assert out_pool == out_shuf


def test_robustness_score_bounded_and_zero_when_identical():
    baseline = {
        "traceHash": "ha",
        "coordinate_coverage": 0.5,
        "artifact_diversity": 0.4,
        "redundancy_rate": 0.1,
        "round_coherence": 0.8,
    }
    assert robustness_score(baseline, baseline) == 0.0
    faulted = dict(baseline)
    faulted["traceHash"] = "hb"
    faulted["coordinate_coverage"] = 0.9  # diff 0.4
    faulted["redundancy_rate"] = 0.3  # diff 0.2
    score = robustness_score(baseline, faulted)
    assert 0.0 <= score <= 1.0
    # mean of |.4|,0,|.2|,0 over 4 keys = 0.15
    assert abs(score - 0.15) < 1e-12


def test_robustness_score_deterministic():
    a = {"traceHash": "a", "m": 0.2, "n": 0.7}
    b = {"traceHash": "b", "m": 0.5, "n": 0.1}
    assert robustness_score(a, b) == robustness_score(a, b)


def test_robustness_score_no_shared_keys_is_zero():
    assert robustness_score({"traceHash": "a"}, {"traceHash": "b"}) == 0.0


def test_robustness_metadata_carries_both_trace_hashes():
    a = {"traceHash": "hash-a", "m": 0.2}
    b = {"traceHash": "hash-b", "m": 0.8}
    md = robustness_score_with_metadata(a, b)
    assert md["baselineTraceHash"] == "hash-a"
    assert md["faultedTraceHash"] == "hash-b"
    assert md["sharedKeys"] == ["m"]
    assert abs(md["value"] - 0.6) < 1e-12
    # D-012: primary label and legacy alias must be present in metadata.
    assert md["metric"] == "sensitivity_score", (
        f"robustness_score_with_metadata must set metric='sensitivity_score', got {md['metric']!r}"
    )
    assert md["legacyAlias"] == "robustness_score", (
        f"robustness_score_with_metadata must set legacyAlias='robustness_score', "
        f"got {md['legacyAlias']!r}"
    )


def test_sensitivity_score_is_identical_to_robustness_score():
    # D-008: sensitivity_score is an unambiguously-named alias of the SAME value.
    # It exists only to make report direction readable; it must never diverge.
    baseline = {
        "traceHash": "ha",
        "coordinate_coverage": 0.5,
        "artifact_diversity": 0.4,
        "redundancy_rate": 0.1,
        "round_coherence": 0.8,
    }
    faulted = dict(baseline)
    faulted["traceHash"] = "hb"
    faulted["coordinate_coverage"] = 0.9
    faulted["redundancy_rate"] = 0.3
    assert sensitivity_score(baseline, faulted) == robustness_score(baseline, faulted)
    assert sensitivity_score(baseline, baseline) == 0.0


def test_robustness_direction_note_is_present_and_unambiguous():
    # D-008: the direction must be explicit — 0.0 = maximal robustness.
    assert isinstance(ROBUSTNESS_DIRECTION, str)
    assert "0.0" in ROBUSTNESS_DIRECTION
    assert "robust" in ROBUSTNESS_DIRECTION.lower()


def test_robustness_metadata_surfaces_sensitivity_and_direction():
    # D-008: metadata carries the identically-valued sensitivity field and the
    # direction note, without changing the existing `value`.
    a = {"traceHash": "hash-a", "m": 0.2}
    b = {"traceHash": "hash-b", "m": 0.8}
    md = robustness_score_with_metadata(a, b)
    assert md["sensitivityScore"] == md["value"]
    assert md["direction"] == ROBUSTNESS_DIRECTION


# ---- D-010 review: CORE_METRIC_KEYS pinning tests ----

def test_robustness_score_with_keys_pinning_ignores_extra_keys():
    """Pinning to 4 keys excludes the 3 D-010 distribution keys from the mean.

    Two dicts share 7 numeric keys (the full METRIC_KEYS set after D-010).
    Pinned to CORE_METRIC_KEYS (4 keys), only those four contribute to the mean
    absolute difference. The value is verified by hand.
    """
    baseline = {
        "traceHash": "base",
        # CORE_METRIC_KEYS (4)
        "coordinate_coverage": 0.5,
        "artifact_diversity": 0.4,
        "redundancy_rate": 0.1,
        "round_coherence": 0.8,
        # D-010 distribution extras (3): deliberately set large diffs to
        # confirm they are excluded from the denominator under pinning.
        "coordinate_entropy": 0.0,
        "cell_visit_gini": 0.0,
        "time_to_saturation": 0.0,
    }
    faulted = {
        "traceHash": "fault",
        "coordinate_coverage": 0.9,   # diff 0.4
        "artifact_diversity": 0.4,    # diff 0.0
        "redundancy_rate": 0.3,       # diff 0.2
        "round_coherence": 0.8,       # diff 0.0
        "coordinate_entropy": 1.0,    # diff 1.0 — should be ignored when pinned
        "cell_visit_gini": 1.0,       # diff 1.0 — should be ignored when pinned
        "time_to_saturation": 1.0,    # diff 1.0 — should be ignored when pinned
    }

    # Pinned to 4 core keys: mean of |0.4|, |0.0|, |0.2|, |0.0| = 0.15
    pinned = robustness_score(baseline, faulted, keys=CORE_METRIC_KEYS)
    assert abs(pinned - 0.15) < 1e-12, f"expected 0.15, got {pinned}"

    # Without pinning: all 7 shared keys average including the 1.0 diffs.
    # mean of |0.4|, |0.0|, |0.2|, |0.0|, |1.0|, |1.0|, |1.0| = 3.6/7
    unpinned = robustness_score(baseline, faulted)
    expected_unpinned = (0.4 + 0.0 + 0.2 + 0.0 + 1.0 + 1.0 + 1.0) / 7
    assert abs(unpinned - expected_unpinned) < 1e-12

    # Confirm the values differ, showing pinning has a real effect.
    assert abs(pinned - unpinned) > 0.1


def test_robustness_score_keys_none_is_default_behaviour():
    """Passing keys=None gives the same result as the zero-arg call."""
    a = {"traceHash": "a", "x": 0.3, "y": 0.7}
    b = {"traceHash": "b", "x": 0.5, "y": 0.1}
    assert robustness_score(a, b, keys=None) == robustness_score(a, b)


def test_robustness_metadata_records_metric_keys_when_pinned():
    """robustness_score_with_metadata carries 'metricKeys' when keys is supplied."""
    baseline = {
        "traceHash": "base",
        "coordinate_coverage": 0.5,
        "artifact_diversity": 0.4,
        "redundancy_rate": 0.1,
        "round_coherence": 0.8,
        "coordinate_entropy": 0.0,
        "cell_visit_gini": 0.0,
        "time_to_saturation": 0.0,
    }
    faulted = dict(baseline)
    faulted["traceHash"] = "fault"
    faulted["coordinate_coverage"] = 0.9

    md_pinned = robustness_score_with_metadata(
        baseline, faulted, keys=CORE_METRIC_KEYS
    )
    # metricKeys is self-describing: records the four pinned keys.
    assert md_pinned["metricKeys"] == list(CORE_METRIC_KEYS)
    # sharedKeys reflects only the four resolved keys.
    assert set(md_pinned["sharedKeys"]) == set(CORE_METRIC_KEYS)

    # Without pinning, metricKeys is None (dynamic intersection).
    md_dynamic = robustness_score_with_metadata(baseline, faulted)
    assert md_dynamic["metricKeys"] is None
