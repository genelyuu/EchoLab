"""Tests for echo_bench.env.constraints (Task B-003, minimal)."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo_bench.basis.schema import load_bases
from echo_bench.env.constraints import (
    REASON_INSUFFICIENT_BASES,
    REASON_K_MISMATCH,
    apply_permutation,
    check_slate,
)

_BASES_PATH = Path(__file__).resolve().parents[1] / "configs" / "basis" / "bases.yaml"


@pytest.fixture(scope="module")
def bases_cfg():
    return load_bases(_BASES_PATH)


def _card(card_id: str, basis: str) -> dict:
    """Construct a minimal card dict (only the fields constraints reads)."""
    return {
        "cardId": card_id,
        "basis": basis,
        "complexityBand": "mid",
        "salienceScore": 0.5,
    }


# ---------------------------------------------------------------------------
# k == 4
# ---------------------------------------------------------------------------


def test_k4_three_distinct_bases_ok(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B1"),
    ]
    ok, reason, perm = check_slate(slate, 4, bases_cfg, seed=7)
    assert ok is True
    assert reason is None
    assert len(perm) == 4
    assert sorted(perm) == [0, 1, 2, 3]


def test_k4_four_distinct_bases_ok(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B4"),
    ]
    ok, reason, perm = check_slate(slate, 4, bases_cfg, seed=7)
    assert ok is True
    assert reason is None
    assert len(perm) == 4


def test_k4_only_two_bases_rejected(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B1"),
        _card("c4", "B2"),
    ]
    ok, reason, perm = check_slate(slate, 4, bases_cfg, seed=7)
    assert ok is False
    assert reason == REASON_INSUFFICIENT_BASES
    assert perm == []


# ---------------------------------------------------------------------------
# k == 2
# ---------------------------------------------------------------------------


def test_k2_two_different_bases_ok(bases_cfg):
    slate = [_card("c1", "B1"), _card("c2", "B2")]
    ok, reason, perm = check_slate(slate, 2, bases_cfg, seed=3)
    assert ok is True
    assert reason is None
    assert sorted(perm) == [0, 1]


def test_k2_same_basis_twice_rejected(bases_cfg):
    slate = [_card("c1", "B1"), _card("c2", "B1")]
    ok, reason, perm = check_slate(slate, 2, bases_cfg, seed=3)
    assert ok is False
    assert reason == REASON_INSUFFICIENT_BASES
    assert perm == []


# ---------------------------------------------------------------------------
# k == 6 (prefer 4 distinct, 3 still accepted)
# ---------------------------------------------------------------------------


def test_k6_three_bases_ok_with_note(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B1"),
        _card("c5", "B2"),
        _card("c6", "B3"),
    ]
    ok, reason, perm = check_slate(slate, 6, bases_cfg, seed=11)
    assert ok is True
    assert reason is None
    assert len(perm) == 6


def test_k6_two_bases_rejected(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B1"),
        _card("c4", "B2"),
        _card("c5", "B1"),
        _card("c6", "B2"),
    ]
    ok, reason, perm = check_slate(slate, 6, bases_cfg, seed=11)
    assert ok is False
    assert reason == REASON_INSUFFICIENT_BASES


# ---------------------------------------------------------------------------
# k mismatch
# ---------------------------------------------------------------------------


def test_len_mismatch_rejected(bases_cfg):
    slate = [_card("c1", "B1"), _card("c2", "B2"), _card("c3", "B3")]
    ok, reason, perm = check_slate(slate, 4, bases_cfg, seed=7)
    assert ok is False
    assert reason == REASON_K_MISMATCH
    assert perm == []


# ---------------------------------------------------------------------------
# non-standard k
# ---------------------------------------------------------------------------


def test_non_standard_k_default_rule(bases_cfg):
    # k==3 -> needs min(3, 2) = 2 distinct bases.
    slate = [_card("c1", "B1"), _card("c2", "B2"), _card("c3", "B1")]
    ok, reason, perm = check_slate(slate, 3, bases_cfg, seed=5)
    assert ok is True
    assert reason is None
    assert len(perm) == 3

    # Single basis -> rejected.
    slate2 = [_card("c1", "B1"), _card("c2", "B1"), _card("c3", "B1")]
    ok2, reason2, _ = check_slate(slate2, 3, bases_cfg, seed=5)
    assert ok2 is False
    assert reason2 == REASON_INSUFFICIENT_BASES


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_same_inputs_identical_outcome_and_permutation(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B4"),
    ]
    ok1, reason1, perm1 = check_slate(slate, 4, bases_cfg, seed=42)
    ok2, reason2, perm2 = check_slate(list(slate), 4, bases_cfg, seed=42)
    assert (ok1, reason1) == (ok2, reason2)
    assert perm1 == perm2


def test_permutation_invariant_to_input_order(bases_cfg):
    # Same logical slate, different input ordering -> same permutation,
    # because seeding is on sorted cardIds.
    slate_a = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B4"),
    ]
    slate_b = [slate_a[2], slate_a[0], slate_a[3], slate_a[1]]
    _, _, perm_a = check_slate(slate_a, 4, bases_cfg, seed=42)
    _, _, perm_b = check_slate(slate_b, 4, bases_cfg, seed=42)
    assert perm_a == perm_b


def test_different_seed_typically_different_permutation_same_ok(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B4"),
    ]
    perms = set()
    for s in range(8):
        ok, reason, perm = check_slate(slate, 4, bases_cfg, seed=s)
        assert ok is True
        assert reason is None
        perms.add(tuple(perm))
    # Accept outcome is stable across seeds; permutations vary.
    assert len(perms) > 1


# ---------------------------------------------------------------------------
# apply_permutation
# ---------------------------------------------------------------------------


def test_apply_permutation_reorders_correctly():
    slate = ["a", "b", "c", "d"]
    perm = [2, 0, 3, 1]
    assert apply_permutation(slate, perm) == ["c", "a", "d", "b"]


def test_apply_permutation_is_bijection(bases_cfg):
    slate = [
        _card("c1", "B1"),
        _card("c2", "B2"),
        _card("c3", "B3"),
        _card("c4", "B4"),
    ]
    _, _, perm = check_slate(slate, 4, bases_cfg, seed=99)
    reordered = apply_permutation(slate, perm)
    assert len(reordered) == len(slate)
    assert {c["cardId"] for c in reordered} == {c["cardId"] for c in slate}


def test_apply_permutation_rejects_bad_permutation():
    with pytest.raises(ValueError):
        apply_permutation(["a", "b", "c"], [0, 1])
    with pytest.raises(ValueError):
        apply_permutation(["a", "b", "c"], [0, 0, 1])
