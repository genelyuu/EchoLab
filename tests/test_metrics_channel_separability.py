"""Tests for channel-separated probe separability (Task D-016, TRD D-016).

The legacy D-002 selection signature concatenates ``selectedCardId`` + sorted
slate membership into ONE signature, so its NMI conflates two different
phenomena: "the policy showed different slates" (adaptive branching) and "the
probe chose differently" (probe choice separability). D-016 splits the
statistic into three channels over the same observable-fields contract:

- ``slate``     — sorted slate member ids only (no selectedCardId): branching
  of WHAT WAS SHOWN.
- ``selection`` — the rank of ``selectedCardId`` within its name-sorted slate:
  the choice CONDITIONAL on the slate (raw selectedCardId would leak slate
  identity, because card ids differ across slates).
- ``combined``  — the legacy signature, byte-identical to
  ``leakage_proxy``'s, so legacy values do not shift.

Each channel carries the full D-015 null-corrected statistics block.

Covers: (a) identical slates / different selections -> slate excess ~ 0,
selection excess > 0; (b) different slates / identical within-slate rank ->
slate excess > 0, selection excess ~ 0; (c) combined channel == legacy
``leakage_proxy`` / ``null_corrected_separability`` bit-identically;
(d) determinism (insertion-order independent, global RNG untouched);
(e) degenerate inputs fail closed per channel; plus the selection-rank
encoding rationale (no slate-identity leak into the selection channel),
metadata, and validation.
"""

from __future__ import annotations

import math
import random

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.leakage import (
    DEFAULT_NULL_PERMUTATIONS,
    PROXY_DISCLAIMER,
    leakage_proxy,
    null_corrected_separability,
)
from echo_bench.metrics.separability import (
    CHANNEL_NAMES,
    CHANNEL_SEPARABILITY_METRIC_NAME,
    channel_separated_separability,
)

# The channel names promised by docs/12_CLAIM_LADDER.md Section 5 (Track L
# re-enable condition 3: claims must be "channel-separated and channel-named").
_EXPECTED_CHANNELS = ("slate", "selection", "combined")

# D-015 statistics every channel block must carry in full.
_REQUIRED_CHANNEL_FIELDS = (
    "observed_nmi",
    "null_mean",
    "null_std",
    "excess_nmi",
    "excess_z",
    "n_permutations",
    "degenerate",
)

# Convenience top-level keys.
_TOP_LEVEL_EXCESS_KEYS = (
    "slate_excess_nmi",
    "selection_excess_nmi",
    "combined_excess_nmi",
)


def _record(selected: str, slate: list) -> dict:
    return {
        "candidatePoolHash": "poolhash-abc",
        "slate": slate,
        "selectedCardId": selected,
        "coordinateContribution": [0.1, 0.2, 0.3, 0.4],
        "complexityBand": "mid",
        "salienceScore": 0.5,
        "slotPermutation": [0, 1, 2, 3],
    }


def _trace(records: list) -> TraceState:
    t = TraceState()
    for r in records:
        t.append_round(r)
    return t


def _same_slates_different_selections() -> dict:
    """Both probes are shown the IDENTICAL alternating slate sequence, but
    PROBE_A always selects rank 0 and PROBE_B always selects rank 3 of the
    name-sorted slate -> slate channel carries nothing, selection channel
    separates perfectly."""
    s1 = ["a1", "a2", "a3", "a4"]
    s2 = ["b1", "b2", "b3", "b4"]
    a = _trace([_record("a1", s1), _record("b1", s2)] * 3)
    b = _trace([_record("a4", s1), _record("b4", s2)] * 3)
    return {"PROBE_A": a, "PROBE_B": b}


def _different_slates_same_rank_pattern() -> dict:
    """The probes see DISJOINT slates (slate channel separates perfectly), but
    both select the identical within-slate rank pattern (rank 0, rank 1,
    alternating) -> the selection channel carries nothing. Raw selectedCardId
    would WRONGLY separate here (the ids differ across probes only because the
    slates differ); the rank encoding must not."""
    sa1 = ["a1", "a2", "a3", "a4"]
    sa2 = ["b1", "b2", "b3", "b4"]
    sb1 = ["c1", "c2", "c3", "c4"]
    sb2 = ["d1", "d2", "d3", "d4"]
    a = _trace([_record("a1", sa1), _record("b2", sa2)] * 3)
    b = _trace([_record("c1", sb1), _record("d2", sb2)] * 3)
    return {"PROBE_A": a, "PROBE_B": b}


def _fully_identical_traces() -> dict:
    """Both probes behave identically over a single repeated slate/selection:
    every channel has a single unique signature -> degenerate per channel."""
    slate = ["c1", "c2", "c3", "c4"]
    recs = [_record("c1", slate)] * 4
    return {"PROBE_A": _trace(recs), "PROBE_B": _trace(list(recs))}


_ALL_FAMILIES = (
    _same_slates_different_selections,
    _different_slates_same_rank_pattern,
    _fully_identical_traces,
)


# --- result shape / metadata ---------------------------------------------------


def test_channel_names_are_exact():
    assert CHANNEL_NAMES == _EXPECTED_CHANNELS
    out = channel_separated_separability(_same_slates_different_selections())
    assert tuple(out["channelNames"]) == _EXPECTED_CHANNELS
    assert set(out["channels"]) == set(_EXPECTED_CHANNELS)
    assert out["metric"] == CHANNEL_SEPARABILITY_METRIC_NAME


def test_every_channel_carries_full_null_corrected_stats():
    out = channel_separated_separability(_same_slates_different_selections())
    for name, block in out["channels"].items():
        assert block["channel"] == name
        for field in _REQUIRED_CHANNEL_FIELDS:
            assert field in block, f"channel {name} missing field: {field}"
        assert block["isProxy"] is True
        assert block["disclaimer"] == PROXY_DISCLAIMER
        assert block["n_permutations"] == DEFAULT_NULL_PERMUTATIONS


def test_top_level_convenience_keys_match_channel_blocks():
    out = channel_separated_separability(_same_slates_different_selections())
    for key in _TOP_LEVEL_EXCESS_KEYS:
        assert key in out
    assert out["slate_excess_nmi"] == out["channels"]["slate"]["excess_nmi"]
    assert (
        out["selection_excess_nmi"]
        == out["channels"]["selection"]["excess_nmi"]
    )
    assert (
        out["combined_excess_nmi"] == out["channels"]["combined"]["excess_nmi"]
    )
    assert out["n_permutations"] == DEFAULT_NULL_PERMUTATIONS


def test_is_explicitly_a_proxy():
    out = channel_separated_separability(_same_slates_different_selections())
    assert out["isProxy"] is True
    assert out["disclaimer"] == PROXY_DISCLAIMER


def test_invalid_n_permutations_rejected():
    with pytest.raises(ValueError):
        channel_separated_separability(
            _same_slates_different_selections(), n_permutations=0
        )
    with pytest.raises(ValueError):
        channel_separated_separability(
            _same_slates_different_selections(), n_permutations=-5
        )


# --- (a) identical slates, different selections --------------------------------


def test_same_slates_different_selections_separates_only_selection():
    out = channel_separated_separability(_same_slates_different_selections())

    # Slate channel: the shown-slate distribution is identical across probes
    # -> nothing to separate (excess at most ~0; the null is >= 0).
    slate = out["channels"]["slate"]
    assert slate["observed_nmi"] == 0.0
    assert out["slate_excess_nmi"] <= 0.0 + 1e-12

    # Selection channel: rank 0 vs rank 3, every round -> perfectly separable
    # with repeated signatures, so the permutation null sits strictly below.
    selection = out["channels"]["selection"]
    assert selection["observed_nmi"] == 1.0
    assert out["selection_excess_nmi"] > 0.0

    # Combined inherits the selection difference.
    assert out["combined_excess_nmi"] > 0.0


# --- (b) different slates, identical within-slate rank -------------------------


def test_different_slates_same_rank_separates_only_slate():
    out = channel_separated_separability(_different_slates_same_rank_pattern())

    # Slate channel: disjoint slate families -> perfectly separable.
    slate = out["channels"]["slate"]
    assert slate["observed_nmi"] == 1.0
    assert out["slate_excess_nmi"] > 0.0

    # Selection channel: both probes produce the identical rank pattern
    # (0, 1, 0, 1, ...) -> no separation. This is the encoding rationale: raw
    # selectedCardId WOULD separate here purely because slate ids differ.
    selection = out["channels"]["selection"]
    assert selection["observed_nmi"] == 0.0
    assert out["selection_excess_nmi"] <= 0.0 + 1e-12

    # Combined (legacy) conflates the two and still separates -> this is the
    # exact ambiguity the channel split resolves.
    assert out["combined_excess_nmi"] > 0.0


# --- (c) combined channel == legacy ---------------------------------------------


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_combined_observed_nmi_equals_legacy_leakage_proxy(family):
    traces = family()
    out = channel_separated_separability(traces)
    assert out["channels"]["combined"]["observed_nmi"] == leakage_proxy(traces)


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_combined_block_is_bit_identical_to_null_corrected(family):
    """The combined channel IS the legacy D-015 statistic (same signature
    construction AND same data-derived permutation seed), so the whole block —
    including the sampled null — matches null_corrected_separability exactly,
    modulo the added "channel" annotation."""
    traces = family()
    out = channel_separated_separability(traces)
    combined = dict(out["channels"]["combined"])
    assert combined.pop("channel") == "combined"
    assert combined == null_corrected_separability(traces)


# --- (d) determinism ------------------------------------------------------------


def test_determinism_identical_calls():
    out1 = channel_separated_separability(_same_slates_different_selections())
    out2 = channel_separated_separability(_same_slates_different_selections())
    assert out1 == out2


def test_determinism_insertion_order_independent():
    traces = _same_slates_different_selections()
    reordered = {k: traces[k] for k in reversed(list(traces))}
    assert channel_separated_separability(
        traces
    ) == channel_separated_separability(reordered)


def test_determinism_does_not_touch_global_rng():
    random.seed(98765)
    before = random.random()
    random.seed(98765)
    channel_separated_separability(_same_slates_different_selections())
    after = random.random()
    assert before == after


def test_n_permutations_recorded_and_deterministic():
    out50 = channel_separated_separability(
        _same_slates_different_selections(), n_permutations=50
    )
    assert out50["n_permutations"] == 50
    for block in out50["channels"].values():
        assert block["n_permutations"] == 50
    assert out50 == channel_separated_separability(
        _same_slates_different_selections(), n_permutations=50
    )


# --- (e) degenerate inputs fail closed -------------------------------------------


@pytest.mark.parametrize("traces_factory", [dict, _fully_identical_traces])
def test_degenerate_inputs_fail_closed(traces_factory):
    out = channel_separated_separability(traces_factory())
    for name, block in out["channels"].items():
        assert block["degenerate"] is True, f"channel {name} not degenerate"
        assert block["observed_nmi"] == 0.0
        assert block["null_mean"] == 0.0
        assert block["null_std"] == 0.0
        assert block["excess_nmi"] == 0.0
        assert block["excess_z"] == 0.0
    for key in _TOP_LEVEL_EXCESS_KEYS:
        assert out[key] == 0.0


def test_single_probe_fails_closed():
    slate = ["c1", "c2", "c3", "c4"]
    out = channel_separated_separability(
        {"ONLY": _trace([_record("c1", slate), _record("c2", slate)])}
    )
    for block in out["channels"].values():
        assert block["degenerate"] is True
    for key in _TOP_LEVEL_EXCESS_KEYS:
        assert out[key] == 0.0


def test_channels_can_be_degenerate_independently():
    """Identical single slate everywhere (slate channel degenerate: one unique
    signature) while selections still differ (selection channel live)."""
    slate = ["c1", "c2", "c3", "c4"]
    a = _trace([_record("c1", slate)] * 4)
    b = _trace([_record("c4", slate)] * 4)
    out = channel_separated_separability({"PROBE_A": a, "PROBE_B": b})
    assert out["channels"]["slate"]["degenerate"] is True
    assert out["slate_excess_nmi"] == 0.0
    assert out["channels"]["selection"]["degenerate"] is False
    assert out["selection_excess_nmi"] > 0.0


def test_all_excess_values_finite():
    for family in _ALL_FAMILIES:
        out = channel_separated_separability(family())
        for key in _TOP_LEVEL_EXCESS_KEYS:
            assert math.isfinite(out[key])
        for block in out["channels"].values():
            assert math.isfinite(block["excess_nmi"])
            assert math.isfinite(block["excess_z"])


# --- selection-rank encoding edge cases ------------------------------------------


def test_selected_card_absent_from_slate_is_fail_closed_encoding():
    """Defensive: a selectedCardId not present in the slate must not crash and
    must map to a distinct fail-closed rank encoding (deterministically)."""
    slate = ["c1", "c2", "c3", "c4"]
    a = _trace([_record("zz-not-in-slate", slate)] * 3)
    b = _trace([_record("c1", slate)] * 3)
    out = channel_separated_separability({"PROBE_A": a, "PROBE_B": b})
    # The absent-selection encoding differs from any in-slate rank, so the
    # selection channel separates these two probes.
    assert out["channels"]["selection"]["observed_nmi"] == 1.0
    # Deterministic on repeat.
    assert out == channel_separated_separability(
        {"PROBE_A": a, "PROBE_B": b}
    )


def test_selection_rank_is_slate_order_invariant():
    """The rank is computed over the NAME-SORTED slate, so permuting the
    stored slate order does not change the selection signature (consistent
    with the combined channel's order-independent membership semantics)."""
    a1 = _trace([_record("c2", ["c1", "c2", "c3", "c4"])] * 3)
    a2 = _trace([_record("c2", ["c4", "c3", "c2", "c1"])] * 3)
    b = _trace([_record("c3", ["c1", "c2", "c3", "c4"])] * 3)
    out1 = channel_separated_separability({"PROBE_A": a1, "PROBE_B": b})
    out2 = channel_separated_separability({"PROBE_A": a2, "PROBE_B": b})
    assert (
        out1["channels"]["selection"] == out2["channels"]["selection"]
    )


# --- defensive record-shape edge cases (D-016 review follow-up) -------------------


class _DuckTrace:
    """Minimal duck-typed trace: the metric contract only requires
    ``rounds()``. Used to exercise defensive record shapes that TraceState's
    own schema validation would reject (e.g. a missing ``selectedCardId``)."""

    def __init__(self, records: list):
        self._records = records

    def rounds(self) -> list:
        return list(self._records)


def test_record_without_selected_card_id_key_is_handled():
    """A round record missing the ``selectedCardId`` key entirely must not
    crash: the selection channel maps it to the deterministic fail-closed
    absent-from-slate encoding, and the slate channel is unaffected.
    (TraceState itself rejects such records, so this guards the metric's own
    defensive ``record.get(...)`` path over duck-typed traces.)"""
    a = _DuckTrace([{"slate": ["a1", "a2", "a3", "a4"]}] * 3)
    b = _DuckTrace([{"slate": ["b1", "b2", "b3", "b4"]}] * 3)
    out = channel_separated_separability({"PROBE_A": a, "PROBE_B": b})
    # Slates are disjoint -> slate channel still separates.
    assert out["channels"]["slate"]["observed_nmi"] == 1.0
    # Both probes carry the SAME absent-selection encoding -> the selection
    # channel has a single unique signature -> degenerate (fail closed).
    assert out["channels"]["selection"]["degenerate"] is True
    # Deterministic on repeat.
    assert out == channel_separated_separability({"PROBE_A": a, "PROBE_B": b})


def test_empty_slate_is_handled():
    """An empty slate must not crash any channel: the slate signature is the
    empty-membership signature; the selection deterministically maps to the
    absent-from-slate encoding."""
    a = _trace([_record("c1", [])] * 3)
    b = _trace([_record("c1", ["c1", "c2", "c3", "c4"])] * 3)
    out = channel_separated_separability({"PROBE_A": a, "PROBE_B": b})
    # Empty vs non-empty membership separates the slate channel.
    assert out["channels"]["slate"]["observed_nmi"] == 1.0
    # absent-encoding (empty slate) vs rank 0 separates the selection channel.
    assert out["channels"]["selection"]["observed_nmi"] == 1.0
    assert out == channel_separated_separability({"PROBE_A": a, "PROBE_B": b})


def test_single_round_per_probe_traces():
    """One round per probe: nothing crashes, the statistic is computed over
    2 pooled rounds, and the all-unique micro-sample is correctly flagged as
    saturated (fail closed: its absolute NMI is not report-grade)."""
    a = _trace([_record("a1", ["a1", "a2", "a3", "a4"])])
    b = _trace([_record("b1", ["b1", "b2", "b3", "b4"])])
    out = channel_separated_separability({"PROBE_A": a, "PROBE_B": b})
    combined = out["channels"]["combined"]
    assert combined["degenerate"] is False
    assert combined["observed_nmi"] == 1.0
    assert combined["saturation"]["sample_count"] == 2
    assert combined["saturation"]["saturation_flag"] is True
    assert out == channel_separated_separability({"PROBE_A": a, "PROBE_B": b})


def test_duplicate_id_slate_has_deterministic_first_occurrence_rank():
    """A slate containing duplicate card ids yields the FIRST-occurrence rank
    within the name-sorted member list — deterministically, and invariant to
    the stored order of the duplicates."""
    a1 = _trace([_record("c1", ["c1", "c1", "c2", "c3"])] * 3)
    a2 = _trace([_record("c1", ["c3", "c1", "c2", "c1"])] * 3)
    b = _trace([_record("c2", ["c1", "c1", "c2", "c3"])] * 3)
    out1 = channel_separated_separability({"PROBE_A": a1, "PROBE_B": b})
    out2 = channel_separated_separability({"PROBE_A": a2, "PROBE_B": b})
    # Same members (duplicates preserved), same selection -> identical blocks.
    assert out1 == out2
    # c1 (rank 0, first occurrence) vs c2 (rank 2) separates the selection
    # channel; duplicates do not make the encoding ambiguous.
    assert out1["channels"]["selection"]["observed_nmi"] == 1.0
    # Membership is identical across probes -> slate channel degenerate.
    assert out1["channels"]["slate"]["degenerate"] is True
