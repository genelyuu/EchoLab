"""Tests for signature-saturation diagnostics (Task D-017, TRD V-004).

When the signature space is too large relative to the pooled sample —
(nearly) every signature unique — the pooled NMI sits at/near its
observed-achievable maximum under EVERY labeling and its absolute value is
not report-grade. D-017 detects this measurement-failure regime
automatically and reports, per channel:

- ``sample_count``             — pooled signature count,
- ``distinct_signature_count`` — distinct signature count,
- ``unique_signature_rate``    — distinct / sample (0.0 for empty),
- ``cardinality_sample_ratio`` — |distinct signatures| / sample_count
  (coincides with the rate by the TRD definitions; both keys promised),
- ``saturation_flag``          — rate >= SATURATION_UNIQUE_RATE_THRESHOLD
  (and a non-empty sample).

The exact key name ``saturation_flag`` is promised by
docs/12_CLAIM_LADDER.md Section 5 (Track L re-enable condition 2:
``saturation_flag = False``); ``saturation_flag = True`` forbids a headline
leakage claim for that channel/report. The flag is a DIAGNOSTIC, never a
claim.

Covers: (a) core stats fields + threshold boundary; (b) saturated input
(all-unique signatures) -> combined flag True; (c) non-saturated separable
input -> False; (d) per-channel independence (slate saturated while
selection is not); (e) determinism (insertion-order independence, repeat
calls); (f) degenerate inputs (empty / single-unique-signature) -> False;
(g) embedding: every null-corrected block and channel block carries the
``saturation`` sub-block, and the combined-channel bit-identity invariant
survives; (h) the ``precomputed_combined`` fast path of
``channel_separated_separability`` is bit-identical and validated.
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.leakage import (
    DEFAULT_NULL_PERMUTATIONS,
    SATURATION_UNIQUE_RATE_THRESHOLD,
    leakage_proxy_with_metadata,
    null_corrected_separability,
    signature_saturation_stats,
)
from echo_bench.metrics.separability import (
    CHANNEL_NAMES,
    SATURATION_METRIC_NAME,
    channel_separated_separability,
    signature_saturation_diagnostics,
)

# The exact field names of one saturation block (TRD D-017 names; the exact
# key ``saturation_flag`` is promised by docs/12_CLAIM_LADDER.md Section 5).
_SATURATION_FIELDS = (
    "sample_count",
    "distinct_signature_count",
    "unique_signature_rate",
    "cardinality_sample_ratio",
    "saturation_flag",
    "saturationThreshold",
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


def _saturated_traces() -> dict:
    """Every round has a globally unique slate AND selection -> every channel's
    signatures are all distinct -> saturated everywhere."""
    traces = {}
    i = 0
    for probe in ("PROBE_A", "PROBE_B"):
        recs = []
        for _ in range(5):
            slate = [f"u{i}", f"u{i + 1}", f"u{i + 2}", f"u{i + 3}"]
            recs.append(_record(f"u{i}", slate))
            i += 4
        traces[probe] = _trace(recs)
    return traces


def _non_saturated_separable_traces() -> dict:
    """Perfectly separable but with heavily REPEATED signatures: 12 pooled
    rounds, only 4 distinct combined signatures -> rate 1/3, far below the
    threshold."""
    s1 = ["a1", "a2", "a3", "a4"]
    s2 = ["b1", "b2", "b3", "b4"]
    a = _trace([_record("a1", s1), _record("b1", s2)] * 3)
    b = _trace([_record("a4", s1), _record("b4", s2)] * 3)
    return {"PROBE_A": a, "PROBE_B": b}


def _slate_saturated_selection_not_traces() -> dict:
    """Every round shows a DISTINCT slate (slate channel: all-unique ->
    saturated) while the within-slate selection rank is always 0 (selection
    channel: a single repeated signature -> not saturated, degenerate)."""
    traces = {}
    i = 0
    for probe in ("PROBE_A", "PROBE_B"):
        recs = []
        for _ in range(5):
            slate = sorted([f"u{i}", f"u{i + 1}", f"u{i + 2}", f"u{i + 3}"])
            recs.append(_record(slate[0], slate))
            i += 4
        traces[probe] = _trace(recs)
    return traces


# --- (a) core stats: fields, values, threshold boundary -------------------------


def test_core_stats_field_names_are_exact():
    out = signature_saturation_stats(["s1", "s2", "s1"])
    assert set(out) == set(_SATURATION_FIELDS)
    assert out["saturationThreshold"] == SATURATION_UNIQUE_RATE_THRESHOLD


def test_core_stats_counts_and_rates():
    out = signature_saturation_stats(["s1", "s2", "s1", "s3"])
    assert out["sample_count"] == 4
    assert out["distinct_signature_count"] == 3
    assert out["unique_signature_rate"] == pytest.approx(0.75)
    # By the TRD definitions the ratio coincides with the rate; both keys are
    # promised and must agree.
    assert out["cardinality_sample_ratio"] == out["unique_signature_rate"]
    assert out["saturation_flag"] is False


def test_core_stats_all_unique_is_saturated():
    out = signature_saturation_stats([f"s{i}" for i in range(10)])
    assert out["unique_signature_rate"] == 1.0
    assert out["saturation_flag"] is True


def test_core_stats_threshold_boundary():
    # Exactly AT the threshold: 19 distinct / 20 samples = 0.95 -> True
    # (the rule is >=, documented).
    at = signature_saturation_stats([f"s{i}" for i in range(19)] + ["s0"])
    assert at["unique_signature_rate"] == pytest.approx(
        SATURATION_UNIQUE_RATE_THRESHOLD
    )
    assert at["saturation_flag"] is True
    # Just BELOW: 18 distinct / 20 samples = 0.90 -> False.
    below = signature_saturation_stats(
        [f"s{i}" for i in range(18)] + ["s0", "s1"]
    )
    assert below["unique_signature_rate"] < SATURATION_UNIQUE_RATE_THRESHOLD
    assert below["saturation_flag"] is False


def test_core_stats_empty_sample_is_not_saturated():
    out = signature_saturation_stats([])
    assert out["sample_count"] == 0
    assert out["distinct_signature_count"] == 0
    assert out["unique_signature_rate"] == 0.0
    assert out["cardinality_sample_ratio"] == 0.0
    # Empty = nothing measured (degenerate elsewhere), NOT saturated.
    assert out["saturation_flag"] is False


def test_core_stats_single_repeated_signature_is_not_saturated():
    out = signature_saturation_stats(["same"] * 8)
    assert out["distinct_signature_count"] == 1
    assert out["unique_signature_rate"] == pytest.approx(1 / 8)
    assert out["saturation_flag"] is False


def test_core_stats_deterministic():
    sigs = ["a", "b", "a", "c", "c"]
    assert signature_saturation_stats(sigs) == signature_saturation_stats(
        list(sigs)
    )


# --- (b)/(c) diagnostics over traces: saturated vs not ---------------------------


def test_saturated_traces_flag_true_on_combined():
    out = signature_saturation_diagnostics(_saturated_traces())
    assert out["combined_saturation_flag"] is True
    assert out["channels"]["combined"]["saturation_flag"] is True
    # All-unique slates/selections saturate the other channels here too.
    assert out["slate_saturation_flag"] is True


def test_non_saturated_separable_traces_flag_false():
    out = signature_saturation_diagnostics(_non_saturated_separable_traces())
    for channel in CHANNEL_NAMES:
        assert out["channels"][channel]["saturation_flag"] is False
    assert out["combined_saturation_flag"] is False
    assert out["slate_saturation_flag"] is False
    assert out["selection_saturation_flag"] is False


def test_diagnostics_result_shape():
    out = signature_saturation_diagnostics(_non_saturated_separable_traces())
    assert out["metric"] == SATURATION_METRIC_NAME
    assert tuple(out["channelNames"]) == CHANNEL_NAMES
    assert out["saturationThreshold"] == SATURATION_UNIQUE_RATE_THRESHOLD
    for channel in CHANNEL_NAMES:
        block = out["channels"][channel]
        assert block["channel"] == channel
        for field in _SATURATION_FIELDS:
            assert field in block, f"channel {channel} missing {field}"


# --- (d) per-channel independence -------------------------------------------------


def test_slate_saturated_while_selection_not():
    out = signature_saturation_diagnostics(
        _slate_saturated_selection_not_traces()
    )
    assert out["slate_saturation_flag"] is True
    assert out["selection_saturation_flag"] is False
    # Combined inherits the all-unique slates -> also saturated.
    assert out["combined_saturation_flag"] is True


# --- (e) determinism ---------------------------------------------------------------


def test_diagnostics_deterministic_and_order_independent():
    traces = _saturated_traces()
    reordered = {k: traces[k] for k in reversed(list(traces))}
    out1 = signature_saturation_diagnostics(traces)
    out2 = signature_saturation_diagnostics(reordered)
    assert out1 == out2
    assert out1 == signature_saturation_diagnostics(_saturated_traces())


# --- (f) degenerate inputs ----------------------------------------------------------


def test_empty_traces_not_flagged():
    out = signature_saturation_diagnostics({})
    for channel in CHANNEL_NAMES:
        block = out["channels"][channel]
        assert block["sample_count"] == 0
        assert block["saturation_flag"] is False


def test_identical_single_signature_traces_not_flagged():
    slate = ["c1", "c2", "c3", "c4"]
    recs = [_record("c1", slate)] * 4
    out = signature_saturation_diagnostics(
        {"PROBE_A": _trace(recs), "PROBE_B": _trace(list(recs))}
    )
    for channel in CHANNEL_NAMES:
        block = out["channels"][channel]
        assert block["distinct_signature_count"] == 1
        assert block["saturation_flag"] is False


# --- (g) embedding in the null-corrected / channel blocks ---------------------------


def test_null_corrected_block_carries_saturation():
    out = null_corrected_separability(_saturated_traces())
    sat = out["saturation"]
    for field in _SATURATION_FIELDS:
        assert field in sat
    # The motivating saturation regime: observed NMI 1.0 yet zero excess —
    # the diagnostics must flag it.
    assert out["observed_nmi"] == 1.0
    assert sat["saturation_flag"] is True
    assert sat["unique_signature_rate"] == 1.0


def test_null_corrected_saturation_matches_standalone_diagnostics():
    for family in (
        _saturated_traces,
        _non_saturated_separable_traces,
        _slate_saturated_selection_not_traces,
    ):
        nc = null_corrected_separability(family())
        diag_combined = dict(
            signature_saturation_diagnostics(family())["channels"]["combined"]
        )
        assert diag_combined.pop("channel") == "combined"
        assert nc["saturation"] == diag_combined


def test_channel_blocks_carry_saturation_and_top_level_flags():
    out = channel_separated_separability(
        _slate_saturated_selection_not_traces()
    )
    for channel in CHANNEL_NAMES:
        sat = out["channels"][channel]["saturation"]
        for field in _SATURATION_FIELDS:
            assert field in sat
        flag_key = f"{channel}_saturation_flag"
        assert out[flag_key] == sat["saturation_flag"]
    assert out["saturationThreshold"] == SATURATION_UNIQUE_RATE_THRESHOLD
    assert out["slate_saturation_flag"] is True
    assert out["selection_saturation_flag"] is False


def test_combined_bit_identity_survives_saturation_addition():
    traces = _saturated_traces()
    out = channel_separated_separability(traces)
    combined = dict(out["channels"]["combined"])
    assert combined.pop("channel") == "combined"
    assert combined == null_corrected_separability(traces)


def test_degenerate_zero_dict_still_carries_saturation():
    out = null_corrected_separability({})
    assert out["degenerate"] is True
    assert out["saturation"]["saturation_flag"] is False
    assert out["saturation"]["sample_count"] == 0


# --- (h) precomputed combined fast path ----------------------------------------------


def test_precomputed_combined_is_bit_identical():
    traces = _non_saturated_separable_traces()
    md = leakage_proxy_with_metadata(traces)
    with_precomputed = channel_separated_separability(
        traces, precomputed_combined=md["nullCorrected"]
    )
    without = channel_separated_separability(traces)
    assert with_precomputed == without


def test_precomputed_combined_validates_metric_name():
    traces = _non_saturated_separable_traces()
    with pytest.raises(ValueError):
        channel_separated_separability(
            traces, precomputed_combined={"metric": "wrong", "n_permutations": 200}
        )


def test_precomputed_combined_validates_permutation_count():
    traces = _non_saturated_separable_traces()
    block = null_corrected_separability(traces, n_permutations=50)
    # Mismatched n_permutations must fail closed.
    with pytest.raises(ValueError):
        channel_separated_separability(
            traces,
            n_permutations=DEFAULT_NULL_PERMUTATIONS,
            precomputed_combined=block,
        )
    # Matching count is accepted and bit-identical.
    assert channel_separated_separability(
        traces, n_permutations=50, precomputed_combined=block
    ) == channel_separated_separability(traces, n_permutations=50)
