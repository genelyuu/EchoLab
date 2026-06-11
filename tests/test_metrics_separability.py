"""Tests for null-corrected probe separability (Task D-015, TRD D-015).

The naive pooled-NMI ``leakage_proxy`` saturates near 1.0 under
trace-conditioned branching (the signature space explodes, so the observed NMI
is high even under the permutation null). D-015 adds
:func:`null_corrected_separability`, which compares the observed NMI against a
deterministic permutation null over the same signature multiset and reports
``observed_nmi`` / ``null_mean`` / ``null_std`` / ``excess_nmi`` / ``excess_z``.

Interpretation convention: never "NMI is high" — only "there is / is not
information in excess of the permutation null".

Covers: (a) perfectly separable probes -> observed high, excess > 0;
(b) identical-behavior probes -> observed == 0, excess <= 0 (no excess);
(c) determinism (identical inputs -> bit-identical dicts, insertion-order
independent, no wall-clock/global RNG); (d) the motivating saturation failure:
all-unique signatures -> observed == 1.0 but excess == 0.0; (e) degenerate
inputs (one probe / empty) fail closed to the documented zero dict;
(f) the bounded ``null_std < eps -> excess_z = 0.0`` convention (no inf);
plus proxy metadata and the ``leakage_proxy_with_metadata`` extension.
"""

from __future__ import annotations

import math
import random

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.leakage import (
    NULL_STD_EPS,
    PROXY_DISCLAIMER,
    SEPARABILITY_METRIC_NAME,
    leakage_proxy,
    leakage_proxy_with_metadata,
    null_corrected_separability,
)

# The exact field names promised by docs/12_CLAIM_LADDER.md Section 5 (Track L
# re-enable condition 1) and the TRD D-015 spec. These must not drift.
_REQUIRED_FIELDS = (
    "observed_nmi",
    "null_mean",
    "null_std",
    "excess_nmi",
    "excess_z",
    "n_permutations",
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


def _separable_traces() -> dict:
    """Two probes whose selections perfectly separate, with enough rounds that
    the permutation null is strictly below 1.0 (repeated signatures)."""
    slate = ["c1", "c2", "c3", "c4"]
    a = _trace([_record("c1", slate) for _ in range(6)])
    b = _trace([_record("c4", slate) for _ in range(6)])
    return {"PROBE_A": a, "PROBE_B": b}


def _identical_traces() -> dict:
    """Two probes whose selection behaviour is identical -> nothing separable."""
    slate = ["c1", "c2", "c3", "c4"]
    recs = [_record("c1", slate), _record("c2", slate)] * 3
    return {"PROBE_A": _trace(recs), "PROBE_B": _trace(list(recs))}


def _saturated_traces() -> dict:
    """The motivating failure: every round has a UNIQUE signature (per-round
    distinct slates), so pooled NMI == 1.0 for ANY labeling — including every
    permutation-null labeling. Observed is maximal yet carries zero excess."""
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


# --- core result shape -------------------------------------------------------


def test_required_field_names_are_exact():
    out = null_corrected_separability(_separable_traces())
    for field in _REQUIRED_FIELDS:
        assert field in out, f"missing required field: {field}"
    assert out["metric"] == SEPARABILITY_METRIC_NAME
    assert out["n_permutations"] == 200


def test_is_explicitly_a_proxy():
    out = null_corrected_separability(_separable_traces())
    assert out["isProxy"] is True
    assert out["disclaimer"] == PROXY_DISCLAIMER


def test_observed_matches_leakage_proxy():
    # observed_nmi is the SAME statistic as the legacy pooled NMI; the
    # correction only adds the null reference (absolute NMI alone is no longer
    # report-grade, but the value itself must not drift).
    traces = _separable_traces()
    out = null_corrected_separability(traces)
    assert out["observed_nmi"] == leakage_proxy(traces)


# --- (a) perfectly separable -> excess > 0 -----------------------------------


def test_separable_probes_have_positive_excess():
    out = null_corrected_separability(_separable_traces())
    assert out["observed_nmi"] == 1.0
    # Random label permutations over a repeated-signature multiset cannot
    # systematically reach 1.0 -> the null mean sits strictly below observed.
    assert 0.0 <= out["null_mean"] < 1.0
    assert out["excess_nmi"] > 0.0
    assert out["excess_nmi"] == pytest.approx(
        out["observed_nmi"] - out["null_mean"]
    )
    if out["null_std"] >= NULL_STD_EPS:
        assert out["excess_z"] == pytest.approx(
            (out["observed_nmi"] - out["null_mean"]) / out["null_std"]
        )
        assert out["excess_z"] > 0.0


# --- (b) identical behaviour -> no excess -------------------------------------


def test_identical_probes_have_no_excess():
    out = null_corrected_separability(_identical_traces())
    assert out["observed_nmi"] == 0.0
    # The null can only be >= 0, so the excess is at most ~0 (finite-sample
    # permutations may give small positive null NMI -> excess slightly <= 0).
    assert out["excess_nmi"] <= 0.0 + 1e-12
    assert out["excess_nmi"] == pytest.approx(
        out["observed_nmi"] - out["null_mean"]
    )
    assert out["excess_z"] <= 0.0 + 1e-12


# --- (c) determinism ----------------------------------------------------------


def test_determinism_identical_calls():
    out1 = null_corrected_separability(_separable_traces())
    out2 = null_corrected_separability(_separable_traces())
    assert out1 == out2


def test_determinism_insertion_order_independent():
    traces = _separable_traces()
    reordered = {k: traces[k] for k in reversed(list(traces))}
    assert null_corrected_separability(traces) == null_corrected_separability(
        reordered
    )


def test_determinism_does_not_touch_global_rng():
    random.seed(12345)
    before = random.random()
    random.seed(12345)
    null_corrected_separability(_separable_traces())
    after = random.random()
    assert before == after


def test_n_permutations_changes_seed_and_is_recorded():
    out50 = null_corrected_separability(_separable_traces(), n_permutations=50)
    assert out50["n_permutations"] == 50
    # Same input + same n_permutations -> identical again.
    assert out50 == null_corrected_separability(
        _separable_traces(), n_permutations=50
    )


def test_invalid_n_permutations_rejected():
    with pytest.raises(ValueError):
        null_corrected_separability(_separable_traces(), n_permutations=0)
    with pytest.raises(ValueError):
        null_corrected_separability(_separable_traces(), n_permutations=-3)


# --- (d) saturation: observed high, excess ~ 0 --------------------------------


def test_saturated_signatures_observed_high_but_zero_excess():
    """The motivating failure mode: pooled NMI saturates at 1.0 because every
    signature is unique — but the permutation null is ALSO 1.0, so the
    null-corrected excess is exactly 0 and excess_z follows the bounded
    null_std~0 convention."""
    out = null_corrected_separability(_saturated_traces())
    assert out["observed_nmi"] == 1.0
    assert out["null_mean"] == pytest.approx(1.0)
    assert out["null_std"] == pytest.approx(0.0, abs=NULL_STD_EPS)
    assert out["excess_nmi"] == pytest.approx(0.0, abs=1e-12)
    assert out["excess_z"] == 0.0


# --- (e) degenerate inputs fail closed ----------------------------------------


@pytest.mark.parametrize(
    "traces",
    [
        {},
        {"ONLY": None},  # placeholder; replaced below
    ],
)
def test_degenerate_inputs_fail_closed(traces):
    if traces and traces.get("ONLY") is None:
        slate = ["c1", "c2", "c3", "c4"]
        traces = {"ONLY": _trace([_record("c1", slate)])}
    out = null_corrected_separability(traces)
    assert out["degenerate"] is True
    assert out["observed_nmi"] == 0.0
    assert out["null_mean"] == 0.0
    assert out["null_std"] == 0.0
    assert out["excess_nmi"] == 0.0
    assert out["excess_z"] == 0.0


def test_empty_rounds_fail_closed():
    out = null_corrected_separability(
        {"PROBE_A": _trace([]), "PROBE_B": _trace([])}
    )
    assert out["degenerate"] is True
    assert out["excess_nmi"] == 0.0
    assert out["excess_z"] == 0.0


# --- (f) null_std ~ 0 convention -----------------------------------------------


def test_null_std_near_zero_gives_bounded_excess_z():
    # Saturated case has null_std == 0; the documented convention is
    # excess_z = 0.0 (never +/-inf or NaN).
    out = null_corrected_separability(_saturated_traces())
    assert out["excess_z"] == 0.0
    assert math.isfinite(out["excess_z"])
    # Convention is self-describing in the output.
    assert out["nullStdEps"] == NULL_STD_EPS


def test_excess_z_always_finite():
    for traces in (
        _separable_traces(),
        _identical_traces(),
        _saturated_traces(),
        {},
    ):
        out = null_corrected_separability(traces)
        assert math.isfinite(out["excess_z"])
        assert math.isfinite(out["excess_nmi"])


# --- leakage_proxy_with_metadata carries the null-corrected block --------------


def test_metadata_carries_null_corrected_block():
    traces = _separable_traces()
    md = leakage_proxy_with_metadata(traces)
    # Legacy keys unchanged.
    for key in (
        "metric",
        "value",
        "isProxy",
        "disclaimer",
        "traceHashes",
        "probeVersions",
    ):
        assert key in md
    nc = md["nullCorrected"]
    for field in _REQUIRED_FIELDS:
        assert field in nc, f"nullCorrected missing field: {field}"
    # The observed NMI in the null-corrected block IS the legacy value.
    assert nc["observed_nmi"] == md["value"]


def test_metadata_null_corrected_is_deterministic():
    md1 = leakage_proxy_with_metadata(_separable_traces())
    md2 = leakage_proxy_with_metadata(_separable_traces())
    assert md1 == md2
