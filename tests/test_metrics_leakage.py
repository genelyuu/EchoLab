"""Tests for echo_bench.metrics.leakage (Tasks D-002, D-011, G-020).

Covers: bounded range [0, 1]; determinism; the metric is explicitly labeled a
PROXY (not a guarantee); the metric reads NO latent/user field (only observable
slate/selection/probe identity); and the carried metadata holds the traceHashes
and probeVersions. D-011 (TRD alias D-010) adds the comparison-ready metrics
``leakage_delta_vs_random`` / ``utility_per_leakage`` and the documented
``LEAKAGE_RATIO_FLOOR`` constant. G-020 adds the primary terminology label
``probe_separability_proxy`` (``PRIMARY_METRIC_NAME``) carried ADDITIVELY in
the metadata (``primaryMetric`` / ``legacyAlias``) — machine keys/values stay
byte-identical (the ``metric``/``value`` keys are unchanged, D-012 precedent).
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics import leakage
from echo_bench.metrics.leakage import (
    IS_PROXY,
    LEAKAGE_RATIO_FLOOR,
    METRIC_NAME,
    PRIMARY_METRIC_NAME,
    PROXY_DISCLAIMER,
    leakage_delta_vs_random,
    leakage_proxy,
    leakage_proxy_with_metadata,
    utility_per_leakage,
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


def _identical_traces() -> dict:
    """Two probes whose selections are identical -> no separability."""
    slate = ["c1", "c2", "c3", "c4"]
    recs = [_record("c1", slate), _record("c2", slate)]
    return {"PROBE_A": _trace(recs), "PROBE_B": _trace(list(recs))}


def _separable_traces() -> dict:
    """Two probes whose selections perfectly separate by probe."""
    slate = ["c1", "c2", "c3", "c4"]
    a = _trace([_record("c1", slate), _record("c1", slate)])
    b = _trace([_record("c4", slate), _record("c4", slate)])
    return {"PROBE_A": a, "PROBE_B": b}


def test_bounded_range():
    for traces in (_identical_traces(), _separable_traces()):
        v = leakage_proxy(traces)
        assert 0.0 <= v <= 1.0


def test_identical_selection_distribution_is_zero():
    # Same selection signature distribution across probes -> nothing to separate.
    assert leakage_proxy(_identical_traces()) == 0.0


def test_perfectly_separable_is_high():
    # Selections perfectly determined by probe -> NMI == 1.0.
    assert leakage_proxy(_separable_traces()) == 1.0


def test_fewer_than_two_probes_is_zero():
    slate = ["c1", "c2", "c3", "c4"]
    assert leakage_proxy({"ONLY": _trace([_record("c1", slate)])}) == 0.0
    assert leakage_proxy({}) == 0.0


def test_determinism():
    t1 = _separable_traces()
    t2 = _separable_traces()
    assert leakage_proxy(t1) == leakage_proxy(t2)
    # Insertion-order independence.
    reordered = {k: t1[k] for k in reversed(list(t1))}
    assert leakage_proxy(reordered) == leakage_proxy(t1)


def test_is_explicitly_a_proxy_not_a_guarantee():
    assert IS_PROXY is True
    # Module + metadata must state PROXY and disclaim privacy/legal guarantees.
    assert "PROXY" in PROXY_DISCLAIMER
    text = (leakage.__doc__ or "") + PROXY_DISCLAIMER
    lowered = text.lower()
    assert "not a privacy" in lowered or "not a real privacy" in lowered
    for forbidden_framing in ("guarantee", "legal", "compliance"):
        # The disclaimer explicitly NEGATES these framings (it mentions them only
        # to deny them); ensure the word "not" governs the disclaimer.
        assert "not" in PROXY_DISCLAIMER.lower()
    md = leakage_proxy_with_metadata(_separable_traces())
    assert md["isProxy"] is True
    assert md["disclaimer"] == PROXY_DISCLAIMER
    assert md["metric"] == METRIC_NAME


def test_primary_metric_name_constant_g020():
    """G-020: the primary terminology label is probe_separability_proxy;
    METRIC_NAME stays the legacy machine name (unchanged)."""
    assert PRIMARY_METRIC_NAME == "probe_separability_proxy"
    assert METRIC_NAME == "leakage_proxy"
    assert "PRIMARY_METRIC_NAME" in leakage.__all__


def test_metadata_carries_primary_metric_and_legacy_alias_g020():
    """G-020: leakage_proxy_with_metadata ADDITIVELY carries the primary
    label + legacy alias; existing machine keys are byte-identical."""
    md = leakage_proxy_with_metadata(_separable_traces())
    # Additive G-020 label layer (D-012 legacyAlias precedent).
    assert md["primaryMetric"] == PRIMARY_METRIC_NAME == "probe_separability_proxy"
    assert md["legacyAlias"] == METRIC_NAME == "leakage_proxy"
    # Machine keys UNCHANGED: "metric" stays the legacy machine name and the
    # value key keeps its name and statistic.
    assert md["metric"] == METRIC_NAME
    assert md["value"] == md["nullCorrected"]["observed_nmi"]
    assert md["isProxy"] is True
    assert md["disclaimer"] == PROXY_DISCLAIMER


def test_metadata_carries_trace_hashes_and_probe_versions():
    traces = _separable_traces()
    md = leakage_proxy_with_metadata(
        traces, probe_versions={"PROBE_A": "p1", "PROBE_B": "p1"}
    )
    assert set(md["traceHashes"]) == set(traces)
    for name, h in md["traceHashes"].items():
        assert h == traces[name].trace_hash()
    assert md["probeVersions"] == {"PROBE_A": "p1", "PROBE_B": "p1"}
    assert 0.0 <= md["value"] <= 1.0


def test_reads_no_latent_or_user_field():
    """The metric must read ONLY observable slate/selection/probe identity.

    We hand it traces whose round records are wrapped so that reading ANY field
    other than the observable ``slate``/``selectedCardId`` records a violation.
    """
    accessed: set = set()
    allowed = {"slate", "selectedCardId"}

    class _SpyRecord(dict):
        def get(self, key, default=None):  # type: ignore[override]
            accessed.add(key)
            return super().get(key, default)

    class _SpyTrace:
        def __init__(self, records):
            self._records = [_SpyRecord(r) for r in records]

        def rounds(self):
            return self._records

        def trace_hash(self):
            return "spy"

    slate = ["c1", "c2", "c3", "c4"]
    traces = {
        "PROBE_A": _SpyTrace([_record("c1", slate)]),
        "PROBE_B": _SpyTrace([_record("c4", slate)]),
    }
    leakage_proxy(traces)
    # Every field the metric touched on the round must be an observable one.
    forbidden = accessed - allowed
    assert not forbidden, f"leakage_proxy read non-observable fields: {forbidden}"


# --- D-011 (TRD alias D-010): comparison-ready leakage delta / ratio ---------


def test_delta_hand_computed():
    # Known E3 n=10 numbers: TRACE_GREEDY 0.737 vs RANDOM 0.947 -> -0.21.
    assert leakage_delta_vs_random(0.737, 0.947) == pytest.approx(-0.21)
    # Sign flips when the policy leaks MORE than the RANDOM reference.
    assert leakage_delta_vs_random(0.947, 0.737) == pytest.approx(0.21)


def test_delta_random_self_is_exact_zero():
    # The RANDOM reference compared with itself is 0.0 by definition (exact).
    assert leakage_delta_vs_random(0.947, 0.947) == 0.0
    assert leakage_delta_vs_random(0.0, 0.0) == 0.0
    assert leakage_delta_vs_random(1.0, 1.0) == 0.0


def test_delta_bounds_and_input_clamping():
    # Extremes of the [0,1] x [0,1] input space give the [-1, 1] bounds.
    assert leakage_delta_vs_random(0.0, 1.0) == -1.0
    assert leakage_delta_vs_random(1.0, 0.0) == 1.0
    # Out-of-range inputs are clamped into [0,1] before differencing.
    assert leakage_delta_vs_random(-0.5, 2.0) == -1.0
    assert leakage_delta_vs_random(1.5, -1.0) == 1.0


def test_ratio_hand_computed():
    # mean coordinate_coverage 0.8 over leakage 0.737 (above the floor).
    assert utility_per_leakage(0.8, 0.737) == pytest.approx(0.8 / 0.737)
    assert utility_per_leakage(0.5, 1.0) == pytest.approx(0.5)


def test_ratio_floor_constant_is_documented_value():
    assert LEAKAGE_RATIO_FLOOR == 0.05


def test_ratio_floor_behavior_below_floor():
    # Leakage below the floor divides by the floor (no near-zero explosion)...
    assert utility_per_leakage(0.8, 0.01) == pytest.approx(0.8 / LEAKAGE_RATIO_FLOOR)
    assert utility_per_leakage(0.8, 0.0) == pytest.approx(0.8 / LEAKAGE_RATIO_FLOOR)
    # ... and AT the floor the two branches agree (continuity at the floor).
    assert utility_per_leakage(0.8, LEAKAGE_RATIO_FLOOR) == pytest.approx(
        0.8 / LEAKAGE_RATIO_FLOOR
    )


def test_ratio_bounds():
    # Max ratio = 1.0 / floor (full utility, zero leakage floored).
    assert utility_per_leakage(1.0, 0.0) == pytest.approx(1.0 / LEAKAGE_RATIO_FLOOR)
    # Zero utility -> 0.0 regardless of leakage.
    assert utility_per_leakage(0.0, 0.5) == 0.0
    # Out-of-range inputs are clamped into [0,1] before the ratio.
    assert utility_per_leakage(1.5, 2.0) == pytest.approx(1.0)
    assert utility_per_leakage(-0.5, 0.5) == 0.0


def test_ratio_rejects_non_positive_floor():
    with pytest.raises(ValueError):
        utility_per_leakage(0.5, 0.5, floor=0.0)
    with pytest.raises(ValueError):
        utility_per_leakage(0.5, 0.5, floor=-0.05)


def test_delta_and_ratio_docstrings_stay_relative_and_proxy_scoped():
    # The docstrings must carry the RELATIVE-claim framing and reference the
    # proxy disclaimer; they must never assert an absolute privacy property.
    for fn in (leakage_delta_vs_random, utility_per_leakage):
        doc = (fn.__doc__ or "").lower()
        assert "proxy" in doc
        assert "relative" in doc or "trade-off" in doc
        assert "privacy guarantee" not in doc.replace("not a privacy guarantee", "")
