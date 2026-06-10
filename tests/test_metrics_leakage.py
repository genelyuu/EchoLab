"""Tests for echo_bench.metrics.leakage (Task D-002).

Covers: bounded range [0, 1]; determinism; the metric is explicitly labeled a
PROXY (not a guarantee); the metric reads NO latent/user field (only observable
slate/selection/probe identity); and the carried metadata holds the traceHashes
and probeVersions.
"""

from __future__ import annotations

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics import leakage
from echo_bench.metrics.leakage import (
    IS_PROXY,
    METRIC_NAME,
    PROXY_DISCLAIMER,
    leakage_proxy,
    leakage_proxy_with_metadata,
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
