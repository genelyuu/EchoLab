"""Tests for echo_bench.metrics.replay (Task D-004).

Covers: exact-match True for identical chains; False + first_divergent for a
single changed hash; first-divergent follows the canonical key order; missing
keys count as divergence; seedBatchIds carried; determinism.
"""

from __future__ import annotations

from echo_bench.metrics.replay import CHAIN_KEY_ORDER, replay_consistency


def _chain(**overrides) -> dict:
    base = {
        "archiveHash": "arc-1",
        "poolHash": "pool-1",
        "slateHash": "slate-1",
        "traceHash": "trace-1",
        "outputHash": "out-1",
    }
    base.update(overrides)
    return base


def test_identical_chains_are_consistent():
    a = _chain()
    b = _chain()
    res = replay_consistency(a, b)
    assert res["consistent"] is True
    assert res["first_divergent"] is None


def test_single_divergence_flagged_with_first_key():
    a = _chain()
    b = _chain(slateHash="slate-2")
    res = replay_consistency(a, b)
    assert res["consistent"] is False
    assert res["first_divergent"] == "slateHash"


def test_first_divergent_follows_canonical_order():
    # Two keys differ; the canonically-earlier one must be reported.
    a = _chain()
    b = _chain(poolHash="pool-X", outputHash="out-X")
    res = replay_consistency(a, b)
    assert res["first_divergent"] == "poolHash"
    # poolHash precedes outputHash in the canonical order.
    assert CHAIN_KEY_ORDER.index("poolHash") < CHAIN_KEY_ORDER.index("outputHash")


def test_missing_key_counts_as_divergence():
    a = _chain()
    b = _chain()
    del b["traceHash"]
    res = replay_consistency(a, b)
    assert res["consistent"] is False
    assert res["first_divergent"] == "traceHash"


def test_exact_comparison_no_tolerance():
    # A trivially different hash string is a divergence (no fuzzy match).
    a = _chain(traceHash="abc")
    b = _chain(traceHash="abd")
    res = replay_consistency(a, b)
    assert res["consistent"] is False
    assert res["first_divergent"] == "traceHash"


def test_seed_batch_ids_carried():
    a = _chain(seedBatchId="sb-A")
    b = _chain(seedBatchId="sb-B")
    res = replay_consistency(a, b)
    assert res["seedBatchIds"] == {"a": "sb-A", "b": "sb-B"}
    # seedBatchId itself is metadata, not part of the hash comparison.
    assert res["consistent"] is True


def test_extra_non_canonical_keys_compared():
    a = _chain(customHash="x")
    b = _chain(customHash="y")
    res = replay_consistency(a, b)
    assert res["consistent"] is False
    assert res["first_divergent"] == "customHash"


def test_determinism():
    a = _chain(poolHash="p", outputHash="o")
    b = _chain(poolHash="P", outputHash="O")
    assert replay_consistency(a, b) == replay_consistency(a, b)
