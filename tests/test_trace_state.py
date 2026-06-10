"""Tests for echo_bench.env.trace_state (Task B-001)."""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import (
    ALLOWED_FIELDS,
    FORBIDDEN_FIELDS,
    TraceState,
)
from echo_bench.utils.hash import canonical_hash


def _valid_record(selected: str = "c1") -> dict:
    """A minimal valid caller-supplied round record (no ``roundHash``)."""
    return {
        "candidatePoolHash": "poolhash-abc",
        "slate": ["c1", "c2", "c3", "c4"],
        "selectedCardId": selected,
        "coordinateContribution": [0.1, 0.2, 0.3, 0.4],
        "complexityBand": "mid",
        "salienceScore": 0.5,
        "slotPermutation": [0, 1, 2, 3],
    }


def test_append_valid_record_increments_len():
    ts = TraceState()
    assert len(ts) == 0
    ts.append_round(_valid_record())
    assert len(ts) == 1
    ts.append_round(_valid_record("c2"))
    assert len(ts) == 2


def test_appended_record_has_round_hash():
    ts = TraceState()
    ts.append_round(_valid_record())
    rec = ts.rounds()[0]
    assert "roundHash" in rec
    assert set(rec.keys()) == set(ALLOWED_FIELDS)


@pytest.mark.parametrize("field", sorted(FORBIDDEN_FIELDS))
def test_forbidden_field_rejected(field):
    ts = TraceState()
    rec = _valid_record()
    rec[field] = "leak"
    with pytest.raises(ValueError):
        ts.append_round(rec)
    assert len(ts) == 0


def test_user_id_specifically_rejected():
    ts = TraceState()
    rec = _valid_record()
    rec["user_id"] = 42
    with pytest.raises(ValueError):
        ts.append_round(rec)


def test_unexpected_free_text_key_rejected():
    ts = TraceState()
    rec = _valid_record()
    rec["note"] = "this is free text describing the user"
    with pytest.raises(ValueError):
        ts.append_round(rec)
    assert len(ts) == 0


def test_caller_supplied_round_hash_rejected():
    # The trace computes roundHash itself; supplying it is an unexpected key.
    ts = TraceState()
    rec = _valid_record()
    rec["roundHash"] = "forged"
    with pytest.raises(ValueError):
        ts.append_round(rec)


def test_missing_required_key_rejected():
    ts = TraceState()
    rec = _valid_record()
    del rec["slate"]
    with pytest.raises(ValueError):
        ts.append_round(rec)


def test_empty_trace_hash_is_stable_and_canonical():
    a = TraceState()
    b = TraceState()
    assert a.trace_hash() == b.trace_hash()
    assert a.trace_hash() == canonical_hash([])


def test_same_ordered_records_identical_trace_hash():
    recs = [_valid_record("c1"), _valid_record("c2"), _valid_record("c3")]
    a = TraceState()
    b = TraceState()
    for r in recs:
        a.append_round(dict(r))
    for r in recs:
        b.append_round(dict(r))
    assert a.trace_hash() == b.trace_hash()


def test_different_order_different_trace_hash():
    r1 = _valid_record("c1")
    r2 = _valid_record("c2")
    a = TraceState()
    b = TraceState()
    a.append_round(dict(r1))
    a.append_round(dict(r2))
    b.append_round(dict(r2))
    b.append_round(dict(r1))
    assert a.trace_hash() != b.trace_hash()


def test_round_hash_chains_previous_trace():
    # Two traces that differ only in the first round must diverge in the
    # second round's roundHash as well (hash chaining).
    a = TraceState()
    b = TraceState()
    a.append_round(_valid_record("c1"))
    b.append_round(_valid_record("cX"))  # different earlier round

    common_second = _valid_record("c2")
    a.append_round(dict(common_second))
    b.append_round(dict(common_second))

    a_second_hash = a.rounds()[1]["roundHash"]
    b_second_hash = b.rounds()[1]["roundHash"]
    assert a_second_hash != b_second_hash


def test_rounds_view_is_read_only_copy():
    ts = TraceState()
    ts.append_round(_valid_record())
    view = ts.rounds()
    view[0]["selectedCardId"] = "mutated"
    # Internal state must be unaffected by mutating the returned view.
    assert ts.rounds()[0]["selectedCardId"] == "c1"
