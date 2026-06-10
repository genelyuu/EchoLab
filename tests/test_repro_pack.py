"""Tests for echo_bench.logging.repro_pack (Task F-001)."""

from __future__ import annotations

import pytest

from echo_bench.logging.repro_pack import ReproducibilityPack


def _valid_kwargs() -> dict:
    return {
        "configHash": "cfg0",
        "commitHash": "commit0",
        "archiveHash": "arc0",
        "poolHash": "pool0",
        "slateHash": "slate0",
        "traceHash": "trace0",
        "outputHash": "out0",
        "reportHash": "rep0",
        "seedBatchId": "seedbatch0",
    }


def test_round_trip_to_from_dict():
    pack = ReproducibilityPack(**_valid_kwargs())
    d = pack.to_dict()
    rebuilt = ReproducibilityPack.from_dict(d)
    assert rebuilt == pack
    assert rebuilt.to_dict() == d


def test_missing_field_raises():
    kwargs = _valid_kwargs()
    del kwargs["traceHash"]
    with pytest.raises(TypeError):
        ReproducibilityPack(**kwargs)  # dataclass requires the field


def test_from_dict_missing_field_raises():
    d = _valid_kwargs()
    del d["outputHash"]
    with pytest.raises(ValueError):
        ReproducibilityPack.from_dict(d)


def test_empty_field_raises():
    kwargs = _valid_kwargs()
    kwargs["configHash"] = ""
    with pytest.raises(ValueError):
        ReproducibilityPack(**kwargs)


def test_whitespace_field_raises():
    kwargs = _valid_kwargs()
    kwargs["poolHash"] = "   "
    with pytest.raises(ValueError):
        ReproducibilityPack(**kwargs)


def test_non_str_field_raises():
    kwargs = _valid_kwargs()
    kwargs["slateHash"] = 123
    with pytest.raises(ValueError):
        ReproducibilityPack(**kwargs)


def test_from_dict_rejects_extra_forbidden_key():
    d = _valid_kwargs()
    d["user_id"] = "u-42"
    with pytest.raises(ValueError):
        ReproducibilityPack.from_dict(d)


def test_pack_hash_deterministic():
    pack = ReproducibilityPack(**_valid_kwargs())
    assert pack.pack_hash() == pack.pack_hash()
    # Equal content -> equal hash.
    other = ReproducibilityPack.from_dict(pack.to_dict())
    assert pack.pack_hash() == other.pack_hash()


def test_pack_hash_changes_with_content():
    pack = ReproducibilityPack(**_valid_kwargs())
    kwargs = _valid_kwargs()
    kwargs["traceHash"] = "trace-different"
    other = ReproducibilityPack(**kwargs)
    assert pack.pack_hash() != other.pack_hash()
