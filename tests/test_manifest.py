"""Tests for echo_bench.logging.manifest (Task F-003)."""

from __future__ import annotations

import pytest

from echo_bench.logging.manifest import VERSION_FIELDS, RunManifest
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.utils.hash import canonical_hash


def _valid_pack() -> ReproducibilityPack:
    return ReproducibilityPack(
        configHash="cfg0",
        commitHash="commit0",
        archiveHash="arc0",
        poolHash="pool0",
        slateHash="slate0",
        traceHash="trace0",
        outputHash="out0",
        reportHash="rep0",
        seedBatchId="seedbatch0",
    )


def _valid_kwargs() -> dict:
    return {
        "pack": _valid_pack(),
        "rendererVersion": "rend1",
        "policyVersion": "pol1",
        "probeVersion": "prb1",
        "metricVersion": "met1",
        "reportVersion": "rpt1",
        "seedBatchId": "seedbatch0",
    }


def test_embeds_pack_and_all_five_versions():
    m = RunManifest(**_valid_kwargs())
    assert isinstance(m.pack, ReproducibilityPack)
    for name in VERSION_FIELDS:
        assert getattr(m, name)
    assert len(VERSION_FIELDS) == 5
    # Full 9-field pack hash chain is reachable through the manifest.
    assert set(m.pack.to_dict().keys()) == {
        "configHash",
        "commitHash",
        "archiveHash",
        "poolHash",
        "slateHash",
        "traceHash",
        "outputHash",
        "reportHash",
        "seedBatchId",
    }


def test_round_trip_to_from_dict():
    m = RunManifest(**_valid_kwargs())
    d = m.to_dict()
    rebuilt = RunManifest.from_dict(d)
    assert rebuilt == m
    assert rebuilt.to_dict() == d


def test_manifest_hash_deterministic():
    m1 = RunManifest(**_valid_kwargs())
    m2 = RunManifest(**_valid_kwargs())
    assert m1.manifest_hash() == m2.manifest_hash()
    assert m1.manifest_hash() == canonical_hash(m1.to_dict())


def test_manifest_hash_changes_with_version():
    m1 = RunManifest(**_valid_kwargs())
    kwargs = _valid_kwargs()
    kwargs["metricVersion"] = "met2"
    m2 = RunManifest(**kwargs)
    assert m1.manifest_hash() != m2.manifest_hash()


@pytest.mark.parametrize("name", VERSION_FIELDS)
def test_missing_version_raises(name):
    kwargs = _valid_kwargs()
    kwargs[name] = ""
    with pytest.raises(ValueError):
        RunManifest(**kwargs)


@pytest.mark.parametrize("name", VERSION_FIELDS)
def test_whitespace_version_raises(name):
    kwargs = _valid_kwargs()
    kwargs[name] = "   "
    with pytest.raises(ValueError):
        RunManifest(**kwargs)


def test_missing_version_from_dict_raises():
    d = RunManifest(**_valid_kwargs()).to_dict()
    del d["policyVersion"]
    with pytest.raises(ValueError):
        RunManifest.from_dict(d)


def test_missing_pack_hash_raises():
    # Build a manifest dict whose embedded pack drops a hash field.
    d = RunManifest(**_valid_kwargs()).to_dict()
    del d["pack"]["traceHash"]
    with pytest.raises(ValueError):
        RunManifest.from_dict(d)


def test_empty_seed_batch_id_raises():
    kwargs = _valid_kwargs()
    kwargs["seedBatchId"] = ""
    with pytest.raises(ValueError):
        RunManifest(**kwargs)


def test_non_pack_raises():
    kwargs = _valid_kwargs()
    kwargs["pack"] = {"configHash": "x"}
    with pytest.raises(ValueError):
        RunManifest(**kwargs)


def test_forbidden_field_rejected_from_dict():
    d = RunManifest(**_valid_kwargs()).to_dict()
    d["userId"] = "u123"
    with pytest.raises(ValueError):
        RunManifest.from_dict(d)


def test_forbidden_field_in_pack_rejected():
    d = RunManifest(**_valid_kwargs()).to_dict()
    d["pack"]["persona"] = "anxious"
    with pytest.raises(ValueError):
        RunManifest.from_dict(d)


def test_from_dict_rejects_non_dict():
    with pytest.raises(ValueError):
        RunManifest.from_dict(["not", "a", "dict"])
