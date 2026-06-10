"""Tests for echo_bench.logging.report (Task F-004)."""

from __future__ import annotations

import json

import pytest

from echo_bench.logging.manifest import RunManifest
from echo_bench.logging.report import generate_report, write_report
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.utils.hash import canonical_hash


def _manifest() -> RunManifest:
    pack = ReproducibilityPack(
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
    return RunManifest(
        pack=pack,
        rendererVersion="rend1",
        policyVersion="pol1",
        probeVersion="prb1",
        metricVersion="met1",
        reportVersion="rpt1",
        seedBatchId="seedbatch0",
    )


def _metrics() -> dict:
    return {
        "utility_mean": 0.42,
        "coverage": 0.81,
        "diversity": 0.55,
        "n_rounds": 12,
    }


def test_generate_report_deterministic():
    m = _manifest()
    r1 = generate_report(_metrics(), m)
    r2 = generate_report(_metrics(), m)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1 == r2


def test_report_embeds_manifest_hash_and_seed_batch():
    m = _manifest()
    r = generate_report(_metrics(), m)
    assert r["manifestHash"] == m.manifest_hash()
    assert r["seedBatchId"] == m.seedBatchId
    # Full hash chain is reachable via the embedded manifest/pack.
    assert r["manifest"]["pack"]["traceHash"] == "trace0"


def test_report_hash_recomputation_matches():
    m = _manifest()
    r = generate_report(_metrics(), m)
    without = {k: v for k, v in r.items() if k != "reportHash"}
    assert canonical_hash(without) == r["reportHash"]


def test_report_hash_changes_with_metrics():
    m = _manifest()
    r1 = generate_report(_metrics(), m)
    metrics2 = _metrics()
    metrics2["utility_mean"] = 0.99
    r2 = generate_report(metrics2, m)
    assert r1["reportHash"] != r2["reportHash"]


def test_extra_embedded_and_affects_hash():
    m = _manifest()
    r1 = generate_report(_metrics(), m)
    r2 = generate_report(_metrics(), m, extra={"k": 4})
    assert r2["extra"] == {"k": 4}
    assert r1["reportHash"] != r2["reportHash"]


def test_generate_report_rejects_bad_manifest():
    with pytest.raises(ValueError):
        generate_report(_metrics(), {"not": "a manifest"})


def test_generate_report_rejects_bad_metrics():
    with pytest.raises(ValueError):
        generate_report("not-a-table", _manifest())


def test_write_report_reloadable_and_hash_matches(tmp_path):
    m = _manifest()
    r = generate_report(_metrics(), m)
    path = tmp_path / "nested" / "report.json"
    write_report(r, str(path))
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == r
    without = {k: v for k, v in loaded.items() if k != "reportHash"}
    assert canonical_hash(without) == loaded["reportHash"]


def test_write_report_rejects_non_dict(tmp_path):
    with pytest.raises(ValueError):
        write_report(["not", "a", "dict"], str(tmp_path / "x.json"))
