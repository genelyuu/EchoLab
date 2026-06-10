"""Tests for the E-014 multi-seed-family runner (TRD V-016 / GR-007).

Covers, at SMALL parameters only (the production 5-family x n=10 x pool=64
sweep is executed separately by the controller):

- the C-011 config-freeze gate (pass on the real manifest; hard Korean
  ValueError on a doctored frozen hash injected via monkeypatch — the real
  manifest file is never touched);
- the pure cross-family aggregation helpers on SYNTHETIC child reports
  (rank stability with unit=FAMILIES, leakage CI block, E1 long-horizon
  thresholds, replay summary) without running any experiment;
- a real small integration run (2 families, n=2, pool=16, H_e2=4): report
  file written, full hash chain present, every aggregation block structured;
- dry-run writes nothing;
- determinism: identical args -> identical outputHash / reportHash.
"""

from __future__ import annotations

import json

import pytest

import echo_bench.experiments.e_seed_families as mod
from echo_bench.experiments.e_seed_families import (
    _REPORTS_DIR,
    DEFAULT_BASE_SEEDS,
    LONG_HORIZON_POLICY,
    V001_COVERAGE_MIN,
    V001_REDUNDANCY_MAX,
    build_e1_long_horizon_block,
    build_leakage_across_families,
    build_per_family_block,
    build_rank_stability_across_families,
    build_replay_audit_summary,
    extract_e2_policy_metric_means,
    run_seed_families,
    verify_trace_greedy_freeze,
)

# Small but real parameters: 2 families, tiny seed batch, 16-card pool, short
# E2/E3 horizon. E1 has no horizon argument (it sweeps its config-defined H
# set), so the small pool keeps its full sweep CPU-fast — the accepted cost.
_SMALL = dict(base_seeds=(42, 7), n=2, k=4, pool_size=16, H_e2=4)

_REQUIRED_HASHES = (
    "configHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

_FORBIDDEN_KEY_TOKENS = (
    "user_id",
    "persona",
    "emotion",
    "preference",
    "user_model",
)


def _glob_reports():
    return (
        set(_REPORTS_DIR.glob("seed_families_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )


def _iter_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_keys(item)


# ---------------------------------------------------------------------------
# Config-freeze gate (C-011)
# ---------------------------------------------------------------------------


def test_freeze_check_passes_on_real_manifest():
    result = verify_trace_greedy_freeze()
    assert result["verified"] is True
    assert result["policyName"] == "TRACE_GREEDY"
    assert result["frozenHash"] == result["liveHash"]


def test_freeze_check_hard_fails_on_drift(monkeypatch):
    """A doctored frozen hash must hard-fail BEFORE anything runs or writes."""
    real = mod._load_frozen_manifest()
    doctored = dict(real)
    doctored["policyEffectiveConfigHash"] = "0" * 64
    monkeypatch.setattr(mod, "_load_frozen_manifest", lambda: doctored)

    before = _glob_reports()
    with pytest.raises(ValueError, match="동결 검증 실패"):
        run_seed_families(dry_run=True, **_SMALL)
    with pytest.raises(ValueError, match="동결 검증 실패"):
        run_seed_families(dry_run=False, **_SMALL)
    assert _glob_reports() == before


def test_freeze_check_fails_on_missing_hash_key(monkeypatch):
    real = mod._load_frozen_manifest()
    doctored = {k: v for k, v in real.items() if k != "policyEffectiveConfigHash"}
    monkeypatch.setattr(mod, "_load_frozen_manifest", lambda: doctored)
    with pytest.raises(ValueError, match="policyEffectiveConfigHash"):
        verify_trace_greedy_freeze()


# ---------------------------------------------------------------------------
# Pure aggregation helpers on SYNTHETIC child reports (no experiment runs)
# ---------------------------------------------------------------------------


def _synthetic_e2(policy_means):
    """Build a minimal synthetic E2 report from policy -> metric -> mean."""
    metric_keys = sorted(next(iter(policy_means.values())))
    table = [
        {
            "policy": policy,
            "stats": {key: {"mean": mean} for key, mean in means.items()},
        }
        for policy, means in policy_means.items()
    ]
    return {"metricKeys": metric_keys, "table": table}


def test_extract_e2_policy_metric_means():
    report = _synthetic_e2(
        {
            "A": {"coordinate_coverage": 0.9, "redundancy_rate": 0.1},
            "B": {"coordinate_coverage": 0.5, "redundancy_rate": 0.05},
        }
    )
    means = extract_e2_policy_metric_means(report)
    assert means == {
        "A": {"coordinate_coverage": 0.9, "redundancy_rate": 0.1},
        "B": {"coordinate_coverage": 0.5, "redundancy_rate": 0.05},
    }


def test_per_family_block_structure():
    e2_by_family = {
        "42": _synthetic_e2({"A": {"coordinate_coverage": 0.9}}),
        "7": _synthetic_e2({"A": {"coordinate_coverage": 0.8}}),
    }
    block = build_per_family_block(e2_by_family)
    assert set(block) == {"42", "7"}
    assert block["42"]["A"]["coordinate_coverage"] == 0.9
    assert block["7"]["A"]["coordinate_coverage"] == 0.8


def test_rank_stability_units_are_families():
    """Units = FAMILIES: n_units equals the family count, and direction-aware
    ranking is applied to the per-family means."""
    e2_by_family = {
        "42": _synthetic_e2(
            {
                "A": {"coordinate_coverage": 0.9, "redundancy_rate": 0.10},
                "B": {"coordinate_coverage": 0.5, "redundancy_rate": 0.05},
            }
        ),
        "7": _synthetic_e2(
            {
                "A": {"coordinate_coverage": 0.8, "redundancy_rate": 0.20},
                "B": {"coordinate_coverage": 0.6, "redundancy_rate": 0.01},
            }
        ),
        "101": _synthetic_e2(
            {
                "A": {"coordinate_coverage": 0.7, "redundancy_rate": 0.30},
                "B": {"coordinate_coverage": 0.4, "redundancy_rate": 0.02},
            }
        ),
    }
    block = build_rank_stability_across_families(e2_by_family)

    assert block["unit"] == "seed_family"
    assert block["families"] == ["42", "7", "101"]

    cov = block["byMetric"]["coordinate_coverage"]
    assert cov["n_units"] == 3  # units are FAMILIES, not child seeds
    assert cov["direction"] == "higher_is_better"
    # A has the higher coverage in every family -> strictly best in all units.
    assert cov["per_policy"]["A"]["top_rank_probability"] == 1.0
    assert cov["per_policy"]["A"]["mean_rank"] == 1.0
    assert cov["per_policy"]["B"]["top_rank_probability"] == 0.0

    red = block["byMetric"]["redundancy_rate"]
    assert red["direction"] == "lower_is_better"
    # B has the lower redundancy in every family -> best under lower-is-better.
    assert red["per_policy"]["B"]["top_rank_probability"] == 1.0
    assert red["per_policy"]["A"]["mean_rank"] == 2.0


def test_rank_stability_rejects_misaligned_families():
    e2_by_family = {
        "42": _synthetic_e2({"A": {"coordinate_coverage": 0.9}}),
        "7": _synthetic_e2({"B": {"coordinate_coverage": 0.8}}),
    }
    with pytest.raises(ValueError, match="정책 집합"):
        build_rank_stability_across_families(e2_by_family)


def _synthetic_e3(leakage_by_policy):
    """Build a minimal synthetic E3 report from policy -> (proxy, delta)."""
    table = [
        {
            "policy": policy,
            "leakage_proxy": proxy,
            "leakage_delta_vs_random": delta,
        }
        for policy, (proxy, delta) in leakage_by_policy.items()
    ]
    return {
        "leakage": {
            "isProxy": True,
            "disclaimer": "proxy disclaimer",
            "deltaReference": "RANDOM",
            "table": table,
        }
    }


def test_leakage_across_families_block():
    e3_by_family = {
        "42": _synthetic_e3({"RANDOM": (0.10, 0.0), "TRACE_GREEDY": (0.30, 0.20)}),
        "7": _synthetic_e3({"RANDOM": (0.20, 0.0), "TRACE_GREEDY": (0.40, 0.20)}),
        "101": _synthetic_e3({"RANDOM": (0.30, 0.0), "TRACE_GREEDY": (0.50, 0.20)}),
    }
    block = build_leakage_across_families(e3_by_family)

    # THE point of this block: the CI that D-011 marked structurally
    # unavailable at a single family is available across families.
    assert block["ciAvailable"] is True
    assert block["unit"] == "seed_family"
    assert block["isProxy"] is True
    assert block["deltaReference"] == "RANDOM"
    assert block["families"] == ["42", "7", "101"]

    tg = block["perPolicy"]["TRACE_GREEDY"]["leakage_proxy"]
    assert tg["perFamilyValues"] == [0.30, 0.40, 0.50]
    assert tg["mean"] == pytest.approx(0.40)
    # n=3 >= MIN_SUFFICIENT_N -> a real seeded-bootstrap CI.
    assert tg["ci_method"] == "bootstrap" and tg["sufficient_n"] is True
    assert tg["ci_low"] <= tg["mean"] <= tg["ci_high"]

    delta = block["perPolicy"]["TRACE_GREEDY"]["leakage_delta_vs_random"]
    assert delta["perFamilyValues"] == [0.20, 0.20, 0.20]
    assert delta["mean"] == pytest.approx(0.20)
    # The reference's own delta is exactly 0 in every family.
    rnd = block["perPolicy"]["RANDOM"]["leakage_delta_vs_random"]
    assert rnd["mean"] == 0.0


def _synthetic_e1(coverage, redundancy, h_sweep=(4, 20)):
    """Synthetic E1 report with a TRACE_GREEDY row at the max horizon."""
    max_h = max(h_sweep)
    table = [
        {
            "H": h,
            "policy": LONG_HORIZON_POLICY,
            "coordinate_coverage": coverage if h == max_h else 1.0,
            "redundancy_rate": redundancy if h == max_h else 0.0,
        }
        for h in h_sweep
    ]
    return {"config": {"H_sweep": list(h_sweep)}, "table": table}


def test_e1_long_horizon_block_thresholds():
    e1_by_family = {
        "42": _synthetic_e1(coverage=0.95, redundancy=0.02),  # passes both
        "7": _synthetic_e1(coverage=0.40, redundancy=0.50),  # fails both
    }
    block = build_e1_long_horizon_block(e1_by_family)

    assert block["policy"] == LONG_HORIZON_POLICY
    assert block["thresholds"]["coordinate_coverage_min"] == V001_COVERAGE_MIN
    assert block["thresholds"]["redundancy_rate_max"] == V001_REDUNDANCY_MAX

    fam42 = block["perFamily"]["42"]
    assert fam42["H"] == 20  # the max-H cell
    assert fam42["coveragePass"] is True
    assert fam42["redundancyPass"] is True
    assert fam42["familyPass"] is True

    fam7 = block["perFamily"]["7"]
    assert fam7["coveragePass"] is False
    assert fam7["redundancyPass"] is False
    assert fam7["familyPass"] is False

    assert block["allFamiliesPass"] is False

    all_pass = build_e1_long_horizon_block(
        {"42": _synthetic_e1(0.95, 0.02), "7": _synthetic_e1(0.92, 0.05)}
    )
    assert all_pass["allFamiliesPass"] is True


def test_replay_audit_summary_flags():
    ok = {"replayAudit": {"replayable": True}}
    bad = {"replayAudit": {"replayable": False}}
    missing = {}

    summary = build_replay_audit_summary({"42": {"e1": ok, "e2": ok, "e3": ok}})
    assert summary["perFamily"]["42"] == {"e1": True, "e2": True, "e3": True}
    assert summary["allReplayable"] is True

    summary = build_replay_audit_summary({"42": {"e1": ok, "e2": bad}})
    assert summary["allReplayable"] is False

    # A missing audit is None and is NEVER coerced to a pass.
    summary = build_replay_audit_summary({"42": {"e1": ok, "e2": missing}})
    assert summary["perFamily"]["42"]["e2"] is None
    assert summary["allReplayable"] is False


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_rejects_duplicate_seeds_and_unknown_experiments():
    with pytest.raises(ValueError, match="중복"):
        run_seed_families(base_seeds=(42, 42), dry_run=True)
    with pytest.raises(ValueError, match="experiment"):
        run_seed_families(base_seeds=(42,), experiments=("e9",), dry_run=True)
    with pytest.raises(ValueError, match="base_seeds"):
        run_seed_families(base_seeds=(), dry_run=True)


def test_default_base_seeds_are_five_distinct_families():
    assert len(DEFAULT_BASE_SEEDS) >= 5
    assert len(set(DEFAULT_BASE_SEEDS)) == len(DEFAULT_BASE_SEEDS)
    assert 42 in DEFAULT_BASE_SEEDS  # the historical development family


# ---------------------------------------------------------------------------
# Real small integration run (2 families, n=2, pool=16, H_e2=4)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_report():
    return run_seed_families(replay_validate=True, **_SMALL)


def test_small_run_writes_report_with_full_hash_chain(small_report):
    for key in _REQUIRED_HASHES:
        assert isinstance(small_report[key], str) and small_report[key], key
    assert isinstance(small_report["seedBatchId"], str)
    assert "reproducibilityPack" in small_report and "packHash" in small_report

    out_path = (
        _REPORTS_DIR / f"seed_families_{small_report['seedBatchId'][:12]}.json"
    )
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == small_report["reportHash"]


def test_small_run_child_provenance(small_report):
    families = {str(s) for s in _SMALL["base_seeds"]}
    assert set(small_report["childReportHashes"]) == families
    for family in families:
        for exp_id in ("e1", "e2", "e3"):
            assert small_report["childReportHashes"][family][exp_id]
            assert small_report["childSeedBatchIds"][family][exp_id]


def test_small_run_per_family_block(small_report):
    block = small_report["perFamily"]
    assert set(block) == {str(s) for s in _SMALL["base_seeds"]}
    for family, by_policy in block.items():
        assert "TRACE_GREEDY" in by_policy and "RANDOM" in by_policy
        for means in by_policy.values():
            assert "coordinate_coverage" in means
            for value in means.values():
                assert isinstance(value, float)


def test_small_run_rank_stability_units_are_families(small_report):
    block = small_report["rankStabilityAcrossFamilies"]
    assert block["unit"] == "seed_family"
    n_families = len(_SMALL["base_seeds"])
    for metric_block in block["byMetric"].values():
        assert metric_block["n_units"] == n_families


def test_small_run_leakage_ci_block(small_report):
    block = small_report["leakageAcrossFamilies"]
    assert block["ciAvailable"] is True
    assert block["unit"] == "seed_family"
    assert block["isProxy"] is True
    n_families = len(_SMALL["base_seeds"])
    for policy_block in block["perPolicy"].values():
        for key in ("leakage_proxy", "leakage_delta_vs_random"):
            entry = policy_block[key]
            assert len(entry["perFamilyValues"]) == n_families
            assert {"mean", "std", "ci_low", "ci_high"} <= set(entry)
    # The delta reference's own delta is exactly 0 in every family.
    ref = block["deltaReference"]
    assert block["perPolicy"][ref]["leakage_delta_vs_random"]["mean"] == 0.0


def test_small_run_e1_long_horizon_block(small_report):
    block = small_report["e1LongHorizonAcrossFamilies"]
    assert block["policy"] == LONG_HORIZON_POLICY
    for entry in block["perFamily"].values():
        assert entry["H"] == 20  # max of the config H sweep
        assert isinstance(entry["coveragePass"], bool)
        assert isinstance(entry["redundancyPass"], bool)
        assert isinstance(entry["familyPass"], bool)
    assert isinstance(block["allFamiliesPass"], bool)


def test_small_run_replay_audit(small_report):
    # Children self-validated inline (replay_validate=True) and E3 always
    # audits its smoke core -> every flag True at small deterministic params.
    summary = small_report["replayAuditSummary"]
    for family_flags in summary["perFamily"].values():
        assert family_flags == {"e1": True, "e2": True, "e3": True}
    assert summary["allReplayable"] is True

    audit = small_report["replayAudit"]
    assert audit["mode"] == "child_reports"
    assert audit["replayable"] is True


def test_small_run_config_freeze_recorded(small_report):
    freeze = small_report["configFreeze"]
    assert freeze["verified"] is True
    assert freeze["policyName"] == "TRACE_GREEDY"
    assert small_report["config"]["configFreeze"]["taskId"] == "C-011"


def test_small_run_no_forbidden_keys(small_report):
    # PSEUDO_USER_MODEL is the documented, explicitly isolated contrast-
    # baseline POLICY NAME (the only allowed latent-vector policy); its name
    # appearing as a policy key is not a latent user-model field.
    keys = {
        str(key).lower()
        for key in _iter_keys(small_report)
        if not str(key).lower().startswith("pseudo_user_model")
    }
    for token in _FORBIDDEN_KEY_TOKENS:
        assert not any(token in key for key in keys), token


# ---------------------------------------------------------------------------
# Dry run + determinism
# ---------------------------------------------------------------------------


def test_dry_run_validates_and_writes_nothing():
    before = _glob_reports()
    result = run_seed_families(dry_run=True, **_SMALL)

    assert result["dryRun"] is True
    assert result["configFreeze"]["verified"] is True
    assert isinstance(result["configHash"], str) and result["configHash"]
    for family in (str(s) for s in _SMALL["base_seeds"]):
        for exp_id in ("e1", "e2", "e3"):
            plan = result["families"][family][exp_id]
            assert plan["configHash"] and plan["archiveHash"] and plan["poolHash"]

    assert _glob_reports() == before


def test_same_base_seeds_tuple_reproduces_output_hash():
    """Determinism gate: identical args -> identical outputHash/reportHash."""
    kw = dict(base_seeds=(42,), n=2, k=4, pool_size=16, H_e2=4)
    first = run_seed_families(replay_validate=False, **kw)
    second = run_seed_families(replay_validate=False, **kw)
    assert first["outputHash"] == second["outputHash"]
    assert first["reportHash"] == second["reportHash"]
    assert first["seedBatchId"] == second["seedBatchId"]
