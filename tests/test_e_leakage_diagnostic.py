"""Tests for the E-019 expanded leakage diagnostic runner (TRD E-019).

Covers:
- EXPANDED_PROBE_SET is the full B-007 registry (>= 7), a strict superset of
  the frozen DEFAULT_PROBE_SET; E2/E3 stay untouched (their own guards cover
  bit-identity, this file guards the runner's set).
- Pure cross-family aggregation helpers on synthetic rows:
  build_cross_family_excess_block / evaluate_track_l_conditions.
- Small 2-family integration run: per-family rows carry the D-015 / D-016 /
  D-017 fields + leakage_delta_vs_random; per-family B-007 probe-overlap
  audit; cross-family excess block; Track L conditions block (DIAGNOSTIC);
  inline-recompute replay audit; report file + reproducibility pack.
- C-011 freeze gate hard-fails on a doctored manifest (monkeypatch only — the
  real frozen manifest is never touched).
- Fail-closed argument validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import echo_bench.experiments.e_seed_families as seed_families_mod
from echo_bench.experiments.e3_audit import (
    E3_LEAKAGE_DELTA_REFERENCE,
    E3_LEAKAGE_POLICIES,
)
from echo_bench.experiments.e_leakage_diagnostic import (
    EXPANDED_PROBE_SET,
    TRACK_L_MIN_FAMILIES,
    build_cross_family_excess_block,
    build_overlap_caveat,
    evaluate_track_l_conditions,
    run_leakage_diagnostic,
)
from echo_bench.metrics.separability import CHANNEL_NAMES
from echo_bench.probes.probe_overlap import PROBE_OVERLAP_EXCLUDE_THRESHOLD
from echo_bench.probes.strategy_probes import DEFAULT_PROBE_SET, PROBES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

# Small deterministic dev params: 2 families, H=4 (in the allowed horizon
# set), 16-card pool, 5-permutation null (speed; the default 200 is for
# production runs).
_KW = dict(base_seeds=(42, 7), H=4, k=4, pool_size=16, n_permutations=5)

_CACHE: dict = {}


def _report() -> dict:
    """Run the small integration run once and share it across tests."""
    if "report" not in _CACHE:
        _CACHE["report"] = run_leakage_diagnostic(dry_run=False, **_KW)
    return _CACHE["report"]


# ---------------------------------------------------------------------------
# Probe-set contract
# ---------------------------------------------------------------------------


def test_expanded_probe_set_is_full_registry():
    assert EXPANDED_PROBE_SET == tuple(sorted(PROBES))
    assert len(EXPANDED_PROBE_SET) >= 7


def test_expanded_probe_set_strict_superset_of_frozen_trio():
    assert set(DEFAULT_PROBE_SET) < set(EXPANDED_PROBE_SET)


# ---------------------------------------------------------------------------
# Pure helper: build_cross_family_excess_block
# ---------------------------------------------------------------------------


def _row(policy, slate, selection, combined, saturated=False):
    """One synthetic per-family leakage row with the keys the block reads."""
    return {
        "policy": policy,
        "slate_excess_nmi": slate,
        "selection_excess_nmi": selection,
        "combined_excess_nmi": combined,
        "slate_saturation_flag": saturated,
        "selection_saturation_flag": saturated,
        "combined_saturation_flag": saturated,
    }


def test_cross_family_block_structure_and_values():
    rows_by_family = {
        "42": [_row("A", 0.2, 0.1, 0.3), _row("B", -0.1, 0.0, 0.05)],
        "7": [_row("A", 0.4, 0.2, 0.5), _row("B", -0.3, 0.1, 0.15)],
    }
    block = build_cross_family_excess_block(rows_by_family)

    assert block["unit"] == "seed_family"
    assert block["ciAvailable"] is True
    assert block["families"] == ["42", "7"]
    assert block["policies"] == ["A", "B"]
    for channel in CHANNEL_NAMES:
        assert channel in block["perPolicy"]["A"]

    a_slate = block["perPolicy"]["A"]["slate"]
    assert a_slate["perFamilyValues"] == [0.2, 0.4]
    assert a_slate["mean"] == pytest.approx(0.3)
    assert a_slate["n"] == 2
    # n=2 < MIN_SUFFICIENT_N=3: degenerate CI flagged insufficient.
    assert a_slate["sufficient_n"] is False


def test_cross_family_sign_consistency():
    rows_by_family = {
        "42": [_row("A", 0.2, -0.1, 0.0)],
        "7": [_row("A", 0.4, -0.3, 0.1)],
    }
    block = build_cross_family_excess_block(rows_by_family)
    per = block["perPolicy"]["A"]
    assert per["slate"]["signConsistent"] is True  # all strictly positive
    assert per["selection"]["signConsistent"] is True  # all strictly negative
    assert per["combined"]["signConsistent"] is False  # contains an exact 0.0


def test_cross_family_saturation_propagates():
    rows_by_family = {
        "42": [_row("A", 0.2, 0.1, 0.3, saturated=False)],
        "7": [_row("A", 0.4, 0.2, 0.5, saturated=True)],
    }
    block = build_cross_family_excess_block(rows_by_family)
    for channel in CHANNEL_NAMES:
        assert block["perPolicy"]["A"][channel]["allFamiliesUnsaturated"] is False


def test_cross_family_block_rejects_mismatched_policy_sets():
    rows_by_family = {
        "42": [_row("A", 0.2, 0.1, 0.3)],
        "7": [_row("B", 0.4, 0.2, 0.5)],
    }
    with pytest.raises(ValueError):
        build_cross_family_excess_block(rows_by_family)


def test_cross_family_block_rejects_duplicate_policy_rows():
    # A duplicate policy row would silently shadow the earlier one in a dict
    # build — it must fail closed instead.
    rows_by_family = {
        "42": [_row("A", 0.2, 0.1, 0.3), _row("A", 0.9, 0.9, 0.9)],
        "7": [_row("A", 0.4, 0.2, 0.5)],
    }
    with pytest.raises(ValueError):
        build_cross_family_excess_block(rows_by_family)


def test_cross_family_ci_invariant_to_family_order():
    # The bootstrap CI must depend on the family SET, not the argument order:
    # the same per-family values listed in two different family orders give
    # bit-identical CI bounds (perFamilyValues keeps the caller order).
    seeds = ("42", "7", "101", "2025", "31337")
    values = {"42": 0.2, "7": 0.25, "101": 0.3, "2025": 0.22, "31337": 0.28}
    forward = {s: [_row("A", values[s], values[s], values[s])] for s in seeds}
    reversed_order = {
        s: [_row("A", values[s], values[s], values[s])]
        for s in reversed(seeds)
    }
    block_fwd = build_cross_family_excess_block(forward)
    block_rev = build_cross_family_excess_block(reversed_order)
    for channel in CHANNEL_NAMES:
        fwd = block_fwd["perPolicy"]["A"][channel]
        rev = block_rev["perPolicy"]["A"][channel]
        assert fwd["ci_low"] == rev["ci_low"]
        assert fwd["ci_high"] == rev["ci_high"]
        assert fwd["mean"] == pytest.approx(rev["mean"])
        assert fwd["perFamilyValues"] == list(reversed(rev["perFamilyValues"]))


def test_cross_family_block_rejects_empty_input():
    with pytest.raises(ValueError):
        build_cross_family_excess_block({})


# ---------------------------------------------------------------------------
# Pure helper: evaluate_track_l_conditions
# ---------------------------------------------------------------------------


def _five_family_block(values, saturated=False):
    """Cross-family block for one policy with the same values per channel."""
    rows_by_family = {
        str(seed): [
            _row("A", values[i], values[i], values[i], saturated=saturated)
        ]
        for i, seed in enumerate((42, 7, 101, 2025, 31337))
    }
    return build_cross_family_excess_block(rows_by_family)


def test_track_l_all_conditions_hold_on_positive_stable_unsaturated():
    block = _five_family_block([0.2, 0.25, 0.3, 0.22, 0.28])
    conditions = evaluate_track_l_conditions(block)
    for channel in CHANNEL_NAMES:
        cond = conditions["perPolicy"]["A"][channel]
        assert cond["condition1_excess_positive_ci"] is True
        assert cond["condition2_not_saturated"] is True
        assert cond["condition3_channel_named"] is True
        assert cond["condition4_cross_family_stable"] is True
        assert cond["allConditionsHold"] is True


def test_track_l_fails_below_min_families():
    rows_by_family = {
        "42": [_row("A", 0.2, 0.2, 0.2)],
        "7": [_row("A", 0.3, 0.3, 0.3)],
    }
    block = build_cross_family_excess_block(rows_by_family)
    conditions = evaluate_track_l_conditions(block)
    cond = conditions["perPolicy"]["A"]["combined"]
    assert cond["condition4_cross_family_stable"] is False
    assert cond["allConditionsHold"] is False


def test_track_l_fails_on_saturation():
    block = _five_family_block([0.2, 0.25, 0.3, 0.22, 0.28], saturated=True)
    conditions = evaluate_track_l_conditions(block)
    cond = conditions["perPolicy"]["A"]["combined"]
    assert cond["condition2_not_saturated"] is False
    assert cond["allConditionsHold"] is False


def test_track_l_negative_stable_effect_fails_condition1():
    # A consistently NEGATIVE excess is cross-family stable (condition 4) but
    # is not a positive excess (condition 1) — never an unlock.
    block = _five_family_block([-0.2, -0.25, -0.3, -0.22, -0.28])
    conditions = evaluate_track_l_conditions(block)
    cond = conditions["perPolicy"]["A"]["combined"]
    assert cond["condition1_excess_positive_ci"] is False
    assert cond["condition4_cross_family_stable"] is True
    assert cond["allConditionsHold"] is False


def test_track_l_block_is_explicitly_diagnostic():
    block = _five_family_block([0.2, 0.25, 0.3, 0.22, 0.28])
    conditions = evaluate_track_l_conditions(block)
    assert conditions["minFamilies"] == TRACK_L_MIN_FAMILIES == 5
    note = conditions["note"]
    assert "DIAGNOSTIC" in note or "diagnostic" in note
    assert "G-009" in note
    assert "12_CLAIM_LADDER" in conditions["ladderRef"]


# ---------------------------------------------------------------------------
# Pure helper: build_overlap_caveat (B-009)
# ---------------------------------------------------------------------------


def _overlap_audit(flagged=(), exclude=()):
    """Synthetic probeOverlap audit carrying just the keys the caveat reads."""
    return {
        "high_overlap_pairs": [
            {"probe_a": a, "probe_b": b, "overlap": o} for a, b, o in flagged
        ],
        "exclude_merge_candidates": [
            {"probe_a": a, "probe_b": b, "overlap": o} for a, b, o in exclude
        ],
    }


def test_overlap_caveat_minority_flag_is_diagnostic_caveat():
    # The E-019 production shape: 1 flagged pair in 1 of 5 families, nothing
    # at the exclude tier -> a diagnostic caveat, NOT a blocking failure.
    overlap_by_family = {
        "42": _overlap_audit(
            flagged=[("PREFER_LOW_COORD_MAGNITUDE", "PREFER_LOW_SALIENCE", 0.929)]
        ),
        "7": _overlap_audit(),
        "101": _overlap_audit(),
        "2025": _overlap_audit(),
        "31337": _overlap_audit(),
    }
    caveat = build_overlap_caveat(overlap_by_family)
    assert caveat["flaggedPairsByFamily"]["42"] == [
        "PREFER_LOW_COORD_MAGNITUDE|PREFER_LOW_SALIENCE"
    ]
    assert caveat["flaggedPairsByFamily"]["7"] == []
    assert caveat["excludeMergeCandidatesByFamily"]["42"] == []
    assert caveat["familiesWithFlags"] == 1
    assert caveat["totalFamilies"] == 5
    note = caveat["caveatNote"]
    assert "1 of 5 seed families" in note
    assert "diagnostic caveat rather than a blocking failure" in note


def test_overlap_caveat_exclude_tier_names_curation_candidates():
    overlap_by_family = {
        "42": _overlap_audit(
            flagged=[("P_A", "P_B", 0.97)], exclude=[("P_A", "P_B", 0.97)]
        ),
        "7": _overlap_audit(),
    }
    caveat = build_overlap_caveat(overlap_by_family)
    assert caveat["excludeMergeCandidatesByFamily"]["42"] == ["P_A|P_B"]
    assert caveat["familiesWithFlags"] == 1
    note = caveat["caveatNote"]
    assert "exclude-or-merge candidate" in note
    assert "curation" in note
    assert "not a claim" in note


def test_overlap_caveat_clean_case():
    overlap_by_family = {"42": _overlap_audit(), "7": _overlap_audit()}
    caveat = build_overlap_caveat(overlap_by_family)
    assert caveat["familiesWithFlags"] == 0
    assert caveat["totalFamilies"] == 2
    assert all(v == [] for v in caveat["flaggedPairsByFamily"].values())
    assert "exceeded the flag threshold" in caveat["caveatNote"]
    assert "no" in caveat["caveatNote"].lower()


def test_overlap_caveat_rejects_empty_input():
    with pytest.raises(ValueError):
        build_overlap_caveat({})


# ---------------------------------------------------------------------------
# Integration: small 2-family run
# ---------------------------------------------------------------------------


def test_run_uses_expanded_probe_set():
    report = _report()
    assert report["config"]["probes"] == list(EXPANDED_PROBE_SET)
    assert report["config"]["probeSet"] == "expanded_registry"


def test_per_family_rows_carry_d015_d016_d017_fields():
    report = _report()
    families = report["perFamily"]
    assert sorted(families) == sorted(str(s) for s in _KW["base_seeds"])
    expected_policies = sorted(E3_LEAKAGE_POLICIES)
    for family_block in families.values():
        rows = family_block["table"]
        assert [r["policy"] for r in rows] == expected_policies
        for row in rows:
            for key in (
                "leakage_proxy",
                "observed_nmi",
                "null_mean",
                "null_std",
                "excess_nmi",
                "excess_z",
                "slate_excess_nmi",
                "selection_excess_nmi",
                "combined_excess_nmi",
                "saturation_flag",
                "slate_saturation_flag",
                "selection_saturation_flag",
                "combined_saturation_flag",
                "leakage_delta_vs_random",
                "channels",
            ):
                assert key in row, key
            assert row["combined_excess_nmi"] == row["excess_nmi"]
            assert sorted(row["channels"]) == sorted(CHANNEL_NAMES)


def test_reference_policy_delta_is_zero():
    report = _report()
    for family_block in report["perFamily"].values():
        ref_rows = [
            r
            for r in family_block["table"]
            if r["policy"] == E3_LEAKAGE_DELTA_REFERENCE
        ]
        assert len(ref_rows) == 1
        assert ref_rows[0]["leakage_delta_vs_random"] == 0.0


def test_per_family_probe_overlap_audit():
    report = _report()
    n_probes = len(EXPANDED_PROBE_SET)
    expected_pairs = n_probes * (n_probes - 1) // 2
    for family_block in report["perFamily"].values():
        overlap = family_block["probeOverlap"]
        assert overlap["probes"] == list(EXPANDED_PROBE_SET)
        assert len(overlap["pairwise_probe_overlap"]) == expected_pairs
        # Decision-time contexts: every probe episode contributes H rounds.
        assert overlap["n_contexts"] == n_probes * _KW["H"]
        assert isinstance(overlap["high_overlap_pairs"], list)
        # B-009 two-tier keys travel with every family audit.
        assert overlap["exclude_threshold"] == PROBE_OVERLAP_EXCLUDE_THRESHOLD
        assert isinstance(overlap["exclude_merge_candidates"], list)
        for candidate in overlap["exclude_merge_candidates"]:
            assert candidate in overlap["high_overlap_pairs"]
        assert "exclude-or-merge" in overlap["policyNote"]


def test_overlap_caveat_block_in_report():
    report = _report()
    caveat = report["overlapCaveat"]
    families = report["perFamily"]
    assert sorted(caveat["flaggedPairsByFamily"]) == sorted(families)
    assert sorted(caveat["excludeMergeCandidatesByFamily"]) == sorted(families)
    assert caveat["totalFamilies"] == len(families)
    # The caveat block must be CONSISTENT with the per-family audits.
    expected_flagged = {
        family: [
            f"{p['probe_a']}|{p['probe_b']}"
            for p in block["probeOverlap"]["high_overlap_pairs"]
        ]
        for family, block in families.items()
    }
    assert caveat["flaggedPairsByFamily"] == expected_flagged
    assert caveat["familiesWithFlags"] == sum(
        1 for pairs in expected_flagged.values() if pairs
    )
    assert isinstance(caveat["caveatNote"], str) and caveat["caveatNote"]


def test_cross_family_excess_block_in_report():
    report = _report()
    block = report["crossFamilyExcess"]
    assert block["unit"] == "seed_family"
    assert block["ciAvailable"] is True
    assert block["policies"] == sorted(E3_LEAKAGE_POLICIES)
    for policy in block["policies"]:
        for channel in CHANNEL_NAMES:
            entry = block["perPolicy"][policy][channel]
            assert len(entry["perFamilyValues"]) == len(_KW["base_seeds"])
            assert entry["n"] == len(_KW["base_seeds"])
            assert entry["sufficient_n"] is False  # 2 < MIN_SUFFICIENT_N


def test_track_l_conditions_block_in_report():
    report = _report()
    conditions = report["trackLConditions"]
    assert conditions["minFamilies"] == TRACK_L_MIN_FAMILIES
    for policy in sorted(E3_LEAKAGE_POLICIES):
        for channel in CHANNEL_NAMES:
            cond = conditions["perPolicy"][policy][channel]
            # 2 families < 5: condition 4 can never hold in this dev run.
            assert cond["condition4_cross_family_stable"] is False
            assert cond["allConditionsHold"] is False


def test_replay_audit_inline_recompute():
    report = _report()
    audit = report["replayAudit"]
    assert audit["mode"] == "inline_recompute_first_family"
    assert audit["replayable"] is True
    # The audit must state its partial coverage explicitly.
    assert audit["scope"] == "first_family_only"
    assert "first family" in audit["scopeNote"]


def test_proxy_framing_carried():
    report = _report()
    leakage_meta = report["leakageMeta"]
    assert leakage_meta["isProxy"] is True
    assert "NOT" in leakage_meta["disclaimer"]
    assert leakage_meta["deltaReference"] == E3_LEAKAGE_DELTA_REFERENCE
    assert leakage_meta["nullPermutations"] == _KW["n_permutations"]
    assert (
        leakage_meta["overlapExcludeThreshold"] == PROBE_OVERLAP_EXCLUDE_THRESHOLD
    )


def test_report_file_written_and_hashed():
    report = _report()
    seed_batch_id = report["seedBatchId"]
    out_path = _REPORTS_DIR / f"leakage_diagnostic_{seed_batch_id[:12]}.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["reportHash"] == report["reportHash"]
    pack = report["reproducibilityPack"]
    for key in (
        "configHash",
        "archiveHash",
        "poolHash",
        "slateHash",
        "traceHash",
        "outputHash",
        "reportHash",
        "seedBatchId",
    ):
        assert key in pack, key


def test_dry_run_plans_and_writes_nothing():
    # Scoped to this runner's own report prefix so concurrent writers of
    # other report types (e.g. parallel test workers) cannot race the glob.
    pattern = "leakage_diagnostic_*.json"
    before = (
        set(_REPORTS_DIR.glob(pattern)) if _REPORTS_DIR.exists() else set()
    )
    plan = run_leakage_diagnostic(dry_run=True, **_KW)
    after = (
        set(_REPORTS_DIR.glob(pattern)) if _REPORTS_DIR.exists() else set()
    )
    assert plan["dryRun"] is True
    assert plan["config"]["probes"] == list(EXPANDED_PROBE_SET)
    assert "configHash" in plan
    assert sorted(plan["families"]) == sorted(str(s) for s in _KW["base_seeds"])
    assert after == before


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_freeze_gate_hard_fails_on_drift(monkeypatch):
    real = seed_families_mod._load_frozen_manifest()
    doctored = dict(real)
    doctored["policyEffectiveConfigHash"] = "0" * 64
    monkeypatch.setattr(
        seed_families_mod, "_load_frozen_manifest", lambda: doctored
    )
    with pytest.raises(ValueError):
        run_leakage_diagnostic(dry_run=True, **_KW)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"base_seeds": ()},
        {"base_seeds": (42, 42)},
        {"n_permutations": 0},
        {"pool_size": 0},
        {"pool_size": -1},
    ],
)
def test_argument_validation_fails_closed(bad_kwargs):
    kwargs = dict(_KW)
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        run_leakage_diagnostic(dry_run=True, **kwargs)
