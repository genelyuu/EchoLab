"""Tests for echo_bench.metrics.compare.effect_size_summary (Task D-014).

Covers:
- Hand-computed d_z and mean_diff against direct cohens_dz call
- Magnitude label thresholds (boundary cases 0.2/0.5/0.8 — boundaries are
  inclusive at the LOWER bound of each interval, i.e. |d_z| < 0.2 → negligible;
  0.2 ≤ |d_z| < 0.5 → small; 0.5 ≤ |d_z| < 0.8 → medium; |d_z| ≥ 0.8 → large;
  documented in COHEN_MAGNITUDE_THRESHOLDS)
- Determinism: CI identical across calls (seeded bootstrap)
- Zero-variance edge case: cohens_dz returns 0.0 → label "negligible", no crash
- E2 integration: effectSizes block present; d_z values consistent with
  comparisons block
- precomputed path returns bit-identical output to the internal-call path
- ci_method and sufficient_n fields present and self-consistent
"""

from __future__ import annotations

import math

import pytest

from echo_bench.experiments.e2_policy import (
    COMPARISON_REFERENCE_POLICY,
    E2_METRIC_KEYS,
    run_e2_policy,
)
from echo_bench.metrics.compare import (
    COHEN_MAGNITUDE_THRESHOLDS,
    _magnitude_label,
    cohens_dz,
    compare_reference_to_others,
    effect_size_summary,
    paired_mean_diff,
)

# ---------------------------------------------------------------------------
# Shared small synthetic data (n=8, exact permutation branch)
# ---------------------------------------------------------------------------

# Clearly separated: reference >> other on every seed.
_REF = [0.80, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.82]
_OTHER = [0.30, 0.31, 0.29, 0.32, 0.30, 0.28, 0.31, 0.30]


def _make_per_seed(ref_vals, other_vals, metric="m"):
    """Build per_seed_by_policy dict from two aligned value lists."""
    return {
        "REF": [{metric: v} for v in ref_vals],
        "OTHER": [{metric: v} for v in other_vals],
    }


# ---------------------------------------------------------------------------
# Module-scoped fixture: three identical small-E2 runs (H=4, k=4, pool=16, n=3)
# Shared across all E2 integration tests so the expensive run executes once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_e2_reports():
    """Run E2 three times with the same params; return (r1, r2, r3)."""
    kwargs = dict(base_seed=42, H=4, k=4, pool_size=16, n=3, dry_run=False,
                  replay_validate=False)
    r1 = run_e2_policy(**kwargs)
    r2 = run_e2_policy(**kwargs)
    r3 = run_e2_policy(**kwargs)
    return r1, r2, r3


# ---------------------------------------------------------------------------
# 1. Hand-computed d_z and mean_diff agree with direct function calls
# ---------------------------------------------------------------------------


def test_hand_computed_dz_and_mean_diff():
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])

    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]

    expected_dz = cohens_dz(_REF, _OTHER)
    expected_md = paired_mean_diff(_REF, _OTHER)  # REF − OTHER (sign convention)

    assert math.isclose(entry["d_z"], expected_dz, rel_tol=1e-9), (
        f"d_z mismatch: got {entry['d_z']}, expected {expected_dz}"
    )
    assert math.isclose(entry["mean_diff"], expected_md, rel_tol=1e-9), (
        f"mean_diff mismatch: got {entry['mean_diff']}, expected {expected_md}"
    )


def test_n_recorded():
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    assert entry["n"] == len(_REF)


def test_sign_convention_documented():
    """mean_diff is reference − other (positive when reference outperforms)."""
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    # REF > OTHER so mean_diff must be positive.
    assert entry["mean_diff"] > 0


# ---------------------------------------------------------------------------
# 2. Magnitude threshold tests (boundary cases)
# ---------------------------------------------------------------------------


def test_magnitude_thresholds_constant_present():
    """COHEN_MAGNITUDE_THRESHOLDS must be exported and list the four cutoffs."""
    # Expected structure: sorted list of (upper_exclusive_bound, label) except
    # the top label which has no upper bound. The constant must document the
    # four canonical magnitudes.
    assert isinstance(COHEN_MAGNITUDE_THRESHOLDS, (list, tuple))
    labels = {item[1] for item in COHEN_MAGNITUDE_THRESHOLDS}
    assert "negligible" in labels
    assert "small" in labels
    assert "medium" in labels
    assert "large" in labels


def _label_for_dz(abs_dz: float) -> str:
    """Drive effect_size_summary with a synthetic dataset that yields ~abs_dz.

    Constructs paired values (ref, other) such that:
      mean(ref_i - other_i) = abs_dz * std(ref_i - other_i, ddof=1)
    i.e. cohens_dz(ref, other) ≈ abs_dz.

    Strategy: set std of diffs = 1.0 and mean of diffs = abs_dz, by:
      diffs = abs_dz + zero_mean_unit_variance_noise
    Then ref = diffs, other = zeros, so d_z = mean(diffs)/std(diffs, ddof=1) ≈ abs_dz.
    """
    import numpy as np

    rng = np.random.default_rng(seed=12345)
    n = 20  # large enough that the noise std ≈ 1.0
    # Gaussian noise with std close to 1.0 after normalization.
    noise = rng.normal(0, 1.0, n)
    noise -= noise.mean()  # zero mean
    noise /= noise.std(ddof=1)  # unit std (ddof=1 to match cohens_dz)
    # diffs s.t. mean = abs_dz and std = 1.0 → d_z = abs_dz / 1.0 = abs_dz.
    diffs = noise + abs_dz

    ref = list(diffs)
    other = [0.0] * n

    per_seed = _make_per_seed(ref, other)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    return summary["byPolicy"]["OTHER"]["byMetric"]["m"]["magnitude"]


def test_magnitude_below_0_2_is_negligible():
    # |d_z| < 0.2 → negligible (boundary exclusive at top)
    label = _label_for_dz(0.1)
    assert label == "negligible", f"Expected negligible for |d_z|≈0.1, got {label}"


def test_magnitude_0_3_is_small():
    label = _label_for_dz(0.3)
    assert label == "small", f"Expected small for |d_z|≈0.3, got {label}"


def test_magnitude_0_6_is_medium():
    label = _label_for_dz(0.6)
    assert label == "medium", f"Expected medium for |d_z|≈0.6, got {label}"


def test_magnitude_above_0_8_is_large():
    label = _label_for_dz(1.5)
    assert label == "large", f"Expected large for |d_z|≈1.5, got {label}"


# Boundary tests use _magnitude_label directly to avoid floating-point imprecision
# in the data construction helper.  The boundaries are lower-inclusive:
#   exactly 0.2 → small, exactly 0.5 → medium, exactly 0.8 → large.
def test_magnitude_boundary_0_2_is_small():
    """Boundary 0.2 is the lower bound of 'small' (inclusive)."""
    assert _magnitude_label(0.2) == "small"
    assert _magnitude_label(0.1999) == "negligible"


def test_magnitude_boundary_0_5_is_medium():
    """Boundary 0.5 is the lower bound of 'medium' (inclusive)."""
    assert _magnitude_label(0.5) == "medium"
    assert _magnitude_label(0.4999) == "small"


def test_magnitude_boundary_0_8_is_large():
    """Boundary 0.8 is the lower bound of 'large' (inclusive)."""
    assert _magnitude_label(0.8) == "large"
    assert _magnitude_label(0.7999) == "medium"


# ---------------------------------------------------------------------------
# 3. Determinism: CI identical across calls
# ---------------------------------------------------------------------------


def test_determinism():
    per_seed = _make_per_seed(_REF, _OTHER)
    s1 = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    s2 = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    e1 = s1["byPolicy"]["OTHER"]["byMetric"]["m"]
    e2 = s2["byPolicy"]["OTHER"]["byMetric"]["m"]
    assert e1["ci_low"] == e2["ci_low"], "ci_low differs between calls (non-deterministic)"
    assert e1["ci_high"] == e2["ci_high"], "ci_high differs between calls (non-deterministic)"
    assert e1 == e2, "Full entry differs between calls (non-deterministic)"


# ---------------------------------------------------------------------------
# 4. Zero-variance edge case: no crash, label "negligible"
# ---------------------------------------------------------------------------


def test_zero_variance_no_crash_and_negligible():
    """When diffs are all zero, cohens_dz returns 0.0 → label negligible."""
    zero_ref = [0.5, 0.5, 0.5, 0.5]
    zero_other = [0.5, 0.5, 0.5, 0.5]
    per_seed = _make_per_seed(zero_ref, zero_other)
    # Must not raise.
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    assert entry["d_z"] == 0.0
    assert entry["magnitude"] == "negligible"


# ---------------------------------------------------------------------------
# 5. CI fields present and ordered
# ---------------------------------------------------------------------------


def test_ci_fields_present_and_ordered():
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    assert "ci_low" in entry and "ci_high" in entry
    assert entry["ci_low"] <= entry["mean_diff"] <= entry["ci_high"], (
        f"CI [{entry['ci_low']}, {entry['ci_high']}] does not bracket "
        f"mean_diff={entry['mean_diff']}"
    )


# ---------------------------------------------------------------------------
# 6. p_adjusted field present
# ---------------------------------------------------------------------------


def test_p_adjusted_present_and_in_range():
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    assert "p_adjusted" in entry
    assert 0.0 <= entry["p_adjusted"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Top-level structure (including ci_method and sufficient_n fields)
# ---------------------------------------------------------------------------


def test_top_level_structure():
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    assert summary["reference"] == "REF"
    assert "byPolicy" in summary
    assert "OTHER" in summary["byPolicy"]
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    for field in (
        "d_z", "mean_diff", "ci_low", "ci_high", "p_adjusted", "n", "magnitude",
        "ci_method", "sufficient_n",
    ):
        assert field in entry, f"Missing field: {field}"


def test_ci_method_and_sufficient_n_self_consistent():
    """ci_method and sufficient_n must be consistent: bootstrap → True, degenerate → False."""
    per_seed = _make_per_seed(_REF, _OTHER)
    summary = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])
    entry = summary["byPolicy"]["OTHER"]["byMetric"]["m"]
    if entry["ci_method"] == "bootstrap":
        assert entry["sufficient_n"] is True
    else:
        assert entry["sufficient_n"] is False


# ---------------------------------------------------------------------------
# 8. precomputed path returns bit-identical output to the internal-call path
# ---------------------------------------------------------------------------


def test_precomputed_path_bit_identical():
    """Passing precomputed=compare_reference_to_others(...) must yield the
    exact same output as the internal-call path (no double computation).
    """
    per_seed = _make_per_seed(_REF, _OTHER)
    # Internal-call path (no precomputed).
    s_internal = effect_size_summary(per_seed, reference="REF", metric_keys=["m"])

    # Precomputed path: caller supplies the comparison block.
    cmp = compare_reference_to_others(per_seed, metric_keys=["m"], reference="REF")
    s_precomputed = effect_size_summary(
        per_seed, reference="REF", metric_keys=["m"], precomputed=cmp
    )

    assert s_internal == s_precomputed, (
        "precomputed path produced different output from internal-call path"
    )


# ---------------------------------------------------------------------------
# 9. E2 integration: effectSizes block present and consistent with comparisons
# ---------------------------------------------------------------------------


def test_e2_effect_sizes_block_present(small_e2_reports):
    report, _, _ = small_e2_reports
    assert "effectSizes" in report, "effectSizes block missing from E2 report"
    es = report["effectSizes"]
    assert es["reference"] == COMPARISON_REFERENCE_POLICY
    assert "byPolicy" in es


def test_e2_effect_sizes_dz_consistent_with_comparisons(small_e2_reports):
    """d_z in effectSizes must match the value in the comparisons block.
    Asserts checked > 0 to guarantee at least one pair was verified.
    """
    report, _, _ = small_e2_reports
    comparisons = report["comparisons"]
    effect_sizes = report["effectSizes"]

    checked = 0
    # For each metric key and each other policy present in both blocks, verify
    # the d_z values are identical (same input → same function → same value).
    for metric_key in E2_METRIC_KEYS:
        if metric_key not in comparisons["byMetric"]:
            continue
        cmp_rows = {
            row["policy"]: row
            for row in comparisons["byMetric"][metric_key]["comparisons"]
        }
        for policy_name, policy_es in effect_sizes["byPolicy"].items():
            if metric_key not in policy_es["byMetric"]:
                continue
            es_entry = policy_es["byMetric"][metric_key]
            if policy_name not in cmp_rows:
                continue
            cmp_entry = cmp_rows[policy_name]
            assert math.isclose(
                es_entry["d_z"], cmp_entry["cohens_dz"], rel_tol=1e-9
            ), (
                f"d_z mismatch for policy={policy_name}, metric={metric_key}: "
                f"effectSizes={es_entry['d_z']}, comparisons={cmp_entry['cohens_dz']}"
            )
            checked += 1

    assert checked > 0, "No (policy, metric) pairs were checked — consistency test vacuous"


def test_e2_effect_sizes_covers_all_metric_keys(small_e2_reports):
    report, _, _ = small_e2_reports
    es = report["effectSizes"]
    for policy_name, policy_es in es["byPolicy"].items():
        for key in E2_METRIC_KEYS:
            assert key in policy_es["byMetric"], (
                f"effectSizes missing metric {key} for policy {policy_name}"
            )


def test_e2_effect_sizes_ci_method_sufficient_n_present(small_e2_reports):
    """Every effectSizes entry must carry ci_method and sufficient_n."""
    report, _, _ = small_e2_reports
    es = report["effectSizes"]
    for policy_name, policy_es in es["byPolicy"].items():
        for key in E2_METRIC_KEYS:
            entry = policy_es["byMetric"][key]
            assert "ci_method" in entry, (
                f"ci_method missing from effectSizes[{policy_name}][{key}]"
            )
            assert "sufficient_n" in entry, (
                f"sufficient_n missing from effectSizes[{policy_name}][{key}]"
            )
            assert isinstance(entry["sufficient_n"], bool), (
                f"sufficient_n must be bool, got {type(entry['sufficient_n'])}"
            )


def test_e2_precomputed_outputhash_bit_identical(small_e2_reports):
    """The E2 report's outputHash must be the same across all three runs
    (verifying that the precomputed path produces bit-identical effectSizes).
    """
    r1, r2, r3 = small_e2_reports
    assert r1["outputHash"] == r2["outputHash"] == r3["outputHash"], (
        f"outputHash not bit-identical across runs: "
        f"{r1['outputHash'][:16]} / {r2['outputHash'][:16]} / {r3['outputHash'][:16]}"
    )
