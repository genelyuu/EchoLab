"""Tests for echo_bench.metrics.compare (Task D-007).

Covers paired effect size, the seeded/exact sign-flip permutation test (a
clearly-separated pair is significant, a null pair is not), Holm/BH correction
(reduces false positives, deterministic), rank stability, and the top-level
reference-vs-others comparison. All test data is hand-crafted (n <= 12 → the
exact permutation branch), so results are fully deterministic with no RNG.
"""

from __future__ import annotations

from echo_bench.experiments.e2_policy import (
    COMPARISON_REFERENCE_POLICY,
    run_e2_policy,
)
from echo_bench.metrics.compare import (
    cohens_dz,
    compare_reference_to_others,
    multiple_comparison_correction,
    paired_mean_diff,
    permutation_test_paired,
    rank_stability,
)

# A clearly-separated pair: a >> b on every seed, small within-pair variance.
_A = [0.80, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.82]
_B = [0.30, 0.31, 0.29, 0.32, 0.30, 0.28, 0.31, 0.30]
# A null pair: differences are perfectly symmetric, observed mean diff == 0.
_NULL_A = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
_NULL_B = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


def test_paired_mean_diff_and_effect_size():
    assert paired_mean_diff(_A, _B) > 0.45
    # Large, finite paired effect size for a consistent separation.
    assert cohens_dz(_A, _B) > 2.0


def test_cohens_dz_degenerate_cases_are_zero():
    assert cohens_dz([0.5], [0.2]) == 0.0  # n < 2
    assert cohens_dz([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.0  # zero variance


def test_permutation_separated_pair_is_significant():
    res = permutation_test_paired(_A, _B, key="m")
    assert res["method"] == "exact"
    assert res["p_value"] < 0.05


def test_permutation_null_pair_is_not_significant():
    res = permutation_test_paired(_NULL_A, _NULL_B, key="m")
    assert res["p_value"] == 1.0  # observed mean diff is 0 → nothing more extreme


def test_permutation_is_deterministic():
    r1 = permutation_test_paired(_A, _B, key="m")
    r2 = permutation_test_paired(_A, _B, key="m")
    assert r1 == r2


def test_holm_correction_reduces_false_positives():
    res = multiple_comparison_correction([0.001, 0.04, 0.5, 0.9], method="holm")
    # Only the strongest survives Holm at alpha=0.05.
    assert res["reject"] == [True, False, False, False]
    assert abs(res["adjusted"][0] - 0.004) < 1e-12
    # Adjusted p-values never fall below the raw values.
    assert all(adj >= raw - 1e-12 for adj, raw in zip(res["adjusted"], [0.001, 0.04, 0.5, 0.9]))


def test_bh_correction_monotone_and_deterministic():
    res = multiple_comparison_correction([0.001, 0.04, 0.5, 0.9], method="bh")
    assert res["reject"] == [True, False, False, False]
    assert abs(res["adjusted"][0] - 0.004) < 1e-12
    assert res == multiple_comparison_correction([0.001, 0.04, 0.5, 0.9], method="bh")


def test_rank_stability_dominant_policy_is_stable():
    per_seed = {
        "A": [{"m": 0.9}, {"m": 0.91}, {"m": 0.89}, {"m": 0.92}, {"m": 0.9}, {"m": 0.9}],
        "B": [{"m": 0.5}, {"m": 0.52}, {"m": 0.48}, {"m": 0.5}, {"m": 0.51}, {"m": 0.49}],
        "C": [{"m": 0.1}, {"m": 0.12}, {"m": 0.08}, {"m": 0.1}, {"m": 0.11}, {"m": 0.09}],
    }
    rs = rank_stability(per_seed, "m", n_subbatches=3)
    assert rs["perPolicy"]["A"]["meanRank"] == 1.0
    assert rs["perPolicy"]["A"]["rankStd"] == 0.0
    assert rs["perPolicy"]["C"]["meanRank"] == 3.0


def test_compare_reference_to_others_structure_and_significance():
    # 'coverage' separates TRACE_GREEDY from RANDOM; 'noise' does not.
    per_seed = {
        "TRACE_GREEDY": [{"coverage": a, "noise": 0.5} for a in _A],
        "RANDOM": [{"coverage": b, "noise": 0.5} for b in _B],
    }
    out = compare_reference_to_others(
        per_seed, ["coverage", "noise"], reference="TRACE_GREEDY"
    )
    assert out["reference"] == "TRACE_GREEDY"
    assert out["others"] == ["RANDOM"]

    cov = out["byMetric"]["coverage"]["comparisons"][0]
    assert cov["policy"] == "RANDOM"
    assert cov["mean_diff"] > 0.45
    assert cov["significant"] is True

    noise = out["byMetric"]["noise"]["comparisons"][0]
    assert noise["significant"] is False

    # Deterministic.
    assert out == compare_reference_to_others(
        per_seed, ["coverage", "noise"], reference="TRACE_GREEDY"
    )


def test_e2_report_embeds_comparisons_block():
    # The E2 runner threads per-seed values into a paired comparisons block.
    report = run_e2_policy(
        base_seed=42, H=4, k=4, pool_size=16, n=3, dry_run=False
    )
    comp = report["comparisons"]
    assert comp["reference"] == COMPARISON_REFERENCE_POLICY
    assert "byMetric" in comp and "coordinate_coverage" in comp["byMetric"]
    # Every other policy appears as a comparison row for each metric.
    cov_rows = comp["byMetric"]["coordinate_coverage"]["comparisons"]
    assert {r["policy"] for r in cov_rows} == set(comp["others"])
    for r in cov_rows:
        assert "p_adjusted" in r and isinstance(r["significant"], bool)
