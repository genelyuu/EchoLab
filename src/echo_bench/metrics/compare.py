"""Paired policy-comparison statistics (Tasks D-007, D-014).

Deterministic, *seeded* significance tooling for comparing trace-only policies on
the same seed batch. Because every policy in a run is evaluated over the SAME
ordered child-seed batch (``derive_child_seeds``), per-seed metric values align
pairwise across policies — so a **paired** design is valid and far more powerful
than an unpaired one.

What this module provides
=========================
- :func:`paired_diffs` / :func:`paired_mean_diff` — per-seed differences.
- :func:`cohens_dz` — paired effect size (mean difference / std of differences).
- :func:`permutation_test_paired` — a sign-flip permutation test. Exact
  (enumerates all ``2**n`` sign assignments) when ``n`` is small, otherwise a
  *seeded* resample — so the p-value is a pure function of its inputs and replays
  bit-identically. No global RNG, wall-clock, or process entropy.
- :func:`multiple_comparison_correction` — Holm (FWER) or Benjamini-Hochberg
  (FDR) adjustment over a family of p-values.
- :func:`rank_stability` — how stable a policy ranking is across seed sub-batches.
- :func:`compare_reference_to_others` — the top-level: compare one reference
  policy (e.g. ``TRACE_GREEDY``) against every other policy across a set of
  metrics, with per-metric multiple-comparison correction.
- :func:`effect_size_summary` — D-014 consolidated effect-size table: for each
  non-reference policy and each metric, ``d_z``, ``mean_diff`` (reference −
  other), bootstrap CI of the paired diffs, Holm-adjusted p, ``n``, and a
  magnitude label from :data:`COHEN_MAGNITUDE_THRESHOLDS`. Reuses
  :func:`compare_reference_to_others` internals so statistical logic is never
  duplicated.

Magnitude labels (:data:`COHEN_MAGNITUDE_THRESHOLDS`)
======================================================
Standard Cohen thresholds applied to ``|d_z|``:

+-------------+-----------------------------+
| label       | condition                   |
+=============+=============================+
| negligible  | ``|d_z| < 0.2``             |
| small       | ``0.2 ≤ |d_z| < 0.5``       |
| medium      | ``0.5 ≤ |d_z| < 0.8``       |
| large       | ``|d_z| ≥ 0.8``             |
+-------------+-----------------------------+

Boundaries are **lower-inclusive / upper-exclusive** at each step (i.e. exactly
``0.2`` is "small", exactly ``0.5`` is "medium", exactly ``0.8`` is "large").
The constant :data:`COHEN_MAGNITUDE_THRESHOLDS` is a tuple of
``(upper_exclusive_bound, label)`` pairs in ascending bound order; the final
``(None, "large")`` entry signals "no upper bound".

Scope and guardrails
====================
These are SYSTEM-LEVEL statistics over a controlled testbed. They quantify
whether observable metric differences between policies are distinguishable from
chance under a controlled seed batch — nothing about users, preference, emotion,
wellbeing, privacy, or legal compliance, and no real-world generalization claim.

Identifiers / keys stay English; runtime log lines are Korean per convention.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "paired_diffs",
    "paired_mean_diff",
    "cohens_dz",
    "permutation_test_paired",
    "multiple_comparison_correction",
    "rank_stability",
    "compare_reference_to_others",
    "effect_size_summary",
    "PERMUTATION_RESAMPLES",
    "PERMUTATION_EXACT_MAX_N",
    "DEFAULT_ALPHA",
    "COHEN_MAGNITUDE_THRESHOLDS",
]

_logger = get_logger(__name__)

# Fixed, documented constants so comparisons are deterministic & config-free.
PERMUTATION_RESAMPLES = 2000
# Enumerate all 2**n sign flips exactly when n <= this (2**12 = 4096 ~ resamples);
# above it, fall back to a seeded resample.
PERMUTATION_EXACT_MAX_N = 12
DEFAULT_ALPHA = 0.05

# Salt mixed into the permutation seed derivation; bumping it changes every test.
_PERMUTATION_SALT = "compare-permutation-h1"

# Cohen's d_z magnitude thresholds (D-014, TRD V-011).
#
# Applied to |d_z|; boundaries are lower-inclusive / upper-exclusive:
#   < 0.2          → "negligible"
#   0.2 ≤ |d_z| < 0.5  → "small"
#   0.5 ≤ |d_z| < 0.8  → "medium"
#   |d_z| ≥ 0.8       → "large"
#
# Format: tuple of (upper_exclusive_bound, label) in ascending bound order.
# The final entry uses None to signal "no upper bound".
COHEN_MAGNITUDE_THRESHOLDS: Tuple[Tuple[object, str], ...] = (
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
    (None, "large"),
)


def _as_floats(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def paired_diffs(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Per-seed differences ``a_i - b_i`` (the two batches must be aligned)."""
    a = _as_floats(a)
    b = _as_floats(b)
    if len(a) != len(b):
        raise ValueError(
            f"paired_diffs: 길이가 다른 배치는 짝지을 수 없습니다 "
            f"(len(a)={len(a)}, len(b)={len(b)})"
        )
    return [a[i] - b[i] for i in range(len(a))]


def paired_mean_diff(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean of the per-seed paired differences ``a_i - b_i``."""
    diffs = paired_diffs(a, b)
    return float(np.mean(diffs)) if diffs else 0.0


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired effect size: ``mean(diff) / std(diff, ddof=1)``.

    Returns ``0.0`` when there are fewer than two pairs or when the within-pair
    difference has zero variance (effect size undefined — rely on the
    permutation p-value in that degenerate case). Finite for normal inputs.
    """
    diffs = paired_diffs(a, b)
    n = len(diffs)
    if n < 2:
        return 0.0
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return 0.0
    return mean / std


def _permutation_seed(diffs: Sequence[float], key: str) -> int:
    digest = canonical_hash(
        {"diffs": _as_floats(diffs), "key": key, "salt": _PERMUTATION_SALT}
    )
    return int(digest, 16) % (2 ** 128)


def permutation_test_paired(
    a: Sequence[float], b: Sequence[float], key: str
) -> Dict[str, Any]:
    """Two-sided paired sign-flip permutation test on ``a_i - b_i``.

    Under the null (exchangeable signs), the sign of each paired difference is
    equally likely to flip. We compare the observed mean difference's magnitude
    against the permutation distribution of mean magnitudes.

    Exact when ``n <= PERMUTATION_EXACT_MAX_N`` (enumerate all ``2**n`` sign
    vectors; the identity assignment is included so ``p`` is naturally
    ``>= 1/2**n``). Otherwise a *seeded* resample of ``PERMUTATION_RESAMPLES``
    sign vectors with an add-one correction. Deterministic in both branches.

    Returns ``{"p_value", "observed", "n", "method": "exact"|"sampled",
    "resamples"}``.
    """
    diffs = paired_diffs(a, b)
    n = len(diffs)
    observed = float(np.mean(diffs)) if n else 0.0
    if n == 0 or all(d == 0.0 for d in diffs):
        return {
            "p_value": 1.0,
            "observed": observed,
            "n": n,
            "method": "degenerate",
            "resamples": 0,
        }

    arr = np.asarray(diffs, dtype=np.float64)
    target = abs(observed)
    # Tiny tolerance so the observed assignment counts itself despite float noise.
    eps = 1e-12

    if n <= PERMUTATION_EXACT_MAX_N:
        count = 0
        total = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            perm_mean = float(np.dot(arr, signs) / n)
            if abs(perm_mean) >= target - eps:
                count += 1
            total += 1
        p_value = count / total
        method = "exact"
        resamples = total
    else:
        rng = np.random.default_rng(_permutation_seed(diffs, key))
        signs = rng.choice(
            np.array([1.0, -1.0]), size=(PERMUTATION_RESAMPLES, n)
        )
        perm_means = (signs * arr).mean(axis=1)
        count = int(np.sum(np.abs(perm_means) >= target - eps))
        # Add-one correction (the observed assignment is a valid permutation).
        p_value = (count + 1) / (PERMUTATION_RESAMPLES + 1)
        method = "sampled"
        resamples = PERMUTATION_RESAMPLES

    return {
        "p_value": float(min(1.0, p_value)),
        "observed": observed,
        "n": n,
        "method": method,
        "resamples": resamples,
    }


def multiple_comparison_correction(
    pvalues: Sequence[float],
    method: str = "holm",
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Adjust a family of p-values for multiple comparisons.

    ``method="holm"`` controls the family-wise error rate (Holm step-down);
    ``method="bh"`` controls the false-discovery rate (Benjamini-Hochberg
    step-up). Returns adjusted p-values **in the original input order** plus a
    boolean ``reject`` per hypothesis at ``alpha``. Deterministic.
    """
    p = _as_floats(pvalues)
    m = len(p)
    if m == 0:
        return {"method": method, "alpha": alpha, "adjusted": [], "reject": []}

    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m

    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            val = min(1.0, (m - rank) * p[idx])
            running = max(running, val)  # enforce monotonic non-decreasing
            adjusted[idx] = running
    elif method == "bh":
        running = 1.0
        # Step up from the largest p-value to the smallest.
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            val = min(1.0, p[idx] * m / (rank + 1))
            running = min(running, val)  # enforce monotonic non-increasing
            adjusted[idx] = running
    else:
        raise ValueError(
            f"multiple_comparison_correction: 알 수 없는 method={method!r} "
            "(holm|bh 만 지원)"
        )

    reject = [adjusted[i] < alpha for i in range(m)]
    return {
        "method": method,
        "alpha": float(alpha),
        "adjusted": adjusted,
        "reject": reject,
    }


def rank_stability(
    per_seed_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    key: str,
    n_subbatches: int = 3,
) -> Dict[str, Any]:
    """Rank stability of policies on metric ``key`` across seed sub-batches.

    Splits the shared (ordered) seed batch into ``n_subbatches`` contiguous
    chunks; within each chunk ranks policies by their mean of ``key`` (rank 1 =
    highest mean, regardless of metric direction) and records each policy's rank.
    Returns, per policy, the mean rank and the rank std across chunks (lower std =
    more stable). Deterministic.

    **Disambiguation (D-007 vs D-013):** This function is the D-007 sub-batch
    diagnostic. It always ranks by sub-batch MEANS with higher mean = rank 1,
    irrespective of whether the metric is higher-is-better or lower-is-better.
    The top-level ``rankStability`` block in the E2 report (produced by
    :func:`echo_bench.metrics.aggregate.rank_stability_by_metric`) is the D-013
    unit-level ranking that applies the documented :data:`METRIC_DIRECTIONS` for
    each metric. Both exist for complementary purposes; neither replaces the other.
    """
    policies = sorted(per_seed_by_policy)
    if not policies:
        return {"key": key, "nSubbatches": 0, "perPolicy": {}}

    n = min(len(per_seed_by_policy[p]) for p in policies)
    k = max(1, min(int(n_subbatches), n))
    # Contiguous chunk boundaries over [0, n).
    bounds = [round(i * n / k) for i in range(k + 1)]

    ranks_by_policy: Dict[str, List[int]] = {p: [] for p in policies}
    for c in range(k):
        lo, hi = bounds[c], bounds[c + 1]
        if hi <= lo:
            continue
        means = {
            p: float(
                np.mean([float(per_seed_by_policy[p][i][key]) for i in range(lo, hi)])
            )
            for p in policies
        }
        # Rank 1 = highest mean; deterministic tiebreak by policy name.
        order = sorted(policies, key=lambda p: (-means[p], p))
        for rank, p in enumerate(order, start=1):
            ranks_by_policy[p].append(rank)

    per_policy = {
        p: {
            "meanRank": float(np.mean(ranks)) if ranks else 0.0,
            "rankStd": float(np.std(ranks, ddof=0)) if ranks else 0.0,
            "ranks": ranks,
        }
        for p, ranks in ranks_by_policy.items()
    }
    return {
        "key": key,
        "nSubbatches": k,
        "perPolicy": per_policy,
        # D-007 sub-batch diagnostic: ranks sub-batch MEANS with higher=rank1
        # regardless of metric direction (lower-is-better metrics are NOT
        # inverted here). The direction-aware unit-level ranking is the
        # top-level ``rankStability`` block in the E2 report (D-013,
        # aggregate.rank_stability_by_metric with METRIC_DIRECTIONS).
        "note": (
            "D-007 sub-batch diagnostic: ranks sub-batch means with "
            "higher=rank1 regardless of metric direction. "
            "For direction-aware unit-level rankings see the top-level "
            "rankStability block (D-013, aggregate.rank_stability_by_metric)."
        ),
    }


def _metric_values(
    per_seed: Sequence[Mapping[str, Any]], key: str
) -> List[float]:
    return [float(d[key]) for d in per_seed if key in d and d[key] is not None]


def compare_reference_to_others(
    per_seed_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    metric_keys: Sequence[str],
    reference: str,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "holm",
) -> Dict[str, Any]:
    """Compare a ``reference`` policy against every other policy across metrics.

    For each metric in ``metric_keys`` and each other policy, computes the paired
    mean difference ``reference - other``, the paired effect size
    (:func:`cohens_dz`), and the paired permutation p-value. P-values are
    corrected **within each metric** (the family is "reference vs the K other
    policies on this metric") via :func:`multiple_comparison_correction`. Also
    attaches :func:`rank_stability` per metric. Deterministic.

    Returns::

        {
          "reference": <name>, "alpha": ..., "correction": ...,
          "others": [<names>],
          "byMetric": {
            <key>: {
              "comparisons": [
                {"policy", "mean_diff", "cohens_dz", "p_value",
                 "p_adjusted", "significant", "method", "n"}, ...
              ],
              "rankStability": {...},
            }, ...
          },
        }
    """
    if reference not in per_seed_by_policy:
        raise ValueError(
            f"compare_reference_to_others: reference={reference!r} 가 "
            "per_seed_by_policy 에 없습니다"
        )
    others = sorted(p for p in per_seed_by_policy if p != reference)

    by_metric: Dict[str, Any] = {}
    for key in metric_keys:
        ref_vals = _metric_values(per_seed_by_policy[reference], key)
        raw: List[Dict[str, Any]] = []
        pvals: List[float] = []
        for other in others:
            other_vals = _metric_values(per_seed_by_policy[other], key)
            if len(other_vals) != len(ref_vals) or len(ref_vals) < 2:
                # Cannot pair — record a degenerate, non-significant comparison.
                raw.append(
                    {
                        "policy": other,
                        "mean_diff": 0.0,
                        "cohens_dz": 0.0,
                        "p_value": 1.0,
                        "method": "degenerate",
                        "n": min(len(ref_vals), len(other_vals)),
                    }
                )
                pvals.append(1.0)
                continue
            test = permutation_test_paired(ref_vals, other_vals, key=f"{key}:{other}")
            raw.append(
                {
                    "policy": other,
                    "mean_diff": paired_mean_diff(ref_vals, other_vals),
                    "cohens_dz": cohens_dz(ref_vals, other_vals),
                    "p_value": test["p_value"],
                    "method": test["method"],
                    "n": test["n"],
                }
            )
            pvals.append(test["p_value"])

        corrected = multiple_comparison_correction(pvals, method=correction, alpha=alpha)
        comparisons = []
        for i, comp in enumerate(raw):
            comp = dict(comp)
            comp["p_adjusted"] = corrected["adjusted"][i]
            comp["significant"] = bool(corrected["reject"][i])
            comparisons.append(comp)

        by_metric[key] = {
            "comparisons": comparisons,
            "rankStability": rank_stability(per_seed_by_policy, key),
        }

    _logger.info(
        "정책 비교 통계를 계산했습니다 (reference=%s, others=%d, metrics=%d, "
        "correction=%s)",
        reference,
        len(others),
        len(list(metric_keys)),
        correction,
    )
    return {
        "reference": reference,
        "alpha": float(alpha),
        "correction": correction,
        "others": others,
        "byMetric": by_metric,
    }


def _magnitude_label(abs_dz: float) -> str:
    """Map ``|d_z|`` to a Cohen magnitude label using :data:`COHEN_MAGNITUDE_THRESHOLDS`.

    Boundaries are lower-inclusive / upper-exclusive at each step.
    """
    for upper, label in COHEN_MAGNITUDE_THRESHOLDS:
        if upper is None or abs_dz < float(upper):
            return label
    # Fallback (unreachable given the None sentinel, but makes mypy happy).
    return "large"  # pragma: no cover


def effect_size_summary(
    per_seed_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference: str,
    metric_keys: Sequence[str],
    alpha: float = DEFAULT_ALPHA,
    correction: str = "holm",
) -> Dict[str, Any]:
    """Consolidated effect-size table for the paper (Task D-014, TRD V-011).

    For each non-reference policy and each metric key, produces::

        {
          "reference": <name>,
          "byPolicy": {
            <policy>: {
              "byMetric": {
                <key>: {
                  "d_z":        float,   # cohens_dz(reference, policy)
                  "mean_diff":  float,   # paired mean diff: reference − other
                  "ci_low":     float,   # seeded bootstrap 95 % CI of the diffs
                  "ci_high":    float,
                  "p_adjusted": float,   # Holm-adjusted permutation p-value
                  "n":          int,     # number of aligned paired observations
                  "magnitude":  str,     # Cohen label ("negligible"/"small"/
                                         #               "medium"/"large")
                }, ...
              }
            }, ...
          }
        }

    **Sign convention:** ``mean_diff = mean(reference_i − other_i)``, so a
    positive value means the reference policy outperforms on this metric.

    **Statistical logic reuse:** internally calls :func:`compare_reference_to_others`
    and extracts ``cohens_dz``, ``mean_diff``, and ``p_adjusted`` from its output
    (no duplication of permutation / correction logic). The bootstrap CI is the only
    addition: :func:`echo_bench.metrics.aggregate.aggregate_values` is applied to
    the paired diffs for a deterministic seeded bootstrap CI at the module-level
    ``CI_LEVEL = 0.95`` (same machinery as seed-batch aggregation).

    Deterministic: the seeded bootstrap uses ``canonical_hash`` of the diffs + a
    distinguishing ``key`` string; identical inputs yield bit-identical output.
    """
    # Lazy import to avoid a circular dependency (aggregate imports nothing from
    # compare; compare would import from aggregate only here — no cycle in practice
    # because compare does not re-export aggregate symbols, but the lazy import
    # makes the dependency direction explicit at the call site).
    from echo_bench.metrics.aggregate import aggregate_values  # noqa: PLC0415

    if reference not in per_seed_by_policy:
        raise ValueError(
            f"effect_size_summary: reference={reference!r} 가 "
            "per_seed_by_policy 에 없습니다"
        )

    # Run the full comparison to extract p_adjusted and d_z / mean_diff.
    cmp = compare_reference_to_others(
        per_seed_by_policy,
        metric_keys=metric_keys,
        reference=reference,
        alpha=alpha,
        correction=correction,
    )

    others = cmp["others"]
    ref_vals_by_key: Dict[str, List[float]] = {
        key: _metric_values(per_seed_by_policy[reference], key)
        for key in metric_keys
    }

    by_policy: Dict[str, Any] = {}
    for other in others:
        other_by_metric: Dict[str, Any] = {}
        for key in metric_keys:
            # Extract pre-computed d_z / mean_diff / p_adjusted from compare block.
            comp_rows = {
                row["policy"]: row
                for row in cmp["byMetric"][key]["comparisons"]
            }
            comp_row = comp_rows[other]

            ref_v = ref_vals_by_key[key]
            other_v = _metric_values(per_seed_by_policy[other], key)

            # CI of paired diffs via seeded bootstrap (aggregate_values).
            if len(ref_v) >= 2 and len(other_v) == len(ref_v):
                diffs = paired_diffs(ref_v, other_v)
                ci_agg = aggregate_values(diffs, key=f"effect_size_diff:{key}:{other}")
                ci_low = ci_agg["ci_low"]
                ci_high = ci_agg["ci_high"]
            else:
                # Degenerate: not enough paired observations for CI.
                md = comp_row["mean_diff"]
                ci_low = md
                ci_high = md

            abs_dz = abs(comp_row["cohens_dz"])
            magnitude = _magnitude_label(abs_dz)

            other_by_metric[key] = {
                "d_z": float(comp_row["cohens_dz"]),
                "mean_diff": float(comp_row["mean_diff"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_adjusted": float(comp_row["p_adjusted"]),
                "n": int(comp_row["n"]),
                "magnitude": magnitude,
            }

        by_policy[other] = {"byMetric": other_by_metric}

    _logger.info(
        "효과 크기 요약 테이블을 계산했습니다 (reference=%s, policies=%d, metrics=%d)",
        reference,
        len(others),
        len(list(metric_keys)),
    )
    return {
        "reference": reference,
        "byPolicy": by_policy,
    }
