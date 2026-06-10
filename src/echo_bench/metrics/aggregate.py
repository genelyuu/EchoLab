"""Seed-batch aggregation statistics (Tasks D-006, D-013).

Pure, deterministic aggregation over a list of per-seed metric values: mean,
sample std (ddof=1), and a CONFIDENCE INTERVAL via a *seeded* bootstrap. The
bootstrap RNG is seeded from ``canonical_hash`` of the input values, so the CI is
a pure function of its inputs and replays bit-identically. No global RNG,
wall-clock, or process entropy ever enters the computation.

Task D-013 adds **rank stability** over the same per-unit value lists:
:func:`rank_stability` ranks policies within each resampling *unit* (a unit is
whatever produced one aligned value per policy — a child seed today, a seed
family in E-014, a batch in E-015) and reports each policy's top-rank
probability, mean rank, and rank distribution. RNG-free: pure sorting, so the
output is trivially deterministic. The metric-name -> ranking-direction map
:data:`METRIC_DIRECTIONS` and the :func:`rank_stability_by_metric` wrapper
apply the documented direction per benchmark metric. This module-level
``rank_stability`` (one value per unit per policy) is DISTINCT from the D-007
sub-batch diagnostic of the same name in ``echo_bench.metrics.compare`` (which
ranks sub-batch *means*); both exist, neither replaces the other.

These are SYSTEM-LEVEL statistics over a controlled testbed. A rank says which
policy scored higher on a system-level metric under controlled conditions —
nothing about user preference, emotion, wellbeing, privacy, or legal compliance,
and no real-world generalization claim.

Identifiers / keys stay English; runtime log lines are Korean per convention.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "aggregate_values",
    "aggregate_metric_dicts",
    "rank_stability",
    "rank_stability_by_metric",
    "BOOTSTRAP_RESAMPLES",
    "CI_LEVEL",
    "MIN_SUFFICIENT_N",
    "AGG_FIELDS",
    "METRIC_DIRECTIONS",
]

_logger = get_logger(__name__)

# Fixed, documented constants so the aggregate is deterministic & config-free.
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95
MIN_SUFFICIENT_N = 3
AGG_FIELDS = ("mean", "std", "n", "ci_low", "ci_high", "ci_method", "sufficient_n")

# Salt mixed into the bootstrap seed derivation; bumping it changes every CI.
_BOOTSTRAP_SALT = "aggregate-bootstrap-h1"


def _bootstrap_seed(values: Sequence[float], key: str) -> int:
    """Derive a deterministic 128-bit bootstrap seed from the values + key."""
    digest = canonical_hash(
        {"values": [float(v) for v in values], "key": key, "salt": _BOOTSTRAP_SALT}
    )
    return int(digest, 16) % (2 ** 128)


def aggregate_values(values: Sequence[float], key: str) -> Dict[str, Any]:
    """Aggregate per-seed ``values`` for metric ``key`` into mean/std/CI.

    Returns a dict with exactly :data:`AGG_FIELDS`. For ``n == 0`` returns a
    zeroed degenerate aggregate; for ``1 <= n < MIN_SUFFICIENT_N`` returns a
    degenerate CI (``ci_low == mean == ci_high``) flagged ``sufficient_n=False``;
    for ``n >= MIN_SUFFICIENT_N`` computes a seeded-bootstrap percentile CI.
    """
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    n = int(arr.size)

    if n == 0:
        return {
            "mean": 0.0, "std": 0.0, "n": 0,
            "ci_low": 0.0, "ci_high": 0.0,
            "ci_method": "degenerate", "sufficient_n": False,
        }

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0

    if n < MIN_SUFFICIENT_N:
        return {
            "mean": mean, "std": std, "n": n,
            "ci_low": mean, "ci_high": mean,
            "ci_method": "degenerate", "sufficient_n": False,
        }

    rng = np.random.default_rng(_bootstrap_seed(values, key))
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    resample_means = arr[idx].mean(axis=1)
    alpha = (1.0 - CI_LEVEL) / 2.0
    ci_low = float(np.quantile(resample_means, alpha))
    ci_high = float(np.quantile(resample_means, 1.0 - alpha))

    _logger.info(
        "집계 통계를 계산했습니다 (key=%s, n=%d, mean=%.6f, ci=[%.6f, %.6f])",
        key, n, mean, ci_low, ci_high,
    )
    return {
        "mean": mean, "std": std, "n": n,
        "ci_low": ci_low, "ci_high": ci_high,
        "ci_method": "bootstrap", "sufficient_n": True,
    }


def aggregate_metric_dicts(
    per_seed: List[Dict[str, Any]], keys: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Aggregate a list of per-seed metric dicts into per-key aggregates.

    For each metric ``key`` in ``keys`` collect the value from every per-seed dict
    (missing values are skipped) and aggregate via :func:`aggregate_values`.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        vals = [d[key] for d in per_seed if key in d and d[key] is not None]
        skipped = len(per_seed) - len(vals)
        if skipped > 0:
            _logger.warning(
                "집계 경고: key=%s, 누락된 시드 값 %d/%d 건을 건너뜀",
                key, skipped, len(per_seed),
            )
        out[key] = aggregate_values(vals, key)
    return out


# ---------------------------------------------------------------------------
# Rank stability across resampling units (Task D-013, TRD V-011/V-016/V-017).
# ---------------------------------------------------------------------------

# Ranking direction per reported benchmark metric: True = HIGHER value ranks
# better (rank 1), False = LOWER value ranks better. Directions for the seven
# trace-only utility metrics + strategy_sensitivity + regret_to_oracle follow
# docs/07_METRICS_AND_EVALUATION.md. Two diagnostics are direction-AMBIGUOUS in
# general (`cell_visit_gini`: lower = more even grid usage, but evenness is not
# universally the goal; `time_to_saturation`: lower = final cell coverage
# reached earlier, which can also mean early plateau) — for RANKING purposes we
# fix them as documented below and state the choice in docs/07. Metrics absent
# from this map default to higher-is-better in
# :func:`rank_stability_by_metric` (with a Korean warning log).
METRIC_DIRECTIONS: Dict[str, bool] = {
    "coordinate_coverage": True,    # more distinct cells visited
    "artifact_diversity": True,     # broader complexity-band spread
    "redundancy_rate": False,       # fewer repeated/near-duplicate rounds
    "round_coherence": True,        # smoother progression
    "coordinate_entropy": True,     # more uniform cell-visit distribution
    "cell_visit_gini": False,       # ambiguous; fixed: lower concentration ranks better
    "time_to_saturation": False,    # ambiguous; fixed: earlier saturation ranks better
    "strategy_sensitivity": True,   # responds more to controlled probe changes
    "regret_to_oracle": False,      # smaller shortfall vs the oracle reference
}

# Documented constants describing the (only) implemented ranking conventions.
_TIE_HANDLING = "average_rank"
_TOP_RANK_RULE = "strict_best_only"


def rank_stability(
    values_by_policy: Mapping[str, Sequence[float]],
    *,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Rank stability of policies across aligned resampling units.

    ``values_by_policy`` maps each policy name to an equal-length list of
    per-unit metric values, where unit ``i`` is aligned across policies (the
    same child seed / seed family / batch). Within each unit, policies are
    ranked by value — rank 1 = best given ``higher_is_better`` — and the
    distribution of those ranks is summarized per policy.

    Conventions (fixed, documented, deterministic):

    - **Ties** (exact float equality within a unit) receive the **average
      rank** of the positions they span (two policies tied for best both get
      rank 1.5), so each unit's ranks always sum to ``P*(P+1)/2``.
    - **Top-rank probability** counts only units where the policy is
      **strictly** best; a tie for best counts for nobody, so the top-rank
      probabilities across policies can sum to LESS than 1.0 (never more).
    - ``rank_distribution`` maps rank label -> unit COUNT (counts, not
      fractions; they sum to ``n_units`` per policy). Average ranks yield
      fractional labels, formatted minimally (``"1"``, ``"1.5"``).
    - No RNG anywhere: pure sorting, so identical input yields an identical
      output dict regardless of the input dict's insertion order.

    Returns::

        {
          "n_units": int, "n_policies": int,
          "higher_is_better": bool,
          "tie_handling": "average_rank", "top_rank_rule": "strict_best_only",
          "per_policy": {
            <policy>: {"top_rank_probability": float, "mean_rank": float,
                        "rank_distribution": {<rank>: count}, "n_units": int},
            ...
          },
        }

    Raises a Korean ``ValueError`` for an empty policy map, empty unit lists,
    or unequal list lengths (misaligned units must never be ranked silently).
    """
    policies = sorted(values_by_policy)
    if not policies:
        raise ValueError(
            "rank_stability: values_by_policy 가 비어 있습니다 "
            "(정책이 최소 1개 필요)"
        )

    lengths = {p: len(values_by_policy[p]) for p in policies}
    n_units = lengths[policies[0]]
    if any(length != n_units for length in lengths.values()):
        raise ValueError(
            f"rank_stability: 정책별 단위 값 길이가 서로 다릅니다 "
            f"(단위가 정렬되지 않음): {lengths}"
        )
    if n_units == 0:
        raise ValueError(
            "rank_stability: 단위 값 리스트가 비어 있습니다 (단위가 최소 1개 필요)"
        )

    values = {p: [float(v) for v in values_by_policy[p]] for p in policies}

    top_counts = {p: 0 for p in policies}
    rank_sums = {p: 0.0 for p in policies}
    rank_dist: Dict[str, Dict[str, int]] = {p: {} for p in policies}

    for i in range(n_units):
        # Direction-normalized score: larger score = better rank, always.
        score = {
            p: (values[p][i] if higher_is_better else -values[p][i])
            for p in policies
        }
        # Deterministic order: by score desc, then policy name (the name
        # tiebreak only fixes iteration order — tied scores share a rank).
        order = sorted(policies, key=lambda p: (-score[p], p))

        # Average rank over each tie group (exact float equality).
        pos = 0
        while pos < len(order):
            end = pos
            while end < len(order) and score[order[end]] == score[order[pos]]:
                end += 1
            avg_rank = (pos + 1 + end) / 2.0  # ranks pos+1 .. end
            label = f"{avg_rank:g}"
            for idx in range(pos, end):
                p = order[idx]
                rank_sums[p] += avg_rank
                rank_dist[p][label] = rank_dist[p].get(label, 0) + 1
            pos = end

        # Strictly best only: a tie for the top counts for nobody.
        best = order[0]
        if len(order) == 1 or score[order[1]] != score[best]:
            top_counts[best] += 1

    per_policy = {
        p: {
            "top_rank_probability": top_counts[p] / n_units,
            "mean_rank": rank_sums[p] / n_units,
            "rank_distribution": {
                label: rank_dist[p][label]
                for label in sorted(rank_dist[p], key=float)
            },
            "n_units": n_units,
        }
        for p in policies
    }
    return {
        "n_units": n_units,
        "n_policies": len(policies),
        "higher_is_better": bool(higher_is_better),
        "tie_handling": _TIE_HANDLING,
        "top_rank_rule": _TOP_RANK_RULE,
        "per_policy": per_policy,
    }


def rank_stability_by_metric(
    per_metric_values: Mapping[str, Mapping[str, Sequence[float]]],
) -> Dict[str, Any]:
    """Apply :func:`rank_stability` per metric with the documented direction.

    ``per_metric_values`` maps metric key -> (policy -> per-unit values). Each
    metric's ranking direction comes from :data:`METRIC_DIRECTIONS`; unmapped
    metrics default to higher-is-better with a Korean warning log. Each output
    block additionally carries ``metric`` and a human-readable ``direction``
    (``"higher_is_better"`` / ``"lower_is_better"``). Deterministic.
    """
    out: Dict[str, Any] = {}
    for key, values_by_policy in per_metric_values.items():
        if key in METRIC_DIRECTIONS:
            hib = METRIC_DIRECTIONS[key]
        else:
            hib = True
            _logger.warning(
                "순위 안정성: metric=%s 의 순위 방향이 METRIC_DIRECTIONS 에 "
                "정의되어 있지 않아 기본값 higher_is_better=True 를 적용합니다",
                key,
            )
        block = rank_stability(values_by_policy, higher_is_better=hib)
        block["metric"] = key
        block["direction"] = "higher_is_better" if hib else "lower_is_better"
        out[key] = block

    _logger.info(
        "순위 안정성 블록을 계산했습니다 (metrics=%d, policies=%d, units=%d)",
        len(out),
        next(iter(out.values()))["n_policies"] if out else 0,
        next(iter(out.values()))["n_units"] if out else 0,
    )
    return out
