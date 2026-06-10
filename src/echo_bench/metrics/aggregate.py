"""Seed-batch aggregation statistics (Task D-006).

Pure, deterministic aggregation over a list of per-seed metric values: mean,
sample std (ddof=1), and a CONFIDENCE INTERVAL via a *seeded* bootstrap. The
bootstrap RNG is seeded from ``canonical_hash`` of the input values, so the CI is
a pure function of its inputs and replays bit-identically. No global RNG,
wall-clock, or process entropy ever enters the computation.

These are SYSTEM-LEVEL statistics over a controlled testbed. They are not
measures of user preference, emotion, wellbeing, privacy, or legal compliance and
make no real-world generalization claim.

Identifiers / keys stay English; runtime log lines are Korean per convention.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "aggregate_values",
    "aggregate_metric_dicts",
    "BOOTSTRAP_RESAMPLES",
    "CI_LEVEL",
    "MIN_SUFFICIENT_N",
    "AGG_FIELDS",
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
