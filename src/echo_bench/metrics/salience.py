"""Salience-audit metrics over observable salience scores (Task D-005).

These metrics detect whether a policy's slates **over-concentrate on
high-salience cards**, using only the observable ``salienceScore`` field already
recorded in each trace round and **configured** thresholds (loaded from
``configs/experiments/salience_audit.yaml`` — never hard-coded here). They drive
the S4 salience audit (E-007).

Guardrails
==========
``salienceScore`` is an objective image/structure statistic (A-006), not an
aesthetic, user-preference, emotion, or wellbeing judgment, and this audit
carries **no** privacy / legal-compliance framing. The metrics are *system-level*
statistics over the controlled testbed; they make no real-world generalization
claim.

Metrics
=======
- :func:`salience_outlier_rate` — fraction of slated/selected cards whose
  observable ``salienceScore`` strictly exceeds the configured
  ``outlier_threshold``. In ``[0, 1]``; ``0.0`` when no round has a salience
  score above threshold (or the trace is empty).
- :func:`salience_control` — gap from a configured **target** salience
  distribution: the observed per-band distribution of slated/selected salience
  scores is compared to the target via half the total-variation distance
  (``0.5 * L1``), in ``[0, 1]``; ``0.0`` is a perfect match to the target.

Both are deterministic functions of the trace's ``salienceScore`` fields and the
supplied config; identical traces + config -> identical values. Each returns
just the bounded float; the ``*_with_metadata`` variants carry the ``traceHash``.

Identifiers/keys stay English; runtime logs are Korean. Hashing is delegated to
the trace's own ``trace_hash``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from echo_bench.logging import get_logger

__all__ = [
    "salience_outlier_rate",
    "salience_control",
    "salience_outlier_rate_with_metadata",
    "salience_control_with_metadata",
    "load_salience_config",
    "SALIENCE_AUDIT_CONFIG_PATH",
]

_logger = get_logger(__name__)

# Repo-rooted default config location (resolved relative to the package, not the
# cwd) so the audit works regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
SALIENCE_AUDIT_CONFIG_PATH = (
    _REPO_ROOT / "configs" / "experiments" / "salience_audit.yaml"
)


def load_salience_config(path: Any = None) -> Dict[str, Any]:
    """Load the salience-audit thresholds config (YAML) into a dict.

    Thresholds live in config, never hard-coded. Defaults to
    :data:`SALIENCE_AUDIT_CONFIG_PATH` when ``path`` is ``None``.
    """
    cfg_path = Path(path) if path is not None else SALIENCE_AUDIT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return doc if isinstance(doc, dict) else {}


def _salience_scores(trace: Any) -> List[float]:
    """Return the observable ``salienceScore`` of every round, as floats.

    Reads only the ``salienceScore`` field of each round — never any latent /
    user field. Rounds with a missing/None salience score are skipped.
    """
    scores: List[float] = []
    rounds_fn = getattr(trace, "rounds", None)
    if not callable(rounds_fn):
        return scores
    for record in rounds_fn():
        val = record.get("salienceScore")
        if val is None:
            continue
        scores.append(float(val))
    return scores


def salience_outlier_rate(trace: Any, thresholds: Mapping[str, Any]) -> float:
    """Fraction of slated/selected cards over the configured salience threshold.

    A round is an *outlier* when its observable ``salienceScore`` strictly
    exceeds ``thresholds["outlier_threshold"]``. Returns the outlier fraction in
    ``[0, 1]`` (``0.0`` for an empty trace). Deterministic.

    Parameters
    ----------
    trace
        A ``TraceState`` (or any object with a ``rounds()`` method).
    thresholds
        Config mapping carrying ``outlier_threshold`` (e.g. from
        :func:`load_salience_config`).
    """
    if "outlier_threshold" not in thresholds:
        raise KeyError(
            "salience 설정에 'outlier_threshold' 키가 없습니다 "
            "(configs/experiments/salience_audit.yaml)"
        )
    threshold = float(thresholds["outlier_threshold"])
    scores = _salience_scores(trace)
    n = len(scores)
    if n == 0:
        return 0.0
    outliers = sum(1 for s in scores if s > threshold)
    value = outliers / n
    _logger.info(
        "salience_outlier_rate 를 계산했습니다 (threshold=%.4f, rounds=%d, "
        "outliers=%d, value=%.6f)",
        threshold,
        n,
        outliers,
        value,
    )
    return value


def _normalized_weights(weights: Sequence[float]) -> List[float]:
    """Normalize a non-negative weight vector to sum to 1 (uniform if all zero)."""
    floats = [max(0.0, float(w)) for w in weights]
    total = sum(floats)
    if total <= 0.0:
        n = len(floats)
        return [1.0 / n] * n if n else []
    return [w / total for w in floats]


def _band_index(score: float, bands: Sequence[Mapping[str, Any]]) -> int:
    """Return the index of the band containing ``score`` ([lo, hi), last closed).

    Scores below the first band's ``lo`` clamp into band 0; scores at/above the
    last band's ``hi`` clamp into the last band. Deterministic.
    """
    n = len(bands)
    for i, band in enumerate(bands):
        lo = float(band["lo"])
        hi = float(band["hi"])
        last = i == n - 1
        if score < lo:
            return 0
        if (lo <= score < hi) or (last and score <= hi):
            return i
    return n - 1


def salience_control(trace: Any, target_cfg: Mapping[str, Any]) -> float:
    """Gap from a configured target salience distribution, in ``[0, 1]``.

    The observed ``salienceScore`` values of all rounds are binned into the bands
    declared in ``target_cfg["target_distribution"]["bands"]`` and compared to
    the per-band ``weights`` (normalized) via **half the total-variation
    distance** (``0.5 * sum |observed_p - target_p|``). The result is in
    ``[0, 1]``: ``0.0`` is a perfect match to the target distribution; ``1.0`` is
    maximal divergence. An empty trace returns ``0.0`` (no observed mass to
    compare). Deterministic.

    Parameters
    ----------
    trace
        A ``TraceState`` (or any object with a ``rounds()`` method).
    target_cfg
        Config mapping carrying ``target_distribution`` with ``bands`` and
        ``weights`` (e.g. from :func:`load_salience_config`).
    """
    dist_cfg = target_cfg.get("target_distribution")
    if not isinstance(dist_cfg, Mapping):
        raise KeyError(
            "salience 설정에 'target_distribution' 키가 없습니다 "
            "(configs/experiments/salience_audit.yaml)"
        )
    bands = dist_cfg.get("bands")
    weights = dist_cfg.get("weights")
    if not bands or weights is None:
        raise KeyError(
            "target_distribution 에는 'bands' 와 'weights' 가 모두 필요합니다"
        )
    if len(bands) != len(weights):
        raise ValueError(
            "target_distribution 의 'bands' 와 'weights' 길이가 다릅니다 "
            f"(bands={len(bands)}, weights={len(weights)})"
        )

    target = _normalized_weights(weights)
    scores = _salience_scores(trace)
    n = len(scores)
    if n == 0:
        return 0.0

    observed_counts = [0] * len(bands)
    for s in scores:
        observed_counts[_band_index(s, bands)] += 1
    observed = [c / n for c in observed_counts]

    tv = 0.5 * sum(abs(o - t) for o, t in zip(observed, target))
    # Clamp defensively into [0, 1].
    if tv <= 0.0:
        value = 0.0
    elif tv >= 1.0:
        value = 1.0
    else:
        value = float(tv)
    _logger.info(
        "salience_control 를 계산했습니다 (bands=%d, rounds=%d, value=%.6f)",
        len(bands),
        n,
        value,
    )
    return value


def salience_outlier_rate_with_metadata(
    trace: Any, thresholds: Mapping[str, Any]
) -> Dict[str, Any]:
    """:func:`salience_outlier_rate` carrying the ``traceHash``."""
    trace_hash_fn = getattr(trace, "trace_hash", None)
    return {
        "metric": "salience_outlier_rate",
        "value": salience_outlier_rate(trace, thresholds),
        "traceHash": trace_hash_fn() if callable(trace_hash_fn) else None,
    }


def salience_control_with_metadata(
    trace: Any, target_cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """:func:`salience_control` carrying the ``traceHash``."""
    trace_hash_fn = getattr(trace, "trace_hash", None)
    return {
        "metric": "salience_control",
        "value": salience_control(trace, target_cfg),
        "traceHash": trace_hash_fn() if callable(trace_hash_fn) else None,
    }
