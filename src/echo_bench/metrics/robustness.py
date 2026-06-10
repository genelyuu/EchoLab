"""Robustness metrics: controlled fault transforms + sensitivity (Task D-003).

This module defines a small registry of **controlled, deterministic FAULT
TRANSFORMS** over a candidate pool (a list of card dicts) and a bounded
**sensitivity score** that compares a baseline metric dict against a faulted
metric dict for the same seed batch.

Scope and guardrails
====================
These are *system-level* robustness statistics over the controlled testbed only.
They quantify how much a policy's already-computed utility metrics move under an
injected, fully-specified fault — nothing about users, emotion, wellbeing,
privacy, or legal compliance, and they make **no** real-world robustness /
generalization claim. The fault transforms here are the controlled definitions
also consumed by the E-003 (E3) audit; the coordinate-scramble pattern in E-006
reuses the same pure-transform shape.

Fault transforms
================
Each fault is a **pure function** ``pool -> new_pool`` (the input pool is never
mutated) that is fully determined by its arguments (including an explicit
``seed`` where randomness is involved — a *local* ``random.Random`` only, never
the global RNG or wall-clock):

- :func:`pool_shrink` — drop a deterministic fraction of cards (distribution
  shift: fewer candidates per round).
- :func:`basis_dropout` — remove every card whose ``basis`` is in a dropped set
  (structural shift: a whole basis becomes unavailable).
- :func:`salience_perturb` — add a bounded deterministic perturbation to each
  card's ``salienceScore`` (measurement shift on the salience channel).

They are enumerated in the :data:`FAULTS` registry so the E3 audit can iterate
them by name.

Sensitivity score
=================
:func:`robustness_score` takes two metric dicts (the kind returned by
``echo_bench.metrics.utility.compute_all``) for the same seed batch and returns
a bounded sensitivity in ``[0.0, 1.0]``: the **mean absolute difference over the
shared numeric metric keys**. Because the utility metrics are each bounded to
``[0, 1]``, the per-key absolute difference is already in ``[0, 1]`` and so is
their mean. ``0.0`` means the fault left every shared metric unchanged (maximal
robustness); larger values mean greater sensitivity to the fault. The result
carries both ``traceHash`` references.

Determinism: every transform and the score are pure deterministic functions of
their inputs. Identifiers/keys stay English; runtime logs are Korean. Hashing is
delegated to :func:`echo_bench.utils.hash.canonical_hash`.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "pool_shrink",
    "basis_dropout",
    "salience_perturb",
    "FAULTS",
    "robustness_score",
    "sensitivity_score",
    "robustness_score_with_metadata",
    "ROBUSTNESS_DIRECTION",
    "SALIENCE_SCORE_MIN",
    "SALIENCE_SCORE_MAX",
]

_logger = get_logger(__name__)

# Explicit direction note (Tasks D-008, D-012). The score is a SENSITIVITY
# magnitude: larger means the fault moved the metrics more. Surfacing this
# verbatim in reports removes the "is high good or bad?" ambiguity.
# The phrase "0.0 = max robustness" is machine-readable and must be preserved
# verbatim; its presence is pinned by tests and by report self-description
# (D-012).  The claim validator does NOT scan this constant.
ROBUSTNESS_DIRECTION = (
    "0.0 = max robustness (fault changed no shared metric); "
    "higher = more sensitive = less robust"
)

# Bounds the salience channel is clamped to after perturbation, matching the
# card schema's salience range ([0, 1]).
SALIENCE_SCORE_MIN = 0.0
SALIENCE_SCORE_MAX = 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    if value <= lo:
        return lo
    if value >= hi:
        return hi
    return float(value)


def _card_key(card: Mapping[str, Any]) -> str:
    """Stable per-card key for deterministic ordering (cardId, else hash)."""
    cid = card.get("cardId")
    if cid is not None:
        return str(cid)
    return canonical_hash(dict(card))


def pool_shrink(
    pool: Sequence[Mapping[str, Any]], frac: float, seed: Any
) -> List[Dict[str, Any]]:
    """Deterministically drop a ``frac`` fraction of the pool.

    Keeps ``round((1 - frac) * len(pool))`` cards, selected by a *local* seeded
    RNG over the cards ordered by their stable :func:`_card_key`. Pure: the input
    ``pool`` is never mutated; each kept card is shallow-copied.

    Parameters
    ----------
    pool
        List of card dicts.
    frac
        Fraction to drop, clamped to ``[0.0, 1.0]``. ``0.0`` keeps the whole
        pool; ``1.0`` drops everything.
    seed
        Seed for the *local* ``random.Random`` (mixed via ``canonical_hash`` so
        it is platform-stable). The global RNG is never touched.

    Returns
    -------
    list[dict]
        A new list of kept card dicts, in their original pool order.
    """
    frac = _clamp(float(frac), 0.0, 1.0)
    n = len(pool)
    keep_n = round((1.0 - frac) * n)
    if keep_n >= n:
        return [dict(c) for c in pool]
    if keep_n <= 0:
        return []

    ordered_idx = sorted(range(n), key=lambda i: _card_key(pool[i]))
    rng = random.Random(int(canonical_hash({"seed": seed, "fault": "pool_shrink"}), 16))
    kept_idx = set(rng.sample(ordered_idx, keep_n))
    # Preserve original pool order among kept cards.
    return [dict(pool[i]) for i in range(n) if i in kept_idx]


def basis_dropout(
    pool: Sequence[Mapping[str, Any]], drop_basis: Any
) -> List[Dict[str, Any]]:
    """Remove every card whose observable ``basis`` is in ``drop_basis``.

    Structural shift: one or more whole bases become unavailable. Pure: the
    input pool is never mutated; surviving cards are shallow-copied.

    Parameters
    ----------
    pool
        List of card dicts.
    drop_basis
        A single basis label (str) or an iterable of basis labels to drop.

    Returns
    -------
    list[dict]
        Cards whose ``basis`` is not dropped, in original order.
    """
    if isinstance(drop_basis, str):
        dropped = {drop_basis}
    else:
        dropped = set(drop_basis)
    return [dict(c) for c in pool if c.get("basis") not in dropped]


def salience_perturb(
    pool: Sequence[Mapping[str, Any]], delta: float, seed: Any
) -> List[Dict[str, Any]]:
    """Add a bounded deterministic perturbation to each card's salience.

    Measurement shift on the salience channel: each card's ``salienceScore`` is
    shifted by a per-card value drawn from ``[-delta, +delta]`` by a *local*
    seeded RNG, then clamped to ``[SALIENCE_SCORE_MIN, SALIENCE_SCORE_MAX]``.
    Pure: the input pool is never mutated; each card is shallow-copied. The
    per-card draw is keyed by the card's stable key so the perturbation is
    reproducible regardless of pool order.

    Parameters
    ----------
    pool
        List of card dicts.
    delta
        Maximum absolute perturbation magnitude (clamped to ``>= 0``).
    seed
        Seed for the *local* ``random.Random``. The global RNG is never touched.

    Returns
    -------
    list[dict]
        New card dicts with perturbed, clamped ``salienceScore``.
    """
    delta = abs(float(delta))
    out: List[Dict[str, Any]] = []
    for card in pool:
        new_card = dict(card)
        base = float(card.get("salienceScore", 0.0))
        # Per-card deterministic draw, order-independent (keyed by card key).
        local_seed = canonical_hash(
            {"seed": seed, "fault": "salience_perturb", "card": _card_key(card)}
        )
        rng = random.Random(int(local_seed, 16))
        shift = rng.uniform(-delta, delta) if delta > 0.0 else 0.0
        new_card["salienceScore"] = _clamp(
            base + shift, SALIENCE_SCORE_MIN, SALIENCE_SCORE_MAX
        )
        out.append(new_card)
    return out


# Registry of controlled fault transforms, by English machine-read name. The E3
# audit (E-003) iterates these; each value is the pure transform callable.
FAULTS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "pool_shrink": pool_shrink,
    "basis_dropout": basis_dropout,
    "salience_perturb": salience_perturb,
}


def _shared_numeric_keys(
    baseline: Mapping[str, Any], faulted: Mapping[str, Any]
) -> List[str]:
    """Return sorted keys present in both dicts with numeric (non-bool) values."""
    shared = []
    for key in baseline.keys() & faulted.keys():
        bv = baseline[key]
        fv = faulted[key]
        if isinstance(bv, bool) or isinstance(fv, bool):
            continue
        if isinstance(bv, (int, float)) and isinstance(fv, (int, float)):
            shared.append(key)
    return sorted(shared)


def _effective_keys(
    baseline: Mapping[str, Any],
    faulted: Mapping[str, Any],
    keys: Optional[Tuple[str, ...]],
) -> List[str]:
    """Return the list of metric keys to compare in the sensitivity score.

    When ``keys`` is not ``None`` (pinned mode), returns only the keys from
    the supplied tuple that exist in both dicts with non-bool numeric values.
    The order follows the supplied ``keys`` tuple (i.e. ``CORE_METRIC_KEYS``
    order in the typical pinned call), **not** sorted.

    When ``keys`` is ``None`` (dynamic mode), delegates to
    :func:`_shared_numeric_keys`, which returns sorted keys.
    """
    if keys is not None:
        # Pinned mode: follow the supplied key order (CORE_METRIC_KEYS order).
        # Skip any key absent from either dict or with a non-numeric value
        # (defensive; the pinned key set should always be present in
        # compute_all dicts).
        effective: List[str] = []
        for k in keys:
            bv = baseline.get(k)
            fv = faulted.get(k)
            if bv is None or fv is None:
                continue
            if isinstance(bv, bool) or isinstance(fv, bool):
                continue
            if isinstance(bv, (int, float)) and isinstance(fv, (int, float)):
                effective.append(k)
        return effective
    return _shared_numeric_keys(baseline, faulted)


def robustness_score(
    baseline_metrics: Mapping[str, Any],
    faulted_metrics: Mapping[str, Any],
    keys: Optional[Tuple[str, ...]] = None,
) -> float:
    """Bounded sensitivity of metrics to a fault, in ``[0.0, 1.0]``.

    Computes the **mean absolute difference over the shared numeric metric
    keys** between two metric dicts (the kind returned by
    ``utility.compute_all``) for the same seed batch. Non-numeric keys (e.g.
    ``traceHash``) and booleans are ignored. Because each utility metric is
    bounded to ``[0, 1]``, the per-key absolute difference and thus their mean
    lie in ``[0, 1]``; the result is clamped defensively.

    ``0.0`` means the fault changed no shared metric (maximal robustness); a
    larger value means greater sensitivity. Returns ``0.0`` when there are no
    shared numeric keys. Deterministic: identical inputs -> identical value.

    This function is also exposed as :func:`sensitivity_score` — a
    clearly-named alias where the direction is immediately legible: higher =
    more sensitive = less robust. The ``ROBUSTNESS_DIRECTION`` constant states
    the direction verbatim. The report-level primary label for this score is
    ``sensitivity_score`` (with ``legacyAlias: "robustness_score"``).

    Parameters
    ----------
    baseline_metrics
        Metric dict from the baseline run (e.g. ``utility.compute_all``).
    faulted_metrics
        Metric dict from the faulted run.
    keys
        Optional explicit tuple of metric key names to include in the mean.
        When ``None`` (the default), the denominator is the full dynamic
        intersection of shared numeric keys — the pre-D-010 behaviour,
        unchanged for direct callers and tests. Pass
        ``utility.CORE_METRIC_KEYS`` from call sites that must pin to the
        original four utility keys to preserve comparability across the C-011
        freeze boundary (e.g. E3 robustness audit and S3 scramble sensitivity).
    """
    effective_keys = _effective_keys(baseline_metrics, faulted_metrics, keys)
    if not effective_keys:
        return 0.0
    total = 0.0
    for key in effective_keys:
        total += abs(float(baseline_metrics[key]) - float(faulted_metrics[key]))
    value = _clamp(total / len(effective_keys), 0.0, 1.0)
    _logger.info(
        "민감도 점수(sensitivity_score, legacy robustness_score)를 계산했습니다 "
        "(shared_keys=%d, pinned=%s, value=%.6f)",
        len(effective_keys),
        keys is not None,
        value,
    )
    return value


def sensitivity_score(
    baseline_metrics: Mapping[str, Any],
    faulted_metrics: Mapping[str, Any],
    keys: Optional[Tuple[str, ...]] = None,
) -> float:
    """Unambiguously-named alias of :func:`robustness_score` (Tasks D-008, D-012).

    Returns the IDENTICAL value as :func:`robustness_score` — this is the
    *primary report label* so papers and artifacts read "sensitivity_score"
    (where **higher = more sensitive = less robust**) without readers
    second-guessing the direction. It must never diverge from
    :func:`robustness_score`. The direction is stated verbatim in
    :data:`ROBUSTNESS_DIRECTION`, which contains the machine-readable phrase
    ``"0.0 = max robustness"``.

    Naming relationship: ``sensitivity_score`` is the report-level primary
    label; ``robustness_score`` is the legacy code-level name and is recorded
    in reports as ``legacyAlias: "robustness_score"`` for backward
    compatibility.

    The optional ``keys`` parameter is passed through to :func:`robustness_score`
    unchanged (see that function's documentation).
    """
    return robustness_score(baseline_metrics, faulted_metrics, keys=keys)


def robustness_score_with_metadata(
    baseline_metrics: Mapping[str, Any],
    faulted_metrics: Mapping[str, Any],
    keys: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Compute :func:`robustness_score` and carry both ``traceHash`` references.

    Returns::

        {
            "metric": "sensitivity_score",         # D-012: primary report label
            "legacyAlias": "robustness_score",      # backward-compatibility alias
            "value": float in [0, 1],
            "sensitivityScore": float in [0, 1],   # == value, clearer name (D-008)
            "direction": ROBUSTNESS_DIRECTION,     # contains "0.0 = max robustness"
            "baselineTraceHash": baseline_metrics.get("traceHash"),
            "faultedTraceHash":  faulted_metrics.get("traceHash"),
            "sharedKeys": [...],   # the numeric keys compared
            "metricKeys": [...],   # the pinned keys used, or None if dynamic
        }

    In pinned mode (``keys`` not None), ``sharedKeys`` follows the order of the
    supplied ``keys`` tuple (e.g. ``CORE_METRIC_KEYS`` order) — it is **not**
    sorted. In dynamic mode (``keys`` is None) it is sorted.

    Deterministic: identical inputs -> identical dict. The ``sensitivityScore``
    and ``direction`` fields (Task D-008) are additive clarity only — ``value``
    is unchanged.

    The optional ``keys`` parameter is passed through to :func:`robustness_score`.
    When supplied (e.g. ``utility.CORE_METRIC_KEYS``), ``metricKeys`` in the
    returned dict records the pinned key list, making the report self-describing
    about which keys were used in the denominator.
    """
    effective_keys = _effective_keys(baseline_metrics, faulted_metrics, keys)
    value = robustness_score(baseline_metrics, faulted_metrics, keys=keys)
    return {
        "metric": "sensitivity_score",
        "legacyAlias": "robustness_score",
        "value": value,
        "sensitivityScore": value,
        "direction": ROBUSTNESS_DIRECTION,
        "baselineTraceHash": baseline_metrics.get("traceHash"),
        "faultedTraceHash": faulted_metrics.get("traceHash"),
        "sharedKeys": effective_keys,
        "metricKeys": list(keys) if keys is not None else None,
    }
