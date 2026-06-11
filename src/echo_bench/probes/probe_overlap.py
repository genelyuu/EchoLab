"""Probe overlap diagnostics for ECHO-Bench (Task B-007 / TRD B-008).

GUARDRAIL
=========
Strategy probes are controlled, instrumented INPUT policies, not synthetic
users (see :mod:`echo_bench.probes.strategy_probes`). This module measures how
*redundant* the registered probes are with respect to each other — a probe set
whose members select the same cards almost everywhere cannot support a
meaningful probe-separability measurement (V-004 diagnosis). Nothing here
models, infers, or claims anything about people.

Definitions
===========
A **context** is one observable decision situation: a ``(slate, trace)`` pair
(``trace`` may be ``None`` or an empty :class:`~echo_bench.env.trace_state.TraceState`).
Each probe is evaluated with the same shared ``seed`` on every context.

``pairwise_probe_overlap``
    For each unordered probe pair ``(A, B)``: the fraction of contexts on which
    A and B select the *identical* ``cardId``. 1.0 = the pair is behaviourally
    indistinguishable on these contexts; 0.0 = they never agree.

``probe_entropy``
    For each probe: the Shannon entropy in bits of the probe's empirical
    selection distribution over ``cardId`` values across the contexts. 0.0
    means the probe selected one single card in every context (degenerate /
    constant behaviour on these contexts); higher values mean more varied
    selections (upper bound ``log2(n_contexts)``).

``high_overlap_pairs``
    Every pair whose overlap is **>= :data:`PROBE_OVERLAP_THRESHOLD`** (or the
    caller-supplied threshold). These are reported separately — flagged, not
    silently dropped — satisfying the TRD acceptance criterion that excessive
    overlap pairs are explicitly surfaced in reports.

``exclude_merge_candidates`` (Task B-009)
    The stricter second tier: every pair whose overlap is
    **>= :data:`PROBE_OVERLAP_EXCLUDE_THRESHOLD`** (or the caller-supplied
    ``exclude_threshold``). A flagged pair (>= 0.9) is a diagnostic caveat;
    an exclude-tier pair (>= 0.95) is additionally an exclude-or-merge
    candidate for probe-set curation. Both tiers are diagnostics, never
    claims, and by construction every exclude candidate is also flagged
    (the exclude threshold is validated to be >= the flag threshold).

The audit is a pure deterministic function of ``(contexts, probes, seed,
threshold, exclude_threshold)``. All output keys are English machine-read
identifiers; runtime log messages are Korean per the project logging
convention.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from echo_bench.logging import get_logger
from echo_bench.probes.strategy_probes import PROBES, StrategyProbe

__all__ = [
    "PROBE_OVERLAP_EXCLUDE_THRESHOLD",
    "PROBE_OVERLAP_THRESHOLD",
    "probe_overlap_audit",
]

_logger = get_logger(__name__)

#: Documented flagging threshold: a probe pair agreeing on >= this fraction of
#: contexts is reported in ``high_overlap_pairs`` (과다 중복 쌍 별도 표기).
#: The comparison is inclusive (``overlap >= threshold`` flags the pair).
PROBE_OVERLAP_THRESHOLD = 0.9

#: Documented exclude-or-merge threshold (Task B-009, second tier): a probe
#: pair agreeing on >= this fraction of contexts is additionally reported in
#: ``exclude_merge_candidates`` as an exclude-or-merge candidate for probe-set
#: curation. The comparison is inclusive (``overlap >= exclude_threshold``),
#: mirroring the flag tier, and the exclude threshold must always be >= the
#: flag threshold so exclude candidates are a subset of flagged pairs. Both
#: tiers are diagnostics, never claims.
PROBE_OVERLAP_EXCLUDE_THRESHOLD = 0.95

_ROUND = 12


def _resolve_probes(
    probes: Optional[Mapping[str, StrategyProbe]],
) -> Dict[str, StrategyProbe]:
    """Resolve the probe mapping under audit (defaults to the full registry)."""
    resolved = dict(PROBES) if probes is None else dict(probes)
    if len(resolved) < 2:
        raise ValueError(
            "프로브 overlap 진단에는 최소 2개의 프로브가 필요합니다 "
            f"(제공된 수: {len(resolved)})"
        )
    return resolved


def _selection_entropy_bits(selections: Sequence[str]) -> float:
    """Shannon entropy (bits) of the empirical selection distribution."""
    counts = Counter(selections)
    total = float(len(selections))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, _ROUND)


def probe_overlap_audit(
    contexts: Sequence[Tuple[Sequence[Mapping[str, Any]], Any]],
    probes: Optional[Mapping[str, StrategyProbe]] = None,
    seed: int = 0,
    threshold: float = PROBE_OVERLAP_THRESHOLD,
    exclude_threshold: float = PROBE_OVERLAP_EXCLUDE_THRESHOLD,
) -> Dict[str, Any]:
    """Audit pairwise selection overlap and selection entropy of a probe set.

    Args:
        contexts: sequence of ``(slate, trace)`` pairs — the observable decision
            situations over which the probes are compared. ``slate`` is a
            non-empty sequence of observable card dicts; ``trace`` may be
            ``None`` or a TraceState-like object exposing ``rounds()``.
        probes: mapping ``name -> StrategyProbe`` to audit; defaults to the full
            :data:`~echo_bench.probes.strategy_probes.PROBES` registry. At
            least two probes are required.
        seed: shared deterministic seed forwarded to every ``probe.select``.
        threshold: inclusive flagging threshold for ``high_overlap_pairs``
            (defaults to :data:`PROBE_OVERLAP_THRESHOLD`).
        exclude_threshold: inclusive exclude-or-merge threshold for
            ``exclude_merge_candidates`` (defaults to
            :data:`PROBE_OVERLAP_EXCLUDE_THRESHOLD`); must lie in ``[0, 1]``
            and be >= ``threshold``.

    Returns:
        A dict with English keys:

        - ``probes``: sorted probe names audited.
        - ``probe_versions``: ``{name: probeVersion}``.
        - ``n_contexts``: number of contexts evaluated.
        - ``threshold``: the flagging threshold used.
        - ``selections_by_probe``: ``{name: [selected cardId per context]}``.
        - ``pairwise_probe_overlap``: ``{"A|B": fraction}`` for every unordered
          pair with A < B lexicographically (identical-selection fraction).
        - ``probe_entropy``: ``{name: Shannon entropy in bits}`` of each
          probe's selection distribution over the contexts (see module
          docstring for the exact definition).
        - ``high_overlap_pairs``: list of ``{"probe_a", "probe_b", "overlap"}``
          dicts for every pair with ``overlap >= threshold``, sorted by pair
          key — the automatic excessive-overlap report.
        - ``exclude_threshold``: the exclude-or-merge threshold used (B-009).
        - ``exclude_merge_candidates``: list of ``{"probe_a", "probe_b",
          "overlap"}`` dicts for every pair with ``overlap >=
          exclude_threshold``, sorted like ``high_overlap_pairs``; always a
          subset of ``high_overlap_pairs``.
        - ``policyNote``: English machine-readable note documenting the
          two-tier policy (flag tier = diagnostic caveat, exclude tier =
          exclude-or-merge candidate; diagnostics, never claims).

    Raises:
        ValueError: (Korean message) on empty ``contexts``, fewer than two
            probes, a threshold outside ``[0, 1]``, or an
            ``exclude_threshold`` outside ``[0, 1]`` or below ``threshold``.
    """
    if not contexts:
        raise ValueError("프로브 overlap 진단에는 최소 1개의 컨텍스트가 필요합니다")
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(
            f"overlap 임계값은 [0, 1] 범위여야 합니다 (입력값: {threshold})"
        )
    if not (0.0 <= float(exclude_threshold) <= 1.0):
        raise ValueError(
            "overlap exclude 임계값은 [0, 1] 범위여야 합니다 "
            f"(입력값: {exclude_threshold})"
        )
    if float(exclude_threshold) < float(threshold):
        raise ValueError(
            "overlap exclude 임계값은 flag 임계값 이상이어야 합니다 "
            f"(flag={threshold}, exclude={exclude_threshold})"
        )
    probe_map = _resolve_probes(probes)
    names = sorted(probe_map)
    n_contexts = len(contexts)

    # Deterministic per-probe selections over the shared contexts.
    selections_by_probe: Dict[str, List[str]] = {}
    for name in names:
        probe = probe_map[name]
        selections_by_probe[name] = [
            probe.select(slate, trace, seed) for slate, trace in contexts
        ]

    # Pairwise identical-selection fraction. Both tiers use the same inclusive
    # comparison; exclude_threshold >= threshold (validated above) guarantees
    # exclude_merge_candidates is a subset of high_overlap_pairs.
    pairwise: Dict[str, float] = {}
    high_overlap_pairs: List[Dict[str, Any]] = []
    exclude_merge_candidates: List[Dict[str, Any]] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            sel_a = selections_by_probe[name_a]
            sel_b = selections_by_probe[name_b]
            agree = sum(1 for a, b in zip(sel_a, sel_b) if a == b)
            overlap = round(agree / n_contexts, _ROUND)
            pair_key = f"{name_a}|{name_b}"
            pairwise[pair_key] = overlap
            if overlap >= threshold:
                high_overlap_pairs.append(
                    {"probe_a": name_a, "probe_b": name_b, "overlap": overlap}
                )
            if overlap >= exclude_threshold:
                exclude_merge_candidates.append(
                    {"probe_a": name_a, "probe_b": name_b, "overlap": overlap}
                )

    probe_entropy = {
        name: _selection_entropy_bits(selections_by_probe[name])
        for name in names
    }

    if exclude_merge_candidates:
        _logger.warning(
            "프로브 overlap 제외/병합(exclude-or-merge) 후보 쌍 감지 "
            "(exclude_threshold=%s): %s — 프로브 셋 큐레이션 검토 대상입니다 "
            "(진단이며 클레임이 아닙니다)",
            exclude_threshold,
            [f"{p['probe_a']}|{p['probe_b']}={p['overlap']}"
             for p in exclude_merge_candidates],
        )
    if high_overlap_pairs:
        _logger.warning(
            "프로브 overlap 과다 쌍 감지 (threshold=%s): %s — 보고서에 별도 "
            "표기됩니다",
            threshold,
            [f"{p['probe_a']}|{p['probe_b']}={p['overlap']}"
             for p in high_overlap_pairs],
        )
    else:
        _logger.info(
            "프로브 overlap 진단 완료: 과다 쌍 없음 (threshold=%s, 쌍 수=%d, "
            "컨텍스트 수=%d)",
            threshold,
            len(pairwise),
            n_contexts,
        )

    policy_note = (
        "Two-tier overlap policy (B-009): a pair with overlap >= "
        f"{float(threshold)} is flagged in high_overlap_pairs as a diagnostic "
        f"caveat; a pair with overlap >= {float(exclude_threshold)} is "
        "additionally listed in exclude_merge_candidates as an "
        "exclude-or-merge candidate for probe-set curation. Both tiers are "
        "inclusive comparisons over instrumented input policies; flags are "
        "diagnostics, never claims."
    )

    return {
        "probes": names,
        "probe_versions": {n: probe_map[n].probe_version() for n in names},
        "n_contexts": n_contexts,
        "threshold": float(threshold),
        "exclude_threshold": float(exclude_threshold),
        "selections_by_probe": selections_by_probe,
        "pairwise_probe_overlap": pairwise,
        "probe_entropy": probe_entropy,
        "high_overlap_pairs": high_overlap_pairs,
        "exclude_merge_candidates": exclude_merge_candidates,
        "policyNote": policy_note,
    }
