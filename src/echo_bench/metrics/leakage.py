"""Leakage **proxy** metric over observable slates/selections (Task D-002).

WHAT THIS IS — AND IS NOT
=========================
:func:`leakage_proxy` is a **PROXY**, **NOT** a privacy guarantee, anonymity
proof, identifiability bound, or legal/compliance claim. It is a *system-level*
statistic that measures, over the controlled testbed only, how much a policy's
observable slate/selection distribution **co-varies with the controlled probe
identity** (the B-004 strategy probes — instrumented *input policies*, not
synthetic users). A higher value means the policy's surfaced/selected behaviour
is more separable by which controlled input strategy drove it; a lower value
means the observable behaviour is harder to distinguish across probes.

This number must never be interpreted as "the system protects privacy",
"users cannot be identified", or "this is GDPR-compliant". Those framings are
forbidden by the project guardrails. The metric reads
**only** observable fields — ``slate`` and ``selectedCardId`` from the trace
rounds, plus the probe *name/version* used as a label — and never reads any
latent / user / persona / emotion / preference field (the trace schema has none;
this module additionally refuses to look for one).

DEFINITION
==========
Given a mapping ``probeName -> TraceState`` (the trace a single policy produced
under each controlled probe), we build, per probe, an empirical distribution
over an observable *selection signature*: the ``selectedCardId`` of each round,
combined with that round's slate membership (the set of slate card ids). We then
quantify how separable these per-probe distributions are using the **normalized
mutual information** between the probe-identity label and the observed selection
signature, computed over the pooled rounds.

- ``0.0`` — the selection-signature distribution is identical across probes:
  knowing the signature tells you nothing about which probe produced it (no
  observable co-variation / no proxy leakage).
- ``1.0`` — the signature perfectly determines the probe (and vice versa):
  maximal observable co-variation.

Mutual information ``I(P; S)`` is normalized by ``min(H(P), H(S))`` so the value
is bounded to ``[0.0, 1.0]`` and comparable across probe/signature cardinalities.
With fewer than two probes, or no rounds, there is nothing to separate and the
proxy is ``0.0``.

DETERMINISM
===========
The metric is a pure deterministic function of the observable inputs: identical
``traces_by_probe`` yields an identical value and identical metadata. Probes are
iterated in a stable name-sorted order; counts use canonical signatures. No RNG,
wall-clock, or process entropy enters the computation.

Identifiers, keys, and metric names stay English; runtime log lines are Korean
per the project logging convention. Hashing is delegated to
:func:`echo_bench.utils.hash.canonical_hash`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "leakage_proxy",
    "leakage_proxy_with_metadata",
    "METRIC_NAME",
    "IS_PROXY",
    "PROXY_DISCLAIMER",
]

_logger = get_logger(__name__)

#: English machine-read metric name.
METRIC_NAME = "leakage_proxy"

#: Hard flag stating this metric is a proxy, surfaced in returned metadata so no
#: downstream consumer can mistake it for a guarantee.
IS_PROXY = True

#: Human-readable disclaimer carried in the metric metadata.
PROXY_DISCLAIMER = (
    "leakage_proxy is a SYSTEM-LEVEL PROXY measuring how observable "
    "slate/selection distributions co-vary with controlled probe identity in "
    "this controlled testbed. It is NOT a privacy guarantee, anonymity proof, "
    "identifiability bound, or legal/compliance claim, and makes no real-world "
    "generalization claim."
)

# Fields the metric is permitted to read from a round record. Reading anything
# outside this set (in particular any latent/user field) is a contract
# violation; tests assert the implementation honours it.
_OBSERVABLE_ROUND_FIELDS = ("slate", "selectedCardId")


def _selection_signature(record: Mapping[str, Any]) -> str:
    """Build a canonical observable selection signature for one round.

    The signature combines only observable fields: the round's
    ``selectedCardId`` and the *set* of card ids present in its ``slate``
    (order-independent). It deliberately reads no other field. The signature is
    a stable canonical hash so equal observable rounds map to an equal label.
    """
    slate = record.get("slate") or []
    # Slate entries may be card-id strings or card dicts; reduce to ids,
    # order-independent (a set), so signature is permutation-invariant.
    slate_ids = []
    for entry in slate:
        if isinstance(entry, Mapping):
            slate_ids.append(entry.get("cardId"))
        else:
            slate_ids.append(entry)
    sig = {
        "selectedCardId": record.get("selectedCardId"),
        "slateMembers": sorted(str(s) for s in slate_ids),
    }
    return canonical_hash(sig)


def _probe_version(trace: Any) -> Any:
    """Return a probe's ``probeVersion`` if the value exposes one, else ``None``.

    ``traces_by_probe`` values are TraceStates (no probe version on them), so the
    caller may instead pass a ``probe_versions`` mapping to
    :func:`leakage_proxy_with_metadata`. This helper is defensive only.
    """
    version_fn = getattr(trace, "probe_version", None)
    if callable(version_fn):
        try:
            return version_fn()
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _collect_labeled_signatures(
    traces_by_probe: Mapping[str, Any]
) -> List[Tuple[str, str]]:
    """Return ``(probeName, selectionSignature)`` pairs over all pooled rounds.

    Reads **only** the observable ``slate``/``selectedCardId`` fields of each
    round via :func:`_selection_signature`. Probes are visited in name-sorted
    order so the pooled sequence is deterministic.
    """
    pairs: List[Tuple[str, str]] = []
    for name in sorted(traces_by_probe):
        trace = traces_by_probe[name]
        rounds_fn = getattr(trace, "rounds", None)
        if not callable(rounds_fn):
            continue
        for record in rounds_fn():
            pairs.append((name, _selection_signature(record)))
    return pairs


def _entropy(counts: Mapping[Any, int], total: int) -> float:
    """Shannon entropy (natural log) of a count distribution."""
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def _normalized_mutual_information(pairs: List[Tuple[str, str]]) -> float:
    """Normalized mutual information between probe label and signature.

    ``I(P; S) / min(H(P), H(S))`` over the pooled ``(probe, signature)`` pairs,
    clamped to ``[0.0, 1.0]``. Returns ``0.0`` when either marginal is
    degenerate (a single probe or a single signature: nothing to separate).
    """
    total = len(pairs)
    if total == 0:
        return 0.0

    p_counts: Dict[str, int] = {}
    s_counts: Dict[str, int] = {}
    joint_counts: Dict[Tuple[str, str], int] = {}
    for probe, sig in pairs:
        p_counts[probe] = p_counts.get(probe, 0) + 1
        s_counts[sig] = s_counts.get(sig, 0) + 1
        joint_counts[(probe, sig)] = joint_counts.get((probe, sig), 0) + 1

    if len(p_counts) < 2 or len(s_counts) < 2:
        return 0.0

    h_p = _entropy(p_counts, total)
    h_s = _entropy(s_counts, total)
    denom = min(h_p, h_s)
    if denom <= 0.0:
        return 0.0

    mi = 0.0
    for (probe, sig), c in joint_counts.items():
        p_joint = c / total
        p_p = p_counts[probe] / total
        p_s = s_counts[sig] / total
        mi += p_joint * math.log(p_joint / (p_p * p_s))

    value = mi / denom
    # Clamp to guard against tiny floating-point overshoot.
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def leakage_proxy(traces_by_probe: Mapping[str, Any]) -> float:
    """Deterministic leakage **PROXY** in ``[0.0, 1.0]`` (see module docstring).

    NOT a privacy/identifiability/legal guarantee — see :data:`PROXY_DISCLAIMER`.

    Parameters
    ----------
    traces_by_probe
        Maps ``probeName`` (str) -> ``TraceState`` (the trace a single policy
        produced under that controlled probe). Only the observable ``slate`` and
        ``selectedCardId`` of each round are read.

    Returns
    -------
    float
        Normalized mutual information between probe identity and the observable
        selection signature, in ``[0.0, 1.0]``. ``0.0`` when fewer than two
        probes (or no rounds) are supplied.
    """
    pairs = _collect_labeled_signatures(traces_by_probe)
    value = _normalized_mutual_information(pairs)
    _logger.info(
        "누출 프록시(leakage_proxy)를 계산했습니다 (probes=%d, rounds=%d, "
        "value=%.6f) — 이는 PROXY 이며 프라이버시/법적 보증이 아닙니다",
        len(traces_by_probe),
        len(pairs),
        value,
    )
    return value


def leakage_proxy_with_metadata(
    traces_by_probe: Mapping[str, Any],
    probe_versions: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute :func:`leakage_proxy` and return it with carried metadata.

    The returned dict carries the proxy value, the ``traceHash`` of every probe
    trace it was computed over (name-sorted), the ``probeVersion`` of each probe
    (from ``probe_versions`` when supplied, else from the trace if it exposes
    one, else ``None``), and the explicit proxy flag/disclaimer so no downstream
    consumer can mistake the value for a guarantee::

        {
            "metric": "leakage_proxy",
            "value": float in [0, 1],
            "isProxy": True,
            "disclaimer": PROXY_DISCLAIMER,
            "traceHashes": {probeName: traceHash, ...},
            "probeVersions": {probeName: probeVersion|None, ...},
        }

    Deterministic: identical inputs -> identical dict.
    """
    versions = dict(probe_versions or {})
    trace_hashes: Dict[str, Any] = {}
    probe_version_map: Dict[str, Any] = {}
    for name in sorted(traces_by_probe):
        trace = traces_by_probe[name]
        trace_hash_fn = getattr(trace, "trace_hash", None)
        trace_hashes[name] = trace_hash_fn() if callable(trace_hash_fn) else None
        if name in versions:
            probe_version_map[name] = versions[name]
        else:
            probe_version_map[name] = _probe_version(trace)

    return {
        "metric": METRIC_NAME,
        "value": leakage_proxy(traces_by_probe),
        "isProxy": IS_PROXY,
        "disclaimer": PROXY_DISCLAIMER,
        "traceHashes": trace_hashes,
        "probeVersions": probe_version_map,
    }
