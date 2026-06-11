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
import random
from typing import Any, Dict, List, Mapping, Tuple

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "leakage_proxy",
    "leakage_proxy_with_metadata",
    "null_corrected_separability",
    "leakage_delta_vs_random",
    "utility_per_leakage",
    "METRIC_NAME",
    "SEPARABILITY_METRIC_NAME",
    "IS_PROXY",
    "PROXY_DISCLAIMER",
    "LEAKAGE_RATIO_FLOOR",
    "DEFAULT_NULL_PERMUTATIONS",
    "NULL_STD_EPS",
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

# D-015: English machine-read metric name for the null-corrected statistic.
SEPARABILITY_METRIC_NAME = "null_corrected_separability"

# D-015: default number of deterministic label permutations for the null.
DEFAULT_NULL_PERMUTATIONS = 200

# D-015: documented near-zero threshold for the null standard deviation.
# CONVENTION (bounded, never +/-inf or NaN): when ``null_std < NULL_STD_EPS``
# the permutation null is (numerically) constant — every sampled label
# permutation yields the same NMI — so ``excess_z`` is defined as exactly
# ``0.0``. (In this regime ``excess_nmi`` is typically ~0 as well, e.g. in the
# fully saturated all-unique-signature case where every labeling gives NMI 1.0,
# but a sampled null cannot logically guarantee observed == null, so only
# ``excess_z`` carries the hard convention.)
NULL_STD_EPS = 1e-12

# D-011 (TRD alias D-010): denominator floor for :func:`utility_per_leakage`.
# A leakage value below this floor is replaced by the floor before dividing, so
# a near-zero leakage cannot explode the ratio. Documented constant — every E3
# report section that carries the ratio also records this floor (``ratioFloor``)
# so the reported number is self-describing.
LEAKAGE_RATIO_FLOOR = 0.05

# Fields the metric is permitted to read from a round record. Reading anything
# outside this set (in particular any latent/user field) is a contract
# violation; tests assert the implementation honours it.
_OBSERVABLE_ROUND_FIELDS = ("slate", "selectedCardId")


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval ``[0.0, 1.0]``."""
    v = float(value)
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def leakage_delta_vs_random(
    policy_leakage: float, random_leakage: float
) -> float:
    """RELATIVE leakage-proxy delta of a policy vs the RANDOM reference (D-011).

    ``leakage_delta_vs_random = leakage(policy) - leakage(RANDOM)``, computed
    over seed-aligned runs (same base seed / horizon / pool / probes for both
    policies). Both inputs are :func:`leakage_proxy` values and are clamped into
    ``[0.0, 1.0]`` before differencing, so the delta is bounded to
    ``[-1.0, 1.0]``.

    This is a **relative, comparison-ready** statistic — the supported claim is
    "policy X's leakage proxy is lower/higher than RANDOM's under identical
    controlled conditions", never an absolute leakage level. A negative delta
    means the policy's observable behaviour is LESS separable by controlled
    probe identity than the RANDOM reference's; positive means MORE separable.
    The reference compared with itself yields exactly ``0.0``.

    Both operands are PROXY values (see :data:`PROXY_DISCLAIMER`): the delta is
    likewise a proxy statistic over the controlled testbed and is NOT a privacy
    guarantee, anonymity proof, identifiability bound, or legal/compliance
    claim.
    """
    return _clamp01(policy_leakage) - _clamp01(random_leakage)


def utility_per_leakage(
    mean_utility: float,
    leakage: float,
    floor: float = LEAKAGE_RATIO_FLOOR,
) -> float:
    """Descriptive utility/leakage trade-off ratio for one policy (D-011).

    ``utility_per_leakage = mean_utility / max(leakage, floor)`` where
    ``mean_utility`` is e.g. the mean ``coordinate_coverage`` over the
    policy's per-probe traces (the caller supplies any ``[0,1]``-bounded
    utility aggregate) and ``leakage`` is its :func:`leakage_proxy` value.
    Note: E3 pools per-probe traces at ONE base seed — there are no multiple
    seed-aligned runs to average; the numerator is the per-probe trace mean
    over the probe family at that base seed.
    Both numerator and denominator inputs are clamped into ``[0.0, 1.0]``
    before the ratio; the denominator is then floored at
    :data:`LEAKAGE_RATIO_FLOOR` (default ``0.05``) so a near-zero leakage proxy
    cannot explode the ratio. Bounds: ``[0.0, 1.0 / floor]`` (``[0.0, 20.0]``
    at the default floor); continuous at the floor.

    Higher = a better observed utility-per-leakage trade-off **within the
    controlled testbed**. This is a descriptive, relative ranking aid only: the
    denominator is a PROXY (see :data:`PROXY_DISCLAIMER`), so the ratio
    inherits every proxy limitation and is NOT a privacy guarantee or any
    absolute leakage/utility claim. Report sections carrying this ratio must
    also record the floor (``ratioFloor``) so the number is self-describing.

    Raises ``ValueError`` if ``floor`` is not strictly positive.
    """
    if floor <= 0.0:
        raise ValueError(
            f"utility_per_leakage: floor 는 양수여야 합니다 (받은 값: {floor!r})"
        )
    return _clamp01(mean_utility) / max(_clamp01(leakage), float(floor))


def _slate_member_ids(record: Mapping[str, Any]) -> List[str]:
    """Name-sorted observable slate member ids of one round.

    Slate entries may be card-id strings or card dicts; reduce to ids and sort
    the string forms so the result is independent of stored slate order.
    Reads only the observable ``slate`` field. Shared by the legacy combined
    signature below and the D-016 channel signatures in
    :mod:`echo_bench.metrics.separability`.
    """
    slate = record.get("slate") or []
    slate_ids = []
    for entry in slate:
        if isinstance(entry, Mapping):
            slate_ids.append(entry.get("cardId"))
        else:
            slate_ids.append(entry)
    return sorted(str(s) for s in slate_ids)


def _selection_signature(record: Mapping[str, Any]) -> str:
    """Build a canonical observable selection signature for one round.

    The signature combines only observable fields: the round's
    ``selectedCardId`` and the *set* of card ids present in its ``slate``
    (order-independent). It deliberately reads no other field. The signature is
    a stable canonical hash so equal observable rounds map to an equal label.

    D-016 NOTE: this is the **combined** channel signature — byte-identical
    construction is an invariant relied upon by
    :func:`echo_bench.metrics.separability.channel_separated_separability`
    (its ``combined`` channel must reproduce legacy values exactly).
    """
    sig = {
        "selectedCardId": record.get("selectedCardId"),
        "slateMembers": _slate_member_ids(record),
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
    traces_by_probe: Mapping[str, Any],
    signature_fn: Any = None,
) -> List[Tuple[str, str]]:
    """Return ``(probeName, signature)`` pairs over all pooled rounds.

    ``signature_fn`` maps one round record to its canonical signature string
    and defaults to the legacy combined :func:`_selection_signature`; D-016
    passes per-channel signature functions. Every supported signature function
    reads **only** the observable ``slate``/``selectedCardId`` fields. Probes
    are visited in name-sorted order so the pooled sequence is deterministic.
    """
    if signature_fn is None:
        signature_fn = _selection_signature
    pairs: List[Tuple[str, str]] = []
    for name in sorted(traces_by_probe):
        trace = traces_by_probe[name]
        rounds_fn = getattr(trace, "rounds", None)
        if not callable(rounds_fn):
            continue
        for record in rounds_fn():
            pairs.append((name, signature_fn(record)))
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


def _null_corrected_stats(
    pairs: List[Tuple[str, str]],
    n_permutations: int,
    seed_metric: str,
) -> Dict[str, Any]:
    """Core observed-vs-permutation-null statistics over labeled pairs.

    Shared machinery for :func:`null_corrected_separability` (D-015) and the
    channel-separated variant in :mod:`echo_bench.metrics.separability`
    (D-016): given pooled ``(label, signature)`` pairs, compute the observed
    pooled NMI, a deterministic permutation null over the same signature
    multiset, and the excess statistics.

    DETERMINISM: the permutation RNG is a ``random.Random`` seeded **only**
    from ``seed_metric`` + the pooled pairs + ``n_permutations`` via
    ``canonical_hash`` — no wall-clock, process entropy, or global RNG state.
    Different ``seed_metric`` strings (e.g. per-channel names) intentionally
    decorrelate the null samples of otherwise-identical pair sequences.

    DEGENERACY (fail closed, no permutations executed): no pairs, fewer than
    two distinct labels, or fewer than two distinct signatures — matching
    :func:`_normalized_mutual_information`'s degeneracy rule (a single unique
    signature always yields NMI 0 under every labeling, so the permutation
    loop would be pointless).

    Raises ``ValueError`` if ``n_permutations < 1``.
    """
    if n_permutations < 1:
        raise ValueError(
            f"{seed_metric}: n_permutations 는 1 이상이어야 "
            f"합니다 (받은 값: {n_permutations!r})"
        )

    labels = [probe for probe, _sig in pairs]
    signatures = [sig for _probe, sig in pairs]
    observed = _normalized_mutual_information(pairs)
    degenerate = (
        len(pairs) == 0
        or len(set(labels)) < 2
        or len(set(signatures)) < 2
    )

    if degenerate:
        # In every degenerate case ``observed`` is already 0.0 (the NMI is
        # defined as 0 for a degenerate marginal) — the zero dict is exact.
        null_mean = 0.0
        null_std = 0.0
        excess_nmi = 0.0
        excess_z = 0.0
    else:
        # Deterministic, data-derived permutation seed: canonical hash of the
        # pooled labeled-signature sequence (already name-sorted) plus the
        # metric/channel name and permutation count. NO wall-clock / global RNG.
        seed = int(
            canonical_hash(
                {
                    "metric": seed_metric,
                    "pairs": [[probe, sig] for probe, sig in pairs],
                    "nPermutations": int(n_permutations),
                }
            ),
            16,
        )
        rng = random.Random(seed)
        permuted = list(labels)
        null_values: List[float] = []
        for _ in range(int(n_permutations)):
            rng.shuffle(permuted)
            null_values.append(
                _normalized_mutual_information(list(zip(permuted, signatures)))
            )
        null_mean = sum(null_values) / len(null_values)
        null_var = sum((v - null_mean) ** 2 for v in null_values) / len(
            null_values
        )
        null_std = math.sqrt(null_var)
        excess_nmi = observed - null_mean
        if null_std < NULL_STD_EPS:
            # Documented bounded convention: constant null -> zero excess z.
            excess_z = 0.0
        else:
            excess_z = excess_nmi / null_std

    return {
        "observed_nmi": float(observed),
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "excess_nmi": float(excess_nmi),
        "excess_z": float(excess_z),
        "n_permutations": int(n_permutations),
        "degenerate": degenerate,
    }


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


def null_corrected_separability(
    traces_by_probe: Mapping[str, Any],
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
) -> Dict[str, Any]:
    """Null-corrected probe-separability **PROXY** (Task D-015, TRD D-015).

    MOTIVATION
    ==========
    The naive pooled NMI of :func:`leakage_proxy` operates in a **saturation
    regime** under trace-conditioned branching: the selection-signature space
    explodes, signatures become (near-)unique, and the pooled NMI sits near
    ``1.0`` even when the probe labels carry no information — i.e. even under
    the null. The absolute NMI value is therefore not report-grade on its own.

    DEFINITION
    ==========
    Compare the observed pooled NMI against a **deterministic permutation
    null**: permute the probe labels over the *same* signature multiset
    ``n_permutations`` times, recompute the NMI per permutation, and report

    - ``observed_nmi``  — :func:`leakage_proxy`'s pooled NMI (same statistic),
    - ``null_mean`` / ``null_std`` — mean / population std (ddof=0) of the
      permutation-null NMI values,
    - ``excess_nmi`` = ``observed_nmi - null_mean``,
    - ``excess_z``  = ``(observed_nmi - null_mean) / null_std``, with the
      bounded convention ``excess_z = 0.0`` whenever
      ``null_std < NULL_STD_EPS`` (a numerically constant null is treated, by
      convention, as carrying no excess; ±inf and NaN are forbidden — see
      :data:`NULL_STD_EPS`).

    Chance adjustment for information-theoretic clustering comparison follows
    the discussion in Vinh et al. (JMLR 2010).

    INTERPRETATION CONVENTION
    =========================
    Never "the NMI is high" — only "the observed signatures carry / do not
    carry information in excess of the permutation null". Every leakage-style
    report block must carry observed/null/excess together; the absolute NMI
    alone is not reportable.

    DETERMINISM
    ===========
    The permutation RNG is seeded **only** from the input data: a
    ``random.Random`` seeded with ``int(canonical_hash(...), 16)`` over the
    pooled ``(probe, signature)`` sequence plus ``n_permutations``. No
    wall-clock, process entropy, or global RNG state enters (the global
    ``random`` module state is untouched). Identical inputs (regardless of
    mapping insertion order) yield a bit-identical output dict, so results are
    replayable on CPU.

    FAIL-CLOSED CONVENTION
    ======================
    Degenerate inputs — fewer than two probes with rounds, no rounds at all,
    or fewer than two distinct signatures (matching the NMI degeneracy rule: a
    single unique signature yields NMI 0 under *every* labeling, so the
    permutation loop would be pointless) — have nothing to separate: the
    result is the defined zero dict
    (``observed_nmi = null_mean = null_std = excess_nmi = excess_z = 0.0``)
    with ``degenerate = True`` and no permutations executed.

    This is a PROXY statistic over the controlled testbed (see
    :data:`PROXY_DISCLAIMER`) — NOT a privacy guarantee, anonymity proof,
    identifiability bound, or legal/compliance claim.

    Raises ``ValueError`` if ``n_permutations < 1``.
    """
    pairs = _collect_labeled_signatures(traces_by_probe)
    stats = _null_corrected_stats(
        pairs, n_permutations, SEPARABILITY_METRIC_NAME
    )

    if stats["degenerate"]:
        _logger.info(
            "널 보정 분리도(null_corrected_separability): 입력이 퇴화 상태라 "
            "fail-closed 0 값으로 보고합니다 (probes=%d, rounds=%d)",
            len(traces_by_probe),
            len(pairs),
        )
    else:
        _logger.info(
            "널 보정 분리도(null_corrected_separability)를 계산했습니다 "
            "(probes=%d, rounds=%d, observed_nmi=%.6f, null_mean=%.6f, "
            "null_std=%.6f, excess_nmi=%+.6f, excess_z=%+.4f, "
            "n_permutations=%d) — 해석은 'null 초과 정보가 있다/없다'로만 "
            "하며, 이는 PROXY 이고 프라이버시/법적 보증이 아닙니다",
            len(traces_by_probe),
            len(pairs),
            stats["observed_nmi"],
            stats["null_mean"],
            stats["null_std"],
            stats["excess_nmi"],
            stats["excess_z"],
            n_permutations,
        )

    return {
        "metric": SEPARABILITY_METRIC_NAME,
        **stats,
        "nullStdEps": NULL_STD_EPS,
        "isProxy": IS_PROXY,
        "disclaimer": PROXY_DISCLAIMER,
    }


def leakage_proxy_with_metadata(
    traces_by_probe: Mapping[str, Any],
    probe_versions: Mapping[str, Any] | None = None,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
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
            "nullCorrected": {observed_nmi, null_mean, null_std,
                              excess_nmi, excess_z, n_permutations, ...},
        }

    D-015: ``nullCorrected`` is the :func:`null_corrected_separability` block
    (``nullCorrected["observed_nmi"] == value`` — same pooled-NMI statistic).
    Every leakage-style report path must surface observed/null/excess together;
    the absolute ``value`` alone is not report-grade (saturation regime).

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

    null_corrected = null_corrected_separability(
        traces_by_probe, n_permutations=n_permutations
    )
    return {
        "metric": METRIC_NAME,
        # Same pooled-NMI statistic as null_corrected["observed_nmi"]; kept as
        # "value" for backward compatibility with existing report consumers.
        "value": null_corrected["observed_nmi"],
        "isProxy": IS_PROXY,
        "disclaimer": PROXY_DISCLAIMER,
        "traceHashes": trace_hashes,
        "probeVersions": probe_version_map,
        "nullCorrected": null_corrected,
    }
