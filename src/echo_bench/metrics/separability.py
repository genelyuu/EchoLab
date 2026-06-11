"""Channel-separated probe-separability **PROXY** (Task D-016, TRD D-016).

WHY CHANNELS
============
The legacy D-002 selection signature concatenates ``selectedCardId`` + the
sorted slate membership into ONE signature, so the pooled NMI of
:func:`echo_bench.metrics.leakage.leakage_proxy` (and its D-015 null-corrected
form) conflates two different phenomena:

- the policy **showed different slates** under different probes — the
  trace-conditioned branching of WHAT WAS SHOWN, and
- the probe **chose differently** given what was shown — choice separability
  CONDITIONAL on the slate.

D-016 splits the statistic into three channels over the same observable-fields
contract (only ``slate`` and ``selectedCardId`` are read; no latent / user /
persona field exists or is touched), each channel carrying the full D-015
null-corrected block (``observed_nmi`` / ``null_mean`` / ``null_std`` /
``excess_nmi`` / ``excess_z``):

- ``slate``     — the name-sorted slate member ids only (no ``selectedCardId``).
- ``selection`` — the RANK of ``selectedCardId`` within its name-sorted slate.
- ``combined``  — the legacy signature, byte-identical to
  :func:`echo_bench.metrics.leakage._selection_signature`, so legacy
  ``leakage_proxy`` / D-015 ``observed_nmi`` values do not shift.

SELECTION-RANK ENCODING RATIONALE
=================================
Raw ``selectedCardId`` would leak slate identity into the selection channel:
card ids differ across slates, so two probes that both apply "pick the first
card of whatever was shown" would look maximally different whenever their
slates differ — exactly the slate-channel information the split is meant to
remove. Encoding the selection as the **rank within the name-sorted slate**
makes the channel invariant both to slate identity and to stored slate order
(consistent with the combined channel's order-independent membership
semantics). A ``selectedCardId`` absent from its slate maps deterministically
to the distinct fail-closed marker :data:`SELECTION_ABSENT_MARKER` — it never
crashes and never collides with an in-slate rank.

DETERMINISM
===========
Per-channel permutation seeds derive ONLY from the input data plus the channel
identity via ``canonical_hash`` (see
:func:`echo_bench.metrics.leakage.null_corrected_stats`): no wall-clock,
process entropy, or global RNG state; the global ``random`` module state is
untouched. The ``combined`` channel intentionally reuses the legacy D-015
seed identity (``SEPARABILITY_METRIC_NAME``) so its whole null-corrected
block — including the sampled null — is bit-identical to
:func:`echo_bench.metrics.leakage.null_corrected_separability`.

SATURATION DIAGNOSTICS (D-017)
==============================
Every channel block additionally carries a ``saturation`` sub-block
(:func:`echo_bench.metrics.leakage.signature_saturation_stats`):
``sample_count`` / ``distinct_signature_count`` / ``unique_signature_rate`` /
``cardinality_sample_ratio`` / ``saturation_flag``. The flag automatically
detects the measurement-failure regime where (nearly) every signature is
unique and the absolute NMI is not report-grade; ``saturation_flag = True``
forbids a headline leakage claim for that channel (docs/12_CLAIM_LADDER.md
Section 5, Track L re-enable condition 2). The standalone, permutation-free
form is :func:`signature_saturation_diagnostics`. Flags are diagnostics, not
claims.

FAIL-CLOSED / PROXY FRAMING
===========================
Each channel fails closed independently to the D-015 zero dict when its own
signature marginal is degenerate. Every value here is a PROXY statistic over
the controlled testbed (see
:data:`echo_bench.metrics.leakage.PROXY_DISCLAIMER`) — NOT a privacy
guarantee, anonymity proof, identifiability bound, or legal/compliance claim.
Interpretation stays "the observed signatures carry / do not carry information
in excess of the permutation null", per channel.

Identifiers, keys, and metric names stay English; runtime log lines are Korean
per the project logging convention.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

from echo_bench.logging import get_logger
from echo_bench.metrics.leakage import (
    DEFAULT_NULL_PERMUTATIONS,
    IS_PROXY,
    NULL_STD_EPS,
    PROXY_DISCLAIMER,
    SATURATION_UNIQUE_RATE_THRESHOLD,
    SEPARABILITY_METRIC_NAME,
    collect_labeled_signatures,
    null_corrected_separability,
    null_corrected_stats,
    signature_saturation_stats,
    slate_member_ids,
)
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "channel_separated_separability",
    "signature_saturation_diagnostics",
    "separability_row_fields",
    "CHANNEL_NAMES",
    "CHANNEL_SEPARABILITY_METRIC_NAME",
    "SATURATION_METRIC_NAME",
    "SATURATION_UNIQUE_RATE_THRESHOLD",
    "SELECTION_ABSENT_MARKER",
]

_logger = get_logger(__name__)

#: English machine-read metric name for the channel-separated statistic.
CHANNEL_SEPARABILITY_METRIC_NAME = "channel_separated_separability"

#: English machine-read metric name for the D-017 saturation diagnostics.
SATURATION_METRIC_NAME = "signature_saturation_diagnostics"

#: The exact channel names, in reporting order (docs/12_CLAIM_LADDER.md
#: Section 5 Track L re-enable condition 3: channel-separated AND
#: channel-named).
CHANNEL_NAMES = ("slate", "selection", "combined")

#: Fail-closed selection-channel encoding for a ``selectedCardId`` that is not
#: a member of its own slate (defensive; cannot collide with an integer rank).
SELECTION_ABSENT_MARKER = "ABSENT_FROM_SLATE"


def _slate_signature(record: Mapping[str, Any]) -> str:
    """Slate-channel signature: the name-sorted slate member ids only.

    Measures trace-conditioned branching of WHAT WAS SHOWN; deliberately blind
    to ``selectedCardId``.
    """
    return canonical_hash({"slateMembers": slate_member_ids(record)})


def _selection_rank_signature(record: Mapping[str, Any]) -> str:
    """Selection-channel signature: rank of the choice within its slate.

    The rank of ``selectedCardId`` within the NAME-SORTED slate — see the
    module docstring for why raw ``selectedCardId`` is forbidden here (it
    would leak slate identity into the selection channel). An absent
    ``selectedCardId`` maps to :data:`SELECTION_ABSENT_MARKER`.
    """
    members = slate_member_ids(record)
    selected = str(record.get("selectedCardId"))
    try:
        # ``list.index`` returns the FIRST occurrence, so a duplicate-id slate
        # still yields a deterministic rank.
        rank: Union[int, str] = members.index(selected)
    except ValueError:
        rank = SELECTION_ABSENT_MARKER
    return canonical_hash({"selectionRank": rank})


# Per-channel signature functions for the channels computed locally. The
# ``combined`` channel is NOT listed here: it is delegated verbatim to
# null_corrected_separability so the legacy block (signature bytes AND
# permutation-seed identity) is reproduced exactly.
_CHANNEL_SIGNATURE_FNS = {
    "slate": _slate_signature,
    "selection": _selection_rank_signature,
}


def channel_separated_separability(
    traces_by_probe: Mapping[str, Any],
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    precomputed_combined: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Channel-separated null-corrected probe-separability **PROXY** (D-016).

    Parameters
    ----------
    traces_by_probe
        Maps ``probeName`` (str) -> ``TraceState``; only the observable
        ``slate`` / ``selectedCardId`` fields of each round are read.
    n_permutations
        Deterministic label permutations per channel null (default
        :data:`echo_bench.metrics.leakage.DEFAULT_NULL_PERMUTATIONS`).
    precomputed_combined
        Optional: a block previously returned by
        :func:`echo_bench.metrics.leakage.null_corrected_separability` for the
        SAME ``traces_by_probe`` / ``n_permutations`` (e.g.
        ``leakage_proxy_with_metadata(...)['nullCorrected']``). When supplied,
        the combined channel reuses it instead of recomputing the
        ``n_permutations``-permutation null a second time (D-016 review
        follow-up: E3 previously ran the combined null twice per policy).
        Because ``null_corrected_separability`` is a pure deterministic
        function of those inputs, the result is bit-identical either way; the
        block is validated defensively (metric name + permutation count) and a
        mismatch raises ``ValueError`` (fail closed). The CALLER is
        responsible for passing a block computed over the same traces.

    Returns
    -------
    dict
        ``{"metric", "channelNames", "channels": {"slate": {...},
        "selection": {...}, "combined": {...}}, "slate_excess_nmi",
        "selection_excess_nmi", "combined_excess_nmi", "slate_saturation_flag",
        "selection_saturation_flag", "combined_saturation_flag",
        "saturationThreshold", "n_permutations", "nullStdEps", "isProxy",
        "disclaimer"}`` — each channel block is a full D-015 null-corrected
        statistics dict (including the D-017 ``saturation`` sub-block) plus a
        ``channel`` name.

    INVARIANTS
    ==========
    - The ``combined`` block equals
      :func:`echo_bench.metrics.leakage.null_corrected_separability` exactly
      (modulo the added ``channel`` key); in particular
      ``channels["combined"]["observed_nmi"] == leakage_proxy(traces_by_probe)``.
    - Deterministic and CPU-replayable: identical inputs (regardless of
      mapping insertion order) yield a bit-identical dict; the global RNG is
      untouched.
    - Each channel fails closed independently (degenerate marginal -> the
      D-015 zero dict with ``degenerate = True``, no permutations executed).
    - D-017: each channel carries its own ``saturation`` diagnostics block
      (``saturation_flag = True`` is a DIAGNOSTIC, never a claim — it forbids
      a headline leakage claim for that channel per docs/12_CLAIM_LADDER.md
      Section 5 condition 2).

    Raises ``ValueError`` if ``n_permutations < 1`` or if
    ``precomputed_combined`` fails validation.
    """
    if n_permutations < 1:
        raise ValueError(
            f"{CHANNEL_SEPARABILITY_METRIC_NAME}: n_permutations 는 1 이상이어야 "
            f"합니다 (받은 값: {n_permutations!r})"
        )

    channels: Dict[str, Dict[str, Any]] = {}
    for channel, signature_fn in _CHANNEL_SIGNATURE_FNS.items():
        pairs = collect_labeled_signatures(
            traces_by_probe, signature_fn=signature_fn
        )
        stats = null_corrected_stats(
            pairs,
            n_permutations,
            f"{CHANNEL_SEPARABILITY_METRIC_NAME}:{channel}",
        )
        channels[channel] = {
            "metric": SEPARABILITY_METRIC_NAME,
            **stats,
            "nullStdEps": NULL_STD_EPS,
            "isProxy": IS_PROXY,
            "disclaimer": PROXY_DISCLAIMER,
            "channel": channel,
        }

    # Combined channel: delegated verbatim to the legacy D-015 statistic so
    # the block (legacy signature bytes + legacy permutation-seed identity) is
    # bit-identical to null_corrected_separability / leakage_proxy. A caller
    # that already holds that exact block may pass it in to avoid recomputing
    # the permutation null (deterministic -> bit-identical either way).
    if precomputed_combined is not None:
        if (
            precomputed_combined.get("metric") != SEPARABILITY_METRIC_NAME
            or precomputed_combined.get("n_permutations") != int(n_permutations)
            or "saturation" not in precomputed_combined
        ):
            raise ValueError(
                f"{CHANNEL_SEPARABILITY_METRIC_NAME}: precomputed_combined 이 "
                f"유효한 {SEPARABILITY_METRIC_NAME} 블록이 아니거나 "
                f"n_permutations={n_permutations} 와 일치하지 않습니다 "
                f"(받은 metric={precomputed_combined.get('metric')!r}, "
                f"n_permutations="
                f"{precomputed_combined.get('n_permutations')!r})"
            )
        combined = dict(precomputed_combined)
    else:
        combined = dict(
            null_corrected_separability(
                traces_by_probe, n_permutations=n_permutations
            )
        )
    combined["channel"] = "combined"
    channels["combined"] = combined

    result: Dict[str, Any] = {
        "metric": CHANNEL_SEPARABILITY_METRIC_NAME,
        "channelNames": list(CHANNEL_NAMES),
        "channels": {name: channels[name] for name in CHANNEL_NAMES},
        "slate_excess_nmi": channels["slate"]["excess_nmi"],
        "selection_excess_nmi": channels["selection"]["excess_nmi"],
        "combined_excess_nmi": channels["combined"]["excess_nmi"],
        # D-017: per-channel saturation flags (diagnostics, not claims),
        # mirroring the excess trio for flat report consumption.
        "slate_saturation_flag": channels["slate"]["saturation"][
            "saturation_flag"
        ],
        "selection_saturation_flag": channels["selection"]["saturation"][
            "saturation_flag"
        ],
        "combined_saturation_flag": channels["combined"]["saturation"][
            "saturation_flag"
        ],
        "saturationThreshold": SATURATION_UNIQUE_RATE_THRESHOLD,
        "n_permutations": int(n_permutations),
        "nullStdEps": NULL_STD_EPS,
        "isProxy": IS_PROXY,
        "disclaimer": PROXY_DISCLAIMER,
    }
    _logger.info(
        "채널 분리 분리도(channel_separated_separability)를 계산했습니다 "
        "(probes=%d, slate_excess_nmi=%+.6f, selection_excess_nmi=%+.6f, "
        "combined_excess_nmi=%+.6f, n_permutations=%d) — 해석은 채널별 "
        "'null 초과 정보가 있다/없다'로만 하며, 이는 PROXY 이고 "
        "프라이버시/법적 보증이 아닙니다",
        len(traces_by_probe),
        result["slate_excess_nmi"],
        result["selection_excess_nmi"],
        result["combined_excess_nmi"],
        n_permutations,
    )
    saturated = [
        name
        for name in CHANNEL_NAMES
        if channels[name]["saturation"]["saturation_flag"]
    ]
    if saturated:
        _logger.warning(
            "시그니처 포화 감지(D-017): 채널 %s 에서 saturation_flag=True — "
            "해당 채널의 절대 NMI/누출 수치는 헤드라인 클레임에 사용할 수 "
            "없습니다 (진단이지 클레임이 아닙니다; 임계값=%s)",
            saturated,
            SATURATION_UNIQUE_RATE_THRESHOLD,
        )
    return result


def signature_saturation_diagnostics(
    traces_by_probe: Mapping[str, Any],
) -> Dict[str, Any]:
    """Per-channel signature-saturation **DIAGNOSTICS** (Task D-017, TRD V-004).

    Detects automatically when NMI saturation occurs: if the signature space
    is too large relative to the pooled sample — (nearly) every signature
    unique — the absolute pooled NMI is not report-grade for that channel.
    This is the standalone, permutation-free form of the diagnostics; the same
    blocks are embedded per channel in :func:`channel_separated_separability`
    and in :func:`echo_bench.metrics.leakage.null_corrected_separability`
    (``saturation`` sub-block).

    Parameters
    ----------
    traces_by_probe
        Maps ``probeName`` (str) -> ``TraceState``; only the observable
        ``slate`` / ``selectedCardId`` fields of each round are read.

    Returns
    -------
    dict
        ``{"metric", "channelNames", "channels": {channel: {"sample_count",
        "distinct_signature_count", "unique_signature_rate",
        "cardinality_sample_ratio", "saturation_flag", "saturationThreshold",
        "channel"}}, "slate_saturation_flag", "selection_saturation_flag",
        "combined_saturation_flag", "saturationThreshold"}`` — see
        :func:`echo_bench.metrics.leakage.signature_saturation_stats` for the
        per-channel field definitions.

    INTERPRETATION CONVENTION: the flags are DIAGNOSTICS that catch
    measurement failure, never claims. ``saturation_flag = True`` for a
    channel forbids a headline leakage claim citing that channel
    (docs/12_CLAIM_LADDER.md Section 5, Track L re-enable condition 2 requires
    ``saturation_flag = False``). Pure deterministic function of the
    observable inputs — no RNG, no permutations, mapping insertion-order
    independent.
    """
    channels: Dict[str, Dict[str, Any]] = {}
    for channel in CHANNEL_NAMES:
        # The combined channel uses collect_labeled_signatures' default —
        # the legacy combined signature (same bytes as leakage_proxy's).
        signature_fn = _CHANNEL_SIGNATURE_FNS.get(channel)
        pairs = collect_labeled_signatures(
            traces_by_probe, signature_fn=signature_fn
        )
        block = signature_saturation_stats([sig for _probe, sig in pairs])
        block["channel"] = channel
        channels[channel] = block

    result: Dict[str, Any] = {
        "metric": SATURATION_METRIC_NAME,
        "channelNames": list(CHANNEL_NAMES),
        "channels": channels,
        "slate_saturation_flag": channels["slate"]["saturation_flag"],
        "selection_saturation_flag": channels["selection"]["saturation_flag"],
        "combined_saturation_flag": channels["combined"]["saturation_flag"],
        "saturationThreshold": SATURATION_UNIQUE_RATE_THRESHOLD,
    }
    _logger.info(
        "시그니처 포화 진단(signature_saturation_diagnostics)을 계산했습니다 "
        "(probes=%d, slate_saturation_flag=%s, selection_saturation_flag=%s, "
        "combined_saturation_flag=%s, 임계값=%s) — 플래그는 측정 실패를 잡는 "
        "진단이지 클레임이 아닙니다",
        len(traces_by_probe),
        result["slate_saturation_flag"],
        result["selection_saturation_flag"],
        result["combined_saturation_flag"],
        SATURATION_UNIQUE_RATE_THRESHOLD,
    )
    return result


def separability_row_fields(
    null_corrected: Mapping[str, Any],
    channel_sep: Mapping[str, Any],
) -> Dict[str, Any]:
    """The shared D-015 + D-016 + D-017 report-row fragment (E-019 review).

    Maps one D-015 null-corrected block and one D-016/D-017 channel-separated
    block to the exact row keys every leakage-style report table carries —
    single source of truth so the E3 leakage rows and the E-019 diagnostic
    rows can never silently diverge:

    - D-015 quintet: ``observed_nmi`` / ``null_mean`` / ``null_std`` /
      ``excess_nmi`` / ``excess_z`` (observed/null/excess always travel
      together; absolute NMI alone is not report-grade),
    - D-016 trio: ``slate_excess_nmi`` / ``selection_excess_nmi`` /
      ``combined_excess_nmi``,
    - D-017 quartet: ``saturation_flag`` (the headline gate — equals the
      combined channel's flag) plus the three per-channel flags
      (diagnostics, never claims).
    """
    return {
        "observed_nmi": null_corrected["observed_nmi"],
        "null_mean": null_corrected["null_mean"],
        "null_std": null_corrected["null_std"],
        "excess_nmi": null_corrected["excess_nmi"],
        "excess_z": null_corrected["excess_z"],
        "slate_excess_nmi": channel_sep["slate_excess_nmi"],
        "selection_excess_nmi": channel_sep["selection_excess_nmi"],
        "combined_excess_nmi": channel_sep["combined_excess_nmi"],
        "saturation_flag": channel_sep["combined_saturation_flag"],
        "slate_saturation_flag": channel_sep["slate_saturation_flag"],
        "selection_saturation_flag": channel_sep["selection_saturation_flag"],
        "combined_saturation_flag": channel_sep["combined_saturation_flag"],
    }
