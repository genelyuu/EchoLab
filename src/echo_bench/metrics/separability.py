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
:func:`echo_bench.metrics.leakage._null_corrected_stats`): no wall-clock,
process entropy, or global RNG state; the global ``random`` module state is
untouched. The ``combined`` channel intentionally reuses the legacy D-015
seed identity (``SEPARABILITY_METRIC_NAME``) so its whole null-corrected
block — including the sampled null — is bit-identical to
:func:`echo_bench.metrics.leakage.null_corrected_separability`.

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

from typing import Any, Dict, Mapping

from echo_bench.logging import get_logger
from echo_bench.metrics.leakage import (
    DEFAULT_NULL_PERMUTATIONS,
    IS_PROXY,
    NULL_STD_EPS,
    PROXY_DISCLAIMER,
    SEPARABILITY_METRIC_NAME,
    _collect_labeled_signatures,
    _null_corrected_stats,
    _slate_member_ids,
    null_corrected_separability,
)
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "channel_separated_separability",
    "CHANNEL_NAMES",
    "CHANNEL_SEPARABILITY_METRIC_NAME",
    "SELECTION_ABSENT_MARKER",
]

_logger = get_logger(__name__)

#: English machine-read metric name for the channel-separated statistic.
CHANNEL_SEPARABILITY_METRIC_NAME = "channel_separated_separability"

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
    return canonical_hash({"slateMembers": _slate_member_ids(record)})


def _selection_rank_signature(record: Mapping[str, Any]) -> str:
    """Selection-channel signature: rank of the choice within its slate.

    The rank of ``selectedCardId`` within the NAME-SORTED slate — see the
    module docstring for why raw ``selectedCardId`` is forbidden here (it
    would leak slate identity into the selection channel). An absent
    ``selectedCardId`` maps to :data:`SELECTION_ABSENT_MARKER`.
    """
    members = _slate_member_ids(record)
    selected = str(record.get("selectedCardId"))
    try:
        rank: Any = members.index(selected)
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

    Returns
    -------
    dict
        ``{"metric", "channelNames", "channels": {"slate": {...},
        "selection": {...}, "combined": {...}}, "slate_excess_nmi",
        "selection_excess_nmi", "combined_excess_nmi", "n_permutations",
        "nullStdEps", "isProxy", "disclaimer"}`` — each channel block is a
        full D-015 null-corrected statistics dict plus a ``channel`` name.

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

    Raises ``ValueError`` if ``n_permutations < 1``.
    """
    if n_permutations < 1:
        raise ValueError(
            f"{CHANNEL_SEPARABILITY_METRIC_NAME}: n_permutations 는 1 이상이어야 "
            f"합니다 (받은 값: {n_permutations!r})"
        )

    channels: Dict[str, Dict[str, Any]] = {}
    for channel, signature_fn in _CHANNEL_SIGNATURE_FNS.items():
        pairs = _collect_labeled_signatures(
            traces_by_probe, signature_fn=signature_fn
        )
        stats = _null_corrected_stats(
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
    # bit-identical to null_corrected_separability / leakage_proxy.
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
    return result
