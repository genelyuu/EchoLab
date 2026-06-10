"""Replay-consistency / determinism metric (Task D-004).

:func:`replay_consistency` compares two run **hash chains** and reports, with
**EXACT** comparison (no tolerance, no fuzzy / approximate matching), whether
the two runs are byte-for-byte identical and, if not, the first chain key at
which they diverge.

Why exact
=========
The reproducibility invariant (CLAUDE.md, docs/09_REPRODUCIBILITY.md) is that a
run is reproducible only if it replays to identical hashes. Any tolerance would
silently smooth over non-determinism; non-determinism must instead be **reported
as a value**. So the comparison is plain ``==`` over hash strings, and a single
differing hash flips the result to inconsistent and names the divergent key.

Chain shape
===========
A hash chain is a dict keyed by the reproducibility hash names, e.g.
``archiveHash``, ``poolHash``, ``slateHash``, ``traceHash``, ``outputHash``
(any subset / superset is accepted). Keys are compared in a fixed canonical
order (:data:`CHAIN_KEY_ORDER`) so "first divergent" is well-defined and
deterministic; keys outside that order are compared afterwards in sorted order.
A key present in one chain but not the other counts as a divergence at that key.

Return value
============
``replay_consistency`` returns::

    {
        "consistent": bool,            # True iff every compared key matches
        "first_divergent": str|None,   # first divergent key, or None if consistent
        "seedBatchIds": {              # carried when present on either chain
            "a": <chain_a["seedBatchId"]|None>,
            "b": <chain_b["seedBatchId"]|None>,
        },
    }

A perfectly reproduced run -> ``{"consistent": True, "first_divergent": None}``;
any single divergence -> ``{"consistent": False, "first_divergent": <key>}``.

Determinism: the comparison is a pure deterministic function of the two chains.
Identifiers/keys stay English; runtime logs are Korean per project convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from echo_bench.logging import get_logger

__all__ = ["replay_consistency", "CHAIN_KEY_ORDER", "SEED_BATCH_ID_KEY"]

_logger = get_logger(__name__)

# Canonical comparison order for the run hash-chain keys. "First divergent" is
# resolved against this order; any extra keys are compared afterwards in sorted
# order. This makes the reported divergence point deterministic.
CHAIN_KEY_ORDER = (
    "configHash",
    "codeCommitHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

# Key carrying a run's seed-batch identity; carried through into the result when
# present. Not part of the hash-chain comparison itself.
SEED_BATCH_ID_KEY = "seedBatchId"


def _comparison_keys(chain_a: Mapping[str, Any], chain_b: Mapping[str, Any]) -> List[str]:
    """Return all hash-chain keys to compare, in deterministic order.

    Canonical-order keys (that appear in either chain) come first in
    :data:`CHAIN_KEY_ORDER` order, then any remaining keys (excluding the
    seed-batch id) in sorted order. The seed-batch id is metadata, not a
    hash-chain link, so it is never part of the comparison.
    """
    present = (set(chain_a) | set(chain_b)) - {SEED_BATCH_ID_KEY}
    ordered = [k for k in CHAIN_KEY_ORDER if k in present]
    extra = sorted(present - set(CHAIN_KEY_ORDER))
    return ordered + extra


def replay_consistency(
    chain_a: Mapping[str, Any], chain_b: Mapping[str, Any]
) -> Dict[str, Any]:
    """Exactly compare two run hash chains; report consistency + first divergence.

    EXACT comparison only — no tolerance, no fuzzy matching. A key present in one
    chain but missing from the other is a divergence at that key. The result
    carries both ``seedBatchId`` values when present.

    Parameters
    ----------
    chain_a, chain_b
        Run hash chains (dicts keyed by hash names; see module docstring).

    Returns
    -------
    dict
        ``{"consistent": bool, "first_divergent": key|None, "seedBatchIds": {...}}``.
    """
    keys = _comparison_keys(chain_a, chain_b)

    first_divergent: Optional[str] = None
    _MISSING = object()
    for key in keys:
        va = chain_a.get(key, _MISSING)
        vb = chain_b.get(key, _MISSING)
        # Missing-on-one-side is a divergence; otherwise exact equality.
        if va is _MISSING or vb is _MISSING or va != vb:
            first_divergent = key
            break

    consistent = first_divergent is None
    result = {
        "consistent": consistent,
        "first_divergent": first_divergent,
        "seedBatchIds": {
            "a": chain_a.get(SEED_BATCH_ID_KEY),
            "b": chain_b.get(SEED_BATCH_ID_KEY),
        },
    }

    if consistent:
        _logger.info(
            "리플레이 일관성: 두 해시 체인이 정확히 일치합니다 "
            "(compared_keys=%d, seedBatchIds=%s/%s)",
            len(keys),
            result["seedBatchIds"]["a"],
            result["seedBatchIds"]["b"],
        )
    else:
        _logger.info(
            "리플레이 비일관성: 해시 체인이 '%s' 키에서 처음으로 어긋났습니다 "
            "(비결정성을 값으로 보고합니다, seedBatchIds=%s/%s)",
            first_divergent,
            result["seedBatchIds"]["a"],
            result["seedBatchIds"]["b"],
        )
    return result
