"""Observable trace-history schema for ECHO-Bench (Task B-001).

The :class:`TraceState` records *what the system surfaced and what was
selected* across adaptation decision steps. A **round** is one adaptation
decision step (candidate pool -> policy -> slate -> selection), never a clock
or UX event. Accordingly, the trace stores only observable, system-level
fields and explicitly rejects any user-model / persona / emotion / preference
/ free-text field, as well as wall-clock timestamps and session/UX telemetry.

The trace is the single observable interface every policy reads from (B-002),
so its key set and rejection rules are a hard contract that other agents code
against. Hashing is delegated to the Wave 1 utility
:func:`echo_bench.utils.hash.canonical_hash` — this module never reimplements
hashing.

Identifiers, keys, and paths stay English; runtime log lines are Korean per the
project logging convention.
"""

from __future__ import annotations

from typing import Any, Dict, List

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = ["TraceState", "ALLOWED_FIELDS", "FORBIDDEN_FIELDS"]

_logger = get_logger(__name__)

# Exactly the keys a round record may contain. No more, no fewer.
ALLOWED_FIELDS = frozenset(
    {
        "candidatePoolHash",
        "slate",
        "selectedCardId",
        "coordinateContribution",
        "complexityBand",
        "salienceScore",
        "slotPermutation",
        "roundHash",
    }
)

# Keys whose presence indicates a user-model / persona / emotion / preference
# leak. Their presence fails closed with a ValueError. The schema also rejects
# any unexpected key (see :meth:`TraceState.append_round`), which covers
# arbitrary free-text fields not enumerated here.
FORBIDDEN_FIELDS = {
    "user_id",
    "persona",
    "emotion",
    "preference",
    "user_model",
}

# ``roundHash`` is computed by :meth:`append_round`; callers supply every other
# allowed key. This is the set the caller is expected to provide.
_CALLER_REQUIRED_FIELDS = ALLOWED_FIELDS - {"roundHash"}


class TraceState:
    """Ordered, append-only history of adaptation decision rounds.

    Each appended round record must contain exactly the caller-supplied allowed
    keys; ``roundHash`` is computed and set on append, chaining the previous
    trace hash so that mutating an earlier round changes every later
    ``roundHash``. The whole-trace :meth:`trace_hash` is a deterministic
    function of the ordered round records.
    """

    def __init__(self) -> None:
        """Create an empty trace (no rounds)."""
        self._rounds: List[Dict[str, Any]] = []

    def append_round(self, record: dict) -> None:
        """Validate, hash-chain, and append one adaptation-decision round.

        The ``record`` must be a dict whose keys are exactly the caller-supplied
        allowed keys (every member of :data:`ALLOWED_FIELDS` except
        ``roundHash``). Any forbidden field, any unexpected/free-text key, or any
        missing required key raises :class:`ValueError` (with a Korean message).

        ``roundHash`` is computed as ``canonical_hash`` over the previous trace
        hash plus this round's ``slate``, ``selectedCardId`` and
        ``slotPermutation``, then stored on the record before it is appended.
        """
        if not isinstance(record, dict):
            raise ValueError(
                "라운드 레코드는 dict 여야 합니다 (type="
                f"{type(record).__name__})"
            )

        keys = set(record.keys())

        # Fail closed on any user-model / persona / emotion / preference leak.
        leaked = keys & FORBIDDEN_FIELDS
        if leaked:
            raise ValueError(
                "금지된 필드가 라운드 레코드에 포함되어 있습니다: "
                f"{sorted(leaked)}"
            )

        # Reject anything outside the caller-supplied allowed key set. This also
        # rejects arbitrary free-text fields and a prematurely supplied
        # ``roundHash`` (which the trace computes itself).
        unexpected = keys - _CALLER_REQUIRED_FIELDS
        if unexpected:
            raise ValueError(
                "허용되지 않은 키가 라운드 레코드에 포함되어 있습니다: "
                f"{sorted(unexpected)}"
            )

        # Reject incomplete records.
        missing = _CALLER_REQUIRED_FIELDS - keys
        if missing:
            raise ValueError(
                "필수 키가 라운드 레코드에서 누락되었습니다: "
                f"{sorted(missing)}"
            )

        prev_trace_hash = self.trace_hash()
        round_hash = canonical_hash(
            {
                "prev_trace_hash": prev_trace_hash,
                "slate": record["slate"],
                "selectedCardId": record["selectedCardId"],
                "slotPermutation": record["slotPermutation"],
            }
        )

        stored = dict(record)
        stored["roundHash"] = round_hash
        self._rounds.append(stored)

        _logger.info(
            "라운드를 트레이스에 추가했습니다 (roundHash=%s, len=%d)",
            round_hash,
            len(self._rounds),
        )

    def trace_hash(self) -> str:
        """Return a deterministic hash over the ordered round records.

        Identical ordered history yields an identical hash; a different order
        yields a different hash. An empty trace returns the stable canonical
        hash of ``[]``.
        """
        return canonical_hash(self._rounds)

    def rounds(self) -> list:
        """Return a read-only (copied) view of the ordered round records.

        Each record is shallow-copied so callers cannot mutate the internal
        trace state and silently break the hash chain.
        """
        return [dict(r) for r in self._rounds]

    def __len__(self) -> int:
        """Return the number of rounds recorded in the trace."""
        return len(self._rounds)
