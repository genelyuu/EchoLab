"""k-slate constraint engine for ECHO-Bench (Task B-003, minimal).

Enforces the configurable slate-size ``k`` plus basis-diversity rules on every
produced slate, and computes a deterministic, seeded slot permutation that the
round runner (B-002) logs into the trace record.

Contract (Wave 4 RANDOM + round runner code against this):

    check_slate(slate, k, bases_cfg, seed)
        -> (ok: bool, reason: str | None, slotPermutation: list[int])

    apply_permutation(slate, slotPermutation) -> list

Basis-diversity rules by ``k`` (the documented spec):

    - k == 4: requires >= 3 distinct bases.
    - k == 6: requires >= 3 distinct bases (prefer 4 — only 3 is NOT a hard
      fail, but emits a Korean info note).
    - k == 2: requires two different bases.
    - any other k: requires at least ``min(k, 2)`` distinct bases (sane
      default) and emits a Korean note that k is non-standard.

On violation the function returns ``(False, reason_code, [])`` where
``reason_code`` is a machine-readable string (``"k_mismatch"``,
``"insufficient_bases"``). On success it returns ``(True, None, perm)`` where
``perm`` is a permutation of ``range(k)`` drawn from a *seeded local*
``random.Random`` — never the global RNG or wall-clock — so that the same
``(slate, k, seed)`` always yields the same accept/reject outcome and the same
permutation.

All identifiers stay English. Runtime log messages are Korean per project
convention (see :mod:`echo_bench.logging`).
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "check_slate",
    "apply_permutation",
    "required_distinct_bases",
    "REASON_K_MISMATCH",
    "REASON_INSUFFICIENT_BASES",
]

_logger = get_logger(__name__)

# Machine-readable reason codes (English identifiers, never translated).
REASON_K_MISMATCH = "k_mismatch"
REASON_INSUFFICIENT_BASES = "insufficient_bases"


def _basis_of(card: Mapping[str, Any]) -> str:
    """Return the basis id of a card dict, failing closed if absent."""
    if not isinstance(card, Mapping):
        raise TypeError(
            f"check_slate: each slate entry must be a card mapping, got "
            f"{type(card).__name__!r}"
        )
    if "basis" not in card:
        raise ValueError("check_slate: card is missing required 'basis' field")
    return str(card["basis"])


def _card_id_of(card: Mapping[str, Any]) -> str:
    """Return the cardId of a card dict, failing closed if absent."""
    if "cardId" not in card:
        raise ValueError("check_slate: card is missing required 'cardId' field")
    return str(card["cardId"])


def required_distinct_bases(k: int) -> tuple[int, bool, bool]:
    """Resolve the basis-diversity rule for a given ``k``.

    Returns a triple ``(min_required, prefer_four, non_standard)``:

        min_required:  hard minimum number of distinct bases for acceptance.
        prefer_four:   ``True`` for k==6, where 4 bases are preferred but 3 is
                       still accepted (with a Korean info note).
        non_standard:  ``True`` for any k outside {2, 4, 6}.

    The k-rule mapping is the documented spec and is intentionally explicit;
    only *tunable* thresholds (not this mapping) would come from ``bases_cfg``.
    """
    if k == 4:
        return (3, False, False)
    if k == 6:
        return (3, True, False)
    if k == 2:
        return (2, False, False)
    # Sane default for non-standard k: at least min(k, 2) distinct bases.
    return (min(k, 2), False, True)


def check_slate(
    slate: Sequence[Mapping[str, Any]],
    k: int,
    bases_cfg: Any,
    seed: int,
) -> tuple[bool, str | None, list[int]]:
    """Validate a slate against its ``k``-rule and compute a seeded permutation.

    Args:
        slate: list of card dicts (each with at least ``cardId`` and ``basis``).
        k: required slate size (configurable; rules per :func:`required_distinct_bases`).
        bases_cfg: loaded basis config (e.g. ``{basis_id: BasisSpec}``). Reserved
            for any naturally-tunable thresholds; the k-rule mapping itself is
            the documented spec and is not overridden here.
        seed: integer seed mixed into the local RNG (never the global RNG).

    Returns:
        ``(ok, reason, slotPermutation)``. On violation ``ok`` is ``False``,
        ``reason`` is a machine-readable code, and ``slotPermutation`` is ``[]``.
        On success ``ok`` is ``True``, ``reason`` is ``None``, and
        ``slotPermutation`` is a permutation of ``range(k)``.
    """
    slate = list(slate)

    # --- Rule 1: slate size must equal k. ---------------------------------
    if len(slate) != k:
        log_ko(
            _logger,
            f"슬레이트 거부: 슬레이트 크기 {len(slate)} 가 k={k} 와 일치하지 않음 "
            f"(reason={REASON_K_MISMATCH})",
        )
        return (False, REASON_K_MISMATCH, [])

    # --- Resolve the basis-diversity rule for this k. ---------------------
    min_required, prefer_four, non_standard = required_distinct_bases(k)
    if non_standard:
        log_ko(
            _logger,
            f"비표준 k 값 감지: k={k} (표준은 2/4/6). 최소 {min_required} 개의 "
            f"서로 다른 basis 를 요구함",
        )

    # --- Rule 2: basis diversity. -----------------------------------------
    bases = [_basis_of(c) for c in slate]
    distinct = len(set(bases))
    if distinct < min_required:
        log_ko(
            _logger,
            f"슬레이트 거부: 서로 다른 basis 수 {distinct} 가 최소 요구치 "
            f"{min_required} 미만 (k={k}, reason={REASON_INSUFFICIENT_BASES})",
        )
        return (False, REASON_INSUFFICIENT_BASES, [])

    if prefer_four and distinct == 3:
        # k==6 with only 3 distinct bases: accepted, but noted (prefer 4).
        log_ko(
            _logger,
            f"안내: k={k} 슬레이트가 basis 4 종을 선호하지만 3 종만 사용함 "
            f"(허용됨)",
        )

    # --- Deterministic seeded slot permutation. ---------------------------
    # Seed material is a stable hash of (sorted cardIds, k, seed) so the same
    # logical slate yields the same permutation regardless of input ordering or
    # process. canonical_hash gives a hex digest; reduce it to an int seed.
    card_ids = sorted(_card_id_of(c) for c in slate)
    seed_material = canonical_hash(
        {"cardIds": card_ids, "k": k, "seed": seed}
    )
    rng = random.Random(int(seed_material, 16))

    slot_permutation = list(range(k))
    rng.shuffle(slot_permutation)

    log_ko(
        _logger,
        f"슬레이트 승인: k={k}, basis {distinct} 종, slotPermutation="
        f"{slot_permutation}",
    )
    return (True, None, slot_permutation)


def apply_permutation(
    slate: Sequence[Any], slot_permutation: Sequence[int]
) -> list:
    """Reorder ``slate`` by ``slot_permutation``.

    ``result[i] = slate[slot_permutation[i]]``. The permutation must be a
    bijection over ``range(len(slate))``.

    Raises:
        ValueError: if lengths differ or the permutation is not a valid
            permutation of ``range(len(slate))``.
    """
    slate = list(slate)
    perm = list(slot_permutation)
    n = len(slate)
    if len(perm) != n:
        raise ValueError(
            f"apply_permutation: permutation length {len(perm)} != slate length {n}"
        )
    if sorted(perm) != list(range(n)):
        raise ValueError(
            f"apply_permutation: {perm!r} is not a permutation of range({n})"
        )
    return [slate[i] for i in perm]
