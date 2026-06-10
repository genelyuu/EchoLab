"""Round runner for ECHO-Bench (Task B-002).

Drives one adaptation **decision step** (a *round*): take a candidate pool,
invoke a policy to produce a constrained slate, deterministically record the
selection, and append exactly one round record to the trace. A round is an
adaptation decision step (pool -> policy -> slate -> selection), never a clock
or UX event.

Pipeline order is fixed and the output is a pure function of
``(candidatePoolHash, policy, seed, prior traceHash)``:

    1. ``candidatePoolHash = canonical_hash([cardId for card in pool])``.
    2. ``slate_ids = policy.select(pool, trace, seed, {"k": k})``.
    3. Map ``slate_ids`` back to card dicts and call
       :func:`echo_bench.env.constraints.check_slate`; reject (ValueError, Korean
       message) on any violation, surfacing the machine-readable reason code.
    4. Selection (Phase 1, controlled & deterministic): apply ``slotPermutation``
       to the slate via :func:`echo_bench.env.constraints.apply_permutation` and
       take slot 0. This is the controlled, deterministic selection rule for
       Phase 1. In Phase 2 a *strategy probe* (B-004) would replace this fixed
       slot-0 rule with an instrumented, versioned input policy; the probe is a
       controlled instrumented input, never a synthetic user.
    5. Aggregate the selected card's ``coordinateContribution``,
       ``complexityBand`` and ``salienceScore`` into the round record.
    6. Build the 7 caller-supplied keys and call ``trace.append_round(record)``
       (which validates and computes ``roundHash``); return the appended record.

Determinism: identical ``(poolHash, policy, seed, prior traceHash)`` yields an
identical round record and an identical resulting ``traceHash``. CPU-only; no
global RNG, no wall-clock.

All identifiers, keys, metric names and paths stay English; runtime log lines
are Korean per the project logging convention.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from echo_bench.env.constraints import apply_permutation, check_slate
from echo_bench.env.trace_state import TraceState
from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_round", "run_episode"]

_logger = get_logger(__name__)


def run_round(
    pool: Sequence[Mapping[str, Any]],
    policy: Any,
    trace: TraceState,
    seed: int,
    k: int,
    bases_cfg: Any,
    select_fn: Optional[Callable[[List[Mapping[str, Any]], TraceState, int], Any]] = None,
) -> Dict[str, Any]:
    """Advance exactly one adaptation decision step and append one round record.

    Args:
        pool: candidate pool, a sequence of card dicts (each with at least
            ``cardId``, ``basis``, ``complexityBand``, ``salienceScore``,
            ``coordinateContribution``).
        policy: a :class:`echo_bench.policies.base.Policy` whose
            ``select(pool, trace, seed, config)`` returns a list of ``cardId``.
        trace: the :class:`TraceState` to append the round to.
        seed: integer seed (no global RNG / wall-clock is used).
        k: required slate size, forwarded to the policy and the constraint check.
        bases_cfg: loaded basis config, forwarded to ``check_slate``.
        select_fn: optional, backward-compatible selection hook (Phase 2,
            B-004 strategy probes). When ``None`` (the default) the controlled,
            deterministic Phase-1 rule applies: permute the slate by
            ``slotPermutation`` and take slot 0 (byte-identical to prior
            behaviour). When provided it is called as
            ``select_fn(slate_dicts, trace, seed) -> cardId`` to choose the
            selected card from the *constraint-validated* slate; the slate
            ordering / ``slotPermutation`` are recorded unchanged. A strategy
            probe is a valid hook via
            ``lambda slate, trace, seed: probe.select(slate, trace, seed)``.
            The probe is a controlled instrumented input, never a synthetic user.

    Returns:
        The appended round record (the 8-key dict including ``roundHash``).

    Raises:
        ValueError: if a returned ``cardId`` is not in the pool, or if the slate
            violates its ``k``-rule (the constraint reason code is surfaced).
    """
    pool = list(pool)

    # 1. Candidate pool hash over the ordered cardIds.
    candidate_pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 2. Policy proposes a slate of cardIds.
    slate_ids = policy.select(pool, trace, seed, {"k": k})

    # 3. Map cardIds back to their card dicts (fail closed on unknown ids).
    by_id: Dict[Any, Mapping[str, Any]] = {c["cardId"]: c for c in pool}
    slate_dicts: List[Mapping[str, Any]] = []
    for cid in slate_ids:
        if cid not in by_id:
            raise ValueError(
                "정책이 후보 풀에 없는 cardId 를 반환했습니다: "
                f"{cid!r}"
            )
        slate_dicts.append(by_id[cid])

    # Validate the slate against its k-rule; reject with the reason code.
    ok, reason, slot_permutation = check_slate(slate_dicts, k, bases_cfg, seed)
    if not ok:
        raise ValueError(
            "슬레이트가 제약 조건을 위반했습니다 "
            f"(reason={reason}, k={k})"
        )

    # 4. Selection rule.
    #    Default (select_fn is None): the controlled, deterministic Phase-1
    #    rule -- permute the slate by slotPermutation and take slot 0. This path
    #    is byte-identical to prior behaviour.
    #    Provided (Phase 2, B-004): a strategy probe / instrumented input hook
    #    chooses the selected card from the constraint-validated slate. The probe
    #    is a controlled instrumented input, not a synthetic user; the slate
    #    ordering and slotPermutation are still recorded unchanged.
    if select_fn is None:
        permuted = apply_permutation(slate_dicts, slot_permutation)
        selected_card = permuted[0]
        selected_card_id = selected_card["cardId"]
    else:
        selected_card_id = select_fn(slate_dicts, trace, seed)
        if selected_card_id not in by_id:
            raise ValueError(
                "select_fn 이 후보 풀에 없는 cardId 를 반환했습니다: "
                f"{selected_card_id!r}"
            )
        if selected_card_id not in slate_ids:
            raise ValueError(
                "select_fn 이 슬레이트에 없는 cardId 를 반환했습니다: "
                f"{selected_card_id!r}"
            )
        selected_card = by_id[selected_card_id]

    # 5. Aggregate the selected card's observable contributions.
    record = {
        "candidatePoolHash": candidate_pool_hash,
        "slate": list(slate_ids),
        "selectedCardId": selected_card_id,
        "coordinateContribution": selected_card["coordinateContribution"],
        "complexityBand": selected_card["complexityBand"],
        "salienceScore": selected_card["salienceScore"],
        "slotPermutation": slot_permutation,
    }

    # 6. Append (validates & computes roundHash) and return the stored record.
    trace.append_round(record)
    appended = trace.rounds()[-1]

    log_ko(
        _logger,
        f"라운드 완료: 선택={selected_card_id}, traceHash={trace.trace_hash()}",
    )
    return appended


def run_episode(
    pool: Sequence[Mapping[str, Any]],
    policy: Any,
    seed: int,
    H: int,
    k: int,
    bases_cfg: Any,
    trace: TraceState | None = None,
    select_fn: Optional[Callable[[List[Mapping[str, Any]], TraceState, int], Any]] = None,
) -> TraceState:
    """Run ``H`` rounds on a fresh (or given) trace and return it.

    Each round ``i`` uses a per-round derived seed
    ``int(canonical_hash({"seed": seed, "round": i}), 16)`` so that rounds differ
    deterministically while remaining a pure function of the base ``seed``.

    Args:
        pool: candidate pool of card dicts.
        policy: the slate-selection policy.
        seed: base integer seed for the episode.
        H: number of adaptation decision steps (rounds) to run.
        k: required slate size.
        bases_cfg: loaded basis config.
        trace: an existing :class:`TraceState` to extend; a fresh one is created
            when ``None``.
        select_fn: optional selection hook threaded to :func:`run_round` (see
            its docstring). ``None`` (default) keeps the byte-identical Phase-1
            slot-0 selection; a strategy probe may be supplied for Phase 2.

    Returns:
        The :class:`TraceState` holding the ``H`` appended rounds.
    """
    if trace is None:
        trace = TraceState()

    for i in range(H):
        round_seed = int(canonical_hash({"seed": seed, "round": i}), 16)
        run_round(pool, policy, trace, round_seed, k, bases_cfg, select_fn=select_fn)

    log_ko(
        _logger,
        f"에피소드 완료: H={H}, k={k}, traceHash={trace.trace_hash()}",
    )
    return trace
