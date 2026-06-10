"""Seed batch runner for ECHO-Bench (Task B-006, minimal subset).

Runs ``n`` seeds reproducibly through the round runner and emits a stable
``seedBatchId`` identifying the full batch. Child seeds are derived
deterministically from a base seed, and batch output is ordered by child index.

Contract:

    derive_child_seeds(base_seed, n) -> list[int]
        ``n`` deterministic child seeds via
        ``int(canonical_hash({"base": base_seed, "i": i}), 16)``.

    seed_batch_id(base_seed, n, policy_version, H) -> str
        Stable hash of ``(base_seed, n, policy_version, H)``.

    run_seed_batch(pool, policy, base_seed, n, H, k, bases_cfg) -> dict
        Run :func:`echo_bench.env.round_runner.run_episode` for each child seed
        and return ``{"seedBatchId": ..., "traces": [trace_hash, ...], "n": n}``.

Reproducibility: re-running the same batch yields identical per-seed
``traceHash`` values and an identical ``seedBatchId``. No wall-clock or process
entropy ever enters a seed or a hash.

``seedBatchId``, paths and metric names stay English; runtime progress logging
is Korean per the project logging convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from echo_bench.env.round_runner import run_episode
from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

__all__ = ["derive_child_seeds", "seed_batch_id", "run_seed_batch"]

_logger = get_logger(__name__)


def derive_child_seeds(base_seed: int, n: int) -> List[int]:
    """Return ``n`` deterministic child seeds derived from ``base_seed``.

    Child ``i`` is ``int(canonical_hash({"base": base_seed, "i": i}), 16)``, so
    the sequence is a pure function of ``(base_seed, n)`` and is ordered by index.
    """
    if n < 0:
        raise ValueError(f"derive_child_seeds: n 은 음수일 수 없습니다 (n={n})")
    return [
        int(canonical_hash({"base": base_seed, "i": i}), 16) for i in range(n)
    ]


def seed_batch_id(base_seed: int, n: int, policy_version: str, H: int) -> str:
    """Return a stable ``seedBatchId`` for ``(base_seed, n, policy_version, H)``."""
    return canonical_hash(
        {
            "base_seed": base_seed,
            "n": n,
            "policy_version": policy_version,
            "H": H,
        }
    )


def run_seed_batch(
    pool: Sequence[Mapping[str, Any]],
    policy: Any,
    base_seed: int,
    n: int,
    H: int,
    k: int,
    bases_cfg: Any,
) -> Dict[str, Any]:
    """Run ``run_episode`` for each child seed and return the batch summary.

    Args:
        pool: candidate pool of card dicts.
        policy: the slate-selection policy (must expose ``policy_version()``).
        base_seed: base integer seed from which child seeds are derived.
        n: number of child seeds / episodes in the batch.
        H: horizon (rounds per episode).
        k: required slate size.
        bases_cfg: loaded basis config.

    Returns:
        ``{"seedBatchId": str, "traces": list[str], "n": int}`` where ``traces``
        holds the per-seed ``traceHash`` ordered by child-seed index.
    """
    batch_id = seed_batch_id(base_seed, n, policy.policy_version(), H)
    child_seeds = derive_child_seeds(base_seed, n)

    log_ko(
        _logger,
        f"시드 배치 시작: seedBatchId={batch_id}, n={n}, H={H}, k={k}",
    )

    trace_hashes: List[str] = []
    for i, child_seed in enumerate(child_seeds):
        trace = run_episode(pool, policy, child_seed, H, k, bases_cfg)
        trace_hashes.append(trace.trace_hash())
        log_ko(
            _logger,
            f"시드 배치 진행: index={i}/{n}, traceHash={trace_hashes[-1]}",
        )

    log_ko(
        _logger,
        f"시드 배치 완료: seedBatchId={batch_id}, traces={len(trace_hashes)}",
    )
    return {"seedBatchId": batch_id, "traces": trace_hashes, "n": n}
