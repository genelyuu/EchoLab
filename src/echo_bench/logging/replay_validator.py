"""Replay validator — the reproducibility gate (Task F-005).

If a run cannot be replayed it cannot support a main claim — this validator is
the gate that enforces that rule. It re-executes a run from its config + seed and
asserts that the full hash chain (``archiveHash``, ``poolHash``, ``slateHash``,
``traceHash``, ``outputHash``, ``reportHash``) reproduces **exactly**.

How it works
============
:func:`validate_replay` runs ``run_fn(**run_kwargs)`` **twice**. Each call
re-runs the benchmark from configuration plus seed and returns a report dict that
carries the run's hash chain. The two chains are compared with
:func:`echo_bench.metrics.replay.replay_consistency`, which is an **EXACT**
comparison (no tolerance, no fuzzy matching). A single differing hash flips the
result to non-replayable and names the first divergent key.

:func:`assert_replayable` is the hard gate: it raises :class:`ReplayError` (with
a Korean message) when a run is not replayable, so a non-deterministic run cannot
silently back a benchmark claim.

:func:`validate_report_file` is the gate for *stored* runs: it loads a
previously written report json, re-runs from config + seed, and asserts the
re-run's hash chain equals the recorded one. Any tampering with a recorded hash
(including ``reportHash``) is therefore detected.

Logging convention: runtime log messages are Korean; hash names, keys, versions,
and file paths stay English (machine-read identifiers).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional

from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.replay import replay_consistency

__all__ = [
    "validate_replay",
    "assert_replayable",
    "validate_report_file",
    "extract_chain",
    "ReplayError",
    "CHAIN_HASH_KEYS",
]

_logger = get_logger("echo_bench.logging.replay_validator")

# The full run hash chain the validator asserts on, in canonical order. Exactly
# these keys (and no tolerance) define replayability for F-005.
CHAIN_HASH_KEYS = (
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

# Carried through (not part of the hash-chain comparison itself).
_SEED_BATCH_ID_KEY = "seedBatchId"


class ReplayError(AssertionError):
    """Raised when a run is not replayable.

    A non-replayable run cannot support a main claim; this is a hard gate, so the
    error subclasses :class:`AssertionError` and carries a Korean message.
    """


def extract_chain(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the run hash chain (+ ``seedBatchId``) from a report dict.

    Pulls exactly the :data:`CHAIN_HASH_KEYS` that are present plus the
    ``seedBatchId`` metadata key. A missing hash key is simply absent from the
    returned chain; :func:`replay_consistency` treats a key present in one chain
    but not the other as a divergence, so missing-hash runs fail the gate.
    """
    if not isinstance(report, Mapping):
        raise ReplayError(
            "재현 검증: 보고서는 dict 여야 합니다 "
            f"(현재 타입: {type(report).__name__})."
        )
    chain: Dict[str, Any] = {
        key: report[key] for key in CHAIN_HASH_KEYS if key in report
    }
    if _SEED_BATCH_ID_KEY in report:
        chain[_SEED_BATCH_ID_KEY] = report[_SEED_BATCH_ID_KEY]
    return chain


def validate_replay(
    run_fn: Callable[..., Mapping[str, Any]], run_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Re-execute a run twice and check the hash chain reproduces exactly.

    Runs ``run_fn(**run_kwargs)`` two independent times (each re-runs the
    benchmark from config + seed), extracts the hash chain from each report, and
    compares them with EXACT (no-tolerance) :func:`replay_consistency`.

    Parameters
    ----------
    run_fn:
        The replayable run (e.g. ``echo_bench.experiments.smoke.run_smoke``); it
        must return a report dict carrying the run hash chain.
    run_kwargs:
        Keyword arguments forwarded verbatim to ``run_fn`` on both executions.

    Returns
    -------
    dict
        ``{"replayable": bool, "first_divergent": key|None, "chain": <chain>,
        "seedBatchId": ...}``. On full match the returned ``chain`` is the
        reproduced first-run chain; on divergence it is still the first-run chain
        and ``first_divergent`` names the first mismatching hash.
    """
    log_ko(
        _logger,
        f"재현 검증 시작: run_fn={getattr(run_fn, '__name__', run_fn)!r}, "
        f"run_kwargs={run_kwargs} (동일 config+seed 로 2회 재실행).",
    )

    report_a = run_fn(**run_kwargs)
    report_b = run_fn(**run_kwargs)

    chain_a = extract_chain(report_a)
    chain_b = extract_chain(report_b)

    result = replay_consistency(chain_a, chain_b)
    replayable = bool(result["consistent"])
    first_divergent: Optional[str] = result["first_divergent"]
    seed_batch_id = chain_a.get(_SEED_BATCH_ID_KEY)

    if replayable:
        log_ko(
            _logger,
            "재현 검증 통과: 두 재실행의 해시 체인이 정확히 일치합니다 "
            f"(seedBatchId={seed_batch_id}).",
        )
    else:
        log_ko(
            _logger,
            "재현 검증 실패: 해시 체인이 '"
            f"{first_divergent}' 키에서 처음으로 어긋났습니다 "
            f"(seedBatchId={seed_batch_id}). 재현 불가한 실행입니다.",
        )

    return {
        "replayable": replayable,
        "first_divergent": first_divergent,
        "chain": chain_a,
        "seedBatchId": seed_batch_id,
    }


def assert_replayable(
    run_fn: Callable[..., Mapping[str, Any]], run_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Gate: raise :class:`ReplayError` if the run is not replayable.

    Runs :func:`validate_replay` and, on any hash-chain divergence, raises
    :class:`ReplayError` with a Korean message — a non-replayable run cannot
    support a main claim. Returns the :func:`validate_replay` result on success.
    """
    result = validate_replay(run_fn, run_kwargs)
    if not result["replayable"]:
        raise ReplayError(
            "재현 불가한 실행은 주요 주장을 뒷받침할 수 없습니다 "
            f"(first_divergent={result['first_divergent']}, "
            f"seedBatchId={result['seedBatchId']})."
        )
    return result


def validate_report_file(
    path: str,
    run_fn: Callable[..., Mapping[str, Any]],
    run_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Gate for a stored run: re-run and assert it matches the recorded chain.

    Loads the report json at ``path``, extracts its recorded hash chain, re-runs
    ``run_fn(**run_kwargs)`` from config + seed, and asserts (EXACT, no
    tolerance) that the re-run's hash chain equals the recorded one. Raises
    :class:`ReplayError` on any divergence — so tampering with any stored hash
    (including ``reportHash``) is caught.

    Returns
    -------
    dict
        ``{"replayable": True, "first_divergent": None, "chain": <re-run chain>,
        "recordedChain": <stored chain>, "seedBatchId": ...}`` on success.
    """
    log_ko(_logger, f"저장된 보고서 재현 검증 시작: path={path}.")
    with open(path, "r", encoding="utf-8") as handle:
        recorded_report = json.load(handle)

    recorded_chain = extract_chain(recorded_report)

    rerun_report = run_fn(**run_kwargs)
    rerun_chain = extract_chain(rerun_report)

    result = replay_consistency(recorded_chain, rerun_chain)
    seed_batch_id = rerun_chain.get(_SEED_BATCH_ID_KEY)

    if not result["consistent"]:
        log_ko(
            _logger,
            "저장된 보고서 재현 검증 실패: 기록된 체인과 재실행 체인이 '"
            f"{result['first_divergent']}' 키에서 어긋났습니다 (path={path}).",
        )
        raise ReplayError(
            "재현 불가한 실행은 주요 주장을 뒷받침할 수 없습니다 "
            f"(저장 보고서 path={path}, "
            f"first_divergent={result['first_divergent']}, "
            f"seedBatchId={seed_batch_id})."
        )

    log_ko(
        _logger,
        "저장된 보고서 재현 검증 통과: 기록된 해시 체인이 재실행과 정확히 "
        f"일치합니다 (path={path}, seedBatchId={seed_batch_id}).",
    )
    return {
        "replayable": True,
        "first_divergent": None,
        "chain": rerun_chain,
        "recordedChain": recorded_chain,
        "seedBatchId": seed_batch_id,
    }
