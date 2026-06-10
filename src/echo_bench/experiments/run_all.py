# src/echo_bench/experiments/run_all.py
"""Consolidated ECHO-Bench benchmark driver (Task E-009).

Runs E1 (horizon sweep), E2 (policy utility), and E3 (leakage / robustness /
replay audit) with standard parameters and writes a single hashed
``benchmark_index.json`` linking every report's
``reportHash``/``packHash``/``seedBatchId``/``designVersion`` plus its replay
status.  S1–S4 supplements have their own dedicated runners and are **not**
aggregated here.

Replayability notes in the index:

- **E3** sets ``"replayable"`` from ``validate_replay``; source is recorded in
  ``"replayableSource"``.
- **E1/E2** set ``"replayable": null`` — determinism is guaranteed by
  construction (fixed seed → fixed trace), and replay is verified by the F-006
  regression guard, not measured inline here.  See ``"replayableNote"`` for the
  rationale.

GPU note: ``prefer_gpu`` is accepted and recorded in the index ``config`` only —
it is NOT forwarded to any experiment runner (E1/E2/E3 are CPU trace producers
and take no GPU argument). GPU eligibility is confined to the F-007 whitelist
(batch card generation / neural baselines), so the CPU-replayable core is
untouched regardless of this flag.

System-level only; no real-world generalization claim. Identifiers/paths stay
English; runtime logs are Korean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from echo_bench.experiments.e1_horizon import run_e1_horizon
from echo_bench.experiments.e2_policy import run_e2_policy
from echo_bench.experiments.e3_audit import run_e3_audit
from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_all", "main", "BENCHMARK_INDEX_SCHEMA"]

_logger = get_logger(__name__)

BENCHMARK_INDEX_SCHEMA = "echo_bench.benchmark.index"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"
_DESIGN_POLICY = "docs/11_EXPERIMENT_DESIGN_POLICY.md"


def run_all(
    base_seed: int = 42, n: int = 10, dry_run: bool = False, prefer_gpu: bool = False
) -> dict:
    """Run the experiment matrix and write/return the hashed benchmark index."""
    planned = [
        {"experiment": "E1_HORIZON_SWEEP", "designVersion": "e1-design-1"},
        {"experiment": "E2_POLICY_UTILITY", "designVersion": "e2-design-1"},
        {"experiment": "E3_AUDIT", "designVersion": "e3-design-1"},
    ]

    if dry_run:
        log_ko(
            _logger,
            f"run_all 드라이런: experiments={[p['experiment'] for p in planned]}, "
            f"n={n}, prefer_gpu={prefer_gpu} (파일 미작성)",
        )
        return {"dryRun": True, "experiments": planned, "n": int(n)}

    entries: List[Dict[str, Any]] = []

    _REPLAY_NOTE = (
        "determinism-by-construction; replay verified by the F-006 regression guard, "
        "not measured inline here"
    )

    e1 = run_e1_horizon(base_seed=base_seed, n=n)
    entries.append({
        "experiment": "E1_HORIZON_SWEEP", "designVersion": "e1-design-1",
        "reportHash": e1["reportHash"], "packHash": e1["packHash"],
        "seedBatchId": e1["seedBatchId"],
        "replayable": None, "replayableNote": _REPLAY_NOTE,
    })

    e2 = run_e2_policy(base_seed=base_seed, n=n)
    entries.append({
        "experiment": "E2_POLICY_UTILITY", "designVersion": "e2-design-1",
        "reportHash": e2["reportHash"], "packHash": e2["packHash"],
        "seedBatchId": e2["seedBatchId"],
        "replayable": None, "replayableNote": _REPLAY_NOTE,
    })

    e3 = run_e3_audit(base_seed=base_seed)
    entries.append({
        "experiment": "E3_AUDIT", "designVersion": "e3-design-1",
        "reportHash": e3["reportHash"], "packHash": e3["packHash"],
        "seedBatchId": e3["seedBatchId"],
        "replayable": bool(e3["replayAudit"]["replayable"]),
        "replayableSource": "validate_replay",
    })

    index: Dict[str, Any] = {
        "schema": BENCHMARK_INDEX_SCHEMA,
        "designPolicy": _DESIGN_POLICY,
        "config": {"base_seed": int(base_seed), "n": int(n), "prefer_gpu": bool(prefer_gpu)},
        "experiments": entries,
    }
    index["indexHash"] = canonical_hash(index)

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / "benchmark_index.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        f"벤치마크 인덱스 작성 완료: path={out_path}, "
        f"indexHash={index['indexHash'][:12]}, experiments={len(entries)}",
    )
    return index


def main() -> None:
    """CLI entry point for the consolidated driver."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.run_all",
        description="ECHO-Bench consolidated benchmark driver.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_all(
        base_seed=args.seed, n=args.n, dry_run=args.dry_run, prefer_gpu=args.prefer_gpu
    )
    if result.get("dryRun"):
        log_ko(_logger, f"run_all 드라이런 완료: {len(result['experiments'])} 실험 계획")
    else:
        log_ko(_logger, f"run_all 완료: indexHash={result['indexHash'][:12]}")


if __name__ == "__main__":
    main()
