"""S1 k-Sensitivity runner for ECHO-Bench (Task E-004, Phase 3 supplementary).

S1 sweeps the **slate size** ``k in {2, 4, 6}`` with the candidate pool size
fixed at 64 (E-004 forbids varying the pool away from 64 within this study) over
a representative policy set and reports how the trace-only utility metrics move
with ``k``.

How ``k`` interacts with the B-003 basis-diversity constraint
------------------------------------------------------------
The constraint engine (``echo_bench.env.constraints.required_distinct_bases``)
enforces a per-``k`` basis-diversity rule on every produced slate:

- ``k == 2`` requires **two** different bases (>= 2 distinct).
- ``k == 4`` requires **>= 3** distinct bases.
- ``k == 6`` requires **>= 3** distinct bases (4 are preferred; only 3 is not a
  hard fail, but emits a Korean info note).

So every ``k`` in this sweep enforces its own B-003 rule — ``k=2`` exercises the
two-base rule, while ``k=4`` and ``k=6`` exercise the >= 3-base rule. The full
archive (B1..B4) keeps every ``k`` feasible.

Policy set (representative, trace-only + the contextual bandit)
--------------------------------------------------------------
``RANDOM``, ``FIXED_BALANCED``, ``TRACE_GREEDY``, ``TRACE_LIN_UCB``. The isolated
contrast baseline (``PSEUDO_USER_MODEL``) and the oracle are intentionally absent
here; this study is about utility sensitivity to ``k``, not the oracle gap, so
``regret_to_oracle`` is never computed.

Pipeline (mirrors ``experiments/e1_horizon.py`` / ``experiments/e2_policy.py``)
-------------------------------------------------------------------------------
1. Load the bases / archive / policy configs and the horizon config; derive a
   ``configHash`` over the merged config dicts plus the run parameters.
2. Build one reproducible candidate archive (``archiveHash``) and take the first
   ``pool_size`` cards as the deterministic candidate pool (``poolHash``) — the
   pool is shared across every ``(k, policy)`` cell.
3. For each ``(k, policy)`` cell, run a small seed batch of ``n`` child seeds
   through :func:`run_episode` (default Phase-1 slot-0 selection), compute the
   four trace-only metrics per seed via :func:`compute_all`, and aggregate
   (mean) them plus a ``compute_cost`` proxy (total rounds = ``H * n``).
4. Assemble a results table (one row per ``k x policy``), hash everything, build
   a :class:`ReproducibilityPack`, and write
   ``outputs/reports/s1_k_sensitivity_<id>.json``.

``--dry-run`` validates the config + computes the hashes available before
executing any seed batch and writes **no** files.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
the pool, the traces, the metrics, or the report. The report carries only
system-level hashes and trace-only utility metrics; it makes no claim about user
preference, experience, emotion, wellbeing, or legal compliance, and no
real-world generalization claim.

All identifiers, metric names, config keys and file paths stay English; runtime
log messages are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.horizon import default_h, load_horizon, validate_h
from echo_bench.env.round_runner import run_episode
from echo_bench.env.seed_batch import derive_child_seeds, seed_batch_id
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.utility import METRIC_KEYS, compute_all
from echo_bench.policies.fixed_balanced import FixedBalancedPolicy
from echo_bench.policies.random import RandomPolicy
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_s1_k_sensitivity", "main", "S1_POLICIES", "S1_K_SWEEP"]

_logger = get_logger(__name__)

S1_SCHEMA = "echo_bench.s1_k_sensitivity.report"
S1_SCHEMA_VERSION = "1"

# The slate sizes swept by S1. Each enforces its own B-003 basis-diversity rule
# (k=2 -> two bases; k=4/6 -> >= 3 bases).
S1_K_SWEEP = (2, 4, 6)

# The representative S1 policy set: RANDOM + FIXED_BALANCED + the two
# coordinate-driven policies (TRACE_GREEDY, TRACE_LIN_UCB). All trace-only; no
# PSEUDO_USER_MODEL / oracle here.
S1_POLICIES = {
    "RANDOM": (RandomPolicy, "random.yaml"),
    "FIXED_BALANCED": (FixedBalancedPolicy, "fixed_balanced.yaml"),
    "TRACE_GREEDY": (TraceGreedyPolicy, "trace_greedy.yaml"),
    "TRACE_LIN_UCB": (TraceLinUcbPolicy, "trace_lin_ucb.yaml"),
}

# The aggregated trace-only metric keys reported per (k, policy) cell.
S1_METRIC_KEYS = tuple(METRIC_KEYS)

# Repo-rooted config locations (resolved relative to the package, not the cwd).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASES_CFG_PATH = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG_PATH = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_POLICY_CFG_DIR = _REPO_ROOT / "configs" / "policies"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into a dict (empty dict if the doc is empty)."""
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return doc if isinstance(doc, dict) else {}


def _git_commit_hash() -> str:
    """Return the short git commit hash, or ``"uncommitted"`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = out.stdout.strip()
        return commit if commit else "uncommitted"
    except (subprocess.SubprocessError, OSError):
        return "uncommitted"


def _mean(values: List[float]) -> float:
    """Arithmetic mean of a list (0.0 for an empty list)."""
    return float(sum(values) / len(values)) if values else 0.0


def _required_distinct_bases(k: int) -> int:
    """Documented minimum distinct bases for ``k`` (B-003 rule), for the report.

    Mirrors ``echo_bench.env.constraints.required_distinct_bases`` for the row's
    self-documenting ``requiredDistinctBases`` field without depending on its
    return-tuple shape: k=2 -> 2, k=4/6 -> 3.
    """
    if k == 2:
        return 2
    if k in (4, 6):
        return 3
    return min(k, 2)


def run_s1_k_sensitivity(
    base_seed: int = 42,
    n: int = 2,
    H: int | None = None,
    pool_size: int = 64,
    dry_run: bool = False,
) -> dict:
    """Run the S1 k-sensitivity sweep and return a fully hashed report.

    For every ``k`` in :data:`S1_K_SWEEP` and every policy in
    :data:`S1_POLICIES`, run a seed batch of ``n`` child seeds over the shared
    pool and aggregate the four trace-only utility metrics (mean) plus a
    ``compute_cost`` proxy. The pool size is fixed at 64 by E-004.

    Args:
        base_seed: base integer seed; child seeds are derived from it.
        n: number of child seeds (episodes) per (k, policy) cell.
        H: horizon (rounds); defaults to the horizon config's default when None.
        pool_size: candidate pool size (E-004 fixes this at 64).
        dry_run: when true, validate config + compute config/archive/pool hashes
            and the planned cell list, write no files, run no seed batch.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and resolve the horizon.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    k_sweep = list(S1_K_SWEEP)

    # Policy configs (k is supplied per-cell at run time, so it is NOT baked into
    # the stored config dict here -- it is recorded in run_params instead).
    policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in S1_POLICIES.items():
        policy_cfgs[name] = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))

    run_params = {
        "base_seed": int(base_seed),
        "n": int(n),
        "H": int(H),
        "pool_size": int(pool_size),
        "k_sweep": [int(k) for k in k_sweep],
        "policies": sorted(S1_POLICIES),
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "policy_cfgs": policy_cfgs,
            "run_params": run_params,
        }
    )

    # 2. Build the archive and take the deterministic candidate pool (fixed 64).
    archive = build_archive(bases, archive_cfg, base_seed)
    archive_hash = archive["archiveHash"]
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"S1 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 3. Dry run: emit the plan, write nothing.
    if dry_run:
        cells = [
            {
                "k": int(k),
                "policy": name,
                "requiredDistinctBases": _required_distinct_bases(int(k)),
            }
            for k in k_sweep
            for name in sorted(S1_POLICIES)
        ]
        log_ko(
            _logger,
            "S1 드라이런 요약: "
            f"k_sweep={run_params['k_sweep']}, policies={sorted(S1_POLICIES)}, "
            f"n={n}, H={H}, pool_size={pool_size}, cells={len(cells)}, "
            f"archiveHash={archive_hash[:12]}, poolHash={pool_hash[:12]}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "cells": cells,
            "configHash": config_hash,
            "archiveHash": archive_hash,
            "poolHash": pool_hash,
        }

    # 4. Run the sweep. Build the results table (one row per k x policy).
    child_seeds = derive_child_seeds(base_seed, n)
    table: List[Dict[str, Any]] = []
    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []
    seed_batch_ids: Dict[str, str] = {}

    for k in k_sweep:
        for name in sorted(S1_POLICIES):
            cls, _cfg_file = S1_POLICIES[name]
            cfg = dict(policy_cfgs[name])
            cfg["k"] = int(k)  # the cell's k overrides the config default
            policy = cls(cfg)
            batch_id = seed_batch_id(base_seed, n, policy.policy_version(), H)
            seed_batch_ids[f"{name}@k{k}"] = batch_id

            per_seed_metrics: List[Dict[str, Any]] = []
            cell_trace_hashes: List[str] = []
            for child_seed in child_seeds:
                trace = run_episode(pool, policy, child_seed, H, int(k), bases)
                metrics = compute_all(trace)
                per_seed_metrics.append(metrics)
                cell_trace_hashes.append(trace.trace_hash())
                all_trace_hashes.append(trace.trace_hash())
                all_slate_hashes.append(
                    canonical_hash([r["slate"] for r in trace.rounds()])
                )

            row = {
                "k": int(k),
                "policy": name,
                "policyVersion": policy.policy_version(),
                "requiredDistinctBases": _required_distinct_bases(int(k)),
                "seedBatchId": batch_id,
                "n": n,
                "compute_cost": int(H * n),
                "traceHashes": cell_trace_hashes,
            }
            for key in S1_METRIC_KEYS:
                row[key] = _mean([m[key] for m in per_seed_metrics])
            table.append(row)

            log_ko(
                _logger,
                "S1 셀 완료: "
                f"k={k}, policy={name}, seedBatchId={batch_id[:12]}, "
                f"requiredDistinctBases={row['requiredDistinctBases']}, "
                f"coverage={row['coordinate_coverage']:.4f}",
            )

    # A single sweep-level seedBatchId summarizing the whole table.
    sweep_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "n": n,
            "H": H,
            "k_sweep": run_params["k_sweep"],
            "policies": sorted(S1_POLICIES),
            "perCellSeedBatchIds": seed_batch_ids,
        }
    )

    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 5. Results body + output hash.
    results_body = {
        "config": run_params,
        "metricKeys": list(S1_METRIC_KEYS),
        "table": table,
    }
    output_hash = canonical_hash(results_body)

    # 6. Report (reportHash is computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": S1_SCHEMA,
        "schemaVersion": S1_SCHEMA_VERSION,
        "experiment": "S1_K_SENSITIVITY",
        "phaseNote": (
            "S1 sweeps slate size k in {2, 4, 6} with pool fixed at 64 over "
            "RANDOM, FIXED_BALANCED, TRACE_GREEDY, TRACE_LIN_UCB. Each k "
            "enforces its B-003 basis-diversity rule: k=2 needs two bases, "
            "k=4/6 need >= 3 distinct bases. Trace-only utility metrics; "
            "regret_to_oracle is not computed here. System-level metrics over a "
            "controlled testbed; no real-world generalization claim."
        ),
        "config": run_params,
        "metricKeys": list(S1_METRIC_KEYS),
        "seedBatchId": sweep_seed_batch_id,
        "perCellSeedBatchIds": seed_batch_ids,
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
        "table": table,
    }
    report_hash = canonical_hash(report)
    report["reportHash"] = report_hash

    # 7. Reproducibility pack.
    pack = ReproducibilityPack(
        configHash=config_hash,
        commitHash=_git_commit_hash(),
        archiveHash=archive_hash,
        poolHash=pool_hash,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=report_hash,
        seedBatchId=sweep_seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 8. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"s1_k_sensitivity_{sweep_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "S1 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={sweep_seed_batch_id[:12]}, rows={len(table)}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run S1, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.s1_k_sensitivity",
        description="ECHO-Bench S1 k-sensitivity runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument(
        "--n", type=int, default=2, help="child seeds per (k, policy) cell"
    )
    parser.add_argument(
        "--H", type=int, default=None, help="horizon (default: config default)"
    )
    parser.add_argument(
        "--pool-size", type=int, default=64, help="candidate pool size (fixed 64)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_s1_k_sensitivity(
        base_seed=args.seed,
        n=args.n,
        H=args.H,
        pool_size=args.pool_size,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "S1 드라이런 완료: "
            f"cells={len(result['cells'])}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "S1 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"rows={len(result['table'])}",
        )


if __name__ == "__main__":
    main()
