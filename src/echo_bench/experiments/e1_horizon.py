"""E1 Horizon Sweep runner for ECHO-Bench (Task E-001, Phase 2).

E1 sweeps the interaction horizon ``H in {4, 6, 8, 12, 16, 20}`` (loaded from
``configs/experiments/horizon.yaml``) at ``k=4`` across the trace-only policies
``RANDOM``, ``FIXED_LOW_TO_HIGH`` and ``TRACE_GREEDY``, measuring how the
system-level trace-only utility metrics evolve as the horizon grows.

Pipeline (mirrors ``experiments/smoke.py``)
-------------------------------------------
1. Load the bases / archive / policy configs and the horizon config; derive a
   ``configHash`` over the merged config dicts plus the run parameters.
2. Build a reproducible candidate archive (``archiveHash``) and take the first
   ``pool_size`` cards as the deterministic candidate pool (``poolHash``).
3. For each ``(H, policy)`` cell: run a small seed batch of ``n`` child seeds
   through :func:`run_episode` (no probe -> the default, Phase-1 slot-0
   selection), compute the four trace-only metrics per seed via
   :func:`compute_all`, and aggregate (mean) them, plus a ``compute_cost``
   proxy (total rounds = ``H * n``).
4. Assemble a results table (one row per ``H x policy``), hash everything
   (``configHash`` / ``archiveHash`` / ``poolHash`` / ``slateHash`` /
   ``traceHash`` aggregate / ``outputHash`` / ``reportHash``), build a
   :class:`ReproducibilityPack`, and write
   ``outputs/reports/e1_horizon_<id>.json``.

``--dry-run`` validates the config + computes the hashes available before
executing any seed batch and writes **no** files.

Phase scope
-----------
The policy set here is the E1 set (``RANDOM``, ``FIXED_LOW_TO_HIGH``,
``TRACE_GREEDY``); these are all trace-only. ``strategy_sensitivity`` (probe
driven) belongs to E2; ``regret_to_oracle`` is deferred to Phase 3 and is never
computed here.

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
from echo_bench.env.horizon import load_horizon, validate_h
from echo_bench.env.round_runner import run_episode
from echo_bench.env.seed_batch import derive_child_seeds, seed_batch_id
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.logging.replay_validator import inline_replay_audit
from echo_bench.metrics.aggregate import aggregate_metric_dicts
from echo_bench.metrics.utility import METRIC_KEYS, compute_all
from echo_bench.policies.fixed_low_to_high import FixedLowToHighPolicy
from echo_bench.policies.random import RandomPolicy
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_e1_horizon", "main", "E1_POLICIES"]

_logger = get_logger(__name__)

E1_SCHEMA = "echo_bench.e1_horizon.report"
E1_SCHEMA_VERSION = "1"

# The E1 policy set (all trace-only). Each entry maps the English policy name to
# the (class, config-filename) used to build it. FIXED_BALANCED is an E2 policy;
# TRACE_LIN_UCB / PSEUDO_USER_MODEL / ORACLE_STRATEGY arrive in Phase 3.
E1_POLICIES = {
    "RANDOM": (RandomPolicy, "random.yaml"),
    "FIXED_LOW_TO_HIGH": (FixedLowToHighPolicy, "fixed_low_to_high.yaml"),
    "TRACE_GREEDY": (TraceGreedyPolicy, "trace_greedy.yaml"),
}

# The aggregated trace-only metric keys reported per (H, policy) cell.
E1_METRIC_KEYS = tuple(METRIC_KEYS)

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


def run_e1_horizon(
    base_seed: int = 42,
    n: int = 3,
    k: int = 4,
    pool_size: int = 64,
    dry_run: bool = False,
    replay_validate: bool = True,
) -> dict:
    """Run the E1 horizon sweep and return a fully hashed report.

    For every ``H`` in the horizon config's allowed set and every policy in
    :data:`E1_POLICIES`, run a seed batch of ``n`` child seeds and aggregate the
    four trace-only utility metrics (mean) plus a ``compute_cost`` proxy.

    Args:
        base_seed: base integer seed; child seeds are derived from it.
        n: number of child seeds (episodes) per (H, policy) cell.
        k: slate size (E1 fixes ``k=4``).
        pool_size: candidate pool size (E1 main parameter is 64).
        dry_run: when true, validate config + compute config/archive/pool hashes
            and the planned cell list, write no files, run no seed batch.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and the horizon sweep set.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    h_sweep = list(horizon_cfg["H_allowed"])

    policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in E1_POLICIES.items():
        cfg = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))
        cfg["k"] = k  # the run's k overrides the config default
        policy_cfgs[name] = cfg

    run_params = {
        "base_seed": int(base_seed),
        "n": int(n),
        "k": int(k),
        "pool_size": int(pool_size),
        "designVersion": "e1-design-1",
        "H_sweep": [int(h) for h in h_sweep],
        "policies": sorted(E1_POLICIES),
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "policy_cfgs": policy_cfgs,
            "run_params": run_params,
        }
    )

    # 2. Build the archive and take the deterministic candidate pool.
    archive = build_archive(bases, archive_cfg, base_seed)
    archive_hash = archive["archiveHash"]
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"E1 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 3. Dry run: emit the plan, write nothing.
    if dry_run:
        cells = [
            {"H": int(h), "policy": name}
            for h in h_sweep
            for name in sorted(E1_POLICIES)
        ]
        log_ko(
            _logger,
            "E1 드라이런 요약: "
            f"H_sweep={run_params['H_sweep']}, policies={sorted(E1_POLICIES)}, "
            f"n={n}, k={k}, pool_size={pool_size}, cells={len(cells)}, "
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

    # 4. Run the sweep. Build the results table (one row per H x policy).
    child_seeds = derive_child_seeds(base_seed, n)
    table: List[Dict[str, Any]] = []
    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []
    seed_batch_ids: Dict[str, str] = {}

    for h in h_sweep:
        H = validate_h(int(h), horizon_cfg)
        for name in sorted(E1_POLICIES):
            cls, _cfg_file = E1_POLICIES[name]
            cfg = policy_cfgs[name]
            policy = cls(cfg)
            batch_id = seed_batch_id(base_seed, n, policy.policy_version(), H)
            seed_batch_ids[f"{name}@H{H}"] = batch_id

            per_seed_metrics: List[Dict[str, Any]] = []
            cell_trace_hashes: List[str] = []
            for child_seed in child_seeds:
                trace = run_episode(pool, policy, child_seed, H, k, bases)
                metrics = compute_all(trace)
                per_seed_metrics.append(metrics)
                cell_trace_hashes.append(trace.trace_hash())
                all_trace_hashes.append(trace.trace_hash())
                all_slate_hashes.append(
                    canonical_hash([r["slate"] for r in trace.rounds()])
                )

            row = {
                "H": H,
                "policy": name,
                "policyVersion": policy.policy_version(),
                "seedBatchId": batch_id,
                "n": n,
                "compute_cost": int(H * n),
                "traceHashes": cell_trace_hashes,
            }
            for key in E1_METRIC_KEYS:
                row[key] = _mean([m[key] for m in per_seed_metrics])
            row["stats"] = aggregate_metric_dicts(per_seed_metrics, E1_METRIC_KEYS)
            table.append(row)

            log_ko(
                _logger,
                "E1 셀 완료: "
                f"H={H}, policy={name}, seedBatchId={batch_id[:12]}, "
                f"compute_cost={row['compute_cost']}, "
                f"coverage={row['coordinate_coverage']:.4f}",
            )

    # A single sweep-level seedBatchId summarizing the whole table.
    sweep_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "n": n,
            "H_sweep": run_params["H_sweep"],
            "policies": sorted(E1_POLICIES),
            "perCellSeedBatchIds": seed_batch_ids,
        }
    )

    # Aggregate slate / trace hashes over every episode in the sweep.
    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 5. Results body + output hash.
    results_body = {
        "config": run_params,
        "metricKeys": list(E1_METRIC_KEYS),
        "table": table,
    }
    output_hash = canonical_hash(results_body)

    # 6. Report (reportHash is computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": E1_SCHEMA,
        "schemaVersion": E1_SCHEMA_VERSION,
        "experiment": "E1_HORIZON_SWEEP",
        "phaseNote": (
            "E1 sweeps H over trace-only policies (RANDOM, FIXED_LOW_TO_HIGH, "
            "TRACE_GREEDY); strategy_sensitivity is E2 and regret_to_oracle is "
            "deferred to Phase 3. System-level metrics over a controlled "
            "testbed; no real-world generalization claim."
        ),
        "config": run_params,
        "metricKeys": list(E1_METRIC_KEYS),
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

    # 7b. Inline replay validation (Task E-012): re-run once from config+seed and
    #     confirm the hash chain reproduces EXACTLY. The audit is attached AFTER
    #     reportHash so it is not part of the hashed body (keeping the
    #     self-comparison consistent); the re-run sets replay_validate=False to
    #     avoid unbounded recursion.
    if replay_validate:
        report["replayAudit"] = inline_replay_audit(
            report,
            run_e1_horizon,
            dict(
                base_seed=base_seed,
                n=n,
                k=k,
                pool_size=pool_size,
                dry_run=False,
                replay_validate=False,
            ),
        )

    # 8. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"e1_horizon_{sweep_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "E1 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={sweep_seed_batch_id[:12]}, rows={len(table)}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run E1, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.e1_horizon",
        description="ECHO-Bench E1 horizon sweep runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument(
        "--n", type=int, default=3, help="child seeds per (H, policy) cell"
    )
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=64, help="candidate pool size"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_e1_horizon(
        base_seed=args.seed,
        n=args.n,
        k=args.k,
        pool_size=args.pool_size,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "E1 드라이런 완료: "
            f"cells={len(result['cells'])}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "E1 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"rows={len(result['table'])}",
        )


if __name__ == "__main__":
    main()
