"""S2 Basis Ablation runner for ECHO-Bench (Task E-005, Phase 3 supplementary).

S2 ablates the procedural basis set and measures the effect on archive coverage
and trace-only utility. It compares the **full** basis set ``{B1, B2, B3, B4}``
against every **3-basis subset** (each dropping exactly one basis), building a
fresh, independently-hashed archive per arm and running a coordinate-driven
policy (``TRACE_GREEDY``) over each.

k-feasibility and the skipped 2-base ablations (no silent cap)
--------------------------------------------------------------
S2 runs at ``k=4``, whose B-003 basis-diversity rule requires **>= 3 distinct
bases** per slate. A 2-basis ablation could never satisfy that rule, so those
arms are **skipped and explicitly logged** (a Korean info note) and recorded in
the report's ``skippedAblations`` list — never silently capped or dropped. The
full set and the four 3-subsets are all k=4-feasible.

Per-arm pipeline (fresh archive per arm — never a stale reuse)
--------------------------------------------------------------
For each ablation arm (a basis subset):

1. Build a fresh archive from ONLY that subset's bases (the builder round-robins
   over the supplied bases), giving the arm its own ``archiveHash``.
2. Take the first ``pool_size`` cards as the arm's pool (``poolHash``).
3. Run :func:`coverage_report` over the arm's archive (basis distribution,
   complexity-band spread, coordinate coverage / cell occupancy = diversity).
4. Run a small seed batch of ``n`` child seeds of ``TRACE_GREEDY`` through
   :func:`run_episode` and aggregate the four trace-only utility metrics (mean).

Comparing the full arm to each drop-one arm shows the contribution of the
dropped basis to coverage and diversity.

Pipeline mirrors ``experiments/e1_horizon.py`` / ``experiments/e2_policy.py``:
load configs, build archives, run, aggregate, hash everything, build a
:class:`ReproducibilityPack`, and write
``outputs/reports/s2_basis_ablation_<id>.json``. ``--dry-run`` validates config +
computes the per-arm archive/pool hashes and the planned/skipped arms, and writes
nothing.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
the archives, the traces, the metrics, or the report. The report carries only
system-level coverage/diversity statistics and trace-only utility metrics; it
makes no claim about user preference, experience, emotion, wellbeing, or legal
compliance, and no real-world generalization claim.

All identifiers, metric names, config keys and file paths stay English; runtime
log messages are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.archive.coverage import coverage_report
from echo_bench.basis.schema import load_bases
from echo_bench.env.horizon import default_h, load_horizon, validate_h
from echo_bench.env.round_runner import run_episode
from echo_bench.env.seed_batch import derive_child_seeds, seed_batch_id
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.utility import METRIC_KEYS, compute_all
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_s2_basis_ablation", "main", "S2_POLICY", "ALL_BASES"]

_logger = get_logger(__name__)

S2_SCHEMA = "echo_bench.s2_basis_ablation.report"
S2_SCHEMA_VERSION = "1"

# The full procedural basis set. Ablation arms are the full set plus every
# subset that drops exactly one basis (the drop-one 3-subsets).
ALL_BASES = ("B1", "B2", "B3", "B4")

# S2 runs one coordinate-driven policy per arm (TRACE_GREEDY).
S2_POLICY = ("TRACE_GREEDY", TraceGreedyPolicy, "trace_greedy.yaml")

# The aggregated trace-only metric keys reported per arm.
S2_METRIC_KEYS = tuple(METRIC_KEYS)

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


def _arm_id(subset: tuple[str, ...]) -> str:
    """Stable English arm id for a basis subset (e.g. "FULL", "DROP_B3")."""
    subset = tuple(sorted(subset))
    if set(subset) == set(ALL_BASES):
        return "FULL"
    dropped = [b for b in ALL_BASES if b not in subset]
    if len(dropped) == 1:
        return f"DROP_{dropped[0]}"
    return "BASES_" + "_".join(subset)


def _ablation_subsets() -> List[tuple[str, ...]]:
    """Return the candidate ablation arms: full set + drop-one + drop-two.

    Enumerates the full basis set, every drop-one 3-subset, and every drop-two
    2-subset. The 2-subsets are *candidates* that the runner will skip (and log)
    when ``k``'s basis-diversity rule needs more than 2 bases (e.g. k=4) -- so
    the skip is over a real, enumerated arm, never a silent omission. Ordered
    deterministically: full set, then 3-subsets, then 2-subsets, each in
    canonical order.
    """
    arms: List[tuple[str, ...]] = [tuple(ALL_BASES)]
    for subset in itertools.combinations(ALL_BASES, len(ALL_BASES) - 1):
        arms.append(subset)
    for subset in itertools.combinations(ALL_BASES, len(ALL_BASES) - 2):
        arms.append(subset)
    return arms


def run_s2_basis_ablation(
    base_seed: int = 42,
    n: int = 2,
    H: int | None = None,
    k: int = 4,
    pool_size: int = 48,
    dry_run: bool = False,
) -> dict:
    """Run the S2 basis-ablation study and return a fully hashed report.

    Builds a fresh archive per ablation arm (the full basis set + every
    drop-one 3-subset), runs a coverage report and a ``TRACE_GREEDY`` seed batch
    over each k=4-feasible arm, and aggregates per-arm coverage/diversity +
    trace-only utility. 2-base ablations are skipped and logged (k=4 needs >= 3
    bases).

    Args:
        base_seed: base integer seed; child seeds are derived from it.
        n: number of child seeds (episodes) per arm.
        H: horizon (rounds); defaults to the horizon config's default when None.
        k: slate size (S2 fixes ``k=4``; its rule needs >= 3 bases per slate).
        pool_size: candidate pool size per arm. Defaults to 48: a drop-one
            3-subset cannot always fill 64 cards from the configured candidate
            budget (the most-constrained arm yields ~55), so 48 keeps every
            feasible arm fillable; raise it only if the archive budget allows.
        dry_run: when true, validate config + compute per-arm archive/pool hashes
            and the planned/skipped arms, write no files, run no seed batch.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and resolve the horizon.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    policy_name, policy_cls, policy_cfg_file = S2_POLICY
    policy_cfg = dict(_load_yaml(_POLICY_CFG_DIR / policy_cfg_file))
    policy_cfg["k"] = int(k)

    # Resolve the ablation arms, splitting into k-feasible (>= 3 bases) and
    # skipped (< 3 bases -- not k=4-feasible). No silent cap: skips are recorded.
    min_required = 3 if int(k) in (4, 6) else (2 if int(k) == 2 else min(int(k), 2))
    all_arms = _ablation_subsets()
    feasible_arms: List[tuple[str, ...]] = []
    skipped: List[Dict[str, Any]] = []
    for subset in all_arms:
        if len(subset) >= min_required:
            feasible_arms.append(subset)
        else:
            skipped.append(
                {
                    "arm": _arm_id(subset),
                    "bases": list(subset),
                    "reason": "insufficient_bases_for_k",
                    "requiredDistinctBases": min_required,
                }
            )

    run_params = {
        "base_seed": int(base_seed),
        "n": int(n),
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
        "policy": policy_name,
        "allBases": list(ALL_BASES),
        "requiredDistinctBases": min_required,
        "arms": [_arm_id(s) for s in feasible_arms],
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "policy_cfg": policy_cfg,
            "run_params": run_params,
        }
    )

    # 2. Build the per-arm archives (fresh per arm) and pools.
    arm_archives: Dict[str, Dict[str, Any]] = {}
    arm_pools: Dict[str, List[Dict[str, Any]]] = {}
    arm_pool_hashes: Dict[str, str] = {}
    for subset in feasible_arms:
        arm = _arm_id(subset)
        arm_bases = {b: bases[b] for b in subset if b in bases}
        archive = build_archive(arm_bases, archive_cfg, base_seed)
        pool = archive["cards"][:pool_size]
        if len(pool) < pool_size:
            raise ValueError(
                f"S2 실행: arm={arm} 아카이브 카드 수 {len(pool)} 가 "
                f"pool_size={pool_size} 보다 작습니다"
            )
        arm_archives[arm] = archive
        arm_pools[arm] = pool
        arm_pool_hashes[arm] = canonical_hash([c["cardId"] for c in pool])

    for s in skipped:
        log_ko(
            _logger,
            "S2 아블레이션 건너뜀(로그 기록): "
            f"arm={s['arm']}, bases={s['bases']}, "
            f"k={k} 에 필요한 최소 basis 수 {min_required} 미만 "
            f"(reason={s['reason']})",
        )

    # 3. Dry run: emit the plan (per-arm archive/pool hashes + skips), write none.
    if dry_run:
        plan_arms = [
            {
                "arm": _arm_id(s),
                "bases": list(s),
                "archiveHash": arm_archives[_arm_id(s)]["archiveHash"],
                "poolHash": arm_pool_hashes[_arm_id(s)],
            }
            for s in feasible_arms
        ]
        log_ko(
            _logger,
            "S2 드라이런 요약: "
            f"arms={[a['arm'] for a in plan_arms]}, "
            f"skipped={[s['arm'] for s in skipped]}, "
            f"n={n}, H={H}, k={k}, pool_size={pool_size}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "arms": plan_arms,
            "skippedAblations": skipped,
            "configHash": config_hash,
        }

    # 4. Run each feasible arm: coverage report + a TRACE_GREEDY seed batch.
    child_seeds = derive_child_seeds(base_seed, n)
    table: List[Dict[str, Any]] = []
    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []
    all_archive_hashes: List[str] = []
    seed_batch_ids: Dict[str, str] = {}

    for subset in feasible_arms:
        arm = _arm_id(subset)
        archive = arm_archives[arm]
        pool = arm_pools[arm]
        archive_hash = archive["archiveHash"]
        all_archive_hashes.append(archive_hash)

        cov = coverage_report(archive)

        policy = policy_cls(dict(policy_cfg))
        batch_id = seed_batch_id(base_seed, n, policy.policy_version(), H)
        seed_batch_ids[arm] = batch_id

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
            "arm": arm,
            "bases": list(subset),
            "policy": policy_name,
            "policyVersion": policy.policy_version(),
            "seedBatchId": batch_id,
            "n": n,
            "archiveHash": archive_hash,
            "poolHash": arm_pool_hashes[arm],
            "coverageReportHash": cov["reportHash"],
            "archiveSize": cov["archiveSize"],
            "perBasisCounts": cov["perBasisCounts"],
            "cellOccupancy": cov["coordinateCoverage"]["cellOccupancy"],
            "occupiedCells": cov["coordinateCoverage"]["occupiedCells"],
            "traceHashes": cell_trace_hashes,
        }
        for key in S2_METRIC_KEYS:
            row[key] = _mean([m[key] for m in per_seed_metrics])
        table.append(row)

        log_ko(
            _logger,
            "S2 아블레이션 완료: "
            f"arm={arm}, archiveHash={archive_hash[:12]}, "
            f"cellOccupancy={row['cellOccupancy']:.4f}, "
            f"coverage={row['coordinate_coverage']:.4f}",
        )

    # An experiment-level seedBatchId summarizing the whole ablation set.
    exp_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "n": n,
            "H": H,
            "k": k,
            "policy": policy_name,
            "arms": run_params["arms"],
            "skipped": [s["arm"] for s in skipped],
            "perArmSeedBatchIds": seed_batch_ids,
        }
    )

    # Aggregate hashes. The archive hash for the report is over every arm's
    # archive hash (one report covers many fresh archives).
    archive_hash_agg = canonical_hash(all_archive_hashes)
    pool_hash_agg = canonical_hash(
        [arm_pool_hashes[_arm_id(s)] for s in feasible_arms]
    )
    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 5. Results body + output hash.
    results_body = {
        "config": run_params,
        "metricKeys": list(S2_METRIC_KEYS),
        "table": table,
        "skippedAblations": skipped,
    }
    output_hash = canonical_hash(results_body)

    # 6. Report.
    report: Dict[str, Any] = {
        "schema": S2_SCHEMA,
        "schemaVersion": S2_SCHEMA_VERSION,
        "experiment": "S2_BASIS_ABLATION",
        "phaseNote": (
            "S2 ablates the basis set (full {B1,B2,B3,B4} vs every drop-one "
            "3-subset), rebuilding a fresh independently-hashed archive per arm "
            "and running coverage + a TRACE_GREEDY batch over each. At k=4 each "
            "slate needs >= 3 distinct bases, so 2-base ablations are skipped "
            "and logged (see skippedAblations) -- never silently capped. "
            "System-level coverage/diversity + trace-only utility over a "
            "controlled testbed; no real-world generalization claim."
        ),
        "config": run_params,
        "metricKeys": list(S2_METRIC_KEYS),
        "seedBatchId": exp_seed_batch_id,
        "perArmSeedBatchIds": seed_batch_ids,
        "skippedAblations": skipped,
        "configHash": config_hash,
        "archiveHash": archive_hash_agg,
        "poolHash": pool_hash_agg,
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
        archiveHash=archive_hash_agg,
        poolHash=pool_hash_agg,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=report_hash,
        seedBatchId=exp_seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 8. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"s2_basis_ablation_{exp_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "S2 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={exp_seed_batch_id[:12]}, arms={len(table)}, "
        f"skipped={len(skipped)}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run S2, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.s2_basis_ablation",
        description="ECHO-Bench S2 basis-ablation runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument("--n", type=int, default=2, help="child seeds per arm")
    parser.add_argument(
        "--H", type=int, default=None, help="horizon (default: config default)"
    )
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=48, help="candidate pool size per arm"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_s2_basis_ablation(
        base_seed=args.seed,
        n=args.n,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "S2 드라이런 완료: "
            f"arms={len(result['arms'])}, "
            f"skipped={len(result['skippedAblations'])}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "S2 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"arms={len(result['table'])}, "
            f"skipped={len(result['skippedAblations'])}",
        )


if __name__ == "__main__":
    main()
