"""S4 Salience Audit runner for ECHO-Bench (Task E-007, Phase 3 supplementary).

S4 audits whether a policy's slates/selections **over-concentrate on
high-salience cards**, using the D-005 salience-audit metrics and the configured
thresholds in ``configs/experiments/salience_audit.yaml`` (never hard-coded
here). It reports, per policy, on a shared seed batch:

- ``salience_outlier_rate`` — fraction of selected cards whose observable
  ``salienceScore`` strictly exceeds the configured ``outlier_threshold``
  (``[0, 1]``; higher = more concentration on high-salience cards).
- ``salience_control`` — half the total-variation gap between the observed
  per-band salience distribution and the configured target distribution
  (``[0, 1]``; ``0`` = a perfect match to the target).

A policy "over-concentrates" when its ``salience_outlier_rate`` exceeds the
configured ``outlier_threshold`` (the same documented threshold), surfaced as a
per-policy ``overConcentratesOnHighSalience`` flag so the report names which
policies over-concentrate.

Guardrails (D-005, restated)
----------------------------
``salienceScore`` is an objective image/structure statistic (A-006), **not** an
aesthetic, user-preference, emotion, or wellbeing judgment, and this audit
carries **no** privacy / legal-compliance framing. No user / persona / emotion /
preference / user_model / free-text field enters the pool, the traces, the
metrics, or the report. The metrics are *system-level* statistics over the
controlled testbed; they make no real-world generalization claim.

Pipeline mirrors ``experiments/e1_horizon.py`` / ``experiments/e2_policy.py``:
load configs, build the archive + pool, run a small seed batch per policy,
compute the two salience metrics per seed, aggregate (mean), hash everything,
build a :class:`ReproducibilityPack`, and write
``outputs/reports/s4_salience_audit_<id>.json``. ``--dry-run`` validates config +
computes the available hashes and writes nothing.

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
from echo_bench.metrics.aggregate import aggregate_values
from echo_bench.metrics.salience import (
    SALIENCE_AUDIT_CONFIG_PATH,
    load_salience_config,
    salience_control,
    salience_outlier_rate,
)
from echo_bench.policies.fixed_balanced import FixedBalancedPolicy
from echo_bench.policies.fixed_low_to_high import FixedLowToHighPolicy
from echo_bench.policies.random import RandomPolicy
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_s4_salience_audit", "main", "S4_POLICIES", "S4_METRIC_KEYS"]

_logger = get_logger(__name__)

S4_SCHEMA = "echo_bench.s4_salience_audit.report"
S4_SCHEMA_VERSION = "1"

# The audited policy set: RANDOM, the two fixed policies, and the two
# coordinate-driven policies. All trace-only; the isolated contrast baseline
# (PSEUDO_USER_MODEL) and the oracle are intentionally absent.
S4_POLICIES = {
    "RANDOM": (RandomPolicy, "random.yaml"),
    "FIXED_LOW_TO_HIGH": (FixedLowToHighPolicy, "fixed_low_to_high.yaml"),
    "FIXED_BALANCED": (FixedBalancedPolicy, "fixed_balanced.yaml"),
    "TRACE_GREEDY": (TraceGreedyPolicy, "trace_greedy.yaml"),
    "TRACE_LIN_UCB": (TraceLinUcbPolicy, "trace_lin_ucb.yaml"),
}

# The two salience-audit metric keys reported per policy.
S4_METRIC_KEYS = ("salience_outlier_rate", "salience_control")

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


def run_s4_salience_audit(
    base_seed: int = 42,
    n: int = 10,
    H: int | None = None,
    k: int = 4,
    pool_size: int = 64,
    dry_run: bool = False,
) -> dict:
    """Run the S4 salience audit and return a fully hashed report.

    For every policy in :data:`S4_POLICIES`, run a seed batch of ``n`` child
    seeds over the shared pool, compute ``salience_outlier_rate`` and
    ``salience_control`` (thresholds from
    ``configs/experiments/salience_audit.yaml``) per seed, aggregate (mean), and
    flag policies that over-concentrate on high-salience cards.

    Args:
        base_seed: base integer seed; child seeds are derived from it.
        n: number of child seeds (episodes) per policy.
        H: horizon (rounds); defaults to the horizon config's default when None.
        k: slate size (S4 fixes ``k=4``).
        pool_size: candidate pool size.
        dry_run: when true, validate config + compute config/archive/pool hashes
            and the planned policy list, write no files, run no seed batch.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs, resolve the horizon, and load the salience thresholds.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    salience_cfg = load_salience_config(SALIENCE_AUDIT_CONFIG_PATH)
    outlier_threshold = float(salience_cfg["outlier_threshold"])

    policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in S4_POLICIES.items():
        cfg = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))
        cfg["k"] = int(k)
        policy_cfgs[name] = cfg

    run_params = {
        "base_seed": int(base_seed),
        "n": int(n),
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
        "policies": sorted(S4_POLICIES),
        "outlier_threshold": outlier_threshold,
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "policy_cfgs": policy_cfgs,
            "salience_cfg": salience_cfg,
            "run_params": run_params,
        }
    )

    # 2. Build the archive and take the deterministic candidate pool.
    archive = build_archive(bases, archive_cfg, base_seed)
    archive_hash = archive["archiveHash"]
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"S4 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 3. Dry run: emit the plan, write nothing.
    if dry_run:
        log_ko(
            _logger,
            "S4 드라이런 요약: "
            f"policies={sorted(S4_POLICIES)}, n={n}, H={H}, k={k}, "
            f"pool_size={pool_size}, outlier_threshold={outlier_threshold}, "
            f"archiveHash={archive_hash[:12]}, poolHash={pool_hash[:12]}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "configHash": config_hash,
            "archiveHash": archive_hash,
            "poolHash": pool_hash,
        }

    # 4. Run each policy's seed batch; compute the two salience metrics per seed.
    child_seeds = derive_child_seeds(base_seed, n)
    table: List[Dict[str, Any]] = []
    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []
    seed_batch_ids: Dict[str, str] = {}

    for name in sorted(S4_POLICIES):
        cls, _cfg_file = S4_POLICIES[name]
        cfg = policy_cfgs[name]
        policy = cls(dict(cfg))
        batch_id = seed_batch_id(base_seed, n, policy.policy_version(), H)
        seed_batch_ids[name] = batch_id

        outlier_rates: List[float] = []
        controls: List[float] = []
        cell_trace_hashes: List[str] = []
        for child_seed in child_seeds:
            trace = run_episode(pool, policy, child_seed, H, int(k), bases)
            outlier_rates.append(salience_outlier_rate(trace, salience_cfg))
            controls.append(salience_control(trace, salience_cfg))
            cell_trace_hashes.append(trace.trace_hash())
            all_trace_hashes.append(trace.trace_hash())
            all_slate_hashes.append(
                canonical_hash([r["slate"] for r in trace.rounds()])
            )

        outlier_rate = _mean(outlier_rates)
        control = _mean(controls)
        over_concentrates = bool(outlier_rate > outlier_threshold)

        row = {
            "policy": name,
            "policyVersion": policy.policy_version(),
            "seedBatchId": batch_id,
            "n": n,
            "salience_outlier_rate": outlier_rate,
            "salience_control": control,
            "overConcentratesOnHighSalience": over_concentrates,
            "stats": {
                "salience_outlier_rate": aggregate_values(
                    outlier_rates, "salience_outlier_rate"
                ),
                "salience_control": aggregate_values(controls, "salience_control"),
            },
            "traceHashes": cell_trace_hashes,
        }
        table.append(row)

        log_ko(
            _logger,
            "S4 정책 완료: "
            f"policy={name}, seedBatchId={batch_id[:12]}, "
            f"salience_outlier_rate={outlier_rate:.4f}, "
            f"salience_control={control:.4f}, "
            f"overConcentrates={over_concentrates}",
        )

    # An experiment-level seedBatchId summarizing the whole audit.
    exp_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "n": n,
            "H": H,
            "k": k,
            "policies": sorted(S4_POLICIES),
            "perPolicySeedBatchIds": seed_batch_ids,
        }
    )

    over_concentrating = [
        row["policy"] for row in table if row["overConcentratesOnHighSalience"]
    ]

    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 5. Results body + output hash.
    results_body = {
        "config": run_params,
        "metricKeys": list(S4_METRIC_KEYS),
        "table": table,
        "overConcentratingPolicies": over_concentrating,
    }
    output_hash = canonical_hash(results_body)

    # 6. Report.
    report: Dict[str, Any] = {
        "schema": S4_SCHEMA,
        "schemaVersion": S4_SCHEMA_VERSION,
        "experiment": "S4_SALIENCE_AUDIT",
        "phaseNote": (
            "S4 audits salience concentration per policy via the D-005 metrics "
            "(salience_outlier_rate, salience_control) using the configured "
            "thresholds in configs/experiments/salience_audit.yaml. salienceScore "
            "is an objective image/structure statistic, not an aesthetic / "
            "user-preference / emotion / wellbeing judgment, and this audit "
            "carries no privacy/legal framing. System-level metrics over a "
            "controlled testbed; no real-world generalization claim."
        ),
        "config": run_params,
        "metricKeys": list(S4_METRIC_KEYS),
        "outlier_threshold": outlier_threshold,
        "seedBatchId": exp_seed_batch_id,
        "perPolicySeedBatchIds": seed_batch_ids,
        "overConcentratingPolicies": over_concentrating,
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
        seedBatchId=exp_seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 8. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"s4_salience_audit_{exp_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "S4 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={exp_seed_batch_id[:12]}, rows={len(table)}, "
        f"overConcentrating={over_concentrating}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run S4, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.s4_salience_audit",
        description="ECHO-Bench S4 salience-audit runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument("--n", type=int, default=10, help="child seeds per policy")
    parser.add_argument(
        "--H", type=int, default=None, help="horizon (default: config default)"
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

    result = run_s4_salience_audit(
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
            "S4 드라이런 완료: "
            f"policies={result['config']['policies']}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "S4 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"rows={len(result['table'])}, "
            f"overConcentrating={result['overConcentratingPolicies']}",
        )


if __name__ == "__main__":
    main()
