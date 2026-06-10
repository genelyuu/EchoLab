"""End-to-end smoke runner for ECHO-Bench (Task E, Phase 1 Wave 5).

The smoke runner is the smallest end-to-end benchmark: load the basis config,
build a reproducible candidate archive, take a deterministic candidate pool,
run a single RANDOM episode over a short horizon, compute the trace-only
utility metrics, and assemble a fully hashed report plus a reproducibility
pack. It exists to prove that the Phase 1 modules wire together and that a run
*replays* -- re-running the same config + seed reproduces identical
``traceHash`` and ``reportHash``.

Phase scope
-----------
This runner executes **RANDOM only** (Phase 1). The full smoke flow -- which
additionally runs
``TRACE_GREEDY`` and a cross-policy replay validation -- activates in **Phase
2**, once the TRACE_GREEDY policy (C-004) lands. Until then the runner is
intentionally single-policy so it stays CPU-replayable and free of any
not-yet-implemented dependency.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
the pool, the trace, the metrics, or the report. The report carries only
system-level hashes and trace-only utility metrics; it makes no claim about
user preference, experience, emotion, wellbeing, or legal compliance, and no
real-world generalization claim.

All identifiers, metric names, config keys and file paths stay English; runtime
log messages are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.round_runner import run_episode
from echo_bench.env.seed_batch import seed_batch_id
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.utility import compute_all
from echo_bench.policies.random import RandomPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = ["run_smoke", "main"]

_logger = get_logger(__name__)

# Phase 1 smoke runs RANDOM only. TRACE_GREEDY (the full run-smoke flow) is
# Phase 2; recorded here so the report self-documents its scope.
SMOKE_POLICY = "RANDOM"
SMOKE_SCHEMA = "echo_bench.smoke.report"
SMOKE_SCHEMA_VERSION = "1"

# Repo-rooted config locations (resolved relative to the package, not the cwd,
# so the runner works regardless of the working directory).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASES_CFG_PATH = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG_PATH = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_POLICY_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "random.yaml"
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


def run_smoke(
    base_seed: int = 42,
    H: int = 8,
    k: int = 4,
    pool_size: int = 16,
    dry_run: bool = False,
) -> dict:
    """Run the Phase 1 RANDOM smoke episode and return a fully hashed report.

    Pipeline (RANDOM only):

    1. Load bases / archive / policy configs and derive ``configHash`` over the
       merged config dicts plus the run parameters.
    2. Build a reproducible archive (``archiveHash``) and take the first
       ``pool_size`` cards as the deterministic candidate pool (``poolHash``).
    3. Construct the RANDOM policy and the ``seedBatchId``.
    4. ``dry_run``: log a Korean plan summary and return the planned config plus
       the hashes computed so far -- writing **no** files.
    5. Otherwise run one ``H``-round episode, compute trace-only metrics, hash
       the slate / trace / output / report, build the reproducibility pack, and
       write ``outputs/reports/smoke_<seedBatchId[:12]>.json``.

    Returns:
        The report dict (dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and derive the config hash.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    policy_cfg = dict(_load_yaml(_POLICY_CFG_PATH))
    policy_cfg["k"] = k  # the run's k overrides the config default

    run_params = {
        "base_seed": int(base_seed),
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "policy_cfg": policy_cfg,
            "run_params": run_params,
            "policy": SMOKE_POLICY,
        }
    )

    # 2. Build the archive and take the deterministic candidate pool.
    archive = build_archive(bases, archive_cfg, base_seed)
    archive_hash = archive["archiveHash"]
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"스모크 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 3. Policy + seed batch id.
    policy = RandomPolicy(policy_cfg)
    seed_batch = seed_batch_id(base_seed, 1, policy.policy_version(), H)

    # 4. Dry run: emit the plan, write nothing.
    if dry_run:
        log_ko(
            _logger,
            "드라이런 요약: "
            f"policy={SMOKE_POLICY}, base_seed={base_seed}, H={H}, k={k}, "
            f"pool_size={pool_size}, archiveHash={archive_hash[:12]}, "
            f"poolHash={pool_hash[:12]}, configHash={config_hash[:12]}, "
            f"seedBatchId={seed_batch[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "policy": SMOKE_POLICY,
            "config": run_params,
            "configHash": config_hash,
            "archiveHash": archive_hash,
            "poolHash": pool_hash,
            "seedBatchId": seed_batch,
        }

    # 6. Run one RANDOM episode and compute trace-only metrics.
    trace = run_episode(pool, policy, base_seed, H, k, bases_cfg=bases)
    metrics = compute_all(trace)
    trace_hash = trace.trace_hash()

    # 7. Slate hash over every round's slate.
    rounds = trace.rounds()
    slate_hash = canonical_hash([r["slate"] for r in rounds])

    # Per-round, system-level summaries only (no forbidden fields).
    rounds_summary = [
        {
            "selectedCardId": r["selectedCardId"],
            "complexityBand": r["complexityBand"],
            "salienceScore": r["salienceScore"],
        }
        for r in rounds
    ]

    # 8. Results body + output hash.
    results_body = {
        "policy": SMOKE_POLICY,
        "config": run_params,
        "metrics": metrics,
        "rounds": rounds_summary,
    }
    output_hash = canonical_hash(results_body)

    # 9. Report (reportHash is computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": SMOKE_SCHEMA,
        "schemaVersion": SMOKE_SCHEMA_VERSION,
        "policy": SMOKE_POLICY,
        "phaseNote": (
            "Phase 1 smoke runs RANDOM only; TRACE_GREEDY activates in Phase 2."
        ),
        "config": run_params,
        "seedBatchId": seed_batch,
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
        "metrics": metrics,
        "rounds": rounds_summary,
    }
    report_hash = canonical_hash(report)
    report["reportHash"] = report_hash

    # 10. Reproducibility pack.
    pack = ReproducibilityPack(
        configHash=config_hash,
        commitHash=_git_commit_hash(),
        archiveHash=archive_hash,
        poolHash=pool_hash,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=report_hash,
        seedBatchId=seed_batch,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 11. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"smoke_{seed_batch[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "스모크 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"traceHash={trace_hash[:12]}, seedBatchId={seed_batch[:12]}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run the smoke, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.smoke",
        description="ECHO-Bench Phase 1 RANDOM smoke runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument("--H", type=int, default=8, help="horizon (rounds)")
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=16, help="candidate pool size"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_smoke(
        base_seed=args.seed,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "스모크 드라이런 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "스모크 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"traceHash={result['traceHash'][:12]}, "
            f"metrics={result['metrics']}",
        )


if __name__ == "__main__":
    main()
