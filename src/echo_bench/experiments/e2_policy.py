"""E2 Policy Utility runner for ECHO-Bench (Task E-002, Phase 3 complete).

E2 compares all 7 policies at a fixed horizon and pool (``k=4``, ``pool=64``) on
the trace-only utility metrics, the probe-driven ``strategy_sensitivity``, and
``regret_to_oracle`` (the C-007 oracle reference).

Policy set (Phase 3 — all 7)
----------------------------
``RANDOM``, ``FIXED_LOW_TO_HIGH``, ``FIXED_BALANCED``, ``TRACE_GREEDY``,
``TRACE_LIN_UCB`` -- all trace-only (they never consume any user model, persona,
emotion, or preference signal). ``ORACLE_STRATEGY`` (C-007) is the regret
reference: its per-round achieved coordinate-novelty objective defines the oracle
reference every policy's ``regret_to_oracle`` is measured against (the oracle's
own self-regret is therefore ~0). ``PSEUDO_USER_MODEL`` (C-006) is the ONLY
policy permitted a latent vector and is kept as an explicitly **isolated contrast
baseline** (``isContrastBaseline=True`` on its row); its synthetic latent scoring
construct never feeds any trace-only policy, the pool, or any other row.

Strategy sensitivity
--------------------
For each policy we run, at a fixed seed and horizon, one episode **per strategy
probe** in :data:`echo_bench.probes.strategy_probes.PROBES` (threading the probe
as the round runner's ``select_fn``) plus one episode under the default,
controlled Phase-1 slot-0 selection. The default-selection trace is the row's
canonical trace; the per-probe traces feed
:func:`compute_all_with_oracle`, so each row carries the four trace-only
metrics + ``strategy_sensitivity`` + ``regret_to_oracle``.

Regret to oracle
----------------
We run ``ORACLE_STRATEGY`` once (default selection) to produce the oracle's
per-round reference -- ``oracle_reference_from_objectives`` of the oracle trace's
achieved coordinate-novelty objective (the same documented system-level objective
the C-007 oracle maximizes under ``PREFER_COORD_NOVELTY``). Each policy's row
then carries ``regret_to_oracle(default_trace, oracle_ref)`` via
:func:`compute_all_with_oracle`, bounded to ``[0, 1]``.

Pipeline mirrors ``experiments/smoke.py`` / ``experiments/e1_horizon.py``: load
configs, build a reproducible archive + pool, run, aggregate, hash everything,
build a :class:`ReproducibilityPack`, and write
``outputs/reports/e2_policy_<id>.json``. ``--dry-run`` validates config +
computes the available hashes and writes nothing.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
the pool, the traces, the metrics, or the report. Strategy probes are controlled
instrumented inputs, never synthetic users. The report carries only system-level
hashes and trace-only utility metrics + strategy_sensitivity; it makes no claim
about user preference, experience, emotion, wellbeing, or legal compliance, and
no real-world generalization claim.

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
from echo_bench.logging.replay_validator import inline_replay_audit
from echo_bench.metrics.aggregate import aggregate_metric_dicts
from echo_bench.metrics.compare import compare_reference_to_others
from echo_bench.metrics.utility import (
    METRIC_KEYS,
    _achieved_coordinate_novelty,
    compute_all_with_oracle,
    oracle_reference_from_objectives,
)
from echo_bench.policies.display_names import (
    DISPLAY_NAMES,
    REFERENCE_NOTE,
    display_name,
    is_reference_policy,
)
from echo_bench.policies.fixed_balanced import FixedBalancedPolicy
from echo_bench.policies.fixed_low_to_high import FixedLowToHighPolicy
from echo_bench.policies.oracle_strategy import OracleStrategyPolicy
from echo_bench.policies.pseudo_user_model import PseudoUserModelPolicy
from echo_bench.policies.random import RandomPolicy
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy
from echo_bench.probes.strategy_probes import PROBES
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "run_e2_policy",
    "main",
    "E2_POLICIES",
    "E2_METRIC_KEYS",
    "CONTRAST_BASELINE_POLICY",
    "ORACLE_POLICY",
    "CONTRAST_BASELINE_POLICIES",
    "ORACLE_POLICIES",
    "COMPARISON_REFERENCE_POLICY",
]

_logger = get_logger(__name__)

E2_SCHEMA = "echo_bench.e2_policy.report"
E2_SCHEMA_VERSION = "1"

# The full (Phase 3) E2 policy set -- all 7. Maps English policy name to
# (class, config-filename). The first five are trace-only; PSEUDO_USER_MODEL is
# an isolated contrast baseline (the only policy with a latent vector) and
# ORACLE_STRATEGY is the regret reference.
E2_POLICIES = {
    "RANDOM": (RandomPolicy, "random.yaml"),
    "FIXED_LOW_TO_HIGH": (FixedLowToHighPolicy, "fixed_low_to_high.yaml"),
    "FIXED_BALANCED": (FixedBalancedPolicy, "fixed_balanced.yaml"),
    "TRACE_GREEDY": (TraceGreedyPolicy, "trace_greedy.yaml"),
    "TRACE_LIN_UCB": (TraceLinUcbPolicy, "trace_lin_ucb.yaml"),
    "PSEUDO_USER_MODEL": (PseudoUserModelPolicy, "pseudo_user_model.yaml"),
    # C-008 strengthened contrast-baseline variants (no longer a straw man).
    "PSEUDO_USER_MODEL_DIVERSITY_REG": (
        PseudoUserModelPolicy,
        "pseudo_user_model_diversity_reg.yaml",
    ),
    "PSEUDO_USER_MODEL_SESSION_EMBEDDING": (
        PseudoUserModelPolicy,
        "pseudo_user_model_session_embedding.yaml",
    ),
    "ORACLE_STRATEGY": (OracleStrategyPolicy, "oracle_strategy.yaml"),
    # C-010 objective-specific oracle references (oracle is NOT a universal upper
    # bound; each is the reference for its own objective).
    "ORACLE_COVERAGE": (OracleStrategyPolicy, "oracle_coverage.yaml"),
    "ORACLE_DIVERSITY": (OracleStrategyPolicy, "oracle_diversity.yaml"),
}

# The coord-novelty oracle that defines the canonical ``regret_to_oracle``
# reference, and the original isolated contrast baseline. Kept singular because
# the per-seed regret reference is derived specifically from ORACLE_STRATEGY.
CONTRAST_BASELINE_POLICY = "PSEUDO_USER_MODEL"
ORACLE_POLICY = "ORACLE_STRATEGY"

# Families flagged in the report. The contrast-baseline family (all latent-vector
# policies) and the oracle family (all probe/objective-reference policies) are
# NOT trace-only; their rows are flagged so they are never read as trace-only.
CONTRAST_BASELINE_POLICIES = {
    "PSEUDO_USER_MODEL",
    "PSEUDO_USER_MODEL_DIVERSITY_REG",
    "PSEUDO_USER_MODEL_SESSION_EMBEDDING",
}
# Derived from DISPLAY_NAMES (C-014): every key in that mapping is an oracle /
# objective-specific reference policy. Using frozenset(DISPLAY_NAMES) keeps this
# set as the single source of truth — add a new oracle policy only in
# display_names.py and it is automatically reflected here.
ORACLE_POLICIES = frozenset(DISPLAY_NAMES)

# The reference policy the D-007 paired comparison statistics are computed for
# (TRACE_GREEDY vs every other policy on each reported metric).
COMPARISON_REFERENCE_POLICY = "TRACE_GREEDY"

# The reported metric keys: the four trace-only metrics + strategy_sensitivity +
# regret_to_oracle.
E2_METRIC_KEYS = tuple(METRIC_KEYS) + ("strategy_sensitivity", "regret_to_oracle")

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


def run_e2_policy(
    base_seed: int = 42,
    H: int | None = None,
    k: int = 4,
    pool_size: int = 64,
    n: int = 10,
    dry_run: bool = False,
    replay_validate: bool = True,
) -> dict:
    """Run the E2 policy-utility comparison and return a fully hashed report.

    For every policy in :data:`E2_POLICIES`, run a batch of ``n`` child seeds
    (derived from ``base_seed`` via :func:`derive_child_seeds`). For each child
    seed: derive that seed's oracle reference by running ``ORACLE_STRATEGY``
    (default selection) and taking ``oracle_reference_from_objectives`` of its
    achieved coordinate-novelty objective; then run -- at that seed and horizon --
    one episode per strategy probe (probe threaded as ``select_fn``) plus one
    episode under the default selection, and compute the four trace-only metrics +
    ``strategy_sensitivity`` + ``regret_to_oracle`` via
    :func:`compute_all_with_oracle` (the oracle's own self-regret is ~0). The
    per-seed metric dicts are aggregated into a ``stats`` block (mean ± CI via
    :func:`aggregate_metric_dicts`); a flat per-metric mean is retained for
    back-compat. ``PSEUDO_USER_MODEL`` is reported as an isolated contrast
    baseline row.

    Args:
        base_seed: base integer seed; child seeds are derived from it.
        H: horizon (rounds); defaults to the horizon config's default when None.
        k: slate size (E2 fixes ``k=4``).
        pool_size: candidate pool size (E2 main parameter is 64).
        n: number of child seeds (seed batch) per policy.
        dry_run: when true, validate config + compute config/archive/pool hashes
            and the planned policy/probe list, write no files, run no episodes.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and resolve the horizon.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    probe_names = sorted(PROBES)

    policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in E2_POLICIES.items():
        cfg = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))
        cfg["k"] = k  # the run's k overrides the config default
        policy_cfgs[name] = cfg

    run_params = {
        "base_seed": int(base_seed),
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
        "n": int(n),
        "designVersion": "e2-design-1",
        "policies": sorted(E2_POLICIES),
        "probes": probe_names,
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
            f"E2 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 3. Dry run: emit the plan, write nothing.
    if dry_run:
        log_ko(
            _logger,
            "E2 드라이런 요약: "
            f"policies={sorted(E2_POLICIES)}, probes={probe_names}, "
            f"H={H}, k={k}, pool_size={pool_size}, "
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

    # 4. Resolve the oracle class. The oracle per-round reference is now derived
    #    PER child seed inside the loop (so the regret reference matches the seed
    #    each policy is actually evaluated on): for each child seed, run
    #    ORACLE_STRATEGY (C-007) under the default selection and take
    #    oracle_reference_from_objectives of its achieved coordinate-novelty
    #    objective (the same documented system-level objective the oracle
    #    maximizes under PREFER_COORD_NOVELTY). The oracle's own self-regret is
    #    therefore ~0 on every seed.
    _oracle_cls, _oracle_cfg_file = E2_POLICIES[ORACLE_POLICY]

    # 5. Run each policy across a batch of n child seeds: per seed compute the
    #    default-selection trace + one trace per probe + a per-seed oracle
    #    reference, then aggregate (mean ± CI) over the seed batch.
    child_seeds = derive_child_seeds(base_seed, n)
    table: List[Dict[str, Any]] = []
    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []
    seed_batch_ids: Dict[str, str] = {}
    # Per-policy per-seed metric vectors, aligned by the shared child-seed batch,
    # retained for the paired policy-comparison statistics (Task D-007).
    per_seed_by_policy: Dict[str, List[Dict[str, Any]]] = {}

    for name in sorted(E2_POLICIES):
        cls, _cfg_file = E2_POLICIES[name]
        cfg = policy_cfgs[name]
        policy = cls(cfg)
        batch_id = seed_batch_id(base_seed, n, policy.policy_version(), H)
        seed_batch_ids[name] = batch_id

        is_contrast_baseline = name in CONTRAST_BASELINE_POLICIES
        is_oracle = name in ORACLE_POLICIES
        # Only the coord-novelty ORACLE_STRATEGY defines the canonical
        # regret_to_oracle reference; the objective-specific oracles
        # (ORACLE_COVERAGE/ORACLE_DIVERSITY) are evaluated AGAINST it like any
        # other row (their coord-novelty regret is expected > 0 — they optimize a
        # different objective, underscoring that the oracle is objective-specific).
        is_reference_oracle = name == ORACLE_POLICY
        # Trace-only iff it consumes no latent/oracle signal. The contrast
        # baseline family (latent vector) and the oracle family (probe/objective
        # access) are NOT trace-only; their rows are flagged so they are never
        # read as such.
        trace_only = not (is_contrast_baseline or is_oracle)

        per_seed_metrics: List[Dict[str, Any]] = []
        default_trace_hashes: List[str] = []
        for child_seed in child_seeds:
            # Canonical (default-selection) trace -- the controlled Phase-1
            # slot-0 rule, identical to E1's selection.
            # Must be computed BEFORE the oracle reference so the is_oracle
            # short-circuit can derive the reference from the already-run trace.
            default_trace = run_episode(pool, policy, child_seed, H, k, bases)

            # Per-seed oracle reference.  When this row IS the coord-novelty
            # reference oracle, default_trace is already that oracle's trace at
            # this seed (same policy, same seed → same trace), so derive the
            # reference directly from it instead of running a redundant episode.
            if is_reference_oracle:
                oracle_seed_ref = oracle_reference_from_objectives(
                    _achieved_coordinate_novelty(default_trace)
                )
            else:
                oracle_seed_policy = _oracle_cls(policy_cfgs[ORACLE_POLICY])
                oracle_seed_trace = run_episode(
                    pool, oracle_seed_policy, child_seed, H, k, bases
                )
                oracle_seed_ref = oracle_reference_from_objectives(
                    _achieved_coordinate_novelty(oracle_seed_trace)
                )
            default_trace_hashes.append(default_trace.trace_hash())
            all_trace_hashes.append(default_trace.trace_hash())
            all_slate_hashes.append(
                canonical_hash([r["slate"] for r in default_trace.rounds()])
            )

            # One trace per controlled strategy probe (probe = select_fn).
            # Probes are instrumented inputs, not synthetic users.
            traces_by_probe: Dict[str, Any] = {}
            for probe_name in probe_names:
                probe = PROBES[probe_name]
                probe_trace = run_episode(
                    pool,
                    policy,
                    child_seed,
                    H,
                    k,
                    bases,
                    select_fn=lambda slate, trace, seed, _p=probe: _p.select(
                        slate, trace, seed
                    ),
                )
                traces_by_probe[probe_name] = probe_trace
                all_trace_hashes.append(probe_trace.trace_hash())
                all_slate_hashes.append(
                    canonical_hash([r["slate"] for r in probe_trace.rounds()])
                )

            per_seed_metrics.append(
                compute_all_with_oracle(
                    default_trace, traces_by_probe, oracle_seed_ref
                )
            )

        stats = aggregate_metric_dicts(per_seed_metrics, E2_METRIC_KEYS)
        per_seed_by_policy[name] = per_seed_metrics
        row = {
            "policy": name,
            "policyVersion": policy.policy_version(),
            "seedBatchId": batch_id,
            "n": int(n),
            "traceOnly": trace_only,
            "isContrastBaseline": is_contrast_baseline,
            "isOracle": is_oracle,
            "defaultTraceHashes": default_trace_hashes,
            "stats": stats,
        }
        # Keep a flat mean per metric for back-compat with existing readers.
        for key in E2_METRIC_KEYS:
            row[key] = stats[key]["mean"]
        # C-014 display-name layer: oracle rows get a mapped display name and
        # the reference note; non-oracle rows carry displayName == policy name
        # with no referenceNote (so the field is only present where meaningful).
        row["displayName"] = display_name(name)
        if is_oracle:  # already computed at row-start; avoids a second predicate call
            row["referenceNote"] = REFERENCE_NOTE
        table.append(row)

        log_ko(
            _logger,
            "E2 정책 완료: "
            f"policy={name}, seedBatchId={batch_id[:12]}, n={n}, "
            f"strategy_sensitivity_mean={stats['strategy_sensitivity']['mean']:.4f}, "
            f"regret_to_oracle_mean={stats['regret_to_oracle']['mean']:.4f}, "
            f"coverage_mean={stats['coordinate_coverage']['mean']:.4f}, "
            f"isContrastBaseline={is_contrast_baseline}",
        )

    # A single experiment-level seedBatchId summarizing the whole comparison.
    exp_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "H": H,
            "k": k,
            "n": int(n),
            "policies": sorted(E2_POLICIES),
            "probes": probe_names,
            "perPolicySeedBatchIds": seed_batch_ids,
        }
    )

    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 6. Results body + output hash. The oracle reference is now derived per
    #    child seed inside the loop (designVersion e2-design-1), so there is no
    #    single pre-loop oracle reference hash; n enters the body for provenance.
    # Paired policy-comparison statistics (Task D-007): TRACE_GREEDY vs every
    # other policy across the reported metrics, over the shared child-seed batch
    # (paired permutation tests + per-metric multiple-comparison correction).
    # Deterministic, so the inline replay re-run reproduces it bit-identically.
    comparisons = None
    if COMPARISON_REFERENCE_POLICY in per_seed_by_policy:
        comparisons = compare_reference_to_others(
            per_seed_by_policy,
            E2_METRIC_KEYS,
            reference=COMPARISON_REFERENCE_POLICY,
        )

    results_body = {
        "config": run_params,
        "metricKeys": list(E2_METRIC_KEYS),
        "n": int(n),
        "table": table,
        "comparisons": comparisons,
    }
    output_hash = canonical_hash(results_body)

    # 7. Report (reportHash is computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": E2_SCHEMA,
        "schemaVersion": E2_SCHEMA_VERSION,
        "experiment": "E2_POLICY_UTILITY",
        "phaseNote": (
            "Phase 3 complete: E2 compares all 7 policies (RANDOM, "
            "FIXED_LOW_TO_HIGH, FIXED_BALANCED, TRACE_GREEDY, TRACE_LIN_UCB, "
            "PSEUDO_USER_MODEL, ORACLE_STRATEGY) on the four trace-only metrics + "
            "strategy_sensitivity + regret_to_oracle. regret_to_oracle is measured "
            "against the ORACLE_STRATEGY per-round coordinate-novelty reference "
            "(oracle self-regret ~0). PSEUDO_USER_MODEL is an ISOLATED CONTRAST "
            "BASELINE (the only latent-vector policy, flagged isContrastBaseline) "
            "and its synthetic latent scoring construct never feeds any trace-only "
            "policy, the pool, or any other row. Strategy probes are controlled "
            "instrumented inputs, not synthetic users. System-level metrics over a "
            "controlled testbed; no real-world generalization claim."
        ),
        "config": run_params,
        "metricKeys": list(E2_METRIC_KEYS),
        "contrastBaselinePolicy": CONTRAST_BASELINE_POLICY,
        "oraclePolicy": ORACLE_POLICY,
        # C-014: display name for the canonical oracle + the reference note so
        # paper artifacts are not misread as global upper bounds.
        "oraclePolicyDisplayName": display_name(ORACLE_POLICY),
        "oracleNote": REFERENCE_NOTE,
        "seedBatchId": exp_seed_batch_id,
        "perPolicySeedBatchIds": seed_batch_ids,
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
        "table": table,
        "comparisons": comparisons,
    }
    report_hash = canonical_hash(report)
    report["reportHash"] = report_hash

    # 8. Reproducibility pack.
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

    # 8b. Inline replay validation (Task E-012): re-run once from config+seed and
    #     confirm the hash chain reproduces EXACTLY. Attached AFTER reportHash so
    #     it is not part of the hashed body; the re-run sets replay_validate=False
    #     to avoid unbounded recursion.
    if replay_validate:
        report["replayAudit"] = inline_replay_audit(
            report,
            run_e2_policy,
            dict(
                base_seed=base_seed,
                H=H,
                k=k,
                pool_size=pool_size,
                n=n,
                dry_run=False,
                replay_validate=False,
            ),
        )

    # 9. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"e2_policy_{exp_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "E2 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={exp_seed_batch_id[:12]}, rows={len(table)}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run E2, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.e2_policy",
        description="ECHO-Bench E2 policy-utility runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument(
        "--H", type=int, default=None, help="horizon (default: config default)"
    )
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=64, help="candidate pool size"
    )
    parser.add_argument(
        "--n", type=int, default=10, help="child seeds per policy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_e2_policy(
        base_seed=args.seed,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        n=args.n,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "E2 드라이런 완료: "
            f"policies={result['config']['policies']}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "E2 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"rows={len(result['table'])}",
        )


if __name__ == "__main__":
    main()
