"""E3 Leakage / Robustness / Replay Audit runner for ECHO-Bench (Task E-003).

E3 is the Phase 3 audit experiment. It audits the policy set along three
controlled, deterministic axes and assembles one fully hashed report:

(a) LEAKAGE (PROXY)
    For a representative policy set, build per-probe traces (each probe threaded
    as the round runner's ``select_fn``) and compute
    :func:`echo_bench.metrics.leakage.leakage_proxy` over the probe-keyed trace
    family. The value is reported **explicitly as a PROXY** (the leakage section
    carries ``isProxy=True`` and the
    :data:`echo_bench.metrics.leakage.PROXY_DISCLAIMER`): it measures how the
    observable slate/selection distribution co-varies with the controlled probe
    identity in this controlled testbed. It is **NOT** a privacy guarantee,
    anonymity proof, identifiability bound, or legal/compliance claim.

(b) ROBUSTNESS
    For a policy, run a baseline episode on the unmodified pool and one episode
    per controlled fault in :data:`echo_bench.metrics.robustness.FAULTS`
    (``pool_shrink``, ``basis_dropout``, ``salience_perturb``). Compute
    :func:`echo_bench.metrics.utility.compute_all` on the baseline and each
    faulted trace, then :func:`echo_bench.metrics.robustness.robustness_score`
    per fault. These are **controlled, fully-specified faults**, not real-world
    shift; the metric is a system-level sensitivity over the controlled testbed.

(c) REPLAY AUDIT
    Call :func:`echo_bench.logging.replay_validator.validate_replay` on
    :func:`echo_bench.experiments.smoke.run_smoke` (re-run twice from
    config + seed) and embed ``{replayable, first_divergent}``. The audit
    **FAILS LOUDLY** -- it reports ``replayable=False`` and surfaces the first
    divergent hash key -- if the run does not replay; a divergence is never
    hidden.

Pipeline mirrors ``experiments/e2_policy.py`` / ``experiments/s3_coordinate_scramble.py``:
load configs, build a reproducible archive + pool, run the three audit sections,
hash everything (the full chain plus a per-section ``outputHash``), record a
``seedBatchId``, build a :class:`ReproducibilityPack`, and write
``outputs/reports/e3_audit_<id>.json``. ``--dry-run`` validates config + computes
the available hashes and the planned audit sections, runs no episodes, and writes
nothing. Re-running the same config + seed reproduces an identical ``reportHash``.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
the pool, the traces, the metrics, or the report. The leakage value is a PROXY,
never a privacy/anonymity/legal claim; the faults are controlled, not
real-world; the metrics are system-level statistics over a controlled testbed
and make no real-world generalization claim.

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
from echo_bench.env.seed_batch import seed_batch_id
from echo_bench.experiments.smoke import run_smoke
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.logging.replay_validator import validate_replay
from echo_bench.metrics.leakage import (
    IS_PROXY,
    LEAKAGE_RATIO_FLOOR,
    PROXY_DISCLAIMER,
    leakage_delta_vs_random,
    leakage_proxy_with_metadata,
    utility_per_leakage,
)
from echo_bench.metrics.robustness import (
    FAULTS,
    ROBUSTNESS_DIRECTION,
    robustness_score_with_metadata,
)
from echo_bench.metrics.utility import (
    CORE_METRIC_KEYS,
    compute_all,
    coordinate_coverage,
)
from echo_bench.policies.fixed_balanced import FixedBalancedPolicy
from echo_bench.policies.fixed_low_to_high import FixedLowToHighPolicy
from echo_bench.policies.random import RandomPolicy
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy
from echo_bench.probes.strategy_probes import PROBES, get_probe
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "run_e3_audit",
    "main",
    "E3_LEAKAGE_POLICIES",
    "E3_LEAKAGE_DELTA_REFERENCE",
    "E3_ROBUSTNESS_POLICY",
    "FAULT_PARAMS",
]

_logger = get_logger(__name__)

E3_SCHEMA = "echo_bench.e3_audit.report"
E3_SCHEMA_VERSION = "1"

# The representative trace-only policy set used for the LEAKAGE-proxy section.
# (Trace-only policies whose observable behaviour varies under the controlled
# probes; the contrast baseline / oracle are not part of the leakage proxy set.)
E3_LEAKAGE_POLICIES = {
    "RANDOM": (RandomPolicy, "random.yaml"),
    "FIXED_LOW_TO_HIGH": (FixedLowToHighPolicy, "fixed_low_to_high.yaml"),
    "FIXED_BALANCED": (FixedBalancedPolicy, "fixed_balanced.yaml"),
    "TRACE_GREEDY": (TraceGreedyPolicy, "trace_greedy.yaml"),
    "TRACE_LIN_UCB": (TraceLinUcbPolicy, "trace_lin_ucb.yaml"),
}

# D-011 (TRD alias D-010): the reference policy for the RELATIVE leakage claim.
# Every leakage row reports ``leakage_delta_vs_random`` = leakage(policy) -
# leakage(reference), seed-aligned; the reference's own delta is exactly 0.0.
E3_LEAKAGE_DELTA_REFERENCE = "RANDOM"

# D-011: why the leakage delta carries no confidence interval in this report.
# leakage_proxy is ONE scalar per policy (pooled NMI over the probe-keyed trace
# family at a single base seed) — there are no per-seed leakage values to pair,
# so the bootstrap CI machinery (metrics/aggregate.py) has nothing to resample.
# leakage_proxy is NOT redefined to manufacture per-seed values.
_E3_LEAKAGE_CI_UNAVAILABLE_REASON = (
    "leakage_proxy is a single scalar per policy (pooled NMI over the "
    "probe-keyed trace family at one base seed); no per-seed leakage values "
    "exist to pair, so no bootstrap CI can be computed without redefining the "
    "metric."
)

# The policy whose robustness to each controlled fault is audited.
E3_ROBUSTNESS_POLICY = ("TRACE_GREEDY", "trace_greedy.yaml")

# Controlled, fully-specified parameters for each fault transform. Documented
# constants (not config-derived) so the audit is deterministic and self-contained.
# ``basis_dropout`` drops a fixed basis label; the others take a magnitude + seed.
FAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "pool_shrink": {"frac": 0.25},
    "basis_dropout": {"drop_basis": "B1"},
    "salience_perturb": {"delta": 0.1},
}

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


def _apply_fault(name: str, pool: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Apply the controlled fault ``name`` to ``pool`` with its documented params.

    Pure: delegates to the registered :data:`FAULTS` transform with the fixed
    :data:`FAULT_PARAMS`; the seed is threaded into the seeded transforms
    (``pool_shrink`` / ``salience_perturb``) and ignored by ``basis_dropout``.
    """
    fault = FAULTS[name]
    params = dict(FAULT_PARAMS[name])
    if name == "basis_dropout":
        return fault(pool, params["drop_basis"])
    if name == "pool_shrink":
        return fault(pool, params["frac"], seed)
    if name == "salience_perturb":
        return fault(pool, params["delta"], seed)
    raise ValueError(f"E3 감사: 알 수 없는 fault 이름입니다: {name!r}")


def run_e3_audit(
    base_seed: int = 42,
    H: int | None = None,
    k: int = 4,
    pool_size: int = 64,
    dry_run: bool = False,
) -> dict:
    """Run the E3 audit (leakage proxy / robustness / replay) and return a report.

    Three deterministic, fully-hashed audit sections:

    (a) ``leakage`` -- per-policy ``leakage_proxy`` over the probe-keyed trace
        family, carried with the explicit proxy flag + disclaimer.
    (b) ``robustness`` -- baseline vs each controlled fault (FAULTS) for the
        audited policy, scored with ``robustness_score`` per fault.
    (c) ``replayAudit`` -- ``validate_replay(run_smoke, {...})`` embedding
        ``{replayable, first_divergent}``; reports ``replayable=False`` loudly on
        any hash-chain divergence (never hidden).

    Args:
        base_seed: base integer seed for every section's episodes.
        H: horizon (rounds); defaults to the horizon config's default when None.
        k: slate size (E3 fixes ``k=4``).
        pool_size: candidate pool size (E3 main parameter is 64).
        dry_run: when true, validate config + compute config/archive/pool hashes
            and the planned audit sections, write no files, run no episodes.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and resolve the horizon.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    probe_names = sorted(PROBES)
    fault_names = sorted(FAULTS)

    leakage_policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in E3_LEAKAGE_POLICIES.items():
        cfg = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))
        cfg["k"] = int(k)
        leakage_policy_cfgs[name] = cfg

    rob_name, rob_cfg_file = E3_ROBUSTNESS_POLICY
    rob_cfg = dict(_load_yaml(_POLICY_CFG_DIR / rob_cfg_file))
    rob_cfg["k"] = int(k)

    # The smoke run kwargs the replay audit re-runs twice (CPU-replayable core).
    smoke_kwargs = {
        "base_seed": int(base_seed),
        "H": int(H),
        "k": int(k),
        "pool_size": min(int(pool_size), 16),
    }

    run_params = {
        "base_seed": int(base_seed),
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
        "leakagePolicies": sorted(E3_LEAKAGE_POLICIES),
        "robustnessPolicy": rob_name,
        "probes": probe_names,
        "faults": fault_names,
        "faultParams": FAULT_PARAMS,
        "replaySmokeKwargs": smoke_kwargs,
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "leakage_policy_cfgs": leakage_policy_cfgs,
            "robustness_policy_cfg": rob_cfg,
            "run_params": run_params,
        }
    )

    # 2. Build the archive and take the deterministic candidate pool.
    archive = build_archive(bases, archive_cfg, base_seed)
    archive_hash = archive["archiveHash"]
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"E3 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # 3. Dry run: emit the plan, write nothing.
    if dry_run:
        log_ko(
            _logger,
            "E3 드라이런 요약: "
            f"leakagePolicies={sorted(E3_LEAKAGE_POLICIES)}, "
            f"robustnessPolicy={rob_name}, probes={probe_names}, "
            f"faults={fault_names}, H={H}, k={k}, pool_size={pool_size}, "
            f"archiveHash={archive_hash[:12]}, poolHash={pool_hash[:12]}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "configHash": config_hash,
            "archiveHash": archive_hash,
            "poolHash": pool_hash,
            "sections": ["leakage", "robustness", "replayAudit"],
        }

    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []

    # 4. (a) LEAKAGE PROXY section. Per policy, build the probe-keyed trace family
    #    and compute leakage_proxy (carried with the explicit proxy disclaimer).
    leakage_rows: List[Dict[str, Any]] = []
    for name in sorted(E3_LEAKAGE_POLICIES):
        cls, _cfg_file = E3_LEAKAGE_POLICIES[name]
        cfg = leakage_policy_cfgs[name]

        traces_by_probe: Dict[str, Any] = {}
        probe_versions: Dict[str, Any] = {}
        for probe_name in probe_names:
            probe = PROBES[probe_name]
            probe_versions[probe_name] = get_probe(probe_name).probe_version()
            probe_trace = run_episode(
                pool,
                cls(dict(cfg)),
                base_seed,
                H,
                int(k),
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

        leak = leakage_proxy_with_metadata(traces_by_probe, probe_versions)
        # D-011: mean coordinate_coverage over this policy's E3-aligned
        # per-probe traces — the numerator of utility_per_leakage.
        coverages = [
            coordinate_coverage(traces_by_probe[pn]) for pn in probe_names
        ]
        mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        leakage_rows.append(
            {
                "policy": name,
                "policyVersion": cls(dict(cfg)).policy_version(),
                "leakage_proxy": leak["value"],
                "isProxy": leak["isProxy"],
                "mean_coordinate_coverage": mean_coverage,
                "utility_per_leakage": utility_per_leakage(
                    mean_coverage, leak["value"]
                ),
                "traceHashes": leak["traceHashes"],
                "probeVersions": leak["probeVersions"],
            }
        )
        log_ko(
            _logger,
            "E3 누출 프록시 완료: "
            f"policy={name}, leakage_proxy={leak['value']:.6f} "
            "(이는 PROXY 이며 프라이버시/법적 보증이 아닙니다)",
        )

    # D-011 (TRD alias D-010): second pass — RELATIVE delta vs the RANDOM
    # reference (seed-aligned: every policy ran the identical seed / horizon /
    # pool / probe family above). The reference's own delta is exactly 0.0.
    # D-012 ride-along: explicit guard so a missing RANDOM reference raises a
    # clear error instead of an opaque StopIteration.
    _ref_matches = [row for row in leakage_rows if row["policy"] == E3_LEAKAGE_DELTA_REFERENCE]
    if not _ref_matches:
        raise ValueError(
            f"E3 감사: leakage_rows 에서 deltaReference 정책을 찾을 수 없습니다 "
            f"(E3_LEAKAGE_DELTA_REFERENCE={E3_LEAKAGE_DELTA_REFERENCE!r}). "
            "E3_LEAKAGE_POLICIES 에 해당 정책이 포함되어 있는지 확인하세요."
        )
    reference_row = _ref_matches[0]
    reference_leakage = reference_row["leakage_proxy"]
    for row in leakage_rows:
        row["leakage_delta_vs_random"] = leakage_delta_vs_random(
            row["leakage_proxy"], reference_leakage
        )
        log_ko(
            _logger,
            "E3 누출 상대 지표(D-011): "
            f"policy={row['policy']}, "
            f"leakage_delta_vs_random={row['leakage_delta_vs_random']:+.6f} "
            f"(기준={E3_LEAKAGE_DELTA_REFERENCE}), "
            f"utility_per_leakage={row['utility_per_leakage']:.6f} "
            f"(floor={LEAKAGE_RATIO_FLOOR}) — 상대 비교용 PROXY 통계입니다",
        )

    leakage_section = {
        "metric": "leakage_proxy",
        "isProxy": IS_PROXY,
        "disclaimer": PROXY_DISCLAIMER,
        "policies": sorted(E3_LEAKAGE_POLICIES),
        # D-011 self-describing comparison fields: the delta reference policy,
        # the documented ratio floor + utility metric, and the explicit
        # statement that no CI is structurally available (with the reason).
        "deltaReference": E3_LEAKAGE_DELTA_REFERENCE,
        "ratioFloor": LEAKAGE_RATIO_FLOOR,
        "ratioUtilityMetric": "coordinate_coverage",
        "ciAvailable": False,
        "ciUnavailableReason": _E3_LEAKAGE_CI_UNAVAILABLE_REASON,
        "probeVersions": {
            pn: get_probe(pn).probe_version() for pn in probe_names
        },
        "table": leakage_rows,
    }

    # 5. (b) ROBUSTNESS section. Baseline vs each controlled fault for the audited
    #    policy; robustness_score per fault. These are controlled faults.
    rob_cls, _rcf = E3_ROBUSTNESS_POLICY[0], E3_ROBUSTNESS_POLICY[1]
    rob_policy_cls = {
        "RANDOM": RandomPolicy,
        "FIXED_LOW_TO_HIGH": FixedLowToHighPolicy,
        "FIXED_BALANCED": FixedBalancedPolicy,
        "TRACE_GREEDY": TraceGreedyPolicy,
        "TRACE_LIN_UCB": TraceLinUcbPolicy,
    }[rob_name]

    baseline_trace = run_episode(
        pool, rob_policy_cls(dict(rob_cfg)), base_seed, H, int(k), bases
    )
    baseline_metrics = compute_all(baseline_trace)
    all_trace_hashes.append(baseline_trace.trace_hash())
    all_slate_hashes.append(
        canonical_hash([r["slate"] for r in baseline_trace.rounds()])
    )

    robustness_rows: List[Dict[str, Any]] = []
    for fault_name in fault_names:
        faulted_pool = _apply_fault(fault_name, pool, base_seed)
        faulted_trace = run_episode(
            faulted_pool, rob_policy_cls(dict(rob_cfg)), base_seed, H, int(k), bases
        )
        faulted_metrics = compute_all(faulted_trace)
        all_trace_hashes.append(faulted_trace.trace_hash())
        all_slate_hashes.append(
            canonical_hash([r["slate"] for r in faulted_trace.rounds()])
        )

        score = robustness_score_with_metadata(
            baseline_metrics, faulted_metrics, keys=CORE_METRIC_KEYS
        )
        robustness_rows.append(
            {
                "fault": fault_name,
                "faultParams": FAULT_PARAMS[fault_name],
                "faultedPoolSize": len(faulted_pool),
                # D-012: primary label is sensitivity_score; legacy alias kept for
                # backward compatibility.
                "sensitivity_score": score["value"],
                "legacyAlias": "robustness_score",
                "robustness_score": score["value"],
                "baselineTraceHash": score["baselineTraceHash"],
                "faultedTraceHash": score["faultedTraceHash"],
                "sharedKeys": score["sharedKeys"],
                "metricKeys": score["metricKeys"],
            }
        )
        log_ko(
            _logger,
            "E3 강건성 완료: "
            f"policy={rob_name}, fault={fault_name}, "
            f"sensitivity_score={score['value']:.6f} (controlled fault)",
        )

    robustness_section = {
        # D-012: primary metric label is sensitivity_score; legacy alias kept for
        # backward compatibility.
        "metric": "sensitivity_score",
        "legacyAlias": "robustness_score",
        "direction": ROBUSTNESS_DIRECTION,
        "policy": rob_name,
        "policyVersion": rob_policy_cls(dict(rob_cfg)).policy_version(),
        "note": (
            "Controlled, fully-specified fault transforms (FAULTS), NOT "
            "real-world distribution shift; system-level sensitivity over a "
            "controlled testbed. sensitivity_score (legacyAlias: robustness_score) "
            "is a SENSITIVITY magnitude: "
            + ROBUSTNESS_DIRECTION
            + "."
        ),
        "baselineTraceHash": baseline_trace.trace_hash(),
        "faults": fault_names,
        "metricKeys": list(CORE_METRIC_KEYS),
        "table": robustness_rows,
    }

    # 6. (c) REPLAY AUDIT section. Re-run the CPU-replayable smoke core twice and
    #    embed the result; report replayable=False LOUDLY on any divergence.
    replay_result = validate_replay(run_smoke, dict(smoke_kwargs))
    replay_section = {
        "validator": "validate_replay",
        "runFn": "echo_bench.experiments.smoke.run_smoke",
        "runKwargs": smoke_kwargs,
        "replayable": bool(replay_result["replayable"]),
        "first_divergent": replay_result["first_divergent"],
        "seedBatchId": replay_result["seedBatchId"],
        "chain": replay_result["chain"],
    }
    if not replay_section["replayable"]:
        log_ko(
            _logger,
            "E3 리플레이 감사 실패(LOUD): "
            f"replayable=False, first_divergent={replay_section['first_divergent']} "
            "— 재현 불가는 숨기지 않고 보고합니다",
        )
    else:
        log_ko(
            _logger,
            "E3 리플레이 감사 통과: replayable=True "
            f"(seedBatchId={str(replay_section['seedBatchId'])[:12]})",
        )

    # 7. An experiment-level seedBatchId summarizing the whole audit.
    exp_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "H": H,
            "k": k,
            "leakagePolicies": sorted(E3_LEAKAGE_POLICIES),
            "robustnessPolicy": rob_name,
            "probes": probe_names,
            "faults": fault_names,
        }
    )

    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 8. Results body + output hash.
    results_body = {
        "config": run_params,
        "leakage": leakage_section,
        "robustness": robustness_section,
        "replayAudit": replay_section,
    }
    output_hash = canonical_hash(results_body)

    # 9. Report (reportHash is computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": E3_SCHEMA,
        "schemaVersion": E3_SCHEMA_VERSION,
        "experiment": "E3_AUDIT",
        "phaseNote": (
            "Phase 3 complete: E3 audits the policy set along three controlled, "
            "deterministic axes. (a) leakage_proxy is a SYSTEM-LEVEL PROXY for how "
            "observable slate/selection distributions co-vary with controlled "
            "probe identity; it is NOT a privacy guarantee, anonymity proof, "
            "identifiability bound, or legal/compliance claim. (b) robustness "
            "uses controlled, fully-specified fault transforms (pool_shrink, "
            "basis_dropout, salience_perturb), NOT real-world shift. (c) the "
            "replay audit re-runs the CPU-replayable smoke core twice and reports "
            "replayable=False LOUDLY on any hash-chain divergence (never hidden). "
            "Strategy probes are controlled instrumented inputs, not synthetic "
            "users. System-level metrics over a controlled testbed; no real-world "
            "generalization claim."
        ),
        "config": run_params,
        "seedBatchId": exp_seed_batch_id,
        "leakage": leakage_section,
        "robustness": robustness_section,
        "replayAudit": replay_section,
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
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
        seedBatchId=exp_seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 11. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"e3_audit_{exp_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "E3 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={exp_seed_batch_id[:12]}, "
        f"replayable={replay_section['replayable']}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run E3, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.e3_audit",
        description="ECHO-Bench E3 leakage/robustness/replay audit runner.",
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
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_e3_audit(
        base_seed=args.seed,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "E3 드라이런 완료: "
            f"sections={result['sections']}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "E3 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"replayable={result['replayAudit']['replayable']}",
        )


if __name__ == "__main__":
    main()
