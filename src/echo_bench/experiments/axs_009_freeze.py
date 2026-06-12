"""AXS-009: Freeze 실험 러너 (AXS-P0 T4).

4개 freeze 설정 (freeze_at_1/quarter/half/none) 에 대해 두 메트릭 측정:
- slate_excess_nmi (run_arm_family에서 직접)
- post_freeze_incremental_divergence = NMI(full) − NMI(prefix[:f])

Guardrails
----------
- Trace-only 정책만 사용; user_id/persona/emotion/preference 벡터 없음.
- 수치는 모두 plain Python float (numpy scalar 금지).
- 런타임 로그는 한국어; 식별자·키·경로는 영어.

All identifiers, config keys, and paths stay English; runtime log messages
are Korean per the project logging convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from echo_bench.env.horizon import default_h, load_horizon
from echo_bench.experiments.axs_common import (
    bootstrap_block,
    build_arm_entry,
    build_axs_report,
    dry_run_plan,
    load_default_configs,
    make_axs_arg_parser,
    parse_base_seeds,
    register_report,
    reportable_block,
    run_arm_family,
    write_report,
)
from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.leakage import DEFAULT_NULL_PERMUTATIONS
from echo_bench.metrics.separability import channel_separated_separability
from echo_bench.policies.axs_ucb import AxsUcbPolicy, TraceView
from echo_bench.policies.random import RandomPolicy

__all__ = ["run_axs_009", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc if isinstance(doc, dict) else {}


def _resolve_h(H: Optional[int]) -> int:
    if H is not None:
        return H
    try:
        return default_h(load_horizon(_HORIZON_CFG_PATH))
    except Exception:
        return 8


def _compute_prefix_nmi(
    rounds_by_probe: Dict[str, List[Dict[str, Any]]],
    freeze_round: int,
    n_permutations: int,
) -> float:
    """prefix[:freeze_round] roundsByProbe 에서 slate_excess_nmi 재계산.

    run_arm_family 와 동일한 channel_separated_separability 호출 방식.
    TraceView(rounds[:f]) 로 프로브별 trace 를 잘라 전달.
    """
    # roundsByProbe 키가 name-sorted probe 이름
    prefix_traces: Dict[str, Any] = {}
    for probe_name, rounds in rounds_by_probe.items():
        prefix_traces[probe_name] = TraceView(rounds[:freeze_round])

    channel_sep = channel_separated_separability(
        prefix_traces,
        n_permutations=n_permutations,
    )
    return float(channel_sep["slate_excess_nmi"])


def _arm_freeze_round(arm_id: str, H: int) -> Optional[int]:
    """arm_id → 해당 arm 의 freeze_round 값."""
    if arm_id == "freeze_at_1":
        return 1
    elif arm_id == "freeze_at_quarter":
        return max(1, H // 4)
    elif arm_id == "freeze_at_half":
        return max(1, H // 2)
    elif arm_id == "freeze_none":
        return None
    else:
        raise ValueError(f"알 수 없는 arm_id: {arm_id!r}")


def run_axs_009(
    base_seeds: Sequence[int],
    *,
    H: Optional[int] = None,
    k: int = 4,
    pool_size: int = 64,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    replay_mode: str = "first_family",
    replay_sample_size: int = 2,
    prereg_path: Any = _PREREG_PATH,
    git_runner: Optional[Callable[[List[str]], str]] = None,
    dry_run: bool = False,
    reports_dir: Optional[Path] = None,
    register_ledger: bool = False,
) -> Dict[str, Any]:
    """AXS-009 freeze 실험 실행.

    4개 freeze 설정에 대해 slate_excess_nmi 와
    post_freeze_incremental_divergence 보고.

    Returns:
        dry_run=True 이면 계획 dict, 그 외 리포트 dict.
    """
    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _REPORTS_DIR

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_yaml(_AXS_UCB_CFG_PATH)

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-009",
    }

    if dry_run:
        log_ko(_logger, f"AXS-009 드라이런: seeds={list(base_seeds)}, H={H_eff}")
        return dry_run_plan("AXS-009", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-009 실험 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )

    ARM_IDS = ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]
    freeze_rounds = {arm_id: _arm_freeze_round(arm_id, H_eff) for arm_id in ARM_IDS}

    log_ko(_logger, f"AXS-009 freeze 설정: {freeze_rounds}")

    # ---- per-arm, per-family runs ----
    # arm_id -> family -> raw block (includes roundsByProbe)
    arm_raw: Dict[str, Dict[str, Dict[str, Any]]] = {arm_id: {} for arm_id in ARM_IDS}
    random_raw_by_family: Dict[str, Dict[str, Any]] = {}

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-009 패밀리 실행: seed={seed}")

        for arm_id in ARM_IDS:
            fr = freeze_rounds[arm_id]
            arm_cfg = dict(base_cfg)
            arm_cfg["k"] = k
            arm_cfg["freeze_round"] = fr

            raw = run_arm_family(
                lambda cfg=arm_cfg: AxsUcbPolicy(dict(cfg)),
                seed,
                H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
                bases=bases, archive_cfg=archive_cfg,
            )
            arm_raw[arm_id][fam] = raw

        # RANDOM baseline — stored per family for single-source binding
        random_raw_by_family[fam] = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )

    # ---- compute post_freeze_incremental_divergence ----
    # divergence[arm_id][fam] = nmi_full - nmi_prefix
    families = [str(s) for s in base_seeds]
    divergence: Dict[str, Dict[str, float]] = {}
    for arm_id in ARM_IDS:
        divergence[arm_id] = {}
        fr = freeze_rounds[arm_id]
        for fam in families:
            raw = arm_raw[arm_id][fam]
            nmi_full = float(raw["slate_excess_nmi"])
            if fr is None:
                # freeze_none: divergence ≡ 0.0
                div = 0.0
            else:
                rounds_by_probe = raw["roundsByProbe"]
                nmi_prefix = _compute_prefix_nmi(rounds_by_probe, fr, n_permutations)
                div = float(nmi_full - nmi_prefix)
            divergence[arm_id][fam] = div
            log_ko(
                _logger,
                f"AXS-009 divergence: arm={arm_id}, family={fam}, "
                f"nmi_full={nmi_full:+.6f}, div={div:+.6f}",
            )

    # ---- family_blocks: single authoritative source for all values ----
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        random_raw = random_raw_by_family[fam]
        block = {
            **reportable_block(arm_raw["freeze_none"][fam]),
            # RANDOM baseline
            "random_coordinate_coverage_mean": float(random_raw["coordinate_coverage_mean"]),
            "random_archiveHash": random_raw["archiveHash"],
            "random_poolHash": random_raw["poolHash"],
            "random_traceHashes": random_raw["traceHashes"],
        }
        # Include all arms' key metrics
        for arm_id in ARM_IDS:
            block[f"{arm_id}_slate_excess_nmi"] = float(
                arm_raw[arm_id][fam]["slate_excess_nmi"]
            )
            block[f"{arm_id}_post_freeze_incremental_divergence"] = float(
                divergence[arm_id][fam]
            )
            block[f"{arm_id}_coordinate_coverage_mean"] = float(
                arm_raw[arm_id][fam]["coordinate_coverage_mean"]
            )
        family_blocks[fam] = block

    # ---- derive all report values exclusively from family_blocks ----
    random_coverage_mean = float(
        sum(family_blocks[fam]["random_coordinate_coverage_mean"] for fam in families)
        / len(families)
    )

    # ---- build arm entries ----
    metric_nmi = "slate_excess_nmi"
    metric_div = "post_freeze_incremental_divergence"

    arms_report: Dict[str, Any] = {}
    for arm_id in ARM_IDS:
        nmi_by_fam = {fam: float(family_blocks[fam][f"{arm_id}_slate_excess_nmi"])
                      for fam in families}
        div_by_fam = {fam: float(family_blocks[fam][f"{arm_id}_post_freeze_incremental_divergence"])
                      for fam in families}
        cov_mean = float(
            sum(family_blocks[fam][f"{arm_id}_coordinate_coverage_mean"] for fam in families)
            / len(families)
        )

        # Build base entry via build_arm_entry (handles degenerate logic)
        entry = build_arm_entry(
            metric_nmi, nmi_by_fam, cov_mean, random_coverage_mean,
            degenerate_reason_prefix=arm_id,
        )

        # Merge divergence into perFamily
        for fam in entry["perFamily"]:
            entry["perFamily"][fam][metric_div] = float(div_by_fam[fam])

        # Add divergence bootstrap block
        div_bs = bootstrap_block(
            {fam: float(div_by_fam[fam]) for fam in div_by_fam},
            key=metric_div,
        )
        entry["bootstrap"][metric_div] = div_bs

        arms_report[arm_id] = entry

    def recompute_fn(family: str) -> Dict[str, Any]:
        seed = int(family)
        result: Dict[str, Any] = {}
        primary_raw = None
        for arm_id in ARM_IDS:
            fr = freeze_rounds[arm_id]
            arm_cfg = dict(base_cfg)
            arm_cfg["k"] = k
            arm_cfg["freeze_round"] = fr
            raw = run_arm_family(
                lambda cfg=arm_cfg: AxsUcbPolicy(dict(cfg)),
                seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
                bases=bases, archive_cfg=archive_cfg,
            )
            if arm_id == "freeze_none":
                primary_raw = raw
            nmi_full = float(raw["slate_excess_nmi"])
            if fr is None:
                div = 0.0
            else:
                nmi_prefix = _compute_prefix_nmi(raw["roundsByProbe"], fr, n_permutations)
                div = float(nmi_full - nmi_prefix)
            result[f"{arm_id}_slate_excess_nmi"] = nmi_full
            result[f"{arm_id}_post_freeze_incremental_divergence"] = div
            result[f"{arm_id}_coordinate_coverage_mean"] = float(raw["coordinate_coverage_mean"])
        r = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        result["random_coordinate_coverage_mean"] = float(r["coordinate_coverage_mean"])
        result["random_archiveHash"] = r["archiveHash"]
        result["random_poolHash"] = r["poolHash"]
        result["random_traceHashes"] = r["traceHashes"]
        assert primary_raw is not None
        return {**reportable_block(primary_raw), **result}

    body_extra = {
        "arms": arms_report,
        "baselines": {
            "RANDOM": {"coordinate_coverage_mean": random_coverage_mean},
        },
        "freezeRounds": {arm_id: freeze_rounds[arm_id] for arm_id in ARM_IDS},
    }

    report = build_axs_report(
        "AXS-009",
        body_extra=body_extra,
        family_blocks=family_blocks,
        recompute_fn=recompute_fn,
        run_params=run_params,
        prereg_path=prereg_path,
        replay_mode=replay_mode,
        replay_sample_size=replay_sample_size,
        git_runner=git_runner,
    )

    # Log summary
    for arm_id in ARM_IDS:
        nmi_mean = report["arms"][arm_id]["bootstrap"][metric_nmi]["mean"]
        div_mean = report["arms"][arm_id]["bootstrap"][metric_div]["mean"]
        log_ko(
            _logger,
            f"AXS-009 {arm_id}: nmi_mean={nmi_mean:+.6f}, div_mean={div_mean:+.6f}",
        )

    log_ko(
        _logger,
        f"AXS-009 완료: reportHash={report['reportHash'][:12]}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-009] 완료\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        + "\n".join(
            f"  {arm_id} slate_excess_nmi mean: "
            f"{report['arms'][arm_id]['bootstrap'][metric_nmi]['mean']:+.6f}, "
            f"divergence mean: "
            f"{report['arms'][arm_id]['bootstrap'][metric_div]['mean']:+.6f}"
            for arm_id in ARM_IDS
        )
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_009_freeze [options]
    """
    parser = make_axs_arg_parser("AXS-009 freeze 실험 (4개 freeze 설정 비교)")
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_009(
        base_seeds_list,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        n_permutations=args.n_permutations,
        replay_mode=args.replay_mode,
        replay_sample_size=args.replay_sample_size,
        prereg_path=_PREREG_PATH,
        git_runner=None,
        dry_run=args.dry_run,
        reports_dir=Path(args.reports_dir),
        register_ledger=args.register_ledger,
    )


if __name__ == "__main__":
    main()
