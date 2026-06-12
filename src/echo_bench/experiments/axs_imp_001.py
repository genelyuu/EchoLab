"""AXS-IMP-001: Freeze-imprint 실험 러너 (AXS-V3 N4).

4개 freeze 설정 (freeze_at_1/quarter/half/none) 에 대해 slate_excess_nmi 측정.
v1 AXS-009 의 v3 버전 — divergence 메트릭 제거, 사전등록 v3 draft 기본.

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
from echo_bench.policies.axs_ucb import AxsUcbPolicy
from echo_bench.policies.random import RandomPolicy

__all__ = ["run_axs_imp_001", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v3_draft.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

_ARM_IDS = ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]


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


def _arm_freeze_round(arm_id: str, H: int) -> Optional[int]:
    """arm_id → freeze_round 값 반환."""
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


def run_axs_imp_001(
    base_seeds: Sequence[int],
    *,
    H: Optional[int] = None,
    k: int = 4,
    pool_size: int = 64,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    replay_mode: str = "first_family",
    replay_sample_size: int = 2,
    prereg_path: Any = None,
    git_runner: Optional[Callable[[List[str]], str]] = None,
    dry_run: bool = False,
    reports_dir: Optional[Path] = None,
    register_ledger: bool = False,
) -> Dict[str, Any]:
    """AXS-IMP-001 freeze-imprint 실험 실행.

    4개 freeze 설정에 대해 slate_excess_nmi 보고.
    v3 prereg 계약: arms + baselines + freezeRounds; delta 블록 없음(게이트가 계산).

    Args:
        base_seeds: 평가 패밀리 base-seed 목록.
        H: 라운드 수 (None → horizon.yaml 기본값).
        k: 슬레이트 크기.
        pool_size: 후보 풀 크기.
        n_permutations: D-015 null 치환 수.
        replay_mode: 인라인 리플레이 감사 모드.
        replay_sample_size: sampled_families 모드 재계산 패밀리 수.
        prereg_path: 사전등록 JSON 경로 (None → 기본 v3 draft).
        git_runner: 테스트용 git 명령 인젝터.
        dry_run: True 이면 계획만 반환, 파일 미작성.
        reports_dir: 리포트 출력 디렉토리.
        register_ledger: True 이면 원장에 등록.

    Returns:
        dry_run=True 이면 계획 dict, 그 외 리포트 dict.
    """
    eff_prereg = prereg_path
    if eff_prereg is None:
        eff_prereg = _PREREG_PATH

    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _REPORTS_DIR

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_yaml(_AXS_UCB_CFG_PATH)

    freeze_rounds = {arm_id: _arm_freeze_round(arm_id, H_eff) for arm_id in _ARM_IDS}

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-IMP-001",
        "base_seeds": [int(s) for s in base_seeds],
        "freezeRounds": {arm_id: freeze_rounds[arm_id] for arm_id in _ARM_IDS},
    }

    if dry_run:
        log_ko(_logger, f"AXS-IMP-001 드라이런: seeds={list(base_seeds)}, H={H_eff}")
        return dry_run_plan("AXS-IMP-001", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-IMP-001 실험 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )
    log_ko(_logger, f"AXS-IMP-001 freeze 설정: {freeze_rounds}")

    # ---- per-arm, per-family runs ----
    arm_raw: Dict[str, Dict[str, Dict[str, Any]]] = {arm_id: {} for arm_id in _ARM_IDS}
    random_raw_by_family: Dict[str, Dict[str, Any]] = {}

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-IMP-001 패밀리 실행: seed={seed}")

        for arm_id in _ARM_IDS:
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
            log_ko(
                _logger,
                f"AXS-IMP-001 arm={arm_id} family={fam}: "
                f"slate_excess_nmi={raw['slate_excess_nmi']:+.6f}",
            )

        random_raw_by_family[fam] = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )

    # ---- family_blocks: single authoritative source ----
    families = [str(s) for s in base_seeds]
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        random_raw = random_raw_by_family[fam]
        block = {
            **reportable_block(arm_raw["freeze_none"][fam]),
            "random_coordinate_coverage_mean": float(random_raw["coordinate_coverage_mean"]),
            "random_archiveHash": random_raw["archiveHash"],
            "random_poolHash": random_raw["poolHash"],
            "random_traceHashes": random_raw["traceHashes"],
        }
        for arm_id in _ARM_IDS:
            block[f"{arm_id}_slate_excess_nmi"] = float(
                arm_raw[arm_id][fam]["slate_excess_nmi"]
            )
            block[f"{arm_id}_coordinate_coverage_mean"] = float(
                arm_raw[arm_id][fam]["coordinate_coverage_mean"]
            )
            block[f"{arm_id}_traceHashes"] = arm_raw[arm_id][fam]["traceHashes"]
        family_blocks[fam] = block

    # ---- derive all report values exclusively from family_blocks ----
    metric = "slate_excess_nmi"

    random_coverage_mean = float(
        sum(family_blocks[fam]["random_coordinate_coverage_mean"] for fam in families)
        / len(families)
    )

    arms_report: Dict[str, Any] = {}
    for arm_id in _ARM_IDS:
        nmi_by_fam = {
            fam: float(family_blocks[fam][f"{arm_id}_slate_excess_nmi"])
            for fam in families
        }
        cov_mean = float(
            sum(family_blocks[fam][f"{arm_id}_coordinate_coverage_mean"] for fam in families)
            / len(families)
        )
        entry = build_arm_entry(
            metric, nmi_by_fam, cov_mean, random_coverage_mean,
            degenerate_reason_prefix=arm_id,
        )
        arms_report[arm_id] = entry

    def recompute_fn(family: str) -> Dict[str, Any]:
        seed = int(family)
        result: Dict[str, Any] = {}
        primary_raw = None
        for arm_id in _ARM_IDS:
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
            result[f"{arm_id}_slate_excess_nmi"] = float(raw["slate_excess_nmi"])
            result[f"{arm_id}_coordinate_coverage_mean"] = float(raw["coordinate_coverage_mean"])
            result[f"{arm_id}_traceHashes"] = raw["traceHashes"]
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
        "freezeRounds": {arm_id: freeze_rounds[arm_id] for arm_id in _ARM_IDS},
    }

    report = build_axs_report(
        "AXS-IMP-001",
        body_extra=body_extra,
        family_blocks=family_blocks,
        recompute_fn=recompute_fn,
        run_params=run_params,
        prereg_path=eff_prereg,
        replay_mode=replay_mode,
        replay_sample_size=replay_sample_size,
        git_runner=git_runner,
    )

    # Korean summary
    for arm_id in _ARM_IDS:
        nmi_mean = report["arms"][arm_id]["bootstrap"][metric]["mean"]
        log_ko(
            _logger,
            f"AXS-IMP-001 {arm_id}: slate_excess_nmi_mean={nmi_mean:+.6f}",
        )

    log_ko(
        _logger,
        f"AXS-IMP-001 완료: reportHash={report['reportHash'][:12]}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-IMP-001] 완료\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        + "\n".join(
            f"  {arm_id} slate_excess_nmi mean: "
            f"{report['arms'][arm_id]['bootstrap'][metric]['mean']:+.6f}"
            for arm_id in _ARM_IDS
        )
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_imp_001 [options]
    """
    parser = make_axs_arg_parser("AXS-IMP-001 freeze-imprint 실험 (4개 freeze 설정 비교)")
    parser.add_argument(
        "--prereg",
        type=str,
        default=str(_PREREG_PATH),
        dest="prereg",
        help="사전등록 JSON 경로 (기본: v3 draft)",
    )
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_imp_001(
        base_seeds_list,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        n_permutations=args.n_permutations,
        replay_mode=args.replay_mode,
        replay_sample_size=args.replay_sample_size,
        prereg_path=args.prereg,
        git_runner=None,
        dry_run=args.dry_run,
        reports_dir=Path(args.reports_dir),
        register_ledger=args.register_ledger,
    )


if __name__ == "__main__":
    main()
