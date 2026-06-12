"""AXS-004c: Yoked-bonus 실험 러너 (AXS-P0 T4).

axs_ucb_default vs axs_yoked_bonus — slate_excess_nmi.
AxsYokedBonusPolicy 의 trace-독립 보너스가 분리도를 소멸시키는지 검사.

Guardrails
----------
- Trace-only 정책만 사용; user_id/persona/emotion/preference 벡터 없음.
- 수치는 모두 plain Python float (numpy scalar 금지).
- 런타임 로그는 한국어; 식별자·키·경로는 영어.

All identifiers, config keys, and paths stay English; runtime log messages
are Korean per the project logging convention.

yokedScheduleNote: The yoked schedule content is committed in git and pinned by
yokedScheduleHash (schedule_hash in config). To recover the schedule, run:
    git show <runCommit>:configs/prereg/axs_004c_yoked_schedule_v1.json
The schedule file path is NOT part of the policy_version hash — only the
schedule content hash (yokedScheduleHash) is. This ensures path-independence:
the same schedule content at any filesystem location yields identical results.
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
from echo_bench.policies.axs_ucb import AxsUcbPolicy, AxsYokedBonusPolicy
from echo_bench.policies.random import RandomPolicy

__all__ = ["run_axs_004c", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"
_DEFAULT_SCHEDULE_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_004c_yoked_schedule_v1.json"


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


def _load_schedule_hash(schedule_path: str) -> str:
    """스케줄 파일에서 scheduleHash 추출. 파일 없으면 한국어 ValueError."""
    p = Path(schedule_path)
    if not p.exists():
        raise ValueError(
            f"AXS-004c 스케줄 파일을 찾을 수 없습니다: {schedule_path}\n"
            "스케줄 파일 경로를 --schedule 플래그로 지정하거나, "
            "axs_004c_schedule_gen.py 를 실행해 생성하세요."
        )
    try:
        import json as _json
        with open(p, "r", encoding="utf-8") as f:
            body = _json.load(f)
    except Exception as exc:
        raise ValueError(
            f"AXS-004c 스케줄 파일 로드 실패: {schedule_path} — {exc}"
        ) from exc

    schedule_hash = body.get("scheduleHash", "")
    if not schedule_hash:
        raise ValueError(
            f"AXS-004c 스케줄 파일에 scheduleHash 키가 없습니다: {schedule_path}"
        )
    return str(schedule_hash)


def run_axs_004c(
    base_seeds: Sequence[int],
    *,
    H: Optional[int] = None,
    k: int = 4,
    pool_size: int = 64,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    schedule_path: Optional[str] = None,
    replay_mode: str = "first_family",
    replay_sample_size: int = 2,
    prereg_path: Any = _PREREG_PATH,
    git_runner: Optional[Callable[[List[str]], str]] = None,
    dry_run: bool = False,
    reports_dir: Optional[Path] = None,
    register_ledger: bool = False,
) -> Dict[str, Any]:
    """AXS-004c yoked-bonus 실험 실행.

    Args:
        base_seeds: 평가 패밀리 base-seed 목록.
        H: 라운드 수 (None → horizon.yaml 기본값).
        k: 슬레이트 크기.
        pool_size: 후보 풀 크기.
        n_permutations: D-015 null 치환 수.
        schedule_path: yoked 스케줄 JSON 경로 (None → 기본 경로, 없으면 fail-closed).
        replay_mode: 인라인 리플레이 감사 모드.
        replay_sample_size: sampled_families 모드 재계산 패밀리 수.
        prereg_path: 사전등록 JSON 경로.
        git_runner: 테스트용 git 명령 인젝터.
        dry_run: True 이면 계획만 반환, 파일 미작성.
        reports_dir: 리포트 출력 디렉토리 (None → 기본 outputs/reports).
        register_ledger: True 이면 원장에 등록.

    Returns:
        dry_run=True 이면 계획 dict, 그 외 리포트 dict.

    Raises:
        ValueError: schedule 파일이 없거나 유효하지 않을 때 (한국어 메시지).
    """
    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _REPORTS_DIR

    # ---- 스케줄 파일 경로 결정 + fail-closed ----
    eff_schedule_path = schedule_path if schedule_path is not None else str(_DEFAULT_SCHEDULE_PATH)
    schedule_hash = _load_schedule_hash(eff_schedule_path)

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_yaml(_AXS_UCB_CFG_PATH)

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-004c",
        "schedule_path": eff_schedule_path,
        "yokedScheduleHash": schedule_hash,
    }

    if dry_run:
        log_ko(_logger, f"AXS-004c 드라이런: seeds={list(base_seeds)}, H={H_eff}")
        return dry_run_plan("AXS-004c", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-004c 실험 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}, "
        f"scheduleHash={schedule_hash[:12]}",
    )

    # ---- arm configs ----
    default_cfg = dict(base_cfg)
    default_cfg["k"] = k

    yoked_cfg = dict(base_cfg)
    yoked_cfg["k"] = k
    yoked_cfg["schedule_path"] = eff_schedule_path
    yoked_cfg["schedule_hash"] = schedule_hash

    # ---- per-family runs ----
    metric = "slate_excess_nmi"

    default_raw_by_family: Dict[str, Dict[str, Any]] = {}
    yoked_raw_by_family: Dict[str, Dict[str, Any]] = {}
    random_raw_by_family: Dict[str, Dict[str, Any]] = {}

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-004c 패밀리 실행: seed={seed}")

        default_raw_by_family[fam] = run_arm_family(
            lambda cfg=default_cfg: AxsUcbPolicy(dict(cfg)),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        yoked_raw_by_family[fam] = run_arm_family(
            lambda cfg=yoked_cfg: AxsYokedBonusPolicy(dict(cfg)),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        # RANDOM baseline — stored per family for single-source binding
        random_raw_by_family[fam] = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )

    # ---- family_blocks: single authoritative source for all values ----
    families = [str(s) for s in base_seeds]
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        d_raw = default_raw_by_family[fam]
        y_raw = yoked_raw_by_family[fam]
        random_raw = random_raw_by_family[fam]
        family_blocks[fam] = {
            **reportable_block(d_raw),
            "axs_yoked_bonus_slate_excess_nmi": float(y_raw["slate_excess_nmi"]),
            "axs_yoked_bonus_coordinate_coverage_mean": float(y_raw["coordinate_coverage_mean"]),
            "axs_yoked_bonus_archiveHash": y_raw["archiveHash"],
            "axs_yoked_bonus_poolHash": y_raw["poolHash"],
            # RANDOM baseline
            "random_coordinate_coverage_mean": float(random_raw["coordinate_coverage_mean"]),
            "random_archiveHash": random_raw["archiveHash"],
            "random_poolHash": random_raw["poolHash"],
            "random_traceHashes": random_raw["traceHashes"],
        }

    # ---- derive all report values exclusively from family_blocks ----
    default_nmi = {fam: float(family_blocks[fam]["slate_excess_nmi"]) for fam in families}
    yoked_nmi = {fam: float(family_blocks[fam]["axs_yoked_bonus_slate_excess_nmi"]) for fam in families}

    default_cov_mean = float(
        sum(family_blocks[fam]["coordinate_coverage_mean"] for fam in families) / len(families)
    )
    yoked_cov_mean = float(
        sum(family_blocks[fam]["axs_yoked_bonus_coordinate_coverage_mean"] for fam in families)
        / len(families)
    )
    random_coverage_mean = float(
        sum(family_blocks[fam]["random_coordinate_coverage_mean"] for fam in families) / len(families)
    )

    default_entry = build_arm_entry(
        metric, default_nmi, default_cov_mean, random_coverage_mean,
        degenerate_reason_prefix="axs_ucb_default",
    )
    yoked_entry = build_arm_entry(
        metric, yoked_nmi, yoked_cov_mean, random_coverage_mean,
        degenerate_reason_prefix="axs_yoked_bonus",
    )

    eff_sched_path_capture = eff_schedule_path
    schedule_hash_capture = schedule_hash

    def recompute_fn(family: str) -> Dict[str, Any]:
        seed = int(family)
        d = run_arm_family(
            lambda cfg=default_cfg: AxsUcbPolicy(dict(cfg)),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        y_cfg = dict(base_cfg)
        y_cfg["k"] = k
        y_cfg["schedule_path"] = eff_sched_path_capture
        y_cfg["schedule_hash"] = schedule_hash_capture
        y = run_arm_family(
            lambda cfg=y_cfg: AxsYokedBonusPolicy(dict(cfg)),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        r = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        return {
            **reportable_block(d),
            "axs_yoked_bonus_slate_excess_nmi": float(y["slate_excess_nmi"]),
            "axs_yoked_bonus_coordinate_coverage_mean": float(y["coordinate_coverage_mean"]),
            "axs_yoked_bonus_archiveHash": y["archiveHash"],
            "axs_yoked_bonus_poolHash": y["poolHash"],
            "random_coordinate_coverage_mean": float(r["coordinate_coverage_mean"]),
            "random_archiveHash": r["archiveHash"],
            "random_poolHash": r["poolHash"],
            "random_traceHashes": r["traceHashes"],
        }

    body_extra = {
        "arms": {
            "axs_ucb_default": default_entry,
            "axs_yoked_bonus": yoked_entry,
        },
        "baselines": {
            "RANDOM": {"coordinate_coverage_mean": random_coverage_mean},
        },
        "yokedScheduleHash": schedule_hash,
        "yokedScheduleNote": (
            "The yoked schedule content is committed in git (recover via "
            "git show runCommit:configs/prereg/axs_004c_yoked_schedule_v1.json) "
            "and pinned here by yokedScheduleHash."
        ),
    }

    report = build_axs_report(
        "AXS-004c",
        body_extra=body_extra,
        family_blocks=family_blocks,
        recompute_fn=recompute_fn,
        run_params=run_params,
        prereg_path=prereg_path,
        replay_mode=replay_mode,
        replay_sample_size=replay_sample_size,
        git_runner=git_runner,
    )

    log_ko(
        _logger,
        f"AXS-004c 완료: reportHash={report['reportHash'][:12]}, "
        f"default_mean={bootstrap_block(default_nmi, key=metric)['mean']:+.6f}, "
        f"yoked_mean={bootstrap_block(yoked_nmi, key=metric)['mean']:+.6f}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-004c] 완료\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        f"  yokedScheduleHash: {schedule_hash}\n"
        f"  axs_ucb_default  slate_excess_nmi mean: "
        f"{report['arms']['axs_ucb_default']['bootstrap'][metric]['mean']:+.6f}\n"
        f"  axs_yoked_bonus  slate_excess_nmi mean: "
        f"{report['arms']['axs_yoked_bonus']['bootstrap'][metric]['mean']:+.6f}"
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_004c_yoked [options]
    """
    parser = make_axs_arg_parser("AXS-004c yoked-bonus 실험 (axs_ucb_default vs axs_yoked_bonus)")
    parser.add_argument(
        "--schedule",
        type=str,
        default=str(_DEFAULT_SCHEDULE_PATH),
        dest="schedule",
        help="yoked 스케줄 JSON 경로 (기본: configs/prereg/axs_004c_yoked_schedule_v1.json)",
    )
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_004c(
        base_seeds_list,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        n_permutations=args.n_permutations,
        schedule_path=args.schedule,
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
