"""AXS-003: Alpha-kill 실험 러너 (AXS-P0 T4).

axs_ucb_default (alpha=1.0) vs axs_ucb_alpha0 (alpha=0.0) — slate_excess_nmi.

Guardrails
----------
- Trace-only 정책만 사용; user_id/persona/emotion/preference 벡터 없음.
- 수치는 모두 plain Python float (numpy scalar 금지).
- 런타임 로그는 한국어; 식별자·키·경로는 영어.

All identifiers, config keys, and paths stay English; runtime log messages
are Korean per the project logging convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from echo_bench.env.horizon import default_h, load_horizon
from echo_bench.experiments.axs_common import (
    REPLAY_MODES,
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
from echo_bench.experiments.e_seed_families import DEFAULT_BASE_SEEDS
from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.leakage import DEFAULT_NULL_PERMUTATIONS
from echo_bench.policies.axs_ucb import AxsUcbPolicy
from echo_bench.policies.random import RandomPolicy

__all__ = ["run_axs_003", "main"]

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


def _load_axs_ucb_cfg() -> Dict[str, Any]:
    return _load_yaml(_AXS_UCB_CFG_PATH)


def _resolve_h(H: Optional[int]) -> int:
    if H is not None:
        return H
    try:
        from echo_bench.env.horizon import load_horizon, default_h
        return default_h(load_horizon(_HORIZON_CFG_PATH))
    except Exception:
        return 8


def run_axs_003(
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
    """AXS-003 alpha-kill 실험 실행.

    axs_ucb_default (alpha=1.0) 대비 axs_ucb_alpha0 (alpha=0.0) 로
    slate_excess_nmi 분리도 소멸 여부 검사.

    Args:
        base_seeds: 평가 패밀리 base-seed 목록.
        H: 라운드 수 (None → horizon.yaml 기본값).
        k: 슬레이트 크기.
        pool_size: 후보 풀 크기.
        n_permutations: D-015 null 치환 수.
        replay_mode: 인라인 리플레이 감사 모드.
        replay_sample_size: sampled_families 모드 재계산 패밀리 수.
        prereg_path: 사전등록 JSON 경로.
        git_runner: 테스트용 git 명령 인젝터.
        dry_run: True 이면 계획만 반환, 파일 미작성.
        reports_dir: 리포트 출력 디렉토리 (None → 기본 outputs/reports).
        register_ledger: True 이면 원장에 등록.

    Returns:
        dry_run=True 이면 계획 dict, 그 외 리포트 dict.
    """
    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _REPORTS_DIR

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_axs_ucb_cfg()

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-003",
    }

    if dry_run:
        log_ko(_logger, f"AXS-003 드라이런: seeds={list(base_seeds)}, H={H_eff}, pool_size={pool_size}")
        return dry_run_plan("AXS-003", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-003 실험 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )

    # ---- arm configs ----
    default_cfg = dict(base_cfg)
    default_cfg["k"] = k

    alpha0_cfg = dict(base_cfg)
    alpha0_cfg["k"] = k
    alpha0_cfg["alpha"] = 0.0

    # ---- per-family runners ----
    metric = "slate_excess_nmi"

    default_raw_by_family: Dict[str, Dict[str, Any]] = {}
    alpha0_raw_by_family: Dict[str, Dict[str, Any]] = {}
    random_coverage_by_family: Dict[str, float] = {}

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-003 패밀리 실행: seed={seed}")

        default_raw_by_family[fam] = run_arm_family(
            lambda cfg=default_cfg: AxsUcbPolicy(dict(cfg)),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        alpha0_raw_by_family[fam] = run_arm_family(
            lambda cfg=alpha0_cfg: AxsUcbPolicy(dict(cfg)),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        # RANDOM baseline coverage
        random_raw = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        random_coverage_by_family[fam] = float(random_raw["coordinate_coverage_mean"])

    # RANDOM baseline: mean over families
    random_coverage_mean = float(
        sum(random_coverage_by_family.values()) / len(random_coverage_by_family)
    )

    # ---- build arm entries ----
    default_nmi = {fam: float(raw["slate_excess_nmi"]) for fam, raw in default_raw_by_family.items()}
    alpha0_nmi = {fam: float(raw["slate_excess_nmi"]) for fam, raw in alpha0_raw_by_family.items()}

    default_cov_mean = float(
        sum(raw["coordinate_coverage_mean"] for raw in default_raw_by_family.values())
        / len(default_raw_by_family)
    )
    alpha0_cov_mean = float(
        sum(raw["coordinate_coverage_mean"] for raw in alpha0_raw_by_family.values())
        / len(alpha0_raw_by_family)
    )

    default_entry = build_arm_entry(
        metric, default_nmi, default_cov_mean, random_coverage_mean,
        degenerate_reason_prefix="axs_ucb_default",
    )
    alpha0_entry = build_arm_entry(
        metric, alpha0_nmi, alpha0_cov_mean, random_coverage_mean,
        degenerate_reason_prefix="axs_ucb_alpha0",
    )

    # ---- family_blocks for replay ----
    # Each family block: reportable of the default arm (primary); all arm hashes included
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in [str(s) for s in base_seeds]:
        d_raw = default_raw_by_family[fam]
        a_raw = alpha0_raw_by_family[fam]
        combined = {
            **reportable_block(d_raw),
            "axs_ucb_alpha0_slate_excess_nmi": float(a_raw["slate_excess_nmi"]),
            "axs_ucb_alpha0_archiveHash": a_raw["archiveHash"],
            "axs_ucb_alpha0_poolHash": a_raw["poolHash"],
        }
        family_blocks[fam] = combined

    def recompute_fn(family: str) -> Dict[str, Any]:
        seed = int(family)
        d = run_arm_family(
            lambda cfg=default_cfg: AxsUcbPolicy(dict(cfg)),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        a = run_arm_family(
            lambda cfg=alpha0_cfg: AxsUcbPolicy(dict(cfg)),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        result = {
            **reportable_block(d),
            "axs_ucb_alpha0_slate_excess_nmi": float(a["slate_excess_nmi"]),
            "axs_ucb_alpha0_archiveHash": a["archiveHash"],
            "axs_ucb_alpha0_poolHash": a["poolHash"],
        }
        return result

    body_extra = {
        "arms": {
            "axs_ucb_default": default_entry,
            "axs_ucb_alpha0": alpha0_entry,
        },
        "baselines": {
            "RANDOM": {"coordinate_coverage_mean": random_coverage_mean},
        },
    }

    report = build_axs_report(
        "AXS-003",
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
        f"AXS-003 완료: reportHash={report['reportHash'][:12]}, "
        f"default_mean={bootstrap_block(default_nmi, key=metric)['mean']:+.6f}, "
        f"alpha0_mean={bootstrap_block(alpha0_nmi, key=metric)['mean']:+.6f}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-003] 완료\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        f"  axs_ucb_default slate_excess_nmi mean: "
        f"{report['arms']['axs_ucb_default']['bootstrap'][metric]['mean']:+.6f}\n"
        f"  axs_ucb_alpha0  slate_excess_nmi mean: "
        f"{report['arms']['axs_ucb_alpha0']['bootstrap'][metric]['mean']:+.6f}"
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_003_alpha_kill [options]
    """
    parser = make_axs_arg_parser("AXS-003 alpha-kill 실험 (axs_ucb_default vs axs_ucb_alpha0)")
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_003(
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
