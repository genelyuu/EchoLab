"""AXS-ALPHA-EXP: 알파 탐색 실험 러너 (AXS-V3 N4).

axs_ucb_default (alpha=1.0) vs axs_ucb_alpha0 (alpha=0.0) — 탐색 전용.
v1 AXS-003 의 v3 버전 — 새 실험 ID + exploratory 표시 + v3 사전등록 경로.

[노클레임 라이선스] 이 실험은 탐색 전용이며 어떤 주장도 라이선스하지 않음.
가족 이질성(비평 파일럿에서 124 가족 역전됨)으로 인해 확인적 주장 불가.
원장 등록은 여전히 필요함 — 묵시적 폐기 없음.

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
    build_arm_entry_v3,
    build_axs_report,
    dry_run_plan,
    load_default_configs,
    load_prereg_doc,
    make_axs_arg_parser,
    parse_base_seeds,
    register_report,
    reportable_block,
    run_arm_family,
    slate_sequence_hashes_from_block,
    validate_v3_utility_guard,
    write_report,
)
from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.leakage import DEFAULT_NULL_PERMUTATIONS
from echo_bench.policies.axs_ucb import AxsUcbPolicy
from echo_bench.policies.random import RandomPolicy

__all__ = ["run_axs_alpha_exp", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v3_draft.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

_EXPLORATORY_NOTE = (
    "Family-heterogeneous in critic pilot (reversed in family 124); "
    "licenses no claim at any rung."
)


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


def run_axs_alpha_exp(
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
    """AXS-ALPHA-EXP 알파 탐색 실험 실행.

    axs_ucb_default (alpha=1.0) vs axs_ucb_alpha0 (alpha=0.0).
    탐색 전용 — 어떤 확인적 주장도 라이선스하지 않음.

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
        dry_run: True 이면 계획만 반환.
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

    # v3.1 역할별 가드 파라미터 — 실행 전 fail-closed 검증 (한국어 ValueError)
    prereg_doc = load_prereg_doc(eff_prereg)
    validate_v3_utility_guard(prereg_doc, "AXS-ALPHA-EXP")

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_yaml(_AXS_UCB_CFG_PATH)

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-ALPHA-EXP",
        "base_seeds": [int(s) for s in base_seeds],
    }

    if dry_run:
        log_ko(_logger, f"AXS-ALPHA-EXP 드라이런: seeds={list(base_seeds)}, H={H_eff}")
        return dry_run_plan("AXS-ALPHA-EXP", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-ALPHA-EXP 실험 시작 (탐색): seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )

    # ---- arm configs ----
    default_cfg = dict(base_cfg)
    default_cfg["k"] = k

    alpha0_cfg = dict(base_cfg)
    alpha0_cfg["k"] = k
    alpha0_cfg["alpha"] = 0.0

    # ---- per-family runs ----
    metric = "slate_excess_nmi"
    default_raw_by_family: Dict[str, Dict[str, Any]] = {}
    alpha0_raw_by_family: Dict[str, Dict[str, Any]] = {}
    random_raw_by_family: Dict[str, Dict[str, Any]] = {}

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-ALPHA-EXP 패밀리 실행: seed={seed}")

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
        random_raw_by_family[fam] = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed,
            H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        log_ko(
            _logger,
            f"AXS-ALPHA-EXP family={fam}: "
            f"default_nmi={default_raw_by_family[fam][metric]:+.6f}, "
            f"alpha0_nmi={alpha0_raw_by_family[fam][metric]:+.6f}",
        )

    # ---- family_blocks ----
    families = [str(s) for s in base_seeds]
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        d_raw = default_raw_by_family[fam]
        a_raw = alpha0_raw_by_family[fam]
        random_raw = random_raw_by_family[fam]
        family_blocks[fam] = {
            **reportable_block(d_raw),
            "axs_ucb_alpha0_slate_excess_nmi": float(a_raw[metric]),
            "axs_ucb_alpha0_coordinate_coverage_mean": float(a_raw["coordinate_coverage_mean"]),
            "axs_ucb_alpha0_archiveHash": a_raw["archiveHash"],
            "axs_ucb_alpha0_poolHash": a_raw["poolHash"],
            "axs_ucb_alpha0_traceHashes": a_raw["traceHashes"],
            "axs_ucb_alpha0_slateHashes": a_raw["slateHashes"],
            "random_coordinate_coverage_mean": float(random_raw["coordinate_coverage_mean"]),
            "random_archiveHash": random_raw["archiveHash"],
            "random_poolHash": random_raw["poolHash"],
            "random_traceHashes": random_raw["traceHashes"],
        }

    # ---- derive from family_blocks ----
    default_nmi = {fam: float(family_blocks[fam][metric]) for fam in families}
    alpha0_nmi = {
        fam: float(family_blocks[fam]["axs_ucb_alpha0_slate_excess_nmi"]) for fam in families
    }
    default_cov_mean = float(
        sum(family_blocks[fam]["coordinate_coverage_mean"] for fam in families) / len(families)
    )
    alpha0_cov_mean = float(
        sum(family_blocks[fam]["axs_ucb_alpha0_coordinate_coverage_mean"] for fam in families)
        / len(families)
    )
    random_coverage_mean = float(
        sum(family_blocks[fam]["random_coordinate_coverage_mean"] for fam in families)
        / len(families)
    )

    default_seq_hashes = {
        fam: slate_sequence_hashes_from_block(
            {"slateHashes": family_blocks[fam]["slateHashes"]}
        )
        for fam in families
    }
    alpha0_seq_hashes = {
        fam: slate_sequence_hashes_from_block(
            {"slateHashes": family_blocks[fam]["axs_ucb_alpha0_slateHashes"]}
        )
        for fam in families
    }

    default_entry = build_arm_entry_v3(
        metric, default_nmi, default_cov_mean,
        prereg=prereg_doc,
        experiment_id="AXS-ALPHA-EXP",
        arm_id="axs_ucb_default",
        live_default_coverage_mean=default_cov_mean,
        slate_sequence_hashes=default_seq_hashes,
        degenerate_reason_prefix="axs_ucb_default",
    )
    alpha0_entry = build_arm_entry_v3(
        metric, alpha0_nmi, alpha0_cov_mean,
        prereg=prereg_doc,
        experiment_id="AXS-ALPHA-EXP",
        arm_id="axs_ucb_alpha0",
        live_default_coverage_mean=default_cov_mean,
        slate_sequence_hashes=alpha0_seq_hashes,
        degenerate_reason_prefix="axs_ucb_alpha0",
    )

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
        r = run_arm_family(
            lambda: RandomPolicy({"k": k}),
            seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
            bases=bases, archive_cfg=archive_cfg,
        )
        return {
            **reportable_block(d),
            "axs_ucb_alpha0_slate_excess_nmi": float(a[metric]),
            "axs_ucb_alpha0_coordinate_coverage_mean": float(a["coordinate_coverage_mean"]),
            "axs_ucb_alpha0_archiveHash": a["archiveHash"],
            "axs_ucb_alpha0_poolHash": a["poolHash"],
            "axs_ucb_alpha0_traceHashes": a["traceHashes"],
            "axs_ucb_alpha0_slateHashes": a["slateHashes"],
            "random_coordinate_coverage_mean": float(r["coordinate_coverage_mean"]),
            "random_archiveHash": r["archiveHash"],
            "random_poolHash": r["poolHash"],
            "random_traceHashes": r["traceHashes"],
        }

    body_extra = {
        "arms": {
            "axs_ucb_default": default_entry,
            "axs_ucb_alpha0": alpha0_entry,
        },
        "baselines": {
            "RANDOM": {"coordinate_coverage_mean": random_coverage_mean},
        },
        "exploratory": True,
        "noClaimLicense": True,
        "exploratoryNote": _EXPLORATORY_NOTE,
    }

    report = build_axs_report(
        "AXS-ALPHA-EXP",
        body_extra=body_extra,
        family_blocks=family_blocks,
        recompute_fn=recompute_fn,
        run_params=run_params,
        prereg_path=eff_prereg,
        replay_mode=replay_mode,
        replay_sample_size=replay_sample_size,
        git_runner=git_runner,
    )

    log_ko(
        _logger,
        f"AXS-ALPHA-EXP 완료 (탐색): reportHash={report['reportHash'][:12]}, "
        f"default_mean={report['arms']['axs_ucb_default']['bootstrap'][metric]['mean']:+.6f}, "
        f"alpha0_mean={report['arms']['axs_ucb_alpha0']['bootstrap'][metric]['mean']:+.6f}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-ALPHA-EXP] 완료 (탐색)\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        f"  exploratoryNote: {_EXPLORATORY_NOTE}\n"
        f"  axs_ucb_default slate_excess_nmi mean: "
        f"{report['arms']['axs_ucb_default']['bootstrap'][metric]['mean']:+.6f}\n"
        f"  axs_ucb_alpha0  slate_excess_nmi mean: "
        f"{report['arms']['axs_ucb_alpha0']['bootstrap'][metric]['mean']:+.6f}"
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_alpha_exp [options]
    """
    parser = make_axs_arg_parser(
        "AXS-ALPHA-EXP 알파 탐색 실험 (axs_ucb_default vs axs_ucb_alpha0, 탐색 전용)"
    )
    parser.add_argument(
        "--prereg",
        type=str,
        default=str(_PREREG_PATH),
        dest="prereg",
        help="사전등록 JSON 경로 (기본: v3 draft)",
    )
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_alpha_exp(
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
