"""AXS-TB-001: Tie-break 불변성 실험 러너 (AXS-V3 N4).

delta_imp 결론의 tie-break 불변성 검사:
  - 4개 scheme: canonical(기준), reverse, hash_seeded, feature_lexicographic
  - 각 scheme 에 대해 freeze_at_1 + freeze_none arm 실행
  - per-family delta_imp = sep(freeze_at_1) − sep(freeze_none) 계산
  - bootstrap over deltas → 블록 {sign, estimate, ciLower, ciUpper, trackDecision}
  - 리포트 바디: tieBreak {baseline, variants} — arms/baselines 키 없음

family_blocks: 모든 scheme + arm 의 per-family 값 포함 (재현 커버리지)
재계산 함수: 모든 arm + scheme 재실행 → canonical_hash 동일성 보장.

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

__all__ = ["run_axs_tb_001", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v3_draft.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

_FREEZE_ARMS = ["freeze_at_1", "freeze_none"]
_TIE_BREAK_ORDERS = ["canonical", "reverse", "hash_seeded", "feature_lexicographic"]
_BASELINE_ORDER = "canonical"
_VARIANT_ORDERS = ["reverse", "hash_seeded", "feature_lexicographic"]


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


def _arm_freeze_round_tb(arm_id: str, H: int) -> Optional[int]:
    """AXS-TB-001 용 freeze_round."""
    if arm_id == "freeze_at_1":
        return 1
    elif arm_id == "freeze_none":
        return None
    else:
        raise ValueError(f"AXS-TB-001 알 수 없는 arm_id: {arm_id!r}")


def run_axs_tb_001(
    base_seeds: Sequence[int],
    *,
    H: Optional[int] = None,
    k: int = 4,
    pool_size: int = 64,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    replay_mode: str = "first_family",
    replay_sample_size: int = 2,
    prereg_path: Any = None,
    rereg_path: Any = None,
    git_runner: Optional[Callable[[List[str]], str]] = None,
    dry_run: bool = False,
    reports_dir: Optional[Path] = None,
    register_ledger: bool = False,
) -> Dict[str, Any]:
    """AXS-TB-001 tie-break 불변성 실험 실행.

    각 tie_break_order 에 대해 freeze_at_1 + freeze_none arm 실행,
    per-family delta_imp 계산, bootstrap → tieBreak 블록 구성.

    보고 형식: tieBreak {baseline, variants} — arms/baselines 키 없음.
    family_blocks: 모든 scheme × arm 의 per-family NMI 포함.

    Args:
        base_seeds: 평가 패밀리 base-seed 목록.
        H: 라운드 수.
        k: 슬레이트 크기.
        pool_size: 후보 풀 크기.
        n_permutations: null 치환 수.
        replay_mode: 리플레이 감사 모드.
        replay_sample_size: sampled_families 모드 재계산 패밀리 수.
        prereg_path: 사전등록 JSON 경로 (None → v3 draft).
        rereg_path: prereg_path 의 별칭 (테스트 주입용).
        git_runner: 테스트용 git 명령 인젝터.
        dry_run: True 이면 계획만 반환.
        reports_dir: 리포트 출력 디렉토리.
        register_ledger: True 이면 원장에 등록.

    Returns:
        dry_run=True 이면 계획 dict, 그 외 리포트 dict.
    """
    eff_prereg = rereg_path if rereg_path is not None else prereg_path
    if eff_prereg is None:
        eff_prereg = _PREREG_PATH

    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _REPORTS_DIR

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_yaml(_AXS_UCB_CFG_PATH)

    freeze_rounds = {arm_id: _arm_freeze_round_tb(arm_id, H_eff) for arm_id in _FREEZE_ARMS}

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-TB-001",
        "base_seeds": [int(s) for s in base_seeds],
        "tieBreakOrders": _TIE_BREAK_ORDERS,
    }

    if dry_run:
        log_ko(_logger, f"AXS-TB-001 드라이런: seeds={list(base_seeds)}, H={H_eff}")
        return dry_run_plan("AXS-TB-001", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-TB-001 실험 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )

    # ---- per-scheme, per-arm, per-family runs ----
    # raw_data[scheme][arm_id][fam] = raw block
    raw_data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        scheme: {arm_id: {} for arm_id in _FREEZE_ARMS}
        for scheme in _TIE_BREAK_ORDERS
    }

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-TB-001 패밀리 실행: seed={seed}")
        for scheme in _TIE_BREAK_ORDERS:
            for arm_id in _FREEZE_ARMS:
                fr = freeze_rounds[arm_id]
                arm_cfg = dict(base_cfg)
                arm_cfg["k"] = k
                arm_cfg["freeze_round"] = fr
                arm_cfg["tie_break_order"] = scheme
                raw = run_arm_family(
                    lambda cfg=arm_cfg: AxsUcbPolicy(dict(cfg)),
                    seed,
                    H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
                    bases=bases, archive_cfg=archive_cfg,
                )
                raw_data[scheme][arm_id][fam] = raw
                log_ko(
                    _logger,
                    f"AXS-TB-001 scheme={scheme!r} arm={arm_id} family={fam}: "
                    f"nmi={raw['slate_excess_nmi']:+.6f}",
                )

    # ---- family_blocks: ALL schemes × arms per-family values ----
    # Primary block: canonical freeze_at_1 (deterministic anchor)
    families = [str(s) for s in base_seeds]
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        primary_raw = raw_data[_BASELINE_ORDER]["freeze_at_1"][fam]
        block = dict(reportable_block(primary_raw))
        for scheme in _TIE_BREAK_ORDERS:
            for arm_id in _FREEZE_ARMS:
                key = f"{scheme}_{arm_id}_slate_excess_nmi"
                block[key] = float(raw_data[scheme][arm_id][fam]["slate_excess_nmi"])
                block[f"{scheme}_{arm_id}_traceHashes"] = raw_data[scheme][arm_id][fam]["traceHashes"]
        family_blocks[fam] = block

    # ---- build tieBreak blocks ----
    # per-family delta_imp[scheme][fam] = nmi(freeze_at_1) - nmi(freeze_none)
    metric = "slate_excess_nmi"
    delta_key = "delta_imp"

    def _build_tie_block(scheme: str) -> Dict[str, Any]:
        delta_by_fam: Dict[str, float] = {}
        for fam in families:
            f1_nmi = float(family_blocks[fam][f"{scheme}_freeze_at_1_{metric}"])
            fn_nmi = float(family_blocks[fam][f"{scheme}_freeze_none_{metric}"])
            delta_by_fam[fam] = f1_nmi - fn_nmi

        bs = bootstrap_block(delta_by_fam, key=delta_key)
        mean = bs["mean"]
        sign = "+" if mean > 0 else "-"

        # trackDecision: delta_imp_pass if >=4/5 positive AND ciLower > 0
        n_pos = sum(1 for v in delta_by_fam.values() if v > 0)
        n_total = len(delta_by_fam)
        min_consistent = max(4, n_total - 1) if n_total >= 5 else n_total
        ci_lower = float(bs["ciLower"])
        # Use 4/5 rule: >=4 of 5 positive AND ciLower > 0
        if n_total >= 5:
            pass_threshold = 4
        else:
            pass_threshold = n_total  # smoke scale: require all
        track = (
            "delta_imp_pass"
            if (n_pos >= pass_threshold and ci_lower > 0.0)
            else "delta_imp_fail"
        )

        return {
            "sign": sign,
            "estimate": float(mean),
            "ciLower": ci_lower,
            "ciUpper": float(bs["ciUpper"]),
            "trackDecision": track,
        }

    baseline_block_data = _build_tie_block(_BASELINE_ORDER)
    variant_blocks = {scheme: _build_tie_block(scheme) for scheme in _VARIANT_ORDERS}

    log_ko(
        _logger,
        f"AXS-TB-001 baseline(canonical): sign={baseline_block_data['sign']}, "
        f"estimate={baseline_block_data['estimate']:+.6f}, "
        f"trackDecision={baseline_block_data['trackDecision']}",
    )
    for scheme, block in variant_blocks.items():
        log_ko(
            _logger,
            f"AXS-TB-001 variant({scheme!r}): sign={block['sign']}, "
            f"estimate={block['estimate']:+.6f}, "
            f"trackDecision={block['trackDecision']}",
        )

    def recompute_fn(family: str) -> Dict[str, Any]:
        seed = int(family)
        primary_raw = None
        result: Dict[str, Any] = {}
        for scheme in _TIE_BREAK_ORDERS:
            for arm_id in _FREEZE_ARMS:
                fr = freeze_rounds[arm_id]
                arm_cfg = dict(base_cfg)
                arm_cfg["k"] = k
                arm_cfg["freeze_round"] = fr
                arm_cfg["tie_break_order"] = scheme
                raw = run_arm_family(
                    lambda cfg=arm_cfg: AxsUcbPolicy(dict(cfg)),
                    seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
                    bases=bases, archive_cfg=archive_cfg,
                )
                if scheme == _BASELINE_ORDER and arm_id == "freeze_at_1":
                    primary_raw = raw
                result[f"{scheme}_{arm_id}_{metric}"] = float(raw[metric])
                result[f"{scheme}_{arm_id}_traceHashes"] = raw["traceHashes"]
        assert primary_raw is not None
        return {**reportable_block(primary_raw), **result}

    # tieBreak only — no arms/baselines keys (gate TB-001 branch checks tieBreak only)
    body_extra = {
        "tieBreak": {
            "baseline": baseline_block_data,
            "variants": variant_blocks,
        },
    }

    report = build_axs_report(
        "AXS-TB-001",
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
        f"AXS-TB-001 완료: reportHash={report['reportHash'][:12]}, "
        f"baseline_sign={baseline_block_data['sign']}, "
        f"baseline_track={baseline_block_data['trackDecision']}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-TB-001] 완료\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        f"  baseline(canonical): sign={baseline_block_data['sign']}, "
        f"estimate={baseline_block_data['estimate']:+.6f}, "
        f"trackDecision={baseline_block_data['trackDecision']}\n"
        + "\n".join(
            f"  variant({scheme!r}): sign={block['sign']}, "
            f"estimate={block['estimate']:+.6f}, "
            f"trackDecision={block['trackDecision']}"
            for scheme, block in variant_blocks.items()
        )
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_tb_001 [options]
    """
    parser = make_axs_arg_parser("AXS-TB-001 tie-break 불변성 실험")
    parser.add_argument(
        "--prereg",
        type=str,
        default=str(_PREREG_PATH),
        dest="prereg",
        help="사전등록 JSON 경로 (기본: v3 draft)",
    )
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_tb_001(
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
