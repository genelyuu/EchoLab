"""AXS-010: Tie-break invariance 실험 러너 (AXS-P0 T4).

canonical / reverse / hash_seeded / feature_lexicographic 4가지 tie_break_order 에 대해
axs_ucb_default 를 실행, slate_excess_nmi trackDecision invariance 검사.

trackDecision 규칙: prereg signRule/ciRule 에서 minConsistentFamilies, lowerBoundMustExceed 로드.

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

__all__ = ["run_axs_010", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

# Tie-break orders: baseline + 3 variants
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


def _load_prereg(prereg_path: Any) -> Dict[str, Any]:
    with open(str(prereg_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_track_decision(
    per_family_values: Dict[str, float],
    bs_block: Dict[str, float],
    prereg: Dict[str, Any],
) -> str:
    """prereg signRule/ciRule 에서 trackDecision 계산.

    separability_present 조건:
        count(perFamily[fam] > 0) >= minConsistentFamilies
        AND bootstrap.ciLower > lowerBoundMustExceed
    """
    min_fam = prereg["signRule"]["minConsistentFamilies"]
    ci_bound = float(prereg["ciRule"]["lowerBoundMustExceed"])

    count_positive = sum(1 for v in per_family_values.values() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)
    ci_lower = float(bs_block.get("ciLower", float("-inf")))

    if count_positive >= min_fam and ci_lower > ci_bound:
        return "separability_present"
    else:
        return "separability_absent"


def run_axs_010(
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
    """AXS-010 tie-break invariance 실험 실행.

    4가지 tie_break_order 에 대해 axs_ucb_default 를 실행,
    slate_excess_nmi 분리도와 trackDecision 비교.

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
        reports_dir: 리포트 출력 디렉토리.
        register_ledger: True 이면 원장에 등록.

    Returns:
        dry_run=True 이면 계획 dict, 그 외 리포트 dict.
    """
    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _REPORTS_DIR

    bases, archive_cfg = load_default_configs()
    base_cfg = _load_yaml(_AXS_UCB_CFG_PATH)
    prereg = _load_prereg(prereg_path)

    run_params = {
        "H": H_eff,
        "k": k,
        "pool_size": pool_size,
        "n_permutations": n_permutations,
        "experiment": "AXS-010",
    }

    if dry_run:
        log_ko(_logger, f"AXS-010 드라이런: seeds={list(base_seeds)}, H={H_eff}")
        return dry_run_plan("AXS-010", run_params, base_seeds, bases, archive_cfg)

    log_ko(
        _logger,
        f"AXS-010 실험 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )

    # ---- per-order, per-family runs ----
    metric = "slate_excess_nmi"
    # order -> family -> raw block
    order_raw: Dict[str, Dict[str, Dict[str, Any]]] = {order: {} for order in _TIE_BREAK_ORDERS}

    for seed in base_seeds:
        fam = str(seed)
        log_ko(_logger, f"AXS-010 패밀리 실행: seed={seed}")
        for order in _TIE_BREAK_ORDERS:
            order_cfg = dict(base_cfg)
            order_cfg["k"] = k
            order_cfg["tie_break_order"] = order
            raw = run_arm_family(
                lambda cfg=order_cfg: AxsUcbPolicy(dict(cfg)),
                seed,
                H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
                bases=bases, archive_cfg=archive_cfg,
            )
            order_raw[order][fam] = raw
            log_ko(_logger, f"AXS-010 order={order!r} family={fam}: nmi={raw['slate_excess_nmi']:+.6f}")

    # ---- build tie-break blocks ----
    def _build_tie_block(order: str) -> Dict[str, Any]:
        fams = [str(s) for s in base_seeds]
        per_fam = {fam: float(order_raw[order][fam]["slate_excess_nmi"]) for fam in fams}
        bs = bootstrap_block(per_fam, key=metric)
        mean = bs["mean"]
        sign = "+" if mean > 0 else "-"
        track = _compute_track_decision(per_fam, bs, prereg)
        return {
            "sign": sign,
            "estimate": float(mean),
            "ciLower": float(bs["ciLower"]),
            "ciUpper": float(bs["ciUpper"]),
            "trackDecision": track,
        }

    baseline_block = _build_tie_block(_BASELINE_ORDER)
    variant_blocks = {order: _build_tie_block(order) for order in _VARIANT_ORDERS}

    log_ko(
        _logger,
        f"AXS-010 baseline(canonical): sign={baseline_block['sign']}, "
        f"estimate={baseline_block['estimate']:+.6f}, "
        f"trackDecision={baseline_block['trackDecision']}",
    )
    for order, block in variant_blocks.items():
        log_ko(
            _logger,
            f"AXS-010 variant({order!r}): sign={block['sign']}, "
            f"estimate={block['estimate']:+.6f}, "
            f"trackDecision={block['trackDecision']}",
        )

    # ---- family_blocks for replay ----
    # Use canonical order's raw block as primary; include all orders' NMI
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for fam in [str(s) for s in base_seeds]:
        primary = reportable_block(order_raw[_BASELINE_ORDER][fam])
        for order in _TIE_BREAK_ORDERS:
            if order != _BASELINE_ORDER:
                primary[f"{order}_slate_excess_nmi"] = float(
                    order_raw[order][fam]["slate_excess_nmi"]
                )
        family_blocks[fam] = primary

    # perFamilyValues: order -> {fam -> nmi} — exposed for test verification
    per_family_values: Dict[str, Dict[str, float]] = {
        order: {
            fam: float(order_raw[order][fam]["slate_excess_nmi"])
            for fam in [str(s) for s in base_seeds]
        }
        for order in _TIE_BREAK_ORDERS
    }

    def recompute_fn(family: str) -> Dict[str, Any]:
        seed = int(family)
        primary_raw = None
        result = {}
        for order in _TIE_BREAK_ORDERS:
            order_cfg = dict(base_cfg)
            order_cfg["k"] = k
            order_cfg["tie_break_order"] = order
            raw = run_arm_family(
                lambda cfg=order_cfg: AxsUcbPolicy(dict(cfg)),
                seed, H=H_eff, k=k, pool_size=pool_size, n_permutations=n_permutations,
                bases=bases, archive_cfg=archive_cfg,
            )
            if order == _BASELINE_ORDER:
                primary_raw = raw
            else:
                result[f"{order}_slate_excess_nmi"] = float(raw["slate_excess_nmi"])
        assert primary_raw is not None
        return {**reportable_block(primary_raw), **result}

    body_extra = {
        "tieBreak": {
            "baseline": baseline_block,
            "variants": variant_blocks,
        },
        "perFamilyValues": per_family_values,
    }

    report = build_axs_report(
        "AXS-010",
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
        f"AXS-010 완료: reportHash={report['reportHash'][:12]}, "
        f"baseline_sign={baseline_block['sign']}, "
        f"baseline_track={baseline_block['trackDecision']}",
    )

    out_path = write_report(report, reports_dir=Path(reports_dir))
    if register_ledger:
        ledger_path = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"
        register_report(report, out_path, ledger_path=ledger_path)

    print(
        f"[AXS-010] 완료\n"
        f"  reportHash: {report['reportHash']}\n"
        f"  path: {out_path}\n"
        f"  baseline(canonical): sign={baseline_block['sign']}, "
        f"estimate={baseline_block['estimate']:+.6f}, "
        f"trackDecision={baseline_block['trackDecision']}\n"
        + "\n".join(
            f"  variant({order!r}): sign={block['sign']}, "
            f"estimate={block['estimate']:+.6f}, "
            f"trackDecision={block['trackDecision']}"
            for order, block in variant_blocks.items()
        )
    )

    return report


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    python -m echo_bench.experiments.axs_010_tiebreak [options]
    """
    parser = make_axs_arg_parser("AXS-010 tie-break invariance 실험")
    args = parser.parse_args(argv)

    base_seeds_list = parse_base_seeds(args.base_seeds)

    run_axs_010(
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
