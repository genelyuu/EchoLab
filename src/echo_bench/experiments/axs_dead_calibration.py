"""AXS-DEAD-CAL: dead-arm 캘리브레이션 러너 (Task N7-1).

Track M v3 prereg 의 role-specific utility guard 가 사용할 절대 coverage floor 를
마법 숫자 없이 캘리브레이션한다. 의도적으로 죽은 정책(DeadConstantPolicy:
trace-blind / probe-blind / 상수 슬레이트)을 4개 critic 캘리브레이션 패밀리
(123, 124, 777, 55555)에서 실행해 dead coverage 를 측정하고, pilot_v3 AXS-009
freeze_at_1 arm 의 live key-arm coverage 와의 중간값으로 floor 를 고정한다:

    absoluteFloor = (max_family(coverage_dead) + min_family(live coverage)) / 2

하드 가드: 평가 패밀리(42, 7, 101, 2025, 31337)는 절대 실행하지 않는다.
원장(run_ledger) 등록도 금지된다.

Guardrails
----------
- DeadConstantPolicy 는 trace-only 그 이하: trace 내용 자체를 읽지 않는다.
  user_id/persona/emotion/preference 벡터 없음.
- 수치는 모두 plain Python float/int (numpy scalar 금지).
- 런타임 로그/에러는 한국어; 식별자·JSON 키·경로는 영어.

All identifiers, config keys, JSON keys, and paths stay English; runtime log
messages are Korean per the project logging convention.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from echo_bench.env.horizon import default_h, load_horizon
from echo_bench.experiments.axs_common import (
    build_arm_entry,
    build_axs_report,
    dry_run_plan,
    load_default_configs,
    make_axs_arg_parser,
    parse_base_seeds,
    reportable_block,
    run_arm_family,
    write_report,
)
from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.leakage import DEFAULT_NULL_PERMUTATIONS
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "ARM_ID",
    "EXPERIMENT_ID",
    "DeadConstantPolicy",
    "run_axs_dead_calibration",
    "derive_floor",
    "extract_dead_calibration",
    "extract_live_key_arm_coverage",
    "write_calibration_summary",
    "main",
]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREREG_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_DEAD_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports" / "dead_calibration"
_PILOT_V3_DIR = _REPO_ROOT / "outputs" / "reports" / "pilot_v3"
_SUMMARY_OUT_PATH = _REPO_ROOT / "configs" / "prereg" / "axs_dead_calibration_v1.json"

EXPERIMENT_ID = "AXS-DEAD-CAL"
ARM_ID = "dead_constant"

# critic 캘리브레이션 패밀리 — 이 캘리브레이션의 기본 대상.
CALIBRATION_FAMILIES: Tuple[int, ...] = (123, 124, 777, 55555)

# 평가 패밀리 — 이 러너에서 실행하면 하드 위반. fail closed.
EVAL_FAMILIES_FORBIDDEN = frozenset({42, 7, 101, 2025, 31337})

# policy_version 해시에 들어가는 고정 salt (기존 정책들과의 구별성 보장).
_DEAD_POLICY_SALT = "axs-dead-constant-v1"

_FLOOR_FORMULA = "midpoint(max(coverage_dead), min(live key-arm coverage))"


# ---------------------------------------------------------------------------
# DeadConstantPolicy
# ---------------------------------------------------------------------------


class DeadConstantPolicy(Policy):
    """의도적으로 죽은 상수 슬레이트 정책 (캘리브레이션 전용).

    trace-blind / probe-blind / seed-blind: 슬레이트는 (pool, k) 만의 순수
    함수다. 풀을 cardId 기준 canonical 정렬한 뒤, 첫 카드를 선택하고 이어서
    아직 등장하지 않은 basis 의 카드를 탐욕적으로 추가해 distinct basis 4종
    (또는 풀 소진)까지 채우고, 남은 슬롯은 canonical 순서로 채운다. 이로써
    k=4 슬레이트는 풀에 3종 이상의 basis 가 존재하는 한 check_slate 의
    ≥3 distinct bases 제약을 만족한다.

    Config keys
    -----------
    k : int
        슬레이트 크기. per-call config 에서는 ``k`` 만 반영되며 그 외 키는
        모두 무시된다.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    def policy_version(self) -> str:
        """클래스명 + config + 고정 salt 의 canonical_hash.

        salt 가 포함되므로 동일 config 의 다른 정책 클래스와 절대 충돌하지
        않는다 (기존 policy_version 컨벤션 + salt).
        """
        return canonical_hash(
            {
                "policy": self.__class__.__name__,
                "config": self.config,
                "salt": _DEAD_POLICY_SALT,
            }
        )

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """(pool, k) 만의 순수 함수로 상수 슬레이트 반환.

        trace 와 seed 는 의도적으로 완전히 무시한다 (dead arm 의 정의).
        per-call config 는 ``k`` 만 반영한다.
        """
        # per-call config 에서 k 만 취한다 — 그 외 키는 무시 (probe/trace-blind).
        k_value = (config or {}).get("k", self.config.get("k"))
        if k_value is None:
            raise ValueError(
                "DEAD_CONSTANT 정책: config 에 'k' 가 없습니다. "
                "슬레이트 크기를 지정해야 합니다."
            )
        k = int(k_value)

        if k > len(pool):
            raise ValueError(
                f"DEAD_CONSTANT 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        # 1. canonical 정렬 (cardId 기준).
        sorted_pool = sorted(pool, key=lambda c: str(c["cardId"]))

        # 2. 탐욕적 distinct-basis 채우기: 첫 카드 → 미등장 basis 카드 →
        #    distinct basis 4종 또는 풀 소진까지.
        chosen: List[Dict[str, Any]] = []
        chosen_ids: set = set()
        seen_bases: set = set()
        for card in sorted_pool:
            if len(chosen) >= k or len(seen_bases) >= 4:
                break
            if not chosen or card["basis"] not in seen_bases:
                chosen.append(card)
                chosen_ids.add(card["cardId"])
                seen_bases.add(card["basis"])

        # 3. 남은 슬롯은 canonical 순서로 채운다.
        for card in sorted_pool:
            if len(chosen) >= k:
                break
            if card["cardId"] not in chosen_ids:
                chosen.append(card)
                chosen_ids.add(card["cardId"])

        if len(chosen) != k:
            raise ValueError(
                f"DEAD_CONSTANT 정책: k={k} 슬레이트를 구성하지 못했습니다 "
                f"(선택={len(chosen)})"
            )

        # 점수 성분: 상수 정책이므로 모두 0 (정보 없음을 명시).
        self.log_score_components(
            {c["cardId"]: {"mean": 0.0, "bonus": 0.0} for c in chosen}
        )
        log_ko(
            _logger,
            f"DEAD_CONSTANT 슬레이트 확정: k={k}, "
            f"basis {len({c['basis'] for c in chosen})} 종 "
            "(trace/seed 무시 — 상수 슬레이트)",
        )
        return [c["cardId"] for c in chosen]


# ---------------------------------------------------------------------------
# 러너
# ---------------------------------------------------------------------------


def _resolve_h(H: Optional[int]) -> int:
    if H is not None:
        return int(H)
    try:
        return default_h(load_horizon(_HORIZON_CFG_PATH))
    except Exception:
        return 8


def _reject_eval_families(base_seeds: Sequence[int]) -> None:
    """평가 패밀리가 포함되어 있으면 즉시 실패 (하드 가드)."""
    forbidden = sorted(int(s) for s in base_seeds if int(s) in EVAL_FAMILIES_FORBIDDEN)
    if forbidden:
        raise ValueError(
            f"AXS-DEAD-CAL: 평가 패밀리 {forbidden} 는 절대 실행할 수 없습니다 "
            "(하드 위반). 캘리브레이션 패밀리(123, 124, 777, 55555)만 허용됩니다."
        )


def run_axs_dead_calibration(
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
) -> Dict[str, Any]:
    """dead_constant 단일 arm 을 캘리브레이션 패밀리들에서 실행.

    패밀리별로 독립 리포트를 reports_dir 에 작성한다 (seedBatchId 가
    base_seeds=[seed] 를 포함하므로 파일명 충돌 없음).

    Returns:
        dry_run=True 이면 계획 dict ({"dryRun": True, ...}),
        그 외 {"dryRun": False, "reports": {family: report},
              "reportPaths": {family: str}}.
    """
    _reject_eval_families(base_seeds)

    H_eff = _resolve_h(H)
    if reports_dir is None:
        reports_dir = _DEAD_REPORTS_DIR

    bases, archive_cfg = load_default_configs()

    if dry_run:
        run_params_plan = {
            "H": H_eff,
            "k": k,
            "pool_size": pool_size,
            "n_permutations": n_permutations,
            "experiment": EXPERIMENT_ID,
            "base_seeds": [int(s) for s in base_seeds],
        }
        log_ko(
            _logger,
            f"AXS-DEAD-CAL 드라이런: seeds={list(base_seeds)}, H={H_eff} (파일 미작성)",
        )
        return dry_run_plan(
            EXPERIMENT_ID, run_params_plan, base_seeds, bases, archive_cfg
        )

    log_ko(
        _logger,
        f"AXS-DEAD-CAL 실행 시작: seeds={list(base_seeds)}, H={H_eff}, k={k}, "
        f"pool_size={pool_size}, n_permutations={n_permutations}",
    )

    dead_policy_version = DeadConstantPolicy({"k": k}).policy_version()

    reports: Dict[str, Any] = {}
    report_paths: Dict[str, str] = {}

    for seed in base_seeds:
        fam = str(int(seed))
        # CRITICAL: base_seeds 누락 시 패밀리 간 파일명 충돌
        # (seedBatchId-filename-collision 버그 재발 방지).
        run_params = {
            "H": H_eff,
            "k": k,
            "pool_size": pool_size,
            "n_permutations": n_permutations,
            "experiment": EXPERIMENT_ID,
            "base_seeds": [int(seed)],
        }

        log_ko(_logger, f"AXS-DEAD-CAL 패밀리 실행: seed={seed}")
        raw = run_arm_family(
            lambda: DeadConstantPolicy({"k": k}),
            int(seed),
            H=H_eff,
            k=k,
            pool_size=pool_size,
            n_permutations=n_permutations,
            bases=bases,
            archive_cfg=archive_cfg,
        )

        family_blocks = {fam: reportable_block(raw)}

        def recompute_fn(family: str, _seed: int = int(seed)) -> Dict[str, Any]:
            if family != str(_seed):
                raise ValueError(
                    f"AXS-DEAD-CAL recompute: 알 수 없는 패밀리 {family!r} "
                    f"(예상: {_seed})"
                )
            return run_arm_family(
                lambda: DeadConstantPolicy({"k": k}),
                _seed,
                H=H_eff,
                k=k,
                pool_size=pool_size,
                n_permutations=n_permutations,
                bases=bases,
                archive_cfg=archive_cfg,
            )

        # RANDOM baseline 부재: 캘리브레이션 단독 실행이므로 degenerate 비교
        # 기준값으로 0.0 을 전달한다 (coverage >= 0 이므로 degenerate 트리플은
        # 부착되지 않음 — dead arm 은 의도적으로 죽은 arm 이며 메커니즘 클레임에
        # 포함되지 않는다).
        entry = build_arm_entry(
            "slate_excess_nmi",
            {fam: float(raw["slate_excess_nmi"])},
            float(raw["coordinate_coverage_mean"]),
            0.0,
            degenerate_reason_prefix=ARM_ID,
        )
        entry["includedInMechanismClaim"] = False

        body_extra = {
            "arms": {ARM_ID: entry},
            "runParams": dict(run_params),
            "deadPolicyVersion": dead_policy_version,
            "calibrationNote": (
                "deliberately dead arm (trace-blind, probe-blind, constant "
                "admissible slate) for absolute utility-floor calibration; "
                "not a mechanism-claim arm"
            ),
        }

        report = build_axs_report(
            EXPERIMENT_ID,
            body_extra=body_extra,
            family_blocks=family_blocks,
            recompute_fn=recompute_fn,
            run_params=run_params,
            prereg_path=prereg_path,
            replay_mode=replay_mode,
            replay_sample_size=replay_sample_size,
            git_runner=git_runner,
        )

        out_path = write_report(report, reports_dir=Path(reports_dir))
        reports[fam] = report
        report_paths[fam] = str(out_path)

        log_ko(
            _logger,
            f"AXS-DEAD-CAL 패밀리 완료: family={fam}, "
            f"coverage_dead={float(raw['coordinate_coverage_mean']):.6f}, "
            f"slate_excess_nmi={float(raw['slate_excess_nmi']):+.6f}, "
            f"reportHash={report['reportHash'][:12]}",
        )

    print(
        "[AXS-DEAD-CAL] 완료\n"
        + "\n".join(
            f"  family={fam}: coverage_dead="
            f"{reports[fam]['arms'][ARM_ID]['utility']['coordinate_coverage_mean']:.6f}, "
            f"slate_excess_nmi="
            f"{reports[fam]['arms'][ARM_ID]['perFamily'][fam]['slate_excess_nmi']:+.6f}, "
            f"path={report_paths[fam]}"
            for fam in reports
        )
    )

    return {"dryRun": False, "reports": reports, "reportPaths": report_paths}


# ---------------------------------------------------------------------------
# floor 계산 + 추출기
# ---------------------------------------------------------------------------


def derive_floor(
    dead_coverage_by_family: Mapping[str, float],
    live_coverage_by_family: Mapping[str, float],
) -> Dict[str, Any]:
    """floor 공식 적용: absoluteFloor = (max(dead) + min(live)) / 2.

    fail-closed 조건:
    - 두 입력의 패밀리 집합이 다르면 ValueError.
    - 입력이 비어 있으면 ValueError.
    - max(dead) >= min(live) 이면 캘리브레이션 무효 — ValueError (숫자 포함).
    """
    if not dead_coverage_by_family or not live_coverage_by_family:
        raise ValueError("floor 계산 실패: dead/live coverage 입력이 비어 있습니다.")

    dead_fams = set(dead_coverage_by_family)
    live_fams = set(live_coverage_by_family)
    if dead_fams != live_fams:
        raise ValueError(
            f"floor 계산 실패: dead 패밀리 {sorted(dead_fams)} 와 "
            f"live 패밀리 {sorted(live_fams)} 가 일치하지 않습니다."
        )

    max_dead = max(float(v) for v in dead_coverage_by_family.values())
    min_live = min(float(v) for v in live_coverage_by_family.values())

    if max_dead >= min_live:
        raise ValueError(
            "floor 계산 실패: max(coverage_dead) "
            f"{max_dead:.6f} >= min(live key-arm coverage) {min_live:.6f} — "
            "dead arm 이 live key-arm 보다 낮지 않아 캘리브레이션이 무효입니다."
        )

    return {
        "formula": _FLOOR_FORMULA,
        "maxCoverageDead": float(max_dead),
        "minLiveKeyArmCoverage": float(min_live),
        "absoluteFloor": float((max_dead + min_live) / 2.0),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"리포트 파일 로드 실패: {path} — {exc}")
    if not isinstance(doc, dict):
        raise ValueError(f"리포트 파일 형식 오류 (dict 아님): {path}")
    return doc


def _single_family_of(arm: Mapping[str, Any], path: Path) -> str:
    fams = list(arm.get("perFamily", {}).keys())
    if len(fams) != 1:
        raise ValueError(
            f"리포트 {path.name} 의 perFamily 가 단일 패밀리가 아닙니다: {fams}"
        )
    return str(fams[0])


def extract_dead_calibration(dead_reports_dir: Path) -> Dict[str, Any]:
    """dead_calibration 리포트들에서 패밀리별 dead 측정값을 추출.

    Returns:
        {"perFamily": {family: {"coordinate_coverage_mean", "slate_excess_nmi_mean"}},
         "reportHashes": {family: reportHash},
         "runParams": 병합된 runParams (base_seeds 는 정렬 합집합),
         "policyVersion": deadPolicyVersion}
    """
    dead_reports_dir = Path(dead_reports_dir)
    files = sorted(dead_reports_dir.glob("axs_dead-cal_*.json"))
    if not files:
        raise ValueError(
            f"dead 리포트가 없습니다: {dead_reports_dir} — "
            "먼저 러너를 실행하십시오 (python -m "
            "echo_bench.experiments.axs_dead_calibration)."
        )

    per_family: Dict[str, Dict[str, float]] = {}
    report_hashes: Dict[str, str] = {}
    shared_params: Optional[Dict[str, Any]] = None
    all_seeds: List[int] = []
    policy_version: Optional[str] = None

    for fp in files:
        report = _load_json(fp)
        if report.get("experimentId") != EXPERIMENT_ID:
            raise ValueError(
                f"리포트 {fp.name} 의 experimentId 가 {EXPERIMENT_ID} 가 아닙니다: "
                f"{report.get('experimentId')!r}"
            )
        arms = report.get("arms", {})
        if ARM_ID not in arms:
            raise ValueError(f"리포트 {fp.name} 에 '{ARM_ID}' arm 이 없습니다.")
        arm = arms[ARM_ID]
        fam = _single_family_of(arm, fp)
        if fam in per_family:
            raise ValueError(
                f"패밀리 {fam} 의 dead 리포트가 중복되었습니다: {fp.name}"
            )

        try:
            coverage = float(arm["utility"]["coordinate_coverage_mean"])
            nmi = float(arm["perFamily"][fam]["slate_excess_nmi"])
            report_hash = str(report["reportHash"])
            run_params = dict(report["runParams"])
            fam_policy_version = str(report["deadPolicyVersion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"리포트 {fp.name} 필드 추출 실패: {exc!r}")

        per_family[fam] = {
            "coordinate_coverage_mean": coverage,
            "slate_excess_nmi_mean": nmi,
        }
        report_hashes[fam] = report_hash

        # fail closed: base_seeds 누락/빈 리스트는 변조 또는 구버전 리포트.
        raw_seeds = run_params.pop("base_seeds", None)
        if not raw_seeds:
            raise ValueError(
                f"리포트 {fp.name} 의 runParams 에 base_seeds 가 없거나 비어 "
                "있습니다 — 캘리브레이션에 사용할 수 없습니다 (fail closed)."
            )
        seeds = [int(s) for s in raw_seeds]
        # fail closed: 단일 패밀리 리포트의 base_seeds 는 정확히 [int(family)].
        if seeds != [int(fam)]:
            raise ValueError(
                f"리포트 {fp.name} 의 base_seeds {seeds} 가 패밀리 [{int(fam)}] 와 "
                "일치하지 않습니다 — 변조되었거나 다른 실행의 리포트입니다 "
                "(fail closed)."
            )
        run_params.pop("experiment", None)
        all_seeds.extend(seeds)

        if shared_params is None:
            shared_params = run_params
        elif shared_params != run_params:
            raise ValueError(
                f"dead 리포트 간 runParams 불일치: {fp.name} — "
                f"{run_params!r} != {shared_params!r}"
            )

        if policy_version is None:
            policy_version = fam_policy_version
        elif policy_version != fam_policy_version:
            raise ValueError(
                f"dead 리포트 간 deadPolicyVersion 불일치: {fp.name}"
            )

    assert shared_params is not None and policy_version is not None
    merged_params = dict(shared_params)
    merged_params["base_seeds"] = sorted(set(all_seeds))

    return {
        "perFamily": per_family,
        "reportHashes": report_hashes,
        "runParams": merged_params,
        "policyVersion": policy_version,
    }


def extract_live_key_arm_coverage(
    pilot_v3_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """pilot_v3 AXS-009 리포트들에서 freeze_at_1 coverage 를 추출.

    Returns:
        (coverage_by_family, report_hash_by_family)
    """
    pilot_v3_dir = Path(pilot_v3_dir)
    files = sorted(pilot_v3_dir.glob("axs_009_*.json"))
    if not files:
        raise ValueError(
            f"pilot_v3 AXS-009 리포트가 없습니다: {pilot_v3_dir}"
        )

    coverage: Dict[str, float] = {}
    report_hashes: Dict[str, str] = {}
    for fp in files:
        report = _load_json(fp)
        # fail closed: 파일명이 axs_009_* 라도 experimentId 가 다르면 외부 리포트.
        if report.get("experimentId") != "AXS-009":
            raise ValueError(
                f"리포트 {fp.name} 의 experimentId 가 AXS-009 가 아닙니다: "
                f"{report.get('experimentId')!r}"
            )
        arms = report.get("arms", {})
        if "freeze_at_1" not in arms:
            raise ValueError(f"리포트 {fp.name} 에 'freeze_at_1' arm 이 없습니다.")
        arm = arms["freeze_at_1"]
        fam = _single_family_of(arm, fp)
        if fam in coverage:
            raise ValueError(
                f"패밀리 {fam} 의 AXS-009 리포트가 중복되었습니다: {fp.name}"
            )
        try:
            coverage[fam] = float(arm["utility"]["coordinate_coverage_mean"])
            report_hashes[fam] = str(report["reportHash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"리포트 {fp.name} 필드 추출 실패: {exc!r}")

    return coverage, report_hashes


# ---------------------------------------------------------------------------
# 요약 작성기
# ---------------------------------------------------------------------------


def _full_commit_sha(git_runner: Optional[Callable[[List[str]], str]] = None) -> str:
    """현재 HEAD 의 전체 커밋 SHA. 확인 불가 시 fail closed."""
    if git_runner is not None:
        out = str(git_runner(["rev-parse", "HEAD"])).strip()
    else:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            out = proc.stdout.strip()
        except (subprocess.SubprocessError, OSError) as exc:
            raise ValueError(f"HEAD 커밋 SHA 확인 실패: {exc!r}")
    if not out:
        raise ValueError(
            "HEAD 커밋 SHA 가 비어 있습니다 — 캘리브레이션 요약에는 "
            "전체 runCommit 이 필수입니다 (fail closed)."
        )
    return out


def write_calibration_summary(
    *,
    dead_reports_dir: Path = _DEAD_REPORTS_DIR,
    pilot_v3_dir: Path = _PILOT_V3_DIR,
    out_path: Path = _SUMMARY_OUT_PATH,
    git_runner: Optional[Callable[[List[str]], str]] = None,
) -> Dict[str, Any]:
    """dead 리포트 + pilot_v3 freeze_at_1 리포트에서 floor 캘리브레이션 요약 작성.

    summaryHash 는 summaryHash 필드를 제외한 본문의 canonical_hash
    (configs/prereg 요약 컨벤션). 원자적 쓰기.
    """
    dead = extract_dead_calibration(Path(dead_reports_dir))
    live_coverage, live_hashes = extract_live_key_arm_coverage(Path(pilot_v3_dir))

    expected_fams = {str(s) for s in CALIBRATION_FAMILIES}
    dead_fams = set(dead["perFamily"].keys())
    live_fams = set(live_coverage.keys())
    if dead_fams != expected_fams:
        raise ValueError(
            f"dead 리포트 패밀리 {sorted(dead_fams)} 가 캘리브레이션 패밀리 "
            f"{sorted(expected_fams)} 와 일치하지 않습니다."
        )
    if live_fams != expected_fams:
        raise ValueError(
            f"pilot_v3 freeze_at_1 패밀리 {sorted(live_fams)} 가 캘리브레이션 "
            f"패밀리 {sorted(expected_fams)} 와 일치하지 않습니다."
        )

    dead_coverage = {
        fam: dead["perFamily"][fam]["coordinate_coverage_mean"]
        for fam in dead["perFamily"]
    }
    floor_derivation = derive_floor(dead_coverage, live_coverage)

    # runParams: 정확히 {H, k, pool_size, n_permutations, base_seeds}.
    rp = dead["runParams"]
    try:
        run_params = {
            "H": int(rp["H"]),
            "k": int(rp["k"]),
            "pool_size": int(rp["pool_size"]),
            "n_permutations": int(rp["n_permutations"]),
            "base_seeds": [int(s) for s in rp["base_seeds"]],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"dead 리포트 runParams 필드 누락/형식 오류: {exc!r}")

    # 패밀리 키 정렬은 캘리브레이션 패밀리 선언 순서(숫자 오름차순 문자열)와 일치.
    fam_order = [str(s) for s in sorted(CALIBRATION_FAMILIES)]

    doc: Dict[str, Any] = {
        "calibrationId": "axs-dead-calibration-v1",
        "purpose": (
            "Absolute coordinate-coverage floor calibration for the v3 "
            "role-specific utility guard on key manipulation arms, fixed "
            "between measured dead-arm coverage and measured minimum live "
            "key-arm coverage."
        ),
        "calibrationFamilies": fam_order,
        "runParams": run_params,
        "deadArm": {
            "policyDescription": (
                "DeadConstantPolicy: deterministic, trace-blind, probe-blind "
                "constant admissible slate — pool sorted canonically by cardId, "
                "greedy distinct-basis fill to 4 bases then canonical-order "
                "fill; identical slate every round regardless of trace content."
            ),
            "policyVersion": dead["policyVersion"],
            "perFamily": {fam: dead["perFamily"][fam] for fam in fam_order},
            "deadReportHashes": {fam: dead["reportHashes"][fam] for fam in fam_order},
        },
        "liveKeyArmSource": {
            "experiment": "AXS-009 (pilot_v3)",
            "arm": "freeze_at_1",
            "reportHashes": {fam: live_hashes[fam] for fam in fam_order},
            "perFamilyCoverage": {fam: live_coverage[fam] for fam in fam_order},
        },
        "floorDerivation": floor_derivation,
        "runCommit": _full_commit_sha(git_runner),
    }

    # summaryHash: summaryHash 필드 제외 본문의 canonical_hash.
    doc["summaryHash"] = canonical_hash(doc)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    log_ko(
        _logger,
        "캘리브레이션 요약 저장 완료: "
        f"path={out_path}, absoluteFloor="
        f"{floor_derivation['absoluteFloor']:.6f}, "
        f"summaryHash={doc['summaryHash'][:12]}",
    )
    print(
        "[AXS-DEAD-CAL 요약] 작성 완료\n"
        f"  maxCoverageDead: {floor_derivation['maxCoverageDead']:.6f}\n"
        f"  minLiveKeyArmCoverage: {floor_derivation['minLiveKeyArmCoverage']:.6f}\n"
        f"  absoluteFloor: {floor_derivation['absoluteFloor']:.6f}\n"
        f"  path: {out_path}"
    )
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    실행: python -m echo_bench.experiments.axs_dead_calibration [options]
    요약: python -m echo_bench.experiments.axs_dead_calibration --write-summary
    """
    parser = make_axs_arg_parser(
        "AXS-DEAD-CAL dead-arm 캘리브레이션 (v3 절대 utility floor)"
    )
    parser.set_defaults(
        base_seeds=",".join(str(s) for s in CALIBRATION_FAMILIES),
        reports_dir=str(_DEAD_REPORTS_DIR),
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        default=False,
        dest="write_summary",
        help="dead 리포트 + pilot_v3 리포트에서 floor 캘리브레이션 요약 작성",
    )
    parser.add_argument(
        "--pilot-v3-dir",
        type=str,
        default=str(_PILOT_V3_DIR),
        dest="pilot_v3_dir",
        help="pilot_v3 AXS-009 리포트 디렉토리",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=str(_SUMMARY_OUT_PATH),
        dest="summary_out",
        help="캘리브레이션 요약 출력 경로",
    )
    args = parser.parse_args(argv)

    if args.register_ledger:
        parser.error(
            "AXS-DEAD-CAL: 실행 원장(run_ledger) 등록은 금지되어 있습니다 "
            "(--register-ledger 사용 불가)."
        )

    if args.write_summary and args.dry_run:
        parser.error(
            "AXS-DEAD-CAL: --write-summary 는 --dry-run 과 함께 사용할 수 "
            "없습니다."
        )

    if args.write_summary:
        write_calibration_summary(
            dead_reports_dir=Path(args.reports_dir),
            pilot_v3_dir=Path(args.pilot_v3_dir),
            out_path=Path(args.summary_out),
        )
        return

    base_seeds_list = parse_base_seeds(args.base_seeds)
    run_axs_dead_calibration(
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
    )


if __name__ == "__main__":
    main()
