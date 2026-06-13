"""ladder_gate — 상태 비저장 메커니즘 라이선스 심사기 (Task G-022b).

사전등록 + 실험 리포트 + 원장 + git 사실로부터 클레임 라이선스를
매 호출 시 재계산한다.

NON-NEGOTIABLE 원칙:
- licenses.json 은 출력 전용 감사 로그. 어떤 코드 경로도 입력으로 읽지 않는다.
- 자가 증명 불리언(preregRemoteVerified, pass, accepted 등)은 무시한다.
  게이트는 원시 숫자·git 사실·파일 해시·집합 연산으로 재계산한다.
- 실패 폐쇄(fail closed). 파일/키/arm 누락 → 검사 실패, 묵시적 통과 없음.
- MVP 범위: 아래 명시된 검사만 구현. 그 이상 없음.

MVP 수용 한계 (문서화된 제약):
- replayAudit.replayable: 소비된 자가증명 불리언 (제한적 방향에서만 사용).
- 동일 버전의 unconsumed_reported 원장 엔트리는 familyHistory 에 공개되지만 M2 를 차단하지 않음.

로그/CLI 메시지: 한국어. 식별자·JSON 키: 영어.
CLI 스타일: src/echo_bench/tools/claim_check.py 참조.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.prereg import (
    entries_for_prereg,
    load_ledger,
    load_prereg,
    prereg_hash,
    run_git,
)
from echo_bench.metrics.aggregate import aggregate_values
from echo_bench.utils.hash import canonical_hash

__all__ = ["evaluate_mechanism_license", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# v3 사전등록에서 사용되는 실험 ID 집합 (하드코딩; 제네릭 규칙 인터프리터 없음)
_V3_EXPERIMENT_IDS = frozenset(
    {"AXS-IMP-001", "AXS-NOISE-001", "AXS-TB-001", "AXS-ALPHA-EXP"}
)

# AXS-IMP-001 정확한 arm 순서
_V3_IMP_ARMS = ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]
# AXS-NOISE-001 arm 집합
_V3_NOISE_ARMS = ["axs_ucb_default", "axs_yoked_bonus"]
# AXS-ALPHA-EXP arm 집합
_V3_ALPHA_ARMS = ["axs_ucb_default", "axs_ucb_alpha0"]
# AXS-TB-001 변형
_V3_TB_VARIANTS = {"reverse", "hash_seeded", "feature_lexicographic"}

# ---------------------------------------------------------------------------
# v3.1 역할별 가드 — 하드코딩 (제네릭 결정 규칙 인터프리터 금지).
# prereg.utilityGuard 와 교차 검증되며, 불일치는 검사 실패 (fail-closed).
# ---------------------------------------------------------------------------
_V31_ARM_ROLES = {
    "AXS-IMP-001": {
        "freeze_none": "live_default",
        "freeze_at_quarter": "intermediate",
        "freeze_at_half": "intermediate",
        "freeze_at_1": "key_manipulation",
    },
    "AXS-NOISE-001": {
        "axs_ucb_default": "live_default",
        "axs_yoked_bonus": "key_manipulation",
    },
    "AXS-ALPHA-EXP": {
        "axs_ucb_default": "live_default",
        "axs_ucb_alpha0": "intermediate",
    },
}
_V31_RHO = 0.70
_V31_PROBE_COUNT = 7
_V31_MIN_DISTINCT_HASHES = 2
# 트랙 평가 대상 실험 (AXS-ALPHA-EXP 는 영구 제외 — m2Prohibited)
_V31_GUARDED_EXPERIMENTS = ("AXS-IMP-001", "AXS-NOISE-001")
# v3.1 5-브랜치
_V31_BRANCHES = (
    "integrity_fail",
    "both_supported",
    "imprint_only_supported",
    "noise_only_supported",
    "no_claim_m1_only",
)


# ---------------------------------------------------------------------------
# Strict numeric type guard (ITEM 3)
# ---------------------------------------------------------------------------


def _strict_number(v: Any) -> bool:
    """True iff v is a plain int or float — booleans and non-numerics return False."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
# Report-hash helper
# ---------------------------------------------------------------------------


def _report_hash(report: Dict[str, Any]) -> str:
    """리포트에서 reportHash 키를 제외한 canonical hash 계산."""
    body = {k: v for k, v in report.items() if k != "reportHash"}
    return canonical_hash(body)


# ---------------------------------------------------------------------------
# Separability primitives
# ---------------------------------------------------------------------------


def _separability_present(
    arm: Dict[str, Any],
    metric: str,
    prereg: Dict[str, Any],
) -> bool:
    """separability_present 판정.

    perFamily[metric] > 0 인 패밀리 수 >= signRule.minConsistentFamilies
    AND bootstrap[metric].ciLower > ciRule.lowerBoundMustExceed.

    perFamily 키 집합이 prereg evaluationFamilies 와 정확히 일치해야 함.
    불일치 시 False 반환 (호출자가 arms_complete 검사에서 별도 실패 처리).
    """
    min_fam = prereg["signRule"]["minConsistentFamilies"]
    ci_bound = prereg["ciRule"]["lowerBoundMustExceed"]
    eval_fams = set(str(f) for f in prereg["evaluationFamilies"])

    per_family = arm.get("perFamily", {})
    # 키 집합 검사 (불일치 → False; arms_complete 이 별도 fail)
    if set(str(k) for k in per_family.keys()) != eval_fams:
        return False

    count = sum(
        1 for fam in eval_fams
        if _strict_number((per_family.get(fam) or {}).get(metric)) and
        (per_family.get(fam) or {}).get(metric, 0) > 0
    )
    if count < min_fam:
        return False

    bootstrap = arm.get("bootstrap", {})
    ci_lower = (bootstrap.get(metric) or {}).get("ciLower")
    if ci_lower is None or not _strict_number(ci_lower):
        return False
    return ci_lower > ci_bound


def _separability_absent(
    arm: Dict[str, Any],
    metric: str,
    prereg: Dict[str, Any],
) -> bool:
    return not _separability_present(arm, metric, prereg)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_prereg_status(prereg: Dict[str, Any]) -> Dict[str, Any]:
    """검사 0: prereg_status.

    - status 키가 있고 값이 "registered" 가 아니면 → 실패 (모든 버전).
    - status 키가 없으면:
        version 1 → 통과 (하위 호환, 관례 이전 버전).
        version >= 2 → 실패 (status 필수).
    """
    status = prereg.get("status")
    version = prereg.get("version", 1)
    if status is not None:
        if status != "registered":
            return {
                "check": "prereg_status",
                "ok": False,
                "detail": (
                    f"design-draft prereg로는 라이선스 발급 불가 "
                    f"(status={status!r}; 'registered' 이어야 함) — "
                    "등록된 사전등록만 증거 심사를 통과할 수 있음"
                ),
            }
        return {
            "check": "prereg_status",
            "ok": True,
            "detail": f"prereg status={status!r} — 등록 확인",
        }
    # status 키 없음
    if version >= 2:
        return {
            "check": "prereg_status",
            "ok": False,
            "detail": (
                f"version {version} 사전등록에 status 키 누락 — "
                "v2 이상은 status='registered' 가 명시되어야 함 (fail closed)"
            ),
        }
    # version 1 + status 없음: v3 실험 ID 가 있으면 버전 위장 우회 차단
    if _is_v3_prereg(prereg):
        return {
            "check": "prereg_status",
            "ok": False,
            "detail": (
                "v3 실험 구조 prereg는 명시적 status=='registered' 필수 — "
                "버전 위장 우회 차단 (version=1 + status 없음으로 grandfather 조항 우회 불가)"
            ),
        }
    # version 1, v3 아님: 하위 호환 허용
    return {
        "check": "prereg_status",
        "ok": True,
        "detail": "version 1 prereg — status 관례 이전 버전, 하위 호환 허용",
    }


def _check_prereg_hash_match(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
    expected_hash: str,
) -> Dict[str, Any]:
    """검사 1: prereg_hash_match."""
    prereg_id = prereg["preregId"]
    prereg_version = prereg["version"]
    ok = True
    details = []
    for r in reports:
        stamp = r.get("preregStamp") or {}
        rh = stamp.get("preregHash", "")
        rid = stamp.get("preregId", "")
        rv = stamp.get("preregVersion")
        if rh != expected_hash:
            ok = False
            details.append(
                f"리포트 {r.get('reportId')!r}: preregHash 불일치 "
                f"(기대={expected_hash[:12]}, 실제={rh[:12] if rh else '없음'})"
            )
        if rid != prereg_id:
            ok = False
            details.append(
                f"리포트 {r.get('reportId')!r}: preregId 불일치 "
                f"(기대={prereg_id!r}, 실제={rid!r})"
            )
        if rv != prereg_version:
            ok = False
            details.append(
                f"리포트 {r.get('reportId')!r}: preregVersion 불일치 "
                f"(기대={prereg_version}, 실제={rv!r})"
            )
    detail = "; ".join(details) if details else "모든 리포트 preregHash/Id/Version 일치"
    return {"check": "prereg_hash_match", "ok": ok, "detail": detail}


def _check_report_hash_integrity(
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """검사 2: report_hash_integrity."""
    ok = True
    details = []
    for r in reports:
        embedded = r.get("reportHash", "")
        computed = _report_hash(r)
        if computed != embedded:
            ok = False
            details.append(
                f"리포트 {r.get('reportId')!r}: 해시 불일치 "
                f"(내장={embedded[:12] if embedded else '없음'}, 재계산={computed[:12]})"
            )
    detail = "; ".join(details) if details else "모든 리포트 해시 무결성 확인"
    return {"check": "report_hash_integrity", "ok": ok, "detail": detail}


def _check_replayable(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """검사 3: replayable."""
    ok = True
    details = []
    for r in reports:
        audit = r.get("replayAudit") or {}
        replayable = audit.get("replayable")
        if replayable is not True:
            ok = False
            details.append(
                f"리포트 {r.get('reportId')!r}: replayAudit.replayable={replayable!r} (True 아님)"
            )
    detail = "; ".join(details) if details else "모든 리포트 재현 가능 확인"
    return {"check": "replayable", "ok": ok, "detail": detail}


def _validate_arm_degenerate_triple(
    exp_id: str,
    arm_id: str,
    arm_data: Dict[str, Any],
    random_cov: float,
    ok_ref: List[bool],
    details: List[str],
) -> None:
    """RANDOM 기준 미달 arm 의 degenerate 3개 필드를 검증하는 공통 헬퍼.

    ok_ref: [True/False] 단일 요소 리스트 — in-place 로 False 로 설정.
    v1/v3 모두 동일 경로.
    """
    degenerate_fields = {"degenerate", "degenerateReason", "includedInMechanismClaim"}
    arm_cov = (arm_data.get("utility") or {}).get("coordinate_coverage_mean")
    if not _strict_number(arm_cov):
        ok_ref[0] = False
        details.append(
            f"{exp_id} arm {arm_id!r}: utility.coordinate_coverage_mean 누락 또는 비숫자 "
            f"(값: {arm_cov!r})"
        )
        return
    if arm_cov < random_cov:
        # RANDOM 미달 → degenerate 3개 필드 필수
        for field in degenerate_fields:
            if field not in arm_data:
                ok_ref[0] = False
                details.append(
                    f"{exp_id} arm {arm_id!r}: RANDOM 기준 미달이나 "
                    f"degenerate 필드 {field!r} 누락"
                )
        if arm_data.get("degenerate") is not True:
            ok_ref[0] = False
            details.append(
                f"{exp_id} arm {arm_id!r}: RANDOM 기준 미달이나 "
                f"degenerate={arm_data.get('degenerate')!r} (True 여야 함)"
            )
        reason = arm_data.get("degenerateReason")
        if not isinstance(reason, str) or not reason.strip():
            ok_ref[0] = False
            details.append(
                f"{exp_id} arm {arm_id!r}: degenerateReason 이 비어 있거나 "
                f"문자열이 아님 ({reason!r})"
            )
        if arm_data.get("includedInMechanismClaim") is not False:
            ok_ref[0] = False
            details.append(
                f"{exp_id} arm {arm_id!r}: RANDOM 기준 미달이나 "
                f"includedInMechanismClaim="
                f"{arm_data.get('includedInMechanismClaim')!r} (False 여야 함)"
            )


def _check_arms_complete(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """검사 4: arms_complete.

    각 실험 리포트의 arm 집합이 prereg 에 등록된 것과 일치하는지 확인.
    RANDOM 기준 미달 arm 은 degenerate 3개 필드 필수.
    AXS-010/AXS-TB-001 은 tieBreak 구조 검사.
    v3 실험 ID: AXS-IMP-001/AXS-NOISE-001/AXS-TB-001/AXS-ALPHA-EXP.
    """
    ok = True
    details = []

    # ITEM 5: 중복 experimentId 검사 (먼저 수행)
    seen_exp_ids: Dict[str, int] = {}
    for i, r in enumerate(reports):
        exp_id = r.get("experimentId", "")
        if exp_id in seen_exp_ids:
            ok = False
            details.append(
                f"중복 experimentId 감지: {exp_id!r} — 인덱스 {seen_exp_ids[exp_id]} 와 {i} 에서 중복"
            )
        else:
            seen_exp_ids[exp_id] = i

    reports_by_exp: Dict[str, Dict[str, Any]] = {}
    for r in reports:
        exp_id = r.get("experimentId", "")
        reports_by_exp[exp_id] = r

    # ITEM 4: 알 수 없는 experimentId 검사
    known_exp_ids = set(prereg.get("experiments", {}).keys()) | {"AXS-010", "AXS-TB-001"}
    for r in reports:
        exp_id = r.get("experimentId", "")
        if exp_id not in known_exp_ids:
            ok = False
            details.append(
                f"알 수 없는 experimentId {exp_id!r}: prereg 에 등록되지 않은 실험"
            )

    for exp_id, exp_cfg in prereg.get("experiments", {}).items():
        r = reports_by_exp.get(exp_id)
        if r is None:
            # 해당 실험 리포트가 소비 목록에 없으면 건너뜀
            continue

        # AXS-010 (v1) 또는 AXS-TB-001 (v3): tieBreak 구조 검사
        if exp_id in ("AXS-010", "AXS-TB-001"):
            tie_break = r.get("tieBreak")
            if not isinstance(tie_break, dict):
                ok = False
                details.append(f"{exp_id}: tieBreak 필드 누락 또는 잘못된 형식")
                continue
            if "baseline" not in tie_break:
                ok = False
                details.append(f"{exp_id}: tieBreak.baseline 누락")
            required_variants = {"reverse", "hash_seeded", "feature_lexicographic"}
            variants = tie_break.get("variants") or {}
            missing_variants = required_variants - set(variants.keys())
            if missing_variants:
                ok = False
                details.append(
                    f"{exp_id}: tieBreak.variants 누락: {sorted(missing_variants)}"
                )
            continue

        # AXS-ALPHA-EXP (v3): arm 집합만 확인, baselines/degenerate 불필요 (noClaimLicense)
        if exp_id == "AXS-ALPHA-EXP":
            prereg_arms = set(exp_cfg.get("arms", []))
            report_arms = set((r.get("arms") or {}).keys())
            missing_arms = prereg_arms - report_arms
            if missing_arms:
                ok = False
                details.append(
                    f"AXS-ALPHA-EXP: arm 누락 {sorted(missing_arms)}"
                )
            continue

        # 일반 실험(v1 및 v3 IMP/NOISE): arms + baselines + degenerate 검사
        prereg_arms = set(exp_cfg.get("arms", []))
        report_arms = set((r.get("arms") or {}).keys())
        missing_arms = prereg_arms - report_arms
        if missing_arms:
            ok = False
            details.append(
                f"{exp_id}: arm 누락 {sorted(missing_arms)}"
            )
            continue

        # ITEM 2: baselines.RANDOM.coordinate_coverage_mean 필수
        random_cov = (r.get("baselines") or {}).get("RANDOM", {}).get(
            "coordinate_coverage_mean"
        )
        if not _strict_number(random_cov):
            ok = False
            details.append(
                f"{exp_id}: baselines.RANDOM.coordinate_coverage_mean 누락 또는 비숫자 "
                f"(값: {random_cov!r}) — RANDOM 기준 없이 degenerate 정책 적용 불가"
            )
            continue

        # v3.1 confirmatory(IMP/NOISE): RANDOM 은 reference-only —
        # RANDOM-parity 기반 degenerate 트리플 검증은 role_specific_guard 의
        # 역할별 재계산으로 대체된다 (v1 경로는 아래 parity 검증 유지).
        if exp_id in _V31_GUARDED_EXPERIMENTS:
            continue

        ok_ref = [ok]
        for arm_id, arm_data in (r.get("arms") or {}).items():
            if arm_id not in prereg_arms:
                continue
            _validate_arm_degenerate_triple(
                exp_id, arm_id, arm_data, random_cov, ok_ref, details
            )
        ok = ok_ref[0]

    detail = "; ".join(details) if details else "모든 arm 완전성 확인"
    return {"check": "arms_complete", "ok": ok, "detail": detail}


def _check_acceptance_recomputed(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """검사 5: acceptance_recomputed.

    requiresPass 실험 전체가 소비된 리포트로 제출되어야 함 (누락 시 즉시 실패).
    각 실험별 판정 재계산:
    AXS-003: present(axs_ucb_default) AND absent(axs_ucb_alpha0) on slate_excess_nmi
    AXS-004c: present(axs_ucb_default) AND absent(axs_yoked_bonus)
    AXS-002: present(utility_matched_contrast, slate_excess_nmi_diff)
    AXS-009: freeze arm 순서 + divergence absent
    """
    ok = True
    details = []

    reports_by_exp: Dict[str, Dict[str, Any]] = {}
    for r in reports:
        exp_id = r.get("experimentId", "")
        reports_by_exp[exp_id] = r

    # requiresPass 강제: 사전등록에 명시된 실험 리포트가 모두 소비되어야 함
    # ITEM 6: 빈 목록 또는 누락 → 즉시 실패 (fail-open 방지)
    requires_pass_exps = list(
        prereg.get("claimTransitions", {}).get("M2", {}).get("requiresPass", [])
    )
    if not requires_pass_exps:
        ok = False
        details.append(
            "claimTransitions.M2.requiresPass 가 비어 있거나 누락됨 — "
            "M2 라이선스 발급 조건 미충족 (fail-open 방지)"
        )
    else:
        missing_exps = [exp for exp in requires_pass_exps if exp not in reports_by_exp]
        if missing_exps:
            ok = False
            details.append(
                f"requiresPass 실험 리포트 누락: {sorted(missing_exps)} — "
                "해당 실험 리포트 없이는 M2 라이선스 발급 불가"
            )

    # AXS-003
    r3 = reports_by_exp.get("AXS-003")
    if r3 is not None:
        arms3 = r3.get("arms") or {}
        default_arm = arms3.get("axs_ucb_default") or {}
        alpha0_arm = arms3.get("axs_ucb_alpha0") or {}
        metric = "slate_excess_nmi"
        # present side must be non-degenerate
        if default_arm.get("degenerate"):
            ok = False
            details.append("AXS-003: axs_ucb_default 는 non-degenerate 여야 함")
        else:
            if not _separability_present(default_arm, metric, prereg):
                ok = False
                details.append(
                    "AXS-003: axs_ucb_default separability_present 실패 (slate_excess_nmi)"
                )
        if not _separability_absent(alpha0_arm, metric, prereg):
            ok = False
            details.append(
                "AXS-003: axs_ucb_alpha0 separability_absent 실패 (slate_excess_nmi)"
            )

    # AXS-004c
    r4 = reports_by_exp.get("AXS-004c")
    if r4 is not None:
        arms4 = r4.get("arms") or {}
        default_arm4 = arms4.get("axs_ucb_default") or {}
        yoked_arm = arms4.get("axs_yoked_bonus") or {}
        metric4 = "slate_excess_nmi"
        if default_arm4.get("degenerate"):
            ok = False
            details.append("AXS-004c: axs_ucb_default 는 non-degenerate 여야 함")
        else:
            if not _separability_present(default_arm4, metric4, prereg):
                ok = False
                details.append(
                    "AXS-004c: axs_ucb_default separability_present 실패"
                )
        if not _separability_absent(yoked_arm, metric4, prereg):
            ok = False
            details.append(
                "AXS-004c: axs_yoked_bonus separability_absent 실패"
            )

    # AXS-002
    r2 = reports_by_exp.get("AXS-002")
    if r2 is not None:
        arms2 = r2.get("arms") or {}
        contrast_arm = arms2.get("utility_matched_contrast") or {}
        metric2 = "slate_excess_nmi_diff"
        if contrast_arm.get("degenerate"):
            ok = False
            details.append("AXS-002: utility_matched_contrast 는 non-degenerate 여야 함")
        else:
            if not _separability_present(contrast_arm, metric2, prereg):
                ok = False
                details.append(
                    "AXS-002: utility_matched_contrast separability_present 실패 "
                    "(slate_excess_nmi_diff)"
                )

    # AXS-009
    r9 = reports_by_exp.get("AXS-009")
    if r9 is not None:
        arms9 = r9.get("arms") or {}
        freeze_order = ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]
        metric9_div = "post_freeze_incremental_divergence"
        metric9_nmi = "slate_excess_nmi"

        # postFreezeAccrualAbsent: frozen arms' bootstrap ciLower <= 0
        for arm_id in ["freeze_at_1", "freeze_at_quarter", "freeze_at_half"]:
            arm_data = arms9.get(arm_id) or {}
            bootstrap = arm_data.get("bootstrap") or {}
            ci_lower = (bootstrap.get(metric9_div) or {}).get("ciLower")
            if ci_lower is None:
                ok = False
                details.append(
                    f"AXS-009 arm {arm_id!r}: bootstrap[post_freeze_incremental_divergence].ciLower 누락"
                )
            elif ci_lower > 0:
                ok = False
                details.append(
                    f"AXS-009 arm {arm_id!r}: ciLower={ci_lower} > 0 "
                    "(postFreezeAccrualAbsent 조건 실패)"
                )

        # doseResponseOrdering: means non-decreasing freeze_at_1 ≤ quarter ≤ half ≤ freeze_none
        means = []
        for arm_id in freeze_order:
            arm_data = arms9.get(arm_id) or {}
            bootstrap = arm_data.get("bootstrap") or {}
            mean_val = (bootstrap.get(metric9_nmi) or {}).get("mean")
            means.append((arm_id, mean_val))

        for i in range(len(means) - 1):
            a_id, a_mean = means[i]
            b_id, b_mean = means[i + 1]
            if a_mean is None or b_mean is None:
                ok = False
                details.append(
                    f"AXS-009: bootstrap[slate_excess_nmi].mean 누락 "
                    f"({a_id} 또는 {b_id})"
                )
            elif a_mean > b_mean:
                ok = False
                details.append(
                    f"AXS-009: dose-response 순서 위반 "
                    f"{a_id}({a_mean}) > {b_id}({b_mean})"
                )

    detail = "; ".join(details) if details else "모든 실험 판정 재계산 통과"
    return {"check": "acceptance_recomputed", "ok": ok, "detail": detail}


# ---------------------------------------------------------------------------
# v3 delta computation helpers
# ---------------------------------------------------------------------------


def _compute_delta_from_arms(
    arm_a: Dict[str, Any],
    arm_b: Dict[str, Any],
    metric: str,
    eval_fams: List[str],
) -> tuple[Optional[Dict[str, float]], Optional[str]]:
    """arm_a[fam][metric] - arm_b[fam][metric] 의 per-family delta 계산.

    Returns (delta_dict, error_msg).
    delta_dict: name-sorted family keys → float delta.
    error_msg: None 이면 성공; 실패 이유 문자열.

    - 패밀리 집합이 두 arm 간에 정확히 일치해야 함 (불일치 → 오류).
    - 모든 값이 strict_number 이어야 함 (불리언 위장 → 오류).
    """
    fams_set = set(str(f) for f in eval_fams)

    pf_a = arm_a.get("perFamily") or {}
    pf_b = arm_b.get("perFamily") or {}

    keys_a = set(str(k) for k in pf_a.keys())
    keys_b = set(str(k) for k in pf_b.keys())

    if keys_a != fams_set:
        return None, (
            f"arm_a perFamily 키 집합 {sorted(keys_a)} 이 "
            f"evaluationFamilies {sorted(fams_set)} 와 불일치"
        )
    if keys_b != fams_set:
        return None, (
            f"arm_b perFamily 키 집합 {sorted(keys_b)} 이 "
            f"evaluationFamilies {sorted(fams_set)} 와 불일치"
        )

    delta: Dict[str, float] = {}
    for fam in sorted(fams_set):
        va = (pf_a.get(fam) or {}).get(metric)
        vb = (pf_b.get(fam) or {}).get(metric)
        if not _strict_number(va):
            return None, (
                f"arm_a perFamily[{fam!r}][{metric!r}] 이 strict_number 가 아님: {va!r}"
            )
        if not _strict_number(vb):
            return None, (
                f"arm_b perFamily[{fam!r}][{metric!r}] 이 strict_number 가 아님: {vb!r}"
            )
        delta[fam] = float(va) - float(vb)
    return delta, None


def _delta_primary_pass(
    delta: Dict[str, float],
    min_consistent: int,
    ci_bound: float,
    delta_key: str,
) -> tuple[bool, bool, float]:
    """primary pass 규칙 평가.

    Returns (sign_pass, ci_pass, ci_low).
    sign_pass: >= min_consistent families 에서 delta > 0.
    ci_pass: gate 의 own seeded bootstrap ciLow > ci_bound.
    name-sorted family order 로 aggregate_values 호출.
    """
    sorted_vals = [delta[k] for k in sorted(delta.keys())]
    sign_count = sum(1 for v in sorted_vals if v > 0)
    sign_pass = sign_count >= min_consistent

    agg = aggregate_values(sorted_vals, key=delta_key)
    ci_low = agg["ci_low"]
    ci_pass = ci_low > ci_bound
    return sign_pass, ci_pass, ci_low


def _check_acceptance_recomputed_v3(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> tuple[
    Dict[str, Any],  # check result
    bool,  # delta_imp_pass
    bool,  # delta_noise_pass
]:
    """검사 5-v3: acceptance_recomputed_v3 (v3.1).

    v3 사전등록(AXS-IMP-001/AXS-NOISE-001/AXS-ALPHA-EXP)에 대한 판정 재계산.

    - 리포트에 내장된 delta 블록은 전적으로 무시 (재계산).
    - AXS-ALPHA-EXP 는 구조만 확인, 판정 계산 없음 (트랙 평가 영구 제외).
    - requiresPass: 트랙별 claimTransitions(M2-IMP/M2-NOISE/M2) 강제 +
      AXS-TB-001 리포트 필수 (누락 → 즉시 실패, fail-open 방지).
    - degenerate 여부는 여기서 계산하지 않음 — role_specific_guard 가 담당.
    """
    ok = True
    details = []

    reports_by_exp: Dict[str, Dict[str, Any]] = {}
    for r in reports:
        reports_by_exp[r.get("experimentId", "")] = r

    eval_fams = [str(f) for f in prereg.get("evaluationFamilies", [])]
    min_consistent = prereg.get("signRule", {}).get("minConsistentFamilies", 4)
    ci_bound = prereg.get("ciRule", {}).get("lowerBoundMustExceed", 0.0)
    metric = "slate_excess_nmi"

    # 트랙별 requiresPass 강제 (fail-open 방지)
    transitions = prereg.get("claimTransitions", {}) or {}
    required_exps: set = set()
    for track in ("M2-IMP", "M2-NOISE", "M2"):
        track_rp = list((transitions.get(track) or {}).get("requiresPass", []) or [])
        if not track_rp:
            ok = False
            details.append(
                f"claimTransitions['{track}'].requiresPass 가 비어 있거나 누락됨 — "
                "라이선스 발급 조건 미충족 (fail-open 방지)"
            )
        required_exps.update(track_rp)
    # AXS-TB-001 은 모든 트랙의 requiresInvariance 대상 — 리포트 필수
    required_exps.add("AXS-TB-001")
    missing_exps = sorted(exp for exp in required_exps if exp not in reports_by_exp)
    if missing_exps:
        ok = False
        details.append(
            f"requiresPass/필수 실험 리포트 누락: {missing_exps} — "
            "해당 실험 리포트 없이는 트랙 라이선스 발급 불가"
        )

    # AXS-IMP-001 delta_imp
    delta_imp_pass = False

    r_imp = reports_by_exp.get("AXS-IMP-001")
    if r_imp is not None:
        arms_imp = r_imp.get("arms") or {}
        arm_freeze1 = arms_imp.get("freeze_at_1") or {}
        arm_freeze_none = arms_imp.get("freeze_none") or {}

        # delta_imp = sep(freeze_at_1)[fam] - sep(freeze_none)[fam]
        delta_imp_vals, err = _compute_delta_from_arms(
            arm_freeze1, arm_freeze_none, metric, eval_fams
        )
        if err is not None:
            ok = False
            details.append(f"AXS-IMP-001 delta_imp 계산 실패: {err}")
        else:
            assert delta_imp_vals is not None
            sign_pass, ci_pass, ci_low = _delta_primary_pass(
                delta_imp_vals, min_consistent, ci_bound, "delta_imp"
            )
            delta_imp_pass = sign_pass and ci_pass
            if not sign_pass:
                pos_count = sum(1 for v in delta_imp_vals.values() if v > 0)
                details.append(
                    f"AXS-IMP-001 delta_imp 부호 일관성 불충분: "
                    f"{pos_count}/{len(delta_imp_vals)} (최소 {min_consistent} 필요)"
                )
            if not ci_pass:
                details.append(
                    f"AXS-IMP-001 delta_imp bootstrap ciLow={ci_low:.6f} ≤ {ci_bound} — CI 조건 불충족"
                )

            # 이차 대비 (M2 비차단, 정보 공개용)
            for sec_name, arm_id_sec in [("delta_q", "freeze_at_quarter"), ("delta_h", "freeze_at_half")]:
                arm_sec = arms_imp.get(arm_id_sec) or {}
                try:
                    sec_delta, sec_err = _compute_delta_from_arms(
                        arm_sec, arm_freeze_none, metric, eval_fams
                    )
                except Exception as exc:
                    sec_err = str(exc)
                    sec_delta = None
                if sec_err is not None:
                    details.append(
                        f"AXS-IMP-001 {sec_name} 재계산 불가: {sec_err} (정보 공개, M2 비차단)"
                    )
                elif sec_delta is not None:
                    sec_pos = sum(1 for v in sec_delta.values() if v > 0)
                    details.append(
                        f"AXS-IMP-001 {sec_name} 이차 부호: {sec_pos}/{len(sec_delta)} 양수 (정보 공개, M2 비차단)"
                    )

    # AXS-NOISE-001 delta_noise
    delta_noise_pass = False

    r_noise = reports_by_exp.get("AXS-NOISE-001")
    if r_noise is not None:
        arms_noise = r_noise.get("arms") or {}
        arm_default = arms_noise.get("axs_ucb_default") or {}
        arm_yoked = arms_noise.get("axs_yoked_bonus") or {}

        delta_noise_vals, err = _compute_delta_from_arms(
            arm_default, arm_yoked, metric, eval_fams
        )
        if err is not None:
            ok = False
            details.append(f"AXS-NOISE-001 delta_noise 계산 실패: {err}")
        else:
            assert delta_noise_vals is not None
            sign_pass, ci_pass, ci_low = _delta_primary_pass(
                delta_noise_vals, min_consistent, ci_bound, "delta_noise"
            )
            delta_noise_pass = sign_pass and ci_pass
            if not sign_pass:
                pos_count = sum(1 for v in delta_noise_vals.values() if v > 0)
                details.append(
                    f"AXS-NOISE-001 delta_noise 부호 일관성 불충분: "
                    f"{pos_count}/{len(delta_noise_vals)} (최소 {min_consistent} 필요)"
                )
            if not ci_pass:
                details.append(
                    f"AXS-NOISE-001 delta_noise bootstrap ciLow={ci_low:.6f} ≤ {ci_bound} — CI 조건 불충족"
                )

            # yokedAbsence 이차 공개 (M2 비차단)
            # yokedAbsence: yoked arm 자체의 분리성이 없음
            # = axs_yoked_bonus perFamily[metric] > 0 인 패밀리 수 < min_consistent
            #   OR yoked bootstrap[metric].ciLower <= ci_bound
            pf_yoked = arm_yoked.get("perFamily") or {}
            yoked_sep_pos = sum(
                1 for fam in eval_fams
                if _strict_number((pf_yoked.get(fam) or {}).get(metric))
                and (pf_yoked.get(fam) or {}).get(metric, 0) > 0
            )
            yoked_ci_lower = (
                (arm_yoked.get("bootstrap") or {})
                .get(metric, {})
                .get("ciLower")
            )
            yoked_absence = (yoked_sep_pos < min_consistent) or (
                yoked_ci_lower is None
                or not _strict_number(yoked_ci_lower)
                or yoked_ci_lower <= ci_bound
            )
            details.append(
                f"AXS-NOISE-001 yokedAbsence 이차: {'확인' if yoked_absence else '미확인'} (정보 공개, M2 비차단)"
            )

    detail = "; ".join(details) if details else "v3 모든 실험 판정 재계산 통과"
    return (
        {"check": "acceptance_recomputed", "ok": ok, "detail": detail},
        delta_imp_pass,
        delta_noise_pass,
    )


# ---------------------------------------------------------------------------
# v3.1 — 역할별 가드 재계산 + 알파 M2 금지 + canonical sentences
# ---------------------------------------------------------------------------


# 캘리브레이션 산출물 고정 경로 (저장소 루트 기준 — 유일하게 허용되는 인용 위치).
# 절대 경로·우회 상대 경로(../)·임의 위치 산출물은 모두 거부 (fail-closed).
_CALIBRATION_ARTIFACT_RELPATH = "configs/prereg/axs_dead_calibration_v1.json"


def _verify_floor_calibration(
    key_rule: Dict[str, Any],
    floor: float,
    details: List[str],
) -> bool:
    """absoluteFloor 를 커밋된 dead-arm 캘리브레이션 산출물과 대조 검증.

    - prereg 인용 artifactPath 는 고정 저장소 상대 경로
      (_CALIBRATION_ARTIFACT_RELPATH) 와 정확히 일치해야 함 — 절대 경로,
      경로 우회(../), 다른 위치의 산출물은 거부 (위조 산출물 인용 차단).
    - 고정 경로(저장소 루트 기준) 산출물 로드
    - 산출물 summaryHash 자기 검증 (canonical_hash(doc - {summaryHash}))
    - 인용 summaryHash 일치 + floor 동등성 요구
    실패 시 details 에 한국어 사유 추가 후 False (fail-closed).
    """
    calib = key_rule.get("floorCalibration") or {}
    artifact_path = calib.get("artifactPath")
    cited_hash = calib.get("summaryHash")
    if not isinstance(artifact_path, str) or not artifact_path:
        details.append(
            "keyManipulationRule.floorCalibration.artifactPath 누락 — "
            "floor 캘리브레이션 검증 불가 (fail-closed)"
        )
        return False
    if not isinstance(cited_hash, str) or not cited_hash:
        details.append(
            "keyManipulationRule.floorCalibration.summaryHash 누락 — "
            "floor 캘리브레이션 검증 불가 (fail-closed)"
        )
        return False

    if artifact_path != _CALIBRATION_ARTIFACT_RELPATH:
        details.append(
            f"floorCalibration.artifactPath={artifact_path!r} 가 고정 캘리브레이션 "
            f"경로 {_CALIBRATION_ARTIFACT_RELPATH!r} 와 불일치 — 절대 경로·경로 "
            "우회·임의 위치 산출물 인용 금지 (fail-closed)"
        )
        return False

    p = _REPO_ROOT / _CALIBRATION_ARTIFACT_RELPATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        details.append(
            f"캘리브레이션 산출물 로드 실패: {p} — {exc} (fail-closed)"
        )
        return False
    if not isinstance(artifact, dict):
        details.append(
            f"캘리브레이션 산출물이 JSON 객체가 아님: {p} (fail-closed)"
        )
        return False

    embedded_hash = artifact.get("summaryHash")
    body = {k: v for k, v in artifact.items() if k != "summaryHash"}
    computed_hash = canonical_hash(body)
    if computed_hash != embedded_hash:
        details.append(
            f"캘리브레이션 산출물 summaryHash 자기 검증 실패: "
            f"내장={str(embedded_hash)[:12]}, 재계산={computed_hash[:12]} — 본문 변조 의심"
        )
        return False
    if embedded_hash != cited_hash:
        details.append(
            f"prereg 인용 summaryHash({cited_hash[:12]}) 가 산출물 "
            f"summaryHash({str(embedded_hash)[:12]}) 와 불일치"
        )
        return False

    artifact_floor = (artifact.get("floorDerivation") or {}).get("absoluteFloor")
    if not _strict_number(artifact_floor) or float(artifact_floor) != float(floor):
        details.append(
            f"absoluteFloor 불일치: prereg={floor!r}, "
            f"산출물 floorDerivation.absoluteFloor={artifact_floor!r}"
        )
        return False
    return True


def _validate_seq_hash_structure(
    ssh: Any,
    eval_fams: List[str],
) -> Optional[str]:
    """key arm 의 slateSequenceHashes 구조 검증.

    요구: 키 == 평가 패밀리 집합 정확히, 패밀리당 정확히 7개 probe 키,
    probe 키 집합은 패밀리 간 일치, 값은 비공백 문자열.
    문제 발견 시 한국어 메시지 반환, 정상이면 None.
    """
    if not isinstance(ssh, dict) or not ssh:
        return "slateSequenceHashes 누락 또는 객체가 아님 (key arm 필수, fail-closed)"
    fams_expected = set(str(f) for f in eval_fams)
    fams_actual = set(str(k) for k in ssh.keys())
    if fams_actual != fams_expected:
        return (
            f"slateSequenceHashes 패밀리 키 집합 {sorted(fams_actual)} 이 "
            f"평가 패밀리 {sorted(fams_expected)} 와 불일치"
        )
    ref_probes: Optional[set] = None
    for fam in sorted(fams_actual):
        probes = ssh.get(fam)
        if not isinstance(probes, dict):
            return f"패밀리 {fam!r} 의 slateSequenceHashes 가 객체가 아님"
        if len(probes) != _V31_PROBE_COUNT:
            return (
                f"패밀리 {fam!r} 의 probe 키 수 {len(probes)} ≠ {_V31_PROBE_COUNT}"
            )
        probe_set = set(str(k) for k in probes.keys())
        if ref_probes is None:
            ref_probes = probe_set
        elif probe_set != ref_probes:
            return (
                f"패밀리 {fam!r} 의 probe 키 집합이 다른 패밀리와 불일치 "
                f"({sorted(probe_set ^ ref_probes)})"
            )
        for probe, h in probes.items():
            if not isinstance(h, str) or not h.strip():
                return f"패밀리 {fam!r} probe {probe!r} 의 해시가 비공백 문자열이 아님"
    return None


def _check_role_specific_guard(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, bool]]:
    """검사 v3.1: role_specific_guard — 역할별 utility 가드 재계산.

    자기 보고 플래그를 절대 신뢰하지 않고 원시 리포트 값으로 재계산한다.
    - 하드코딩 armRoles/rho 를 prereg.utilityGuard 와 교차 검증 (불일치 → 실패).
    - absoluteFloor 를 prereg 에서 읽되 커밋된 캘리브레이션 산출물과 대조 검증.
    - control/intermediate: coverage_mean ≥ rho × live_default coverage_mean.
    - key_manipulation: coverage_mean ≥ absoluteFloor AND 프로브 응답성
      (정확한 패밀리/probe 키 집합 + 패밀리당 고유 해시 ≥2).
    - 일관성 규칙: 자기 보고 degenerate 플래그가 재계산 결과와 모순(양방향)
      → 검사 실패 (변조/오계산 리포트, fail-closed).
    - 정직하게 보고된 가드 실패는 검사 실패가 아니라 트랙 강등.

    Returns: (check_result, {experiment_id: 모든 arm 가드 통과 여부})
    """
    ok = True
    details: List[str] = []
    guard_pass: Dict[str, bool] = {exp: False for exp in _V31_GUARDED_EXPERIMENTS}

    guard = prereg.get("utilityGuard")
    if not isinstance(guard, dict):
        return (
            {
                "check": "role_specific_guard",
                "ok": False,
                "detail": (
                    "prereg 에 utilityGuard 없음 — RANDOM-parity 부활 불가, "
                    "역할별 가드 정의 필수 (fail-closed)"
                ),
            },
            guard_pass,
        )

    # 1. armRoles 교차 검증 (하드코딩 vs prereg — 정확 일치)
    arm_roles = guard.get("armRoles")
    if arm_roles != _V31_ARM_ROLES:
        ok = False
        details.append(
            "utilityGuard.armRoles 가 게이트 하드코딩 역할 매핑과 불일치 — "
            f"prereg={arm_roles!r} (fail-closed)"
        )

    # 2. rho 교차 검증
    rho = (guard.get("controlIntermediateRule") or {}).get("rho")
    if not _strict_number(rho) or float(rho) != _V31_RHO:
        ok = False
        details.append(
            f"controlIntermediateRule.rho={rho!r} 가 게이트 하드코딩 "
            f"{_V31_RHO} 와 불일치 (fail-closed)"
        )

    # 3. absoluteFloor (strict number) + 캘리브레이션 산출물 대조
    key_rule = guard.get("keyManipulationRule") or {}
    floor = key_rule.get("absoluteFloor")
    if not _strict_number(floor):
        ok = False
        details.append(
            f"keyManipulationRule.absoluteFloor={floor!r} 누락 또는 비숫자 (fail-closed)"
        )
    else:
        if not _verify_floor_calibration(key_rule, float(floor), details):
            ok = False

    # 4. probeResponsiveness 파라미터 교차 검증
    probe_resp = key_rule.get("probeResponsiveness") or {}
    if probe_resp.get("probeCount") != _V31_PROBE_COUNT or (
        probe_resp.get("minDistinctHashesPerFamily") != _V31_MIN_DISTINCT_HASHES
    ):
        ok = False
        details.append(
            "keyManipulationRule.probeResponsiveness 의 probeCount/"
            "minDistinctHashesPerFamily 가 게이트 하드코딩 "
            f"({_V31_PROBE_COUNT}/{_V31_MIN_DISTINCT_HASHES}) 과 불일치 (fail-closed)"
        )

    if not ok:
        # 구조/파라미터 실패 시 arm 평가 생략 (기준값 신뢰 불가)
        return (
            {
                "check": "role_specific_guard",
                "ok": False,
                "detail": "; ".join(details),
            },
            guard_pass,
        )

    floor_f = float(floor)
    eval_fams = [str(f) for f in prereg.get("evaluationFamilies", [])]

    reports_by_exp: Dict[str, Dict[str, Any]] = {}
    for r in reports:
        reports_by_exp[r.get("experimentId", "")] = r

    for exp_id in _V31_GUARDED_EXPERIMENTS:
        r = reports_by_exp.get(exp_id)
        if r is None:
            details.append(
                f"{exp_id}: 리포트 미제출 — 가드 평가 불가 "
                "(acceptance_recomputed 에서 별도 실패)"
            )
            continue
        roles = _V31_ARM_ROLES[exp_id]
        arms = r.get("arms") or {}

        extra_arms = sorted(set(arms.keys()) - set(roles.keys()))
        if extra_arms:
            ok = False
            details.append(
                f"{exp_id}: 역할 미정의 arm {extra_arms} — armRoles 에 없는 arm (fail-closed)"
            )

        live_arm_id = next(a for a, role in roles.items() if role == "live_default")
        live_cov = ((arms.get(live_arm_id) or {}).get("utility") or {}).get(
            "coordinate_coverage_mean"
        )
        if not _strict_number(live_cov):
            ok = False
            details.append(
                f"{exp_id} live_default arm {live_arm_id!r}: "
                f"coordinate_coverage_mean 누락 또는 비숫자 (값: {live_cov!r}) — "
                "기준값 없이 가드 평가 불가 (fail-closed)"
            )
            continue

        exp_all_pass = True
        for arm_id, role in roles.items():
            arm = arms.get(arm_id)
            if not isinstance(arm, dict):
                ok = False
                exp_all_pass = False
                details.append(f"{exp_id} arm {arm_id!r}: 누락 — 가드 평가 불가")
                continue
            cov = (arm.get("utility") or {}).get("coordinate_coverage_mean")
            if not _strict_number(cov):
                ok = False
                exp_all_pass = False
                details.append(
                    f"{exp_id} arm {arm_id!r}: coordinate_coverage_mean 누락 또는 "
                    f"비숫자 (값: {cov!r})"
                )
                continue
            cov_f = float(cov)

            structural_fail = False
            if role == "live_default":
                arm_pass = True
            elif role == "intermediate":
                arm_pass = cov_f >= _V31_RHO * float(live_cov)
                if not arm_pass:
                    details.append(
                        f"{exp_id} arm {arm_id!r}: intermediate 가드 실패 — "
                        f"coverage {cov_f:.6f} < rho({_V31_RHO}) × live_default "
                        f"{float(live_cov):.6f}"
                    )
            else:  # key_manipulation
                ssh = arm.get("slateSequenceHashes")
                struct_err = _validate_seq_hash_structure(ssh, eval_fams)
                if struct_err is not None:
                    ok = False
                    exp_all_pass = False
                    structural_fail = True
                    arm_pass = False
                    details.append(f"{exp_id} arm {arm_id!r}: {struct_err}")
                else:
                    non_responsive = sorted(
                        str(fam)
                        for fam, probes in ssh.items()
                        if len(set(probes.values())) < _V31_MIN_DISTINCT_HASHES
                    )
                    floor_ok = cov_f >= floor_f
                    arm_pass = floor_ok and not non_responsive
                    if not floor_ok:
                        details.append(
                            f"{exp_id} arm {arm_id!r}: key 가드 실패 — "
                            f"coverage {cov_f:.6f} < absoluteFloor {floor_f:.6f}"
                        )
                    if non_responsive:
                        details.append(
                            f"{exp_id} arm {arm_id!r}: key 가드 실패 — "
                            f"프로브 비응답 패밀리 {non_responsive} "
                            f"(고유 해시 <{_V31_MIN_DISTINCT_HASHES})"
                        )

            if structural_fail:
                continue  # 구조 위반은 이미 검사 실패 — 일관성 비교 생략

            # 일관성 규칙: 자기 보고 vs 재계산 (양방향 모순 → 검사 실패)
            self_degenerate = arm.get("degenerate") is True
            recomputed_fail = not arm_pass
            if self_degenerate != recomputed_fail:
                ok = False
                details.append(
                    f"{exp_id} arm {arm_id!r}: 자기 보고 degenerate={self_degenerate} 가 "
                    f"재계산된 역할별 가드 결과(실패={recomputed_fail})와 모순 — "
                    "변조/오계산 리포트 (fail-closed)"
                )
            elif recomputed_fail:
                # 정직한 가드 실패 — 트리플 관례 검증 후 트랙 강등
                reason = arm.get("degenerateReason")
                if not isinstance(reason, str) or not reason.strip():
                    ok = False
                    details.append(
                        f"{exp_id} arm {arm_id!r}: degenerateReason 이 비어 있거나 "
                        f"문자열이 아님 ({reason!r})"
                    )
                if arm.get("includedInMechanismClaim") is not False:
                    ok = False
                    details.append(
                        f"{exp_id} arm {arm_id!r}: includedInMechanismClaim="
                        f"{arm.get('includedInMechanismClaim')!r} (False 여야 함)"
                    )
                details.append(
                    f"{exp_id} arm {arm_id!r}: 역할별 가드 실패 (정직 보고) — 트랙 강등"
                )

            if not arm_pass:
                exp_all_pass = False

        guard_pass[exp_id] = exp_all_pass

    detail = "; ".join(details) if details else "역할별 가드 재계산 전체 통과"
    return (
        {"check": "role_specific_guard", "ok": ok, "detail": detail},
        guard_pass,
    )


def _check_alpha_m2_prohibition(prereg: Dict[str, Any]) -> Dict[str, Any]:
    """검사 v3.1: alpha_m2_prohibition — AXS-ALPHA-EXP 의 M2 영구 금지.

    - claimTransitions 의 어떤 M2-트랙 requiresPass/requiresTracks 에도
      AXS-ALPHA-EXP 가 등장하면 안 됨.
    - prereg.experiments['AXS-ALPHA-EXP'].m2Prohibited 가 정확히 True 여야 함
      (키 누락/False → 실패).
    - v3 에서 금지 선언은 필수: experiments 에 AXS-ALPHA-EXP 엔트리 자체가
      없어도 실패 (엔트리 삭제 우회 차단, fail-closed).
    """
    ok = True
    details: List[str] = []

    transitions = prereg.get("claimTransitions", {}) or {}
    for track, trans in transitions.items():
        if not str(track).startswith("M2"):
            continue
        trans = trans or {}
        for field in ("requiresPass", "requiresTracks"):
            values = trans.get(field) or []
            if isinstance(values, list) and "AXS-ALPHA-EXP" in values:
                ok = False
                details.append(
                    f"claimTransitions['{track}'].{field} 에 AXS-ALPHA-EXP 포함 — "
                    "알파 실험의 M2 기여는 영구 금지 (fail-closed)"
                )

    alpha_cfg = (prereg.get("experiments") or {}).get("AXS-ALPHA-EXP")
    if not isinstance(alpha_cfg, dict):
        # v3 에서 금지 선언은 필수 — 엔트리 자체의 부재(삭제 우회)도 fail-closed
        ok = False
        details.append(
            "experiments['AXS-ALPHA-EXP'] 엔트리 누락 또는 객체가 아님 "
            f"(값: {alpha_cfg!r}) — v3 사전등록은 알파 M2 영구 금지 선언"
            "(m2Prohibited=True) 이 필수, 엔트리 삭제로 우회 불가 (fail-closed)"
        )
    elif alpha_cfg.get("m2Prohibited") is not True:
        ok = False
        details.append(
            "experiments['AXS-ALPHA-EXP'].m2Prohibited 가 True 가 아님 "
            f"(값: {alpha_cfg.get('m2Prohibited')!r}) — 영구 금지 선언 필수 (fail-closed)"
        )

    detail = "; ".join(details) if details else "알파 M2 영구 금지 확인"
    return {"check": "alpha_m2_prohibition", "ok": ok, "detail": detail}


def _check_canonical_sentences(prereg: Dict[str, Any]) -> Dict[str, Any]:
    """검사 v3.1: canonical_sentences — 라이선스 문장 소스 무결성.

    licensedSentences 는 prereg.canonicalSentences 에서만 발급되므로
    M-IMP / M-NOISE / M2-COMBINED 가 비공백 문자열로 존재해야 한다.
    """
    cs = prereg.get("canonicalSentences")
    if not isinstance(cs, dict):
        return {
            "check": "canonical_sentences",
            "ok": False,
            "detail": "prereg.canonicalSentences 누락 — 라이선스 문장 발급 불가 (fail-closed)",
        }
    missing = [
        key for key in ("M-IMP", "M-NOISE", "M2-COMBINED")
        if not isinstance(cs.get(key), str) or not cs.get(key, "").strip()
    ]
    if missing:
        return {
            "check": "canonical_sentences",
            "ok": False,
            "detail": (
                f"canonicalSentences 누락/비공백 아님: {missing} — "
                "라이선스 문장 발급 불가 (fail-closed)"
            ),
        }
    return {
        "check": "canonical_sentences",
        "ok": True,
        "detail": "canonicalSentences(M-IMP/M-NOISE/M2-COMBINED) 확인",
    }


def _check_ledger_registered(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
    ledger: Dict[str, Any],
    consumed_paths: List[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """검사 6: ledger_registered.

    소비된 리포트가 원장에 등록되어 있는지 확인하고 familyHistory 를 구성.
    Returns: (check_result, consumed_reports, family_history)
    """
    prereg_id = prereg["preregId"]
    prereg_version = prereg["version"]
    ok = True
    details = []

    # 원장에서 이 prereg 의 모든 엔트리 수집
    all_entries = entries_for_prereg(ledger, prereg_id)

    # consumed 리포트별 hash 집합
    consumed_hashes: Dict[str, str] = {}  # reportId -> reportHash
    consumed_reports_out: List[Dict[str, Any]] = []
    for r, path in zip(reports, consumed_paths):
        rh = _report_hash(r)
        r_id = r.get("reportId", "")
        consumed_hashes[r_id] = rh
        # 원장에 등록됐는지 확인
        found = any(e.get("reportHash") == rh for e in all_entries)
        if not found:
            ok = False
            details.append(
                f"리포트 {r_id!r} (hash={rh[:12]}): 원장 미등록"
            )
        consumed_reports_out.append({"path": path, "reportHash": rh})

    # familyHistory 구성
    family_history: List[Dict[str, Any]] = []
    for entry in all_entries:
        entry_hash = entry.get("reportHash", "")
        entry_version = entry.get("preregVersion")
        annotated = dict(entry)
        if entry_hash in consumed_hashes.values():
            annotated["disposition"] = "consumed"
        elif entry_version is not None and entry_version < prereg_version:
            annotated["disposition"] = "superseded_reported"
        else:
            annotated["disposition"] = "unconsumed_reported"
        family_history.append(annotated)

    detail = "; ".join(details) if details else "모든 리포트 원장 등록 확인"
    return (
        {"check": "ledger_registered", "ok": ok, "detail": detail},
        consumed_reports_out,
        family_history,
    )


def _check_ancestry(
    reports: List[Dict[str, Any]],
    git_runner: Optional[Callable[[List[str]], str]],
    release: bool,
) -> Dict[str, Any]:
    """검사 7: ancestry.

    merge-base --is-ancestor <preregCommit> <runCommit>.
    release=True 시 preregCommit 이 원격 브랜치에 존재하는지 추가 확인.
    """
    ok = True
    details = []
    for r in reports:
        stamp = r.get("preregStamp") or {}
        prereg_commit = stamp.get("preregCommit", "")
        run_commit = stamp.get("runCommit", "")
        r_id = r.get("reportId", "")

        if not prereg_commit or not run_commit:
            ok = False
            details.append(
                f"리포트 {r_id!r}: preregCommit 또는 runCommit 누락"
            )
            continue

        # merge-base --is-ancestor
        try:
            run_git(
                ["merge-base", "--is-ancestor", prereg_commit, run_commit],
                git_runner,
            )
        except ValueError:
            ok = False
            details.append(
                f"리포트 {r_id!r}: preregCommit {prereg_commit[:12]} 이 "
                f"runCommit {run_commit[:12]} 의 조상이 아님"
            )
            continue

        if release:
            try:
                remote_out = run_git(
                    ["branch", "-r", "--contains", prereg_commit],
                    git_runner,
                )
                if not remote_out.strip():
                    ok = False
                    details.append(
                        f"리포트 {r_id!r}: preregCommit {prereg_commit[:12]} 이 "
                        "어떤 원격 브랜치에도 없음 (release=True 조건 실패)"
                    )
            except ValueError:
                ok = False
                details.append(
                    f"리포트 {r_id!r}: 원격 브랜치 확인 실패 "
                    f"(preregCommit={prereg_commit[:12]})"
                )

    detail = "; ".join(details) if details else "모든 리포트 git ancestry 확인"
    return {"check": "ancestry", "ok": ok, "detail": detail}


def _check_pilot_disjoint(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """검사 8: pilot_disjoint.

    AXS-004c 리포트가 있고 prereg 에 yokedSchedule 이 있을 때:
    pilotFamily 가 evaluationFamilies 에 포함되지 않아야 함.
    """
    has_004c = any(r.get("experimentId") == "AXS-004c" for r in reports)
    yoked = prereg.get("yokedSchedule")
    if not has_004c or not yoked:
        return {
            "check": "pilot_disjoint",
            "ok": True,
            "detail": "AXS-004c 리포트 없거나 yokedSchedule 없음 — 검사 해당 없음",
        }

    pilot = str(yoked.get("pilotFamily", ""))
    eval_fams = set(str(f) for f in prereg.get("evaluationFamilies", []))
    if pilot in eval_fams:
        return {
            "check": "pilot_disjoint",
            "ok": False,
            "detail": (
                f"pilotFamily {pilot!r} 가 evaluationFamilies {sorted(eval_fams)} 에 포함됨 "
                "(pilot_disjoint 위반)"
            ),
        }
    return {
        "check": "pilot_disjoint",
        "ok": True,
        "detail": f"pilotFamily {pilot!r} ∉ evaluationFamilies — 분리 확인",
    }


def _check_tiebreak_invariance(
    reports: List[Dict[str, Any]],
    experiment_id: str,
) -> tuple[Dict[str, Any], bool, bool, bool]:
    """tieBreak invariance 검사 (AXS-010 또는 AXS-TB-001).

    experiment_id 로 리포트를 조회한다. v1 = "AXS-010", v3 = "AXS-TB-001".
    check 키는 호출 컨텍스트에 따라 결정 ("axs010_invariance" 또는 "axstb001_invariance").

    정직 vs 구조 분리 (역할별 가드와 동일 아키텍처):
    - well-formed 리포트의 정직한 invariance fail(부호/trackDecision 불일치)은
      증거 결과 — 검사 위반이 아니라 라이선스 강등 (ok=True, tb_pass=False).
    - 구조 위반(필수 필드 누락, 비숫자/불리언 값)은 검사 실패 (ok=False).

    Returns: (check_result, caveat_required, present, tb_pass)
    strict_pass → ok=True, tb_pass=True
    soft_pass  → ok=True, caveat_required=True, tb_pass=True
    fail (정직) → ok=True, tb_pass=False (증거 결과 — 트랙/M2 강등)
    구조 위반   → ok=False, tb_pass=False
    absent     → ok=True, present=False, tb_pass=False
    """
    check_name = (
        "axs010_invariance" if experiment_id == "AXS-010" else "axstb001_invariance"
    )
    r = next(
        (rep for rep in reports if rep.get("experimentId") == experiment_id), None
    )
    if r is None:
        return (
            {
                "check": check_name,
                "ok": True,
                "detail": f"{experiment_id} invariance 증거 없음",
            },
            False,
            False,
            False,
        )

    tie_break = r.get("tieBreak") or {}
    baseline = tie_break.get("baseline") or {}
    variants = tie_break.get("variants") or {}

    b_sign = baseline.get("sign")
    b_ci_lower = baseline.get("ciLower")
    b_ci_upper = baseline.get("ciUpper")
    b_track = baseline.get("trackDecision")

    # fail closed — baseline required fields must be present and non-None
    if b_sign is None or b_track is None or b_ci_lower is None or b_ci_upper is None:
        return (
            {
                "check": check_name,
                "ok": False,
                "detail": (
                    f"{experiment_id}: baseline 필수 필드 누락 "
                    f"(sign={b_sign!r}, trackDecision={b_track!r}, "
                    f"ciLower={b_ci_lower!r}, ciUpper={b_ci_upper!r}) — fail closed"
                ),
            },
            False,
            True,
            False,
        )

    # 구조 위반: baseline CI 경계는 strict number 여야 함 (불리언/문자열 위장 차단)
    if not _strict_number(b_ci_lower) or not _strict_number(b_ci_upper):
        return (
            {
                "check": check_name,
                "ok": False,
                "detail": (
                    f"{experiment_id}: baseline ciLower/ciUpper 가 strict number 가 아님 "
                    f"(ciLower={b_ci_lower!r}, ciUpper={b_ci_upper!r}) — fail closed"
                ),
            },
            False,
            True,
            False,
        )

    all_signs_match = True
    all_tracks_match = True
    all_within_ci = True

    for var_id, var_data in variants.items():
        v_sign = var_data.get("sign")
        v_track = var_data.get("trackDecision")
        v_est = var_data.get("estimate")

        # variant required fields must be present and non-None
        if v_sign is None or v_track is None:
            return (
                {
                    "check": check_name,
                    "ok": False,
                    "detail": (
                        f"{experiment_id}: 변형 {var_id!r} 필수 필드 누락 "
                        f"(sign={v_sign!r}, trackDecision={v_track!r}) — fail closed"
                    ),
                },
                False,
                True,
                False,
            )

        # 구조 위반: estimate 가 존재하면 strict number 여야 함
        if v_est is not None and not _strict_number(v_est):
            return (
                {
                    "check": check_name,
                    "ok": False,
                    "detail": (
                        f"{experiment_id}: 변형 {var_id!r} estimate 가 strict number "
                        f"가 아님 ({v_est!r}) — fail closed"
                    ),
                },
                False,
                True,
                False,
            )

        if v_sign != b_sign:
            all_signs_match = False
        if v_track != b_track:
            all_tracks_match = False
        if (
            v_est is not None
            and not (b_ci_lower <= v_est <= b_ci_upper)
        ):
            all_within_ci = False

    if not all_signs_match or not all_tracks_match:
        # 정직한 invariance fail — 증거 결과 (검사 위반 아님, 라이선스 강등)
        return (
            {
                "check": check_name,
                "ok": True,
                "detail": (
                    f"{experiment_id} fail (정직한 증거 결과): 변형 부호 또는 "
                    "trackDecision 불일치 — invariance 미충족, 게이트 무결성 위반 아님 "
                    "(해당 라이선스 강등)"
                ),
            },
            False,
            True,
            False,
        )
    if all_within_ci:
        return (
            {
                "check": check_name,
                "ok": True,
                "detail": f"{experiment_id} strict_pass: 모든 변형 부호·추정치 CI 내 일치",
            },
            False,
            True,
            True,
        )
    else:
        return (
            {
                "check": check_name,
                "ok": True,
                "detail": f"{experiment_id} soft_pass: 부호·trackDecision 일치, CI 외 추정치 존재 → caveatRequired",
            },
            True,
            True,
            True,
        )


def _check_axs010_invariance(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], bool, bool, bool]:
    """검사 9: axs010_invariance (v1 호환 래퍼).

    Returns: (check_result, caveat_required, axs010_present, tb_pass)
    """
    result, caveat, present, tb_pass = _check_tiebreak_invariance(reports, "AXS-010")
    return result, caveat, present, tb_pass


def _check_branch_decision(
    checks_ok: bool,
    m2_imp: bool,
    m2_noise: bool,
) -> Dict[str, Any]:
    """검사 10-v3.1: branch_decision — 5-브랜치 트리 (branchCountFrozen).

    트랙 라이선스(M2-IMP/M2-NOISE)는 호출자가 계산:
        트랙 = 검사 전체 통과 ∧ primary contrast 통과 ∧ 해당 실험의
        역할별 가드 전 arm 통과 ∧ AXS-TB-001 ≥ soft_pass.

    1. 어떤 필수 검사 실패 → "integrity_fail"
    2. 두 트랙 라이선스      → "both_supported"
    3. IMP 트랙만            → "imprint_only_supported"
    4. NOISE 트랙만          → "noise_only_supported"
    5. 그 외                 → "no_claim_m1_only"
    """
    if not checks_ok:
        branch = "integrity_fail"
    elif m2_imp and m2_noise:
        branch = "both_supported"
    elif m2_imp:
        branch = "imprint_only_supported"
    elif m2_noise:
        branch = "noise_only_supported"
    else:
        branch = "no_claim_m1_only"

    assert branch in _V31_BRANCHES
    return {
        "check": "branch_decision",
        "ok": True,  # 브랜치 결정 자체는 항상 ok (결과는 branch 필드에)
        "detail": f"브랜치 결정: {branch!r}",
        "branch": branch,
    }


# ---------------------------------------------------------------------------
# v3 prereg 감지 헬퍼
# ---------------------------------------------------------------------------


def _is_v3_prereg(prereg: Dict[str, Any]) -> bool:
    """prereg 가 v3 실험 ID 집합을 포함하면 True."""
    exp_ids = set(prereg.get("experiments", {}).keys())
    return bool(exp_ids & _V3_EXPERIMENT_IDS)


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------


def evaluate_mechanism_license(
    prereg_path: Any,
    report_paths: List[Any],
    *,
    ledger_path: Any,
    git_runner: Optional[Callable[[List[str]], str]] = None,
    release: bool = False,
) -> Dict[str, Any]:
    """사전등록 + 리포트 + 원장 + git 사실에서 메커니즘 라이선스 재계산.

    Args:
        prereg_path: 사전등록 JSON 파일 경로.
        report_paths: 실험 리포트 JSON 파일 경로 목록.
        ledger_path: 실행 원장 JSON 파일 경로.
        git_runner: 테스트용 git 명령 인젝터 (optional).
        release: True 이면 원격 브랜치 ancestry 추가 확인.

    Returns:
        rungs, checks, caveatRequired, consumedReports, familyHistory, provisional
        포함 dict. v3 prereg 에서는 추가로:
        - rungs["M2-IMP"], rungs["M2-NOISE"]: 트랙별 라이선스 (bool)
        - rungs["M2"]: 결합 라이선스 (두 트랙 모두 통과 시에만 True)
        - branch: 5-브랜치 중 하나 (integrity_fail / both_supported /
          imprint_only_supported / noise_only_supported / no_claim_m1_only)
        - licensedSentences: prereg.canonicalSentences 에서 발급된 정확한
          문장 목록 (M2-IMP → "M-IMP", M2-NOISE → "M-NOISE",
          결합 → 추가로 "M2-COMBINED"; 미라이선스 시 빈 목록)
        v1 prereg 경로는 branch/licensedSentences/트랙 키 없음 (하위 호환).
        caveatRequired 의 의미 불변 (TB soft_pass → True; 모든 라이선스 문장에 적용).

    NON-NEGOTIABLE: licenses.json 은 절대 읽지 않음. 자가 증명 불리언 무시.
    """
    prereg_path = Path(prereg_path)
    ledger_path = Path(ledger_path)

    log_ko(_logger, f"ladder_gate 평가 시작: prereg={prereg_path}, release={release}")

    # 사전등록 로드
    prereg = load_prereg(prereg_path)
    expected_hash = prereg_hash(prereg)
    prereg_id = prereg["preregId"]

    # 원장 로드
    ledger = load_ledger(ledger_path)

    # 리포트 로드
    reports: List[Dict[str, Any]] = []
    str_report_paths: List[str] = []
    for rp in report_paths:
        rp = Path(rp)
        try:
            with open(rp, "r", encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"리포트 파일 로드 실패: {rp} — {exc}") from exc
        # ITEM 7: 리포트가 dict 가 아닌 경우 즉시 실패 (list, string 등 비허용)
        if not isinstance(r, dict):
            raise ValueError(
                f"리포트 파일이 JSON 객체(dict) 가 아닙니다: {rp} "
                f"(타입: {type(r).__name__}) — 유효한 리포트 JSON 이 아님"
            )
        reports.append(r)
        str_report_paths.append(str(rp))

    # v3 사전등록 여부 감지
    is_v3 = _is_v3_prereg(prereg)

    # 검사 실행
    checks: List[Dict[str, Any]] = []

    # 검사 0: prereg_status (모든 prereg 에 적용)
    c0 = _check_prereg_status(prereg)
    checks.append(c0)

    c1 = _check_prereg_hash_match(prereg, reports, expected_hash)
    checks.append(c1)

    c2 = _check_report_hash_integrity(reports)
    checks.append(c2)

    c3 = _check_replayable(reports)
    checks.append(c3)

    c4 = _check_arms_complete(prereg, reports)
    checks.append(c4)

    # 검사 5: v3 와 v1 분기
    delta_imp_pass = False
    delta_noise_pass = False
    guard_pass: Dict[str, bool] = {exp: False for exp in _V31_GUARDED_EXPERIMENTS}

    if is_v3:
        c5, delta_imp_pass, delta_noise_pass = _check_acceptance_recomputed_v3(
            prereg, reports
        )
    else:
        c5 = _check_acceptance_recomputed(prereg, reports)
    checks.append(c5)

    # v3.1 전용 검사: 역할별 가드 + 알파 M2 금지 + canonical sentences
    if is_v3:
        c5b, guard_pass = _check_role_specific_guard(prereg, reports)
        checks.append(c5b)
        c5c = _check_alpha_m2_prohibition(prereg)
        checks.append(c5c)
        c5d = _check_canonical_sentences(prereg)
        checks.append(c5d)

    c6, consumed_reports, family_history = _check_ledger_registered(
        prereg, reports, ledger, str_report_paths
    )
    checks.append(c6)

    c7 = _check_ancestry(reports, git_runner, release)
    checks.append(c7)

    c8 = _check_pilot_disjoint(prereg, reports)
    checks.append(c8)

    # 검사 9: tieBreak invariance
    if is_v3:
        # v3: AXS-TB-001
        c9, caveat_required, tb_present, tb_pass = _check_tiebreak_invariance(
            reports, "AXS-TB-001"
        )
        c9["check"] = "axs010_invariance"  # 하위 호환: check 이름 유지
    else:
        # v1: AXS-010
        c9, caveat_required, tb_present, tb_pass = _check_axs010_invariance(
            prereg, reports
        )
    checks.append(c9)

    # M0/M1 always True (가설은 항상 허용)
    # M3 always False (MVP; AXS-001 필요)
    all_ok = all(c["ok"] for c in checks)

    # branch_decision + 트랙 라이선스 (v3.1 전용)
    branch: Optional[str] = None
    m2_imp = False
    m2_noise = False
    if is_v3:
        # 트랙 라이선스: 검사 전체 통과 ∧ primary contrast ∧ 역할별 가드 ∧ TB ≥ soft_pass
        # tb_pass: 정직한 invariance fail(부호/trackDecision 불일치)은 검사 위반이
        # 아니라 여기서 트랙 비라이선스로 반영된다 (→ no_claim_m1_only).
        tb_ok = tb_present and c9["ok"] and tb_pass
        m2_imp = bool(
            all_ok and delta_imp_pass and guard_pass.get("AXS-IMP-001", False) and tb_ok
        )
        m2_noise = bool(
            all_ok
            and delta_noise_pass
            and guard_pass.get("AXS-NOISE-001", False)
            and tb_ok
        )
        c10 = _check_branch_decision(all_ok, m2_imp, m2_noise)
        checks.append(c10)
        branch = c10["branch"]
        # all_ok 재확인 (branch_decision 추가 후)
        all_ok = all(c["ok"] for c in checks)

    # M2 계산
    if is_v3:
        # v3.1: 결합 M2 는 두 트랙이 모두 라이선스될 때만 (both_supported)
        m2 = m2_imp and m2_noise
    else:
        # v1: 모든 검사 통과 AND AXS-010 >= soft_pass
        # (axs010_present + tb_pass — 정직한 fail 은 검사 위반 없이 M2 강등)
        m2 = all_ok and tb_present and tb_pass

    rungs: Dict[str, Any] = {
        "M0": True,
        "M1": True,
        "M2": m2,
        "M3": False,  # MVP: AXS-001 필요 — 현재 미구현
    }
    if is_v3:
        rungs["M2-IMP"] = m2_imp
        rungs["M2-NOISE"] = m2_noise

    # licensedSentences (v3.1 전용 — prereg.canonicalSentences 에서만 발급)
    licensed_sentences: List[str] = []
    if is_v3:
        cs = prereg.get("canonicalSentences") or {}
        if m2_imp and isinstance(cs.get("M-IMP"), str):
            licensed_sentences.append(cs["M-IMP"])
        if m2_noise and isinstance(cs.get("M-NOISE"), str):
            licensed_sentences.append(cs["M-NOISE"])
        if m2_imp and m2_noise and isinstance(cs.get("M2-COMBINED"), str):
            licensed_sentences.append(cs["M2-COMBINED"])

    log_ko(
        _logger,
        f"ladder_gate 평가 완료: M2={m2}, caveatRequired={caveat_required}, "
        f"검사수={len(checks)}, 실패={sum(1 for c in checks if not c['ok'])}",
    )

    result: Dict[str, Any] = {
        "rungs": rungs,
        "checks": checks,
        "caveatRequired": caveat_required,
        "consumedReports": consumed_reports,
        "familyHistory": family_history,
        "provisional": not release,
    }
    if is_v3:
        result["branch"] = branch
        result["licensedSentences"] = licensed_sentences

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_AUDIT_NOTE = (
    "AUDIT LOG ONLY — 입력으로 사용 금지 "
    "(ladder_gate는 매 호출 시 증거에서 재계산)"
)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 진입점.

    python -m echo_bench.tools.ladder_gate \\
        --prereg P --reports R1 R2... --ledger L [--release] [--licenses-out PATH]

    exit 0: 실패 검사 없음. exit 1: 하나 이상 실패.
    """
    parser = argparse.ArgumentParser(
        prog="ladder_gate",
        description="메커니즘 라이선스 증거 심사기 (Track M)",
    )
    parser.add_argument("--prereg", required=True, help="사전등록 JSON 경로")
    parser.add_argument(
        "--reports", nargs="+", required=True, help="실험 리포트 JSON 경로 목록"
    )
    parser.add_argument("--ledger", required=True, help="실행 원장 JSON 경로")
    parser.add_argument(
        "--release", action="store_true", default=False, help="원격 ancestry 추가 확인"
    )
    parser.add_argument(
        "--licenses-out",
        default="outputs/reports/licenses.json",
        help="라이선스 감사 로그 출력 경로 (기본: outputs/reports/licenses.json)",
    )
    args = parser.parse_args(argv)

    try:
        result = evaluate_mechanism_license(
            args.prereg,
            args.reports,
            ledger_path=args.ledger,
            release=args.release,
        )
    except (ValueError, OSError) as exc:
        print(f"[오류] ladder_gate 실행 실패: {exc}", file=sys.stderr)
        return 1

    # 라이선스 감사 로그 쓰기 (출력 전용 — 절대 읽지 않음)
    out_path = Path(args.licenses_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output = dict(result)
    audit_output["note"] = _AUDIT_NOTE
    tmp_out = out_path.with_suffix(".json.tmp")
    try:
        with open(tmp_out, "w", encoding="utf-8") as fh:
            json.dump(audit_output, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_out, out_path)
    except Exception:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        raise

    # 한국어 요약 출력
    rungs = result["rungs"]
    checks = result["checks"]
    failed_checks = [c for c in checks if not c["ok"]]

    print("=== ladder_gate 메커니즘 라이선스 심사 결과 ===")
    print(f"런그 상태: M0={rungs['M0']} M1={rungs['M1']} M2={rungs['M2']} M3={rungs['M3']}")
    print(f"caveatRequired: {result['caveatRequired']}")
    print(f"provisional: {result['provisional']}")
    if "branch" in result:
        print(f"branch: {result['branch']}")

    if failed_checks:
        print(f"\n실패 검사 ({len(failed_checks)}건):")
        for c in failed_checks:
            print(f"  [실패] {c['check']}: {c['detail']}")
        print(f"\n라이선스 감사 로그 작성 완료: {out_path}")
        print("ladder_gate 심사 실패: 증거 불충분 또는 무결성 위반")
        return 1

    print(f"\n모든 검사 통과 ({len(checks)}건)")
    print(f"라이선스 감사 로그 작성 완료: {out_path}")
    branch = result.get("branch")
    if branch is not None:
        m2_val = rungs["M2"]
        print(
            f"심사 완료: branch={branch}, "
            f"M2-IMP={rungs.get('M2-IMP')}, M2-NOISE={rungs.get('M2-NOISE')}, "
            f"M2(combined)={m2_val}"
        )
        sentences = result.get("licensedSentences") or []
        for s in sentences:
            print(f"  licensed: {s}")
        if m2_val:
            print("ladder_gate 심사 완료: 증거 충분")
        else:
            print("ladder_gate 심사 완료: M2 미충족 (branch 참조)")
    else:
        print("ladder_gate 심사 완료: 증거 충분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
