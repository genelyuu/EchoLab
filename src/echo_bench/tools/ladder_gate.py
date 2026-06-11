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
from echo_bench.utils.hash import canonical_hash

__all__ = ["evaluate_mechanism_license", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _check_arms_complete(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """검사 4: arms_complete.

    각 실험 리포트의 arm 집합이 prereg 에 등록된 것과 일치하는지 확인.
    RANDOM 기준 미달 arm 은 degenerate 3개 필드 필수.
    AXS-010 은 tieBreak 구조 검사.
    """
    ok = True
    details = []
    degenerate_fields = {"degenerate", "degenerateReason", "includedInMechanismClaim"}

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
    known_exp_ids = set(prereg.get("experiments", {}).keys()) | {"AXS-010"}
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

        if exp_id == "AXS-010":
            # AXS-010: tieBreak 구조 검사
            tie_break = r.get("tieBreak")
            if not isinstance(tie_break, dict):
                ok = False
                details.append("AXS-010: tieBreak 필드 누락 또는 잘못된 형식")
                continue
            if "baseline" not in tie_break:
                ok = False
                details.append("AXS-010: tieBreak.baseline 누락")
            required_variants = {"reverse", "hash_seeded", "feature_lexicographic"}
            variants = tie_break.get("variants") or {}
            missing_variants = required_variants - set(variants.keys())
            if missing_variants:
                ok = False
                details.append(
                    f"AXS-010: tieBreak.variants 누락: {sorted(missing_variants)}"
                )
            continue

        # 일반 실험: arms 검사
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

        # ITEM 3: arm coverage 도 strict_number 로 검사
        for arm_id, arm_data in (r.get("arms") or {}).items():
            if arm_id not in prereg_arms:
                continue
            arm_cov = (arm_data.get("utility") or {}).get("coordinate_coverage_mean")
            if not _strict_number(arm_cov):
                ok = False
                details.append(
                    f"{exp_id} arm {arm_id!r}: utility.coordinate_coverage_mean 누락 또는 비숫자 "
                    f"(값: {arm_cov!r})"
                )
                continue
            if arm_cov < random_cov:
                # RANDOM 미달 → degenerate 3개 필드 필수, 값도 정확해야 함
                for field in degenerate_fields:
                    if field not in arm_data:
                        ok = False
                        details.append(
                            f"{exp_id} arm {arm_id!r}: RANDOM 기준 미달이나 "
                            f"degenerate 필드 {field!r} 누락"
                        )
                # 값 검증: degenerate=True, degenerateReason 비어있지 않은 문자열,
                # includedInMechanismClaim=False 이어야 함
                if arm_data.get("degenerate") is not True:
                    ok = False
                    details.append(
                        f"{exp_id} arm {arm_id!r}: RANDOM 기준 미달이나 "
                        f"degenerate={arm_data.get('degenerate')!r} (True 여야 함)"
                    )
                reason = arm_data.get("degenerateReason")
                if not isinstance(reason, str) or not reason.strip():
                    ok = False
                    details.append(
                        f"{exp_id} arm {arm_id!r}: degenerateReason 이 비어 있거나 "
                        f"문자열이 아님 ({reason!r})"
                    )
                if arm_data.get("includedInMechanismClaim") is not False:
                    ok = False
                    details.append(
                        f"{exp_id} arm {arm_id!r}: RANDOM 기준 미달이나 "
                        f"includedInMechanismClaim="
                        f"{arm_data.get('includedInMechanismClaim')!r} (False 여야 함)"
                    )

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


def _check_axs010_invariance(
    prereg: Dict[str, Any],
    reports: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], bool, bool]:
    """검사 9: axs010_invariance.

    Returns: (check_result, caveat_required, axs010_present)
    strict_pass → caveat_required=False
    soft_pass  → caveat_required=True
    fail       → ok=False
    absent     → ok=True (M2=False 이지만 검사 자체는 fail 아님), caveat_required=False
    """
    r10 = next(
        (r for r in reports if r.get("experimentId") == "AXS-010"), None
    )
    if r10 is None:
        return (
            {
                "check": "axs010_invariance",
                "ok": True,
                "detail": "AXS-010 invariance 증거 없음",
            },
            False,
            False,
        )

    tie_break = r10.get("tieBreak") or {}
    baseline = tie_break.get("baseline") or {}
    variants = tie_break.get("variants") or {}

    b_sign = baseline.get("sign")
    b_ci_lower = baseline.get("ciLower")
    b_ci_upper = baseline.get("ciUpper")
    b_track = baseline.get("trackDecision")

    # ITEM 1: fail closed — baseline required fields must be present and non-None
    if b_sign is None or b_track is None or b_ci_lower is None or b_ci_upper is None:
        return (
            {
                "check": "axs010_invariance",
                "ok": False,
                "detail": (
                    "AXS-010: baseline 필수 필드 누락 "
                    f"(sign={b_sign!r}, trackDecision={b_track!r}, "
                    f"ciLower={b_ci_lower!r}, ciUpper={b_ci_upper!r}) — fail closed"
                ),
            },
            False,
            True,
        )

    all_signs_match = True
    all_tracks_match = True
    all_within_ci = True

    for var_id, var_data in variants.items():
        v_sign = var_data.get("sign")
        v_track = var_data.get("trackDecision")
        v_est = var_data.get("estimate")

        # ITEM 1: variant required fields must be present and non-None
        if v_sign is None or v_track is None:
            return (
                {
                    "check": "axs010_invariance",
                    "ok": False,
                    "detail": (
                        f"AXS-010: 변형 {var_id!r} 필수 필드 누락 "
                        f"(sign={v_sign!r}, trackDecision={v_track!r}) — fail closed"
                    ),
                },
                False,
                True,
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
        # fail
        return (
            {
                "check": "axs010_invariance",
                "ok": False,
                "detail": "AXS-010: 변형 부호 또는 trackDecision 불일치 (fail)",
            },
            False,
            True,
        )
    if all_within_ci:
        # strict_pass
        return (
            {
                "check": "axs010_invariance",
                "ok": True,
                "detail": "AXS-010 strict_pass: 모든 변형 부호·추정치 CI 내 일치",
            },
            False,
            True,
        )
    else:
        # soft_pass
        return (
            {
                "check": "axs010_invariance",
                "ok": True,
                "detail": "AXS-010 soft_pass: 부호·trackDecision 일치, CI 외 추정치 존재 → caveatRequired",
            },
            True,
            True,
        )


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
        rungs, checks, caveatRequired, consumedReports, familyHistory, provisional 포함 dict.

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

    # 검사 실행
    checks: List[Dict[str, Any]] = []

    c1 = _check_prereg_hash_match(prereg, reports, expected_hash)
    checks.append(c1)

    c2 = _check_report_hash_integrity(reports)
    checks.append(c2)

    c3 = _check_replayable(reports)
    checks.append(c3)

    c4 = _check_arms_complete(prereg, reports)
    checks.append(c4)

    c5 = _check_acceptance_recomputed(prereg, reports)
    checks.append(c5)

    c6, consumed_reports, family_history = _check_ledger_registered(
        prereg, reports, ledger, str_report_paths
    )
    checks.append(c6)

    c7 = _check_ancestry(reports, git_runner, release)
    checks.append(c7)

    c8 = _check_pilot_disjoint(prereg, reports)
    checks.append(c8)

    c9, caveat_required, axs010_present = _check_axs010_invariance(prereg, reports)
    checks.append(c9)

    # M0/M1 always True (가설은 항상 허용)
    # M3 always False (MVP; AXS-001 필요)
    all_ok = all(c["ok"] for c in checks)

    # M2: 모든 검사 통과 AND AXS-010 >= soft_pass (axs010_present + c9["ok"])
    m2 = all_ok and axs010_present
    # M2 also requires caveat is not a blocker (soft_pass allows M2 with caveat)

    rungs = {
        "M0": True,
        "M1": True,
        "M2": m2,
        "M3": False,  # MVP: AXS-001 필요 — 현재 미구현
    }

    log_ko(
        _logger,
        f"ladder_gate 평가 완료: M2={m2}, caveatRequired={caveat_required}, "
        f"검사수={len(checks)}, 실패={sum(1 for c in checks if not c['ok'])}",
    )

    return {
        "rungs": rungs,
        "checks": checks,
        "caveatRequired": caveat_required,
        "consumedReports": consumed_reports,
        "familyHistory": family_history,
        "provisional": not release,
    }


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

    if failed_checks:
        print(f"\n실패 검사 ({len(failed_checks)}건):")
        for c in failed_checks:
            print(f"  [실패] {c['check']}: {c['detail']}")
        print(f"\n라이선스 감사 로그 작성 완료: {out_path}")
        print("ladder_gate 심사 실패: 증거 불충분 또는 무결성 위반")
        return 1

    print(f"\n모든 검사 통과 ({len(checks)}건)")
    print(f"라이선스 감사 로그 작성 완료: {out_path}")
    print("ladder_gate 심사 완료: 증거 충분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
