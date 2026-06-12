"""tests/test_prereg_v2_draft.py

AXS v2 설계 초안 사전등록 + 파일럿 요약 검증 테스트.

Task: AXS-V2 V1
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG_V1_PATH = REPO / "configs/prereg/axs_mechanism_prereg_v1.json"
PREREG_V2_PATH = REPO / "configs/prereg/axs_mechanism_prereg_v2_draft.json"
SUMMARY_PATH = REPO / "configs/prereg/axs_p0_pilot_summary_v1.json"
PILOT_REPORTS_DIR = REPO / "outputs/reports/pilot_123"

PILOT_REPORTS_EXIST = PILOT_REPORTS_DIR.exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v1() -> dict:
    with open(PREREG_V1_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def v2() -> dict:
    from echo_bench.logging.prereg import load_prereg
    return load_prereg(PREREG_V2_PATH)


@pytest.fixture(scope="module")
def summary() -> dict:
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# load_prereg 통과 + 버전/수정 키 존재
# ---------------------------------------------------------------------------


def test_load_prereg_passes(v2: dict) -> None:
    """load_prereg가 예외 없이 반환되어야 한다."""
    assert isinstance(v2, dict), "load_prereg가 dict를 반환해야 한다"


def test_version_is_2(v2: dict) -> None:
    """version 필드가 정수 2여야 한다."""
    assert v2["version"] == 2, f"version이 2가 아님: {v2['version']!r}"


def test_supersedes_present_and_nonempty(v2: dict) -> None:
    """supersedes 키가 존재하고 비어 있지 않아야 한다."""
    assert "supersedes" in v2, "supersedes 키 누락"
    assert v2["supersedes"], "supersedes가 비어 있음"


def test_changeJustification_present_and_nonempty(v2: dict) -> None:
    """changeJustification 키가 존재하고 비어 있지 않은 배열이어야 한다."""
    assert "changeJustification" in v2, "changeJustification 키 누락"
    cj = v2["changeJustification"]
    assert isinstance(cj, list) and len(cj) > 0, (
        f"changeJustification이 비어 있거나 배열이 아님: {cj!r}"
    )


# ---------------------------------------------------------------------------
# status 정확히 "design-draft"
# ---------------------------------------------------------------------------


def test_status_is_design_draft(v2: dict) -> None:
    """status가 정확히 'design-draft' 문자열이어야 한다."""
    assert v2.get("status") == "design-draft", (
        f"status가 'design-draft'가 아님: {v2.get('status')!r}"
    )


# ---------------------------------------------------------------------------
# preregId 계보 불변
# ---------------------------------------------------------------------------


def test_preregId_lineage(v1: dict, v2: dict) -> None:
    """v2 preregId가 'axs-mechanism'이고 v1과 동일해야 한다."""
    assert v2["preregId"] == "axs-mechanism", (
        f"preregId가 'axs-mechanism'이 아님: {v2['preregId']!r}"
    )
    assert v2["preregId"] == v1["preregId"], (
        f"v2 preregId가 v1과 다름: {v2['preregId']!r} != {v1['preregId']!r}"
    )


# ---------------------------------------------------------------------------
# evaluationFamilies, designContaminatedFamilies, 격리 검증
# ---------------------------------------------------------------------------


def test_evaluationFamilies_identical_to_v1(v1: dict, v2: dict) -> None:
    """evaluationFamilies가 v1과 동일해야 한다."""
    assert sorted(v2["evaluationFamilies"]) == sorted(v1["evaluationFamilies"]), (
        f"evaluationFamilies 불일치: v2={v2['evaluationFamilies']}, "
        f"v1={v1['evaluationFamilies']}"
    )


def test_designContaminatedFamilies_exact(v2: dict) -> None:
    """designContaminatedFamilies가 4개 비평 가족 목록이어야 한다."""
    expected = {"123", "124", "777", "55555"}
    actual = set(v2.get("designContaminatedFamilies", []))
    assert actual == expected, (
        f"designContaminatedFamilies 불일치: {actual} != {expected}"
    )


def test_no_overlap_contaminated_evaluation(v2: dict) -> None:
    """designContaminatedFamilies와 evaluationFamilies는 교집합이 없어야 한다."""
    contaminated = set(v2.get("designContaminatedFamilies", []))
    evaluation = set(v2.get("evaluationFamilies", []))
    overlap = contaminated & evaluation
    assert not overlap, (
        f"designContaminatedFamilies와 evaluationFamilies 사이에 교집합 발생: {overlap}"
    )


def test_999_not_in_contaminated_or_evaluation(v2: dict) -> None:
    """가족 '999'는 designContaminatedFamilies나 evaluationFamilies에 없어야 한다."""
    contaminated = set(v2.get("designContaminatedFamilies", []))
    evaluation = set(v2.get("evaluationFamilies", []))
    assert "999" not in contaminated, "'999'이 designContaminatedFamilies에 포함됨"
    assert "999" not in evaluation, "'999'이 evaluationFamilies에 포함됨"


# ---------------------------------------------------------------------------
# pilotSummaryRef.summaryHash 재계산 검증
# ---------------------------------------------------------------------------


def test_pilotSummaryRef_summaryHash(v2: dict, summary: dict) -> None:
    """pilotSummaryRef.summaryHash가 summary 파일 canonical_hash와 일치해야 한다."""
    from echo_bench.utils.hash import canonical_hash

    ref = v2.get("pilotSummaryRef", {})
    assert "summaryHash" in ref, "pilotSummaryRef에 summaryHash 키 누락"

    recomputed = canonical_hash(summary)
    assert ref["summaryHash"] == recomputed, (
        f"summaryHash 불일치: 저장됨={ref['summaryHash']!r}, "
        f"재계산됨={recomputed!r}"
    )


# ---------------------------------------------------------------------------
# 파일럿 요약 파일 자체 검증
# ---------------------------------------------------------------------------


def test_summary_calibrationAnchor_value(summary: dict) -> None:
    """calibrationAnchor.value가 AXS-003 리포트의 정확한 float이어야 한다."""
    if not PILOT_REPORTS_EXIST:
        pytest.skip("outputs/reports/pilot_123 디렉터리 없음 — CI 안전 스킵")

    anchor = summary.get("calibrationAnchor", {})
    assert "value" in anchor, "calibrationAnchor에 value 키 누락"

    # AXS-003 리포트에서 axs_ucb_default perFamily["123"] slate_excess_nmi 추출
    report_files = list(PILOT_REPORTS_DIR.glob("axs_003_*.json"))
    assert report_files, "AXS-003 리포트 파일을 찾을 수 없음"

    with open(report_files[0], encoding="utf-8") as f:
        report_003 = json.load(f)

    expected = (
        report_003["arms"]["axs_ucb_default"]["perFamily"]["123"]["slate_excess_nmi"]
    )
    assert anchor["value"] == expected, (
        f"calibrationAnchor.value 불일치: 저장됨={anchor['value']!r}, "
        f"AXS-003 리포트={expected!r}"
    )


def test_summary_pilotReportHashes_keys(summary: dict) -> None:
    """pilotReportHashes에 4개 실험 키가 존재해야 한다."""
    hashes = summary.get("pilotReportHashes", {})
    expected_keys = {"AXS-010", "AXS-003", "AXS-009", "AXS-004c"}
    actual_keys = set(hashes.keys())
    assert actual_keys == expected_keys, (
        f"pilotReportHashes 키 불일치: {actual_keys} != {expected_keys}"
    )


def test_summary_reproduction_commands_length(summary: dict) -> None:
    """reproduction.commands에 정확히 4개 명령어가 있어야 한다."""
    repro = summary.get("reproduction", {})
    cmds = repro.get("commands", [])
    assert len(cmds) == 4, (
        f"reproduction.commands 길이가 4가 아님: {len(cmds)}"
    )


# ---------------------------------------------------------------------------
# AXS-ENTRY-001 arm/branch 구조
# ---------------------------------------------------------------------------


def test_axs_entry_001_arms_exact(v2: dict) -> None:
    """AXS-ENTRY-001 arms가 정확히 6개 이름 순서대로여야 한다."""
    exp = v2.get("experiments", {}).get("AXS-ENTRY-001", {})
    expected_arms = [
        "a0_baseline_canonical",
        "a1_baseline_tracefree",
        "a2_ctx0_tracefree",
        "a3_ctx1_tracefree",
        "a4_neverlearn_tracefree",
        "a5_all_entry_cut",
    ]
    assert exp.get("arms") == expected_arms, (
        f"AXS-ENTRY-001 arms 불일치: {exp.get('arms')!r}"
    )


def test_axs_entry_001_branches_exact(v2: dict) -> None:
    """AXS-ENTRY-001 branches가 정확히 6개 branch 이름이어야 한다."""
    exp = v2.get("experiments", {}).get("AXS-ENTRY-001", {})
    branches = exp.get("branches", [])
    expected_names = {
        "strong_e1",
        "weak_joint_utility",
        "e1_e4_joint",
        "e2e3_reemergence",
        "no_claim_m1_only",
        "sanity_fail",
    }
    actual_names = {b["name"] if isinstance(b, dict) else b for b in branches}
    assert actual_names == expected_names, (
        f"AXS-ENTRY-001 branches 불일치: {actual_names} != {expected_names}"
    )


def test_axs_entry_001_branchCountFrozen(v2: dict) -> None:
    """AXS-ENTRY-001 branchCountFrozen이 true여야 한다."""
    exp = v2.get("experiments", {}).get("AXS-ENTRY-001", {})
    assert exp.get("branchCountFrozen") is True, (
        f"branchCountFrozen이 True가 아님: {exp.get('branchCountFrozen')!r}"
    )


# ---------------------------------------------------------------------------
# tieBreakCaveatMarker 계보 불변
# ---------------------------------------------------------------------------


def test_tieBreakCaveatMarker_identical_to_v1(v1: dict, v2: dict) -> None:
    """tieBreakCaveatMarker가 v1과 동일해야 한다."""
    assert v2["tieBreakCaveatMarker"] == v1["tieBreakCaveatMarker"], (
        f"tieBreakCaveatMarker 불일치: "
        f"v2={v2['tieBreakCaveatMarker']!r}, "
        f"v1={v1['tieBreakCaveatMarker']!r}"
    )


# ---------------------------------------------------------------------------
# 소스 리포트 대비 요약 수치 회귀 테스트 (pilot_123 존재 시만 실행)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pilot_reports() -> dict[str, dict]:
    """pilot_123 디렉터리의 4개 리포트를 키(실험 ID)별로 반환한다."""
    if not PILOT_REPORTS_EXIST:
        return {}
    reports: dict[str, dict] = {}
    for exp_id, glob_prefix in [
        ("AXS-003", "axs_003_"),
        ("AXS-004c", "axs_004c_"),
        ("AXS-009", "axs_009_"),
        ("AXS-010", "axs_010_"),
    ]:
        matches = list(PILOT_REPORTS_DIR.glob(f"{glob_prefix}*.json"))
        assert matches, f"{exp_id} 리포트 파일을 찾을 수 없음"
        with open(matches[0], encoding="utf-8") as f:
            reports[exp_id] = json.load(f)
    return reports


def test_pilotReportHashes_match_source(summary: dict, pilot_reports: dict) -> None:
    """pilotReportHashes의 4개 해시가 소스 리포트의 reportHash와 일치해야 한다."""
    if not PILOT_REPORTS_EXIST:
        pytest.skip("outputs/reports/pilot_123 디렉터리 없음 — CI 안전 스킵")

    stored_hashes = summary.get("pilotReportHashes", {})
    for exp_id, report in pilot_reports.items():
        source_hash = report.get("reportHash")
        assert source_hash is not None, f"{exp_id} 리포트에 reportHash 키 없음"
        assert stored_hashes.get(exp_id) == source_hash, (
            f"pilotReportHashes[{exp_id!r}] 불일치: "
            f"저장됨={stored_hashes.get(exp_id)!r}, "
            f"소스 리포트={source_hash!r}"
        )


def test_runCommit_matches_reports(summary: dict, pilot_reports: dict) -> None:
    """summary.runCommit이 각 소스 리포트의 preregStamp.runCommit과 일치해야 한다."""
    if not PILOT_REPORTS_EXIST:
        pytest.skip("outputs/reports/pilot_123 디렉터리 없음 — CI 안전 스킵")

    summary_commit = summary.get("runCommit")
    for exp_id, report in pilot_reports.items():
        report_commit = report.get("preregStamp", {}).get("runCommit")
        assert report_commit is not None, f"{exp_id} 리포트에 preregStamp.runCommit 없음"
        assert summary_commit == report_commit, (
            f"runCommit 불일치 ({exp_id}): "
            f"summary={summary_commit!r}, 리포트={report_commit!r}"
        )


def test_scheduleEmbeddedHash_matches_artifact(summary: dict) -> None:
    """scheduleEmbeddedHash가 axs_004c_yoked_schedule_v1.json 아티팩트의 scheduleHash와 일치해야 한다."""
    if not PILOT_REPORTS_EXIST:
        pytest.skip("outputs/reports/pilot_123 디렉터리 없음 — CI 안전 스킵")

    schedule_path = REPO / "configs/prereg/axs_004c_yoked_schedule_v1.json"
    assert schedule_path.exists(), f"스케줄 아티팩트 파일 없음: {schedule_path}"
    with open(schedule_path, encoding="utf-8") as f:
        schedule = json.load(f)

    artifact_embedded = schedule.get("scheduleHash")
    assert artifact_embedded is not None, "스케줄 아티팩트에 scheduleHash 키 없음"
    summary_embedded = summary.get("scheduleEmbeddedHash")
    assert summary.get("scheduleEmbeddedHash") is not None, "summary에 scheduleEmbeddedHash 키 없음"
    assert summary_embedded == artifact_embedded, (
        f"scheduleEmbeddedHash 불일치: "
        f"summary={summary_embedded!r}, 아티팩트={artifact_embedded!r}"
    )


def test_axs_004c_yoked_mean_matches_source(summary: dict, pilot_reports: dict) -> None:
    """summary AXS-004c axs_yoked_bonus mean이 소스 리포트의 값과 일치해야 한다."""
    if not PILOT_REPORTS_EXIST:
        pytest.skip("outputs/reports/pilot_123 디렉터리 없음 — CI 안전 스킵")

    # 소스 리포트에서 axs_yoked_bonus perFamily["123"] slate_excess_nmi 추출
    report_004c = pilot_reports["AXS-004c"]
    source_value = (
        report_004c["arms"]["axs_yoked_bonus"]["perFamily"]["123"]["slate_excess_nmi"]
    )
    summary_value = (
        summary.get("headlineNumbers", {})
        .get("AXS-004c", {})
        .get("axs_yoked_bonus_bootstrap_slate_excess_nmi_mean")
    )
    assert summary_value is not None, "summary AXS-004c axs_yoked_bonus_bootstrap_slate_excess_nmi_mean 키 없음"
    assert summary_value == source_value, (
        f"AXS-004c yoked mean 불일치: "
        f"summary={summary_value!r}, 소스 리포트={source_value!r}"
    )


def test_axs_009_freeze_divergence_headline_matches_source(summary: dict, pilot_reports: dict) -> None:
    """summary AXS-009 freeze_at_half_post_freeze_incremental_divergence_mean이 소스 리포트 값과 일치해야 한다."""
    if not PILOT_REPORTS_EXIST:
        pytest.skip("outputs/reports/pilot_123 디렉터리 없음 — CI 안전 스킵")

    report_009 = pilot_reports["AXS-009"]
    source_value = (
        report_009["arms"]["freeze_at_half"]["perFamily"]["123"][
            "post_freeze_incremental_divergence"
        ]
    )
    summary_value = (
        summary.get("headlineNumbers", {})
        .get("AXS-009", {})
        .get("freeze_at_half_post_freeze_incremental_divergence_mean")
    )
    assert summary_value is not None, (
        "summary AXS-009 freeze_at_half_post_freeze_incremental_divergence_mean 키 없음"
    )
    assert summary_value == source_value, (
        f"AXS-009 freeze_at_half divergence 불일치: "
        f"summary={summary_value!r}, 소스 리포트={source_value!r}"
    )
