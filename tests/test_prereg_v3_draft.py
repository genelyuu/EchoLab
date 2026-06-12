"""tests/test_prereg_v3_draft.py

AXS v3 설계 초안 사전등록 + 파일럿 요약 v2 검증 테스트 (pilot_v3 4-family 완전 재구축).

Task: AXS-V3 N1
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG_V1_PATH = REPO / "configs/prereg/axs_mechanism_prereg_v1.json"
PREREG_V3_PATH = REPO / "configs/prereg/axs_mechanism_prereg_v3_draft.json"
SUMMARY_V2_PATH = REPO / "configs/prereg/axs_pilot_summary_v2.json"

PILOT_V3_DIR = REPO / "outputs/reports/pilot_v3"

# 이전 단일-family 충돌 디렉터리 — 감사용으로 보존
PILOT_123_DIR = REPO / "outputs/reports/pilot_123_postfix"
PILOT_CRITIC_DIR = REPO / "outputs/reports/pilot_critic_postfix"

PILOT_V3_EXISTS = PILOT_V3_DIR.exists()

# pilot_v3 내 모든 4개 family 디렉터리 확인 — 16개 파일이 있어야 함
PILOT_V3_COMPLETE = (
    PILOT_V3_EXISTS
    and len(list(PILOT_V3_DIR.glob("axs_003_*.json"))) >= 4
    and len(list(PILOT_V3_DIR.glob("axs_009_*.json"))) >= 4
    and len(list(PILOT_V3_DIR.glob("axs_004c_*.json"))) >= 4
    and len(list(PILOT_V3_DIR.glob("axs_010_*.json"))) >= 4
)

# pilot_v3 4-family 리포트 경로 매핑 (파일명 = 실험 prefix + seedBatchId prefix)
PILOT_V3_REPORT_MAP: dict[str, dict[str, str]] = {
    "123": {
        "AXS-003": "axs_003_4145b41c3d60.json",
        "AXS-009": "axs_009_bd3b7be2635d.json",
        "AXS-004c": "axs_004c_c1a36fd19f76.json",
        "AXS-010": "axs_010_9cf850cd0b9d.json",
    },
    "124": {
        "AXS-003": "axs_003_8394f4e0a426.json",
        "AXS-009": "axs_009_018a86e7a721.json",
        "AXS-004c": "axs_004c_2416583d9373.json",
        "AXS-010": "axs_010_8534040653d9.json",
    },
    "777": {
        "AXS-003": "axs_003_090400a72f75.json",
        "AXS-009": "axs_009_1920dac4060d.json",
        "AXS-004c": "axs_004c_9cd233837029.json",
        "AXS-010": "axs_010_972fd96d9fa3.json",
    },
    "55555": {
        "AXS-003": "axs_003_563b27979dc3.json",
        "AXS-009": "axs_009_0390d8a988bd.json",
        "AXS-004c": "axs_004c_ceffbdd4bbae.json",
        "AXS-010": "axs_010_19fd30af5ae8.json",
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v1() -> dict:
    with open(PREREG_V1_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def v3() -> dict:
    from echo_bench.logging.prereg import load_prereg
    return load_prereg(PREREG_V3_PATH)


@pytest.fixture(scope="module")
def summary_v2() -> dict:
    with open(SUMMARY_V2_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pilot_v3_reports() -> dict[str, dict[str, dict]]:
    """pilot_v3 디렉터리의 모든 16개 리포트를 {family -> {exp_id -> report}} 형태로 반환한다."""
    if not PILOT_V3_COMPLETE:
        return {}
    result: dict[str, dict[str, dict]] = {}
    for family, exp_map in PILOT_V3_REPORT_MAP.items():
        result[family] = {}
        for exp_id, filename in exp_map.items():
            path = PILOT_V3_DIR / filename
            with open(path, encoding="utf-8") as f:
                result[family][exp_id] = json.load(f)
    return result


# ---------------------------------------------------------------------------
# load_prereg 통과 + 버전/수정 키 존재
# ---------------------------------------------------------------------------


def test_load_prereg_passes(v3: dict) -> None:
    """load_prereg가 예외 없이 반환되어야 한다."""
    assert isinstance(v3, dict), "load_prereg가 dict를 반환해야 한다"


def test_version_is_3(v3: dict) -> None:
    """version 필드가 정수 3이어야 한다."""
    assert v3["version"] == 3, f"version이 3이 아님: {v3['version']!r}"


def test_supersedes_present_and_nonempty(v3: dict) -> None:
    """supersedes 키가 존재하고 비어 있지 않아야 한다."""
    assert "supersedes" in v3, "supersedes 키 누락"
    assert v3["supersedes"], "supersedes가 비어 있음"


def test_changeJustification_present_and_nonempty(v3: dict) -> None:
    """changeJustification 키가 존재하고 비어 있지 않은 배열이어야 한다."""
    assert "changeJustification" in v3, "changeJustification 키 누락"
    cj = v3["changeJustification"]
    assert isinstance(cj, list) and len(cj) > 0, (
        f"changeJustification이 비어 있거나 배열이 아님: {cj!r}"
    )


def test_changeJustification_mentions_filename_collision_bug(v3: dict) -> None:
    """changeJustification에 seedBatchId/filename-collision 두 번째 버그 기록이 있어야 한다."""
    cj_text = " ".join(str(e) for e in v3.get("changeJustification", []))
    assert "89a8f4832cbca7be80acf0a1f1cb52b4a0855e33" in cj_text, (
        "changeJustification에 seedBatchId 버그 fixCommit(89a8f48 full SHA) 언급 없음"
    )
    assert "pilot_v3" in cj_text, (
        "changeJustification에 pilot_v3 언급 없음"
    )


# ---------------------------------------------------------------------------
# status 정확히 "design-draft"
# ---------------------------------------------------------------------------


def test_status_is_design_draft(v3: dict) -> None:
    """status가 정확히 'design-draft' 문자열이어야 한다."""
    assert v3.get("status") == "design-draft", (
        f"status가 'design-draft'가 아님: {v3.get('status')!r}"
    )


# ---------------------------------------------------------------------------
# preregId 계보 불변
# ---------------------------------------------------------------------------


def test_preregId_lineage(v1: dict, v3: dict) -> None:
    """v3 preregId가 'axs-mechanism'이고 v1과 동일해야 한다."""
    assert v3["preregId"] == "axs-mechanism", (
        f"preregId가 'axs-mechanism'이 아님: {v3['preregId']!r}"
    )
    assert v3["preregId"] == v1["preregId"], (
        f"v3 preregId가 v1과 다름: {v3['preregId']!r} != {v1['preregId']!r}"
    )


# ---------------------------------------------------------------------------
# evaluationFamilies, designContaminatedFamilies, 격리 검증
# ---------------------------------------------------------------------------


def test_evaluationFamilies_identical_to_v1(v1: dict, v3: dict) -> None:
    """evaluationFamilies가 v1과 동일해야 한다."""
    assert sorted(v3["evaluationFamilies"]) == sorted(v1["evaluationFamilies"]), (
        f"evaluationFamilies 불일치: v3={v3['evaluationFamilies']}, "
        f"v1={v1['evaluationFamilies']}"
    )


def test_designContaminatedFamilies_exact(v3: dict) -> None:
    """designContaminatedFamilies가 4개 비평 가족 목록이어야 한다."""
    expected = {"123", "124", "777", "55555"}
    actual = set(v3.get("designContaminatedFamilies", []))
    assert actual == expected, (
        f"designContaminatedFamilies 불일치: {actual} != {expected}"
    )


def test_no_overlap_contaminated_evaluation(v3: dict) -> None:
    """designContaminatedFamilies와 evaluationFamilies는 교집합이 없어야 한다."""
    contaminated = set(v3.get("designContaminatedFamilies", []))
    evaluation = set(v3.get("evaluationFamilies", []))
    overlap = contaminated & evaluation
    assert not overlap, (
        f"designContaminatedFamilies와 evaluationFamilies 사이에 교집합 발생: {overlap}"
    )


def test_999_not_in_contaminated_or_evaluation(v3: dict) -> None:
    """가족 '999'는 designContaminatedFamilies나 evaluationFamilies에 없어야 한다."""
    contaminated = set(v3.get("designContaminatedFamilies", []))
    evaluation = set(v3.get("evaluationFamilies", []))
    assert "999" not in contaminated, "'999'이 designContaminatedFamilies에 포함됨"
    assert "999" not in evaluation, "'999'이 evaluationFamilies에 포함됨"


# ---------------------------------------------------------------------------
# pilotSummaryRef.summaryHash 재계산 검증
# ---------------------------------------------------------------------------


def test_pilotSummaryRef_summaryHash(v3: dict, summary_v2: dict) -> None:
    """pilotSummaryRef.summaryHash가 summary v2 파일 canonical_hash와 일치해야 한다."""
    from echo_bench.utils.hash import canonical_hash

    ref = v3.get("pilotSummaryRef", {})
    assert "summaryHash" in ref, "pilotSummaryRef에 summaryHash 키 누락"

    recomputed = canonical_hash(summary_v2)
    assert ref["summaryHash"] == recomputed, (
        f"summaryHash 불일치: 저장됨={ref['summaryHash']!r}, "
        f"재계산됨={recomputed!r}"
    )


# ---------------------------------------------------------------------------
# Summary v2 자체 검증
# ---------------------------------------------------------------------------


def test_summary_v2_summaryId(summary_v2: dict) -> None:
    """summaryId가 'axs-pilot-summary-v2'여야 한다."""
    assert summary_v2.get("summaryId") == "axs-pilot-summary-v2", (
        f"summaryId 불일치: {summary_v2.get('summaryId')!r}"
    )


def test_summary_v2_supersedesSummary_present(summary_v2: dict) -> None:
    """supersedesSummary 키가 존재하고 올바른 summaryId를 가져야 한다."""
    sup = summary_v2.get("supersedesSummary", {})
    assert sup.get("summaryId") == "axs-p0-pilot-summary-v1", (
        f"supersedesSummary.summaryId 불일치: {sup.get('summaryId')!r}"
    )


def test_summary_v2_bugRecord_is_list_with_two_entries(summary_v2: dict) -> None:
    """bugRecord가 2개 항목을 가진 배열이어야 한다."""
    bug_record = summary_v2.get("bugRecord")
    assert isinstance(bug_record, list), (
        f"bugRecord가 배열이 아님: {type(bug_record)!r}"
    )
    assert len(bug_record) == 2, (
        f"bugRecord 항목 수가 2가 아님: {len(bug_record)}"
    )


def test_summary_v2_bugRecord_first_entry_fixCommit(summary_v2: dict) -> None:
    """bugRecord[0] (config-drop 버그)의 fixCommit이 b7b64b4로 시작해야 한다."""
    bug_record = summary_v2.get("bugRecord", [])
    assert len(bug_record) >= 1, "bugRecord가 비어 있음"
    first = bug_record[0]
    fix_commit = first.get("fixCommit")
    assert fix_commit is not None, "bugRecord[0].fixCommit 키 누락"
    assert fix_commit.startswith("b7b64b4"), (
        f"bugRecord[0].fixCommit이 b7b64b4로 시작하지 않음: {fix_commit!r}"
    )
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", fix_commit],
        capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"bugRecord[0].fixCommit을 git으로 해석할 수 없음: {fix_commit!r}"
    )


def test_summary_v2_bugRecord_second_entry_fixCommit(summary_v2: dict) -> None:
    """bugRecord[1] (seedBatchId 충돌 버그)의 fixCommit이 89a8f48으로 시작해야 한다."""
    bug_record = summary_v2.get("bugRecord", [])
    assert len(bug_record) >= 2, "bugRecord에 두 번째 항목 없음"
    second = bug_record[1]
    fix_commit = second.get("fixCommit")
    assert fix_commit is not None, "bugRecord[1].fixCommit 키 누락"
    assert fix_commit.startswith("89a8f48"), (
        f"bugRecord[1].fixCommit이 89a8f48으로 시작하지 않음: {fix_commit!r}"
    )
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", fix_commit],
        capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"bugRecord[1].fixCommit을 git으로 해석할 수 없음: {fix_commit!r}"
    )


def test_summary_v2_postFixPilot_has_all_4_families(summary_v2: dict) -> None:
    """postFixPilot에 4개 family 키가 모두 있어야 한다."""
    post_fix = summary_v2.get("postFixPilot", {})
    expected_families = {"123", "124", "777", "55555"}
    actual_families = set(post_fix.keys())
    assert actual_families == expected_families, (
        f"postFixPilot family 키 불일치: {actual_families} != {expected_families}"
    )


def test_summary_v2_all_16_report_hashes_match_source(
    summary_v2: dict,
    pilot_v3_reports: dict,
) -> None:
    """summary v2의 postFixPilot 모든 16개 리포트 해시가 소스 리포트의 reportHash와 일치해야 한다."""
    if not PILOT_V3_COMPLETE:
        pytest.skip("pilot_v3 디렉터리 없음 또는 완전하지 않음 — CI 안전 스킵")

    post_fix = summary_v2.get("postFixPilot", {})

    for family in ["123", "124", "777", "55555"]:
        family_data = post_fix.get(family, {})
        for exp_id in ["AXS-003", "AXS-009", "AXS-004c", "AXS-010"]:
            source_report = pilot_v3_reports.get(family, {}).get(exp_id, {})
            source_hash = source_report.get("reportHash")
            stored_hash = family_data.get(exp_id, {}).get("reportHash")
            assert stored_hash is not None, (
                f"postFixPilot[{family}][{exp_id}].reportHash 키 누락"
            )
            assert stored_hash == source_hash, (
                f"postFixPilot[{family}][{exp_id}].reportHash 불일치: "
                f"저장됨={stored_hash!r}, 소스={source_hash!r}"
            )


def _extract_contrasts_from_reports(
    r003: dict, r009: dict, r004c: dict, family: str
) -> dict[str, float]:
    """리포트에서 derived contrast를 계산하여 반환한다."""
    freeze_at_1 = r009["arms"]["freeze_at_1"]["perFamily"][family]["slate_excess_nmi"]
    freeze_none = r009["arms"]["freeze_none"]["perFamily"][family]["slate_excess_nmi"]
    freeze_at_quarter = r009["arms"]["freeze_at_quarter"]["perFamily"][family]["slate_excess_nmi"]
    freeze_at_half = r009["arms"]["freeze_at_half"]["perFamily"][family]["slate_excess_nmi"]
    default_004c = r004c["arms"]["axs_ucb_default"]["perFamily"][family]["slate_excess_nmi"]
    yoked_004c = r004c["arms"]["axs_yoked_bonus"]["perFamily"][family]["slate_excess_nmi"]
    default_003 = r003["arms"]["axs_ucb_default"]["perFamily"][family]["slate_excess_nmi"]
    alpha0_003 = r003["arms"]["axs_ucb_alpha0"]["perFamily"][family]["slate_excess_nmi"]

    return {
        "delta_imp": freeze_at_1 - freeze_none,
        "delta_q": freeze_at_quarter - freeze_none,
        "delta_h": freeze_at_half - freeze_none,
        "delta_noise": default_004c - yoked_004c,
        "delta_alpha": alpha0_003 - default_003,
    }


def test_summary_v2_all_4_families_derived_contrasts_match_source(
    summary_v2: dict,
    pilot_v3_reports: dict,
) -> None:
    """summary v2의 4개 family 전체 derived contrast 값이 소스 리포트 재계산과 일치해야 한다."""
    if not PILOT_V3_COMPLETE:
        pytest.skip("pilot_v3 디렉터리 없음 또는 완전하지 않음 — CI 안전 스킵")

    post_fix = summary_v2.get("postFixPilot", {})

    for family in ["123", "124", "777", "55555"]:
        computed = _extract_contrasts_from_reports(
            pilot_v3_reports[family]["AXS-003"],
            pilot_v3_reports[family]["AXS-009"],
            pilot_v3_reports[family]["AXS-004c"],
            family,
        )
        stored = post_fix.get(family, {})
        for key, expected in computed.items():
            actual = stored.get(key)
            assert actual == expected, (
                f"postFixPilot[{family}][{key}] 불일치: 저장됨={actual!r}, 계산됨={expected!r}"
            )


def test_summary_v2_runCommit_all_16_reports_consistent(
    summary_v2: dict,
    pilot_v3_reports: dict,
) -> None:
    """bugRecord[1].fixCommit이 pilot_v3의 모든 16개 리포트 preregStamp.runCommit과 일치해야 한다."""
    if not PILOT_V3_COMPLETE:
        pytest.skip("pilot_v3 디렉터리 없음 또는 완전하지 않음 — CI 안전 스킵")

    bug_record = summary_v2.get("bugRecord", [])
    assert len(bug_record) >= 2, "bugRecord에 두 번째 항목 없음"
    fix_commit = bug_record[1].get("fixCommit")
    assert fix_commit is not None, "bugRecord[1].fixCommit 없음"

    for family in ["123", "124", "777", "55555"]:
        for exp_id in ["AXS-003", "AXS-009", "AXS-004c", "AXS-010"]:
            report = pilot_v3_reports.get(family, {}).get(exp_id, {})
            run_commit = report.get("preregStamp", {}).get("runCommit")
            assert run_commit == fix_commit, (
                f"pilot_v3 family={family} {exp_id}: "
                f"runCommit={run_commit!r} != fixCommit={fix_commit!r}"
            )


def test_summary_v2_stabilityAssessment_strings_present(summary_v2: dict) -> None:
    """stabilityAssessment 키들이 존재하고 비어 있지 않아야 한다."""
    sa = summary_v2.get("stabilityAssessment", {})
    required = [
        "delta_imp", "delta_q", "delta_h", "strictMonotoneChain",
        "delta_noise", "tieBreakInvariance", "delta_alpha",
    ]
    for key in required:
        assert key in sa, f"stabilityAssessment.{key} 키 누락"
        assert sa[key], f"stabilityAssessment.{key}가 비어 있음"


def test_summary_v2_stabilityAssessment_sign_counts_match_data(
    summary_v2: dict,
    pilot_v3_reports: dict,
) -> None:
    """stabilityAssessment의 '4/4'/'3/4' 표현이 실제 데이터 사인 카운트와 일치해야 한다."""
    if not PILOT_V3_COMPLETE:
        pytest.skip("pilot_v3 디렉터리 없음 또는 완전하지 않음 — CI 안전 스킵")

    families = ["123", "124", "777", "55555"]
    contrasts_per_family = {
        fam: _extract_contrasts_from_reports(
            pilot_v3_reports[fam]["AXS-003"],
            pilot_v3_reports[fam]["AXS-009"],
            pilot_v3_reports[fam]["AXS-004c"],
            fam,
        )
        for fam in families
    }

    sa = summary_v2.get("stabilityAssessment", {})

    # delta_imp: expect 4/4 positive
    delta_imp_pos = sum(1 for f in families if contrasts_per_family[f]["delta_imp"] > 0)
    assert delta_imp_pos == 4, f"delta_imp positive count={delta_imp_pos}, expected 4"
    assert "4/4" in sa["delta_imp"], (
        f"stabilityAssessment.delta_imp에 '4/4' 없음: {sa['delta_imp']!r}"
    )

    # delta_q: expect 4/4 positive
    delta_q_pos = sum(1 for f in families if contrasts_per_family[f]["delta_q"] > 0)
    assert delta_q_pos == 4, f"delta_q positive count={delta_q_pos}, expected 4"
    assert "4/4" in sa["delta_q"], (
        f"stabilityAssessment.delta_q에 '4/4' 없음: {sa['delta_q']!r}"
    )

    # delta_h: expect 4/4 positive
    delta_h_pos = sum(1 for f in families if contrasts_per_family[f]["delta_h"] > 0)
    assert delta_h_pos == 4, f"delta_h positive count={delta_h_pos}, expected 4"
    assert "4/4" in sa["delta_h"], (
        f"stabilityAssessment.delta_h에 '4/4' 없음: {sa['delta_h']!r}"
    )

    # delta_noise: expect 4/4 positive
    delta_noise_pos = sum(1 for f in families if contrasts_per_family[f]["delta_noise"] > 0)
    assert delta_noise_pos == 4, f"delta_noise positive count={delta_noise_pos}, expected 4"
    assert "4/4" in sa["delta_noise"], (
        f"stabilityAssessment.delta_noise에 '4/4' 없음: {sa['delta_noise']!r}"
    )

    # delta_alpha: expect 3/4 positive (124 reversed)
    delta_alpha_pos = sum(1 for f in families if contrasts_per_family[f]["delta_alpha"] > 0)
    assert delta_alpha_pos == 3, f"delta_alpha positive count={delta_alpha_pos}, expected 3"
    assert "3/4" in sa["delta_alpha"], (
        f"stabilityAssessment.delta_alpha에 '3/4' 없음: {sa['delta_alpha']!r}"
    )
    assert "124" in sa["delta_alpha"], (
        f"stabilityAssessment.delta_alpha에 반전 family '124' 언급 없음: {sa['delta_alpha']!r}"
    )

    # strict monotone check: 124 should violate
    monotone_violations = []
    for fam in families:
        fam_reports = pilot_v3_reports[fam]
        d009 = fam_reports["AXS-009"]
        f1 = d009["arms"]["freeze_at_1"]["perFamily"][fam]["slate_excess_nmi"]
        fq = d009["arms"]["freeze_at_quarter"]["perFamily"][fam]["slate_excess_nmi"]
        fh = d009["arms"]["freeze_at_half"]["perFamily"][fam]["slate_excess_nmi"]
        fn = d009["arms"]["freeze_none"]["perFamily"][fam]["slate_excess_nmi"]
        if not (f1 > fq > fh > fn):
            monotone_violations.append(fam)
    assert "124" in monotone_violations, (
        f"family 124가 strict monotone 위반 목록에 없음: {monotone_violations}"
    )
    assert "124" in sa["strictMonotoneChain"], (
        f"stabilityAssessment.strictMonotoneChain에 '124' 언급 없음: {sa['strictMonotoneChain']!r}"
    )


# ---------------------------------------------------------------------------
# v3 draft 실험 구조 검증
# ---------------------------------------------------------------------------


def test_v3_experiments_keys(v3: dict) -> None:
    """v3에 4개 실험 키가 정확히 존재해야 한다."""
    expected = {"AXS-IMP-001", "AXS-NOISE-001", "AXS-TB-001", "AXS-ALPHA-EXP"}
    actual = set(v3.get("experiments", {}).keys())
    assert actual == expected, (
        f"experiments 키 불일치: {actual} != {expected}"
    )


def test_v3_axs_imp_001_arms(v3: dict) -> None:
    """AXS-IMP-001 arms가 정확히 4개 이름을 가져야 한다."""
    exp = v3.get("experiments", {}).get("AXS-IMP-001", {})
    expected = ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]
    assert exp.get("arms") == expected, (
        f"AXS-IMP-001 arms 불일치: {exp.get('arms')!r}"
    )


def test_v3_axs_noise_001_arms(v3: dict) -> None:
    """AXS-NOISE-001 arms가 정확히 2개 이름을 가져야 한다."""
    exp = v3.get("experiments", {}).get("AXS-NOISE-001", {})
    expected = ["axs_ucb_default", "axs_yoked_bonus"]
    assert exp.get("arms") == expected, (
        f"AXS-NOISE-001 arms 불일치: {exp.get('arms')!r}"
    )


def test_v3_axs_alpha_exp_exploratory_flags(v3: dict) -> None:
    """AXS-ALPHA-EXP에 exploratory=true, noClaimLicense=true가 있어야 한다."""
    exp = v3.get("experiments", {}).get("AXS-ALPHA-EXP", {})
    assert exp.get("exploratory") is True, (
        f"AXS-ALPHA-EXP.exploratory가 True가 아님: {exp.get('exploratory')!r}"
    )
    assert exp.get("noClaimLicense") is True, (
        f"AXS-ALPHA-EXP.noClaimLicense가 True가 아님: {exp.get('noClaimLicense')!r}"
    )


def test_v3_axs_imp_001_strictMonotoneChainNote_present(v3: dict) -> None:
    """AXS-IMP-001에 strictMonotoneChainNote 키가 있어야 한다."""
    exp = v3.get("experiments", {}).get("AXS-IMP-001", {})
    note = exp.get("strictMonotoneChainNote")
    assert note is not None and note, (
        f"AXS-IMP-001.strictMonotoneChainNote 누락 또는 비어 있음: {note!r}"
    )


# ---------------------------------------------------------------------------
# 6개 branches + branchCountFrozen (AXS-IMP-001 외부 v3 top-level)
# ---------------------------------------------------------------------------


def test_v3_branches_exactly_6(v3: dict) -> None:
    """v3 최상위 branches가 정확히 6개여야 한다."""
    branches = v3.get("branches", [])
    expected_names = {
        "imprint_washout_supported",
        "imprint_only",
        "noise_only",
        "degenerate_qualified",
        "no_claim_m1_only",
        "integrity_fail",
    }
    actual_names = {b["name"] if isinstance(b, dict) else b for b in branches}
    assert len(branches) == 6, (
        f"branches 개수가 6이 아님: {len(branches)}"
    )
    assert actual_names == expected_names, (
        f"branches 이름 불일치: {actual_names} != {expected_names}"
    )


def test_v3_branchCountFrozen(v3: dict) -> None:
    """branchCountFrozen이 true여야 한다."""
    assert v3.get("branchCountFrozen") is True, (
        f"branchCountFrozen이 True가 아님: {v3.get('branchCountFrozen')!r}"
    )


# ---------------------------------------------------------------------------
# tieBreakCaveatMarker 계보 불변
# ---------------------------------------------------------------------------


def test_tieBreakCaveatMarker_identical_to_v1(v1: dict, v3: dict) -> None:
    """tieBreakCaveatMarker가 v1과 동일해야 한다."""
    assert v3["tieBreakCaveatMarker"] == v1["tieBreakCaveatMarker"], (
        f"tieBreakCaveatMarker 불일치: "
        f"v3={v3['tieBreakCaveatMarker']!r}, "
        f"v1={v1['tieBreakCaveatMarker']!r}"
    )


# ---------------------------------------------------------------------------
# contrastLivenessEvidence 4-family per-family floats가 summary와 일치
# ---------------------------------------------------------------------------


def test_v3_contrastLivenessEvidence_has_all_4_families(v3: dict) -> None:
    """contrastLivenessEvidence에 4개 family 값이 모두 있어야 한다."""
    clue = v3.get("contrastLivenessEvidence", {})
    delta_imp_pf = clue.get("deltaImpPerFamily", {})
    delta_noise_pf = clue.get("deltaNoisePerFamily", {})

    expected_families = {"123", "124", "777", "55555"}
    assert set(delta_imp_pf.keys()) == expected_families, (
        f"deltaImpPerFamily 키 불일치: {set(delta_imp_pf.keys())} != {expected_families}"
    )
    assert set(delta_noise_pf.keys()) == expected_families, (
        f"deltaNoisePerFamily 키 불일치: {set(delta_noise_pf.keys())} != {expected_families}"
    )


def test_v3_contrastLivenessEvidence_matches_pilot_v3_reports(
    v3: dict,
    summary_v2: dict,
    pilot_v3_reports: dict,
) -> None:
    """contrastLivenessEvidence의 4-family per-family float가 pilot_v3 소스에서 도출한 값과 일치해야 한다."""
    if not PILOT_V3_COMPLETE:
        pytest.skip("pilot_v3 디렉터리 없음 또는 완전하지 않음 — CI 안전 스킵")

    clue = v3.get("contrastLivenessEvidence", {})
    delta_imp_pf = clue.get("deltaImpPerFamily", {})
    delta_noise_pf = clue.get("deltaNoisePerFamily", {})

    for family in ["123", "124", "777", "55555"]:
        computed = _extract_contrasts_from_reports(
            pilot_v3_reports[family]["AXS-003"],
            pilot_v3_reports[family]["AXS-009"],
            pilot_v3_reports[family]["AXS-004c"],
            family,
        )
        assert delta_imp_pf.get(family) == computed["delta_imp"], (
            f"deltaImpPerFamily[{family}] 불일치: "
            f"저장됨={delta_imp_pf.get(family)!r}, 계산됨={computed['delta_imp']!r}"
        )
        assert delta_noise_pf.get(family) == computed["delta_noise"], (
            f"deltaNoisePerFamily[{family}] 불일치: "
            f"저장됨={delta_noise_pf.get(family)!r}, 계산됨={computed['delta_noise']!r}"
        )
