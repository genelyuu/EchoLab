"""tests/test_prereg_v3_draft.py

AXS v3 설계 초안 사전등록 + 파일럿 요약 v2 검증 테스트.

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

PILOT_123_DIR = REPO / "outputs/reports/pilot_123_postfix"
PILOT_CRITIC_DIR = REPO / "outputs/reports/pilot_critic_postfix"

POSTFIX_REPORTS_EXIST = PILOT_123_DIR.exists() and PILOT_CRITIC_DIR.exists()


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
def postfix_reports_123() -> dict[str, dict]:
    """pilot_123_postfix 디렉터리의 4개 리포트를 실험 ID별로 반환한다."""
    if not PILOT_123_DIR.exists():
        return {}
    reports: dict[str, dict] = {}
    for exp_id, glob_prefix in [
        ("AXS-003", "axs_003_"),
        ("AXS-004c", "axs_004c_"),
        ("AXS-009", "axs_009_"),
        ("AXS-010", "axs_010_"),
    ]:
        matches = list(PILOT_123_DIR.glob(f"{glob_prefix}*.json"))
        assert matches, f"{exp_id} 리포트 파일을 찾을 수 없음 (pilot_123_postfix)"
        with open(matches[0], encoding="utf-8") as f:
            reports[exp_id] = json.load(f)
    return reports


@pytest.fixture(scope="module")
def postfix_reports_critic() -> dict[str, dict]:
    """pilot_critic_postfix 디렉터리의 4개 리포트를 실험 ID별로 반환한다."""
    if not PILOT_CRITIC_DIR.exists():
        return {}
    reports: dict[str, dict] = {}
    for exp_id, glob_prefix in [
        ("AXS-003", "axs_003_"),
        ("AXS-004c", "axs_004c_"),
        ("AXS-009", "axs_009_"),
        ("AXS-010", "axs_010_"),
    ]:
        matches = list(PILOT_CRITIC_DIR.glob(f"{glob_prefix}*.json"))
        assert matches, f"{exp_id} 리포트 파일을 찾을 수 없음 (pilot_critic_postfix)"
        with open(matches[0], encoding="utf-8") as f:
            reports[exp_id] = json.load(f)
    return reports


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


def test_summary_v2_bugRecord_fixCommit_resolves(summary_v2: dict) -> None:
    """bugRecord.fixCommit이 git으로 해석 가능하고 b7b64b4로 시작해야 한다."""
    if not POSTFIX_REPORTS_EXIST:
        pytest.skip("postfix 리포트 디렉터리 없음 — CI 안전 스킵")

    bug = summary_v2.get("bugRecord", {})
    fix_commit = bug.get("fixCommit")
    assert fix_commit is not None, "bugRecord.fixCommit 키 누락"
    assert fix_commit.startswith("b7b64b4"), (
        f"fixCommit이 b7b64b4로 시작하지 않음: {fix_commit!r}"
    )
    # git rev-parse를 통해 해석 가능 여부 확인
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", fix_commit],
        capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"fixCommit을 git으로 해석할 수 없음: {fix_commit!r} "
        f"(stderr: {result.stderr.strip()})"
    )


def test_summary_v2_postfix_report_hashes_match_source(
    summary_v2: dict,
    postfix_reports_123: dict,
    postfix_reports_critic: dict,
) -> None:
    """summary v2의 postFixPilot 리포트 해시가 소스 리포트의 reportHash와 일치해야 한다."""
    if not POSTFIX_REPORTS_EXIST:
        pytest.skip("postfix 리포트 디렉터리 없음 — CI 안전 스킵")

    post_fix = summary_v2.get("postFixPilot", {})

    # family 123 checks
    family_123 = post_fix.get("123", {})
    for exp_id, report in postfix_reports_123.items():
        source_hash = report.get("reportHash")
        stored_hash = family_123.get(exp_id, {}).get("reportHash")
        assert stored_hash is not None, (
            f"postFixPilot[123][{exp_id}].reportHash 키 누락"
        )
        assert stored_hash == source_hash, (
            f"postFixPilot[123][{exp_id}].reportHash 불일치: "
            f"저장됨={stored_hash!r}, 소스={source_hash!r}"
        )

    # family 55555 checks
    family_55555 = post_fix.get("55555", {})
    for exp_id, report in postfix_reports_critic.items():
        source_hash = report.get("reportHash")
        stored_hash = family_55555.get(exp_id, {}).get("reportHash")
        assert stored_hash is not None, (
            f"postFixPilot[55555][{exp_id}].reportHash 키 누락"
        )
        assert stored_hash == source_hash, (
            f"postFixPilot[55555][{exp_id}].reportHash 불일치: "
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


def test_summary_v2_derived_contrasts_match_source(
    summary_v2: dict,
    postfix_reports_123: dict,
    postfix_reports_critic: dict,
) -> None:
    """summary v2의 derived contrast 값이 소스 리포트에서 재계산한 값과 일치해야 한다."""
    if not POSTFIX_REPORTS_EXIST:
        pytest.skip("postfix 리포트 디렉터리 없음 — CI 안전 스킵")

    post_fix = summary_v2.get("postFixPilot", {})

    # Family 123
    computed_123 = _extract_contrasts_from_reports(
        postfix_reports_123["AXS-003"],
        postfix_reports_123["AXS-009"],
        postfix_reports_123["AXS-004c"],
        "123",
    )
    stored_123 = post_fix.get("123", {})
    for key, expected in computed_123.items():
        stored = stored_123.get(key)
        assert stored == expected, (
            f"postFixPilot[123][{key}] 불일치: 저장됨={stored!r}, 계산됨={expected!r}"
        )

    # Family 55555
    computed_55555 = _extract_contrasts_from_reports(
        postfix_reports_critic["AXS-003"],
        postfix_reports_critic["AXS-009"],
        postfix_reports_critic["AXS-004c"],
        "55555",
    )
    stored_55555 = post_fix.get("55555", {})
    for key, expected in computed_55555.items():
        stored = stored_55555.get(key)
        assert stored == expected, (
            f"postFixPilot[55555][{key}] 불일치: 저장됨={stored!r}, 계산됨={expected!r}"
        )


def test_summary_v2_runCommit_consistent(
    summary_v2: dict,
    postfix_reports_123: dict,
    postfix_reports_critic: dict,
) -> None:
    """bugRecord.fixCommit이 모든 postfix 리포트의 preregStamp.runCommit과 일치해야 한다."""
    if not POSTFIX_REPORTS_EXIST:
        pytest.skip("postfix 리포트 디렉터리 없음 — CI 안전 스킵")

    fix_commit = summary_v2.get("bugRecord", {}).get("fixCommit")
    assert fix_commit is not None, "bugRecord.fixCommit 없음"

    for exp_id, report in postfix_reports_123.items():
        run_commit = report.get("preregStamp", {}).get("runCommit")
        assert run_commit == fix_commit, (
            f"pilot_123_postfix {exp_id}: runCommit={run_commit!r} != fixCommit={fix_commit!r}"
        )

    for exp_id, report in postfix_reports_critic.items():
        run_commit = report.get("preregStamp", {}).get("runCommit")
        assert run_commit == fix_commit, (
            f"pilot_critic_postfix {exp_id}: runCommit={run_commit!r} != fixCommit={fix_commit!r}"
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
# contrastLivenessEvidence per-family floats가 summary와 일치
# ---------------------------------------------------------------------------


def test_v3_contrastLivenessEvidence_matches_summary_contrasts(
    v3: dict,
    summary_v2: dict,
    postfix_reports_123: dict,
    postfix_reports_critic: dict,
) -> None:
    """contrastLivenessEvidence의 per-family float가 summary v2에서 도출한 값과 일치해야 한다."""
    if not POSTFIX_REPORTS_EXIST:
        pytest.skip("postfix 리포트 디렉터리 없음 — CI 안전 스킵")

    clue = v3.get("contrastLivenessEvidence", {})
    delta_imp_pf = clue.get("deltaImpPerFamily", {})
    delta_noise_pf = clue.get("deltaNoisePerFamily", {})

    # Recompute from source reports
    computed_123 = _extract_contrasts_from_reports(
        postfix_reports_123["AXS-003"],
        postfix_reports_123["AXS-009"],
        postfix_reports_123["AXS-004c"],
        "123",
    )
    computed_55555 = _extract_contrasts_from_reports(
        postfix_reports_critic["AXS-003"],
        postfix_reports_critic["AXS-009"],
        postfix_reports_critic["AXS-004c"],
        "55555",
    )

    assert delta_imp_pf.get("123") == computed_123["delta_imp"], (
        f"deltaImpPerFamily[123] 불일치: "
        f"저장됨={delta_imp_pf.get('123')!r}, 계산됨={computed_123['delta_imp']!r}"
    )
    assert delta_imp_pf.get("55555") == computed_55555["delta_imp"], (
        f"deltaImpPerFamily[55555] 불일치: "
        f"저장됨={delta_imp_pf.get('55555')!r}, 계산됨={computed_55555['delta_imp']!r}"
    )
    assert delta_noise_pf.get("123") == computed_123["delta_noise"], (
        f"deltaNoisePerFamily[123] 불일치: "
        f"저장됨={delta_noise_pf.get('123')!r}, 계산됨={computed_123['delta_noise']!r}"
    )
    assert delta_noise_pf.get("55555") == computed_55555["delta_noise"], (
        f"deltaNoisePerFamily[55555] 불일치: "
        f"저장됨={delta_noise_pf.get('55555')!r}, 계산됨={computed_55555['delta_noise']!r}"
    )
