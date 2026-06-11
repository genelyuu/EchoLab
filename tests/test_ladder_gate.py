"""tests/test_ladder_gate.py — ladder_gate 라이브러리 + red-team 픽스처 (G-022b / G-022d).

TDD: failing tests first, then implementation makes them green.

픽스처 전략:
- 실제 prereg(configs/prereg/axs_mechanism_prereg_v1.json) 를 tmp_path 로 복사.
- 유효한 5개 리포트(AXS-003/009/004c/002/010) 를 합성 생성.
- reportHash 는 canonical_hash(report minus reportHash) 로 정확히 계산.
- 원장 엔트리는 append_ledger_entry 로 등록.
- git_runner 를 주입해 merge-base / branch -r 을 결정론적으로 제어.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from echo_bench.logging.prereg import (
    append_ledger_entry,
    load_ledger,
    load_prereg,
    prereg_hash,
)
from echo_bench.tools.ladder_gate import evaluate_mechanism_license, main
from echo_bench.utils.hash import canonical_hash

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_PREREG = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_REAL_LEDGER = _REPO_ROOT / "configs" / "prereg" / "run_ledger.json"


# ---------------------------------------------------------------------------
# git_runner factory helpers
# ---------------------------------------------------------------------------

GOOD_PREREG_COMMIT = "aabbcc1122334455667788990011223344556677"
GOOD_RUN_COMMIT = "ff00ee1122334455667788990011223344556677"

# Real git commits from the repo (for CLI tests that use real git).
# preregCommit: git log -1 --format=%H -- configs/prereg/axs_mechanism_prereg_v1.json
# runCommit:    git rev-parse HEAD
# Resolved dynamically at module load time; None if git is unavailable.
def _resolve_real_sha_pair() -> tuple:
    """런타임에 git 에서 CLI 테스트용 SHA 쌍 해석.

    실패 시 (None, None) 반환 — git checkout 이 아닌 환경에서 건너뜀.
    """
    import subprocess as _sp
    prereg_file = "configs/prereg/axs_mechanism_prereg_v1.json"
    try:
        prereg_commit = _sp.run(
            ["git", "log", "-1", "--format=%H", "--", prereg_file],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        run_commit = _sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if prereg_commit and run_commit:
            return prereg_commit, run_commit
    except Exception:
        pass
    return None, None


_REAL_PREREG_COMMIT, _REAL_RUN_COMMIT = _resolve_real_sha_pair()


def _good_git_runner(args: List[str]) -> str:
    """조상 관계 성립, 원격 브랜치 존재 시뮬레이션."""
    cmd = " ".join(args)
    if "merge-base" in cmd and "--is-ancestor" in cmd:
        return ""  # exit 0 → 조상 관계 성립
    if "branch" in cmd and "-r" in cmd and "--contains" in cmd:
        return "  origin/main"
    if "rev-parse" in cmd:
        return GOOD_RUN_COMMIT
    if "log" in cmd and "--format=%H" in cmd:
        return GOOD_PREREG_COMMIT
    return ""


def _bad_ancestor_git_runner(args: List[str]) -> str:
    """조상 관계 성립하지 않는 git_runner."""
    cmd = " ".join(args)
    if "merge-base" in cmd and "--is-ancestor" in cmd:
        raise ValueError("조상 관계 없음 (테스트 시뮬레이션)")
    return _good_git_runner(args)


def _remote_empty_git_runner(args: List[str]) -> str:
    """원격 브랜치 비어 있는 git_runner (release 조건 실패)."""
    cmd = " ".join(args)
    if "branch" in cmd and "-r" in cmd and "--contains" in cmd:
        return ""  # 빈 문자열 → 원격 브랜치 없음
    return _good_git_runner(args)


# ---------------------------------------------------------------------------
# Evaluation families (must match prereg)
# ---------------------------------------------------------------------------

EVAL_FAMILIES = ["42", "7", "101", "2025", "31337"]


def _per_family_present(metric: str, value: float = 0.08) -> Dict[str, Any]:
    """perFamily 구조 생성 — 모든 패밀리 양수값(present 판정)."""
    return {fam: {metric: value} for fam in EVAL_FAMILIES}


def _per_family_absent(metric: str) -> Dict[str, Any]:
    """perFamily 구조 생성 — 모든 패밀리 0(absent 판정)."""
    return {fam: {metric: 0.0} for fam in EVAL_FAMILIES}


def _bootstrap_present(metric: str) -> Dict[str, Any]:
    return {metric: {"mean": 0.08, "ciLower": 0.05, "ciUpper": 0.11}}


def _bootstrap_absent(metric: str) -> Dict[str, Any]:
    """ciLower <= 0 → absent."""
    return {metric: {"mean": -0.01, "ciLower": -0.05, "ciUpper": 0.02}}


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _build_stamp(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    prereg_commit: str = GOOD_PREREG_COMMIT,
    run_commit: str = GOOD_RUN_COMMIT,
) -> Dict[str, Any]:
    return {
        "preregId": prereg["preregId"],
        "preregVersion": prereg["version"],
        "preregPath": prereg_path_str,
        "preregHash": p_hash,
        "preregCommit": prereg_commit,
        "runCommit": run_commit,
    }


def _finalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """reportHash 를 정확히 계산해 삽입."""
    body = {k: v for k, v in report.items() if k != "reportHash"}
    report["reportHash"] = canonical_hash(body)
    return report


def _build_axs003(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    prereg_commit: str = GOOD_PREREG_COMMIT,
    run_commit: str = GOOD_RUN_COMMIT,
) -> Dict[str, Any]:
    metric = "slate_excess_nmi"
    r = {
        "reportId": "rep-axs003-v1",
        "experimentId": "AXS-003",
        "preregStamp": _build_stamp(prereg, p_hash, prereg_path_str, prereg_commit=prereg_commit, run_commit=run_commit),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
        "arms": {
            "axs_ucb_default": {
                "perFamily": _per_family_present(metric),
                "bootstrap": _bootstrap_present(metric),
                "utility": {"coordinate_coverage_mean": 0.93},
            },
            "axs_ucb_alpha0": {
                "perFamily": _per_family_absent(metric),
                "bootstrap": _bootstrap_absent(metric),
                "utility": {"coordinate_coverage_mean": 0.91},
            },
        },
    }
    return _finalize_report(r)


def _build_axs009(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    prereg_commit: str = GOOD_PREREG_COMMIT,
    run_commit: str = GOOD_RUN_COMMIT,
) -> Dict[str, Any]:
    metric_nmi = "slate_excess_nmi"
    metric_div = "post_freeze_incremental_divergence"
    # dose-response: freeze_at_1 ≤ quarter ≤ half ≤ freeze_none
    r = {
        "reportId": "rep-axs009-v1",
        "experimentId": "AXS-009",
        "preregStamp": _build_stamp(prereg, p_hash, prereg_path_str, prereg_commit=prereg_commit, run_commit=run_commit),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
        "arms": {
            "freeze_at_1": {
                "perFamily": _per_family_present(metric_nmi, 0.06),
                "bootstrap": {
                    metric_nmi: {"mean": 0.06, "ciLower": 0.03, "ciUpper": 0.09},
                    metric_div: {"mean": -0.01, "ciLower": -0.04, "ciUpper": 0.00},
                },
                "utility": {"coordinate_coverage_mean": 0.92},
            },
            "freeze_at_quarter": {
                "perFamily": _per_family_present(metric_nmi, 0.07),
                "bootstrap": {
                    metric_nmi: {"mean": 0.07, "ciLower": 0.04, "ciUpper": 0.10},
                    metric_div: {"mean": -0.01, "ciLower": -0.04, "ciUpper": 0.00},
                },
                "utility": {"coordinate_coverage_mean": 0.93},
            },
            "freeze_at_half": {
                "perFamily": _per_family_present(metric_nmi, 0.075),
                "bootstrap": {
                    metric_nmi: {"mean": 0.075, "ciLower": 0.04, "ciUpper": 0.10},
                    metric_div: {"mean": -0.01, "ciLower": -0.04, "ciUpper": 0.00},
                },
                "utility": {"coordinate_coverage_mean": 0.93},
            },
            "freeze_none": {
                "perFamily": _per_family_present(metric_nmi, 0.08),
                "bootstrap": {
                    metric_nmi: {"mean": 0.08, "ciLower": 0.05, "ciUpper": 0.11},
                    metric_div: {"mean": 0.00, "ciLower": -0.01, "ciUpper": 0.01},
                },
                "utility": {"coordinate_coverage_mean": 0.93},
            },
        },
    }
    return _finalize_report(r)


def _build_axs004c(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    prereg_commit: str = GOOD_PREREG_COMMIT,
    run_commit: str = GOOD_RUN_COMMIT,
) -> Dict[str, Any]:
    metric = "slate_excess_nmi"
    r = {
        "reportId": "rep-axs004c-v1",
        "experimentId": "AXS-004c",
        "preregStamp": _build_stamp(prereg, p_hash, prereg_path_str, prereg_commit=prereg_commit, run_commit=run_commit),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
        "arms": {
            "axs_ucb_default": {
                "perFamily": _per_family_present(metric),
                "bootstrap": _bootstrap_present(metric),
                "utility": {"coordinate_coverage_mean": 0.93},
            },
            "axs_yoked_bonus": {
                "perFamily": _per_family_absent(metric),
                "bootstrap": _bootstrap_absent(metric),
                "utility": {"coordinate_coverage_mean": 0.91},
            },
        },
    }
    return _finalize_report(r)


def _build_axs002(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    prereg_commit: str = GOOD_PREREG_COMMIT,
    run_commit: str = GOOD_RUN_COMMIT,
) -> Dict[str, Any]:
    metric = "slate_excess_nmi_diff"
    r = {
        "reportId": "rep-axs002-v1",
        "experimentId": "AXS-002",
        "preregStamp": _build_stamp(prereg, p_hash, prereg_path_str, prereg_commit=prereg_commit, run_commit=run_commit),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
        "arms": {
            "utility_matched_contrast": {
                "perFamily": _per_family_present(metric),
                "bootstrap": _bootstrap_present(metric),
                "utility": {"coordinate_coverage_mean": 0.93},
            },
        },
    }
    return _finalize_report(r)


def _build_axs010_strict(prereg: Dict[str, Any], p_hash: str, prereg_path_str: str) -> Dict[str, Any]:
    """strict_pass: 모든 변형 estimate CI 내."""
    r = {
        "reportId": "rep-axs010-v1",
        "experimentId": "AXS-010",
        "preregStamp": _build_stamp(prereg, p_hash, prereg_path_str),
        "replayAudit": {"replayable": True},
        "tieBreak": {
            "baseline": {
                "sign": "+",
                "estimate": 0.08,
                "ciLower": 0.05,
                "ciUpper": 0.11,
                "trackDecision": "S",
            },
            "variants": {
                "reverse": {"sign": "+", "estimate": 0.079, "trackDecision": "S"},
                "hash_seeded": {"sign": "+", "estimate": 0.081, "trackDecision": "S"},
                "feature_lexicographic": {"sign": "+", "estimate": 0.078, "trackDecision": "S"},
            },
        },
    }
    return _finalize_report(r)


def _build_axs010_soft(prereg: Dict[str, Any], p_hash: str, prereg_path_str: str) -> Dict[str, Any]:
    """soft_pass: 부호/trackDecision 일치, 하나가 CI 밖."""
    r = {
        "reportId": "rep-axs010-v1",
        "experimentId": "AXS-010",
        "preregStamp": _build_stamp(prereg, p_hash, prereg_path_str),
        "replayAudit": {"replayable": True},
        "tieBreak": {
            "baseline": {
                "sign": "+",
                "estimate": 0.08,
                "ciLower": 0.05,
                "ciUpper": 0.11,
                "trackDecision": "S",
            },
            "variants": {
                "reverse": {"sign": "+", "estimate": 0.03, "trackDecision": "S"},  # CI 밖
                "hash_seeded": {"sign": "+", "estimate": 0.081, "trackDecision": "S"},
                "feature_lexicographic": {"sign": "+", "estimate": 0.078, "trackDecision": "S"},
            },
        },
    }
    return _finalize_report(r)


# ---------------------------------------------------------------------------
# Ledger helper
# ---------------------------------------------------------------------------


def _make_ledger(tmp_path: Path) -> Path:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps({"ledgerVersion": 1, "entries": []}, indent=2), encoding="utf-8"
    )
    return ledger_path


def _register_report(
    ledger_path: Path,
    report: Dict[str, Any],
    prereg: Dict[str, Any],
    p_hash: str,
    report_path: Path,
) -> None:
    entry = {
        "reportId": report["reportId"],
        "experimentId": report["experimentId"],
        "preregId": prereg["preregId"],
        "preregVersion": prereg["version"],
        "preregHash": p_hash,
        "reportHash": report["reportHash"],
        "reportPath": str(report_path),
        "runCommit": GOOD_RUN_COMMIT,
    }
    append_ledger_entry(ledger_path, entry)


# ---------------------------------------------------------------------------
# Full-set fixture builder
# ---------------------------------------------------------------------------


def _build_full_set(
    tmp_path: Path,
    axs010_builder=None,
    prereg_override: Optional[Dict[str, Any]] = None,
    *,
    prereg_commit: str = GOOD_PREREG_COMMIT,
    run_commit: str = GOOD_RUN_COMMIT,
):
    """유효한 5-리포트 세트 + 원장 + prereg 복사본 생성.

    Returns: (prereg_path, report_paths, ledger_path, prereg_dict, p_hash)
    """
    # prereg 복사
    prereg_src = _REAL_PREREG
    prereg_path = tmp_path / "prereg.json"
    if prereg_override is not None:
        prereg_path.write_text(
            json.dumps(prereg_override, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        prereg = prereg_override
    else:
        shutil.copy(prereg_src, prereg_path)
        prereg = load_prereg(prereg_path)

    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    if axs010_builder is None:
        axs010_builder = _build_axs010_strict

    # Build stamp builder with injected commits
    def _stamper(p: Dict[str, Any], ph: str, pps: str) -> Dict[str, Any]:
        return _build_stamp(p, ph, pps, prereg_commit=prereg_commit, run_commit=run_commit)

    # AXS-010: rebuild with correct stamp and tieBreak from axs010_builder
    def _axs010_with_stamp(p, ph, pps):
        stamp = _stamper(p, ph, pps)
        base = axs010_builder(p, ph, pps)
        r = {
            "reportId": "rep-axs010-v1",
            "experimentId": "AXS-010",
            "preregStamp": stamp,
            "replayAudit": {"replayable": True},
            "tieBreak": base["tieBreak"],
        }
        return _finalize_report(r)

    reports_data = [
        _build_axs003(prereg, p_hash, ps, prereg_commit=prereg_commit, run_commit=run_commit),
        _build_axs009(prereg, p_hash, ps, prereg_commit=prereg_commit, run_commit=run_commit),
        _build_axs004c(prereg, p_hash, ps, prereg_commit=prereg_commit, run_commit=run_commit),
        _build_axs002(prereg, p_hash, ps, prereg_commit=prereg_commit, run_commit=run_commit),
        _axs010_with_stamp(prereg, p_hash, ps),
    ]

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in reports_data:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    return prereg_path, report_paths, ledger_path, prereg, p_hash


# ===========================================================================
# Tests — Happy paths
# ===========================================================================


def test_happy_path_strict_pass(tmp_path):
    """픽스처 1: 정상 경로 — M2 True, caveatRequired False."""
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result["rungs"]["M0"] is True
    assert result["rungs"]["M1"] is True
    assert result["rungs"]["M2"] is True
    assert result["rungs"]["M3"] is False  # MVP
    assert result["caveatRequired"] is False
    assert result["provisional"] is True  # release=False 기본값
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]


def test_happy_path_soft_pass(tmp_path):
    """픽스처 2: soft_pass variant — M2 True + caveatRequired True."""
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path, axs010_builder=_build_axs010_soft
    )
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result["rungs"]["M2"] is True
    assert result["caveatRequired"] is True
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]


def test_m3_always_false(tmp_path):
    """M3 는 MVP 에서 항상 False."""
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result["rungs"]["M3"] is False


def test_provisional_false_when_release(tmp_path):
    """release=True 이면 provisional=False."""
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
        release=True,
    )
    assert result["provisional"] is False


# ===========================================================================
# Red-team tests — 각각 게이트 실패
# ===========================================================================


def test_redteam_tampered_licenses_ignored(tmp_path):
    """픽스처 3: 사전 작성된 licenses.json M2:true 가 있어도 증거 실패 시 M2 False.
    CLI 가 파일을 덮어씀.
    """
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # preregHash 를 변조해 검사 실패 유도
    bad_reports = []
    for rp in report_paths:
        r = json.loads(rp.read_text())
        r["preregStamp"]["preregHash"] = "deadbeef" * 8
        # reportHash 재계산 (해시 무결성 검사는 통과시키기 위해)
        body = {k: v for k, v in r.items() if k != "reportHash"}
        r["reportHash"] = canonical_hash(body)
        new_rp = tmp_path / f"bad_{rp.name}"
        new_rp.write_text(json.dumps(r, indent=2), encoding="utf-8")
        bad_reports.append(new_rp)

    # 조작된 licenses.json 사전 작성
    licenses_out = tmp_path / "licenses.json"
    licenses_out.write_text(
        json.dumps({"rungs": {"M2": True}, "note": "tampered"}, indent=2),
        encoding="utf-8",
    )

    result = evaluate_mechanism_license(
        prereg_path,
        bad_reports,
        ledger_path=ledger_path,  # 원본 원장 (등록 안 됨 → 실패)
        git_runner=_good_git_runner,
    )
    # prereg_hash_match 실패로 M2 False
    assert result["rungs"]["M2"] is False
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "prereg_hash_match" for c in failed)

    # CLI 실행: licenses.json 을 덮어써야 함
    ret = main(
        [
            "--prereg", str(prereg_path),
            "--reports", *[str(r) for r in bad_reports],
            "--ledger", str(ledger_path),
            "--licenses-out", str(licenses_out),
        ]
    )
    assert ret == 1
    written = json.loads(licenses_out.read_text())
    # 덮어쓴 파일의 M2 는 False
    assert written["rungs"]["M2"] is False
    assert "AUDIT LOG ONLY" in written.get("note", "")


def test_redteam_prereg_hash_mismatch(tmp_path):
    """픽스처 4: preregHash 불일치 → 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # 리포트의 preregHash 를 변조
    bad_reports = []
    for rp in report_paths:
        r = json.loads(rp.read_text())
        r["preregStamp"]["preregHash"] = "badhash" * 9
        body = {k: v for k, v in r.items() if k != "reportHash"}
        r["reportHash"] = canonical_hash(body)
        new_rp = tmp_path / f"mismatch_{rp.name}"
        new_rp.write_text(json.dumps(r, indent=2), encoding="utf-8")
        bad_reports.append(new_rp)

    result = evaluate_mechanism_license(
        prereg_path,
        bad_reports,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "prereg_hash_match" for c in failed)
    assert result["rungs"]["M2"] is False


def test_redteam_missing_arm(tmp_path):
    """픽스처 5a: arm 누락 → 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # AXS-003 리포트에서 axs_ucb_alpha0 arm 제거
    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    del r["arms"]["axs_ucb_alpha0"]
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed)


def test_redteam_below_random_arm_not_marked(tmp_path):
    """픽스처 5b: RANDOM 기준 미달 arm 에 degenerate 미표기 → 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    # axs_ucb_alpha0 arm 을 RANDOM 미달로 설정 (degenerate 미표기)
    r["arms"]["axs_ucb_alpha0"]["utility"]["coordinate_coverage_mean"] = 0.50
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed)


def test_redteam_report_not_in_ledger(tmp_path):
    """픽스처 6: 리포트가 원장에 미등록 → 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # 원장을 비워 재생성
    empty_ledger = tmp_path / "empty_ledger.json"
    empty_ledger.write_text(
        json.dumps({"ledgerVersion": 1, "entries": []}, indent=2), encoding="utf-8"
    )

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=empty_ledger,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "ledger_registered" for c in failed)
    assert result["rungs"]["M2"] is False


def test_redteam_replayable_false(tmp_path):
    """픽스처 7a: replayAudit.replayable=False → 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    r["replayAudit"]["replayable"] = False
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "replayable" for c in failed)


def test_redteam_replayable_missing(tmp_path):
    """픽스처 7b: replayAudit 키 자체 누락 → 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    del r["replayAudit"]
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "replayable" for c in failed)


def test_redteam_self_attested_pass_but_bad_ancestry(tmp_path):
    """픽스처 8: 자가 증명 필드 무시 — git 이 조상 관계 부정 → ancestry 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # 각 리포트에 자가 증명 필드 추가 (무시되어야 함)
    for rp in report_paths:
        r = json.loads(rp.read_text())
        r["preregRemoteVerified"] = True
        r["pass"] = True
        r["accepted"] = True
        body = {k: v for k, v in r.items() if k != "reportHash"}
        r["reportHash"] = canonical_hash(body)
        rp.write_text(json.dumps(r, indent=2), encoding="utf-8")
        # 원장 업데이트
        # 기존 엔트리를 교체하기 위해 원장을 다시 로드하고 해당 엔트리 제거 후 재추가
        ledger = load_ledger(ledger_path)
        new_entries = [
            e for e in ledger["entries"] if e.get("reportId") != r["reportId"]
        ]
        ledger["entries"] = new_entries
        ledger_path.write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _register_report(ledger_path, r, prereg, p_hash, rp)

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_bad_ancestor_git_runner,  # git 이 조상 관계 부정
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "ancestry" for c in failed)
    assert result["rungs"]["M2"] is False


def test_redteam_pilot_family_in_eval_families(tmp_path):
    """픽스처 9: pilotFamily ∈ evaluationFamilies → pilot_disjoint 실패."""
    real_prereg = load_prereg(_REAL_PREREG)
    # pilotFamily 를 evaluationFamilies 중 하나로 변경
    bad_prereg = copy.deepcopy(real_prereg)
    bad_prereg["yokedSchedule"]["pilotFamily"] = "42"  # evaluationFamilies 에 있음

    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(
        tmp_path, prereg_override=bad_prereg
    )

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "pilot_disjoint" for c in failed)
    assert result["rungs"]["M2"] is False


def test_redteam_axs010_sign_flip(tmp_path):
    """픽스처 10a: AXS-010 변형 부호 뒤집기 → axs010_invariance 실패."""
    def _bad_sign_axs010(prereg, p_hash, ps):
        r = _build_axs010_strict(prereg, p_hash, ps)
        # reportHash 제거 후 변조
        r.pop("reportHash", None)
        r["tieBreak"]["variants"]["reverse"]["sign"] = "-"
        return _finalize_report(r)

    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(
        tmp_path, axs010_builder=_bad_sign_axs010
    )

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "axs010_invariance" for c in failed)
    assert result["rungs"]["M2"] is False


def test_redteam_axs010_track_decision_change(tmp_path):
    """픽스처 10b: AXS-010 trackDecision 변경 → axs010_invariance 실패."""
    def _bad_track_axs010(prereg, p_hash, ps):
        r = _build_axs010_strict(prereg, p_hash, ps)
        r.pop("reportHash", None)
        r["tieBreak"]["variants"]["hash_seeded"]["trackDecision"] = "X"
        return _finalize_report(r)

    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(
        tmp_path, axs010_builder=_bad_track_axs010
    )

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "axs010_invariance" for c in failed)
    assert result["rungs"]["M2"] is False


def test_redteam_v2_prereg_superseded_disposition(tmp_path):
    """픽스처 11: v2 prereg — v1 원장 엔트리는 superseded_reported disposition."""
    # v1 세트로 원장 등록
    prereg_path_v1, report_paths_v1, ledger_path, prereg_v1, p_hash_v1 = _build_full_set(tmp_path)

    # v2 prereg 생성 (supersedes / changeJustification 추가)
    real_prereg = load_prereg(_REAL_PREREG)
    v2_prereg = copy.deepcopy(real_prereg)
    v2_prereg["version"] = 2
    v2_prereg["supersedes"] = 1
    v2_prereg["changeJustification"] = "테스트 수정판"

    v2_path = tmp_path / "prereg_v2.json"
    v2_path.write_text(
        json.dumps(v2_prereg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    p_hash_v2 = prereg_hash(v2_prereg)

    # v2 리포트 생성 + 원장 등록
    ps_v2 = str(v2_path)
    v2_reports = [
        _build_axs003(v2_prereg, p_hash_v2, ps_v2),
        _build_axs009(v2_prereg, p_hash_v2, ps_v2),
        _build_axs004c(v2_prereg, p_hash_v2, ps_v2),
        _build_axs002(v2_prereg, p_hash_v2, ps_v2),
        _build_axs010_strict(v2_prereg, p_hash_v2, ps_v2),
    ]
    v2_report_paths = []
    for r in v2_reports:
        rp = tmp_path / f"v2_{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, v2_prereg, p_hash_v2, rp)
        v2_report_paths.append(rp)

    result = evaluate_mechanism_license(
        v2_path,
        v2_report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )

    # v1 엔트리들은 superseded_reported
    v1_hashes = {
        json.loads(rp.read_text())["reportHash"]
        for rp in report_paths_v1
    }
    history = result["familyHistory"]
    for entry in history:
        if entry.get("reportHash") in v1_hashes:
            assert entry["disposition"] == "superseded_reported", (
                f"v1 엔트리 disposition 오류: {entry}"
            )

    # v2 엔트리들은 consumed
    v2_hashes = {r["reportHash"] for r in v2_reports}
    for entry in history:
        if entry.get("reportHash") in v2_hashes:
            assert entry["disposition"] == "consumed", (
                f"v2 엔트리 disposition 오류: {entry}"
            )


def test_redteam_tampered_report_body(tmp_path):
    """픽스처 12: 리포트 본문 변조 (hash 불일치) → report_hash_integrity 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    original_hash = r["reportHash"]
    # 본문 변조 (reportHash 는 그대로)
    r["arms"]["axs_ucb_default"]["utility"]["coordinate_coverage_mean"] = 0.01
    r["reportHash"] = original_hash  # 원래 hash 유지 → 불일치
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "report_hash_integrity" for c in failed)
    assert result["rungs"]["M2"] is False


# ===========================================================================
# AXS-010 absent → caveatRequired False, M2 False
# ===========================================================================


def test_axs010_absent_m2_false(tmp_path):
    """AXS-010 리포트 없으면 M2 False, caveatRequired False."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # AXS-010 리포트 제외
    no_axs010 = [r for r in report_paths if "axs010" not in r.name]

    result = evaluate_mechanism_license(
        prereg_path,
        no_axs010,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result["rungs"]["M2"] is False
    assert result["caveatRequired"] is False
    c10 = next(c for c in result["checks"] if c["check"] == "axs010_invariance")
    assert "AXS-010 invariance 증거 없음" in c10["detail"]


# ===========================================================================
# consumed_reports + familyHistory structure
# ===========================================================================


def test_consumed_reports_and_family_history(tmp_path):
    """consumedReports 에 path + reportHash 포함, familyHistory disposition 확인."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert len(result["consumedReports"]) == 5
    for cr in result["consumedReports"]:
        assert "path" in cr
        assert "reportHash" in cr

    for entry in result["familyHistory"]:
        assert entry["disposition"] in {"consumed", "superseded_reported", "unconsumed_reported"}
    # 5개 모두 consumed
    consumed = [e for e in result["familyHistory"] if e["disposition"] == "consumed"]
    assert len(consumed) == 5


# ===========================================================================
# CLI tests
# ===========================================================================


def test_cli_exit0_happy_path(tmp_path):
    """CLI: 정상 경로 — 실제 git 커밋 동적 해석 사용, exit 0."""
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")
    # CLI 는 실제 git 을 호출하므로 레포에 실제로 존재하는 커밋을 stamp 에 주입
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path,
        prereg_commit=_REAL_PREREG_COMMIT,
        run_commit=_REAL_RUN_COMMIT,
    )
    licenses_out = tmp_path / "licenses.json"
    ret = main(
        [
            "--prereg", str(prereg_path),
            "--reports", *[str(r) for r in report_paths],
            "--ledger", str(ledger_path),
            "--licenses-out", str(licenses_out),
        ]
    )
    assert ret == 0, (
        f"CLI 가 exit 1 을 반환했습니다 — ancestry 실패 의심. "
        f"preregCommit={_REAL_PREREG_COMMIT[:12]}, runCommit={_REAL_RUN_COMMIT[:12]}"
    )
    written = json.loads(licenses_out.read_text())
    assert written["rungs"]["M2"] is True
    assert "AUDIT LOG ONLY" in written["note"]


def test_cli_exit1_on_failure(tmp_path):
    """CLI: 실패 검사 있으면 exit 1 (원장 미등록 유도)."""
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path,
        prereg_commit=_REAL_PREREG_COMMIT,
        run_commit=_REAL_RUN_COMMIT,
    )
    # 원장 비워서 실패 유도
    empty_ledger = tmp_path / "empty.json"
    empty_ledger.write_text(
        json.dumps({"ledgerVersion": 1, "entries": []}, indent=2), encoding="utf-8"
    )
    licenses_out = tmp_path / "licenses.json"
    ret = main(
        [
            "--prereg", str(prereg_path),
            "--reports", *[str(r) for r in report_paths],
            "--ledger", str(empty_ledger),
            "--licenses-out", str(licenses_out),
        ]
    )
    assert ret == 1


def test_cli_audit_note_in_output(tmp_path):
    """CLI 출력 파일에 AUDIT LOG ONLY 노트 포함."""
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path,
        prereg_commit=_REAL_PREREG_COMMIT,
        run_commit=_REAL_RUN_COMMIT,
    )
    licenses_out = tmp_path / "licenses_audit.json"
    main(
        [
            "--prereg", str(prereg_path),
            "--reports", *[str(r) for r in report_paths],
            "--ledger", str(ledger_path),
            "--licenses-out", str(licenses_out),
        ]
    )
    written = json.loads(licenses_out.read_text())
    assert "AUDIT LOG ONLY" in written.get("note", "")
    assert "입력으로 사용 금지" in written.get("note", "")


# ===========================================================================
# Fix 1 (CRITICAL fail-open): requiresPass 강제
# ===========================================================================


def test_redteam_only_axs010_submitted(tmp_path):
    """red-team: AXS-010 strict_pass 단독 제출 — M2 False, acceptance_recomputed 실패.

    requiresPass 에 열거된 AXS-003/009/004c/002 리포트가 없으면
    acceptance_recomputed 가 누락 실험을 한국어로 명시하며 실패해야 한다.
    """
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)

    # AXS-010 리포트만 남김
    only_axs010 = [r for r in report_paths if "axs010" in r.name]
    assert len(only_axs010) == 1, "AXS-010 리포트가 픽스처에 없음"

    result = evaluate_mechanism_license(
        prereg_path,
        only_axs010,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result["rungs"]["M2"] is False, "AXS-010 단독 제출 시 M2 는 False 여야 함"
    failed = [c for c in result["checks"] if not c["ok"]]
    acc_fail = next(
        (c for c in failed if c["check"] == "acceptance_recomputed"), None
    )
    assert acc_fail is not None, (
        "requiresPass 실험 누락 시 acceptance_recomputed 가 실패해야 함"
    )
    # 누락 실험이 detail 에 한국어로 명시되어야 함
    detail = acc_fail["detail"]
    assert "AXS-003" in detail or "AXS-009" in detail or "AXS-004c" in detail or "AXS-002" in detail, (
        f"acceptance_recomputed detail 에 누락 실험이 없음: {detail!r}"
    )


# ===========================================================================
# Fix 3 (minor): arms_complete — degenerate 값 검증
# ===========================================================================


def test_redteam_below_random_arm_false_degenerate_values(tmp_path):
    """red-team: RANDOM 미달 arm 에 degenerate=False 로 표기 → arms_complete 실패.

    키 존재만으로는 통과해선 안 된다.
    degenerate=false, degenerateReason="x", includedInMechanismClaim=true 조합은
    arms_complete 실패를 유발해야 한다.
    """
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    # axs_ucb_alpha0 arm 을 RANDOM 미달로 만들되, degenerate 값은 잘못 설정
    r["arms"]["axs_ucb_alpha0"]["utility"]["coordinate_coverage_mean"] = 0.50  # RANDOM(0.90) 미달
    r["arms"]["axs_ucb_alpha0"]["degenerate"] = False          # 잘못된 값
    r["arms"]["axs_ucb_alpha0"]["degenerateReason"] = "x"      # 비어 있지 않지만 degenerate=False
    r["arms"]["axs_ucb_alpha0"]["includedInMechanismClaim"] = True  # 잘못된 값
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed), (
        "degenerate=False arm 이 arms_complete 을 통과해서는 안 됨"
    )


# ===========================================================================
# Fix 2 (test fragility): CLI 테스트 SHA 동적 해석 — 새 픽스처 검증
# ===========================================================================


def test_cli_dynamic_sha_happy_path(tmp_path):
    """CLI: SHA 를 동적으로 해석해 ancestry 통과 — exit 0.

    Fix 2: 하드코딩된 SHA 대신 런타임에 git 에서 해석된 모듈 수준 변수 사용.
    git 해석 실패 시 skip (Korean reason).
    """
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")

    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path,
        prereg_commit=_REAL_PREREG_COMMIT,
        run_commit=_REAL_RUN_COMMIT,
    )
    licenses_out = tmp_path / "licenses_dynamic.json"
    ret = main(
        [
            "--prereg", str(prereg_path),
            "--reports", *[str(r) for r in report_paths],
            "--ledger", str(ledger_path),
            "--licenses-out", str(licenses_out),
        ]
    )
    assert ret == 0, (
        f"동적 SHA 로 CLI 실행 시 exit 0 기대. "
        f"preregCommit={_REAL_PREREG_COMMIT[:12]}, runCommit={_REAL_RUN_COMMIT[:12]}"
    )
    written = json.loads(licenses_out.read_text())
    assert written["rungs"]["M2"] is True
    assert "AUDIT LOG ONLY" in written["note"]


# ===========================================================================
# ITEM 1: AXS-010 vacuous strict_pass (CRITICAL)
# ===========================================================================


def test_redteam_axs010_empty_tiebreak_fails(tmp_path):
    """ITEM 1: tieBreak with empty baseline/variant objects → axs010_invariance fail, M2 False."""
    def _empty_tiebreak_axs010(prereg, p_hash, ps):
        r = {
            "reportId": "rep-axs010-v1",
            "experimentId": "AXS-010",
            "preregStamp": _build_stamp(prereg, p_hash, ps),
            "replayAudit": {"replayable": True},
            "tieBreak": {
                "baseline": {},
                "variants": {
                    "reverse": {},
                    "hash_seeded": {},
                    "feature_lexicographic": {},
                },
            },
        }
        return _finalize_report(r)

    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path, axs010_builder=_empty_tiebreak_axs010
    )

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "axs010_invariance" for c in failed), (
        "비어 있는 baseline/variant 객체는 axs010_invariance 실패를 유발해야 함"
    )
    assert result["rungs"]["M2"] is False


# ===========================================================================
# ITEM 2: Baseline omission disables degenerate policy
# ===========================================================================


def test_redteam_arms_complete_missing_baseline_coverage(tmp_path):
    """ITEM 2: AXS-003 with RANDOM baseline removed + low-coverage arm → arms_complete fail."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    # Remove baselines.RANDOM entirely
    del r["baselines"]["RANDOM"]
    # Set one arm to very low coverage (would be below RANDOM if RANDOM were present)
    r["arms"]["axs_ucb_alpha0"]["utility"]["coordinate_coverage_mean"] = 0.01
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    # Re-register in ledger
    ledger = load_ledger(ledger_path)
    new_entries = [e for e in ledger["entries"] if e.get("reportId") != r["reportId"]]
    ledger["entries"] = new_entries
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(ledger_path, r, prereg, p_hash, axs003_path)

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed), (
        "baselines.RANDOM 누락 시 arms_complete 이 실패해야 함"
    )


# ===========================================================================
# ITEM 3: Booleans/non-numerics consumed as metric values (CRITICAL)
# ===========================================================================


def test_redteam_boolean_per_family_fails(tmp_path):
    """ITEM 3a: perFamily value is True (boolean) → acceptance_recomputed fails."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    # Set 2 families' perFamily metric to boolean True — enough to push count below minConsistentFamilies=4
    r["arms"]["axs_ucb_default"]["perFamily"]["42"]["slate_excess_nmi"] = True
    r["arms"]["axs_ucb_default"]["perFamily"]["7"]["slate_excess_nmi"] = True
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    # Re-register in ledger
    ledger = load_ledger(ledger_path)
    new_entries = [e for e in ledger["entries"] if e.get("reportId") != r["reportId"]]
    ledger["entries"] = new_entries
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(ledger_path, r, prereg, p_hash, axs003_path)

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "acceptance_recomputed" for c in failed), (
        "perFamily 값이 bool(True)일 때 acceptance_recomputed 가 실패해야 함"
    )
    assert result["rungs"]["M2"] is False


def test_redteam_boolean_ci_lower_fails(tmp_path):
    """ITEM 3b: bootstrap.ciLower is True (boolean) → acceptance_recomputed fails."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    r = json.loads(axs003_path.read_text())
    # Set bootstrap ciLower to boolean True
    r["arms"]["axs_ucb_default"]["bootstrap"]["slate_excess_nmi"]["ciLower"] = True
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    axs003_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

    # Re-register in ledger
    ledger = load_ledger(ledger_path)
    new_entries = [e for e in ledger["entries"] if e.get("reportId") != r["reportId"]]
    ledger["entries"] = new_entries
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(ledger_path, r, prereg, p_hash, axs003_path)

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "acceptance_recomputed" for c in failed), (
        "bootstrap.ciLower 가 bool(True)일 때 acceptance_recomputed 가 실패해야 함"
    )
    assert result["rungs"]["M2"] is False


# ===========================================================================
# ITEM 4: Unknown experimentId laundered
# ===========================================================================


def test_redteam_unknown_experiment_id(tmp_path):
    """ITEM 4: Report with experimentId not in prereg experiments → arms_complete fail."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # Create a fake report with unknown experimentId
    fake_r = {
        "reportId": "rep-axs999-v1",
        "experimentId": "AXS-999",
        "preregStamp": _build_stamp(prereg, p_hash, str(prereg_path)),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
        "arms": {},
    }
    fake_r = _finalize_report(fake_r)
    fake_path = tmp_path / "rep-axs999-v1.json"
    fake_path.write_text(json.dumps(fake_r, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(ledger_path, fake_r, prereg, p_hash, fake_path)

    all_reports = list(report_paths) + [fake_path]
    result = evaluate_mechanism_license(
        prereg_path,
        all_reports,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed), (
        "알 수 없는 experimentId 는 arms_complete 실패를 유발해야 함"
    )


# ===========================================================================
# ITEM 5: Duplicate experimentId silent last-wins
# ===========================================================================


def test_redteam_duplicate_experiment_id(tmp_path):
    """ITEM 5: Two reports for same experimentId → arms_complete fail."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_full_set(tmp_path)

    # Find AXS-003 report and submit it twice
    axs003_path = next(r for r in report_paths if "axs003" in r.name)
    duplicate_reports = list(report_paths) + [axs003_path]

    result = evaluate_mechanism_license(
        prereg_path,
        duplicate_reports,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed), (
        "동일한 experimentId 가 중복 제출되면 arms_complete 이 실패해야 함"
    )


# ===========================================================================
# ITEM 6: requiresPass fail-open default
# ===========================================================================


def test_redteam_empty_requires_pass(tmp_path):
    """ITEM 6: prereg with claimTransitions.M2.requiresPass=[] → acceptance_recomputed fail."""
    real_prereg = load_prereg(_REAL_PREREG)
    bad_prereg = copy.deepcopy(real_prereg)
    bad_prereg["claimTransitions"]["M2"]["requiresPass"] = []

    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(
        tmp_path, prereg_override=bad_prereg
    )

    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "acceptance_recomputed" for c in failed), (
        "requiresPass 가 빈 목록이면 acceptance_recomputed 가 실패해야 함"
    )


# ===========================================================================
# ITEM 7: Non-dict report JSON
# ===========================================================================


def test_cli_non_dict_report_exit1(tmp_path):
    """ITEM 7: Report JSON containing [1,2,3] → exit 1, no traceback."""
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)

    # Write a JSON array as a report file
    bad_report = tmp_path / "bad_report.json"
    bad_report.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    import io
    import sys as _sys
    old_stderr = _sys.stderr
    _sys.stderr = io.StringIO()
    try:
        ret = main(
            [
                "--prereg", str(prereg_path),
                "--reports", str(bad_report),
                "--ledger", str(ledger_path),
                "--licenses-out", str(tmp_path / "licenses.json"),
            ]
        )
        stderr_output = _sys.stderr.getvalue()
    finally:
        _sys.stderr = old_stderr

    assert ret == 1, "비-dict JSON 리포트는 exit 1 을 반환해야 함"
    assert "Traceback" not in stderr_output, "트레이스백이 노출되어서는 안 됨"
