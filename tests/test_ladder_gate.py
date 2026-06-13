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
    """픽스처 10a: AXS-010 변형 부호 뒤집기 — 정직한 invariance fail 은 증거 결과.

    well-formed 리포트의 정직한 fail 은 검사 위반이 아니라 M2 강등:
    axs010_invariance ok=True (정직 fail 명시), 실패 검사 0건, M2=False.
    """
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
    assert failed == [], f"정직한 TB fail 이 검사 실패로 처리됨: {failed!r}"
    inv = next(c for c in result["checks"] if c["check"] == "axs010_invariance")
    assert inv["ok"] is True
    assert "정직" in inv["detail"]
    assert result["rungs"]["M2"] is False
    assert result["caveatRequired"] is False


def test_redteam_axs010_track_decision_change(tmp_path):
    """픽스처 10b: AXS-010 trackDecision 변경 — 정직한 invariance fail → M2 강등.

    검사 위반 아님: 실패 검사 0건, axs010_invariance ok=True, M2=False.
    """
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
    assert failed == [], f"정직한 TB fail 이 검사 실패로 처리됨: {failed!r}"
    inv = next(c for c in result["checks"] if c["check"] == "axs010_invariance")
    assert inv["ok"] is True
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


# ===========================================================================
# v3 prereg 픽스처 빌더 (G-022b-v3 / G-022d-v3)
# ===========================================================================

_REAL_PREREG_V3 = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v3_draft.json"

# v3 평가 패밀리 (v1 과 동일)
EVAL_FAMILIES_V3 = ["42", "7", "101", "2025", "31337"]

# v3.1 가드 상수 — 커밋된 dead-arm 캘리브레이션 산출물과 동일
V31_ABSOLUTE_FLOOR = 0.16964285714285715
V31_RHO = 0.70

# 프로브 7종 (probe-responsiveness 계약)
from echo_bench.experiments.e_leakage_diagnostic import EXPANDED_PROBE_SET as _V31_PROBES_T

V31_PROBES = list(_V31_PROBES_T)

# v3.1 5-브랜치 집합
V31_BRANCHES = {
    "integrity_fail",
    "both_supported",
    "imprint_only_supported",
    "noise_only_supported",
    "no_claim_m1_only",
}


def _v31_seq_hashes(
    *,
    arm_tag: str = "arm",
    families=None,
    collapse_families=(),
    probes=None,
) -> Dict[str, Dict[str, str]]:
    """slateSequenceHashes 합성 — {family: {probe: hash}}.

    collapse_families 에 포함된 패밀리는 모든 프로브 해시가 동일(프로브 비응답).
    """
    fams = families if families is not None else EVAL_FAMILIES_V3
    probe_names = probes if probes is not None else V31_PROBES
    out: Dict[str, Dict[str, str]] = {}
    for fam in fams:
        if fam in collapse_families:
            out[fam] = {p: f"h-{arm_tag}-{fam}-const" for p in probe_names}
        else:
            out[fam] = {p: f"h-{arm_tag}-{fam}-{p}" for p in probe_names}
    return out

# v3 등록 완료 prereg — 원본 draft 를 복사한 뒤 status 를 "registered" 로 변경한 버전
# (모든 정상-통과 픽스처는 이 사본을 사용)


def _make_v3_prereg_registered(tmp_path: Path) -> tuple[Path, Dict[str, Any]]:
    """v3 draft 를 tmp_path 로 복사 + status → 'registered' 로 변경 후 저장."""
    import copy as _copy

    v3_src = _REAL_PREREG_V3
    with open(v3_src, "r", encoding="utf-8") as fh:
        v3_data = json.load(fh)
    v3_reg = _copy.deepcopy(v3_data)
    v3_reg["status"] = "registered"

    prereg_path = tmp_path / "prereg_v3_registered.json"
    prereg_path.write_text(
        json.dumps(v3_reg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return prereg_path, v3_reg


def _v3_per_family(metric: str, value: float = 0.08) -> Dict[str, Any]:
    """v3 perFamily — 모든 5개 패밀리 양수값."""
    return {fam: {metric: value} for fam in EVAL_FAMILIES_V3}


def _v3_per_family_custom(metric: str, vals: Dict[str, float]) -> Dict[str, Any]:
    """v3 perFamily — 패밀리별 지정 값."""
    return {fam: {metric: vals[fam]} for fam in EVAL_FAMILIES_V3}


def _v3_bootstrap(metric: str, ci_low: float = 0.05) -> Dict[str, Any]:
    return {metric: {"mean": 0.08, "ciLower": ci_low, "ciUpper": 0.11}}


def _v3_build_stamp(
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


def _v3_build_imp_report(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    freeze1_vals: Optional[Dict[str, float]] = None,
    freeze_none_vals: Optional[Dict[str, float]] = None,
    random_cov: float = 0.50,
    arm_cov_freeze1: float = 0.60,
    arm_cov_quarter: float = 0.62,
    arm_cov_half: float = 0.64,
    arm_cov_none: float = 0.65,
    freeze1_hashes: Optional[Dict[str, Dict[str, str]]] = None,
    freeze1_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """AXS-IMP-001 리포트 빌더 (v3.1 — slateSequenceHashes 포함).

    freeze1_vals / freeze_none_vals: None 이면 기본 양수값(0.12/0.04).
    freeze1_hashes: freeze_at_1 의 slateSequenceHashes 오버라이드 (None → distinct).
    freeze1_flags: freeze_at_1 에 병합할 degenerate 트리플 등 추가 키.
    기본 coverage 값은 역할별 가드를 모두 통과한다:
      intermediate(quarter/half) ≥ rho × live(0.65)=0.455, key(0.60) ≥ floor.
    """
    metric = "slate_excess_nmi"
    if freeze1_vals is None:
        freeze1_vals = {fam: 0.12 for fam in EVAL_FAMILIES_V3}
    if freeze_none_vals is None:
        freeze_none_vals = {fam: 0.04 for fam in EVAL_FAMILIES_V3}

    # freeze_at_1 arm (key_manipulation)
    arm_freeze1: Dict[str, Any] = {
        "perFamily": _v3_per_family_custom(metric, freeze1_vals),
        "bootstrap": _v3_bootstrap(metric),
        "utility": {"coordinate_coverage_mean": arm_cov_freeze1},
        "slateSequenceHashes": (
            freeze1_hashes if freeze1_hashes is not None
            else _v31_seq_hashes(arm_tag="freeze_at_1")
        ),
    }
    if freeze1_flags:
        arm_freeze1.update(freeze1_flags)

    r = {
        "reportId": "rep-axs-imp-001-v1",
        "experimentId": "AXS-IMP-001",
        "preregStamp": _v3_build_stamp(prereg, p_hash, prereg_path_str),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": random_cov}},
        "arms": {
            "freeze_at_1": arm_freeze1,
            "freeze_at_quarter": {
                "perFamily": _v3_per_family(metric, 0.09),
                "bootstrap": _v3_bootstrap(metric),
                "utility": {"coordinate_coverage_mean": arm_cov_quarter},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="freeze_at_quarter"),
            },
            "freeze_at_half": {
                "perFamily": _v3_per_family(metric, 0.07),
                "bootstrap": _v3_bootstrap(metric),
                "utility": {"coordinate_coverage_mean": arm_cov_half},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="freeze_at_half"),
            },
            "freeze_none": {
                "perFamily": _v3_per_family_custom(metric, freeze_none_vals),
                "bootstrap": _v3_bootstrap(metric, ci_low=-0.01),
                "utility": {"coordinate_coverage_mean": arm_cov_none},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="freeze_none"),
            },
        },
    }
    return _finalize_report(r)


def _v3_build_noise_report(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    default_vals: Optional[Dict[str, float]] = None,
    yoked_vals: Optional[Dict[str, float]] = None,
    random_cov: float = 0.50,
    arm_cov_default: float = 0.60,
    arm_cov_yoked: float = 0.55,
    yoked_hashes: Optional[Dict[str, Dict[str, str]]] = None,
    yoked_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """AXS-NOISE-001 리포트 빌더 (v3.1 — slateSequenceHashes 포함)."""
    metric = "slate_excess_nmi"
    if default_vals is None:
        default_vals = {fam: 0.10 for fam in EVAL_FAMILIES_V3}
    if yoked_vals is None:
        yoked_vals = {fam: 0.02 for fam in EVAL_FAMILIES_V3}

    arm_default: Dict[str, Any] = {
        "perFamily": _v3_per_family_custom(metric, default_vals),
        "bootstrap": _v3_bootstrap(metric),
        "utility": {"coordinate_coverage_mean": arm_cov_default},
        "slateSequenceHashes": _v31_seq_hashes(arm_tag="axs_ucb_default"),
    }

    arm_yoked: Dict[str, Any] = {
        "perFamily": _v3_per_family_custom(metric, yoked_vals),
        "bootstrap": _v3_bootstrap(metric, ci_low=-0.03),
        "utility": {"coordinate_coverage_mean": arm_cov_yoked},
        "slateSequenceHashes": (
            yoked_hashes if yoked_hashes is not None
            else _v31_seq_hashes(arm_tag="axs_yoked_bonus")
        ),
    }
    if yoked_flags:
        arm_yoked.update(yoked_flags)

    r = {
        "reportId": "rep-axs-noise-001-v1",
        "experimentId": "AXS-NOISE-001",
        "preregStamp": _v3_build_stamp(prereg, p_hash, prereg_path_str),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": random_cov}},
        "arms": {
            "axs_ucb_default": arm_default,
            "axs_yoked_bonus": arm_yoked,
        },
    }
    return _finalize_report(r)


def _v3_build_tb001_report(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """AXS-TB-001 리포트 빌더 (strict_pass 또는 soft_pass)."""
    estimate_outside = 0.03 if not strict else 0.079
    r = {
        "reportId": "rep-axs-tb-001-v1",
        "experimentId": "AXS-TB-001",
        "preregStamp": _v3_build_stamp(prereg, p_hash, prereg_path_str),
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
                "reverse": {"sign": "+", "estimate": estimate_outside, "trackDecision": "S"},
                "hash_seeded": {"sign": "+", "estimate": 0.081, "trackDecision": "S"},
                "feature_lexicographic": {"sign": "+", "estimate": 0.078, "trackDecision": "S"},
            },
        },
    }
    return _finalize_report(r)


def _v3_build_alpha_report(
    prereg: Dict[str, Any],
    p_hash: str,
    prereg_path_str: str,
) -> Dict[str, Any]:
    """AXS-ALPHA-EXP 리포트 빌더 (구조만)."""
    metric = "slate_excess_nmi"
    r = {
        "reportId": "rep-axs-alpha-exp-v1",
        "experimentId": "AXS-ALPHA-EXP",
        "preregStamp": _v3_build_stamp(prereg, p_hash, prereg_path_str),
        "replayAudit": {"replayable": True},
        "arms": {
            "axs_ucb_default": {
                "perFamily": _v3_per_family(metric, 0.10),
                "utility": {"coordinate_coverage_mean": 0.60},
            },
            "axs_ucb_alpha0": {
                "perFamily": _v3_per_family(metric, 0.07),
                "utility": {"coordinate_coverage_mean": 0.58},
            },
        },
    }
    return _finalize_report(r)


def _v31_degenerate_triple(reason: str) -> Dict[str, Any]:
    """일관된(정직한) degenerate 트리플 — 가드 실패 arm 의 자기 보고."""
    return {
        "degenerate": True,
        "degenerateReason": reason,
        "includedInMechanismClaim": False,
    }


def _make_v3_prereg_mutated(tmp_path: Path, mutate_fn) -> tuple[Path, Dict[str, Any]]:
    """등록판 v3 prereg 를 만든 뒤 mutate_fn(prereg) 적용 (단일 위치 구조 빌더)."""
    import copy as _copy

    with open(_REAL_PREREG_V3, "r", encoding="utf-8") as fh:
        v3_data = json.load(fh)
    prereg = _copy.deepcopy(v3_data)
    prereg["status"] = "registered"
    mutate_fn(prereg)
    prereg_path = tmp_path / "prereg_v3_mutated.json"
    prereg_path.write_text(
        json.dumps(prereg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return prereg_path, prereg


def _build_v3_full_set(
    tmp_path: Path,
    *,
    tb001_strict: bool = True,
    include_alpha: bool = True,
    imp_kwargs: Optional[Dict[str, Any]] = None,
    noise_kwargs: Optional[Dict[str, Any]] = None,
    imp_report_override: Optional[Dict[str, Any]] = None,
    noise_report_override: Optional[Dict[str, Any]] = None,
    prereg_override: Optional[Dict[str, Any]] = None,
    include_tb: bool = True,
):
    """v3 전체 리포트 세트 + 원장 + prereg 복사본 생성.

    Returns: (prereg_path, report_paths, ledger_path, prereg_dict, p_hash)
    """
    if prereg_override is not None:
        prereg_path = tmp_path / "prereg_v3.json"
        prereg_path.write_text(
            json.dumps(prereg_override, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        prereg = prereg_override
    else:
        prereg_path, prereg = _make_v3_prereg_registered(tmp_path)

    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    # IMP 리포트
    if imp_report_override is not None:
        imp_report = imp_report_override
    else:
        imp_report = _v3_build_imp_report(prereg, p_hash, ps, **(imp_kwargs or {}))

    # NOISE 리포트
    if noise_report_override is not None:
        noise_report = noise_report_override
    else:
        noise_report = _v3_build_noise_report(prereg, p_hash, ps, **(noise_kwargs or {}))

    reports_data = [imp_report, noise_report]
    if include_tb:
        tb_report = _v3_build_tb001_report(prereg, p_hash, ps, strict=tb001_strict)
        reports_data.append(tb_report)
    if include_alpha:
        alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)
        reports_data.append(alpha_report)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in reports_data:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    return prereg_path, report_paths, ledger_path, prereg, p_hash


# ===========================================================================
# v3.1 정상 경로 테스트 — 5-브랜치 트리 + 트랙별 라이선스 + canonical sentences
# ===========================================================================


def test_v3_branch_both_supported_strict(tmp_path):
    """v3.1 정상 경로: branch=both_supported, 두 트랙 + 결합 M2 모두 True (strict_pass)."""
    prereg_path, report_paths, ledger_path, prereg, _ = _build_v3_full_set(
        tmp_path, tb001_strict=True
    )
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result.get("branch") == "both_supported", (
        f"branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2-IMP"] is True
    assert result["rungs"]["M2-NOISE"] is True
    assert result["rungs"]["M2"] is True, "결합 M2 는 True 여야 함"
    assert result["caveatRequired"] is False
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]
    cs = prereg["canonicalSentences"]
    assert result["licensedSentences"] == [
        cs["M-IMP"], cs["M-NOISE"], cs["M2-COMBINED"]
    ], f"licensedSentences 오류: {result['licensedSentences']!r}"


def test_v3_branch_both_supported_soft(tmp_path):
    """v3.1 정상 경로: branch=both_supported, M2=True, caveatRequired=True (soft_pass)."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, tb001_strict=False
    )
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    assert result.get("branch") == "both_supported"
    assert result["rungs"]["M2"] is True
    assert result["caveatRequired"] is True
    assert len(result["licensedSentences"]) == 3


def test_v31_key_arm_below_floor_demotes_track(tmp_path):
    """v3.1 (red-team ii): key arm coverage < floor (정직한 트리플 포함) → IMP 트랙 비지지.

    delta 는 통과하지만 역할별 가드 실패 → noise_only_supported, M2-IMP=False,
    integrity_fail 아님 (정직하게 보고된 가드 실패는 검사 실패가 아니라 트랙 강등).
    """
    prereg_path, report_paths, ledger_path, prereg, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={
            "arm_cov_freeze1": 0.10,  # floor(0.1696...) 미달
            "freeze1_flags": _v31_degenerate_triple(
                "freeze_at_1: 역할별 가드 실패 — coverage 0.10 < absoluteFloor"
            ),
        },
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "noise_only_supported", (
        f"branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2-IMP"] is False
    assert result["rungs"]["M2-NOISE"] is True
    assert result["rungs"]["M2"] is False
    cs = prereg["canonicalSentences"]
    assert result["licensedSentences"] == [cs["M-NOISE"]]
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]


def test_v3_branch_sign_inconsistency_no_claim(tmp_path):
    """v3: delta_imp AND delta_noise 모두 부호 일관성 3/5 → branch=no_claim_m1_only."""
    # 두 delta 모두 실패해야 no_claim_m1_only (한쪽만 실패 시 imprint_only_supported 또는 noise_only_supported)
    three_pos_imp = {fam: (0.05 if fam in ["42", "7", "101"] else -0.02) for fam in EVAL_FAMILIES_V3}
    baseline_vals = {fam: 0.0 for fam in EVAL_FAMILIES_V3}
    three_pos_noise = {fam: (0.05 if fam in ["42", "7", "101"] else -0.02) for fam in EVAL_FAMILIES_V3}
    yoked_baseline = {fam: 0.0 for fam in EVAL_FAMILIES_V3}

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    # delta_imp: freeze_at_1 = three_pos_imp, freeze_none = 0 → 3/5 양수 (fail)
    imp_report = _v3_build_imp_report(
        prereg, p_hash, ps,
        freeze1_vals=three_pos_imp,
        freeze_none_vals=baseline_vals,
    )
    # delta_noise: default = three_pos_noise, yoked = 0 → 3/5 양수 (fail)
    noise_report = _v3_build_noise_report(
        prereg, p_hash, ps,
        default_vals=three_pos_noise,
        yoked_vals=yoked_baseline,
    )
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "no_claim_m1_only", (
        f"두 delta 모두 3/5 일관성 실패 시 branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2"] is False
    assert result["rungs"]["M2-IMP"] is False
    assert result["rungs"]["M2-NOISE"] is False
    assert result["licensedSentences"] == []


def test_v3_branch_imprint_only_supported(tmp_path):
    """v3.1: delta_imp pass, delta_noise fail → branch=imprint_only_supported."""
    # delta_noise fail: default_vals 와 yoked_vals 를 동일하게 → delta = 0 → 양수 아님
    zero_vals = {fam: 0.05 for fam in EVAL_FAMILIES_V3}

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    noise_report = _v3_build_noise_report(
        prereg, p_hash, ps,
        default_vals=zero_vals,
        yoked_vals=zero_vals,  # delta = 0 → sign fail
    )
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "imprint_only_supported", (
        f"delta_noise fail 에서 branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2"] is False
    assert result["rungs"]["M2-IMP"] is True
    assert result["rungs"]["M2-NOISE"] is False
    assert result["licensedSentences"] == [prereg["canonicalSentences"]["M-IMP"]]


def test_v3_branch_noise_only_supported(tmp_path):
    """v3.1: delta_imp fail, delta_noise pass → branch=noise_only_supported."""
    # delta_imp fail: freeze_at_1 와 freeze_none 동일 → delta = 0
    zero_vals = {fam: 0.05 for fam in EVAL_FAMILIES_V3}

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(
        prereg, p_hash, ps,
        freeze1_vals=zero_vals,
        freeze_none_vals=zero_vals,  # delta = 0 → sign fail
    )
    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "noise_only_supported", (
        f"delta_imp fail 에서 branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2"] is False
    assert result["rungs"]["M2-IMP"] is False
    assert result["rungs"]["M2-NOISE"] is True
    assert result["licensedSentences"] == [prereg["canonicalSentences"]["M-NOISE"]]


# ===========================================================================
# v3 red-team 픽스처
# ===========================================================================


def test_v3_redteam_draft_status_prereg_fails(tmp_path):
    """red-team: status='design-draft' prereg + 완벽한 리포트 → prereg_status 실패.

    커밋된 prereg는 N7-7에서 'registered'로 승격됐으므로, 거부 로직 자체를
    검증하려면 status를 명시적으로 'design-draft'로 강등한 사본을 써야 한다
    (커밋 파일 상태에 의존하지 않는다).
    """
    import copy as _copy
    with open(_REAL_PREREG_V3, "r", encoding="utf-8") as fh:
        v3_draft = json.load(fh)

    # 거부 경로 검증: status를 design-draft로 강등 (등록 메타는 제거)
    v3_draft["status"] = "design-draft"
    v3_draft.pop("derivedFromDraftHash", None)
    v3_draft.pop("registeredAt", None)
    prereg_path = tmp_path / "prereg_v3_draft.json"
    prereg_path.write_text(
        json.dumps(v3_draft, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prereg = v3_draft
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "prereg_status" for c in failed), (
        "design-draft prereg 는 prereg_status 실패를 유발해야 함"
    )
    assert result["rungs"]["M2"] is False
    # branch 가 integrity_fail 이어야 함
    assert result.get("branch") == "integrity_fail"


def test_v3_redteam_version2_no_status_fails(tmp_path):
    """red-team: version=2 + status 키 없음 → prereg_status 실패."""
    import copy as _copy
    with open(_REAL_PREREG_V3, "r", encoding="utf-8") as fh:
        v3_data = json.load(fh)
    v2_no_status = _copy.deepcopy(v3_data)
    v2_no_status["version"] = 2
    v2_no_status.pop("status", None)  # status 제거

    prereg_path = tmp_path / "prereg_v2_nostatus.json"
    prereg_path.write_text(
        json.dumps(v2_no_status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    p_hash = prereg_hash(v2_no_status)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(v2_no_status, p_hash, ps)
    noise_report = _v3_build_noise_report(v2_no_status, p_hash, ps)
    tb_report = _v3_build_tb001_report(v2_no_status, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, v2_no_status, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "prereg_status" for c in failed), (
        "version>=2 에서 status 없으면 prereg_status 실패해야 함"
    )
    assert result["rungs"]["M2"] is False


def test_v3_redteam_version1_no_status_passes(tmp_path):
    """v1 prereg 에 status 키 없어도 prereg_status 는 통과 (하위 호환)."""
    # 기존 v1 테스트 세트 실행 — 이미 통과하지만, status 검사 확인
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    status_check = next(
        (c for c in result["checks"] if c["check"] == "prereg_status"), None
    )
    assert status_check is not None, "prereg_status 검사가 결과에 없음"
    assert status_check["ok"] is True, (
        f"v1 prereg 의 prereg_status 가 실패해서는 안 됨: {status_check['detail']}"
    )
    # v1 에서는 branch 없음 — v3.1 트랙 키/licensedSentences 도 없어야 함 (하위 호환)
    assert "branch" not in result, "v1 결과에 branch 가 있어서는 안 됨"
    assert "licensedSentences" not in result, (
        "v1 결과에 licensedSentences 가 있어서는 안 됨"
    )
    assert "M2-IMP" not in result["rungs"], "v1 rungs 에 M2-IMP 가 있어서는 안 됨"
    assert "M2-NOISE" not in result["rungs"], "v1 rungs 에 M2-NOISE 가 있어서는 안 됨"


def test_v3_redteam_arm_missing_from_imp(tmp_path):
    """red-team: AXS-IMP-001 에서 arm 누락 → arms_complete 실패."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    # freeze_at_quarter arm 제거
    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    del imp_report["arms"]["freeze_at_quarter"]
    # reportHash 재계산
    body = {k: v for k, v in imp_report.items() if k != "reportHash"}
    imp_report["reportHash"] = canonical_hash(body)

    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed), (
        "AXS-IMP-001 arm 누락 시 arms_complete 실패해야 함"
    )


def test_v3_redteam_family_set_mismatch(tmp_path):
    """red-team: freeze_at_1 에 5개 패밀리, freeze_none 에 4개 패밀리 → delta_imp 계산 실패."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    metric = "slate_excess_nmi"

    # 정상 빌드 후 freeze_none arm 의 perFamily 를 4개로 직접 교체
    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    imp_report["arms"]["freeze_none"]["perFamily"] = {
        fam: {metric: 0.04} for fam in EVAL_FAMILIES_V3 if fam != "31337"
    }
    body = {k: v for k, v in imp_report.items() if k != "reportHash"}
    imp_report["reportHash"] = canonical_hash(body)

    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    # 패밀리 집합 불일치 → acceptance_recomputed 실패 (delta 계산 오류)
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "acceptance_recomputed" for c in failed), (
        f"패밀리 집합 불일치 시 acceptance_recomputed 실패 해야 함. 실패 검사: {[c['check'] for c in failed]}"
    )
    assert result["rungs"]["M2"] is False


def test_v3_redteam_tampered_embedded_delta_immune(tmp_path):
    """red-team: 리포트에 내장된 delta_imp bootstrap 블록을 변조해도 게이트 결과가 동일해야 함."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_v3_full_set(tmp_path)

    # 정상 결과 먼저 계산
    result_clean = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )

    # IMP 리포트에 absurd 내장 delta 블록 삽입
    imp_path = next(rp for rp in report_paths if "imp" in rp.name)
    r = json.loads(imp_path.read_text())
    r["delta_imp"] = {
        "bootstrap": {"mean": 999.0, "ciLower": 999.0, "ciUpper": 999.0},
        "sign_count": 0,  # 의도적으로 잘못된 값
    }
    body = {k: v for k, v in r.items() if k != "reportHash"}
    r["reportHash"] = canonical_hash(body)
    imp_path.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    # 원장 업데이트: 기존 엔트리 제거 후 새 해시로 재등록
    ledger = load_ledger(ledger_path)
    new_entries = [e for e in ledger["entries"] if e.get("reportId") != r["reportId"]]
    ledger["entries"] = new_entries
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(ledger_path, r, prereg, p_hash, imp_path)

    result_tampered = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )

    # branch 와 M2 결과가 동일해야 함 (내장 delta 무시)
    assert result_tampered.get("branch") == result_clean.get("branch"), (
        f"내장 delta 변조 시 branch 변경됨: "
        f"정상={result_clean.get('branch')!r}, 변조={result_tampered.get('branch')!r}"
    )
    assert result_tampered["rungs"]["M2"] == result_clean["rungs"]["M2"], (
        "내장 delta 변조 시 M2 변경됨 — 재계산 면역성 위반"
    )


def test_v3_redteam_alpha_only_tb001_no_m2(tmp_path):
    """red-team: ALPHA-EXP + TB-001 만 제출 → M2 False, requiresPass IMP/NOISE 누락 명시."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [alpha_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result["rungs"]["M2"] is False
    failed = [c for c in result["checks"] if not c["ok"]]
    acc_fail = next((c for c in failed if c["check"] == "acceptance_recomputed"), None)
    assert acc_fail is not None, "acceptance_recomputed 실패 없음"
    detail = acc_fail["detail"]
    # requiresPass 에 IMP/NOISE 가 명시되어야 함
    assert "AXS-IMP-001" in detail or "AXS-NOISE-001" in detail, (
        f"requiresPass 누락 detail 에 IMP/NOISE 없음: {detail!r}"
    )


def test_v31_redteam_random_parity_marking_contradiction_fails(tmp_path):
    """red-team (iv-b): RANDOM-parity 부활 시도 — RANDOM 미달 기반 degenerate 마킹이
    재계산된 역할별 가드(통과)와 모순 → 검사 실패 (integrity_fail).

    freeze_at_1 coverage 0.40 은 RANDOM(0.50) 미달이지만 floor(0.1696...) 초과 +
    프로브 응답성 충족 → 가드 통과. 그런데 리포트가 RANDOM 기준으로 degenerate
    마킹 → 변조/오계산 리포트로 간주, fail-closed.
    """
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={
            "arm_cov_freeze1": 0.40,
            "freeze1_flags": _v31_degenerate_triple("RANDOM 기준 미달"),
        },
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed), (
        f"가드 모순 마킹이 role_specific_guard 실패를 유발해야 함. "
        f"실패 검사: {[c['check'] for c in failed]}"
    )
    assert result.get("branch") == "integrity_fail"
    assert result["rungs"]["M2"] is False
    assert result["licensedSentences"] == []


def test_v3_redteam_boolean_per_family_fails(tmp_path):
    """red-team: v3 perFamily 값이 boolean (True) → acceptance_recomputed 실패."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    # freeze_at_1 의 두 패밀리 perFamily 값을 boolean 으로 변조
    imp_report["arms"]["freeze_at_1"]["perFamily"]["42"]["slate_excess_nmi"] = True
    imp_report["arms"]["freeze_at_1"]["perFamily"]["7"]["slate_excess_nmi"] = True
    body = {k: v for k, v in imp_report.items() if k != "reportHash"}
    imp_report["reportHash"] = canonical_hash(body)

    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "acceptance_recomputed" for c in failed), (
        "boolean perFamily 값은 acceptance_recomputed 실패를 유발해야 함"
    )
    assert result["rungs"]["M2"] is False


def test_v3_redteam_sign_inconsistency_3of5(tmp_path):
    """red-team: delta_imp AND delta_noise 모두 부호 일관성 3/5 → no_claim_m1_only."""
    # 두 delta 가 모두 실패해야 no_claim_m1_only
    three_pos_imp = {fam: (0.05 if fam in ["42", "7", "101"] else -0.02) for fam in EVAL_FAMILIES_V3}
    baseline_vals = {fam: 0.0 for fam in EVAL_FAMILIES_V3}
    three_pos_noise = {fam: (0.05 if fam in ["42", "7", "101"] else -0.02) for fam in EVAL_FAMILIES_V3}

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(
        prereg, p_hash, ps,
        freeze1_vals=three_pos_imp,
        freeze_none_vals=baseline_vals,
    )
    noise_report = _v3_build_noise_report(
        prereg, p_hash, ps,
        default_vals=three_pos_noise,
        yoked_vals=baseline_vals,
    )
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "no_claim_m1_only"
    assert result["rungs"]["M2"] is False


def test_v3_cli_branch_in_output(tmp_path):
    """v3 CLI: licenses.json 출력에 branch 포함되어야 함."""
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    # 실제 커밋 SHA 를 stamp 에 주입
    def _v3_stamper():
        return {
            "preregId": prereg["preregId"],
            "preregVersion": prereg["version"],
            "preregPath": ps,
            "preregHash": p_hash,
            "preregCommit": _REAL_PREREG_COMMIT,
            "runCommit": _REAL_RUN_COMMIT,
        }

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    imp_report["preregStamp"] = _v3_stamper()
    body = {k: v for k, v in imp_report.items() if k != "reportHash"}
    imp_report["reportHash"] = canonical_hash(body)

    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    noise_report["preregStamp"] = _v3_stamper()
    body = {k: v for k, v in noise_report.items() if k != "reportHash"}
    noise_report["reportHash"] = canonical_hash(body)

    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    tb_report["preregStamp"] = _v3_stamper()
    body = {k: v for k, v in tb_report.items() if k != "reportHash"}
    tb_report["reportHash"] = canonical_hash(body)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    licenses_out = tmp_path / "v3_licenses.json"
    ret = main([
        "--prereg", str(prereg_path),
        "--reports", *[str(rp) for rp in report_paths],
        "--ledger", str(ledger_path),
        "--licenses-out", str(licenses_out),
    ])
    written = json.loads(licenses_out.read_text())
    assert "branch" in written, "v3 CLI 출력에 branch 키가 없음"
    assert written["branch"] in V31_BRANCHES, (
        f"알 수 없는 branch: {written['branch']!r}"
    )
    # v3.1 감사 로그 확장: 트랙별 런그 + licensedSentences
    assert "licensedSentences" in written, "licenses.json 에 licensedSentences 없음"
    assert "M2-IMP" in written["rungs"], "licenses.json rungs 에 M2-IMP 없음"
    assert "M2-NOISE" in written["rungs"], "licenses.json rungs 에 M2-NOISE 없음"


def test_v3_redteam_tb001_strict_m2_true(tmp_path):
    """v3: AXS-TB-001 strict_pass + 정상 delta → M2=True, caveatRequired=False."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, tb001_strict=True
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result["rungs"]["M2"] is True
    assert result["caveatRequired"] is False


def test_v3_redteam_tb001_soft_caveat_required(tmp_path):
    """v3: AXS-TB-001 soft_pass + 정상 delta → M2=True, caveatRequired=True."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, tb001_strict=False
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result["rungs"]["M2"] is True
    assert result["caveatRequired"] is True


# ===========================================================================
# AXS-V3 N3 리뷰 반영 — 4개 게이트 검사
# ===========================================================================


def test_n3_finding1_version_relabel_bypass_blocked(tmp_path):
    """Finding 1: v3 실험 구조 prereg + version=1 + status 없음 → prereg_status 실패.

    probe 4e 시나리오: status=deleted(또는 없음) + version:=1 로 grandfather 우회 시도.
    _is_v3_prereg() 가 True 이면 명시적 status=='registered' 필수.
    """
    import copy as _copy
    with open(_REAL_PREREG_V3, "r", encoding="utf-8") as fh:
        v3_data = json.load(fh)
    # version 을 1 로 낮추고 status 제거 → grandfather 조항 우회 시도
    v3_relabeled = _copy.deepcopy(v3_data)
    v3_relabeled["version"] = 1
    v3_relabeled.pop("status", None)

    prereg_path = tmp_path / "prereg_v3_relabeled.json"
    prereg_path.write_text(
        json.dumps(v3_relabeled, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    p_hash = prereg_hash(v3_relabeled)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(v3_relabeled, p_hash, ps)
    noise_report = _v3_build_noise_report(v3_relabeled, p_hash, ps)
    tb_report = _v3_build_tb001_report(v3_relabeled, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, v3_relabeled, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "prereg_status" for c in failed), (
        "v3 실험 구조 + version=1 + status 없음은 prereg_status 실패를 유발해야 함 "
        "(버전 위장 우회 차단)"
    )
    assert result["rungs"]["M2"] is False
    # 실패 상세에 우회 차단 메시지 포함되어야 함
    status_fail = next(c for c in failed if c["check"] == "prereg_status")
    assert "버전 위장" in status_fail["detail"], (
        f"prereg_status detail 에 '버전 위장' 없음: {status_fail['detail']!r}"
    )


def test_n3_finding1_version_relabel_bypass_status_deleted(tmp_path):
    """Finding 1 probe 4e: status='deleted' + version:=1 → prereg_status 실패.

    status 가 명시적으로 'registered' 가 아닌 값이면 기존 로직에서 먼저 실패해야 함.
    """
    import copy as _copy
    with open(_REAL_PREREG_V3, "r", encoding="utf-8") as fh:
        v3_data = json.load(fh)
    v3_relabeled = _copy.deepcopy(v3_data)
    v3_relabeled["version"] = 1
    v3_relabeled["status"] = "deleted"  # 명시적 비-registered

    prereg_path = tmp_path / "prereg_v3_deleted.json"
    prereg_path.write_text(
        json.dumps(v3_relabeled, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    p_hash = prereg_hash(v3_relabeled)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(v3_relabeled, p_hash, ps)
    noise_report = _v3_build_noise_report(v3_relabeled, p_hash, ps)
    tb_report = _v3_build_tb001_report(v3_relabeled, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, v3_relabeled, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "prereg_status" for c in failed), (
        "status='deleted' + v3 실험 구조는 prereg_status 실패를 유발해야 함"
    )
    assert result["rungs"]["M2"] is False


def test_n3_finding1_v1_non_v3_still_passes(tmp_path):
    """Finding 1 회귀: 비-v3 v1 prereg 는 status 없어도 여전히 grandfather 통과."""
    # 기존 v1 픽스처는 v3 실험 ID 없음 → grandfather 허용
    prereg_path, report_paths, ledger_path, _, _ = _build_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    status_check = next(
        (c for c in result["checks"] if c["check"] == "prereg_status"), None
    )
    assert status_check is not None
    assert status_check["ok"] is True, (
        f"비-v3 v1 prereg 는 prereg_status 통과해야 함: {status_check['detail']}"
    )


def test_n3_finding2_secondary_delta_incomputable_disclosed(tmp_path):
    """Finding 2: freeze_at_quarter perFamily 값 오염 시 delta_q 불가 → 한국어 공개 detail.

    M2 는 기본 delta_imp/delta_noise 로 결정되므로 이차 불가는 M2 미변경.
    """
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    # freeze_at_quarter perFamily 하나를 비-숫자(None)로 오염 → delta_q 계산 불가
    imp_report["arms"]["freeze_at_quarter"]["perFamily"]["42"]["slate_excess_nmi"] = None
    body = {k: v for k, v in imp_report.items() if k != "reportHash"}
    imp_report["reportHash"] = canonical_hash(body)

    noise_report = _v3_build_noise_report(prereg, p_hash, ps)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    # M2 는 delta_imp(freeze_at_1 vs freeze_none) 로 결정 → 이차 불가는 M2 미변경
    assert result["rungs"]["M2"] is True, (
        "이차 delta_q 계산 불가는 M2 를 변경해서는 안 됨"
    )
    # acceptance_recomputed detail 에 delta_q 공개 메시지 포함
    acc_check = next(c for c in result["checks"] if c["check"] == "acceptance_recomputed")
    assert "delta_q 재계산 불가" in acc_check["detail"], (
        f"delta_q 재계산 불가 공개 메시지가 없음: {acc_check['detail']!r}"
    )


def test_n3_finding3_yoked_absence_probe_b(tmp_path):
    """Finding 3 probe B: yoked sep +0.50 in 5/5, default 0.40 → yokedAbsence 미확인.

    yoked arm 자체의 분리성이 확인되면 yokedAbsence = False → '미확인' 공개.
    """
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    metric = "slate_excess_nmi"
    # yoked_bonus: 5/5 양수 + ciLower > 0 → separability present → yokedAbsence = False
    yoked_sep_vals = {fam: 0.50 for fam in EVAL_FAMILIES_V3}
    # default: 0.40 in 5/5 → delta = default - yoked = 0.40 - 0.50 = -0.10 (delta_noise fail)
    default_sep_vals = {fam: 0.40 for fam in EVAL_FAMILIES_V3}

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)

    # noise 리포트 직접 생성 — yoked bootstrap ciLower > 0
    r_noise = {
        "reportId": "rep-axs-noise-001-v1",
        "experimentId": "AXS-NOISE-001",
        "preregStamp": _v3_build_stamp(prereg, p_hash, ps),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.50}},
        "arms": {
            "axs_ucb_default": {
                "perFamily": {fam: {metric: default_sep_vals[fam]} for fam in EVAL_FAMILIES_V3},
                "bootstrap": {metric: {"mean": 0.40, "ciLower": 0.35, "ciUpper": 0.45}},
                "utility": {"coordinate_coverage_mean": 0.60},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="axs_ucb_default"),
            },
            "axs_yoked_bonus": {
                "perFamily": {fam: {metric: yoked_sep_vals[fam]} for fam in EVAL_FAMILIES_V3},
                "bootstrap": {metric: {"mean": 0.50, "ciLower": 0.40, "ciUpper": 0.60}},
                "utility": {"coordinate_coverage_mean": 0.55},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="axs_yoked_bonus"),
            },
        },
    }
    r_noise = _finalize_report(r_noise)

    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, r_noise, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    acc_check = next(c for c in result["checks"] if c["check"] == "acceptance_recomputed")
    # yoked arm 이 separability present → yokedAbsence = False → '미확인'
    assert "yokedAbsence 이차: 미확인" in acc_check["detail"], (
        f"yoked sep 5/5 양수 시 yokedAbsence '미확인' 이어야 함: {acc_check['detail']!r}"
    )


def test_n3_finding3_yoked_absence_confirmed(tmp_path):
    """Finding 3 대칭: yoked sep 0/5 + ciLower <= 0 → yokedAbsence 확인."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    metric = "slate_excess_nmi"
    # yoked_bonus: 0/5 양수 (모두 음수) + ciLower <= 0 → yokedAbsence = True
    yoked_absent_vals = {fam: -0.02 for fam in EVAL_FAMILIES_V3}
    # default: 5/5 양수
    default_present_vals = {fam: 0.10 for fam in EVAL_FAMILIES_V3}

    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    r_noise = {
        "reportId": "rep-axs-noise-001-v1",
        "experimentId": "AXS-NOISE-001",
        "preregStamp": _v3_build_stamp(prereg, p_hash, ps),
        "replayAudit": {"replayable": True},
        "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.50}},
        "arms": {
            "axs_ucb_default": {
                "perFamily": {fam: {metric: default_present_vals[fam]} for fam in EVAL_FAMILIES_V3},
                "bootstrap": {metric: {"mean": 0.10, "ciLower": 0.05, "ciUpper": 0.15}},
                "utility": {"coordinate_coverage_mean": 0.60},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="axs_ucb_default"),
            },
            "axs_yoked_bonus": {
                "perFamily": {fam: {metric: yoked_absent_vals[fam]} for fam in EVAL_FAMILIES_V3},
                "bootstrap": {metric: {"mean": -0.02, "ciLower": -0.05, "ciUpper": 0.00}},
                "utility": {"coordinate_coverage_mean": 0.55},
                "slateSequenceHashes": _v31_seq_hashes(arm_tag="axs_yoked_bonus"),
            },
        },
    }
    r_noise = _finalize_report(r_noise)

    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, r_noise, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    acc_check = next(c for c in result["checks"] if c["check"] == "acceptance_recomputed")
    assert "yokedAbsence 이차: 확인" in acc_check["detail"], (
        f"yoked sep 0/5 양수 시 yokedAbsence '확인' 이어야 함: {acc_check['detail']!r}"
    )


def test_n3_finding4_cli_no_claim_branch_named(tmp_path, capsys):
    """Finding 4: no_claim_m1_only 브랜치에서 '증거 충분' 미출력, branch 이름 명시.

    모든 검사 통과하지만 M2=False (no_claim_m1_only) → 요약에 branch 표시.
    실제 git SHA 사용 (없으면 skip).
    """
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")

    # delta_imp, delta_noise 모두 실패 → no_claim_m1_only (all checks ok, M2 False)
    three_pos = {fam: (0.05 if fam in ["42", "7", "101"] else -0.02) for fam in EVAL_FAMILIES_V3}
    baseline = {fam: 0.0 for fam in EVAL_FAMILIES_V3}

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    def _stamped(r: Dict[str, Any]) -> Dict[str, Any]:
        r["preregStamp"] = _v3_build_stamp(
            prereg, p_hash, ps,
            prereg_commit=_REAL_PREREG_COMMIT,
            run_commit=_REAL_RUN_COMMIT,
        )
        body = {k: v for k, v in r.items() if k != "reportHash"}
        r["reportHash"] = canonical_hash(body)
        return r

    imp_report = _stamped(_v3_build_imp_report(prereg, p_hash, ps, freeze1_vals=three_pos, freeze_none_vals=baseline))
    noise_report = _stamped(_v3_build_noise_report(prereg, p_hash, ps, default_vals=three_pos, yoked_vals=baseline))
    tb_report = _stamped(_v3_build_tb001_report(prereg, p_hash, ps))
    alpha_report = _stamped(_v3_build_alpha_report(prereg, p_hash, ps))

    ledger_path = _make_ledger(tmp_path)
    licenses_out = tmp_path / "licenses_noclaim.json"
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    ret = main([
        "--prereg", str(prereg_path),
        "--reports", *[str(rp) for rp in report_paths],
        "--ledger", str(ledger_path),
        "--licenses-out", str(licenses_out),
    ])
    captured = capsys.readouterr()
    stdout = captured.out
    # exit code: 0 (검사 통과, M2 는 exit code 에 영향 없음)
    assert ret == 0, f"모든 검사 통과 시 exit 0 이어야 함. stdout:\n{stdout}"
    # branch 이름이 stdout 에 포함되어야 함
    assert "no_claim_m1_only" in stdout, (
        f"stdout 에 branch 이름 'no_claim_m1_only' 없음:\n{stdout}"
    )
    # M2=False 인 경우 '증거 충분' 단독 줄 미출력
    lines = stdout.splitlines()
    assert not any(line.strip() == "ladder_gate 심사 완료: 증거 충분" for line in lines), (
        f"M2=False 시 '증거 충분' 단독 줄이 출력되어서는 안 됨:\n{stdout}"
    )


def test_n3_finding4_cli_no_claim_no_suffix_without_git(tmp_path, capsys):
    """Finding 4 (git-free): evaluate_mechanism_license 결과 + CLI 출력 형식 검증.

    git_runner 주입으로 ancestry 통과 후 main() stdout 대신 결과를 직접 검증.
    """
    three_pos = {fam: (0.05 if fam in ["42", "7", "101"] else -0.02) for fam in EVAL_FAMILIES_V3}
    baseline = {fam: 0.0 for fam in EVAL_FAMILIES_V3}

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    imp_report = _v3_build_imp_report(prereg, p_hash, ps, freeze1_vals=three_pos, freeze_none_vals=baseline)
    noise_report = _v3_build_noise_report(prereg, p_hash, ps, default_vals=three_pos, yoked_vals=baseline)
    tb_report = _v3_build_tb001_report(prereg, p_hash, ps)
    alpha_report = _v3_build_alpha_report(prereg, p_hash, ps)

    ledger_path = _make_ledger(tmp_path)
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    # git_runner 주입으로 ancestry 통과
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    # no_claim_m1_only: 모든 검사 ok, M2=False, branch='no_claim_m1_only'
    assert result.get("branch") == "no_claim_m1_only", (
        f"branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2"] is False
    # branch 결과 확인 — CLI 출력 형식 변경은 Finding 4 구현에 포함됨


def test_n3_finding4_cli_both_supported_증거충분(tmp_path, capsys):
    """Finding 4 대칭: both_supported 에서 '증거 충분' + v3.1 요약 줄 출력 확인 (회귀 방지).

    실제 git SHA 사용 (없으면 skip).
    """
    if _REAL_PREREG_COMMIT is None or _REAL_RUN_COMMIT is None:
        pytest.skip("git SHA 동적 해석 실패 — git checkout 이 아닌 환경에서는 건너뜀")

    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    def _stamped(r: Dict[str, Any]) -> Dict[str, Any]:
        r["preregStamp"] = _v3_build_stamp(
            prereg, p_hash, ps,
            prereg_commit=_REAL_PREREG_COMMIT,
            run_commit=_REAL_RUN_COMMIT,
        )
        body = {k: v for k, v in r.items() if k != "reportHash"}
        r["reportHash"] = canonical_hash(body)
        return r

    imp_report = _stamped(_v3_build_imp_report(prereg, p_hash, ps))
    noise_report = _stamped(_v3_build_noise_report(prereg, p_hash, ps))
    tb_report = _stamped(_v3_build_tb001_report(prereg, p_hash, ps, strict=True))
    alpha_report = _stamped(_v3_build_alpha_report(prereg, p_hash, ps))

    ledger_path = _make_ledger(tmp_path)
    licenses_out = tmp_path / "licenses_ws.json"
    report_paths = []
    for r in [imp_report, noise_report, tb_report, alpha_report]:
        rp = tmp_path / f"{r['reportId']}.json"
        rp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_report(ledger_path, r, prereg, p_hash, rp)
        report_paths.append(rp)

    ret = main([
        "--prereg", str(prereg_path),
        "--reports", *[str(rp) for rp in report_paths],
        "--ledger", str(ledger_path),
        "--licenses-out", str(licenses_out),
    ])
    captured = capsys.readouterr()
    stdout = captured.out
    assert ret == 0, f"both_supported exit 0 기대. stdout:\n{stdout}"
    assert "증거 충분" in stdout, (
        f"both_supported + M2=True 에서 '증거 충분' 출력 없음:\n{stdout}"
    )
    # v3.1 CLI 요약 줄 계약
    assert (
        "심사 완료: branch=both_supported, M2-IMP=True, M2-NOISE=True, M2(combined)=True"
        in stdout
    ), f"v3.1 요약 줄 없음:\n{stdout}"


# ===========================================================================
# N7-3 v3.1 — 역할별 가드 red-team 픽스처 (advisor-mandated)
# ===========================================================================


def test_v31_redteam_key_arm_parity_fail_but_floor_pass_track_eligible(tmp_path):
    """red-team (i): key arm 이 parity-equivalent ratio 실패(0.5×live)지만
    floor + 프로브 응답성 통과 → 가드 통과, 트랙 자격 유지 (both_supported)."""
    live_cov = 0.65
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={"arm_cov_freeze1": 0.5 * live_cov, "arm_cov_none": live_cov},
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "both_supported", (
        f"parity 미달(0.325)이지만 floor(0.1696...) 초과인 key arm 은 가드를 통과해야 함. "
        f"branch={result.get('branch')!r}"
    )
    assert result["rungs"]["M2-IMP"] is True
    assert result["rungs"]["M2"] is True


def test_v31_redteam_alpha_in_requires_pass_fails(tmp_path):
    """red-team (iii-a): claimTransitions M2-트랙 requiresPass 에 AXS-ALPHA-EXP 포함 → 실패."""
    def _mutate(p):
        p["claimTransitions"]["M2-IMP"]["requiresPass"] = [
            "AXS-IMP-001", "AXS-ALPHA-EXP",
        ]
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "alpha_m2_prohibition" for c in failed), (
        f"ALPHA 가 M2-트랙 requiresPass 에 있으면 실패해야 함: {[c['check'] for c in failed]}"
    )
    assert result.get("branch") == "integrity_fail"
    assert result["rungs"]["M2"] is False
    assert result["licensedSentences"] == []


def test_v31_redteam_m2prohibited_stripped_fails(tmp_path):
    """red-team (iii-b): experiments.AXS-ALPHA-EXP.m2Prohibited 제거 → 실패."""
    def _mutate(p):
        p["experiments"]["AXS-ALPHA-EXP"].pop("m2Prohibited", None)
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "alpha_m2_prohibition" for c in failed)
    assert result.get("branch") == "integrity_fail"

    # m2Prohibited=False 위장도 동일하게 실패
    def _mutate_false(p):
        p["experiments"]["AXS-ALPHA-EXP"]["m2Prohibited"] = False
    (tmp_path / "f").mkdir(exist_ok=True)
    prereg_path2, prereg2 = _make_v3_prereg_mutated(tmp_path / "f", _mutate_false)
    _, report_paths2, ledger_path2, _, _ = _build_v3_full_set(
        tmp_path / "f", prereg_override=prereg2
    )
    result2 = evaluate_mechanism_license(
        prereg_path2, report_paths2, ledger_path=ledger_path2,
        git_runner=_good_git_runner,
    )
    failed2 = [c for c in result2["checks"] if not c["ok"]]
    assert any(c["check"] == "alpha_m2_prohibition" for c in failed2)


def test_v31_redteam_prereg_missing_arm_roles_fails(tmp_path):
    """red-team (iv-a): utilityGuard.armRoles 제거 (RANDOM-parity 부활 시도) → 실패."""
    def _mutate(p):
        p["utilityGuard"].pop("armRoles", None)
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed), (
        f"armRoles 누락 prereg 는 role_specific_guard 실패해야 함: {[c['check'] for c in failed]}"
    )
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_prereg_missing_key_rule_fails(tmp_path):
    """red-team (iv-a'): utilityGuard.keyManipulationRule 제거 → 실패."""
    def _mutate(p):
        p["utilityGuard"].pop("keyManipulationRule", None)
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_missing_slate_sequence_hashes_key_arm_fails(tmp_path):
    """red-team: key arm 에 slateSequenceHashes 자체가 없음 → 검사 실패 (구조 위반)."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)
    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    del imp_report["arms"]["freeze_at_1"]["slateSequenceHashes"]
    imp_report = _finalize_report(
        {k: v for k, v in imp_report.items() if k != "reportHash"}
    )
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg, imp_report_override=imp_report
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed), (
        f"key arm slateSequenceHashes 누락은 검사 실패여야 함: {[c['check'] for c in failed]}"
    )
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_collapsed_family_consistent_demotes_track(tmp_path):
    """red-team: 한 패밀리의 해시 전부 동일(프로브 비응답) + 정직한 트리플 →
    해당 arm 가드 실패 → 트랙 강등 (noise_only_supported), 검사 실패 아님."""
    prereg_path, report_paths, ledger_path, prereg, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={
            "freeze1_hashes": _v31_seq_hashes(
                arm_tag="freeze_at_1", collapse_families=("101",)
            ),
            "freeze1_flags": _v31_degenerate_triple(
                "freeze_at_1: 프로브 비응답 — 패밀리 101 에서 slateSequenceHashes 고유값 <2"
            ),
        },
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]
    assert result.get("branch") == "noise_only_supported"
    assert result["rungs"]["M2-IMP"] is False
    assert result["licensedSentences"] == [prereg["canonicalSentences"]["M-NOISE"]]


def test_v31_redteam_collapsed_family_unmarked_contradiction_fails(tmp_path):
    """red-team: 프로브 비응답인데 degenerate 미마킹 → 모순 → 검사 실패."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={
            "freeze1_hashes": _v31_seq_hashes(
                arm_tag="freeze_at_1", collapse_families=("101",)
            ),
            # flags 없음 — 자기 보고와 재계산 모순
        },
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_below_floor_unmarked_contradiction_fails(tmp_path):
    """red-team: floor 미달인데 degenerate 미마킹 → 모순 → 검사 실패."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={"arm_cov_freeze1": 0.10},  # floor 미달, flags 없음
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_wrong_family_key_set_fails(tmp_path):
    """red-team: key arm slateSequenceHashes 패밀리 키가 평가 집합과 다름 → 실패."""
    four_fams = [f for f in EVAL_FAMILIES_V3 if f != "31337"]
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={
            "freeze1_hashes": _v31_seq_hashes(
                arm_tag="freeze_at_1", families=four_fams
            ),
        },
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_probe_key_set_mismatch_fails(tmp_path):
    """red-team: 한 패밀리의 probe 키 집합이 다른 패밀리와 불일치 → 실패."""
    hashes = _v31_seq_hashes(arm_tag="freeze_at_1")
    # 패밀리 "7" 의 첫 probe 키를 다른 이름으로 교체 (7개 수는 유지)
    first_probe = V31_PROBES[0]
    hashes["7"]["WRONG_PROBE_NAME"] = hashes["7"].pop(first_probe)
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, imp_kwargs={"freeze1_hashes": hashes}
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_wrong_probe_count_fails(tmp_path):
    """red-team: probe 키가 7개가 아님 → 실패."""
    hashes = _v31_seq_hashes(arm_tag="freeze_at_1")
    for fam in hashes:
        hashes[fam].pop(V31_PROBES[0])  # 6개로 축소 (집합은 패밀리 간 일치)
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, imp_kwargs={"freeze1_hashes": hashes}
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)


def test_v31_redteam_floor_mismatch_fails(tmp_path):
    """red-team: prereg absoluteFloor 이 캘리브레이션 산출물과 불일치 → 실패."""
    def _mutate(p):
        p["utilityGuard"]["keyManipulationRule"]["absoluteFloor"] = 0.25
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_corrupt_artifact_fails(tmp_path, monkeypatch):
    """red-team: 고정 경로의 캘리브레이션 산출물 본문 변조(summaryHash 자기검증 실패) → 실패.

    경로 고정 이후에도 자기검증 sub-check 가 살아 있는지 _REPO_ROOT 대체로 확인.
    """
    import echo_bench.tools.ladder_gate as lg

    artifact_src = _REPO_ROOT / "configs" / "prereg" / "axs_dead_calibration_v1.json"
    with open(artifact_src, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    artifact["floorDerivation"]["maxCoverageDead"] = 0.001  # 본문 변조 (해시 미갱신)
    fake_root = tmp_path / "fakeroot"
    pinned = fake_root / "configs" / "prereg" / "axs_dead_calibration_v1.json"
    pinned.parent.mkdir(parents=True)
    pinned.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(lg, "_REPO_ROOT", fake_root)

    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_missing_artifact_fails(tmp_path, monkeypatch):
    """red-team: 고정 경로에 캘리브레이션 산출물이 존재하지 않음 → 실패."""
    import echo_bench.tools.ladder_gate as lg

    fake_root = tmp_path / "emptyroot"
    fake_root.mkdir()
    monkeypatch.setattr(lg, "_REPO_ROOT", fake_root)

    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)


def test_v31_redteam_cited_summary_hash_mismatch_fails(tmp_path):
    """red-team: prereg 인용 summaryHash 만 변조 → 인용 일치 sub-check 실패."""
    def _mutate(p):
        p["utilityGuard"]["keyManipulationRule"]["floorCalibration"]["summaryHash"] = (
            "0" * 64
        )
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_rho_mismatch_fails(tmp_path):
    """red-team: prereg rho 가 하드코딩 0.70 과 불일치 → 실패 (제네릭 해석기 금지)."""
    def _mutate(p):
        p["utilityGuard"]["controlIntermediateRule"]["rho"] = 0.5
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)


def test_v31_redteam_armroles_mismatch_fails(tmp_path):
    """red-team: prereg armRoles 가 하드코딩 매핑과 불일치 → 실패."""
    def _mutate(p):
        p["utilityGuard"]["armRoles"]["AXS-IMP-001"]["freeze_at_half"] = "key_manipulation"
    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_random_reference_only_contributes_no_decision(tmp_path):
    """v3.1: RANDOM coverage 가 모든 arm 보다 높아도(0.99) parity 미적용 → both_supported.

    RANDOM-parity 가 어떤 형태로든 부활하면 이 테스트가 잡는다 (reference-only).
    """
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path,
        imp_kwargs={"random_cov": 0.99},
        noise_kwargs={"random_cov": 0.99},
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert result.get("branch") == "both_supported", (
        f"RANDOM 이 결정에 기여함 (parity 부활 의심): branch={result.get('branch')!r}, "
        f"실패: {[c['check'] for c in result['checks'] if not c['ok']]}"
    )
    assert result["rungs"]["M2"] is True


def test_v31_random_baseline_missing_still_fails(tmp_path):
    """v3.1: baselines.RANDOM 누락 → arms_complete 실패 (reference 필수, 결정 비기여)."""
    prereg_path, prereg = _make_v3_prereg_registered(tmp_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)
    imp_report = _v3_build_imp_report(prereg, p_hash, ps)
    del imp_report["baselines"]["RANDOM"]
    imp_report = _finalize_report(
        {k: v for k, v in imp_report.items() if k != "reportHash"}
    )
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg, imp_report_override=imp_report
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "arms_complete" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_tb_missing_fails(tmp_path):
    """v3.1: AXS-TB-001 리포트 누락 → acceptance_recomputed 실패 (requiresPass 관례)."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, include_tb=False
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    acc_fail = next((c for c in failed if c["check"] == "acceptance_recomputed"), None)
    assert acc_fail is not None, "TB 누락 시 acceptance_recomputed 실패해야 함"
    assert "AXS-TB-001" in acc_fail["detail"]
    assert result.get("branch") == "integrity_fail"
    assert result["rungs"]["M2"] is False


def test_v31_alpha_present_alters_nothing(tmp_path):
    """v3.1: AXS-ALPHA-EXP 리포트 존재 여부가 branch/rungs/licensedSentences 를 바꾸지 않음."""
    with_alpha_dir = tmp_path / "with_alpha"
    with_alpha_dir.mkdir()
    no_alpha_dir = tmp_path / "no_alpha"
    no_alpha_dir.mkdir()

    p1, rp1, l1, _, _ = _build_v3_full_set(with_alpha_dir, include_alpha=True)
    p2, rp2, l2, _, _ = _build_v3_full_set(no_alpha_dir, include_alpha=False)

    r_with = evaluate_mechanism_license(
        p1, rp1, ledger_path=l1, git_runner=_good_git_runner
    )
    r_without = evaluate_mechanism_license(
        p2, rp2, ledger_path=l2, git_runner=_good_git_runner
    )
    assert r_with["branch"] == r_without["branch"] == "both_supported"
    assert r_with["rungs"] == r_without["rungs"]
    assert r_with["licensedSentences"] == r_without["licensedSentences"]


def test_v31_result_shape_contract(tmp_path):
    """v3.1 결과 형태 계약 (스캐너 팀 연동): rungs 키, branch, licensedSentences."""
    prereg_path, report_paths, ledger_path, _, _ = _build_v3_full_set(tmp_path)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert set(result["rungs"].keys()) == {"M0", "M1", "M2", "M2-IMP", "M2-NOISE", "M3"}
    for key in ("M2", "M2-IMP", "M2-NOISE"):
        assert isinstance(result["rungs"][key], bool)
    assert result["branch"] in V31_BRANCHES
    assert isinstance(result["licensedSentences"], list)
    for s in result["licensedSentences"]:
        assert isinstance(s, str) and s.strip()
    # 기존 키 유지
    for key in ("checks", "caveatRequired", "consumedReports", "familyHistory", "provisional"):
        assert key in result


def test_v31_self_probe_end_to_end_degradation(tmp_path):
    """적대적 자기 점검: both_supported 도달 후 원시값 하나씩 변조 →
    브랜치/라이선스가 명세대로 정확히 강등되는지 확인.

    (a) key-arm coverage < floor (정직 트리플) → noise_only_supported, [M-NOISE]
    (b) 한 패밀리 해시 붕괴 (정직 트리플)     → noise_only_supported, [M-NOISE]
    (c) TB 변형 부호 뒤집기 (strict → fail)   → no_claim_m1_only, []
        (정직한 invariance fail 은 증거 결과 — 검사 위반 아님, 두 트랙 강등)
    """
    # 기준: both_supported
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    p0, rp0, l0, prereg0, _ = _build_v3_full_set(base_dir)
    r0 = evaluate_mechanism_license(p0, rp0, ledger_path=l0, git_runner=_good_git_runner)
    assert r0["branch"] == "both_supported"
    assert len(r0["licensedSentences"]) == 3
    cs = prereg0["canonicalSentences"]

    # (a) key-arm coverage 미달
    dir_a = tmp_path / "mut_a"
    dir_a.mkdir()
    pa, rpa, la, prereg_a, _ = _build_v3_full_set(
        dir_a,
        imp_kwargs={
            "arm_cov_freeze1": 0.10,
            "freeze1_flags": _v31_degenerate_triple(
                "freeze_at_1: 역할별 가드 실패 — coverage 0.10 < absoluteFloor"
            ),
        },
    )
    ra = evaluate_mechanism_license(pa, rpa, ledger_path=la, git_runner=_good_git_runner)
    assert ra["branch"] == "noise_only_supported", f"(a) branch={ra['branch']!r}"
    assert ra["rungs"] == {**ra["rungs"], "M2-IMP": False, "M2-NOISE": True, "M2": False}
    assert ra["licensedSentences"] == [cs["M-NOISE"]]

    # (b) 한 패밀리 해시 붕괴
    dir_b = tmp_path / "mut_b"
    dir_b.mkdir()
    pb, rpb, lb, prereg_b, _ = _build_v3_full_set(
        dir_b,
        imp_kwargs={
            "freeze1_hashes": _v31_seq_hashes(
                arm_tag="freeze_at_1", collapse_families=("2025",)
            ),
            "freeze1_flags": _v31_degenerate_triple(
                "freeze_at_1: 프로브 비응답 — 패밀리 2025 에서 slateSequenceHashes 고유값 <2"
            ),
        },
    )
    rb = evaluate_mechanism_license(pb, rpb, ledger_path=lb, git_runner=_good_git_runner)
    assert rb["branch"] == "noise_only_supported", f"(b) branch={rb['branch']!r}"
    assert rb["licensedSentences"] == [cs["M-NOISE"]]

    # (c) TB strict → fail (변형 부호 뒤집기)
    dir_c = tmp_path / "mut_c"
    dir_c.mkdir()
    pc, rpc, lc, prereg_c, p_hash_c = _build_v3_full_set(dir_c)
    tb_path = next(rp for rp in rpc if "tb" in rp.name)
    tb = json.loads(tb_path.read_text())
    tb["tieBreak"]["variants"]["reverse"]["sign"] = "-"
    tb = _finalize_report({k: v for k, v in tb.items() if k != "reportHash"})
    tb_path.write_text(json.dumps(tb, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(lc, tb, prereg_c, p_hash_c, tb_path)

    rc = evaluate_mechanism_license(pc, rpc, ledger_path=lc, git_runner=_good_git_runner)
    assert rc["branch"] == "no_claim_m1_only", f"(c) branch={rc['branch']!r}"
    assert all(c["ok"] for c in rc["checks"]), [
        c for c in rc["checks"] if not c["ok"]
    ]
    assert rc["rungs"]["M2-IMP"] is False
    assert rc["rungs"]["M2-NOISE"] is False
    assert rc["rungs"]["M2"] is False
    assert rc["licensedSentences"] == []


# ===========================================================================
# 리뷰어 수정 픽스처 (Team A) — TB 정직 fail 강등 / 알파 엔트리 삭제 우회 /
# 캘리브레이션 산출물 경로 고정
# ===========================================================================


def _v3_mutate_tb_report(report_paths, ledger_path, prereg, p_hash, mutate_fn):
    """TB 리포트 파일을 변조 후 재해시·재등록하는 헬퍼 (well-formed 유지)."""
    tb_path = next(rp for rp in report_paths if "tb" in rp.name)
    tb = json.loads(tb_path.read_text(encoding="utf-8"))
    mutate_fn(tb)
    tb = _finalize_report({k: v for k, v in tb.items() if k != "reportHash"})
    tb_path.write_text(json.dumps(tb, indent=2, ensure_ascii=False), encoding="utf-8")
    _register_report(ledger_path, tb, prereg, p_hash, tb_path)
    return tb_path


def test_v31_tb_honest_fail_demotes_to_no_claim(tmp_path):
    """수정 1 (리뷰어 프로브): TB 변형 부호 뒤집기 (정직 fail) → no_claim_m1_only.

    well-formed AXS-TB-001 리포트의 정직한 invariance fail 은 증거 결과 —
    검사 전체 ok, 두 트랙 비라이선스 (TB >= soft_pass 미충족), 라이선스 문장 없음.
    """
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_v3_full_set(tmp_path)

    def _flip_sign(tb):
        tb["tieBreak"]["variants"]["reverse"]["sign"] = "-"

    _v3_mutate_tb_report(report_paths, ledger_path, prereg, p_hash, _flip_sign)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]
    assert result.get("branch") == "no_claim_m1_only", (
        f"branch 오류: {result.get('branch')!r}"
    )
    assert result["rungs"]["M2-IMP"] is False
    assert result["rungs"]["M2-NOISE"] is False
    assert result["rungs"]["M2"] is False
    assert result["licensedSentences"] == []
    assert result["caveatRequired"] is False
    inv = next(c for c in result["checks"] if c["check"] == "axs010_invariance")
    assert "정직" in inv["detail"]


def test_v31_tb_honest_track_change_demotes_to_no_claim(tmp_path):
    """수정 1: TB 변형 trackDecision 변경 (정직 fail) → no_claim_m1_only, 검사 전체 ok."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_v3_full_set(tmp_path)

    def _change_track(tb):
        tb["tieBreak"]["variants"]["hash_seeded"]["trackDecision"] = "X"

    _v3_mutate_tb_report(report_paths, ledger_path, prereg, p_hash, _change_track)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    assert all(c["ok"] for c in result["checks"]), [
        c for c in result["checks"] if not c["ok"]
    ]
    assert result.get("branch") == "no_claim_m1_only"
    assert result["rungs"]["M2-IMP"] is False
    assert result["rungs"]["M2-NOISE"] is False
    assert result["licensedSentences"] == []


def test_v31_tb_structural_missing_field_remains_integrity_fail(tmp_path):
    """수정 1 경계: 구조적 TB 문제(변형 필수 필드 None)는 여전히 검사 실패 → integrity_fail."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_v3_full_set(tmp_path)

    def _null_track(tb):
        tb["tieBreak"]["variants"]["reverse"]["trackDecision"] = None

    _v3_mutate_tb_report(report_paths, ledger_path, prereg, p_hash, _null_track)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "axs010_invariance" for c in failed), (
        f"구조 위반이 검사 실패로 처리되지 않음: {[c['check'] for c in failed]!r}"
    )
    assert result.get("branch") == "integrity_fail"
    assert result["rungs"]["M2"] is False


def test_v31_tb_non_numeric_baseline_ci_remains_integrity_fail(tmp_path):
    """수정 1 경계: baseline ciLower 가 비숫자(문자열) → 구조 위반, 검사 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_v3_full_set(tmp_path)

    def _string_ci(tb):
        tb["tieBreak"]["baseline"]["ciLower"] = "high"

    _v3_mutate_tb_report(report_paths, ledger_path, prereg, p_hash, _string_ci)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "axs010_invariance" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_tb_boolean_estimate_remains_integrity_fail(tmp_path):
    """수정 1 경계: 변형 estimate 가 불리언 위장 → 구조 위반, 검사 실패."""
    prereg_path, report_paths, ledger_path, prereg, p_hash = _build_v3_full_set(tmp_path)

    def _bool_estimate(tb):
        tb["tieBreak"]["variants"]["reverse"]["estimate"] = True

    _v3_mutate_tb_report(report_paths, ledger_path, prereg, p_hash, _bool_estimate)
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "axs010_invariance" for c in failed)
    assert result.get("branch") == "integrity_fail"


def test_v31_redteam_alpha_entry_deleted_fails(tmp_path):
    """수정 2 (red-team): experiments 에서 AXS-ALPHA-EXP 엔트리 삭제 → 검사 실패.

    v3 에서 금지 선언(m2Prohibited=True)은 필수 — 엔트리 부재 = fail-closed.
    """
    def _mutate(p):
        del p["experiments"]["AXS-ALPHA-EXP"]

    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg, include_alpha=False
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "alpha_m2_prohibition" for c in failed), (
        f"알파 엔트리 삭제 우회가 차단되지 않음: {[c['check'] for c in failed]!r}"
    )
    assert result.get("branch") == "integrity_fail"
    assert result["rungs"]["M2"] is False


def test_v31_redteam_forged_artifact_arbitrary_path_fails(tmp_path):
    """수정 3 (리뷰어 프로브): 임의 절대 경로의 자기 일관 위조 산출물 + 위조 prereg (floor 0.05)
    → role_specific_guard 실패 (이전에는 both_supported 도달하던 프로브).
    """
    artifact_src = _REPO_ROOT / "configs" / "prereg" / "axs_dead_calibration_v1.json"
    with open(artifact_src, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    artifact.pop("summaryHash", None)
    artifact["floorDerivation"]["absoluteFloor"] = 0.05
    forged_hash = canonical_hash(artifact)
    artifact["summaryHash"] = forged_hash
    forged_path = tmp_path / "forged_calibration.json"
    forged_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def _mutate(p):
        key_rule = p["utilityGuard"]["keyManipulationRule"]
        key_rule["absoluteFloor"] = 0.05
        key_rule["floorCalibration"]["artifactPath"] = str(forged_path)
        key_rule["floorCalibration"]["summaryHash"] = forged_hash

    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    guard_fail = next(
        (c for c in failed if c["check"] == "role_specific_guard"), None
    )
    assert guard_fail is not None, (
        f"위조 산출물 임의 경로가 차단되지 않음: branch={result.get('branch')!r}, "
        f"실패: {[c['check'] for c in failed]!r}"
    )
    assert "고정" in guard_fail["detail"], (
        f"경로 고정 사유(한국어)가 detail 에 없음: {guard_fail['detail']!r}"
    )
    assert result.get("branch") == "integrity_fail"
    assert result["rungs"]["M2"] is False


def test_v31_redteam_artifact_path_traversal_fails(tmp_path):
    """수정 3: 동일 파일로 해석되는 우회 상대 경로(../)도 고정 경로 불일치로 거부."""
    def _mutate(p):
        p["utilityGuard"]["keyManipulationRule"]["floorCalibration"]["artifactPath"] = (
            "configs/prereg/../prereg/axs_dead_calibration_v1.json"
        )

    prereg_path, prereg = _make_v3_prereg_mutated(tmp_path, _mutate)
    _, report_paths, ledger_path, _, _ = _build_v3_full_set(
        tmp_path, prereg_override=prereg
    )
    result = evaluate_mechanism_license(
        prereg_path, report_paths, ledger_path=ledger_path, git_runner=_good_git_runner
    )
    failed = [c for c in result["checks"] if not c["ok"]]
    assert any(c["check"] == "role_specific_guard" for c in failed)
    assert result.get("branch") == "integrity_fail"
