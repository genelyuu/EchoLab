"""tests/test_axs_runners.py — AXS-P0 T4 TDD 픽스처 (AXS-003/009/004c/010 러너).

TDD: 테스트 먼저 작성(red), 러너 구현 후 green.

테스트 전략:
- 모듈 스코프 픽스처로 각 러너를 1회 실행, 결과를 어설션 전체에서 공유.
- 스모크 설정: H=2, pool_size=8, n_permutations=3, base_seeds=[42,7,101,2025,31337].
- 게이트 계약 테스트: test_ladder_gate 패턴 미러 (git_runner 주입 + 합성 선행 리포트).

로그/에러/CLI 출력: 한국어. 식별자·JSON 키: 영어.
All identifiers, JSON keys, and path strings stay English; log messages are Korean.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_PREREG = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"

# ---------------------------------------------------------------------------
# Smoke parameters
# ---------------------------------------------------------------------------

SMOKE_H = 2
SMOKE_POOL_SIZE = 8
SMOKE_N_PERM = 3
SMOKE_BASE_SEEDS = [42, 7, 101, 2025, 31337]

# ---------------------------------------------------------------------------
# git_runner fixture helpers (copied from test_ladder_gate pattern)
# ---------------------------------------------------------------------------

GOOD_PREREG_COMMIT = "aabbcc1122334455667788990011223344556677"
GOOD_RUN_COMMIT = "ff00ee1122334455667788990011223344556677"


def _good_git_runner(args: List[str]) -> str:
    """조상 관계 성립, 원격 브랜치 존재 시뮬레이션."""
    cmd = " ".join(args)
    if "merge-base" in cmd and "--is-ancestor" in cmd:
        return ""
    if "branch" in cmd and "-r" in cmd and "--contains" in cmd:
        return "  origin/main"
    if "rev-parse" in cmd:
        return GOOD_RUN_COMMIT
    if "log" in cmd and "--format=%H" in cmd:
        return GOOD_PREREG_COMMIT
    return ""


# ---------------------------------------------------------------------------
# Synthetic fixture helpers (mirrored from test_ladder_gate)
# ---------------------------------------------------------------------------

EVAL_FAMILIES = ["42", "7", "101", "2025", "31337"]


def _per_family_present(metric: str, value: float = 0.08) -> Dict[str, Any]:
    return {fam: {metric: value} for fam in EVAL_FAMILIES}


def _per_family_absent(metric: str) -> Dict[str, Any]:
    return {fam: {metric: 0.0} for fam in EVAL_FAMILIES}


def _bootstrap_present(metric: str) -> Dict[str, Any]:
    return {metric: {"mean": 0.08, "ciLower": 0.05, "ciUpper": 0.11}}


def _bootstrap_absent(metric: str) -> Dict[str, Any]:
    return {metric: {"mean": -0.01, "ciLower": -0.05, "ciUpper": 0.02}}


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
    from echo_bench.utils.hash import canonical_hash
    body = {k: v for k, v in report.items() if k != "reportHash"}
    report["reportHash"] = canonical_hash(body)
    return report


def _build_synthetic_axs002(prereg: Dict[str, Any], p_hash: str, ps: str) -> Dict[str, Any]:
    metric = "slate_excess_nmi_diff"
    r = {
        "reportId": "rep-axs002-synth",
        "experimentId": "AXS-002",
        "preregStamp": _build_stamp(prereg, p_hash, ps),
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


def _build_synthetic_axs010_strict(prereg: Dict[str, Any], p_hash: str, ps: str) -> Dict[str, Any]:
    r = {
        "reportId": "rep-axs010-synth",
        "experimentId": "AXS-010",
        "preregStamp": _build_stamp(prereg, p_hash, ps),
        "replayAudit": {"replayable": True},
        "tieBreak": {
            "baseline": {
                "sign": "+",
                "estimate": 0.08,
                "ciLower": 0.05,
                "ciUpper": 0.11,
                "trackDecision": "separability_present",
            },
            "variants": {
                "reverse": {"sign": "+", "estimate": 0.079, "trackDecision": "separability_present", "ciLower": 0.04, "ciUpper": 0.11},
                "hash_seeded": {"sign": "+", "estimate": 0.081, "trackDecision": "separability_present", "ciLower": 0.05, "ciUpper": 0.12},
                "feature_lexicographic": {"sign": "+", "estimate": 0.078, "trackDecision": "separability_present", "ciLower": 0.04, "ciUpper": 0.11},
            },
        },
    }
    return _finalize_report(r)


def _make_ledger(tmp_path: Path) -> Path:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps({"ledgerVersion": 1, "entries": []}, indent=2), encoding="utf-8"
    )
    return ledger_path


def _register_synthetic(
    ledger_path: Path,
    report: Dict[str, Any],
    prereg: Dict[str, Any],
    p_hash: str,
    report_path: Path,
) -> None:
    from echo_bench.logging.prereg import append_ledger_entry
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


# ===========================================================================
# Module-scoped smoke fixtures
# Each runner is invoked once; the resulting report is reused across tests.
# ===========================================================================


@pytest.fixture(scope="module")
def _smoke_configs():
    """기본 bases + archive_cfg 로드 (모듈 1회)."""
    from echo_bench.experiments.axs_common import load_default_configs
    return load_default_configs()


@pytest.fixture(scope="module")
def _prereg_path(tmp_path_factory):
    """prereg JSON 경로 (tmp 복사본)."""
    tmp = tmp_path_factory.mktemp("prereg")
    dest = tmp / "prereg.json"
    shutil.copy(_REAL_PREREG, dest)
    return dest


@pytest.fixture(scope="module")
def smoke_axs003(tmp_path_factory, _prereg_path, _smoke_configs):
    """AXS-003 스모크 실행 결과 캐시."""
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003
    bases, archive_cfg = _smoke_configs
    tmp = tmp_path_factory.mktemp("reports_003")
    report = run_axs_003(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=tmp,
    )
    return report


@pytest.fixture(scope="module")
def smoke_axs009(tmp_path_factory, _prereg_path, _smoke_configs):
    """AXS-009 스모크 실행 결과 캐시."""
    from echo_bench.experiments.axs_009_freeze import run_axs_009
    tmp = tmp_path_factory.mktemp("reports_009")
    report = run_axs_009(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=tmp,
    )
    return report


@pytest.fixture(scope="module")
def smoke_axs004c(tmp_path_factory, _prereg_path, _smoke_configs):
    """AXS-004c 스모크 실행 결과 캐시 (smoke schedule 생성 포함)."""
    from echo_bench.experiments.axs_004c_schedule_gen import generate_yoked_schedule, write_schedule
    from echo_bench.experiments.axs_004c_yoked import run_axs_004c

    bases, archive_cfg = _smoke_configs
    # generate a smoke schedule
    tmp = tmp_path_factory.mktemp("axs004c_sched")
    sched_path = tmp / "smoke_schedule.json"
    schedule = generate_yoked_schedule(
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        base_seed=999,
        bases=bases,
        archive_cfg=archive_cfg,
        policy_config={"k": 4, "alpha": 1.0, "lambda_reg": 1.0,
                       "features": ["coordinate_gap", "band_progress", "redundancy", "bias"],
                       "freeze_round": None, "tie_break_order": "canonical"},
    )
    write_schedule(schedule, sched_path)

    tmp_r = tmp_path_factory.mktemp("reports_004c")
    report = run_axs_004c(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        schedule_path=str(sched_path),
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=tmp_r,
    )
    return report


@pytest.fixture(scope="module")
def smoke_axs010(tmp_path_factory, _prereg_path, _smoke_configs):
    """AXS-010 스모크 실행 결과 캐시."""
    from echo_bench.experiments.axs_010_tiebreak import run_axs_010
    tmp = tmp_path_factory.mktemp("reports_010")
    report = run_axs_010(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=tmp,
    )
    return report


# ===========================================================================
# (a) dry_run returns plan dict + writes nothing
# ===========================================================================


def test_axs003_dry_run(tmp_path, _prereg_path):
    """AXS-003 dry_run: 계획 dict 반환, 파일 미작성."""
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003
    plan = run_axs_003(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        dry_run=True,
        reports_dir=tmp_path / "r",
    )
    assert plan.get("dryRun") is True
    assert not list((tmp_path / "r").glob("*.json")) if (tmp_path / "r").exists() else True


def test_axs009_dry_run(tmp_path, _prereg_path):
    """AXS-009 dry_run: 계획 dict 반환, 파일 미작성."""
    from echo_bench.experiments.axs_009_freeze import run_axs_009
    plan = run_axs_009(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        dry_run=True,
        reports_dir=tmp_path / "r",
    )
    assert plan.get("dryRun") is True


def test_axs004c_dry_run(tmp_path, _prereg_path, _smoke_configs):
    """AXS-004c dry_run: 계획 dict 반환, 파일 미작성."""
    from echo_bench.experiments.axs_004c_schedule_gen import generate_yoked_schedule, write_schedule
    from echo_bench.experiments.axs_004c_yoked import run_axs_004c

    bases, archive_cfg = _smoke_configs
    sched_path = tmp_path / "sched.json"
    schedule = generate_yoked_schedule(
        H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, base_seed=999,
        bases=bases, archive_cfg=archive_cfg,
        policy_config={"k": 4, "alpha": 1.0, "lambda_reg": 1.0,
                       "features": ["coordinate_gap", "band_progress", "redundancy", "bias"],
                       "freeze_round": None, "tie_break_order": "canonical"},
    )
    write_schedule(schedule, sched_path)

    plan = run_axs_004c(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
        schedule_path=str(sched_path),
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        dry_run=True,
        reports_dir=tmp_path / "r",
    )
    assert plan.get("dryRun") is True


def test_axs010_dry_run(tmp_path, _prereg_path):
    """AXS-010 dry_run: 계획 dict 반환, 파일 미작성."""
    from echo_bench.experiments.axs_010_tiebreak import run_axs_010
    plan = run_axs_010(
        SMOKE_BASE_SEEDS,
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        dry_run=True,
        reports_dir=tmp_path / "r",
    )
    assert plan.get("dryRun") is True


# ===========================================================================
# (b) real smoke run: expected top-level keys + reportHash self-consistent
# ===========================================================================

REQUIRED_TOP_LEVEL_KEYS = {
    "reportId", "experimentId", "preregStamp", "replayAudit",
    "reportHash", "seedBatchId", "configFreeze", "outputHash",
    "reproducibilityPack", "packHash",
}


def test_axs003_top_level_keys(smoke_axs003):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(smoke_axs003.keys()), (
        f"누락 키: {REQUIRED_TOP_LEVEL_KEYS - smoke_axs003.keys()}"
    )
    assert smoke_axs003["experimentId"] == "AXS-003"


def test_axs003_report_hash_self_consistent(smoke_axs003):
    from echo_bench.utils.hash import canonical_hash
    body = {k: v for k, v in smoke_axs003.items() if k != "reportHash"}
    assert canonical_hash(body) == smoke_axs003["reportHash"]


def test_axs009_top_level_keys(smoke_axs009):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(smoke_axs009.keys()), (
        f"누락 키: {REQUIRED_TOP_LEVEL_KEYS - smoke_axs009.keys()}"
    )
    assert smoke_axs009["experimentId"] == "AXS-009"


def test_axs009_report_hash_self_consistent(smoke_axs009):
    from echo_bench.utils.hash import canonical_hash
    body = {k: v for k, v in smoke_axs009.items() if k != "reportHash"}
    assert canonical_hash(body) == smoke_axs009["reportHash"]


def test_axs004c_top_level_keys(smoke_axs004c):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(smoke_axs004c.keys()), (
        f"누락 키: {REQUIRED_TOP_LEVEL_KEYS - smoke_axs004c.keys()}"
    )
    assert smoke_axs004c["experimentId"] == "AXS-004c"


def test_axs004c_report_hash_self_consistent(smoke_axs004c):
    from echo_bench.utils.hash import canonical_hash
    body = {k: v for k, v in smoke_axs004c.items() if k != "reportHash"}
    assert canonical_hash(body) == smoke_axs004c["reportHash"]


def test_axs010_top_level_keys(smoke_axs010):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(smoke_axs010.keys()), (
        f"누락 키: {REQUIRED_TOP_LEVEL_KEYS - smoke_axs010.keys()}"
    )
    assert smoke_axs010["experimentId"] == "AXS-010"


def test_axs010_report_hash_self_consistent(smoke_axs010):
    from echo_bench.utils.hash import canonical_hash
    body = {k: v for k, v in smoke_axs010.items() if k != "reportHash"}
    assert canonical_hash(body) == smoke_axs010["reportHash"]


# (b) write_report: expected path
def test_axs003_write_report(tmp_path, smoke_axs003):
    from echo_bench.experiments.axs_common import write_report
    out = write_report(smoke_axs003, reports_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["reportHash"] == smoke_axs003["reportHash"]


def test_axs010_write_report(tmp_path, smoke_axs010):
    from echo_bench.experiments.axs_common import write_report
    out = write_report(smoke_axs010, reports_dir=tmp_path)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["reportHash"] == smoke_axs010["reportHash"]


# ===========================================================================
# (c) determinism: two runs → identical reportHash
# ===========================================================================


def test_axs003_deterministic(tmp_path, _prereg_path):
    """AXS-003: 동일 인수 → 동일 reportHash."""
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003
    kw = dict(
        H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path, git_runner=_good_git_runner,
        reports_dir=tmp_path / "det003",
    )
    r1 = run_axs_003(SMOKE_BASE_SEEDS, **kw)
    r2 = run_axs_003(SMOKE_BASE_SEEDS, **kw)
    assert r1["reportHash"] == r2["reportHash"], (
        f"AXS-003 비결정론: {r1['reportHash'][:12]} != {r2['reportHash'][:12]}"
    )


def test_axs010_deterministic(tmp_path, _prereg_path):
    """AXS-010: 동일 인수 → 동일 reportHash."""
    from echo_bench.experiments.axs_010_tiebreak import run_axs_010
    kw = dict(
        H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path, git_runner=_good_git_runner,
        reports_dir=tmp_path / "det010",
    )
    r1 = run_axs_010(SMOKE_BASE_SEEDS, **kw)
    r2 = run_axs_010(SMOKE_BASE_SEEDS, **kw)
    assert r1["reportHash"] == r2["reportHash"], (
        f"AXS-010 비결정론: {r1['reportHash'][:12]} != {r2['reportHash'][:12]}"
    )


# ===========================================================================
# (d) gate contract tests
# ===========================================================================


def _build_gate_fixture(tmp_path, reports_to_submit, prereg_path, smoke_reports):
    """리포트 파일 작성 + 원장 등록 + 합성 선행 리포트 추가 후 evaluate_mechanism_license 호출."""
    from echo_bench.experiments.axs_common import write_report, register_report
    from echo_bench.logging.prereg import load_prereg, prereg_hash

    ledger_path = _make_ledger(tmp_path)
    prereg = load_prereg(prereg_path)
    p_hash = prereg_hash(prereg)
    ps = str(prereg_path)

    # Write and register each submitted smoke report
    report_paths = []
    for report in reports_to_submit:
        rp = write_report(report, reports_dir=tmp_path)
        register_report(report, rp, ledger_path=ledger_path)
        report_paths.append(rp)

    # Add synthetic AXS-002 (requiresPass coverage)
    if not any(r.get("experimentId") == "AXS-002" for r in reports_to_submit):
        synth_002 = _build_synthetic_axs002(prereg, p_hash, ps)
        rp_002 = tmp_path / "rep-axs002-synth.json"
        rp_002.write_text(json.dumps(synth_002, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_synthetic(ledger_path, synth_002, prereg, p_hash, rp_002)
        report_paths.append(rp_002)

    # Add synthetic AXS-010 (for M2) unless already submitted
    if not any(r.get("experimentId") == "AXS-010" for r in reports_to_submit):
        synth_010 = _build_synthetic_axs010_strict(prereg, p_hash, ps)
        rp_010 = tmp_path / "rep-axs010-synth.json"
        rp_010.write_text(json.dumps(synth_010, indent=2, ensure_ascii=False), encoding="utf-8")
        _register_synthetic(ledger_path, synth_010, prereg, p_hash, rp_010)
        report_paths.append(rp_010)

    return report_paths, ledger_path


def test_axs003_gate_contract(tmp_path, smoke_axs003, smoke_axs009, smoke_axs004c, _prereg_path):
    """AXS-003: gate checks ok for our fields (perFamily keys, hash integrity, replayable)."""
    from echo_bench.tools.ladder_gate import evaluate_mechanism_license

    report_paths, ledger_path = _build_gate_fixture(
        tmp_path,
        [smoke_axs003, smoke_axs009, smoke_axs004c],
        _prereg_path,
        None,
    )
    result = evaluate_mechanism_license(
        _prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    checks_by_name = {c["check"]: c for c in result["checks"]}
    assert checks_by_name["prereg_hash_match"]["ok"], checks_by_name["prereg_hash_match"]["detail"]
    assert checks_by_name["report_hash_integrity"]["ok"], checks_by_name["report_hash_integrity"]["detail"]
    assert checks_by_name["replayable"]["ok"], checks_by_name["replayable"]["detail"]
    assert checks_by_name["ancestry"]["ok"], checks_by_name["ancestry"]["detail"]
    assert checks_by_name["ledger_registered"]["ok"], checks_by_name["ledger_registered"]["detail"]
    # arms_complete: AXS-003 must have its two arms
    arms_check = checks_by_name["arms_complete"]
    assert arms_check["ok"] is True, (
        f"arms_complete 실패 (prereg 기반 실제 체크): {arms_check['detail']}"
    )
    # pilot_disjoint (AXS-004c present)
    assert checks_by_name["pilot_disjoint"]["ok"], checks_by_name["pilot_disjoint"]["detail"]


def test_axs010_gate_axs010_invariance_no_schema_failure(tmp_path, smoke_axs010, smoke_axs003, smoke_axs009, smoke_axs004c, _prereg_path):
    """AXS-010: axs010_invariance check가 스키마/누락 키 문제로 실패하지 않음."""
    from echo_bench.tools.ladder_gate import evaluate_mechanism_license

    report_paths, ledger_path = _build_gate_fixture(
        tmp_path,
        [smoke_axs003, smoke_axs009, smoke_axs004c, smoke_axs010],
        _prereg_path,
        None,
    )
    result = evaluate_mechanism_license(
        _prereg_path,
        report_paths,
        ledger_path=ledger_path,
        git_runner=_good_git_runner,
    )
    checks_by_name = {c["check"]: c for c in result["checks"]}
    inv_check = checks_by_name["axs010_invariance"]
    # If it fails, it must be about values/CI, NOT about missing keys or format
    if not inv_check["ok"]:
        detail = inv_check["detail"]
        schema_keywords = ["누락", "형식", "missing", "schema", "format"]
        has_schema_issue = any(kw in detail for kw in schema_keywords)
        # Allowed failure reasons: sign flip, CI, estimate, trackDecision값 — not schema
        assert not has_schema_issue, (
            f"axs010_invariance가 스키마 문제로 실패해서는 안 됩니다: {detail!r}"
        )


# ===========================================================================
# (e) AXS-009 specifics
# ===========================================================================


def test_axs009_freeze_none_divergence_zero(smoke_axs009):
    """freeze_none arm: post_freeze_incremental_divergence ≡ 0.0 per family."""
    arms = smoke_axs009.get("arms", {})
    freeze_none = arms.get("freeze_none", {})
    per_family = freeze_none.get("perFamily", {})
    for fam, row in per_family.items():
        div = row.get("post_freeze_incremental_divergence")
        assert div == 0.0, (
            f"freeze_none family {fam!r}: divergence={div!r} (0.0 이어야 함)"
        )


def test_axs009_freeze_rounds_correct(smoke_axs009):
    """freezeRounds 필드 존재 + smoke H=2 에 맞는 값."""
    freeze_rounds = smoke_axs009.get("freezeRounds", {})
    assert "freeze_at_1" in freeze_rounds
    assert "freeze_at_quarter" in freeze_rounds
    assert "freeze_at_half" in freeze_rounds
    assert "freeze_none" in freeze_rounds

    # H=2: freeze_at_1=1, freeze_at_quarter=max(1,2//4)=max(1,0)=1, freeze_at_half=max(1,2//2)=1
    assert freeze_rounds["freeze_at_1"] == 1
    assert freeze_rounds["freeze_at_quarter"] == max(1, SMOKE_H // 4)
    assert freeze_rounds["freeze_at_half"] == max(1, SMOKE_H // 2)
    assert freeze_rounds["freeze_none"] is None


def test_axs009_both_bootstrap_blocks_present(smoke_axs009):
    """각 arm: bootstrap에 slate_excess_nmi + post_freeze_incremental_divergence 모두 존재."""
    arms = smoke_axs009.get("arms", {})
    for arm_id in ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]:
        arm = arms.get(arm_id, {})
        bootstrap = arm.get("bootstrap", {})
        assert "slate_excess_nmi" in bootstrap, (
            f"{arm_id}: bootstrap['slate_excess_nmi'] 누락"
        )
        assert "post_freeze_incremental_divergence" in bootstrap, (
            f"{arm_id}: bootstrap['post_freeze_incremental_divergence'] 누락"
        )
        for metric_key in ["slate_excess_nmi", "post_freeze_incremental_divergence"]:
            block = bootstrap[metric_key]
            assert "mean" in block, f"{arm_id}.bootstrap.{metric_key}: mean 누락"
            assert "ciLower" in block, f"{arm_id}.bootstrap.{metric_key}: ciLower 누락"
            assert "ciUpper" in block, f"{arm_id}.bootstrap.{metric_key}: ciUpper 누락"


def test_axs009_per_family_has_both_metrics(smoke_axs009):
    """각 arm의 perFamily: 두 메트릭 모두 존재."""
    arms = smoke_axs009.get("arms", {})
    for arm_id in ["freeze_at_1", "freeze_at_quarter", "freeze_at_half", "freeze_none"]:
        arm = arms.get(arm_id, {})
        per_family = arm.get("perFamily", {})
        # 5개 패밀리 모두 있어야 함
        assert set(per_family.keys()) == set(EVAL_FAMILIES), (
            f"{arm_id} perFamily 키: {set(per_family.keys())} != {set(EVAL_FAMILIES)}"
        )
        for fam, row in per_family.items():
            assert "slate_excess_nmi" in row, f"{arm_id} family {fam}: slate_excess_nmi 누락"
            assert "post_freeze_incremental_divergence" in row, (
                f"{arm_id} family {fam}: post_freeze_incremental_divergence 누락"
            )


# ===========================================================================
# (f) AXS-010 specifics
# ===========================================================================


def test_axs010_tiebreak_baseline_fields(smoke_axs010):
    """tieBreak.baseline에 5개 필수 필드 존재."""
    tb = smoke_axs010.get("tieBreak", {})
    baseline = tb.get("baseline", {})
    required = {"sign", "estimate", "ciLower", "ciUpper", "trackDecision"}
    assert required.issubset(baseline.keys()), (
        f"tieBreak.baseline 누락 필드: {required - baseline.keys()}"
    )


def test_axs010_tiebreak_variants_keys(smoke_axs010):
    """tieBreak.variants에 정확히 3개 키."""
    tb = smoke_axs010.get("tieBreak", {})
    variants = tb.get("variants", {})
    required = {"reverse", "hash_seeded", "feature_lexicographic"}
    assert required == set(variants.keys()), (
        f"variants 키 불일치: {set(variants.keys())} != {required}"
    )


def test_axs010_tiebreak_variants_have_ci(smoke_axs010):
    """variants 에도 ciLower/ciUpper 존재."""
    variants = smoke_axs010.get("tieBreak", {}).get("variants", {})
    for var_id, var_data in variants.items():
        assert "ciLower" in var_data, f"variant {var_id!r}: ciLower 누락"
        assert "ciUpper" in var_data, f"variant {var_id!r}: ciUpper 누락"


def test_axs010_track_decision_valid(smoke_axs010):
    """trackDecision은 두 허용값 중 하나."""
    allowed = {"separability_present", "separability_absent"}
    tb = smoke_axs010.get("tieBreak", {})
    baseline_td = tb.get("baseline", {}).get("trackDecision")
    assert baseline_td in allowed, (
        f"tieBreak.baseline.trackDecision={baseline_td!r} (허용값: {allowed})"
    )
    for var_id, var_data in tb.get("variants", {}).items():
        var_td = var_data.get("trackDecision")
        assert var_td in allowed, (
            f"variants[{var_id!r}].trackDecision={var_td!r} (허용값: {allowed})"
        )


# ===========================================================================
# (g) AXS-004c specifics
# ===========================================================================


def test_axs004c_fails_without_schedule(tmp_path, _prereg_path):
    """AXS-004c: schedule 파일 없으면 한국어 ValueError."""
    from echo_bench.experiments.axs_004c_yoked import run_axs_004c
    missing = tmp_path / "nonexistent_schedule.json"
    with pytest.raises((ValueError, FileNotFoundError), match="스케줄|schedule|파일"):
        run_axs_004c(
            SMOKE_BASE_SEEDS,
            H=SMOKE_H,
            k=4,
            pool_size=SMOKE_POOL_SIZE,
            n_permutations=SMOKE_N_PERM,
            schedule_path=str(missing),
            prereg_path=_prereg_path,
            git_runner=_good_git_runner,
        )


def test_axs004c_yoked_schedule_hash_in_body(smoke_axs004c):
    """yokedScheduleHash가 리포트 바디에 존재."""
    assert "yokedScheduleHash" in smoke_axs004c, (
        "yokedScheduleHash가 AXS-004c 리포트 바디에 없음"
    )
    assert isinstance(smoke_axs004c["yokedScheduleHash"], str)
    assert len(smoke_axs004c["yokedScheduleHash"]) > 0


def test_axs004c_yoked_schedule_hash_matches_artifact(_smoke_configs, _prereg_path, tmp_path, smoke_axs004c):
    """yokedScheduleHash == 생성된 schedule의 scheduleHash."""
    from echo_bench.experiments.axs_004c_schedule_gen import generate_yoked_schedule
    bases, archive_cfg = _smoke_configs
    # regenerate to get scheduleHash
    schedule = generate_yoked_schedule(
        H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, base_seed=999,
        bases=bases, archive_cfg=archive_cfg,
        policy_config={"k": 4, "alpha": 1.0, "lambda_reg": 1.0,
                       "features": ["coordinate_gap", "band_progress", "redundancy", "bias"],
                       "freeze_round": None, "tie_break_order": "canonical"},
    )
    assert smoke_axs004c["yokedScheduleHash"] == schedule["scheduleHash"], (
        f"yokedScheduleHash 불일치: "
        f"{smoke_axs004c['yokedScheduleHash'][:12]} != {schedule['scheduleHash'][:12]}"
    )


# ===========================================================================
# Additional: perFamily coverage (all 5 families present in each arm)
# ===========================================================================


def test_axs003_per_family_keys(smoke_axs003):
    """AXS-003: 두 arm의 perFamily에 5개 패밀리 모두 존재."""
    arms = smoke_axs003.get("arms", {})
    for arm_id in ["axs_ucb_default", "axs_ucb_alpha0"]:
        per_family = arms.get(arm_id, {}).get("perFamily", {})
        assert set(per_family.keys()) == set(EVAL_FAMILIES), (
            f"AXS-003 {arm_id} perFamily 키: {set(per_family.keys())} != {set(EVAL_FAMILIES)}"
        )


def test_axs004c_per_family_keys(smoke_axs004c):
    """AXS-004c: 두 arm의 perFamily에 5개 패밀리 모두 존재."""
    arms = smoke_axs004c.get("arms", {})
    for arm_id in ["axs_ucb_default", "axs_yoked_bonus"]:
        per_family = arms.get(arm_id, {}).get("perFamily", {})
        assert set(per_family.keys()) == set(EVAL_FAMILIES), (
            f"AXS-004c {arm_id} perFamily: {set(per_family.keys())} != {set(EVAL_FAMILIES)}"
        )


def test_axs010_per_family_all_orders(smoke_axs010):
    """AXS-010: 4개 tie_break_order 모두 perFamilyValues에 5 패밀리."""
    pfv = smoke_axs010.get("perFamilyValues", {})
    for order in ["canonical", "reverse", "hash_seeded", "feature_lexicographic"]:
        order_data = pfv.get(order, {})
        assert set(order_data.keys()) == set(EVAL_FAMILIES), (
            f"AXS-010 perFamilyValues[{order!r}] 키: {set(order_data.keys())} != {set(EVAL_FAMILIES)}"
        )


def test_axs003_baselines_random_present(smoke_axs003):
    """AXS-003: baselines.RANDOM.coordinate_coverage_mean 존재."""
    baselines = smoke_axs003.get("baselines", {})
    rand = baselines.get("RANDOM", {})
    assert "coordinate_coverage_mean" in rand
    assert isinstance(rand["coordinate_coverage_mean"], float)


def test_axs009_baselines_random_present(smoke_axs009):
    """AXS-009: baselines.RANDOM.coordinate_coverage_mean 존재."""
    baselines = smoke_axs009.get("baselines", {})
    rand = baselines.get("RANDOM", {})
    assert "coordinate_coverage_mean" in rand
    assert isinstance(rand["coordinate_coverage_mean"], float)


# ===========================================================================
# (h) Tamper-detection regression tests — single-source binding
#
# Approach: inspect.getsource code-level guard.
#
# Why: the old tests only checked isinstance(value, float) — a parallel
# *_raw_by_family[fam] variable feeding arms/baselines directly would still
# pass. The real regression is a runner that derives report numbers from a
# variable OTHER than family_blocks after the construction block.
#
# This test reads the runner source and asserts two structural invariants:
#   1. The "derive values from family_blocks" section references
#      `family_blocks[fam]` (not just family_blocks as a dict key container).
#   2. The per-arm raw-result dicts (_raw_by_family / arm_raw) do NOT appear
#      inside the arms/baselines derivation section — i.e. only family_blocks
#      is the source. We check that the source fragment between the
#      "derive all report values" marker comment and the build_arm_entry /
#      body_extra construction contains `family_blocks[fam]` and does NOT
#      contain any `_raw_by_family[fam]` or `arm_raw[` reads (which would
#      indicate a parallel source bypassing the single-source block).
#
# This is a code-level guard: it would fail if someone reintroduces a
# parallel non-family_blocks variable feeding arms/baselines.
# It does NOT test runtime values — the existing smoke fixtures cover that.
# ===========================================================================

def _extract_derive_section(source: str, marker: str = "derive all report values") -> str:
    """Return source lines from the 'derive all report values' comment onward.

    Stops at the first 'def ' line (recompute_fn / next function) so the
    guard is scoped to the derivation block only and not the recompute closure.
    """
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if marker in line:
            start = i
            break
    if start is None:
        return ""
    result_lines = []
    for line in lines[start:]:
        # Stop at the recompute_fn closure or next top-level def
        stripped = line.lstrip()
        if stripped.startswith("def recompute_fn") or stripped.startswith("def _"):
            break
        result_lines.append(line)
    return "\n".join(result_lines)


def test_axs003_family_blocks_contain_all_arm_coverage(smoke_axs003):
    """family_blocks single-source guard (AXS-003): source-level assertion.

    Verifies that in the derivation section the runner reads from family_blocks
    exclusively — no parallel *_raw_by_family[fam] bypasses the single-source
    block when building arm entries and baselines.
    """
    import inspect
    import echo_bench.experiments.axs_003_alpha_kill as _mod003

    src = inspect.getsource(_mod003)
    derive = _extract_derive_section(src)

    assert derive, (
        "AXS-003 러너에서 'derive all report values' 마커 섹션을 찾을 수 없음 — "
        "single-source 계약 주석이 삭제되었거나 이동되었습니다."
    )
    assert "family_blocks[fam]" in derive, (
        "AXS-003 derivation section에 family_blocks[fam] 참조가 없습니다 — "
        "단일 소스 계약 위반: arms/baselines 값이 family_blocks 에서 파생되어야 합니다."
    )
    # Parallel raw-dict reads in the derivation section indicate a bypass
    forbidden_patterns = [
        "default_raw_by_family[fam]",
        "alpha0_raw_by_family[fam]",
        "random_raw_by_family[fam]",
    ]
    for pat in forbidden_patterns:
        assert pat not in derive, (
            f"AXS-003 derivation section에 {pat!r} 가 있습니다 — "
            "family_blocks 를 우회하는 병렬 원시 변수 접근이 탐지되었습니다."
        )


def test_axs009_family_blocks_contain_all_arm_coverage(smoke_axs009):
    """family_blocks single-source guard (AXS-009): source-level assertion.

    Verifies that in the derivation section arm coverage values and baselines
    are derived exclusively from family_blocks, not from arm_raw directly.
    """
    import inspect
    import echo_bench.experiments.axs_009_freeze as _mod009

    src = inspect.getsource(_mod009)
    derive = _extract_derive_section(src)

    assert derive, (
        "AXS-009 러너에서 'derive all report values' 마커 섹션을 찾을 수 없음 — "
        "single-source 계약 주석이 삭제되었거나 이동되었습니다."
    )
    assert "family_blocks[fam]" in derive, (
        "AXS-009 derivation section에 family_blocks[fam] 참조가 없습니다 — "
        "단일 소스 계약 위반: arms/baselines 값이 family_blocks 에서 파생되어야 합니다."
    )
    # arm_raw[arm_id][fam] reads outside the family_blocks construction block
    # indicate a bypass of the single-source invariant
    assert "arm_raw[arm_id][fam]" not in derive, (
        "AXS-009 derivation section에 arm_raw[arm_id][fam] 가 있습니다 — "
        "family_blocks 를 우회하는 병렬 원시 변수 접근이 탐지되었습니다."
    )
    assert "random_raw_by_family[fam]" not in derive, (
        "AXS-009 derivation section에 random_raw_by_family[fam] 가 있습니다 — "
        "RANDOM baseline 이 family_blocks 를 우회하고 있습니다."
    )


def test_axs004c_family_blocks_contain_all_arm_coverage(smoke_axs004c):
    """family_blocks single-source guard (AXS-004c): source-level assertion.

    Verifies that in the derivation section arm coverage values and baselines
    are derived exclusively from family_blocks, not from the raw-by-family dicts.
    """
    import inspect
    import echo_bench.experiments.axs_004c_yoked as _mod004c

    src = inspect.getsource(_mod004c)
    derive = _extract_derive_section(src)

    assert derive, (
        "AXS-004c 러너에서 'derive all report values' 마커 섹션을 찾을 수 없음 — "
        "single-source 계약 주석이 삭제되었거나 이동되었습니다."
    )
    assert "family_blocks[fam]" in derive, (
        "AXS-004c derivation section에 family_blocks[fam] 참조가 없습니다 — "
        "단일 소스 계약 위반: arms/baselines 값이 family_blocks 에서 파생되어야 합니다."
    )
    forbidden_patterns = [
        "default_raw_by_family[fam]",
        "yoked_raw_by_family[fam]",
        "random_raw_by_family[fam]",
    ]
    for pat in forbidden_patterns:
        assert pat not in derive, (
            f"AXS-004c derivation section에 {pat!r} 가 있습니다 — "
            "family_blocks 를 우회하는 병렬 원시 변수 접근이 탐지되었습니다."
        )


def test_axs003_recompute_fn_covers_random_baseline(smoke_axs003, _prereg_path):
    """recompute_fn covers RANDOM baseline + full-coverage replay audit binding.

    Two bindings:
    1. Determinism: a second run with identical args produces the same
       reportHash, proving arms/baselines values are stable single-source.
    2. Full-replay familyBlockHash coverage: running with replay_mode="full"
       produces replayAudit.perFamily[fam].familyBlockHash for ALL 5 families,
       confirming the replay audit actually audited every family block.
       If arms/baselines came from a parallel variable diverging from
       family_blocks, the recomputed block would hash differently and
       replayable would be False — caught here.
    """
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003

    baselines = smoke_axs003.get("baselines", {})
    rand_cov = baselines.get("RANDOM", {}).get("coordinate_coverage_mean")
    assert rand_cov is not None, "baselines.RANDOM.coordinate_coverage_mean 누락"

    # --- binding 1: determinism (uses module-level Path and tempfile) ---
    with tempfile.TemporaryDirectory() as tmp:
        r2 = run_axs_003(
            SMOKE_BASE_SEEDS,
            H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
            prereg_path=_prereg_path, git_runner=_good_git_runner,
            reports_dir=Path(tmp),
        )
    rand_cov2 = r2.get("baselines", {}).get("RANDOM", {}).get("coordinate_coverage_mean")
    assert rand_cov == rand_cov2, (
        f"RANDOM baseline coverage not deterministic: {rand_cov} != {rand_cov2}"
    )
    assert smoke_axs003["reportHash"] == r2["reportHash"], (
        "Determinism check: report hashes must match"
    )

    # --- binding 2: full-replay familyBlockHash present for all 5 families ---
    # replay_mode="full" forces the runner to recompute every family block and
    # record familyBlockHash in replayAudit.perFamily. If any arm or baseline
    # value was sourced from a parallel variable rather than family_blocks, the
    # recomputed block hash would diverge and replayable would be False.
    with tempfile.TemporaryDirectory() as tmp_full:
        r_full = run_axs_003(
            SMOKE_BASE_SEEDS,
            H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
            prereg_path=_prereg_path, git_runner=_good_git_runner,
            reports_dir=Path(tmp_full),
            replay_mode="full",
        )
    audit = r_full.get("replayAudit", {})
    assert audit.get("replayable") is True, (
        "AXS-003 full-replay 감사 실패: replayAudit.replayable 이 True 가 아닙니다 — "
        f"perFamily={audit.get('perFamily')}"
    )
    per_family_audit = audit.get("perFamily", {})
    expected_families = [str(s) for s in SMOKE_BASE_SEEDS]
    for fam in expected_families:
        assert fam in per_family_audit, (
            f"AXS-003 full-replay 감사 누락: replayAudit.perFamily[{fam!r}] 없음"
        )
        fam_entry = per_family_audit[fam]
        assert "familyBlockHash" in fam_entry, (
            f"AXS-003 full-replay 감사 누락: perFamily[{fam!r}].familyBlockHash 없음"
        )
        assert fam_entry.get("replayable") is True, (
            f"AXS-003 full-replay 감사 실패: perFamily[{fam!r}].replayable 이 True 가 아닙니다"
        )


# ===========================================================================
# (i) AXS-V3 N1: base_seeds 누락 봉쇄 — seedBatchId 충돌 방지
#
# 버그: 네 러너의 run_params 에 base_seeds 가 없어 서로 다른 family set 이
#       동일한 seedBatchId → 동일한 파일명을 생성, 리포트가 덮어쓰여지는 문제.
# 픽스: run_params 에 "base_seeds": [int(s) for s in base_seeds] 추가.
# ===========================================================================


def test_axs003_different_base_seeds_produce_different_seedbatchid_and_filename(
    tmp_path, _prereg_path
):
    """AXS-003: 서로 다른 base_seeds → 다른 seedBatchId + 다른 파일명 (덮어쓰기 없음).

    두 실행을 동일한 reports_dir 에 기록해 파일이 2개 생성되는지 검증한다.
    (기존 버그: 두 파일 모두 axs_003_d61d7a9deb3a.json 으로 중복 생성됨)
    """
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003

    reports_dir = tmp_path / "reports_collision"
    reports_dir.mkdir()

    # family set A: seeds [42, 7]
    r_a = run_axs_003(
        [42, 7],
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=reports_dir,
    )
    # family set B: seeds [123, 55555] — completely different families
    r_b = run_axs_003(
        [123, 55555],
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=reports_dir,
    )

    assert r_a["seedBatchId"] != r_b["seedBatchId"], (
        "AXS-003: 서로 다른 base_seeds 임에도 seedBatchId 가 동일합니다 — "
        f"base_seeds A={[42, 7]}, B={[123, 55555]}, "
        f"seedBatchId={r_a['seedBatchId'][:12]}"
    )
    assert r_a["reportHash"] != r_b["reportHash"], (
        "AXS-003: 서로 다른 base_seeds 임에도 reportHash 가 동일합니다."
    )

    # 파일이 두 개 생성되어야 함 (덮어쓰기 없음)
    json_files = sorted(reports_dir.glob("axs_003_*.json"))
    assert len(json_files) == 2, (
        f"AXS-003: 서로 다른 family set 으로 실행했는데 파일이 {len(json_files)}개입니다 "
        f"(2개 기대) — 파일명 충돌로 덮어쓰기가 발생했습니다.\n"
        f"파일 목록: {[f.name for f in json_files]}"
    )


def test_axs003_same_base_seeds_produce_identical_seedbatchid(tmp_path, _prereg_path):
    """AXS-003: 동일 base_seeds → 동일 seedBatchId + 동일 reportHash (결정론성 보존)."""
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003

    kw = dict(
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=tmp_path / "det_same_a",
    )
    r1 = run_axs_003([42, 7], **kw)
    kw2 = dict(kw, reports_dir=tmp_path / "det_same_b")
    r2 = run_axs_003([42, 7], **kw2)

    assert r1["seedBatchId"] == r2["seedBatchId"], (
        "AXS-003: 동일 base_seeds 임에도 seedBatchId 가 달라졌습니다 — 결정론성 파괴."
    )
    assert r1["reportHash"] == r2["reportHash"], (
        "AXS-003: 동일 base_seeds 임에도 reportHash 가 달라졌습니다 — 결정론성 파괴."
    )


def test_axs003_seedbatchid_includes_base_seeds(tmp_path, _prereg_path):
    """AXS-003: seedBatchId 계산에 base_seeds 가 포함됨을 인테스트 재구성으로 검증.

    build_axs_report 내 stable 필터가 base_seeds 를 pick-up 하는지 확인한다.
    run_params 에 base_seeds 가 없으면 재구성 해시가 불일치한다.
    """
    from echo_bench.experiments.axs_003_alpha_kill import run_axs_003
    from echo_bench.experiments.e_leakage_diagnostic import EXPANDED_PROBE_SET
    from echo_bench.utils.hash import canonical_hash

    r = run_axs_003(
        [42, 7],
        H=SMOKE_H,
        k=4,
        pool_size=SMOKE_POOL_SIZE,
        n_permutations=SMOKE_N_PERM,
        prereg_path=_prereg_path,
        git_runner=_good_git_runner,
        reports_dir=tmp_path / "bs_check",
    )

    # base_seeds 포함 재구성
    run_params_with_seeds = {
        "H": SMOKE_H,
        "k": 4,
        "pool_size": SMOKE_POOL_SIZE,
        "n_permutations": SMOKE_N_PERM,
        "experiment": "AXS-003",
        "base_seeds": [42, 7],
    }
    stable_with = {
        k: run_params_with_seeds[k]
        for k in sorted(run_params_with_seeds)
        if k != "configFreeze"
    }
    expected_with = canonical_hash(
        {"experiment": "AXS-003", "probes": list(EXPANDED_PROBE_SET), **stable_with}
    )

    # base_seeds 제외 재구성
    run_params_without_seeds = {
        "H": SMOKE_H,
        "k": 4,
        "pool_size": SMOKE_POOL_SIZE,
        "n_permutations": SMOKE_N_PERM,
        "experiment": "AXS-003",
    }
    stable_without = {
        k: run_params_without_seeds[k]
        for k in sorted(run_params_without_seeds)
        if k != "configFreeze"
    }
    expected_without = canonical_hash(
        {"experiment": "AXS-003", "probes": list(EXPANDED_PROBE_SET), **stable_without}
    )

    actual = r["seedBatchId"]
    assert actual == expected_with, (
        f"AXS-003 seedBatchId 가 base_seeds 를 포함하지 않습니다.\n"
        f"  actual:          {actual[:12]}\n"
        f"  expected_with:   {expected_with[:12]}\n"
        f"  expected_without:{expected_without[:12]}"
    )
    assert actual != expected_without, (
        "AXS-003 seedBatchId 가 base_seeds 없는 해시와 동일 — base_seeds 가 실제로 누락됨."
    )


@pytest.mark.parametrize("runner_name,base_seeds_a,base_seeds_b,experiment_id", [
    ("axs_009_freeze", [42, 7], [123, 55555], "AXS-009"),
    ("axs_004c_yoked", [42, 7], [123, 55555], "AXS-004c"),
    ("axs_010_tiebreak", [42, 7], [123, 55555], "AXS-010"),
])
def test_other_runners_different_base_seeds_produce_different_seedbatchid(
    runner_name, base_seeds_a, base_seeds_b, experiment_id,
    tmp_path, _prereg_path, _smoke_configs
):
    """AXS-009/004c/010: 서로 다른 base_seeds → 다른 seedBatchId (덮어쓰기 봉쇄 검증).

    AXS-004c 는 smoke schedule 이 필요하므로 별도 생성.
    """
    if runner_name == "axs_009_freeze":
        from echo_bench.experiments.axs_009_freeze import run_axs_009 as run_fn
        def call_runner(seeds, rd):
            return run_fn(
                seeds,
                H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
                prereg_path=_prereg_path, git_runner=_good_git_runner,
                reports_dir=rd,
            )
    elif runner_name == "axs_004c_yoked":
        from echo_bench.experiments.axs_004c_schedule_gen import generate_yoked_schedule, write_schedule
        from echo_bench.experiments.axs_004c_yoked import run_axs_004c
        bases, archive_cfg = _smoke_configs
        sched_path = tmp_path / "sched_collision.json"
        schedule = generate_yoked_schedule(
            H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, base_seed=999,
            bases=bases, archive_cfg=archive_cfg,
            policy_config={"k": 4, "alpha": 1.0, "lambda_reg": 1.0,
                           "features": ["coordinate_gap", "band_progress", "redundancy", "bias"],
                           "freeze_round": None, "tie_break_order": "canonical"},
        )
        write_schedule(schedule, sched_path)
        def call_runner(seeds, rd):
            return run_axs_004c(
                seeds,
                H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
                schedule_path=str(sched_path),
                prereg_path=_prereg_path, git_runner=_good_git_runner,
                reports_dir=rd,
            )
    else:  # axs_010_tiebreak
        from echo_bench.experiments.axs_010_tiebreak import run_axs_010
        def call_runner(seeds, rd):
            return run_axs_010(
                seeds,
                H=SMOKE_H, k=4, pool_size=SMOKE_POOL_SIZE, n_permutations=SMOKE_N_PERM,
                prereg_path=_prereg_path, git_runner=_good_git_runner,
                reports_dir=rd,
            )

    reports_dir = tmp_path / f"collision_{runner_name}"
    reports_dir.mkdir()

    r_a = call_runner(base_seeds_a, reports_dir)
    r_b = call_runner(base_seeds_b, reports_dir)

    assert r_a["seedBatchId"] != r_b["seedBatchId"], (
        f"{experiment_id}: 서로 다른 base_seeds 임에도 seedBatchId 가 동일합니다 — "
        f"base_seeds A={base_seeds_a}, B={base_seeds_b}, "
        f"seedBatchId={r_a['seedBatchId'][:12]}"
    )

    # 파일이 두 개 생성되어야 함 (덮어쓰기 없음)
    short_id = experiment_id.upper().replace("AXS-", "").lower()
    json_files = sorted(reports_dir.glob(f"axs_{short_id}_*.json"))
    assert len(json_files) == 2, (
        f"{experiment_id}: 서로 다른 family set 으로 실행했는데 파일이 {len(json_files)}개입니다 "
        f"(2개 기대) — 파일명 충돌로 덮어쓰기가 발생했습니다.\n"
        f"파일 목록: {[f.name for f in json_files]}"
    )
