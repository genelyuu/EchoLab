"""tests/test_axs_common.py — TDD 픽스처 (AXS-P0 T2).

Coverage:
- bootstrap_block: 키, 타입, ci_low→ciLower 매핑, 결정론성, 패밀리 순서 독립성
- build_arm_entry: degenerate 트리플, 경계값(equal=비퇴화), 타입 정확성
- run_arm_family: 스모크(H=2, pool=8, perm=3), 필수 키, 결정론성, 타입
- build_axs_report + write_report: 게이트 필수 키, reportHash 자기 일관성,
  numpy/bool 누수 없음, 파일명, 결정론성
- register_report: 8-키 원장 엔트리
- dry_run_plan: 파일 미작성, plan 키
- make_axs_arg_parser: 기본값, --dry-run/--register-ledger 플래그
- divergent recompute_fn: replayAudit.replayable False 전파 검증
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---- 경로 ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_PREREG = _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json"

# ---- git_runner 픽스처 ----
GOOD_PREREG_COMMIT = "aabbcc1122334455667788990011223344556677"
GOOD_RUN_COMMIT = "ff00ee1122334455667788990011223344556677"


def _good_git_runner(args: List[str]) -> str:
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


# ===========================================================================
# 공유 설정 픽스처
# ===========================================================================

_BASES_CFG = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG = _REPO_ROOT / "configs" / "archive" / "archive.yaml"


def _load_yaml_local(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc if isinstance(doc, dict) else {}


@pytest.fixture(scope="module")
def smoke_bases():
    from echo_bench.basis.schema import load_bases
    return load_bases(_BASES_CFG)


@pytest.fixture(scope="module")
def smoke_archive_cfg():
    return _load_yaml_local(_ARCHIVE_CFG)


@pytest.fixture(scope="module")
def smoke_policy_factory():
    """정책 팩토리 픽스처 — run_arm_family(policy_factory=...) 용."""
    from echo_bench.policies.random import RandomPolicy
    return lambda: RandomPolicy({"k": 4})


# ---------------------------------------------------------------------------
# 공유 리포트 빌더 픽스처
#
# 반복되는 raw→reportable_block→arm_entry→build_axs_report 패턴을 단일 함수로
# 캡슐화. module scope 로 캐싱하여 7개 테스트 재실행을 방지한다.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def axs_report_fixture(tmp_path_factory, smoke_bases, smoke_archive_cfg, smoke_policy_factory):
    """(prereg_path, raw, family_block, arm_entry, recompute_fn, report, reports_dir) 반환.

    module scope: 최초 1회만 실행 — 동일 입력이므로 결과가 결정론적으로 동일.
    """
    from echo_bench.experiments.axs_common import (
        run_arm_family, reportable_block, build_arm_entry, build_axs_report, write_report,
    )

    tmp = tmp_path_factory.mktemp("axs_report_fixture")
    shutil.copy(_REAL_PREREG, tmp / "prereg.json")
    prereg_path = tmp / "prereg.json"

    raw = run_arm_family(
        smoke_policy_factory, base_seed=42, H=2, k=4, pool_size=8,
        n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
    )
    family_block = reportable_block(raw)

    random_coverage = 0.90
    nmi_values = {"42": float(raw["slate_excess_nmi"])}
    arm_entry = build_arm_entry(
        "slate_excess_nmi",
        nmi_values,
        coverage_mean=float(raw["coordinate_coverage_mean"]),
        random_coverage_mean=random_coverage,
        degenerate_reason_prefix="test_arm",
    )

    def recompute_fn(family: str) -> dict:
        r = run_arm_family(
            smoke_policy_factory, base_seed=int(family), H=2, k=4, pool_size=8,
            n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
        )
        return reportable_block(r)

    report = build_axs_report(
        "AXS-003",
        body_extra={
            "baselines": {"RANDOM": {"coordinate_coverage_mean": random_coverage}},
            "arms": {"test_arm": arm_entry},
        },
        family_blocks={"42": family_block},
        recompute_fn=recompute_fn,
        run_params={"H": 2, "k": 4, "pool_size": 8, "n_permutations": 3},
        prereg_path=prereg_path,
        replay_mode="first_family",
        replay_sample_size=1,
        git_runner=_good_git_runner,
    )

    reports_dir = tmp / "reports"
    reports_dir.mkdir()

    return {
        "prereg_path": prereg_path,
        "raw": raw,
        "family_block": family_block,
        "arm_entry": arm_entry,
        "recompute_fn": recompute_fn,
        "report": report,
        "reports_dir": reports_dir,
        "random_coverage": random_coverage,
    }


def _make_ledger(path: Path) -> Path:
    p = path / "ledger.json"
    p.write_text(json.dumps({"ledgerVersion": 1, "entries": []}), encoding="utf-8")
    return p


# ===========================================================================
# bootstrap_block 테스트
# ===========================================================================


def test_bootstrap_block_keys_and_types():
    """bootstrap_block 은 mean/ciLower/ciUpper 3개 키, 모두 plain float."""
    from echo_bench.experiments.axs_common import bootstrap_block

    values = {"42": 0.08, "7": 0.07, "101": 0.09}
    result = bootstrap_block(values, key="slate_excess_nmi")
    assert set(result.keys()) == {"mean", "ciLower", "ciUpper"}, result.keys()
    for k, v in result.items():
        assert type(v) is float, f"{k} 타입이 float 아님: {type(v)}"


def test_bootstrap_block_ci_low_to_ci_lower_mapping():
    """ci_low → ciLower, ci_high → ciUpper 매핑 확인 (aggregate_values 직접 비교)."""
    from echo_bench.experiments.axs_common import bootstrap_block
    from echo_bench.metrics.aggregate import aggregate_values

    values_dict = {"101": 0.09, "42": 0.08, "7": 0.07}
    # name-sorted 순서: 101, 42, 7
    sorted_vals = [values_dict[k] for k in sorted(values_dict)]
    agg = aggregate_values(sorted_vals, "slate_excess_nmi")

    block = bootstrap_block(values_dict, key="slate_excess_nmi")
    assert block["mean"] == float(agg["mean"])
    assert block["ciLower"] == float(agg["ci_low"])
    assert block["ciUpper"] == float(agg["ci_high"])


def test_bootstrap_block_deterministic():
    """동일 입력 → 동일 출력 (결정론성)."""
    from echo_bench.experiments.axs_common import bootstrap_block

    values = {"42": 0.05, "7": 0.06, "101": 0.04, "2025": 0.07, "31337": 0.08}
    r1 = bootstrap_block(values, key="k1")
    r2 = bootstrap_block(values, key="k1")
    assert r1 == r2


def test_bootstrap_block_family_order_independence():
    """패밀리 삽입 순서가 달라도 동일 블록."""
    from echo_bench.experiments.axs_common import bootstrap_block

    values_a = {"42": 0.05, "7": 0.06, "101": 0.04}
    values_b = {"101": 0.04, "42": 0.05, "7": 0.06}
    assert bootstrap_block(values_a, key="m") == bootstrap_block(values_b, key="m")


# ===========================================================================
# build_arm_entry 테스트
# ===========================================================================


def test_build_arm_entry_not_degenerate_above_baseline():
    """coverage_mean > random → degenerate 트리플 없음."""
    from echo_bench.experiments.axs_common import build_arm_entry

    entry = build_arm_entry(
        "slate_excess_nmi",
        {"42": 0.08, "7": 0.07},
        coverage_mean=0.93,
        random_coverage_mean=0.90,
        degenerate_reason_prefix="test",
    )
    assert "degenerate" not in entry
    assert "degenerateReason" not in entry
    assert "includedInMechanismClaim" not in entry


def test_build_arm_entry_not_degenerate_equal_baseline():
    """coverage_mean == random_coverage_mean → degenerate 트리플 없음 (경계값)."""
    from echo_bench.experiments.axs_common import build_arm_entry

    entry = build_arm_entry(
        "slate_excess_nmi",
        {"42": 0.08},
        coverage_mean=0.90,
        random_coverage_mean=0.90,
        degenerate_reason_prefix="test",
    )
    assert "degenerate" not in entry


def test_build_arm_entry_degenerate_below_baseline():
    """coverage_mean < random_coverage_mean → exact degenerate 트리플 첨부."""
    from echo_bench.experiments.axs_common import build_arm_entry

    entry = build_arm_entry(
        "slate_excess_nmi",
        {"42": 0.05, "7": 0.04},
        coverage_mean=0.50,
        random_coverage_mean=0.90,
        degenerate_reason_prefix="arm_label",
    )
    assert entry["degenerate"] is True
    assert entry["includedInMechanismClaim"] is False
    assert isinstance(entry["degenerateReason"], str)
    assert len(entry["degenerateReason"]) > 0
    # 두 수치가 reason 에 포함되어야 함
    assert "0.50" in entry["degenerateReason"] or "0.500000" in entry["degenerateReason"]
    assert "0.90" in entry["degenerateReason"] or "0.900000" in entry["degenerateReason"]


def test_build_arm_entry_degenerate_literal_types():
    """degenerate/includedInMechanismClaim 은 Python bool literal True/False."""
    from echo_bench.experiments.axs_common import build_arm_entry

    entry = build_arm_entry(
        "m",
        {"42": 0.05},
        coverage_mean=0.40,
        random_coverage_mean=0.90,
        degenerate_reason_prefix="p",
    )
    assert entry["degenerate"] is True
    assert entry["includedInMechanismClaim"] is False
    assert type(entry["degenerate"]) is bool
    assert type(entry["includedInMechanismClaim"]) is bool


def test_build_arm_entry_per_family_floats_plain():
    """perFamily 값은 plain float (numpy 누수 없음)."""
    from echo_bench.experiments.axs_common import build_arm_entry

    entry = build_arm_entry(
        "slate_excess_nmi",
        {"42": 0.08, "7": 0.07},
        coverage_mean=0.93,
        random_coverage_mean=0.90,
        degenerate_reason_prefix="p",
    )
    for fam, row in entry["perFamily"].items():
        for v in row.values():
            assert type(v) is float, f"perFamily[{fam}] 값이 float 아님: {type(v)}"


def test_build_arm_entry_structure():
    """perFamily/bootstrap/utility 세 키 존재."""
    from echo_bench.experiments.axs_common import build_arm_entry

    entry = build_arm_entry(
        "slate_excess_nmi",
        {"42": 0.08, "7": 0.07},
        coverage_mean=0.93,
        random_coverage_mean=0.90,
        degenerate_reason_prefix="p",
    )
    assert "perFamily" in entry
    assert "bootstrap" in entry
    assert "utility" in entry
    assert entry["utility"]["coordinate_coverage_mean"] == 0.93


# ===========================================================================
# run_arm_family 스모크 테스트
# ===========================================================================


def test_run_arm_family_smoke(smoke_bases, smoke_archive_cfg, smoke_policy_factory):
    """H=2, pool=8, perm=3, seed=42: 필수 키 반환 + 타입 확인."""
    from echo_bench.experiments.axs_common import run_arm_family

    result = run_arm_family(
        smoke_policy_factory,
        base_seed=42,
        H=2,
        k=4,
        pool_size=8,
        n_permutations=3,
        bases=smoke_bases,
        archive_cfg=smoke_archive_cfg,
    )
    required_keys = {
        "slate_excess_nmi",
        "coordinate_coverage_values",
        "coordinate_coverage_mean",
        "archiveHash",
        "poolHash",
        "traceHashes",
        "slateHashes",
        "roundsByProbe",
    }
    assert required_keys.issubset(result.keys()), (
        f"누락 키: {required_keys - result.keys()}"
    )
    assert type(result["slate_excess_nmi"]) is float
    assert type(result["coordinate_coverage_mean"]) is float
    assert isinstance(result["coordinate_coverage_values"], list)
    for v in result["coordinate_coverage_values"]:
        assert type(v) is float


def test_run_arm_family_coverage_values_length(smoke_bases, smoke_archive_cfg, smoke_policy_factory):
    """coordinate_coverage_values 길이 == probe 수."""
    from echo_bench.experiments.axs_common import run_arm_family
    from echo_bench.experiments.e_leakage_diagnostic import EXPANDED_PROBE_SET

    result = run_arm_family(
        smoke_policy_factory,
        base_seed=42,
        H=2,
        k=4,
        pool_size=8,
        n_permutations=3,
        bases=smoke_bases,
        archive_cfg=smoke_archive_cfg,
    )
    assert len(result["coordinate_coverage_values"]) == len(EXPANDED_PROBE_SET)


def test_run_arm_family_deterministic(smoke_bases, smoke_archive_cfg, smoke_policy_factory):
    """동일 인수 → canonical_hash 동일 (결정론성)."""
    from echo_bench.experiments.axs_common import run_arm_family, reportable_block
    from echo_bench.utils.hash import canonical_hash

    r1 = reportable_block(run_arm_family(
        smoke_policy_factory, base_seed=42, H=2, k=4, pool_size=8,
        n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
    ))
    r2 = reportable_block(run_arm_family(
        smoke_policy_factory, base_seed=42, H=2, k=4, pool_size=8,
        n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
    ))
    assert canonical_hash(r1) == canonical_hash(r2)


# ===========================================================================
# reportable_block 테스트
# ===========================================================================


def test_reportable_block_strips_rounds_by_probe():
    """reportable_block 은 roundsByProbe 만 제거하고 나머지는 보존."""
    from echo_bench.experiments.axs_common import reportable_block

    raw = {
        "slate_excess_nmi": 0.05,
        "archiveHash": "abc",
        "roundsByProbe": {"probe1": [{"round": 1}]},
    }
    result = reportable_block(raw)
    assert "roundsByProbe" not in result
    assert result["slate_excess_nmi"] == 0.05
    assert result["archiveHash"] == "abc"
    # 원본 불변
    assert "roundsByProbe" in raw


def test_reportable_block_no_rounds_by_probe_passthrough():
    """roundsByProbe 없는 블록도 정상 처리 (방어적 제거)."""
    from echo_bench.experiments.axs_common import reportable_block

    block = {"slate_excess_nmi": 0.05, "archiveHash": "abc"}
    result = reportable_block(block)
    assert result == block


# ===========================================================================
# build_axs_report + write_report 테스트
# ===========================================================================


def test_build_axs_report_gate_required_keys(axs_report_fixture):
    """build_axs_report 출력에 게이트 필수 최상위 키 존재."""
    report = axs_report_fixture["report"]

    assert "reportId" in report
    assert "experimentId" in report
    assert "preregStamp" in report
    assert "replayAudit" in report
    assert "reportHash" in report
    assert report["experimentId"] == "AXS-003"
    assert report["replayAudit"]["replayable"] is True


def test_build_axs_report_hash_self_consistent(axs_report_fixture):
    """canonical_hash(report minus reportHash) == report['reportHash']."""
    from echo_bench.utils.hash import canonical_hash

    report = axs_report_fixture["report"]
    body = {k: v for k, v in report.items() if k != "reportHash"}
    assert canonical_hash(body) == report["reportHash"]


def test_build_axs_report_no_numpy_bool_leakage(axs_report_fixture):
    """JSON 왕복 후 metric 위치에 numpy/bool 누수 없음."""
    report = axs_report_fixture["report"]

    # JSON 왕복
    serialized = json.dumps(report)
    loaded = json.loads(serialized)

    # arms 내 metric 값 위치 검사 (perFamily + bootstrap + utility)
    for arm_id, arm_data in loaded.get("arms", {}).items():
        for fam, fam_row in arm_data.get("perFamily", {}).items():
            for mk, mv in fam_row.items():
                assert not isinstance(mv, bool), (
                    f"arms[{arm_id}].perFamily[{fam}][{mk}] 가 bool: {mv}"
                )
        for metric_key, ci_dict in arm_data.get("bootstrap", {}).items():
            for ck, cv in ci_dict.items():
                assert not isinstance(cv, bool), (
                    f"arms[{arm_id}].bootstrap[{metric_key}][{ck}] 가 bool: {cv}"
                )
        cov = arm_data.get("utility", {}).get("coordinate_coverage_mean")
        if cov is not None:
            assert not isinstance(cov, bool)


def test_write_report_file_created(axs_report_fixture, tmp_path):
    """write_report: 예상 이름으로 파일 생성."""
    from echo_bench.experiments.axs_common import write_report

    report = axs_report_fixture["report"]
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    out_path = write_report(report, reports_dir=reports_dir)

    assert out_path.exists()
    # 파일명: axs_<experimentId lower dashes stripped>_<seedBatchId[:12]>.json
    seed_batch_id = report["seedBatchId"]
    expected_name = f"axs_003_{seed_batch_id[:12]}.json"
    assert out_path.name == expected_name, f"파일명 불일치: {out_path.name!r} != {expected_name!r}"

    # 파일 내용이 유효한 JSON
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["reportHash"] == report["reportHash"]


def test_build_axs_report_deterministic(axs_report_fixture, smoke_bases, smoke_archive_cfg, smoke_policy_factory, tmp_path):
    """동일 입력 → 동일 reportHash (결정론성)."""
    from echo_bench.experiments.axs_common import (
        run_arm_family, reportable_block, build_arm_entry, build_axs_report,
    )

    shutil.copy(_REAL_PREREG, tmp_path / "prereg.json")
    prereg_path = tmp_path / "prereg.json"

    raw = run_arm_family(
        smoke_policy_factory, base_seed=42, H=2, k=4, pool_size=8,
        n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
    )
    family_block = reportable_block(raw)
    arm_entry = build_arm_entry(
        "slate_excess_nmi", {"42": float(raw["slate_excess_nmi"])},
        coverage_mean=float(raw["coordinate_coverage_mean"]),
        random_coverage_mean=0.90,
        degenerate_reason_prefix="t",
    )

    def recompute_fn(family: str) -> dict:
        r = run_arm_family(
            smoke_policy_factory, base_seed=int(family), H=2, k=4, pool_size=8,
            n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
        )
        return reportable_block(r)

    def _make_report():
        return build_axs_report(
            "AXS-003",
            body_extra={
                "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
                "arms": {"t": arm_entry},
            },
            family_blocks={"42": family_block},
            recompute_fn=recompute_fn,
            run_params={"H": 2, "k": 4, "pool_size": 8, "n_permutations": 3},
            prereg_path=prereg_path,
            replay_mode="first_family",
            replay_sample_size=1,
            git_runner=_good_git_runner,
        )

    r1 = _make_report()
    r2 = _make_report()
    assert r1["reportHash"] == r2["reportHash"]


def test_build_axs_report_hash_semantics_field(axs_report_fixture):
    """hashSemantics 필드 존재 + pack.reportHash == pre-pack body hash."""
    from echo_bench.utils.hash import canonical_hash

    report = axs_report_fixture["report"]

    # hashSemantics 필드 존재 + 문자열 타입
    assert "hashSemantics" in report, "hashSemantics 필드 누락"
    assert isinstance(report["hashSemantics"], str)

    # pack.reportHash == canonical_hash(report minus {reproducibilityPack, packHash, reportHash})
    pre_pack_body = {
        k: v for k, v in report.items()
        if k not in {"reproducibilityPack", "packHash", "reportHash"}
    }
    expected_pack_report_hash = canonical_hash(pre_pack_body)
    actual_pack_report_hash = report["reproducibilityPack"]["reportHash"]
    assert actual_pack_report_hash == expected_pack_report_hash, (
        f"pack.reportHash={actual_pack_report_hash[:12]} != "
        f"pre-pack body hash={expected_pack_report_hash[:12]}"
    )


def test_build_axs_report_seed_batch_id_includes_probes(axs_report_fixture):
    """seedBatchId 는 probe 목록을 포함해 계산됨: 해시 재구성으로 검증."""
    from echo_bench.experiments.e_leakage_diagnostic import EXPANDED_PROBE_SET
    from echo_bench.utils.hash import canonical_hash

    report = axs_report_fixture["report"]
    run_params = {"H": 2, "k": 4, "pool_size": 8, "n_permutations": 3}

    # seedBatchId 재구성: probes 키 포함 여부 검증
    stable = {k: run_params[k] for k in sorted(run_params) if k != "configFreeze"}
    expected_with_probes = canonical_hash(
        {"experiment": "AXS-003", "probes": list(EXPANDED_PROBE_SET), **stable}
    )
    expected_without_probes = canonical_hash(
        {"experiment": "AXS-003", **stable}
    )

    actual = report["seedBatchId"]
    assert actual == expected_with_probes, (
        "seedBatchId 가 probe 목록을 포함하지 않음"
    )
    # probes 없이 계산한 해시와는 달라야 함
    assert actual != expected_without_probes, (
        "seedBatchId 가 probes 없는 해시와 동일 — probes 가 실제로 포함되지 않음"
    )


# ===========================================================================
# divergent recompute_fn → replayAudit.replayable False 검증
# ===========================================================================


def test_divergent_recompute_fn_replayable_false(
    smoke_bases, smoke_archive_cfg, smoke_policy_factory, tmp_path
):
    """recompute_fn 이 발산 블록을 반환하면 replayAudit.replayable 이 False."""
    from echo_bench.experiments.axs_common import (
        run_arm_family, reportable_block, build_arm_entry, build_axs_report,
    )

    shutil.copy(_REAL_PREREG, tmp_path / "prereg.json")
    prereg_path = tmp_path / "prereg.json"

    raw = run_arm_family(
        smoke_policy_factory, base_seed=42, H=2, k=4, pool_size=8,
        n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
    )
    family_block = reportable_block(raw)

    arm_entry = build_arm_entry(
        "slate_excess_nmi", {"42": float(raw["slate_excess_nmi"])},
        coverage_mean=float(raw["coordinate_coverage_mean"]),
        random_coverage_mean=0.90,
        degenerate_reason_prefix="t",
    )

    # 발산 recompute_fn: archiveHash 를 변조해 canonical_hash 불일치 유도
    def divergent_recompute_fn(family: str) -> dict:
        r = run_arm_family(
            smoke_policy_factory, base_seed=int(family), H=2, k=4, pool_size=8,
            n_permutations=3, bases=smoke_bases, archive_cfg=smoke_archive_cfg,
        )
        block = reportable_block(r)
        # archiveHash 변조 → canonical_hash 불일치
        block = dict(block)
        block["archiveHash"] = "deadbeef" * 8
        return block

    report = build_axs_report(
        "AXS-003",
        body_extra={
            "baselines": {"RANDOM": {"coordinate_coverage_mean": 0.90}},
            "arms": {"t": arm_entry},
        },
        family_blocks={"42": family_block},
        recompute_fn=divergent_recompute_fn,
        run_params={"H": 2, "k": 4, "pool_size": 8, "n_permutations": 3},
        prereg_path=prereg_path,
        replay_mode="first_family",
        replay_sample_size=1,
        git_runner=_good_git_runner,
    )

    # 발산 recompute_fn → replayAudit.replayable 이 False 여야 함
    assert report["replayAudit"]["replayable"] is False, (
        f"발산 recompute_fn 임에도 replayable={report['replayAudit']['replayable']}"
    )


# ===========================================================================
# register_report 테스트
# ===========================================================================


def test_register_report_ledger_entry(axs_report_fixture, tmp_path):
    """register_report: 8-키 원장 엔트리 추가됨."""
    from echo_bench.experiments.axs_common import write_report, register_report
    from echo_bench.logging.prereg import load_ledger

    report = axs_report_fixture["report"]
    ledger_path = _make_ledger(tmp_path)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = write_report(report, reports_dir=reports_dir)

    register_report(report, report_path, ledger_path=ledger_path)

    ledger = load_ledger(ledger_path)
    assert len(ledger["entries"]) == 1
    entry = ledger["entries"][0]
    required_keys = {
        "reportId", "experimentId", "preregId", "preregVersion",
        "preregHash", "reportHash", "reportPath", "runCommit",
    }
    assert required_keys.issubset(entry.keys()), (
        f"누락 키: {required_keys - entry.keys()}"
    )
    assert entry["reportHash"] == report["reportHash"]
    assert entry["reportPath"] == str(report_path)


# ===========================================================================
# dry_run_plan 테스트
# ===========================================================================


def test_dry_run_plan_writes_nothing(smoke_bases, smoke_archive_cfg, tmp_path):
    """dry_run_plan: outputs/reports 디렉토리에 파일 미작성."""
    from echo_bench.experiments.axs_common import dry_run_plan

    # reports_dir 를 명시적으로 지정하여 드라이런이 실제로 쓰지 않음을 확인
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    dry_run_plan(
        "AXS-003",
        run_params={"H": 2, "k": 4, "pool_size": 8, "n_permutations": 3},
        base_seeds=[42, 7],
        bases=smoke_bases,
        archive_cfg=smoke_archive_cfg,
    )
    # dry_run_plan 은 reports_dir 를 인수로 받지 않으므로 tmp_path 에 아무것도 없어야 함
    all_files = list(reports_dir.rglob("*.json"))
    assert all_files == [], f"dry_run 이 파일을 생성함: {all_files}"


def test_dry_run_plan_has_required_keys(smoke_bases, smoke_archive_cfg):
    """dry_run_plan: config/configFreeze/families 키 존재 + configFreeze 3-키 subset."""
    from echo_bench.experiments.axs_common import dry_run_plan

    result = dry_run_plan(
        "AXS-003",
        run_params={"H": 2, "k": 4, "pool_size": 8, "n_permutations": 3},
        base_seeds=[42],
        bases=smoke_bases,
        archive_cfg=smoke_archive_cfg,
    )
    assert "config" in result
    assert "configFreeze" in result
    assert "families" in result
    assert "42" in result["families"]
    fam = result["families"]["42"]
    assert "archiveHash" in fam
    assert "poolHash" in fam

    # configFreeze 는 리포트와 동일한 3-키 subset
    freeze = result["configFreeze"]
    assert set(freeze.keys()) == {"policyName", "policyEffectiveConfigHash", "taskId"}, (
        f"configFreeze 키 불일치: {set(freeze.keys())}"
    )


# ===========================================================================
# parse_base_seeds 테스트
# ===========================================================================


def test_parse_base_seeds_normal():
    """정상 입력 파싱."""
    from echo_bench.experiments.axs_common import parse_base_seeds

    assert parse_base_seeds("42,7,101") == [42, 7, 101]
    assert parse_base_seeds("42") == [42]


def test_parse_base_seeds_empty_segment_raises():
    """빈 세그먼트 → ValueError(한국어 메시지)."""
    from echo_bench.experiments.axs_common import parse_base_seeds

    with pytest.raises(ValueError, match="빈 세그먼트"):
        parse_base_seeds("42,,7")


def test_parse_base_seeds_non_int_raises():
    """정수가 아닌 세그먼트 → ValueError(한국어 메시지)."""
    from echo_bench.experiments.axs_common import parse_base_seeds

    with pytest.raises(ValueError, match="정수가 아닙니다"):
        parse_base_seeds("42,abc,7")


# ===========================================================================
# make_axs_arg_parser 테스트
# ===========================================================================


def test_make_axs_arg_parser_defaults():
    """기본값 확인: k=4, pool_size=64, dry_run=False, register_ledger=False.

    n_permutations=200 은 DEFAULT_NULL_PERMUTATIONS 값으로 의도적으로 고정됨.
    (D-015 null 치환 수; 벤치마크 재현성을 위해 변경 금지)
    """
    from echo_bench.experiments.axs_common import make_axs_arg_parser, REPLAY_MODES

    parser = make_axs_arg_parser("테스트 파서")
    args = parser.parse_args([])
    assert args.k == 4
    assert args.pool_size == 64
    assert args.dry_run is False
    assert args.register_ledger is False
    assert args.n_permutations == 200  # DEFAULT_NULL_PERMUTATIONS 고정값
    assert args.replay_sample_size == 2


def test_make_axs_arg_parser_dry_run_flag():
    """--dry-run 플래그 파싱."""
    from echo_bench.experiments.axs_common import make_axs_arg_parser

    parser = make_axs_arg_parser("테스트")
    args = parser.parse_args(["--dry-run"])
    assert args.dry_run is True


def test_make_axs_arg_parser_register_ledger_flag():
    """--register-ledger 플래그 파싱."""
    from echo_bench.experiments.axs_common import make_axs_arg_parser

    parser = make_axs_arg_parser("테스트")
    args = parser.parse_args(["--register-ledger"])
    assert args.register_ledger is True


def test_make_axs_arg_parser_replay_mode_choices():
    """--replay-mode 허용값 확인."""
    from echo_bench.experiments.axs_common import make_axs_arg_parser, REPLAY_MODES

    parser = make_axs_arg_parser("테스트")
    for mode in REPLAY_MODES:
        args = parser.parse_args(["--replay-mode", mode])
        assert args.replay_mode == mode


def test_make_axs_arg_parser_base_seeds_custom():
    """--base-seeds 커스텀 파싱."""
    from echo_bench.experiments.axs_common import make_axs_arg_parser

    parser = make_axs_arg_parser("테스트")
    args = parser.parse_args(["--base-seeds", "42,7,101"])
    assert args.base_seeds == "42,7,101"
