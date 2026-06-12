"""tests/test_axs_dead_calibration.py — N7-1 TDD 픽스처 (dead-arm 캘리브레이션).

v3 절대 utility floor 캘리브레이션용 DeadConstantPolicy + 러너 + 요약 작성기 테스트.

TDD: 테스트 먼저 작성(red), 구현 후 green.

테스트 전략:
- DeadConstantPolicy: trace-blind / probe-blind / 상수 슬레이트 / 결정론 / 허용성.
- 러너: --dry-run 파일 미작성, 평가 패밀리(42,7,101,2025,31337) 거부.
- 요약: floor 공식(합성 입력), pilot_v3 freeze_at_1 추출 통합(존재 시),
  합성 리포트 end-to-end 요약 작성 + summaryHash 자기일관성.

주의: 평가 패밀리는 어떤 테스트에서도 실행하지 않는다. 풀 생성은
critic 캘리브레이션 패밀리(123)만 사용한다.

로그/에러 메시지: 한국어. 식별자·JSON 키: 영어.
All identifiers, JSON keys, and path strings stay English; log messages are Korean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from echo_bench.env.constraints import check_slate
from echo_bench.env.round_runner import run_episode
from echo_bench.experiments.axs_common import load_default_configs
from echo_bench.experiments.axs_dead_calibration import (
    ARM_ID,
    EXPERIMENT_ID,
    DeadConstantPolicy,
    derive_floor,
    extract_dead_calibration,
    extract_live_key_arm_coverage,
    main,
    run_axs_dead_calibration,
    write_calibration_summary,
)
from echo_bench.experiments.e_leakage_diagnostic import _build_family_pool
from echo_bench.policies.axs_ucb import AxsUcbPolicy, TraceView
from echo_bench.policies.random import RandomPolicy
from echo_bench.probes.strategy_probes import get_probe
from echo_bench.utils.hash import canonical_hash

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PILOT_V3_DIR = _REPO_ROOT / "outputs" / "reports" / "pilot_v3"

# 캘리브레이션 패밀리만 사용 (평가 패밀리 42/7/101/2025/31337 은 금지).
_CAL_SEED = 123
_SMOKE_POOL_SIZE = 8


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _configs():
    return load_default_configs()


@pytest.fixture(scope="module")
def _pool123(_configs):
    bases, archive_cfg = _configs
    _archive_hash, pool, _pool_hash = _build_family_pool(
        bases, archive_cfg, _CAL_SEED, _SMOKE_POOL_SIZE
    )
    return pool


# ---------------------------------------------------------------------------
# 1. DeadConstantPolicy — trace-blind 상수성
# ---------------------------------------------------------------------------


def test_dead_policy_constant_across_traces(_pool123) -> None:
    """서로 다른 두 trace(길이/내용 상이) + 다른 seed → 동일 슬레이트."""
    policy = DeadConstantPolicy({"k": 4})

    trace_empty = TraceView([])
    trace_long = TraceView(
        [
            {"slate": ["a", "b", "c", "d"], "selectedCardId": "a"},
            {"slate": ["e", "f", "g", "h"], "selectedCardId": "h"},
            {"slate": ["a", "b", "c", "d"], "selectedCardId": "c"},
        ]
    )

    slate_1 = policy.select(_pool123, trace_empty, 111, {"k": 4})
    slate_2 = policy.select(_pool123, trace_long, 222, {"k": 4})

    assert slate_1 == slate_2, "trace/seed 가 달라도 슬레이트는 동일해야 한다"


# ---------------------------------------------------------------------------
# 2. 허용성: 실제 패밀리 풀에서 k=4 제약(≥3 distinct bases) 충족
# ---------------------------------------------------------------------------


def test_dead_policy_slate_admissible(_pool123, _configs) -> None:
    bases, _archive_cfg = _configs
    policy = DeadConstantPolicy({"k": 4})

    slate_ids = policy.select(_pool123, TraceView([]), 0, {"k": 4})

    assert len(slate_ids) == 4
    assert len(set(slate_ids)) == 4, "슬레이트 카드는 모두 서로 달라야 한다"

    by_id = {c["cardId"]: c for c in _pool123}
    slate_cards = [by_id[cid] for cid in slate_ids]
    distinct_bases = len({c["basis"] for c in slate_cards})
    assert distinct_bases >= 3, "k=4 슬레이트는 최소 3개 distinct basis 필요"

    ok, reason, _perm = check_slate(slate_cards, 4, bases, 0)
    assert ok, f"check_slate 거부: reason={reason}"


# ---------------------------------------------------------------------------
# 3. 결정론 + policy_version 안정성/구별성
# ---------------------------------------------------------------------------


def test_dead_policy_deterministic_and_version_distinct(_pool123) -> None:
    a = DeadConstantPolicy({"k": 4})
    b = DeadConstantPolicy({"k": 4})

    slate_a = a.select(_pool123, TraceView([]), 7_777, {"k": 4})
    slate_b = b.select(_pool123, TraceView([]), 7_777, {"k": 4})
    assert slate_a == slate_b, "동일 입력 → 동일 슬레이트"

    # policy_version 안정성 (인스턴스 간 동일)
    assert a.policy_version() == b.policy_version()

    # 기존 정책들과의 구별성
    assert a.policy_version() != AxsUcbPolicy({"k": 4}).policy_version()
    assert a.policy_version() != RandomPolicy({"k": 4}).policy_version()


# ---------------------------------------------------------------------------
# 4. 실제 하니스 경로(run_episode + probe select_fn)에서의 상수성
# ---------------------------------------------------------------------------


def test_dead_policy_constant_through_harness(_pool123, _configs) -> None:
    """짧은 에피소드(H=3)에서 모든 라운드의 슬레이트가 동일하고 probe 간에도 동일."""
    bases, _archive_cfg = _configs

    first_slate_by_probe: Dict[str, Any] = {}
    for probe_name in ("PREFER_HIGH_COMPLEXITY", "PREFER_LOW_SALIENCE"):
        probe = get_probe(probe_name)
        trace = run_episode(
            _pool123,
            DeadConstantPolicy({"k": 4}),
            _CAL_SEED,
            3,
            4,
            bases,
            select_fn=lambda slate, tr, seed, _p=probe: _p.select(slate, tr, seed),
        )
        slates = [r["slate"] for r in trace.rounds()]
        assert len(slates) == 3
        assert all(s == slates[0] for s in slates), (
            f"probe={probe_name}: 라운드 간 슬레이트가 상수가 아님: {slates}"
        )
        first_slate_by_probe[probe_name] = slates[0]

    vals = list(first_slate_by_probe.values())
    assert vals[0] == vals[1], "probe 가 달라도 슬레이트는 동일해야 한다 (probe-blind)"


# ---------------------------------------------------------------------------
# 5. --dry-run 은 파일을 쓰지 않는다 + 평가 패밀리 거부
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path) -> None:
    plan = run_axs_dead_calibration(
        [_CAL_SEED],
        H=2,
        k=4,
        pool_size=_SMOKE_POOL_SIZE,
        n_permutations=3,
        dry_run=True,
        reports_dir=tmp_path,
    )
    assert plan["dryRun"] is True
    assert list(tmp_path.iterdir()) == [], "dry-run 은 어떤 파일도 쓰면 안 된다"


@pytest.mark.parametrize("eval_seed", [42, 7, 101, 2025, 31337])
def test_runner_rejects_eval_families(tmp_path, eval_seed) -> None:
    with pytest.raises(ValueError, match="평가 패밀리"):
        run_axs_dead_calibration(
            [eval_seed],
            H=2,
            k=4,
            pool_size=_SMOKE_POOL_SIZE,
            n_permutations=3,
            dry_run=True,
            reports_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# 6. floor 공식 (합성 입력)
# ---------------------------------------------------------------------------


def test_derive_floor_synthetic() -> None:
    dead = {"123": 0.05, "124": 0.0625, "777": 0.04, "55555": 0.03}
    live = {"123": 0.1875, "124": 0.25, "777": 0.3, "55555": 0.2}

    fd = derive_floor(dead, live)

    assert fd["maxCoverageDead"] == pytest.approx(0.0625)
    assert fd["minLiveKeyArmCoverage"] == pytest.approx(0.1875)
    assert fd["absoluteFloor"] == pytest.approx((0.0625 + 0.1875) / 2.0)
    assert fd["formula"] == "midpoint(max(coverage_dead), min(live key-arm coverage))"


def test_derive_floor_rejects_inverted_ordering() -> None:
    """max(dead) >= min(live) 이면 캘리브레이션 무효 — fail closed."""
    with pytest.raises(ValueError, match="캘리브레이션이 무효"):
        derive_floor({"123": 0.5}, {"123": 0.2})


def test_derive_floor_rejects_family_mismatch() -> None:
    with pytest.raises(ValueError, match="dead 패밀리"):
        derive_floor({"123": 0.05}, {"124": 0.2})


# ---------------------------------------------------------------------------
# 7. pilot_v3 freeze_at_1 커버리지 추출 통합 (리포트 존재 시)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PILOT_V3_DIR.exists()
    or len(list(_PILOT_V3_DIR.glob("axs_009_*.json"))) == 0,
    reason="outputs/reports/pilot_v3 의 axs_009 리포트가 없음 (gitignored 로컬 산출물)",
)
def test_extract_live_key_arm_coverage_matches_files() -> None:
    coverage, report_hashes = extract_live_key_arm_coverage(_PILOT_V3_DIR)

    expected_cov: Dict[str, float] = {}
    expected_hashes: Dict[str, str] = {}
    for fp in sorted(_PILOT_V3_DIR.glob("axs_009_*.json")):
        report = json.loads(fp.read_text(encoding="utf-8"))
        arm = report["arms"]["freeze_at_1"]
        fams = list(arm["perFamily"].keys())
        assert len(fams) == 1, f"pilot_v3 리포트는 단일 패밀리여야 함: {fp.name}"
        expected_cov[fams[0]] = float(arm["utility"]["coordinate_coverage_mean"])
        expected_hashes[fams[0]] = report["reportHash"]

    assert coverage == expected_cov
    assert report_hashes == expected_hashes


# ---------------------------------------------------------------------------
# 8. 요약 작성 end-to-end (합성 리포트)
# ---------------------------------------------------------------------------

_FAMS = ["123", "124", "777", "55555"]


def _fake_dead_report(fam: str, cov: float, nmi: float) -> Dict[str, Any]:
    return {
        "experimentId": EXPERIMENT_ID,
        "reportHash": f"deadhash-{fam}",
        "runParams": {
            "H": 8,
            "k": 4,
            "pool_size": 64,
            "n_permutations": 200,
            "base_seeds": [int(fam)],
            "experiment": EXPERIMENT_ID,
        },
        "deadPolicyVersion": "dead-policy-version-synthetic",
        "arms": {
            ARM_ID: {
                "perFamily": {fam: {"slate_excess_nmi": nmi}},
                "bootstrap": {
                    "slate_excess_nmi": {"mean": nmi, "ciLower": nmi, "ciUpper": nmi}
                },
                "utility": {"coordinate_coverage_mean": cov},
            }
        },
    }


def _fake_live_report(fam: str, cov: float) -> Dict[str, Any]:
    return {
        "experimentId": "AXS-009",
        "reportHash": f"livehash-{fam}",
        "arms": {
            "freeze_at_1": {
                "perFamily": {fam: {"slate_excess_nmi": 0.3}},
                "utility": {"coordinate_coverage_mean": cov},
            }
        },
    }


def test_write_calibration_summary_synthetic(tmp_path) -> None:
    dead_dir = tmp_path / "dead"
    pilot_dir = tmp_path / "pilot"
    dead_dir.mkdir()
    pilot_dir.mkdir()

    dead_cov = {"123": 0.05, "124": 0.0625, "777": 0.04, "55555": 0.03}
    dead_nmi = {"123": -0.01, "124": 0.0, "777": -0.02, "55555": 0.01}
    live_cov = {"123": 0.1875, "124": 0.25, "777": 0.3, "55555": 0.2}

    for fam in _FAMS:
        (dead_dir / f"axs_dead-cal_{fam}.json").write_text(
            json.dumps(_fake_dead_report(fam, dead_cov[fam], dead_nmi[fam])),
            encoding="utf-8",
        )
        (pilot_dir / f"axs_009_{fam}.json").write_text(
            json.dumps(_fake_live_report(fam, live_cov[fam])), encoding="utf-8"
        )

    out_path = tmp_path / "axs_dead_calibration_v1.json"
    fake_sha = "f" * 40

    doc = write_calibration_summary(
        dead_reports_dir=dead_dir,
        pilot_v3_dir=pilot_dir,
        out_path=out_path,
        git_runner=lambda args: fake_sha,
    )

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == doc

    # 필수 키 구조 (스펙 EXACT)
    assert set(doc.keys()) == {
        "calibrationId",
        "purpose",
        "calibrationFamilies",
        "runParams",
        "deadArm",
        "liveKeyArmSource",
        "floorDerivation",
        "runCommit",
        "summaryHash",
    }
    assert doc["calibrationId"] == "axs-dead-calibration-v1"
    assert doc["calibrationFamilies"] == ["123", "124", "777", "55555"]
    assert doc["runCommit"] == fake_sha

    # runParams: 정확히 5개 키
    assert set(doc["runParams"].keys()) == {
        "H",
        "k",
        "pool_size",
        "n_permutations",
        "base_seeds",
    }
    assert doc["runParams"]["base_seeds"] == [123, 124, 777, 55555]

    # deadArm
    dead_arm = doc["deadArm"]
    assert set(dead_arm.keys()) == {
        "policyDescription",
        "policyVersion",
        "perFamily",
        "deadReportHashes",
    }
    for fam in _FAMS:
        assert dead_arm["perFamily"][fam]["coordinate_coverage_mean"] == pytest.approx(
            dead_cov[fam]
        )
        assert dead_arm["perFamily"][fam]["slate_excess_nmi_mean"] == pytest.approx(
            dead_nmi[fam]
        )
        assert dead_arm["deadReportHashes"][fam] == f"deadhash-{fam}"

    # liveKeyArmSource
    live_src = doc["liveKeyArmSource"]
    assert live_src["experiment"] == "AXS-009 (pilot_v3)"
    assert live_src["arm"] == "freeze_at_1"
    for fam in _FAMS:
        assert live_src["perFamilyCoverage"][fam] == pytest.approx(live_cov[fam])
        assert live_src["reportHashes"][fam] == f"livehash-{fam}"

    # floorDerivation
    fd = doc["floorDerivation"]
    assert fd["maxCoverageDead"] == pytest.approx(0.0625)
    assert fd["minLiveKeyArmCoverage"] == pytest.approx(0.1875)
    assert fd["absoluteFloor"] == pytest.approx((0.0625 + 0.1875) / 2.0)

    # summaryHash 자기일관성: summaryHash 필드를 제외한 canonical_hash
    body = {k: v for k, v in doc.items() if k != "summaryHash"}
    assert doc["summaryHash"] == canonical_hash(body)


# ---------------------------------------------------------------------------
# 9. fail-closed 추출기 가드 + CLI 플래그 조합 거부
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["absent", "empty"])
def test_extract_dead_rejects_missing_or_empty_base_seeds(tmp_path, mode) -> None:
    """runParams.base_seeds 누락/빈 리스트 → fail closed (한국어 ValueError)."""
    dead_dir = tmp_path / "dead"
    dead_dir.mkdir()
    report = _fake_dead_report("123", 0.05, 0.0)
    if mode == "absent":
        del report["runParams"]["base_seeds"]
    else:
        report["runParams"]["base_seeds"] = []
    (dead_dir / "axs_dead-cal_123.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="base_seeds 가 없거나 비어"):
        extract_dead_calibration(dead_dir)


def test_extract_dead_rejects_foreign_base_seeds(tmp_path) -> None:
    """base_seeds 가 [int(family)] 와 다르면 변조/외부 리포트 — fail closed."""
    dead_dir = tmp_path / "dead"
    dead_dir.mkdir()
    report = _fake_dead_report("123", 0.05, 0.0)
    report["runParams"]["base_seeds"] = [124]
    (dead_dir / "axs_dead-cal_123.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="패밀리 .* 일치하지 않습니다"):
        extract_dead_calibration(dead_dir)


def test_main_rejects_write_summary_with_dry_run(capsys) -> None:
    """--write-summary + --dry-run 동시 사용 → parser.error (SystemExit)."""
    with pytest.raises(SystemExit):
        main(["--write-summary", "--dry-run"])
    err = capsys.readouterr().err
    assert "--write-summary" in err
    assert "--dry-run" in err
    assert "함께 사용할 수 없습니다" in err


def test_extract_live_rejects_wrong_experiment_id(tmp_path) -> None:
    """axs_009 파일명이라도 experimentId != AXS-009 면 거부 (한국어 ValueError)."""
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    report = _fake_live_report("123", 0.2)
    report["experimentId"] = "AXS-DEAD-CAL"
    (pilot_dir / "axs_009_123.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="experimentId 가 AXS-009 가 아닙니다"):
        extract_live_key_arm_coverage(pilot_dir)
