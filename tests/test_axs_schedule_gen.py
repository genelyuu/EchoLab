"""tests/test_axs_schedule_gen.py — TDD 픽스처 (AXS-P0 T3).

Coverage:
1. 결정론성: 동일 입력으로 두 번 호출 → 아티팩트 canonical_hash 동일
2. pilotFamily == "999", perRoundBonus 길이 == H, 모두 plain finite float
3. scheduleHash 자기 일관성: canonical_hash(body minus scheduleHash) == scheduleHash
4. 라운드트립: write_schedule → AxsYokedBonusPolicy 로드 및 슬레이트 선택
5. perRoundBonus 다양성 (H>=3): 비-동일 또는 전원 >= 0 유한 (약 보증)
6. CLI --dry-run: 출력 파일 미생성
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# ---- 경로 ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASES_CFG = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_AXS_UCB_CFG = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc if isinstance(doc, dict) else {}


# ===========================================================================
# 공유 픽스처
# ===========================================================================


@pytest.fixture(scope="module")
def smoke_bases():
    from echo_bench.basis.schema import load_bases
    return load_bases(_BASES_CFG)


@pytest.fixture(scope="module")
def smoke_archive_cfg():
    return _load_yaml(_ARCHIVE_CFG)


@pytest.fixture(scope="module")
def smoke_policy_config():
    """axs_ucb.yaml 을 로드하여 k=4 설정 반환 (smoke 파라미터 오버라이드 없음)."""
    return _load_yaml(_AXS_UCB_CFG)


# ---------------------------------------------------------------------------
# 스모크 매개변수: H=3, pool_size=8 (빠른 실행)
# ---------------------------------------------------------------------------
SMOKE_H = 3
SMOKE_K = 4
SMOKE_POOL = 8
SMOKE_BASE_SEED = 999


@pytest.fixture(scope="module")
def smoke_schedule(smoke_bases, smoke_archive_cfg, smoke_policy_config):
    """모듈 스코프 단일 스케줄 — 반복 생성 방지."""
    from echo_bench.experiments.axs_004c_schedule_gen import generate_yoked_schedule
    return generate_yoked_schedule(
        H=SMOKE_H,
        k=SMOKE_K,
        pool_size=SMOKE_POOL,
        base_seed=SMOKE_BASE_SEED,
        bases=smoke_bases,
        archive_cfg=smoke_archive_cfg,
        policy_config=smoke_policy_config,
    )


# ===========================================================================
# T1: 결정론성
# ===========================================================================


def test_determinism(smoke_bases, smoke_archive_cfg, smoke_policy_config):
    """동일 인수로 두 번 호출 → canonical_hash 동일."""
    from echo_bench.experiments.axs_004c_schedule_gen import generate_yoked_schedule
    from echo_bench.utils.hash import canonical_hash

    s1 = generate_yoked_schedule(
        H=SMOKE_H, k=SMOKE_K, pool_size=SMOKE_POOL, base_seed=SMOKE_BASE_SEED,
        bases=smoke_bases, archive_cfg=smoke_archive_cfg,
        policy_config=smoke_policy_config,
    )
    s2 = generate_yoked_schedule(
        H=SMOKE_H, k=SMOKE_K, pool_size=SMOKE_POOL, base_seed=SMOKE_BASE_SEED,
        bases=smoke_bases, archive_cfg=smoke_archive_cfg,
        policy_config=smoke_policy_config,
    )
    # scheduleHash 포함 전체 바디 동일해야 함
    assert canonical_hash(s1) == canonical_hash(s2), (
        "generate_yoked_schedule 가 결정론적이어야 합니다: 두 결과의 canonical_hash 불일치"
    )


# ===========================================================================
# T2: pilotFamily, perRoundBonus 길이, 타입
# ===========================================================================


def test_pilot_family_and_bonus_length(smoke_schedule):
    """pilotFamily == '999', perRoundBonus 길이 == H."""
    assert smoke_schedule["pilotFamily"] == "999", (
        f"pilotFamily 가 '999' 이어야 합니다 (실제: {smoke_schedule['pilotFamily']!r})"
    )
    assert len(smoke_schedule["perRoundBonus"]) == SMOKE_H, (
        f"perRoundBonus 길이가 H={SMOKE_H} 이어야 합니다 "
        f"(실제: {len(smoke_schedule['perRoundBonus'])})"
    )


def test_bonus_values_are_plain_finite_floats(smoke_schedule):
    """perRoundBonus 모든 값이 plain float, 유한, numpy/bool 아님."""
    for i, v in enumerate(smoke_schedule["perRoundBonus"]):
        # bool 은 int 서브클래스이므로 명시 체크
        assert not isinstance(v, bool), (
            f"perRoundBonus[{i}]={v!r} 가 bool 이어서는 안 됩니다"
        )
        assert isinstance(v, float), (
            f"perRoundBonus[{i}]={v!r} 가 plain Python float 이어야 합니다 "
            f"(type={type(v).__name__})"
        )
        assert math.isfinite(v), (
            f"perRoundBonus[{i}]={v!r} 가 유한해야 합니다"
        )


# ===========================================================================
# T3: scheduleHash 자기 일관성
# ===========================================================================


def test_schedule_hash_self_consistent(smoke_schedule):
    """canonical_hash(body minus scheduleHash) == scheduleHash."""
    from echo_bench.utils.hash import canonical_hash

    schedule_hash = smoke_schedule["scheduleHash"]
    body_without_hash = {k: v for k, v in smoke_schedule.items() if k != "scheduleHash"}
    recomputed = canonical_hash(body_without_hash)
    assert recomputed == schedule_hash, (
        f"scheduleHash 자기 일관성 실패: recomputed={recomputed!r}, "
        f"stored={schedule_hash!r}"
    )


# ===========================================================================
# T4: 라운드트립 — write_schedule → AxsYokedBonusPolicy 로드 및 선택
# ===========================================================================


def test_round_trip_load_and_select(tmp_path, smoke_bases, smoke_archive_cfg, smoke_schedule):
    """write_schedule 후 AxsYokedBonusPolicy 가 로드 + 선택 가능."""
    from echo_bench.experiments.axs_004c_schedule_gen import write_schedule
    from echo_bench.experiments.e_leakage_diagnostic import _build_family_pool
    from echo_bench.policies.axs_ucb import AxsYokedBonusPolicy
    from echo_bench.env.trace_state import TraceState

    out_path = tmp_path / "yoked_schedule_roundtrip.json"
    write_schedule(smoke_schedule, out_path)

    assert out_path.exists(), "write_schedule 후 파일이 존재해야 합니다"

    # AxsYokedBonusPolicy 로드
    policy_cfg = {
        "k": SMOKE_K,
        "schedule_path": str(out_path),
        "schedule_hash": smoke_schedule["scheduleHash"],
    }
    policy = AxsYokedBonusPolicy(policy_cfg)

    # 슬레이트 선택 테스트: 스모크 풀 준비
    _, pool, _ = _build_family_pool(smoke_bases, smoke_archive_cfg, SMOKE_BASE_SEED, SMOKE_POOL)
    trace = TraceState()

    # 첫 번째 라운드 선택
    from echo_bench.utils.hash import canonical_hash
    seed = int(canonical_hash({"seed": SMOKE_BASE_SEED, "round": 0}), 16)
    slate = policy.select(pool, trace, seed)
    assert len(slate) == SMOKE_K, (
        f"선택된 슬레이트 크기가 k={SMOKE_K} 이어야 합니다 (실제: {len(slate)})"
    )
    # 슬레이트 cardId 는 모두 풀 내 있어야 함
    pool_ids = {c["cardId"] for c in pool}
    for cid in slate:
        assert cid in pool_ids, f"슬레이트 cardId={cid!r} 가 풀에 없습니다"


# ===========================================================================
# T5: perRoundBonus 다양성 (H>=3 일 때 약 보증)
# ===========================================================================


def test_bonus_diversity_or_nonneg(smoke_schedule):
    """H>=3 케이스: perRoundBonus 가 모두 같지 않거나 (약 보증), 전원 >= 0 유한.

    배경:
    - UCB 보너스는 탐색 이력에 따라 라운드마다 변한다. H=3, pool=8 같은 스모크
      스케일에서는 분산이 작을 수 있으므로, 전원 동일 or 모두 >= 0 유한이면 통과.
    - 주 목표: numpy scalar / 무한 / NaN 누수 없음.
    """
    bonuses = smoke_schedule["perRoundBonus"]
    assert len(bonuses) >= 3, "이 테스트는 H>=3 에서만 의미 있습니다"

    all_finite = all(math.isfinite(b) for b in bonuses)
    all_nonneg = all(b >= 0.0 for b in bonuses)

    # 1차 보증: 모두 유한이고 음수 없음
    assert all_finite, "perRoundBonus 에 유한하지 않은 값이 있습니다"
    assert all_nonneg, (
        f"UCB 보너스(항상 ≥ 0)가 음수인 경우가 있습니다: {bonuses}"
    )

    # 2차 약 보증: 서로 다른 값이 있으면 better (분산 있음)
    # 스모크 스케일에서 플래키할 수 있으므로 경고 수준으로만 기록 — assert 하지 않음
    unique_count = len(set(bonuses))
    # 로그로 남겨두어 CI에서 확인 가능
    if unique_count == 1:
        import warnings
        warnings.warn(
            f"perRoundBonus 가 모두 동일합니다 ({bonuses[0]:.6f}). "
            "스모크 스케일에서는 분산이 없을 수 있습니다 (약 보증 통과).",
            stacklevel=2,
        )


# ===========================================================================
# T6: CLI --dry-run 파일 미생성
# ===========================================================================


def test_cli_dry_run_writes_nothing(tmp_path):
    """--dry-run 플래그 시 --out 경로에 파일이 생성되지 않아야 한다."""
    out_path = tmp_path / "should_not_exist.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "echo_bench.experiments.axs_004c_schedule_gen",
            "--H", str(SMOKE_H),
            "--k", str(SMOKE_K),
            "--pool-size", str(SMOKE_POOL),
            "--out", str(out_path),
            "--dry-run",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"--dry-run CLI 가 returncode=0 을 반환해야 합니다\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert not out_path.exists(), (
        "--dry-run 모드에서는 출력 파일이 생성되어서는 안 됩니다"
    )


# ===========================================================================
# 추가: 필수 키 확인
# ===========================================================================


def test_required_keys_present(smoke_schedule):
    """아티팩트 바디에 필수 키가 모두 있어야 한다."""
    required = {
        "scheduleId", "preregId", "preregVersion", "pilotFamily",
        "referenceArm", "derivation", "H", "k", "pool_size",
        "perRoundBonus", "configHash", "scheduleHash",
    }
    missing = required - set(smoke_schedule.keys())
    assert not missing, f"아티팩트에서 누락된 필수 키: {missing}"


def test_schedule_id_and_prereg(smoke_schedule):
    """scheduleId, preregId, preregVersion, referenceArm 값 확인."""
    assert smoke_schedule["scheduleId"] == "axs-004c-yoked-v1"
    assert smoke_schedule["preregId"] == "axs-mechanism"
    assert smoke_schedule["preregVersion"] == 1
    assert smoke_schedule["referenceArm"] == "axs_ucb_default"


def test_h_k_pool_in_artifact(smoke_schedule):
    """아티팩트 내 H, k, pool_size 가 호출 인수와 일치해야 한다."""
    assert smoke_schedule["H"] == SMOKE_H
    assert smoke_schedule["k"] == SMOKE_K
    assert smoke_schedule["pool_size"] == SMOKE_POOL
