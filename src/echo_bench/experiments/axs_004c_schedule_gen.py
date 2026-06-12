"""AXS-004c yoked-bonus 스케줄 생성기 (AXS-P0 T3).

파일럿 패밀리(base_seed 999)에서 referenceArm(AXS_UCB_DEFAULT)의 라운드별
탐색 보너스를 측정하여 결정론적 yoked 스케줄을 생성한다.

Guardrails
----------
- trace-only 정책만 사용; user_id/persona/emotion/preference 벡터 없음.
- 수치는 모두 plain Python float (numpy scalar 금지).
- 런타임 로그는 한국어; 식별자·키·경로는 영어.

AxsYokedBonusPolicy 로드 검증:
    생성된 아티팩트를 반환 전에 AxsYokedBonusPolicy 의 자체 로더로 검증하여
    fail-closed 보장. 검증 실패 시 한국어 ValueError.

All identifiers, config keys, and paths stay English; runtime log messages
are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from echo_bench.basis.schema import load_bases
from echo_bench.env.horizon import default_h, load_horizon
from echo_bench.env.round_runner import run_round
from echo_bench.env.trace_state import TraceState
from echo_bench.experiments.axs_common import load_default_configs
from echo_bench.experiments.e_leakage_diagnostic import (
    EXPANDED_PROBE_SET,
    _build_family_pool,
)
from echo_bench.experiments.e_seed_families import verify_trace_greedy_freeze
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.axs_ucb import AxsUcbPolicy, AxsYokedBonusPolicy
from echo_bench.probes.strategy_probes import get_probe
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "generate_yoked_schedule",
    "write_schedule",
    "main",
]

_logger = get_logger(__name__)

# 저장소 루트: src/echo_bench/experiments/ -> parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASES_CFG_PATH = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG_PATH = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_AXS_UCB_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "axs_ucb.yaml"

# 사전등록된 스케줄 아티팩트 기본 경로 (prereg scheduleArtifactPath)
_DEFAULT_SCHEDULE_OUT = _REPO_ROOT / "configs" / "prereg" / "axs_004c_yoked_schedule_v1.json"

# 파일럿 패밀리 base_seed (prereg yokedSchedule.pilotFamily)
_PILOT_BASE_SEED = 999


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return doc if isinstance(doc, dict) else {}


def _run_probe_episode_manual(
    pool: List[Dict[str, Any]],
    policy: AxsUcbPolicy,
    base_seed: int,
    H: int,
    k: int,
    bases_cfg: Any,
    probe: Any,
) -> List[float]:
    """한 프로브에 대해 수동 라운드 루프를 실행하여 라운드별 bonus 목록 반환.

    run_episode 의 per-round seed 유도 방식을 정확히 복제:
        round_seed = int(canonical_hash({"seed": base_seed, "round": i}), 16)

    각 라운드 후 policy.last_score_components 에서 선택된 슬레이트 카드들의
    bonus 평균 → bonus_t.

    반환: [bonus_0, bonus_1, ..., bonus_{H-1}] (plain Python float 리스트)
    """
    trace = TraceState()
    bonus_by_round: List[float] = []

    for i in range(H):
        round_seed = int(canonical_hash({"seed": base_seed, "round": i}), 16)

        # run_round 는 policy.select() 를 내부 호출하므로 select_fn 으로 프로브 사용
        run_round(
            pool,
            policy,
            trace,
            round_seed,
            k,
            bases_cfg,
            select_fn=lambda slate, tr, seed, _p=probe: _p.select(slate, tr, seed),
        )

        # 방금 선택된 슬레이트 카드의 bonus 값 추출
        # policy.last_score_components: {cardId: {"mean": ..., "bonus": ...}}
        components = policy.last_score_components
        if not components:
            raise ValueError(
                f"AXS-004c 스케줄 생성: 라운드 {i} 에서 policy.last_score_components "
                "가 비어 있습니다. policy 구현을 확인하세요."
            )

        # 슬레이트에 있는 카드들의 bonus 평균 (last_score_components 는 슬레이트 카드만 포함)
        bonus_values = [
            float(comp["bonus"])
            for comp in components.values()
            if "bonus" in comp
        ]
        if not bonus_values:
            raise ValueError(
                f"AXS-004c 스케줄 생성: 라운드 {i} last_score_components 에 "
                "'bonus' 키가 없습니다."
            )
        if not all(math.isfinite(b) for b in bonus_values):
            raise ValueError(
                f"AXS-004c 스케줄 생성: 라운드 {i} bonus 값에 유한하지 않은 값이 "
                f"있습니다: {bonus_values}"
            )

        mean_bonus = sum(bonus_values) / len(bonus_values)
        bonus_by_round.append(float(mean_bonus))

    return bonus_by_round


def generate_yoked_schedule(
    *,
    H: int,
    k: int,
    pool_size: int,
    base_seed: int = _PILOT_BASE_SEED,
    bases: Any,
    archive_cfg: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """파일럿 패밀리에서 결정론적 yoked 스케줄 생성.

    EXPANDED_PROBE_SET (name-sorted) 각 프로브별 AXS_UCB_DEFAULT 에피소드를
    수동 실행하여 라운드별 bonus 를 추출한 뒤, 프로브 평균을 perRoundBonus 로 기록.

    Args:
        H: 라운드 수 (horizon).
        k: 슬레이트 크기.
        pool_size: 후보 풀 크기.
        base_seed: 파일럿 패밀리 base_seed (prereg 기본값 999).
        bases: 로드된 basis 설정 (load_bases 반환값).
        archive_cfg: 아카이브 설정 dict.
        policy_config: AxsUcbPolicy 설정 dict (axs_ucb.yaml 로드값).

    Returns:
        스케줄 아티팩트 dict. scheduleHash 가 마지막 키로 삽입됨.

    Raises:
        ValueError: 생성 실패, 유효성 검증 실패 (한국어 메시지).
    """
    # policy_config 에 k 오버라이드 (호출자 k 우선)
    effective_cfg: Dict[str, Any] = dict(policy_config)
    effective_cfg["k"] = k

    log_ko(
        _logger,
        f"AXS-004c 스케줄 생성 시작: H={H}, k={k}, pool_size={pool_size}, "
        f"base_seed={base_seed}, probes={len(EXPANDED_PROBE_SET)}종",
    )

    # 1. 파일럿 패밀리 풀 구성
    archive_hash, pool, pool_hash = _build_family_pool(
        bases, dict(archive_cfg), int(base_seed), pool_size
    )

    log_ko(
        _logger,
        f"AXS-004c 파일럿 풀 준비 완료: poolHash={pool_hash[:12]}, "
        f"archiveHash={archive_hash[:12]}",
    )

    # 2. EXPANDED_PROBE_SET (name-sorted) 각 프로브별 에피소드 실행
    # 프로브 순서: sorted(EXPANDED_PROBE_SET) — EXPANDED_PROBE_SET 는 이미 tuple(sorted())
    probe_names = sorted(EXPANDED_PROBE_SET)
    bonus_by_probe: Dict[str, List[float]] = {}

    for probe_name in probe_names:
        probe = get_probe(probe_name)
        # 프로브별 신규 정책 인스턴스 — 상태 누적 차단
        policy = AxsUcbPolicy(dict(effective_cfg))

        log_ko(
            _logger,
            f"AXS-004c 프로브 에피소드 실행: probe={probe_name}",
        )

        bonuses = _run_probe_episode_manual(
            pool, policy, int(base_seed), H, k, bases, probe
        )
        bonus_by_probe[probe_name] = bonuses

        log_ko(
            _logger,
            f"AXS-004c 프로브 완료: probe={probe_name}, "
            f"bonuses={[round(b, 6) for b in bonuses]}",
        )

    # 3. perRoundBonus[t] = 프로브 평균(bonus_t)
    per_round_bonus: List[float] = []
    for t in range(H):
        probe_bonuses = [bonus_by_probe[pn][t] for pn in probe_names]
        mean_t = sum(probe_bonuses) / len(probe_bonuses)
        per_round_bonus.append(float(mean_t))

    # 4. configHash
    config_hash = canonical_hash(dict(effective_cfg))

    # 5. 아티팩트 바디 구성 (scheduleHash 제외)
    body: Dict[str, Any] = {
        "scheduleId": "axs-004c-yoked-v1",
        "preregId": "axs-mechanism",
        "preregVersion": 1,
        "pilotFamily": str(base_seed),
        "referenceArm": "axs_ucb_default",
        "derivation": (
            "deterministic from pilot config + seed: mean per-round slate bonus "
            "of axs_ucb_default over EXPANDED_PROBE_SET on pilot family 999"
        ),
        "H": H,
        "k": k,
        "pool_size": pool_size,
        "perRoundBonus": per_round_bonus,
        "configHash": config_hash,
    }

    # 6. scheduleHash = canonical_hash(body minus scheduleHash) — 마지막 삽입
    schedule_hash = canonical_hash(body)
    schedule: Dict[str, Any] = dict(body)
    schedule["scheduleHash"] = schedule_hash

    # 7. AxsYokedBonusPolicy 자체 로더로 검증 (fail-closed)
    _validate_with_yoked_policy(schedule)

    log_ko(
        _logger,
        f"AXS-004c 스케줄 생성 완료: scheduleId=axs-004c-yoked-v1, "
        f"scheduleHash={schedule_hash[:12]}, H={H}, "
        f"perRoundBonus={[round(b, 6) for b in per_round_bonus]}",
    )

    return schedule


def _validate_with_yoked_policy(schedule: Dict[str, Any]) -> None:
    """AxsYokedBonusPolicy 의 자체 로더로 아티팩트 검증.

    임시 파일에 스케줄을 기록한 뒤 AxsYokedBonusPolicy 인스턴스를 생성하여
    _load_and_verify_schedule() 를 호출. 실패 시 한국어 ValueError 를 전파.

    Args:
        schedule: 생성된 스케줄 아티팩트 dict (scheduleHash 포함).

    Raises:
        ValueError: 검증 실패 (AxsYokedBonusPolicy 의 한국어 메시지 전파).
    """
    # 임시 파일에 기록
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp_f:
        tmp_path = tmp_f.name
        json.dump(schedule, tmp_f, indent=2, sort_keys=True, ensure_ascii=True)
        tmp_f.flush()
        os.fsync(tmp_f.fileno())

    try:
        policy_cfg = {
            "k": schedule["k"],
            "schedule_path": tmp_path,
            "schedule_hash": schedule["scheduleHash"],
        }
        policy = AxsYokedBonusPolicy(policy_cfg)
        # 실제 로드 + 검증 트리거
        policy._load_and_verify_schedule()
        log_ko(
            _logger,
            "AXS-004c 스케줄 AxsYokedBonusPolicy 검증 통과: "
            f"scheduleHash={schedule['scheduleHash'][:12]}",
        )
    except ValueError as exc:
        raise ValueError(
            f"AXS-004c 스케줄이 AxsYokedBonusPolicy 검증을 통과하지 못했습니다: {exc}"
        ) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def write_schedule(schedule: Dict[str, Any], path: Any) -> None:
    """스케줄 아티팩트를 JSON 으로 원자적 저장.

    tmp 형제 파일 → flush + fsync → os.replace 원자적 교체
    (axs_common.write_report 패턴 미러).

    Args:
        schedule: 생성된 스케줄 아티팩트 dict.
        path: 저장 경로 (str 또는 Path).

    Raises:
        OSError: 파일 시스템 오류.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(schedule, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    log_ko(
        _logger,
        f"AXS-004c 스케줄 저장 완료: path={out_path}, "
        f"scheduleHash={schedule.get('scheduleHash', 'N/A')[:12]}",
    )


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 진입점.

    플래그: --H --k --pool-size --out --dry-run
    --dry-run: 계획만 출력, 파일 미작성.
    """
    # horizon 기본값 로드
    try:
        horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
        h_default = default_h(horizon_cfg)
    except Exception:
        log_ko(_logger, "horizon 설정 파일 로드 실패 — 기본값 H=8 사용")
        h_default = 8

    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.axs_004c_schedule_gen",
        description="AXS-004c yoked 스케줄 생성기",
    )
    parser.add_argument(
        "--H", type=int, default=None,
        help=f"horizon (기본값: config default {h_default})",
    )
    parser.add_argument("--k", type=int, default=4, help="슬레이트 크기")
    parser.add_argument(
        "--pool-size", type=int, default=64, dest="pool_size",
        help="후보 풀 크기",
    )
    parser.add_argument(
        "--out", type=str, default=str(_DEFAULT_SCHEDULE_OUT), dest="out",
        help="스케줄 출력 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="계획만 출력, 파일 미작성",
    )

    args = parser.parse_args(argv)

    H = args.H if args.H is not None else h_default

    # 기본 설정 로드
    bases, archive_cfg = load_default_configs()
    policy_config = _load_yaml(_AXS_UCB_CFG_PATH)

    if args.dry_run:
        # 드라이런: 파일럿 풀 해시만 계산하고 파일 미작성
        from echo_bench.experiments.e_leakage_diagnostic import _build_family_pool

        archive_hash, _pool, pool_hash = _build_family_pool(
            bases, dict(archive_cfg), _PILOT_BASE_SEED, args.pool_size
        )
        config_hash = canonical_hash(dict(policy_config))

        log_ko(
            _logger,
            "AXS-004c 드라이런 완료 (파일 미작성):\n"
            f"  H={H}, k={args.k}, pool_size={args.pool_size}\n"
            f"  base_seed={_PILOT_BASE_SEED} (파일럿 패밀리)\n"
            f"  poolHash={pool_hash[:12]}\n"
            f"  archiveHash={archive_hash[:12]}\n"
            f"  configHash={config_hash[:12]}\n"
            f"  out={args.out}",
        )
        print(
            f"[AXS-004c 드라이런] H={H}, k={args.k}, pool_size={args.pool_size}, "
            f"base_seed={_PILOT_BASE_SEED}\n"
            f"poolHash={pool_hash}\narchiveHash={archive_hash}\nconfigHash={config_hash}\n"
            f"(파일 미작성: --dry-run 모드)"
        )
        return

    # 실제 실행 전 verify_trace_greedy_freeze 게이트
    verify_trace_greedy_freeze()

    schedule = generate_yoked_schedule(
        H=H,
        k=args.k,
        pool_size=args.pool_size,
        base_seed=_PILOT_BASE_SEED,
        bases=bases,
        archive_cfg=dict(archive_cfg),
        policy_config=policy_config,
    )

    out_path = Path(args.out)
    write_schedule(schedule, out_path)

    log_ko(
        _logger,
        f"AXS-004c 스케줄 생성 및 저장 완료: path={out_path}, "
        f"scheduleHash={schedule['scheduleHash'][:12]}",
    )
    print(
        f"[AXS-004c] 스케줄 생성 완료\n"
        f"path: {out_path}\n"
        f"scheduleHash: {schedule['scheduleHash']}\n"
        f"H={H}, k={args.k}, pool_size={args.pool_size}, "
        f"perRoundBonus={[round(b, 6) for b in schedule['perRoundBonus']]}"
    )


if __name__ == "__main__":
    main()
