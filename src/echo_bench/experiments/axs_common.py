"""AXS 메커니즘 실험 공유 핵심 모듈 (AXS-P0 T2).

AXS-010/003/009/004c 네 개 실험 러너(이후 태스크에서 작성)가 공유하는
재사용 가능한 함수 모음. ladder_gate 가 요구하는 리포트 계약을 충족시키는
JSON 구조를 생성한다.

Hash semantics note
-------------------
AXS reportHash 와 e_leakage reportHash 의 계산 범위가 다르다.
- AXS: ``reportHash = canonical_hash(report - {reportHash})`` — 즉 pack/packHash
  포함 전체 바디에 대해 계산 (hashSemantics 필드 참조).
- e_leakage: ``reportHash = canonical_hash(pre-pack body)`` — pack 삽입 전 계산.
이 차이는 각 리포트의 ``hashSemantics`` 필드에 문서화되어 있다.

Guardrails
----------
- 모든 정책은 trace-only. user_id/persona/emotion/preference 벡터 없음.
- 수치는 모두 plain Python float/int (numpy scalar 금지).
- 모든 런타임 로그는 한국어.
- 식별자·JSON 키·경로는 영어 유지.

All identifiers, config keys, and paths stay English; runtime log messages
are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.horizon import default_h, load_horizon, validate_h
from echo_bench.env.round_runner import run_episode
from echo_bench.experiments.e_leakage_diagnostic import (
    EXPANDED_PROBE_SET,
    REPLAY_MODES,
    _build_family_pool,
    build_replay_section,
)
from echo_bench.metrics.leakage import DEFAULT_NULL_PERMUTATIONS
from echo_bench.policies.base import Policy
from echo_bench.experiments.e_seed_families import (
    DEFAULT_BASE_SEEDS,
    verify_trace_greedy_freeze,
)
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.prereg import append_ledger_entry, build_prereg_stamp
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.aggregate import aggregate_values
from echo_bench.metrics.separability import channel_separated_separability
from echo_bench.metrics.utility import coordinate_coverage
from echo_bench.probes.strategy_probes import get_probe
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "run_arm_family",
    "reportable_block",
    "bootstrap_block",
    "build_arm_entry",
    "build_axs_report",
    "write_report",
    "register_report",
    "make_axs_arg_parser",
    "dry_run_plan",
    "parse_base_seeds",
    "load_default_configs",
    "REPLAY_MODES",
]

_logger = get_logger(__name__)

# 저장소 루트: src/echo_bench/experiments/axs_common.py → parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASES_CFG_PATH = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG_PATH = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return doc if isinstance(doc, dict) else {}


def load_default_configs() -> tuple:
    """기본 configs/basis/bases.yaml + configs/archive/archive.yaml 로드.

    Returns:
        (bases, archive_cfg) — bases 는 load_bases 결과, archive_cfg 는 dict.

    실험 러너가 bases/archive 설정을 직접 로드할 때 사용.
    (e_leakage_diagnostic 의 _load_yaml 패턴 미러.)
    """
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    return bases, archive_cfg


def parse_base_seeds(s: str) -> List[int]:
    """쉼표 구분 base-seed 문자열 → int 리스트.

    빈 세그먼트나 정수가 아닌 세그먼트는 ValueError(한국어 메시지) 발생.

    Args:
        s: 예) "42,7,101"

    Returns:
        [42, 7, 101]
    """
    seeds: List[int] = []
    for segment in s.split(","):
        segment = segment.strip()
        if not segment:
            raise ValueError(
                f"base-seeds 파싱 실패: 빈 세그먼트가 포함되어 있습니다. 입력: {s!r}"
            )
        try:
            seeds.append(int(segment))
        except (ValueError, TypeError):
            raise ValueError(
                f"base-seeds 파싱 실패: '{segment}' 는 정수가 아닙니다. 입력: {s!r}"
            )
    return seeds


def _git_commit_hash(git_runner: Optional[Callable[[List[str]], str]] = None) -> str:
    """현재 HEAD 의 짧은 git 커밋 해시 반환. 실패 시 'uncommitted'."""
    if git_runner is not None:
        try:
            result = git_runner(["rev-parse", "--short", "HEAD"])
            return result.strip() if result.strip() else "uncommitted"
        except Exception:
            return "uncommitted"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = out.stdout.strip()
        return commit if commit else "uncommitted"
    except (subprocess.SubprocessError, OSError):
        return "uncommitted"


# ---------------------------------------------------------------------------
# 1. reportable_block
# ---------------------------------------------------------------------------


def reportable_block(raw_family_block: Dict[str, Any]) -> Dict[str, Any]:
    """raw run_arm_family 출력에서 roundsByProbe 를 제거한 리포트용 블록 반환.

    roundsByProbe 는 워킹 데이터(AXS-009 prefix 재계산용)로, 리포트 바디에서
    제외되어야 한다. 이 함수를 통해 stripping 계약을 단일 위치에서 관리한다.

    Args:
        raw_family_block: run_arm_family() 반환값 (roundsByProbe 포함 가능).

    Returns:
        roundsByProbe 를 제거한 새 dict. 원본은 변경되지 않는다.
    """
    return {k: v for k, v in raw_family_block.items() if k != "roundsByProbe"}


# ---------------------------------------------------------------------------
# 2. run_arm_family
# ---------------------------------------------------------------------------


def run_arm_family(
    policy_factory: Callable[[], Policy],
    base_seed: int,
    *,
    H: int,
    k: int,
    pool_size: int,
    n_permutations: int,
    bases: Any,
    archive_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """하나의 패밀리(base_seed)에 대해 전체 프로브 에피소드를 실행하고 블록 반환.

    e_leakage_diagnostic._run_family 패턴을 AXS 실험에 맞게 적용.
    policy_factory 는 프로브별 새 인스턴스를 생성하는 호출 가능 객체.
    EXPANDED_PROBE_SET 의 각 프로브별로 에피소드 1회씩 실행.

    정책은 probe별 새 인스턴스 — 상태 누적 차단.

    recompute_fn 계약:
        recompute_fn 이 이 함수의 출력을 기반으로 패밀리 블록을 재계산할 때,
        반환 블록은 reportable_block() 적용 후 원래 블록과 canonical_hash 가
        동일해야 한다. 즉:
            canonical_hash(reportable_block(recompute_fn(family)))
            == canonical_hash(reportable_block(original_block))
        build_axs_report 는 비교 전에 양쪽에 모두 reportable_block() 을 적용한다.

    반환:
        dict — slate_excess_nmi(float), coordinate_coverage_values(list[float]),
        coordinate_coverage_mean(float), archiveHash, poolHash,
        traceHashes(list[str]), slateHashes(list[str]),
        roundsByProbe(dict — 워킹 데이터, 리포트 바디에서 제외됨).
        reportable_block() 을 통해 roundsByProbe 를 제거하면 리포트용 블록이 된다.
    """
    archive_hash, pool, pool_hash = _build_family_pool(
        bases, archive_cfg, int(base_seed), pool_size
    )

    traces_by_probe: Dict[str, Any] = {}
    slate_hashes: List[str] = []
    trace_hashes: List[str] = []

    for probe_name in EXPANDED_PROBE_SET:
        probe = get_probe(probe_name)
        # 프로브별 신규 정책 인스턴스: e_leakage_diagnostic._run_family 와 동일 방식
        probe_trace = run_episode(
            pool,
            policy_factory(),
            int(base_seed),
            H,
            int(k),
            bases,
            select_fn=lambda slate, trace, seed, _p=probe: _p.select(
                slate, trace, seed
            ),
        )
        traces_by_probe[probe_name] = probe_trace
        trace_hashes.append(probe_trace.trace_hash())
        slate_hashes.append(
            canonical_hash([r["slate"] for r in probe_trace.rounds()])
        )

    # slate 채널 excess NMI (D-016)
    channel_sep = channel_separated_separability(
        traces_by_probe,
        n_permutations=n_permutations,
    )
    # 반올림 없이 raw float 반환 — e_leakage separability_row_fields 와의 바이트 호환성 유지
    slate_excess_nmi = float(channel_sep["slate_excess_nmi"])

    # coordinate_coverage: name-sorted 프로브 순서
    coverage_values: List[float] = []
    for probe_name in sorted(traces_by_probe.keys()):
        cov = float(coordinate_coverage(traces_by_probe[probe_name]))
        coverage_values.append(cov)

    coverage_mean = float(sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0

    # roundsByProbe: 리포트 바디에서는 제외하지만 AXS-009 prefix 재계산에 필요
    rounds_by_probe = {
        probe_name: [dict(r) for r in trace.rounds()]
        for probe_name, trace in sorted(traces_by_probe.items())
    }

    log_ko(
        _logger,
        "AXS 패밀리 실행 완료: "
        f"base_seed={base_seed}, probes={len(EXPANDED_PROBE_SET)}종, "
        f"slate_excess_nmi={slate_excess_nmi:+.6f}, "
        f"coverage_mean={coverage_mean:.6f}, poolHash={pool_hash[:12]}",
    )

    return {
        "slate_excess_nmi": slate_excess_nmi,
        "coordinate_coverage_values": coverage_values,
        "coordinate_coverage_mean": coverage_mean,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "traceHashes": trace_hashes,
        "slateHashes": slate_hashes,
        "roundsByProbe": rounds_by_probe,
    }


# ---------------------------------------------------------------------------
# 3. bootstrap_block
# ---------------------------------------------------------------------------


def bootstrap_block(
    values_by_family: Mapping[str, float],
    *,
    key: str,
) -> Dict[str, float]:
    """패밀리 값들을 name-sorted 순서로 집계해 CI 블록 반환.

    aggregate_values(ci_low→ciLower, ci_high→ciUpper) 키 매핑.
    모든 값은 plain float.
    """
    sorted_vals = [float(values_by_family[k]) for k in sorted(values_by_family)]
    agg = aggregate_values(sorted_vals, key)
    return {
        "mean": float(agg["mean"]),
        "ciLower": float(agg["ci_low"]),
        "ciUpper": float(agg["ci_high"]),
    }


# ---------------------------------------------------------------------------
# 4. build_arm_entry
# ---------------------------------------------------------------------------


def build_arm_entry(
    metric_name: str,
    per_family_values: Mapping[str, float],
    coverage_mean: float,
    random_coverage_mean: float,
    *,
    degenerate_reason_prefix: str,
) -> Dict[str, Any]:
    """arm 리포트 엔트리 조립.

    coverage_mean < random_coverage_mean 이면 정확한 degenerate 트리플 첨부.
    경계값(같은 경우)은 degenerate 로 처리하지 않는다.
    """
    per_family: Dict[str, Dict[str, float]] = {
        fam: {metric_name: float(v)}
        for fam, v in per_family_values.items()
    }

    bs = bootstrap_block(per_family_values, key=metric_name)

    entry: Dict[str, Any] = {
        "perFamily": per_family,
        "bootstrap": {metric_name: bs},
        "utility": {"coordinate_coverage_mean": float(coverage_mean)},
    }

    if float(coverage_mean) < float(random_coverage_mean):
        reason = (
            f"{degenerate_reason_prefix}: coordinate_coverage_mean "
            f"{float(coverage_mean):.6f} below RANDOM baseline "
            f"{float(random_coverage_mean):.6f}"
        )
        entry["degenerate"] = True
        entry["degenerateReason"] = reason
        entry["includedInMechanismClaim"] = False

    return entry


# ---------------------------------------------------------------------------
# 5. build_axs_report
# ---------------------------------------------------------------------------


def build_axs_report(
    experiment_id: str,
    *,
    body_extra: Dict[str, Any],
    family_blocks: Mapping[str, Dict[str, Any]],
    recompute_fn: Callable[[str], Mapping[str, Any]],
    run_params: Dict[str, Any],
    prereg_path: Any,
    replay_mode: str = "first_family",
    replay_sample_size: int = 2,
    git_runner: Optional[Callable[[List[str]], str]] = None,
) -> Dict[str, Any]:
    """AXS 실험 리포트 조립 (ladder_gate 계약 충족).

    body_extra 는 함수 진입 시 deep-copy 되어 원본 불변성이 보장된다(거버넌스 요건).

    roundsByProbe stripping 계약:
        family_blocks 의 각 값에서 roundsByProbe 를 제거한 뒤 해시 체인에 사용한다.
        recompute_fn 이 반환하는 블록도 동일하게 reportable_block() 을 적용한 후
        비교한다. 따라서 recompute_fn 은 roundsByProbe 포함·미포함 모두 허용되지만,
        reportable_block() 적용 후 원래 블록과 canonical_hash 가 일치해야 한다.

    recompute_fn 계약:
        recompute_fn(family: str) -> Mapping[str, Any]
        반환값은 reportable_block() 적용 후 원래 family_blocks[family] 와
        canonical_hash 가 동일해야 한다(재현 가능성 검증).
        roundsByProbe 는 stripping 후 비교되므로 포함 여부는 무관하다.

    순서:
    1. body_extra deep-copy
    2. verify_trace_greedy_freeze() → configFreeze
    3. build_prereg_stamp() → preregStamp
    4. seedBatchId 계산
    5. family_blocks 에서 roundsByProbe 방어적 제거
    6. recompute_fn 래퍼: 반환 블록에서도 roundsByProbe 방어적 제거
    7. build_replay_section()
    8. 해시 체인
    9. reportId 생성
    10. 리포트 바디 조립
    11. outputHash
    12. hashSemantics 삽입
    13. ReproducibilityPack (pre-pack body hash)
    14. reportHash (마지막)
    """
    # 1. body_extra deep-copy — 거버넌스 리포트 불변성 보장
    body_extra = copy.deepcopy(body_extra)

    # 2. C-011 config-freeze 게이트
    freeze = verify_trace_greedy_freeze()
    log_ko(_logger, f"AXS 리포트 조립 시작: experiment_id={experiment_id}")

    # 3. prereg 스탬프
    prereg_stamp = build_prereg_stamp(prereg_path, git_runner=git_runner)

    # 4. seedBatchId — run_params 에서 안정적 키들만 사용 + probe 목록 포함
    stable = {
        k: run_params[k]
        for k in sorted(run_params)
        if k not in ("configFreeze",)  # mutable 제외
    }
    seed_batch_id = canonical_hash(
        {
            "experiment": experiment_id,
            "probes": list(EXPANDED_PROBE_SET),
            **stable,
        }
    )

    # 5. family_blocks 에서 roundsByProbe 방어적 제거 (symmetric stripping)
    clean_family_blocks: Dict[str, Dict[str, Any]] = {
        family: reportable_block(block)
        for family, block in family_blocks.items()
    }

    # 6. recompute_fn 래퍼: 반환 블록에서도 roundsByProbe 방어적 제거
    def _clean_recompute_fn(family: str) -> Dict[str, Any]:
        return reportable_block(dict(recompute_fn(family)))

    # 7. 인라인 리플레이 감사
    replay_section = build_replay_section(
        clean_family_blocks,
        _clean_recompute_fn,
        replay_mode=replay_mode,
        replay_sample_size=replay_sample_size,
        config_key=seed_batch_id,
    )

    # 8. 해시 체인
    archive_hash = canonical_hash(
        {family: block["archiveHash"] for family, block in clean_family_blocks.items()}
    )
    pool_hash = canonical_hash(
        {family: block["poolHash"] for family, block in clean_family_blocks.items()}
    )
    slate_hash = canonical_hash(
        {family: block["slateHashes"] for family, block in clean_family_blocks.items()}
    )
    trace_hash = canonical_hash(
        {family: block["traceHashes"] for family, block in clean_family_blocks.items()}
    )

    # 9. reportId: 결정론적
    report_id = f"{experiment_id.lower().replace('-', '')}-{seed_batch_id[:12]}"

    # 10. 리포트 바디 조립 (reportHash 제외)
    report: Dict[str, Any] = {
        "reportId": report_id,
        "experimentId": experiment_id,
        "preregStamp": prereg_stamp,
        "configFreeze": {
            "policyName": freeze["policyName"],
            "policyEffectiveConfigHash": freeze["frozenHash"],
            "taskId": freeze["taskId"],
        },
        "seedBatchId": seed_batch_id,
        "replayAudit": replay_section,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        **body_extra,
    }

    # 11. outputHash: 현재까지 조립된 바디 기반
    output_hash = canonical_hash(dict(report))
    report["outputHash"] = output_hash

    # 12. hashSemantics: 두 reportHash 필드의 의미 명시.
    # AXS top-level reportHash = canonical_hash(전체 바디 - {reportHash}).
    # pack.reportHash = pack/packHash/reportHash 삽입 전 바디 해시 (pre-pack body hash).
    # e_leakage 는 pre-pack 방식만 사용; AXS 는 양쪽 모두 기록.
    report["hashSemantics"] = (
        "top-level reportHash = canonical_hash of the complete report minus the "
        "reportHash key (i.e. canonical_hash(report - {reportHash})); "
        "reproducibilityPack.reportHash = hash of the report body before the "
        "pack / packHash / reportHash fields were inserted (pre-pack body hash)."
    )

    # 13. pre-pack body hash: pack/packHash/reportHash 삽입 전 바디 해시
    #     → reproducibilityPack.reportHash 에 저장 (hashSemantics 참조)
    pre_pack_body = {
        k: v for k, v in report.items()
        if k not in {"reproducibilityPack", "packHash", "reportHash"}
    }
    pre_pack_body_hash = canonical_hash(pre_pack_body)

    # pack 생성 (pre-pack body hash 포함)
    commit_hash = _git_commit_hash(git_runner)
    pack = ReproducibilityPack(
        configHash=canonical_hash(run_params),
        commitHash=commit_hash,
        archiveHash=archive_hash,
        poolHash=pool_hash,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=pre_pack_body_hash,
        seedBatchId=seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 14. reportHash: pack+packHash 포함된 전체 바디에 대해 계산(마지막)
    report_without_hash = {k: v for k, v in report.items() if k != "reportHash"}
    report_hash = canonical_hash(report_without_hash)
    report["reportHash"] = report_hash

    log_ko(
        _logger,
        "AXS 리포트 조립 완료: "
        f"experimentId={experiment_id}, reportId={report_id}, "
        f"reportHash={report_hash[:12]}, seedBatchId={seed_batch_id[:12]}, "
        f"replayable={replay_section.get('replayable')}",
    )
    return report


# ---------------------------------------------------------------------------
# 6. write_report
# ---------------------------------------------------------------------------


def write_report(report: Dict[str, Any], *, reports_dir: Path) -> Path:
    """리포트를 JSON 으로 원자적 저장하고 경로 반환.

    tmp 형제 파일 → flush + fsync → os.replace 원자적 교체
    (append_ledger_entry 패턴 미러, src/echo_bench/logging/prereg.py 참조).

    파일명: axs_<experimentId lower dashes stripped>_<seedBatchId[:12]>.json
    예: axs_003_<12hex>.json (AXS-003), axs_010_<12hex>.json (AXS-010)
    """
    experiment_id: str = report["experimentId"]
    seed_batch_id: str = report["seedBatchId"]
    # "AXS-003" → "003", "AXS-010" → "010"
    short_id = experiment_id.upper().replace("AXS-", "").lower()
    file_name = f"axs_{short_id}_{seed_batch_id[:12]}.json"
    out_path = Path(reports_dir) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    log_ko(
        _logger,
        f"AXS 리포트 저장 완료: path={out_path}, "
        f"reportHash={report['reportHash'][:12]}",
    )
    return out_path


# ---------------------------------------------------------------------------
# 7. register_report
# ---------------------------------------------------------------------------


def register_report(
    report: Dict[str, Any],
    report_path: Path,
    *,
    ledger_path: Path,
) -> None:
    """리포트를 실행 원장에 등록 (8-키 엔트리).

    키: reportId, experimentId, preregId, preregVersion, preregHash,
        reportHash, reportPath, runCommit
    """
    stamp = report["preregStamp"]
    entry = {
        "reportId": report["reportId"],
        "experimentId": report["experimentId"],
        "preregId": stamp["preregId"],
        "preregVersion": stamp["preregVersion"],
        "preregHash": stamp["preregHash"],
        "reportHash": report["reportHash"],
        "reportPath": str(report_path),
        "runCommit": stamp["runCommit"],
    }
    append_ledger_entry(ledger_path, entry)
    log_ko(
        _logger,
        f"AXS 리포트 원장 등록 완료: reportId={entry['reportId']}, "
        f"reportHash={entry['reportHash'][:12]}",
    )


# ---------------------------------------------------------------------------
# 8. make_axs_arg_parser
# ---------------------------------------------------------------------------


def make_axs_arg_parser(description: str) -> argparse.ArgumentParser:
    """AXS 실험 러너 공유 CLI 인수 파서 반환.

    공통 플래그: --H --k --pool-size --n-permutations --base-seeds
    --replay-mode --replay-sample-size --reports-dir --dry-run --register-ledger
    """
    # horizon 기본값 로드
    try:
        horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
        h_default = default_h(horizon_cfg)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        log_ko(_logger, "horizon 설정 파일 로드 실패 — 기본값 H=8 사용")
        h_default = 8  # fallback

    parser = argparse.ArgumentParser(description=description)
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
        "--n-permutations", type=int, default=DEFAULT_NULL_PERMUTATIONS,
        dest="n_permutations",
        help=f"D-015 null 치환 수 (기본값: {DEFAULT_NULL_PERMUTATIONS})",
    )
    parser.add_argument(
        "--base-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_BASE_SEEDS),
        dest="base_seeds",
        help="쉼표 구분 base-seed 패밀리 목록",
    )
    parser.add_argument(
        "--replay-mode",
        type=str,
        choices=list(REPLAY_MODES),
        default="first_family",
        dest="replay_mode",
        help="F-008 인라인 리플레이 감사 모드",
    )
    parser.add_argument(
        "--replay-sample-size",
        type=int,
        default=2,
        dest="replay_sample_size",
        help="sampled_families 모드에서 재계산할 패밀리 수",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(_REPORTS_DIR),
        dest="reports_dir",
        help="리포트 출력 디렉토리",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="설정 검증 + 계획만 수행, 파일 미작성",
    )
    parser.add_argument(
        "--register-ledger",
        action="store_true",
        default=False,
        dest="register_ledger",
        help="실행 원장에 리포트 등록 (기본값: 미등록)",
    )
    return parser


# ---------------------------------------------------------------------------
# 9. dry_run_plan
# ---------------------------------------------------------------------------


def dry_run_plan(
    experiment_id: str,
    run_params: Dict[str, Any],
    base_seeds: Sequence[int],
    bases: Any,
    archive_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """드라이런: 파일 미작성, config + freeze + 패밀리별 아카이브/풀 해시 계획 반환.

    e_leakage_diagnostic.run_leakage_diagnostic dry_run 블록 미러.

    configFreeze 는 리포트와 동일한 3-키 subset 으로 노출된다:
        {policyName, policyEffectiveConfigHash, taskId}
    """
    freeze = verify_trace_greedy_freeze()
    seeds = [int(s) for s in base_seeds]

    families_plan: Dict[str, Any] = {}
    for seed in seeds:
        archive_hash, _pool, pool_hash = _build_family_pool(
            bases, archive_cfg, seed,
            int(run_params.get("pool_size", 64)),
        )
        families_plan[str(seed)] = {
            "archiveHash": archive_hash,
            "poolHash": pool_hash,
        }

    log_ko(
        _logger,
        f"AXS 드라이런 완료: experiment={experiment_id}, "
        f"families={seeds}, "
        f"pool_size={run_params.get('pool_size', 64)} (파일 미작성)",
    )

    return {
        "dryRun": True,
        "config": dict(run_params),
        "configFreeze": {
            "policyName": freeze["policyName"],
            "policyEffectiveConfigHash": freeze["frozenHash"],
            "taskId": freeze["taskId"],
        },
        "families": families_plan,
    }
