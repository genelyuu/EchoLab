"""Reproducibility + guardrail regression guard (Task F-006).

Runs four invariant checks and fails (exit 1) if any regresses:
  - smoke_replay:    run_smoke twice -> identical reportHash
  - e2_replay:       run_e2_policy twice (small) -> identical reportHash
  - claim_scan:      the context-aware claim scanner finds no forbidden claim
  - forbidden_fields: a produced trace contains no FORBIDDEN_FIELDS key

No check may be allowlisted to pass; a real divergence is reported, never hidden.
Identifiers/paths stay English; runtime detail strings are Korean.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.round_runner import run_episode
from echo_bench.env.trace_state import FORBIDDEN_FIELDS
from echo_bench.experiments.e2_policy import run_e2_policy
from echo_bench.experiments.smoke import run_smoke
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.random import RandomPolicy
from echo_bench.tools.claim_check import scan_paths

__all__ = ["run_regression_guard", "main"]

_logger = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASES_CFG_PATH = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG_PATH = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_POLICY_CFG_PATH = _REPO_ROOT / "configs" / "policies" / "random.yaml"


def _check_smoke_replay() -> Dict[str, Any]:
    a = run_smoke(base_seed=1, H=4, k=4, pool_size=16)
    b = run_smoke(base_seed=1, H=4, k=4, pool_size=16)
    ok = a["reportHash"] == b["reportHash"]
    return {"name": "smoke_replay", "passed": ok,
            "detail": "스모크 reportHash 동일" if ok else "스모크 reportHash 불일치"}


def _check_e2_replay() -> Dict[str, Any]:
    a = run_e2_policy(base_seed=2, H=4, k=4, pool_size=16, n=3)
    b = run_e2_policy(base_seed=2, H=4, k=4, pool_size=16, n=3)
    ok = a["reportHash"] == b["reportHash"]
    return {"name": "e2_replay", "passed": ok,
            "detail": "E2 reportHash 동일" if ok else "E2 reportHash 불일치"}


def _check_claim_scan() -> Dict[str, Any]:
    findings = scan_paths([_REPO_ROOT / "docs", _REPO_ROOT / "outputs" / "reports"])
    ok = len(findings) == 0
    return {"name": "claim_scan", "passed": ok,
            "detail": "금지된 주장 없음" if ok else f"금지된 주장 {len(findings)}건"}


def _produce_trace_rounds() -> List[Dict[str, Any]]:
    """Produce a real episode trace via the same path ``run_smoke`` uses.

    Loads the bases, builds the reproducible archive, takes a deterministic
    candidate pool, and runs one RANDOM episode. Returns the trace's round
    records so the forbidden-field scan inspects genuine produced data (not a
    report summary that pre-filters its fields).
    """
    import yaml

    bases = load_bases(_BASES_CFG_PATH)
    with open(_ARCHIVE_CFG_PATH, "r", encoding="utf-8") as handle:
        archive_cfg = yaml.safe_load(handle) or {}
    with open(_POLICY_CFG_PATH, "r", encoding="utf-8") as handle:
        policy_cfg = dict(yaml.safe_load(handle) or {})
    policy_cfg["k"] = 4

    archive = build_archive(bases, archive_cfg, 3)
    pool = archive["cards"][:16]
    policy = RandomPolicy(policy_cfg)
    trace = run_episode(pool, policy, 3, 4, 4, bases_cfg=bases)
    return trace.rounds()


def _check_forbidden_fields() -> Dict[str, Any]:
    """Assert no FORBIDDEN_FIELDS key appears in a real produced trace.

    Inspects the round records of an actual episode trace (not a report
    summary). The trace's own ``append_round`` rejects forbidden fields at
    write time; this check independently re-verifies the produced data.
    """
    rounds = _produce_trace_rounds()
    bad: List[str] = []
    for record in rounds:
        bad.extend(key for key in record if key in FORBIDDEN_FIELDS)
    ok = not bad
    return {"name": "forbidden_fields", "passed": ok,
            "detail": "금지 필드 없음" if ok else f"금지 필드 발견: {sorted(set(bad))}"}


def run_regression_guard() -> Dict[str, Any]:
    """Run every invariant check and return ``{"ok": bool, "checks": [...]}``."""
    checks = [
        _check_smoke_replay(),
        _check_e2_replay(),
        _check_claim_scan(),
        _check_forbidden_fields(),
    ]
    ok = all(c["passed"] for c in checks)
    for c in checks:
        log_ko(_logger, f"리그레션 점검: name={c['name']}, passed={c['passed']}, {c['detail']}")
    return {"ok": ok, "checks": checks}


def main() -> None:
    """CLI entry point: exit 0 if all checks pass, 1 otherwise."""
    result = run_regression_guard()
    if result["ok"]:
        log_ko(_logger, "리그레션 가드 통과: 모든 불변식 유지")
        sys.exit(0)
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    log_ko(_logger, f"리그레션 가드 실패: {failed}")
    sys.exit(1)


if __name__ == "__main__":
    main()
