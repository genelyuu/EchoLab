"""Preregistration hash chain + append-only run ledger (Task F-009).

Governance infrastructure for the Track M mechanism-claim ladder.

Design principles:
- The prereg is a committed, hashed contract; the ledger is an append-only
  registry so no evidence run can be silently buried.
- Every failure is fail-closed: a missing/corrupt prereg or ledger raises
  ``ValueError`` with a Korean operator message rather than returning a
  degraded result.
- git calls are injectable (``git_runner`` parameter) for deterministic tests.

All runtime log/exception messages are in Korean; identifiers, JSON keys,
metric names, and file paths stay English per the project convention.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "load_prereg",
    "prereg_hash",
    "build_prereg_stamp",
    "append_ledger_entry",
    "load_ledger",
    "entries_for_prereg",
]

_logger = get_logger(__name__)

# Repo root (resolved relative to this file so it is always correct regardless
# of the working directory when the module is imported).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Required top-level keys every prereg version must carry (fail-closed on any
# absence — the gate cannot operate on an incomplete contract).
_REQUIRED_PREREG_KEYS = (
    "preregId",
    "version",
    "primaryEndpoint",
    "utilityGuard",
    "pairedUnit",
    "evaluationFamilies",
    "signRule",
    "ciRule",
    "tieBreakCaveatMarker",
    "experiments",
    "claimTransitions",
    "degenerateArmPolicy",
    "amendmentPolicy",
)

# Additional keys required for v2+ amendments (amendment policy enforcement).
_AMENDMENT_REQUIRED_KEYS = ("supersedes", "changeJustification")

# Required fields in a ledger entry.
_REQUIRED_LEDGER_ENTRY_KEYS = (
    "reportId",
    "experimentId",
    "preregId",
    "preregVersion",
    "preregHash",
    "reportHash",
    "reportPath",
    "runCommit",
)


# ---------------------------------------------------------------------------
# load_prereg
# ---------------------------------------------------------------------------


def load_prereg(path: Any) -> Dict[str, Any]:
    """JSON-load a prereg file and validate required keys (fail-closed).

    Args:
        path: path-like pointing at the prereg JSON file.

    Returns:
        The prereg dict.

    Raises:
        ValueError: if any required key is missing, or if ``version > 1`` and
            ``supersedes`` / ``changeJustification`` are absent (amendment
            policy enforcement).
        OSError: if the file cannot be read.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            prereg = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"사전등록 파일 JSON 파싱 실패: {path} — {exc}"
        ) from exc

    if not isinstance(prereg, dict):
        raise ValueError(
            f"사전등록 파일이 JSON 객체(dict)가 아닙니다: {path}"
        )

    # Validate all required top-level keys.
    missing = [k for k in _REQUIRED_PREREG_KEYS if k not in prereg]
    if missing:
        raise ValueError(
            f"사전등록 파일에 필수 키가 누락되었습니다: {sorted(missing)} "
            f"(path={path})."
        )

    # version must be a plain int >= 1.  Reject bool (isinstance(True, int) is
    # True in Python), strings, floats, and zero/negative values — any of these
    # would silently bypass the amendment-policy check below.
    version = prereg.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(
            f"사전등록 'version' 필드는 1 이상의 정수여야 합니다 "
            f"(현재 값: {version!r}, 타입: {type(version).__name__}, path={path})."
        )

    # Amendment policy: version > 1 requires supersedes + changeJustification.
    if version > 1:
        amendment_missing = [k for k in _AMENDMENT_REQUIRED_KEYS if k not in prereg]
        if amendment_missing:
            raise ValueError(
                f"사전등록 수정판(version={version})에 필수 수정 키가 누락되었습니다: "
                f"{sorted(amendment_missing)}. 수정 시 supersedes 및 "
                f"changeJustification 을 모두 기입해야 합니다 (path={path})."
            )

    log_ko(
        _logger,
        "사전등록 파일 로드 완료: "
        f"preregId={prereg.get('preregId')}, version={version}, "
        f"path={path}.",
    )
    return prereg


# ---------------------------------------------------------------------------
# prereg_hash
# ---------------------------------------------------------------------------


def prereg_hash(prereg: Dict[str, Any]) -> str:
    """Return a deterministic sha256 hex digest of the prereg dict.

    Reuses ``canonical_hash`` from ``utils.hash`` (stable JSON canonicalization
    + sha256 + version tag). The hash is always computed externally — the prereg
    file itself never contains its own hash.

    Args:
        prereg: the prereg dict (as returned by :func:`load_prereg`).

    Returns:
        The hex digest string.
    """
    return canonical_hash(prereg)


# ---------------------------------------------------------------------------
# build_prereg_stamp
# ---------------------------------------------------------------------------


def _run_git(args: List[str], git_runner: Optional[Callable[[List[str]], str]]) -> str:
    """Execute a git command and return its stripped stdout.

    If ``git_runner`` is provided, delegates to it (test injection). Otherwise
    runs the real git subprocess in the repo root.
    """
    if git_runner is not None:
        return git_runner(args).strip()
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        stderr_detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr_detail = f" (stderr: {exc.stderr.strip()})"
        raise ValueError(
            f"git 명령 실행 실패: {args} — {exc}{stderr_detail}"
        ) from exc


def build_prereg_stamp(
    prereg_path: Any,
    *,
    git_runner: Optional[Callable[[List[str]], str]] = None,
) -> Dict[str, Any]:
    """Build a provenance stamp for embedding into experiment reports.

    Returns a dict with keys:
    - ``preregId``: the prereg's ID.
    - ``preregVersion``: the prereg's version.
    - ``preregPath``: the absolute path to the prereg file (str).
    - ``preregHash``: the canonical hash of the loaded prereg dict.
    - ``preregCommit``: the git commit that last touched the prereg file
      (``git log -1 --format=%H -- <path>``).
    - ``runCommit``: ``git rev-parse HEAD`` at stamp-build time.

    Args:
        prereg_path: path-like pointing at the prereg JSON file.
        git_runner: injectable callable ``(args: list[str]) -> str`` for tests.
            Defaults to running real git via subprocess.

    Raises:
        ValueError: if the prereg file does not validate, if ``preregCommit``
            is empty (file not yet committed), or if ``runCommit`` is empty.
    """
    prereg_path = Path(prereg_path)
    prereg = load_prereg(prereg_path)
    p_hash = prereg_hash(prereg)

    # preregCommit: git log -1 --format=%H -- <path>
    prereg_commit = _run_git(
        ["log", "-1", "--format=%H", "--", str(prereg_path)],
        git_runner,
    )
    if not prereg_commit:
        raise ValueError(
            "사전등록 파일의 git commit 해시를 가져올 수 없습니다. "
            "사전등록 파일이 커밋되지 않았습니다 — "
            "먼저 커밋한 뒤 스탬프를 생성하세요 "
            f"(path={prereg_path})."
        )

    # runCommit: git rev-parse HEAD
    run_commit = _run_git(["rev-parse", "HEAD"], git_runner)
    if not run_commit:
        raise ValueError(
            "현재 git HEAD 커밋 해시를 가져올 수 없습니다 "
            f"(git rev-parse HEAD 실패, repo_root={_REPO_ROOT})."
        )

    stamp = {
        "preregId": prereg["preregId"],
        "preregVersion": prereg["version"],
        "preregPath": str(prereg_path),
        "preregHash": p_hash,
        "preregCommit": prereg_commit,
        "runCommit": run_commit,
    }
    log_ko(
        _logger,
        "사전등록 스탬프 생성 완료: "
        f"preregId={stamp['preregId']}, version={stamp['preregVersion']}, "
        f"preregHash={p_hash[:12]}, preregCommit={prereg_commit[:12]}, "
        f"runCommit={run_commit[:12]}.",
    )
    return stamp


# ---------------------------------------------------------------------------
# load_ledger
# ---------------------------------------------------------------------------


def load_ledger(ledger_path: Any) -> Dict[str, Any]:
    """Load and validate a run ledger file.

    Args:
        ledger_path: path-like pointing at the ledger JSON file.

    Returns:
        The ledger dict (guaranteed to have an ``entries`` list).

    Raises:
        ValueError: if the file is not valid JSON, or if it is missing the
            ``entries`` list.
    """
    ledger_path = Path(ledger_path)
    try:
        with open(ledger_path, "r", encoding="utf-8") as handle:
            ledger = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"실행 원장 JSON 파싱 실패: {ledger_path} — {exc}"
        ) from exc

    if not isinstance(ledger, dict):
        raise ValueError(
            f"실행 원장이 JSON 객체(dict)가 아닙니다: {ledger_path}"
        )
    if "entries" not in ledger or not isinstance(ledger["entries"], list):
        raise ValueError(
            f"실행 원장에 'entries' 목록이 없습니다: {ledger_path}."
        )
    return ledger


# ---------------------------------------------------------------------------
# append_ledger_entry
# ---------------------------------------------------------------------------


def append_ledger_entry(ledger_path: Any, entry: Dict[str, Any]) -> None:
    """Append an entry to the run ledger (append-only; idempotent on exact duplicate).

    Duplicate detection:
    - If an existing entry has the same ``reportHash`` AND is byte-for-byte
      identical to *entry*, the call is a no-op (idempotent re-append).
    - If an existing entry has the same ``reportHash`` but ANY field differs,
      ``ValueError`` is raised — this is an integrity anomaly.

    Atomic write: the ledger is written to a ``.json.tmp`` sibling, fsynced,
    then atomically replaced via ``os.replace``.  Concurrent appends are not
    locked (single-operator CLI assumption).

    Args:
        ledger_path: path-like pointing at the ledger JSON file.
        entry: a dict with required keys ``{reportId, experimentId, preregId,
            preregVersion, preregHash, reportHash, reportPath, runCommit}``.

    Raises:
        ValueError: if ``entry`` is missing required keys, if the ledger file
            is corrupted/invalid JSON, if it lacks an ``entries`` list, or if
            an existing entry shares ``reportHash`` but has conflicting metadata.
    """
    # Validate entry keys up front (fail-closed).
    if not isinstance(entry, dict):
        raise ValueError(
            "원장 엔트리가 dict 가 아닙니다 "
            f"(현재 타입: {type(entry).__name__})."
        )
    missing = [k for k in _REQUIRED_LEDGER_ENTRY_KEYS if k not in entry]
    if missing:
        raise ValueError(
            f"원장 엔트리에 필수 키가 누락되었습니다: {sorted(missing)}."
        )

    ledger = load_ledger(ledger_path)

    # Duplicate / conflict check by reportHash.
    entry_hash = entry["reportHash"]
    for existing in ledger["entries"]:
        if existing.get("reportHash") == entry_hash:
            if existing == dict(entry):
                # Exact duplicate — idempotent no-op.
                log_ko(
                    _logger,
                    "원장 중복 등록 무시(no-op): "
                    f"reportHash={entry_hash[:12]} 가 이미 존재합니다 "
                    f"(reportId={entry.get('reportId')}).",
                )
                return
            # Same hash, different content — integrity anomaly.
            raise ValueError(
                "원장 무결성 이상: 같은 reportHash에 상이한 메타데이터가 감지되었습니다 "
                f"(reportHash={entry_hash[:12]}, "
                f"기존 reportId={existing.get('reportId')!r}, "
                f"신규 reportId={entry.get('reportId')!r})."
            )

    ledger["entries"].append(dict(entry))

    ledger_path = Path(ledger_path)
    tmp_path = ledger_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, ledger_path)
    except Exception:
        # Clean up stale tmp file if replacement failed.
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    log_ko(
        _logger,
        "원장 엔트리 추가 완료: "
        f"reportId={entry.get('reportId')}, "
        f"experimentId={entry.get('experimentId')}, "
        f"preregId={entry.get('preregId')}, "
        f"reportHash={entry['reportHash'][:12]}, "
        f"총 엔트리 수={len(ledger['entries'])}.",
    )


# ---------------------------------------------------------------------------
# entries_for_prereg
# ---------------------------------------------------------------------------


def entries_for_prereg(ledger: Dict[str, Any], prereg_id: str) -> List[Dict[str, Any]]:
    """Return all ledger entries whose ``preregId`` matches ``prereg_id``.

    Renamed from ``family_entries`` to avoid collision with the repo's
    seed-family vocabulary; this function filters by ``preregId``, not by
    seed family.

    Args:
        ledger: a ledger dict as returned by :func:`load_ledger`.
        prereg_id: the preregistration ID to filter by.

    Returns:
        A list of matching entry dicts (may be empty).
    """
    return [
        e for e in ledger.get("entries", [])
        if e.get("preregId") == prereg_id
    ]
