"""Deterministic report generator (Task F-004).

:func:`generate_report` builds a deterministic report json from a run's metrics
table and its :class:`~echo_bench.logging.manifest.RunManifest`. The report is a
pure function of ``(metrics, manifest, extra)``: identical inputs yield an
identical ``reportHash``.

The report embeds ``manifestHash`` and ``seedBatchId`` so every report is
traceable back to the run that produced it, and it carries the full
``ReproducibilityPack`` hash chain. ``reportHash`` is computed over the report
with that field removed, then inserted — closing the hash chain started by the
archive and trace hashes.

Per project guardrails the report contains ONLY system-level numeric claims; no
user-facing, satisfaction, emotion, wellbeing, privacy, legal, or generalization
language is produced here. Field names, metric names, and paths stay English; the
runtime log line emitted on write is Korean.
"""

from __future__ import annotations

import json
import os
from typing import Any, Union

from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.manifest import RunManifest
from echo_bench.utils.hash import canonical_hash

__all__ = ["generate_report", "write_report", "REPORT_FORMAT_VERSION"]

_logger = get_logger("echo_bench.logging.report")

# Format/version tag for the report envelope itself. Bumping it changes every
# produced reportHash (an explicit, machine-read version, not a content claim).
REPORT_FORMAT_VERSION = "report-fmt-1"


def generate_report(
    metrics_table: Union[dict, list],
    manifest: RunManifest,
    extra: Union[dict, None] = None,
) -> dict:
    """Return a deterministic report dict for a run.

    Parameters
    ----------
    metrics_table:
        System-level numeric metrics (dict or list). Stored verbatim under
        ``"metrics"``; callers are responsible for keeping it free of
        user-facing/generalization language per guardrails.
    manifest:
        The run's :class:`RunManifest`. Its ``manifestHash``, ``seedBatchId``,
        embedded ``pack`` hash chain, and component versions are embedded so the
        report is traceable to the run.
    extra:
        Optional additional system-level numeric/string metadata to embed under
        ``"extra"``. Defaults to an empty dict.

    Returns
    -------
    dict
        The report including a ``reportHash`` computed via
        :func:`~echo_bench.utils.hash.canonical_hash` over the report with the
        ``reportHash`` field removed. Identical inputs produce an identical hash.
    """
    if not isinstance(manifest, RunManifest):
        raise ValueError(
            "generate_report: manifest 는 RunManifest 이어야 합니다 "
            f"(현재 타입: {type(manifest).__name__})."
        )
    if not isinstance(metrics_table, (dict, list)):
        raise ValueError(
            "generate_report: metrics_table 는 dict 또는 list 여야 합니다 "
            f"(현재 타입: {type(metrics_table).__name__})."
        )
    if extra is not None and not isinstance(extra, dict):
        raise ValueError(
            "generate_report: extra 는 dict 또는 None 이어야 합니다 "
            f"(현재 타입: {type(extra).__name__})."
        )

    report: dict[str, Any] = {
        "reportFormatVersion": REPORT_FORMAT_VERSION,
        "seedBatchId": manifest.seedBatchId,
        "manifestHash": manifest.manifest_hash(),
        "manifest": manifest.to_dict(),
        "metrics": metrics_table,
        "extra": dict(extra) if extra is not None else {},
    }
    # reportHash closes over the entire report minus the reportHash field itself.
    report["reportHash"] = canonical_hash(report)
    return report


def write_report(report: dict, path: str) -> None:
    """Write a report dict to ``path`` as pretty json, creating dirs as needed.

    The file is written deterministically with sorted keys so a re-load of the
    file (and recomputation of ``reportHash`` over the report minus that field)
    reproduces the same hash. Emits a Korean runtime log line; ``path`` stays
    English.
    """
    if not isinstance(report, dict):
        raise ValueError(
            "write_report: report 는 dict 여야 합니다 "
            f"(현재 타입: {type(report).__name__})."
        )
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    log_ko(_logger, f"리포트를 기록했습니다: path={path}.")
