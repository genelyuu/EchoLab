"""Utility×leakage frontier figures (Task E-010).

Builds a HASHED, replayable frontier *data* artifact by joining E2 utility with
E3 leakage-proxy per policy. The data JSON (carrying a ``dataHash`` and the source
``reportHashes``) is the citable, replayable artifact; PNG rendering via
matplotlib is an OPTIONAL convenience whose provenance is the data hash. PNG bytes
are not a main-claim artifact (matplotlib versions differ) — the hashed data is.

The leakage axis is a PROXY (see docs/01); these are system-level coordinates over
a controlled testbed, not privacy, UX, or real-world claims.

Identifiers/paths stay English; runtime logs are Korean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "build_frontier_data",
    "write_frontier_data",
    "render_figures",
    "FRONTIER_SCHEMA",
]

_logger = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTIER_SCHEMA = "echo_bench.frontier.data"


def _latest(reports_dir: Path, prefix: str) -> Dict[str, Any]:
    """Load the lexicographically-last report JSON with the given prefix.

    Note: ``sorted(...)[-1]`` returns the lexicographically-last match, which
    equals temporal order only when filenames are ISO-8601 / sortable-prefixed.
    Report filenames are hash-prefixed (NOT time-sortable), so this is only a
    fallback for when no ``benchmark_index.json`` pins the current run — see
    :func:`_indexed_or_latest`.
    """
    matches = sorted(reports_dir.glob(f"{prefix}*.json"))
    if not matches:
        return {}
    with open(matches[-1], "r", encoding="utf-8") as handle:
        return json.load(handle)


# Maps the benchmark-index experiment name to its report filename prefix.
_INDEX_PREFIX = {
    "E2_POLICY_UTILITY": "e2_policy_",
    "E3_AUDIT": "e3_audit_",
}


def _indexed_or_latest(reports_dir: Path, exp_name: str, prefix: str) -> Dict[str, Any]:
    """Load the report the ``benchmark_index.json`` pins for ``exp_name``.

    Because report filenames are hash-prefixed (not time-sortable), a plain
    lexicographic ``_latest`` glob can pick a STALE report (e.g. a small
    regression-guard test run) instead of the current benchmark run. When a
    ``benchmark_index.json`` exists it authoritatively names the current run's
    reports via ``seedBatchId``; we load that file. Falls back to
    :func:`_latest` when there is no index or the pinned file is missing.
    """
    index_path = reports_dir / "benchmark_index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            for entry in index.get("experiments", []):
                if entry.get("experiment") == exp_name:
                    pinned = reports_dir / f"{prefix}{entry['seedBatchId'][:12]}.json"
                    if pinned.exists():
                        with open(pinned, "r", encoding="utf-8") as fh:
                            return json.load(fh)
        except (OSError, ValueError, KeyError):
            pass  # fall through to the lexicographic fallback
    return _latest(reports_dir, prefix)


def build_frontier_data(reports_dir: Any) -> Dict[str, Any]:
    """Join E2 utility and E3 leakage proxy per policy into hashed frontier data.

    When ``benchmark_index.json`` is present, the E2/E3 reports of the CURRENT
    benchmark run (pinned by the index) are used; otherwise it falls back to the
    lexicographically-last matching report.
    """
    rdir = Path(reports_dir)
    e2 = _indexed_or_latest(rdir, "E2_POLICY_UTILITY", "e2_policy_")
    e3 = _indexed_or_latest(rdir, "E3_AUDIT", "e3_audit_")

    if not e2:
        log_ko(_logger, "E2 보고서(e2_policy_*.json)를 찾을 수 없어 프론티어 포인트가 비어 있을 수 있습니다.")
    if not e3:
        log_ko(_logger, "E3 보고서(e3_audit_*.json)를 찾을 수 없어 프론티어 포인트가 비어 있을 수 있습니다.")

    utility_by_policy: Dict[str, Dict[str, Any]] = {}
    for row in e2.get("table", []):
        utility_by_policy[row["policy"]] = {
            "utility": float(row.get("coordinate_coverage", 0.0)),
            "traceOnly": bool(row.get("traceOnly", True)),
        }
    leakage_by_policy: Dict[str, float] = {}
    for row in e3.get("leakage", {}).get("table", []):
        leakage_by_policy[row["policy"]] = float(row.get("leakage_proxy", 0.0))

    points: List[Dict[str, Any]] = []
    for policy in sorted(set(utility_by_policy) & set(leakage_by_policy)):
        u = utility_by_policy[policy]
        points.append({
            "policy": policy,
            "utility": u["utility"],
            "leakage_proxy": leakage_by_policy[policy],
            "traceOnly": u["traceOnly"],
        })

    source_hashes = sorted(
        h for h in (e2.get("reportHash"), e3.get("reportHash")) if h
    )
    data: Dict[str, Any] = {
        "schema": FRONTIER_SCHEMA,
        "note": (
            "Utility (coordinate_coverage) vs leakage_proxy per policy over a "
            "controlled testbed. leakage_proxy is a PROXY, not a privacy claim. "
            "No real-world generalization."
        ),
        "points": points,
        "sourceReportHashes": source_hashes,
    }
    data["dataHash"] = canonical_hash(data)
    log_ko(_logger, f"프론티어 데이터 생성: points={len(points)}, dataHash={data['dataHash'][:12]}")
    return data


def write_frontier_data(frontier_data: Dict[str, Any], reports_dir: Any = None) -> str:
    """Persist ``frontier_data`` to ``outputs/reports/frontier_data.json``.

    The citable, replayable artifact is written here; ``build_frontier_data``
    stays pure (returns a dict, writes nothing). ``reports_dir`` may override the
    repo-rooted default (for tests).
    """
    rdir = Path(reports_dir) if reports_dir is not None else _REPO_ROOT / "outputs" / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "frontier_data.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(frontier_data, handle, indent=2, sort_keys=True, ensure_ascii=True)
    log_ko(_logger, f"프론티어 데이터 작성: path={path}, dataHash={frontier_data.get('dataHash', '')[:12]}")
    return str(path)


def render_figures(frontier_data: Dict[str, Any], out_dir: Any) -> List[str]:
    """Render PNGs if matplotlib is importable; otherwise return [] (no error)."""
    try:
        import matplotlib  # noqa: WPS433
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log_ko(_logger, f"matplotlib 미설치로 PNG 렌더링을 건너뜁니다: {type(exc).__name__}")
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pts = frontier_data["points"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([p["utility"] for p in pts], [p["leakage_proxy"] for p in pts])
    for p in pts:
        ax.annotate(p["policy"], (p["utility"], p["leakage_proxy"]))
    ax.set_xlabel("utility (coordinate_coverage)")
    ax.set_ylabel("leakage_proxy (PROXY)")
    ax.set_title("ECHO-Bench utility-leakage frontier (controlled testbed)")
    path = out / f"frontier_{frontier_data['dataHash'][:12]}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_ko(_logger, f"프론티어 PNG 작성: path={path}")
    return [str(path)]
