"""Utility×leakage frontier figures (Task E-010) and the channel-resolved
utility×separability frontier (Task E-022, TRD E-022).

Builds HASHED, replayable frontier *data* artifacts. E-010 joins E2 utility
with E3 leakage-proxy per policy; E-022 joins E2 utility with the E-019
expanded leakage-diagnostic per policy × channel (slate / selection /
combined), carrying the null-corrected excess separability CIs and the
Track S/N assignment of claim ladder v2 (docs/12_CLAIM_LADDER.md). The data
JSON (carrying a ``dataHash`` and the source ``reportHashes``) is the citable,
replayable artifact; PNG rendering via matplotlib is an OPTIONAL convenience
whose provenance is the data hash. PNG bytes are not a main-claim artifact
(matplotlib versions differ) — the hashed data is.

The separability axis is a PROXY DIAGNOSTIC (G-020 primary label
``probe_separability_proxy``; see docs/01 and docs/12): these are system-level
coordinates over a controlled testbed, not privacy, UX, or real-world claims,
and never an improvement claim.

Identifiers/paths stay English; runtime logs are Korean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.leakage import PRIMARY_METRIC_NAME
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "build_frontier_data",
    "write_frontier_data",
    "render_figures",
    "FRONTIER_SCHEMA",
    "build_separability_frontier_data",
    "write_separability_frontier_data",
    "render_separability_figures",
    "SEPARABILITY_FRONTIER_SCHEMA",
    "SEPARABILITY_CHANNELS",
    "main",
]

_logger = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTIER_SCHEMA = "echo_bench.frontier.data"
SEPARABILITY_FRONTIER_SCHEMA = "echo_bench.frontier.separability_data"

#: D-016 channel order, mirrored in the E-019 diagnostic report.
SEPARABILITY_CHANNELS = ("slate", "selection", "combined")


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


# ---------------------------------------------------------------------------
# E-022 (TRD E-022): channel-resolved utility×separability frontier
# ---------------------------------------------------------------------------

#: The excess-block fields carried verbatim into each frontier point (a strict
#: subset of the E-019 ``crossFamilyExcess`` per-channel block).
_EXCESS_CARRY_KEYS = (
    "mean",
    "ci_low",
    "ci_high",
    "sufficient_n",
    "signConsistent",
    "allFamiliesUnsaturated",
)

#: English documentation of the trackAssignment derivation, embedded in the
#: artifact so the rule travels with the data (claim ladder v2, docs/12).
_TRACK_ASSIGNMENT_RULE = (
    "trackAssignment derivation: 'S' when the combined-channel Track S "
    "activation conditions all hold (trackLConditions allConditionsHold, "
    "docs/12_CLAIM_LADDER.md Section 5); otherwise 'N' when the "
    "combined-channel excess shows no positive above-null probe separability "
    "(mean <= 0, or ci_low <= 0 so the CI does not positively exclude 0) AND "
    "all families are unsaturated (allFamiliesUnsaturated true); otherwise "
    "'indeterminate' (positive excess without the activation conditions, any "
    "saturation, or a policy absent from the diagnostic report). Diagnostic "
    "register only — not a comparative, privacy, or improvement claim."
)

_SEPARABILITY_NOTE = (
    f"Utility (coordinate_coverage) vs {PRIMARY_METRIC_NAME} excess "
    "separability (null-corrected excess NMI, per channel: slate / selection "
    "/ combined) per policy over a controlled testbed. Probe separability is "
    "a DIAGNOSTIC axis (claim ladder v2 Track S / Track N, "
    "docs/12_CLAIM_LADDER.md) computed against instrumented strategy probes; "
    "the value is a PROXY, not a privacy claim, and a frontier position is "
    "never an improvement claim. This hashed data JSON is the citable "
    "artifact; PNG renderings are a non-citable convenience. No real-world "
    "generalization."
)

_MISSING_REPORT_NOTE = (
    " One or more source reports (E2 policy utility / expanded leakage "
    "diagnostic) were missing when this artifact was built, so points may be "
    "empty or one-sided."
)


def _derive_track_assignment(
    combined_excess: Optional[Dict[str, Any]],
    combined_conditions_hold: Optional[bool],
) -> str:
    """Derive the Track S / Track N / indeterminate assignment (combined channel).

    - ``"S"``: the four Track S activation conditions all hold on the combined
      channel (machine-evaluated by E-019; the ladder decision itself stays a
      documented G-009 decision — this field is a diagnostic echo).
    - ``"N"``: no positive above-null separability evidence on the combined
      channel (``mean <= 0`` or ``ci_low <= 0``) with NO saturation in any
      family. Saturated runs cannot support a non-separability statement, so
      they fall through to indeterminate.
    - ``"indeterminate"``: everything else (positive CI without the activation
      conditions, saturation present, or missing diagnostic data).
    """
    if combined_conditions_hold is True:
        return "S"
    if not isinstance(combined_excess, dict):
        return "indeterminate"
    mean = combined_excess.get("mean")
    ci_low = combined_excess.get("ci_low")
    unsaturated = combined_excess.get("allFamiliesUnsaturated")
    if mean is None or ci_low is None or unsaturated is not True:
        return "indeterminate"
    if mean <= 0 or ci_low <= 0:
        return "N"
    return "indeterminate"


def _as_dict(value: Any) -> Dict[str, Any]:
    """Defensive read: return ``value`` if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def build_separability_frontier_data(reports_dir: Any = None) -> Dict[str, Any]:
    """Join E2 utility with the E-019 channel-resolved excess separability.

    PURE (writes nothing). Joins the latest ``e2_policy_*.json`` and the latest
    ``leakage_diagnostic_*.json`` by policy name (``benchmark_index.json``
    pinning is honored when present, mirroring :func:`build_frontier_data`).
    Reads the diagnostic structure defensively: reports with or without the
    B-009 ``overlapCaveat`` block and with either ``leakageMeta.metric`` label
    (pre/post G-020) are accepted. Policies present in only one source are
    INCLUDED with explicit nulls and surfaced in ``joinWarnings`` — never
    silently dropped.
    """
    rdir = Path(reports_dir) if reports_dir is not None else _REPO_ROOT / "outputs" / "reports"
    e2 = _indexed_or_latest(rdir, "E2_POLICY_UTILITY", "e2_policy_")
    diag = _indexed_or_latest(rdir, "E_LEAKAGE_DIAGNOSTIC", "leakage_diagnostic_")

    if not e2:
        log_ko(_logger, "E2 보고서(e2_policy_*.json)를 찾을 수 없어 separability 프론티어의 utility 축이 비어 있습니다.")
    if not diag:
        log_ko(_logger, "확장 누출 진단 보고서(leakage_diagnostic_*.json)를 찾을 수 없어 separability 축이 비어 있습니다.")

    utility_by_policy: Dict[str, Dict[str, Any]] = {}
    for row in e2.get("table", []):
        utility_by_policy[row["policy"]] = {
            "utility": float(row.get("coordinate_coverage", 0.0)),
            "traceOnly": bool(row.get("traceOnly", True)),
        }

    cross_family = _as_dict(diag.get("crossFamilyExcess"))
    excess_per_policy = _as_dict(cross_family.get("perPolicy"))
    track_per_policy = _as_dict(_as_dict(diag.get("trackLConditions")).get("perPolicy"))
    families = list(cross_family.get("families") or [])
    leakage_meta = _as_dict(diag.get("leakageMeta"))

    channels = list(SEPARABILITY_CHANNELS)
    join_warnings: List[str] = []
    points: List[Dict[str, Any]] = []
    for policy in sorted(set(utility_by_policy) | set(excess_per_policy)):
        in_e2 = policy in utility_by_policy
        in_diag = policy in excess_per_policy
        if not in_e2:
            join_warnings.append(
                f"policy {policy} present only in the leakage diagnostic report; utility/traceOnly set to null"
            )
        if not in_diag:
            join_warnings.append(
                f"policy {policy} present only in the E2 report; excess/trackSConditionsHold set to null"
            )

        excess: Dict[str, Optional[Dict[str, Any]]] = {}
        policy_excess = _as_dict(excess_per_policy.get(policy))
        for channel in channels:
            block = policy_excess.get(channel)
            if isinstance(block, dict):
                excess[channel] = {key: block.get(key) for key in _EXCESS_CARRY_KEYS}
            else:
                excess[channel] = None

        conditions_hold: Dict[str, Optional[bool]] = {}
        policy_conditions = _as_dict(track_per_policy.get(policy))
        for channel in channels:
            block = policy_conditions.get(channel)
            if isinstance(block, dict) and "allConditionsHold" in block:
                conditions_hold[channel] = bool(block["allConditionsHold"])
            else:
                conditions_hold[channel] = None

        u = utility_by_policy.get(policy, {})
        points.append({
            "policy": policy,
            "utility": u.get("utility"),
            "traceOnly": u.get("traceOnly"),
            "excess": excess,
            "trackSConditionsHold": conditions_hold,
            "trackAssignment": _derive_track_assignment(
                excess["combined"], conditions_hold["combined"]
            ),
        })

    note = _SEPARABILITY_NOTE
    if not e2 or not diag:
        note += _MISSING_REPORT_NOTE

    data: Dict[str, Any] = {
        "schema": SEPARABILITY_FRONTIER_SCHEMA,
        "note": note,
        "trackAssignmentRule": _TRACK_ASSIGNMENT_RULE,
        "channels": channels,
        "points": points,
        "families": families,
        "nFamilies": len(families),
        "sourceReportHashes": {
            "e2": e2.get("reportHash"),
            "leakageDiagnostic": diag.get("reportHash"),
        },
        "leakageMetaEcho": {
            "isProxy": leakage_meta.get("isProxy"),
            "disclaimer": leakage_meta.get("disclaimer"),
            "primaryLabel": PRIMARY_METRIC_NAME,
            "sourceMetricLabel": leakage_meta.get("metric"),
        },
        "joinWarnings": join_warnings,
    }
    data["dataHash"] = canonical_hash(data)
    log_ko(
        _logger,
        f"separability 프론티어 데이터 생성: points={len(points)}, "
        f"joinWarnings={len(join_warnings)}, dataHash={data['dataHash'][:12]}",
    )
    return data


def write_separability_frontier_data(data: Dict[str, Any], reports_dir: Any = None) -> str:
    """Persist to ``outputs/reports/frontier_separability_data.json`` (E-010 pattern).

    ``build_separability_frontier_data`` stays pure; this writer owns the side
    effect. ``reports_dir`` may override the repo-rooted default (for tests).
    """
    rdir = Path(reports_dir) if reports_dir is not None else _REPO_ROOT / "outputs" / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "frontier_separability_data.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
    log_ko(
        _logger,
        f"separability 프론티어 데이터 작성: path={path}, dataHash={data.get('dataHash', '')[:12]}",
    )
    return str(path)


def render_separability_figures(data: Dict[str, Any], out_dir: Any) -> List[str]:
    """Render one scatter PNG per channel if matplotlib is importable, else [].

    x = excess mean (with CI error bars), y = utility, annotated with policy
    names. Points lacking either coordinate for a channel are skipped in that
    channel's figure (they remain in the hashed data). PNGs are a convenience;
    the hashed data JSON is the citable artifact.
    """
    try:
        import matplotlib  # noqa: WPS433
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log_ko(_logger, f"matplotlib 미설치로 separability PNG 렌더링을 건너뜁니다: {type(exc).__name__}")
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for channel in data["channels"]:
        plottable = [
            p for p in data["points"]
            if p.get("utility") is not None
            and isinstance(p.get("excess", {}).get(channel), dict)
            and p["excess"][channel].get("mean") is not None
        ]
        if not plottable:
            log_ko(_logger, f"채널 {channel}: 그릴 수 있는 포인트가 없어 PNG를 건너뜁니다.")
            continue
        xs = [p["excess"][channel]["mean"] for p in plottable]
        ys = [p["utility"] for p in plottable]
        xerr_low = [
            x - p["excess"][channel]["ci_low"]
            if p["excess"][channel].get("ci_low") is not None else 0.0
            for x, p in zip(xs, plottable)
        ]
        xerr_high = [
            p["excess"][channel]["ci_high"] - x
            if p["excess"][channel].get("ci_high") is not None else 0.0
            for x, p in zip(xs, plottable)
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(xs, ys, xerr=[xerr_low, xerr_high], fmt="o", capsize=3)
        for x, y, p in zip(xs, ys, plottable):
            ax.annotate(p["policy"], (x, y))
        ax.axvline(0.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"excess separability ({PRIMARY_METRIC_NAME}, {channel}, PROXY diagnostic)")
        ax.set_ylabel("utility (coordinate_coverage)")
        ax.set_title(f"ECHO-Bench utility vs excess separability — {channel} (controlled testbed)")
        path = out / f"frontier_separability_{channel}_{data['dataHash'][:12]}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
        log_ko(_logger, f"separability 프론티어 PNG 작성: path={path}")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: build + write a frontier data artifact (optionally render PNGs).

    ``python -m echo_bench.experiments.figures`` builds the E-010 utility×
    leakage frontier; ``--separability`` builds the E-022 channel-resolved
    utility×separability frontier instead. ``--render`` additionally writes
    convenience PNGs (skipped silently when matplotlib is absent).
    """
    import argparse

    parser = argparse.ArgumentParser(prog="echo_bench.experiments.figures")
    parser.add_argument("--separability", action="store_true",
                        help="build the E-022 utility x separability frontier data")
    parser.add_argument("--reports-dir", default=None,
                        help="reports directory (default: outputs/reports)")
    parser.add_argument("--render", action="store_true",
                        help="also render convenience PNGs (requires matplotlib)")
    parser.add_argument("--out-dir", default=None,
                        help="PNG output directory (default: outputs/artifacts)")
    args = parser.parse_args(argv)

    reports_dir = Path(args.reports_dir) if args.reports_dir else _REPO_ROOT / "outputs" / "reports"
    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "outputs" / "artifacts"

    if args.separability:
        data = build_separability_frontier_data(reports_dir)
        path = write_separability_frontier_data(data, reports_dir=reports_dir)
        if args.render:
            render_separability_figures(data, out_dir)
    else:
        data = build_frontier_data(reports_dir)
        path = write_frontier_data(data, reports_dir=reports_dir)
        if args.render:
            render_figures(data, out_dir)
    log_ko(_logger, f"프론티어 아티팩트 완료: path={path}, dataHash={data['dataHash'][:12]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
