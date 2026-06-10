"""Multi-seed-family runner for ECHO-Bench (Task E-014, TRD V-016 / GR-007).

All n=10 development results so far rest on a single seed family
(``base_seed=42``). This runner re-executes the core experiments (E1 horizon
sweep, E2 policy utility, E3 leakage/robustness/replay audit) across **multiple
base-seed families** and aggregates per-family metrics plus cross-family
stability statistics, so the development conclusions can be shown not to hinge
on one seed family.

Structure
---------
1. **Config-freeze check first** (C-011): the live ``TRACE_GREEDY`` policy's
   ``policy_version()`` must equal the hash recorded in
   ``configs/policies/frozen/trace_greedy_frozen.json``. On any mismatch the
   runner hard-fails with a Korean :class:`ValueError` before running anything.
2. **Per family**: call :func:`run_e1_horizon`, :func:`run_e2_policy`, and
   :func:`run_e3_audit` as library functions with that family's ``base_seed``
   (E2 at ``H=H_e2``; E1 sweeps its own config-defined horizon set). Each child
   writes its own fully hashed report; this runner collects each child's
   ``reportHash`` / ``seedBatchId`` / replay flag.
3. **Cross-family aggregation** (pure, RNG-free except the *seeded* bootstrap,
   which is a pure function of its inputs):

   - ``perFamily``: family -> policy -> E2 metric means (from each E2 table
     row's ``stats[<metric>]["mean"]``).
   - ``rankStabilityAcrossFamilies``: D-013 :func:`rank_stability_by_metric`
     where the resampling **unit is a seed FAMILY** (seeds within a family are
     already averaged into the family mean) — NOT a child seed as in E2's own
     ``rankStability`` block.
   - ``leakageAcrossFamilies``: per policy, the per-family ``leakage_proxy``
     and ``leakage_delta_vs_random`` values with mean + seeded-bootstrap CI via
     :func:`aggregate_values`. This delivers the confidence interval that
     D-011 marked structurally unavailable at a single seed family (one pooled
     NMI scalar per policy per family -> across families there ARE multiple
     values to resample): ``ciAvailable=True``, ``unit="seed_family"``.
   - ``e1LongHorizonAcrossFamilies``: per family, TRACE_GREEDY's max-H (H=20 in
     the production horizon config) ``coordinate_coverage`` and
     ``redundancy_rate``, with per-family booleans for the TRD V-001 thresholds
     (coverage >= 0.90, redundancy <= 0.05) and an overall ``allFamiliesPass``.

4. **Replay audit** (``mode="child_reports"``): an inline whole-sweep replay
   would re-run every child experiment a second time (and each E1/E2 child
   already self-validates inline per E-012). Instead the report documents that
   the children carry their own replay audits and that the cross-family
   aggregation layer is deterministic (pure functions of the child reports), so
   the sweep replays whenever its children replay. The summary block carries
   every child ``replayable`` flag plus an overall boolean.

``--dry-run`` validates the freeze + configs (delegating to each child's own
dry-run), computes the available hashes, and writes nothing.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
any pool, trace, metric, or report. All statements are system-level statistics
over a controlled testbed: cross-family stability of system-level metrics, not
user preference, emotion, wellbeing, privacy, legal compliance, or real-world
generalization. The leakage values aggregated here remain a PROXY exactly as in
the E3 children.

All identifiers, metric names, config keys and file paths stay English; runtime
log messages are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from echo_bench.experiments.e1_horizon import run_e1_horizon
from echo_bench.experiments.e2_policy import run_e2_policy
from echo_bench.experiments.e3_audit import run_e3_audit
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.aggregate import aggregate_values, rank_stability_by_metric
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "run_seed_families",
    "main",
    "verify_trace_greedy_freeze",
    "extract_e2_policy_metric_means",
    "build_per_family_block",
    "build_rank_stability_across_families",
    "build_leakage_across_families",
    "build_e1_long_horizon_block",
    "build_replay_audit_summary",
    "DEFAULT_BASE_SEEDS",
    "SEED_FAMILY_EXPERIMENTS",
    "V001_COVERAGE_MIN",
    "V001_REDUNDANCY_MAX",
    "LONG_HORIZON_POLICY",
]

_logger = get_logger(__name__)

SEED_FAMILIES_SCHEMA = "echo_bench.e_seed_families.report"
SEED_FAMILIES_SCHEMA_VERSION = "1"

# The default base-seed families (>= 5 per TRD V-016). 42 is the historical
# development family; the others are fresh, arbitrary, documented constants.
DEFAULT_BASE_SEEDS: Tuple[int, ...] = (42, 7, 101, 2025, 31337)

# The core experiments a family runs, in execution order.
SEED_FAMILY_EXPERIMENTS: Tuple[str, ...] = ("e1", "e2", "e3")

# TRD V-001 thresholds applied to TRACE_GREEDY's max-H E1 cell per family.
V001_COVERAGE_MIN = 0.90
V001_REDUNDANCY_MAX = 0.05

# The policy audited by the long-horizon block (and by the C-011 freeze).
LONG_HORIZON_POLICY = "TRACE_GREEDY"

# The leakage metrics aggregated across families (per E3 leakage-table row).
_LEAKAGE_KEYS = ("leakage_proxy", "leakage_delta_vs_random")

# Repo-rooted locations (resolved relative to the package, not the cwd).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_CFG_DIR = _REPO_ROOT / "configs" / "policies"
_FROZEN_MANIFEST_PATH = (
    _REPO_ROOT / "configs" / "policies" / "frozen" / "trace_greedy_frozen.json"
)
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

# The E2 run-level k the C-011 manifest was frozen at is read from the frozen
# manifest itself (effectiveConfig.k), so the gate can never drift from the
# artifact it validates.  Set lazily in verify_trace_greedy_freeze() from
# the loaded manifest; _FREEZE_K is the module-level placeholder used only
# when the manifest hasn't been loaded yet (tests that monkeypatch
# _load_frozen_manifest supply their own k).
_FREEZE_K: int | None = None  # resolved from manifest at runtime


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into a dict (empty dict if the doc is empty)."""
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return doc if isinstance(doc, dict) else {}


def _git_commit_hash() -> str:
    """Return the short git commit hash, or ``"uncommitted"`` if unavailable."""
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


def _load_frozen_manifest() -> Dict[str, Any]:
    """Load the C-011 frozen TRACE_GREEDY manifest from disk.

    Module-level seam so tests can monkeypatch a doctored manifest WITHOUT ever
    touching the real frozen file.
    """
    with open(_FROZEN_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_trace_greedy_freeze() -> Dict[str, Any]:
    """C-011 drift gate: live TRACE_GREEDY hash must equal the frozen hash.

    Mirrors the construction path the freeze manifest (and
    ``tests/test_config_freeze.py``) documents: load
    ``configs/policies/trace_greedy.yaml``, override ``k`` with the value
    recorded in the manifest's ``effectiveConfig.k`` (so the gate is always
    consistent with the artifact it validates), instantiate
    :class:`TraceGreedyPolicy`, and compare its ``policy_version()`` to the
    manifest's ``policyEffectiveConfigHash``.

    Returns a small provenance dict on success; raises a Korean
    :class:`ValueError` (hard fail, nothing runs) on any mismatch or on a
    malformed manifest.
    """
    manifest = _load_frozen_manifest()
    frozen_hash = manifest.get("policyEffectiveConfigHash")
    if not isinstance(frozen_hash, str) or not frozen_hash:
        raise ValueError(
            "E-014 동결 검증 실패: 동결 매니페스트에 policyEffectiveConfigHash 가 "
            f"없습니다 (path={_FROZEN_MANIFEST_PATH})."
        )

    # Read k from the manifest so this gate cannot drift from the artifact it
    # validates.  The manifest must carry effectiveConfig.k (set at freeze time
    # by C-011); a missing key is a hard error rather than a silent fallback.
    effective_cfg = manifest.get("effectiveConfig")
    if not isinstance(effective_cfg, dict) or "k" not in effective_cfg:
        raise ValueError(
            "E-014 동결 검증 실패: 동결 매니페스트에 effectiveConfig.k 가 없습니다 "
            f"(path={_FROZEN_MANIFEST_PATH}). C-011 동결 매니페스트를 확인하세요."
        )
    freeze_k = int(effective_cfg["k"])

    cfg = dict(_load_yaml(_POLICY_CFG_DIR / "trace_greedy.yaml"))
    cfg["k"] = freeze_k
    live_hash = TraceGreedyPolicy(cfg).policy_version()

    if live_hash != frozen_hash:
        raise ValueError(
            "E-014 동결 검증 실패: TRACE_GREEDY 의 현재 policy_version 이 "
            "C-011 동결 해시와 다릅니다. 가중치/클래스명/yaml 설정이 변경된 "
            "상태에서는 멀티 시드 패밀리 실행을 시작할 수 없습니다 "
            f"(frozen={frozen_hash}, live={live_hash}). 의도된 변경이라면 "
            "동결 매니페스트를 명시적 결정으로 갱신한 뒤 다시 실행하세요."
        )

    log_ko(
        _logger,
        "E-014 동결 검증 통과: TRACE_GREEDY policy_version 이 C-011 동결 해시와 "
        f"일치합니다 (hash={frozen_hash[:12]}, k={freeze_k}, "
        f"frozenAt={manifest.get('frozenAt')}).",
    )
    return {
        "policyName": manifest.get("policyName"),
        "frozenHash": frozen_hash,
        "liveHash": live_hash,
        "freezeK": freeze_k,
        "frozenAt": manifest.get("frozenAt"),
        "taskId": manifest.get("taskId"),
        "verified": True,
    }


# ---------------------------------------------------------------------------
# Pure cross-family aggregation helpers (unit-testable on synthetic reports).
# Each takes child report dicts keyed by family (str(base_seed)) in family
# order and returns a JSON-ready block. No RNG enters any of them except the
# SEEDED bootstrap inside aggregate_values (a pure function of its inputs), so
# the whole aggregation layer replays bit-identically.
# ---------------------------------------------------------------------------


def extract_e2_policy_metric_means(
    e2_report: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Extract policy -> metric -> mean from one E2 report's table.

    Reads each row's ``stats[<metric>]["mean"]`` for every key in the report's
    ``metricKeys`` (the authoritative aggregate; the flat per-row means are a
    back-compat mirror of the same values).

    Raises a Korean :class:`ValueError` if the report is structurally malformed
    (missing top-level keys, missing per-row stats entry, or missing mean).
    """
    try:
        metric_keys = list(e2_report["metricKeys"])
    except KeyError:
        raise ValueError(
            "E-014 집계: E2 보고서에 metricKeys 키가 없습니다. "
            "보고서 구조를 확인하세요."
        )
    try:
        rows = list(e2_report["table"])
    except KeyError:
        raise ValueError(
            "E-014 집계: E2 보고서에 table 키가 없습니다. "
            "보고서 구조를 확인하세요."
        )
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        policy = row.get("policy")
        if policy is None:
            raise ValueError(
                "E-014 집계: E2 table 행에 policy 키가 없습니다."
            )
        stats = row.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(
                f"E-014 집계: E2 table 행(policy={policy})에 stats 딕셔너리가 없습니다."
            )
        means: Dict[str, float] = {}
        for key in metric_keys:
            stat_entry = stats.get(key)
            if not isinstance(stat_entry, dict) or "mean" not in stat_entry:
                raise ValueError(
                    f"E-014 집계: E2 table 행(policy={policy})에 "
                    f"stats[{key!r}]['mean'] 이 없습니다."
                )
            means[key] = float(stat_entry["mean"])
        out[policy] = means
    return out


def build_per_family_block(
    e2_by_family: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """``perFamily``: family -> policy -> E2 metric means."""
    return {
        family: extract_e2_policy_metric_means(report)
        for family, report in e2_by_family.items()
    }


def build_rank_stability_across_families(
    e2_by_family: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """D-013 rank stability where the resampling UNIT is a seed FAMILY.

    For each E2 metric, ``values_by_policy`` is the list of per-family mean
    values (one aligned value per family, in family order). Seeds within a
    family are already averaged into that family's mean, so a "unit" here is a
    whole base-seed family — distinct from E2's own ``rankStability`` block
    whose units are child seeds inside one family.

    Raises a Korean :class:`ValueError` if the families disagree on the metric
    or policy sets (misaligned units must never be ranked silently).
    """
    if not e2_by_family:
        raise ValueError(
            "E-014 집계: e2_by_family 가 비어 있습니다 (패밀리가 최소 1개 필요)."
        )

    families = list(e2_by_family)
    means_by_family: Dict[str, Dict[str, Dict[str, float]]] = {}
    for family, report in e2_by_family.items():
        # extract_e2_policy_metric_means already raises a descriptive Korean
        # ValueError on malformed input, so no bare KeyError escapes here.
        means_by_family[family] = extract_e2_policy_metric_means(report)

    first = families[0]
    try:
        metric_keys = list(e2_by_family[first]["metricKeys"])
    except KeyError:
        raise ValueError(
            "E-014 집계: 첫 번째 패밀리 E2 보고서에 metricKeys 키가 없습니다 "
            f"(family={first})."
        )
    policies = sorted(means_by_family[first])
    for family in families[1:]:
        if list(e2_by_family[family]["metricKeys"]) != metric_keys:
            raise ValueError(
                "E-014 집계: 패밀리 간 E2 metricKeys 가 일치하지 않습니다 "
                f"(family={family})."
            )
        if sorted(means_by_family[family]) != policies:
            raise ValueError(
                "E-014 집계: 패밀리 간 E2 정책 집합이 일치하지 않습니다 "
                f"(family={family})."
            )

    per_metric_values = {
        key: {
            policy: [means_by_family[family][policy][key] for family in families]
            for policy in policies
        }
        for key in metric_keys
    }
    by_metric = rank_stability_by_metric(per_metric_values)
    return {
        "unit": "seed_family",
        "note": (
            "Each resampling unit is one base-seed FAMILY; the n child seeds "
            "within a family are already averaged into the family mean before "
            "ranking. Distinct from the per-child-seed rankStability block "
            "inside each E2 child report (D-013, same ranking function)."
        ),
        "families": families,
        "policies": policies,
        "byMetric": by_metric,
    }


def build_leakage_across_families(
    e3_by_family: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Cross-family leakage-proxy aggregation with a seeded-bootstrap CI.

    Per policy, collects the per-family ``leakage_proxy`` and
    ``leakage_delta_vs_random`` scalars from each E3 child's leakage table and
    aggregates them via :func:`aggregate_values` (mean / std / seeded-bootstrap
    CI). D-011 documented that NO CI is structurally available at a single seed
    family (one pooled-NMI scalar per policy); across families there are
    multiple values to resample, so this block carries ``ciAvailable=True``
    with ``unit="seed_family"``. The values remain a PROXY exactly as in E3.
    """
    if not e3_by_family:
        raise ValueError(
            "E-014 집계: e3_by_family 가 비어 있습니다 (패밀리가 최소 1개 필요)."
        )

    families = list(e3_by_family)
    first_section = e3_by_family[families[0]]["leakage"]
    policies = sorted(
        row["policy"] for row in first_section["table"]
    )

    rows_by_family: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for family in families:
        section = e3_by_family[family]["leakage"]
        rows = {row["policy"]: row for row in section["table"]}
        if sorted(rows) != policies:
            raise ValueError(
                "E-014 집계: 패밀리 간 E3 누출 정책 집합이 일치하지 않습니다 "
                f"(family={family})."
            )
        rows_by_family[family] = rows

    per_policy: Dict[str, Any] = {}
    for policy in policies:
        block: Dict[str, Any] = {}
        for key in _LEAKAGE_KEYS:
            values = [
                float(rows_by_family[family][policy][key]) for family in families
            ]
            agg = aggregate_values(values, f"{key}@{policy}")
            block[key] = {"perFamilyValues": values, **agg}
        per_policy[policy] = block

    return {
        "unit": "seed_family",
        "ciAvailable": True,
        "ciNote": (
            "D-011 documented that no CI is structurally available within a "
            "single seed family (leakage_proxy is one pooled-NMI scalar per "
            "policy per family). Across seed families there are multiple "
            "aligned values per policy, so a seeded-bootstrap CI over "
            "families IS available here without redefining the metric."
        ),
        "isProxy": bool(first_section.get("isProxy", True)),
        "disclaimer": first_section.get("disclaimer"),
        "deltaReference": first_section.get("deltaReference"),
        "families": families,
        "policies": policies,
        "perPolicy": per_policy,
    }


def _max_h_row(
    e1_report: Mapping[str, Any], policy: str
) -> Tuple[int, Mapping[str, Any]]:
    """Return ``(max_H, row)`` for ``policy`` at the report's largest horizon."""
    max_h = max(int(h) for h in e1_report["config"]["H_sweep"])
    for row in e1_report["table"]:
        if row["policy"] == policy and int(row["H"]) == max_h:
            return max_h, row
    raise ValueError(
        "E-014 집계: E1 보고서에서 최대 horizon 행을 찾을 수 없습니다 "
        f"(policy={policy}, H={max_h})."
    )


def build_e1_long_horizon_block(
    e1_by_family: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """TRD V-001 long-horizon check per family at TRACE_GREEDY's max-H cell.

    Per family: TRACE_GREEDY's max-H (production: H=20) ``coordinate_coverage``
    and ``redundancy_rate`` means, plus booleans against the V-001 thresholds
    (coverage >= 0.90, redundancy <= 0.05) and an overall ``allFamiliesPass``.
    """
    if not e1_by_family:
        raise ValueError(
            "E-014 집계: e1_by_family 가 비어 있습니다 (패밀리가 최소 1개 필요)."
        )

    per_family: Dict[str, Any] = {}
    for family, report in e1_by_family.items():
        max_h, row = _max_h_row(report, LONG_HORIZON_POLICY)
        coverage = float(row["coordinate_coverage"])
        redundancy = float(row["redundancy_rate"])
        coverage_pass = coverage >= V001_COVERAGE_MIN
        redundancy_pass = redundancy <= V001_REDUNDANCY_MAX
        per_family[family] = {
            "H": max_h,
            "coordinate_coverage": coverage,
            "redundancy_rate": redundancy,
            "coveragePass": coverage_pass,
            "redundancyPass": redundancy_pass,
            "familyPass": coverage_pass and redundancy_pass,
        }

    return {
        "policy": LONG_HORIZON_POLICY,
        "thresholds": {
            "coordinate_coverage_min": V001_COVERAGE_MIN,
            "redundancy_rate_max": V001_REDUNDANCY_MAX,
            "source": "TRD V-001",
        },
        "perFamily": per_family,
        "allFamiliesPass": all(
            entry["familyPass"] for entry in per_family.values()
        ),
    }


def build_replay_audit_summary(
    children_by_family: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Collect each child report's ``replayable`` flag + an overall boolean.

    ``children_by_family`` maps family -> experiment id -> child report. E1/E2
    carry ``replayAudit.replayable`` only when their inline replay ran
    (``replay_validate=True``); a missing audit is reported as ``None`` (never
    silently coerced to ``True``). E3 always carries its smoke-core replay
    audit. ``allReplayable`` is ``True`` only when EVERY collected flag is
    exactly ``True``.
    """
    per_family: Dict[str, Dict[str, Any]] = {}
    flags: List[Any] = []
    for family, children in children_by_family.items():
        per_family[family] = {}
        for exp_id, report in children.items():
            audit = report.get("replayAudit")
            flag = audit.get("replayable") if isinstance(audit, dict) else None
            per_family[family][exp_id] = flag
            flags.append(flag)
    return {
        "perFamily": per_family,
        "allReplayable": bool(flags) and all(flag is True for flag in flags),
    }


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


def _validate_args(
    base_seeds: Sequence[int], experiments: Sequence[str]
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Validate / normalize base_seeds + experiments, failing closed (Korean)."""
    seeds = tuple(int(s) for s in base_seeds)
    if not seeds:
        raise ValueError(
            "E-014 실행: base_seeds 가 비어 있습니다 (패밀리가 최소 1개 필요)."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"E-014 실행: base_seeds 에 중복된 시드가 있습니다: {list(seeds)}."
        )
    exps = tuple(str(e) for e in experiments)
    unknown = [e for e in exps if e not in SEED_FAMILY_EXPERIMENTS]
    if unknown:
        raise ValueError(
            f"E-014 실행: 알 수 없는 experiment 식별자입니다: {unknown} "
            f"(허용: {list(SEED_FAMILY_EXPERIMENTS)})."
        )
    if not exps:
        raise ValueError(
            "E-014 실행: experiments 가 비어 있습니다 (최소 1개 필요)."
        )
    # Canonical execution order regardless of input order.
    ordered = tuple(e for e in SEED_FAMILY_EXPERIMENTS if e in exps)
    return seeds, ordered


def run_seed_families(
    base_seeds: Sequence[int] = DEFAULT_BASE_SEEDS,
    n: int = 10,
    k: int = 4,
    pool_size: int = 64,
    H_e2: int = 8,
    dry_run: bool = False,
    replay_validate: bool = True,
    experiments: Sequence[str] = SEED_FAMILY_EXPERIMENTS,
) -> dict:
    """Run E1/E2/E3 across multiple base-seed families and aggregate.

    Args:
        base_seeds: the base-seed families (>= 5 for TRD V-016 production use;
            smaller tuples are allowed for development/tests).
        n: child seeds per family (passed through to E1/E2).
        k: slate size (passed through to all children).
        pool_size: candidate pool size (passed through to all children).
        H_e2: horizon for the E2 and E3 children (E1 sweeps its own
            config-defined horizon set and takes no H argument).
        dry_run: when true, run the freeze check + every child's own dry-run
            (config validation + plan hashes), write no files.
        replay_validate: forwarded to the E1/E2 children's inline replay audit
            (E-012). E3 always runs its smoke-core replay audit.
        experiments: which children to run per family (subset of
            ``("e1", "e2", "e3")``); aggregation blocks that depend on a
            skipped child experiment are emitted as
            ``{"skipped": True, "reason": "<description>"}`` rather than
            ``null``, so downstream readers can distinguish "not computed"
            from a computed null result.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    seeds, exps = _validate_args(base_seeds, experiments)

    # 0. C-011 config-freeze gate. Hard-fails (Korean) before anything runs.
    freeze = verify_trace_greedy_freeze()

    run_params = {
        "base_seeds": [int(s) for s in seeds],
        "n": int(n),
        "k": int(k),
        "pool_size": int(pool_size),
        "H_e2": int(H_e2),
        "experiments": list(exps),
        "designVersion": "e014-design-1",
        "configFreeze": {
            "policyName": freeze["policyName"],
            "policyEffectiveConfigHash": freeze["frozenHash"],
            "taskId": freeze["taskId"],
        },
    }
    config_hash = canonical_hash({"run_params": run_params})

    # Per-experiment child kwargs (E1 has no H argument: it sweeps the
    # config-defined horizon set; E3 has no n / replay_validate arguments: it
    # is single-seed per section and always replay-audits its smoke core).
    def _child_kwargs(exp_id: str, seed: int, child_dry: bool) -> Dict[str, Any]:
        if exp_id == "e1":
            return dict(
                base_seed=seed,
                n=int(n),
                k=int(k),
                pool_size=int(pool_size),
                dry_run=child_dry,
                replay_validate=bool(replay_validate),
            )
        if exp_id == "e2":
            return dict(
                base_seed=seed,
                H=int(H_e2),
                k=int(k),
                pool_size=int(pool_size),
                n=int(n),
                dry_run=child_dry,
                replay_validate=bool(replay_validate),
            )
        return dict(
            base_seed=seed,
            H=int(H_e2),
            k=int(k),
            pool_size=int(pool_size),
            dry_run=child_dry,
        )

    _RUNNERS = {"e1": run_e1_horizon, "e2": run_e2_policy, "e3": run_e3_audit}

    # 1. Dry run: freeze check done; validate every child config via the
    #    child's own dry-run (which also writes nothing), collect plan hashes.
    if dry_run:
        families_plan: Dict[str, Any] = {}
        for seed in seeds:
            family = str(seed)
            log_ko(
                _logger,
                f"E-014 드라이런: 패밀리 base_seed={seed} 자식 설정 검증 시작 "
                f"(experiments={list(exps)}).",
            )
            families_plan[family] = {}
            for exp_id in exps:
                child = _RUNNERS[exp_id](**_child_kwargs(exp_id, seed, True))
                families_plan[family][exp_id] = {
                    "configHash": child["configHash"],
                    "archiveHash": child["archiveHash"],
                    "poolHash": child["poolHash"],
                }
        log_ko(
            _logger,
            "E-014 드라이런 요약: "
            f"families={[int(s) for s in seeds]}, experiments={list(exps)}, "
            f"n={n}, k={k}, pool_size={pool_size}, H_e2={H_e2}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "configHash": config_hash,
            "configFreeze": freeze,
            "families": families_plan,
        }

    # 2. Run every family. Families are the slow unit: log start/end of each.
    children_by_family: Dict[str, Dict[str, dict]] = {}
    for idx, seed in enumerate(seeds, start=1):
        family = str(seed)
        log_ko(
            _logger,
            f"E-014 패밀리 시작 ({idx}/{len(seeds)}): base_seed={seed}, "
            f"experiments={list(exps)}, n={n}, k={k}, pool_size={pool_size}, "
            f"H_e2={H_e2}.",
        )
        children_by_family[family] = {}
        for exp_id in exps:
            child = _RUNNERS[exp_id](**_child_kwargs(exp_id, seed, False))
            children_by_family[family][exp_id] = child
            log_ko(
                _logger,
                f"E-014 자식 완료: family={seed}, experiment={exp_id}, "
                f"reportHash={child['reportHash'][:12]}, "
                f"seedBatchId={child['seedBatchId'][:12]}.",
            )
        log_ko(
            _logger,
            f"E-014 패밀리 완료 ({idx}/{len(seeds)}): base_seed={seed}.",
        )

    # 3. Child provenance: per family/experiment reportHash + seedBatchId.
    child_report_hashes = {
        family: {exp_id: child["reportHash"] for exp_id, child in children.items()}
        for family, children in children_by_family.items()
    }
    child_seed_batch_ids = {
        family: {exp_id: child["seedBatchId"] for exp_id, child in children.items()}
        for family, children in children_by_family.items()
    }

    # 4. Cross-family aggregation (pure functions of the child reports).
    e1_by_family = {
        family: children["e1"]
        for family, children in children_by_family.items()
        if "e1" in children
    }
    e2_by_family = {
        family: children["e2"]
        for family, children in children_by_family.items()
        if "e2" in children
    }
    e3_by_family = {
        family: children["e3"]
        for family, children in children_by_family.items()
        if "e3" in children
    }

    # When a child experiment was not in `experiments`, the blocks derived from
    # it are emitted as an explicit skip marker (not null) so downstream readers
    # can distinguish "not computed" from "computed and was None".
    _SKIPPED = "experiment not in this run's experiments selection"

    def _skip_block() -> Dict[str, Any]:
        return {"skipped": True, "reason": _SKIPPED}

    per_family_block: Any = (
        build_per_family_block(e2_by_family) if e2_by_family else _skip_block()
    )
    rank_block: Any = (
        build_rank_stability_across_families(e2_by_family)
        if e2_by_family
        else _skip_block()
    )
    leakage_block: Any = (
        build_leakage_across_families(e3_by_family)
        if e3_by_family
        else _skip_block()
    )
    e1_long_block: Any = (
        build_e1_long_horizon_block(e1_by_family)
        if e1_by_family
        else _skip_block()
    )
    replay_summary = build_replay_audit_summary(children_by_family)

    def _block_status(block: Any) -> str:
        return "생략(스킵)" if isinstance(block, dict) and block.get("skipped") else "있음"

    log_ko(
        _logger,
        "E-014 교차 패밀리 집계 완료: "
        f"families={len(seeds)}, "
        f"rankStability={_block_status(rank_block)}, "
        f"leakageCI={_block_status(leakage_block)}, "
        f"e1LongHorizon={_block_status(e1_long_block)}, "
        f"allReplayable={replay_summary['allReplayable']}.",
    )

    # 5. Sweep-level identifiers + aggregate hash chain. Each family builds its
    #    own archive/pool from its base seed, so the sweep-level hashes are
    #    canonical hashes OVER the per-family child hashes.
    sweep_seed_batch_id = canonical_hash(
        {
            "base_seeds": run_params["base_seeds"],
            "n": int(n),
            "k": int(k),
            "pool_size": int(pool_size),
            "H_e2": int(H_e2),
            "experiments": list(exps),
            "childSeedBatchIds": child_seed_batch_ids,
        }
    )

    def _child_hash_map(key: str) -> Dict[str, Dict[str, str]]:
        return {
            family: {exp_id: child[key] for exp_id, child in children.items()}
            for family, children in children_by_family.items()
        }

    archive_hash = canonical_hash(_child_hash_map("archiveHash"))
    pool_hash = canonical_hash(_child_hash_map("poolHash"))
    slate_hash = canonical_hash(_child_hash_map("slateHash"))
    trace_hash = canonical_hash(_child_hash_map("traceHash"))

    # 6. Results body + output hash. The replay summary is part of the hashed
    #    body (E3 precedent): it is a deterministic function of the children
    #    for fixed run args.
    results_body = {
        "config": run_params,
        "childReportHashes": child_report_hashes,
        "childSeedBatchIds": child_seed_batch_ids,
        "perFamily": per_family_block,
        "rankStabilityAcrossFamilies": rank_block,
        "leakageAcrossFamilies": leakage_block,
        "e1LongHorizonAcrossFamilies": e1_long_block,
        "replayAuditSummary": replay_summary,
    }
    output_hash = canonical_hash(results_body)

    # 7. Report (reportHash is computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": SEED_FAMILIES_SCHEMA,
        "schemaVersion": SEED_FAMILIES_SCHEMA_VERSION,
        "experiment": "E_SEED_FAMILIES",
        "phaseNote": (
            "E-014 (TRD V-016 / GR-007): re-executes the core experiments "
            "(E1, E2, E3) across multiple base-seed families and aggregates "
            "per-family means plus cross-family stability (rank stability "
            "with unit=seed_family; leakage-proxy mean + seeded-bootstrap CI "
            "over families; TRACE_GREEDY long-horizon V-001 thresholds per "
            "family). Shows whether development conclusions hinge on the "
            "single base_seed=42 family. All values are system-level "
            "statistics over a controlled testbed; the leakage values remain "
            "a PROXY (no privacy/anonymity/legal claim); no real-world "
            "generalization claim."
        ),
        "config": run_params,
        "configFreeze": freeze,
        "seedBatchId": sweep_seed_batch_id,
        "childReportHashes": child_report_hashes,
        "childSeedBatchIds": child_seed_batch_ids,
        "perFamily": per_family_block,
        "rankStabilityAcrossFamilies": rank_block,
        "leakageAcrossFamilies": leakage_block,
        "e1LongHorizonAcrossFamilies": e1_long_block,
        "replayAuditSummary": replay_summary,
        # Replay-audit mode for the sweep itself: the children carry their own
        # replay audits (E1/E2 inline per E-012; E3 smoke-core), and the
        # cross-family aggregation layer is deterministic (pure functions of
        # the child reports; the only randomness is the SEEDED bootstrap,
        # a pure function of its inputs), so no whole-sweep inline re-run is
        # performed -- it would double the dominant child cost without adding
        # information beyond the children's own audits.
        "replayAudit": {
            "mode": "child_reports",
            "note": (
                "Child reports carry their own replay audits; the "
                "cross-family aggregation is deterministic (RNG-free except "
                "the seeded bootstrap, a pure function of its inputs), so "
                "the sweep replays exactly when its children replay."
            ),
            "inlineValidate": bool(replay_validate),
            "perFamily": replay_summary["perFamily"],
            "replayable": replay_summary["allReplayable"],
        },
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
        # Describes what the sweep-level hashes cover so readers need not infer.
        "hashSemantics": (
            "Sweep-level archiveHash / poolHash / slateHash / traceHash are "
            "canonical hashes computed over the mapping of per-family child "
            "hashes (family -> experiment -> child hash).  They are NOT a "
            "re-hash of raw artifacts; their integrity depends on the child "
            "reports' own hash chains."
        ),
    }
    report_hash = canonical_hash(report)
    report["reportHash"] = report_hash

    # 8. Reproducibility pack (sweep-level hashes aggregate the children's).
    pack = ReproducibilityPack(
        configHash=config_hash,
        commitHash=_git_commit_hash(),
        archiveHash=archive_hash,
        poolHash=pool_hash,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=report_hash,
        seedBatchId=sweep_seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 9. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"seed_families_{sweep_seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "E-014 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={sweep_seed_batch_id[:12]}, families={len(seeds)}, "
        f"allReplayable={replay_summary['allReplayable']}.",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run the sweep, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.e_seed_families",
        description="ECHO-Bench E-014 multi-seed-family runner.",
    )
    parser.add_argument(
        "--base-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_BASE_SEEDS),
        help="comma-separated base-seed families (default: 42,7,101,2025,31337)",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="child seeds per family"
    )
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=64, help="candidate pool size"
    )
    parser.add_argument(
        "--H-e2", type=int, default=8, help="horizon for the E2/E3 children"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(SEED_FAMILY_EXPERIMENTS),
        help="comma-separated experiment subset (default: e1,e2,e3)",
    )
    parser.add_argument(
        "--no-replay-validate",
        action="store_true",
        help="skip the E1/E2 children's inline replay audit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    base_seeds = tuple(
        int(token) for token in args.base_seeds.split(",") if token.strip()
    )
    experiments = tuple(
        token.strip() for token in args.experiments.split(",") if token.strip()
    )

    result = run_seed_families(
        base_seeds=base_seeds,
        n=args.n,
        k=args.k,
        pool_size=args.pool_size,
        H_e2=args.H_e2,
        dry_run=args.dry_run,
        replay_validate=not args.no_replay_validate,
        experiments=experiments,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "E-014 드라이런 완료: "
            f"families={result['config']['base_seeds']}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "E-014 실행 완료: "
            f"families={result['config']['base_seeds']}, "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"allReplayable={result['replayAuditSummary']['allReplayable']}.",
        )


if __name__ == "__main__":
    main()
