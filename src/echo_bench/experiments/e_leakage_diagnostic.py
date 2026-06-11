"""Expanded leakage diagnostic runner for ECHO-Bench (Task E-019, TRD E-019).

The E3 leakage section measures probe-separability over the FROZEN B-004
probe trio (``DEFAULT_PROBE_SET``) at one base seed. This runner is the
Track L hardening consumer named "E-019" by the B-007 / D-015 / D-016 / D-017
task notes: it re-measures probe-separability with the EXPANDED 7-probe
registry (B-007 / TRD B-008) across multiple base-seed families and reports,
per policy and per channel:

- the full D-015 null-corrected block (``observed_nmi`` / ``null_mean`` /
  ``null_std`` / ``excess_nmi`` / ``excess_z``),
- the D-016 channel separation (``slate`` / ``selection`` / ``combined``),
- the D-017 saturation diagnostics (``saturation_flag`` per channel),
- the B-007 probe-overlap audit over decision-time contexts,
- a cross-family seeded-bootstrap CI of ``excess_nmi`` — the confidence
  interval D-015 marked as "future E-019" (unit = seed FAMILY, the E-014
  convention),
- a machine evaluation of the four Track L re-enable conditions
  (docs/12_CLAIM_LADDER.md Section 5) per policy x channel.

GUARDRAILS
----------
The Track L conditions block is a **DIAGNOSTIC**: it states whether the
ladder's re-enable conditions hold in this run's data. Unlocking the
conditional rung is a documented ladder decision (G-009 / TRD G-013) — it is
NEVER an automatic effect of this report, and this report never phrases a
comparative leakage claim. Every separability value remains a PROXY
(``isProxy=True`` + disclaimer): a system-level statistic over a controlled
testbed, NOT a privacy guarantee, anonymity proof, identifiability bound, or
legal/compliance claim. Strategy probes are controlled instrumented input
policies, never synthetic users. E2/E3 stay on the frozen
``DEFAULT_PROBE_SET`` — this runner imports their policy set but never
modifies their behaviour.

TERMINOLOGY (G-020)
-------------------
The primary report label for the measured statistic is
``probe_separability_proxy`` (the probe-separability terminology this module
already uses in prose); ``leakage_proxy`` is the legacy machine name, carried
as ``legacyAlias`` in ``leakageMeta``. Machine keys and values — per-family
row keys such as ``leakage_proxy``, ``excess_nmi`` and friends — stay
byte-identical (label layer only, D-012 precedent). Module/file names keep
the legacy "leakage diagnostic" wording; renaming files is out of scope.

All identifiers, metric names, config keys and file paths stay English;
runtime log messages are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.horizon import default_h, load_horizon, validate_h
from echo_bench.env.round_runner import run_episode
from echo_bench.env.trace_state import TraceState
from echo_bench.experiments.e3_audit import (
    E3_LEAKAGE_DELTA_REFERENCE,
    E3_LEAKAGE_POLICIES,
)
from echo_bench.experiments.e_seed_families import (
    DEFAULT_BASE_SEEDS,
    verify_trace_greedy_freeze,
)
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.aggregate import aggregate_values
from echo_bench.metrics.leakage import (
    DEFAULT_NULL_PERMUTATIONS,
    IS_PROXY,
    PROXY_DISCLAIMER,
    leakage_delta_vs_random,
    leakage_proxy_with_metadata,
)
from echo_bench.metrics.separability import (
    CHANNEL_NAMES,
    SATURATION_UNIQUE_RATE_THRESHOLD,
    channel_separated_separability,
    separability_row_fields,
)
from echo_bench.probes.probe_overlap import (
    PROBE_OVERLAP_EXCLUDE_THRESHOLD,
    PROBE_OVERLAP_THRESHOLD,
    probe_overlap_audit,
)
from echo_bench.probes.strategy_probes import PROBES, get_probe
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "run_leakage_diagnostic",
    "main",
    "EXPANDED_PROBE_SET",
    "TRACK_L_MIN_FAMILIES",
    "build_cross_family_excess_block",
    "build_overlap_caveat",
    "evaluate_track_l_conditions",
]

_logger = get_logger(__name__)

DIAGNOSTIC_SCHEMA = "echo_bench.e_leakage_diagnostic.report"
DIAGNOSTIC_SCHEMA_VERSION = "1"

#: The EXPANDED probe set: the full B-007 registry, name-sorted. A strict
#: superset of the frozen ``DEFAULT_PROBE_SET`` E2/E3 keep running on.
EXPANDED_PROBE_SET: Tuple[str, ...] = tuple(sorted(PROBES))

#: Track L re-enable condition 4 requires >= this many seed families
#: (docs/12_CLAIM_LADDER.md Section 5; E-014 machinery convention).
TRACK_L_MIN_FAMILIES = 5

#: Where the four conditions are defined; recorded in the report so the
#: diagnostic block is self-describing.
TRACK_L_LADDER_REF = "docs/12_CLAIM_LADDER.md Section 5"

_TRACK_L_NOTE = (
    "DIAGNOSTIC only: this block states whether the Track L re-enable "
    "conditions (docs/12_CLAIM_LADDER.md Section 5) hold in this run's data. "
    "Unlocking the conditional rung is a documented ladder decision "
    "(G-009 / TRD G-013) and is NEVER an automatic effect of this report; "
    "no comparative leakage claim is made or implied here. All values remain "
    "a PROXY over a controlled testbed."
)

# Repo-rooted config locations (resolved relative to the package, not the cwd).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASES_CFG_PATH = _REPO_ROOT / "configs" / "basis" / "bases.yaml"
_ARCHIVE_CFG_PATH = _REPO_ROOT / "configs" / "archive" / "archive.yaml"
_HORIZON_CFG_PATH = _REPO_ROOT / "configs" / "experiments" / "horizon.yaml"
_POLICY_CFG_DIR = _REPO_ROOT / "configs" / "policies"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"


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


# ---------------------------------------------------------------------------
# Pure cross-family aggregation helpers (unit-testable on synthetic rows).
# ---------------------------------------------------------------------------


def build_cross_family_excess_block(
    rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate per-channel ``excess_nmi`` across seed families (E-019 core).

    ``rows_by_family`` maps family (str(base_seed), in family order) to that
    family's per-policy leakage rows; each row must carry
    ``<channel>_excess_nmi`` and ``<channel>_saturation_flag`` for every
    channel in :data:`CHANNEL_NAMES` (the D-016 / D-017 row keys).

    Per policy x channel the block carries:

    - ``perFamilyValues``: the aligned per-family ``excess_nmi`` values,
    - the :func:`aggregate_values` stats (mean / std / n / seeded-bootstrap
      CI / ``ci_method`` / ``sufficient_n``) — the cross-family CI that
      D-015 marked as "future E-019"; the resampling unit is a seed FAMILY,
    - ``signConsistent``: all values strictly positive or all strictly
      negative (an exact 0.0 breaks consistency — fail closed),
    - ``allFamiliesUnsaturated``: that channel's ``saturation_flag`` is
      False in EVERY family (D-017; one saturated family poisons the well).

    Raises a Korean :class:`ValueError` on an empty mapping or when the
    families disagree on the policy set (misaligned units are never
    aggregated silently).
    """
    if not rows_by_family:
        raise ValueError(
            "E-019 집계: rows_by_family 가 비어 있습니다 (패밀리가 최소 1개 필요)."
        )

    families = list(rows_by_family)
    policies = sorted({str(row["policy"]) for row in rows_by_family[families[0]]})

    rows_map: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for family in families:
        family_rows = list(rows_by_family[family])
        by_policy = {str(row["policy"]): row for row in family_rows}
        # A duplicate policy row would silently shadow an earlier one in the
        # dict build above — reject it (misaligned units are never aggregated
        # silently).
        if len(by_policy) != len(family_rows):
            raise ValueError(
                "E-019 집계: 한 패밀리에 동일 정책 행이 중복되어 있습니다 "
                f"(family={family}, rows={len(family_rows)}, "
                f"policies={len(by_policy)})."
            )
        if sorted(by_policy) != policies:
            raise ValueError(
                "E-019 집계: 패밀리 간 정책 집합이 일치하지 않습니다 "
                f"(family={family}, expected={policies}, got={sorted(by_policy)})."
            )
        rows_map[family] = by_policy

    # The seeded bootstrap inside aggregate_values derives its seed from the
    # ORDERED values list. Feed it the values in name-sorted family order so
    # the CI — and the boolean Track L verdicts derived from it — depend on
    # the family SET, never on the --base-seeds argument order.
    canonical_families = sorted(families)

    per_policy: Dict[str, Any] = {}
    for policy in policies:
        channel_blocks: Dict[str, Any] = {}
        for channel in CHANNEL_NAMES:
            value_key = f"{channel}_excess_nmi"
            flag_key = f"{channel}_saturation_flag"
            values = [
                float(rows_map[family][policy][value_key]) for family in families
            ]
            canonical_values = [
                float(rows_map[family][policy][value_key])
                for family in canonical_families
            ]
            agg = aggregate_values(
                canonical_values, f"excess_nmi@{policy}@{channel}"
            )
            sign_consistent = all(v > 0.0 for v in values) or all(
                v < 0.0 for v in values
            )
            all_unsaturated = all(
                not bool(rows_map[family][policy][flag_key])
                for family in families
            )
            channel_blocks[channel] = {
                "channel": channel,
                "perFamilyValues": values,
                **agg,
                "signConsistent": sign_consistent,
                "allFamiliesUnsaturated": all_unsaturated,
            }
        per_policy[policy] = channel_blocks

    return {
        "metric": "excess_nmi",
        "unit": "seed_family",
        "ciAvailable": True,
        "ciNote": (
            "Cross-family seeded-bootstrap CI of the null-corrected excess "
            "NMI (D-015); the resampling unit is a seed FAMILY (E-014 "
            "convention). This is the CI the D-015 task marked as 'future "
            "E-019'. The bootstrap input is ordered by name-sorted family "
            "key, so the CI is invariant to the base_seeds argument order; "
            "perFamilyValues stays in caller family order for readability. "
            "The values remain a PROXY."
        ),
        "families": families,
        "policies": policies,
        "perPolicy": per_policy,
    }


def evaluate_track_l_conditions(
    cross_block: Mapping[str, Any],
    min_families: int = TRACK_L_MIN_FAMILIES,
) -> Dict[str, Any]:
    """Machine-evaluate the four Track L re-enable conditions (DIAGNOSTIC).

    Per policy x channel of ``cross_block`` (a
    :func:`build_cross_family_excess_block` result):

    1. ``condition1_excess_positive_ci`` — the excess is positive with a CI
       excluding 0 on the positive side (``mean > 0`` and ``ci_low > 0``)
       and the CI is a real bootstrap CI (``sufficient_n``).
    2. ``condition2_not_saturated`` — the channel's ``saturation_flag`` is
       False in every family (D-017).
    3. ``condition3_channel_named`` — structurally True here: the block is
       keyed by channel, so any citation of it names its channel (D-016).
    4. ``condition4_cross_family_stable`` — at least ``min_families``
       families, consistent sign across all of them, and a sufficient-n CI
       excluding 0 (either side).

    ``allConditionsHold`` is the conjunction. The block is a DIAGNOSTIC:
    it gates claims and never supports one; the unlock decision belongs to
    the claim ladder (G-009), never to this function.
    """
    per_policy_out: Dict[str, Any] = {}
    for policy in cross_block["policies"]:
        channel_out: Dict[str, Any] = {}
        for channel, entry in cross_block["perPolicy"][policy].items():
            sufficient = bool(entry["sufficient_n"])
            ci_excludes_zero = (
                float(entry["ci_low"]) > 0.0 or float(entry["ci_high"]) < 0.0
            )
            condition1 = (
                sufficient
                and float(entry["mean"]) > 0.0
                and float(entry["ci_low"]) > 0.0
            )
            condition2 = bool(entry["allFamiliesUnsaturated"])
            condition3 = True
            condition4 = (
                int(entry["n"]) >= int(min_families)
                and bool(entry["signConsistent"])
                and sufficient
                and ci_excludes_zero
            )
            channel_out[channel] = {
                "channel": channel,
                "condition1_excess_positive_ci": condition1,
                "condition2_not_saturated": condition2,
                "condition3_channel_named": condition3,
                "condition4_cross_family_stable": condition4,
                "allConditionsHold": (
                    condition1 and condition2 and condition3 and condition4
                ),
            }
        per_policy_out[policy] = channel_out

    return {
        "minFamilies": int(min_families),
        "ladderRef": TRACK_L_LADDER_REF,
        "note": _TRACK_L_NOTE,
        "perPolicy": per_policy_out,
    }


def _pair_keys(pairs: Any) -> List[str]:
    """``"A|B"`` strings for a list of overlap pair dicts (audit order kept)."""
    return [f"{p['probe_a']}|{p['probe_b']}" for p in pairs]


def build_overlap_caveat(
    overlap_by_family: Mapping[str, Mapping],
) -> Dict[str, Any]:
    """Build the cross-family probe-overlap caveat block (Task B-009, pure).

    ``overlap_by_family`` maps family key to that family's B-007/B-009
    ``probe_overlap_audit`` result (only ``high_overlap_pairs`` and
    ``exclude_merge_candidates`` are read). The block summarises the
    two-tier overlap policy outcome across families:

    - ``flaggedPairsByFamily``: family -> ``"A|B"`` strings from that
      family's ``high_overlap_pairs`` (>= flag threshold, diagnostic caveat),
    - ``excludeMergeCandidatesByFamily``: same shape for the exclude tier
      (>= exclude threshold, exclude-or-merge candidates),
    - ``familiesWithFlags`` / ``totalFamilies``: flag prevalence counts,
    - ``caveatNote``: an English note generated FROM the data. A flagged
      pair in a strict minority of families (and nothing at the exclude
      tier) is a diagnostic caveat, NOT a blocking failure — the E-019
      production observation (1 pair in family 42 only, overlap 0.929) is
      exactly this shape. Exclude-tier candidates are named as
      exclude-or-merge candidates for probe-set curation. Every wording is
      a diagnostic over instrumented input policies, never a claim.

    Raises a Korean :class:`ValueError` on an empty mapping (fail closed).
    """
    if not overlap_by_family:
        raise ValueError(
            "E-019 overlap 캐비앳: overlap_by_family 가 비어 있습니다 "
            "(패밀리가 최소 1개 필요)."
        )

    flagged_by_family: Dict[str, List[str]] = {}
    exclude_by_family: Dict[str, List[str]] = {}
    for family in overlap_by_family:
        audit = overlap_by_family[family]
        flagged_by_family[str(family)] = _pair_keys(
            audit.get("high_overlap_pairs", [])
        )
        exclude_by_family[str(family)] = _pair_keys(
            audit.get("exclude_merge_candidates", [])
        )

    total_families = len(flagged_by_family)
    families_with_flags = sum(
        1 for pairs in flagged_by_family.values() if pairs
    )
    families_with_exclude = sum(
        1 for pairs in exclude_by_family.values() if pairs
    )

    if families_with_exclude > 0:
        caveat_note = (
            "Probe pair(s) at or above the exclude threshold were detected "
            f"in {families_with_exclude} of {total_families} seed families; "
            "these pairs are exclude-or-merge candidates for probe-set "
            "curation. This remains a diagnostic over instrumented input "
            "policies on a controlled testbed, not a claim."
        )
    elif families_with_flags == 0:
        caveat_note = (
            "No probe pair exceeded the flag threshold in any seed family; "
            "no overlap caveat applies."
        )
    elif 2 * families_with_flags < total_families:
        caveat_note = (
            "High-overlap probe pair(s) were detected in "
            f"{families_with_flags} of {total_families} seed families and "
            "should be treated as a diagnostic caveat rather than a blocking "
            "failure; no pair reached the exclude threshold. This is a "
            "diagnostic, not a claim."
        )
    else:
        caveat_note = (
            "High-overlap probe pair(s) were detected in "
            f"{families_with_flags} of {total_families} seed families (not a "
            "strict minority); the flagged pairs warrant probe-set review "
            "before interpreting probe-separability values, though no pair "
            "reached the exclude threshold. This is a diagnostic, not a "
            "claim."
        )

    return {
        "flaggedPairsByFamily": flagged_by_family,
        "excludeMergeCandidatesByFamily": exclude_by_family,
        "familiesWithFlags": families_with_flags,
        "totalFamilies": total_families,
        "caveatNote": caveat_note,
    }


# ---------------------------------------------------------------------------
# Per-family computation.
# ---------------------------------------------------------------------------


class _PrefixTraceView:
    """Read-only TraceState-like view over a fixed round-record prefix.

    ``probe_overlap_audit`` documents its ``trace`` as "``None`` or a
    TraceState-like object exposing ``rounds()``", and every registered probe
    reads the trace ONLY through ``rounds()``. A view over the already
    validated, already hash-chained stored records therefore reproduces the
    decision-time history exactly, without re-running ``append_round``'s
    validation + chained re-hashing per prefix (which would be O(H^2) hashed
    appends per probe).
    """

    def __init__(self, rounds: Sequence[Mapping[str, Any]]) -> None:
        self._rounds = list(rounds)

    def rounds(self) -> List[Mapping[str, Any]]:
        return list(self._rounds)

    def __len__(self) -> int:
        return len(self._rounds)


def _decision_contexts(
    traces_by_probe: Mapping[str, TraceState],
    pool: Sequence[Mapping[str, Any]],
) -> List[Tuple[List[Mapping[str, Any]], _PrefixTraceView]]:
    """Reconstruct decision-time ``(slate dicts, prefix trace)`` contexts.

    For every probe episode and every round ``t``, the context is the slate
    the probe saw at round ``t`` (cardIds mapped back to observable card
    dicts) together with the trace prefix of rounds ``0..t-1`` (the history
    available at decision time), exposed as a read-only
    :class:`_PrefixTraceView`. Contexts are a pure function of the recorded
    traces.
    """
    by_id = {card["cardId"]: card for card in pool}
    contexts: List[Tuple[List[Mapping[str, Any]], _PrefixTraceView]] = []
    for probe_name in sorted(traces_by_probe):
        rounds = traces_by_probe[probe_name].rounds()
        for t in range(len(rounds)):
            slate_dicts = [by_id[cid] for cid in rounds[t]["slate"]]
            contexts.append((slate_dicts, _PrefixTraceView(rounds[:t])))
    return contexts


def _build_family_pool(
    bases: Any,
    archive_cfg: Mapping[str, Any],
    base_seed: int,
    pool_size: int,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Build one family's archive + deterministic pool, fail closed.

    Single source of the pool derivation (archive build, head slice, size
    check, pool hash) shared by the dry-run plan and the real run, so the
    plan hashes can never silently diverge from the executed run's hashes.

    Returns ``(archiveHash, pool, poolHash)``.
    """
    archive = build_archive(bases, dict(archive_cfg), int(base_seed))
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"E-019 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            f"보다 작습니다 (base_seed={base_seed})"
        )
    return (
        archive["archiveHash"],
        pool,
        canonical_hash([c["cardId"] for c in pool]),
    )


def _run_family(
    base_seed: int,
    H: int,
    k: int,
    pool_size: int,
    n_permutations: int,
    bases: Any,
    archive_cfg: Mapping[str, Any],
    policy_cfgs: Mapping[str, Mapping[str, Any]],
    probe_versions: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run the expanded-probe leakage diagnostic for ONE seed family.

    Pure function of its arguments (every episode, null permutation and
    overlap audit is seeded from ``base_seed`` / the data itself), so the
    inline replay audit can recompute it bit-identically.

    Returns the JSON-ready family block: ``archiveHash`` / ``poolHash``,
    the per-policy ``table`` (D-015 + D-016 + D-017 + delta keys, plus the
    full per-channel ``channels`` blocks), the B-007 ``probeOverlap`` audit,
    and the family's slate/trace hash lists for the report hash chain.
    """
    archive_hash, pool, pool_hash = _build_family_pool(
        bases, archive_cfg, base_seed, pool_size
    )

    rows: List[Dict[str, Any]] = []
    slate_hashes: List[str] = []
    trace_hashes: List[str] = []
    reference_traces: Mapping[str, TraceState] | None = None

    for policy_name in sorted(E3_LEAKAGE_POLICIES):
        policy_cls, _cfg_file = E3_LEAKAGE_POLICIES[policy_name]
        cfg = policy_cfgs[policy_name]

        traces_by_probe: Dict[str, TraceState] = {}
        for probe_name in EXPANDED_PROBE_SET:
            probe = get_probe(probe_name)
            probe_trace = run_episode(
                pool,
                policy_cls(dict(cfg)),
                base_seed,
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

        if policy_name == E3_LEAKAGE_DELTA_REFERENCE:
            reference_traces = traces_by_probe

        leak = leakage_proxy_with_metadata(
            traces_by_probe, probe_versions, n_permutations=n_permutations
        )
        null_corrected = leak["nullCorrected"]
        channel_sep = channel_separated_separability(
            traces_by_probe,
            n_permutations=n_permutations,
            precomputed_combined=null_corrected,
        )
        rows.append(
            {
                "policy": policy_name,
                "policyVersion": policy_cls(dict(cfg)).policy_version(),
                "leakage_proxy": leak["value"],
                "isProxy": leak["isProxy"],
                # D-015 quintet + D-016 trio + D-017 flag quartet via the
                # shared row fragment (same single source of truth as the E3
                # leakage rows): observed/null/excess always travel together;
                # the saturation flags are diagnostics, never claims.
                **separability_row_fields(null_corrected, channel_sep),
                # E-019 extra: the full per-channel blocks for the diagnostic.
                "channels": channel_sep["channels"],
                "traceHashes": leak["traceHashes"],
                "nullPermutations": int(null_corrected["n_permutations"]),
            }
        )
        log_ko(
            _logger,
            "E-019 정책 진단 완료: "
            f"family={base_seed}, policy={policy_name}, "
            f"excess_nmi={null_corrected['excess_nmi']:+.6f}, "
            f"slate={channel_sep['slate_excess_nmi']:+.6f}, "
            f"selection={channel_sep['selection_excess_nmi']:+.6f}, "
            f"saturation_flag={channel_sep['combined_saturation_flag']} "
            "(확장 프로브 7종; PROXY 통계이며 클레임이 아닙니다)",
        )

    # D-011 convention: RELATIVE delta vs the seed-aligned reference policy.
    reference_rows = [
        row for row in rows if row["policy"] == E3_LEAKAGE_DELTA_REFERENCE
    ]
    if not reference_rows:
        raise ValueError(
            "E-019 실행: 기준 정책 행을 찾을 수 없습니다 "
            f"(deltaReference={E3_LEAKAGE_DELTA_REFERENCE!r})."
        )
    reference_leakage = reference_rows[0]["leakage_proxy"]
    for row in rows:
        row["leakage_delta_vs_random"] = leakage_delta_vs_random(
            row["leakage_proxy"], reference_leakage
        )

    # B-007 probe-overlap audit over the reference policy's decision-time
    # contexts (slate seen + trace prefix available at that round). The
    # audited probe set is passed EXPLICITLY so it stays the set this run
    # measured, even if EXPANDED_PROBE_SET is ever curated away from the
    # full registry default that probe_overlap_audit falls back to.
    assert reference_traces is not None  # guaranteed by the guard above
    contexts = _decision_contexts(reference_traces, pool)
    overlap = probe_overlap_audit(
        contexts,
        probes={name: get_probe(name) for name in EXPANDED_PROBE_SET},
        seed=int(base_seed),
    )

    return {
        "base_seed": int(base_seed),
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "contextPolicy": E3_LEAKAGE_DELTA_REFERENCE,
        "table": rows,
        "probeOverlap": overlap,
        "slateHashes": slate_hashes,
        "traceHashes": trace_hashes,
    }


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


def _validate_args(
    base_seeds: Sequence[int], n_permutations: int, pool_size: int
) -> Tuple[int, ...]:
    """Validate base_seeds / n_permutations / pool_size, failing closed (Korean)."""
    seeds = tuple(int(s) for s in base_seeds)
    if not seeds:
        raise ValueError(
            "E-019 실행: base_seeds 가 비어 있습니다 (패밀리가 최소 1개 필요)."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"E-019 실행: base_seeds 에 중복된 시드가 있습니다: {list(seeds)}."
        )
    if int(n_permutations) < 1:
        raise ValueError(
            f"E-019 실행: n_permutations 는 1 이상이어야 합니다 "
            f"(입력값: {n_permutations})."
        )
    # pool_size < 1 would otherwise slip through: a negative value silently
    # truncates the pool from the END via the head slice and the
    # ``len(pool) < pool_size`` guard can never fire.
    if int(pool_size) < 1:
        raise ValueError(
            f"E-019 실행: pool_size 는 1 이상이어야 합니다 (입력값: {pool_size})."
        )
    return seeds


def run_leakage_diagnostic(
    base_seeds: Sequence[int] = DEFAULT_BASE_SEEDS,
    H: int | None = None,
    k: int = 4,
    pool_size: int = 64,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    dry_run: bool = False,
    replay_validate: bool = True,
) -> dict:
    """Run the E-019 expanded-probe leakage diagnostic and return a report.

    Args:
        base_seeds: the base-seed families (>= TRACK_L_MIN_FAMILIES = 5 for a
            Track L condition-4 evaluation that can pass; smaller tuples are
            allowed for development/tests and simply fail condition 4).
        H: horizon (rounds); the horizon config default when ``None``.
        k: slate size (main parameter 4).
        pool_size: candidate pool size (main parameter 64).
        n_permutations: D-015 permutation-null size per channel (the
            documented default 200 for production; smaller for development).
        dry_run: validate configs + freeze, compute per-family archive/pool
            hashes, write nothing, run no episodes.
        replay_validate: when true, recompute the FIRST family a second time
            and compare canonical hashes (mode ``inline_recompute_first_family``);
            a divergence is reported LOUDLY as ``replayable=False``.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    seeds = _validate_args(base_seeds, n_permutations, pool_size)

    # 0. C-011 config-freeze gate (hard Korean ValueError before anything runs).
    freeze = verify_trace_greedy_freeze()

    # The probe registry is fixed for the whole run: resolve every probe
    # version ONCE and thread the same mapping into each family (and the
    # report metadata), so the self-describing report cannot drift from the
    # versions actually hashed into the rows.
    probe_versions = {
        name: get_probe(name).probe_version() for name in EXPANDED_PROBE_SET
    }

    # 1. Configs + horizon.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in E3_LEAKAGE_POLICIES.items():
        cfg = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))
        cfg["k"] = int(k)
        policy_cfgs[name] = cfg

    run_params = {
        "base_seeds": [int(s) for s in seeds],
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
        "n_permutations": int(n_permutations),
        "probeSet": "expanded_registry",
        "probes": list(EXPANDED_PROBE_SET),
        "policies": sorted(E3_LEAKAGE_POLICIES),
        "deltaReference": E3_LEAKAGE_DELTA_REFERENCE,
        "overlapThreshold": PROBE_OVERLAP_THRESHOLD,
        "overlapExcludeThreshold": PROBE_OVERLAP_EXCLUDE_THRESHOLD,
        "minFamilies": TRACK_L_MIN_FAMILIES,
        "configFreeze": {
            "policyName": freeze["policyName"],
            "policyEffectiveConfigHash": freeze["frozenHash"],
            "taskId": freeze["taskId"],
        },
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "policy_cfgs": policy_cfgs,
            "run_params": run_params,
        }
    )

    # 2. Dry run: per-family archive/pool plan hashes, write nothing.
    if dry_run:
        families_plan: Dict[str, Any] = {}
        for seed in seeds:
            archive_hash, _pool, pool_hash = _build_family_pool(
                bases, archive_cfg, int(seed), pool_size
            )
            families_plan[str(seed)] = {
                "archiveHash": archive_hash,
                "poolHash": pool_hash,
            }
        log_ko(
            _logger,
            "E-019 드라이런 요약: "
            f"families={[int(s) for s in seeds]}, "
            f"probes={len(EXPANDED_PROBE_SET)}종(확장), "
            f"policies={sorted(E3_LEAKAGE_POLICIES)}, H={H}, k={k}, "
            f"pool_size={pool_size}, n_permutations={n_permutations}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "configHash": config_hash,
            "configFreeze": freeze,
            "families": families_plan,
        }

    # 3. Run every family (the slow unit: Korean start/end logs per family).
    family_blocks: Dict[str, Dict[str, Any]] = {}
    for idx, seed in enumerate(seeds, start=1):
        log_ko(
            _logger,
            f"E-019 패밀리 시작 ({idx}/{len(seeds)}): base_seed={seed}, "
            f"probes={len(EXPANDED_PROBE_SET)}종(확장 레지스트리), "
            f"H={H}, k={k}, pool_size={pool_size}.",
        )
        family_blocks[str(seed)] = _run_family(
            int(seed), int(H), int(k), int(pool_size), int(n_permutations),
            bases, archive_cfg, policy_cfgs, probe_versions,
        )
        log_ko(
            _logger,
            f"E-019 패밀리 완료 ({idx}/{len(seeds)}): base_seed={seed}, "
            f"overlap 과다 쌍="
            f"{len(family_blocks[str(seed)]['probeOverlap']['high_overlap_pairs'])}개.",
        )

    # 4. Cross-family aggregation + Track L condition evaluation (pure).
    cross_block = build_cross_family_excess_block(
        {family: block["table"] for family, block in family_blocks.items()}
    )
    track_l = evaluate_track_l_conditions(cross_block)
    # B-009: cross-family two-tier overlap caveat (pure summary of the
    # per-family probeOverlap audits; the full audits stay embedded per family).
    overlap_caveat = build_overlap_caveat(
        {family: block["probeOverlap"] for family, block in family_blocks.items()}
    )
    log_ko(
        _logger,
        "E-019 overlap 캐비앳 요약: "
        f"flag 패밀리 {overlap_caveat['familiesWithFlags']}/"
        f"{overlap_caveat['totalFamilies']}개, exclude 후보 패밀리 "
        f"{sum(1 for v in overlap_caveat['excludeMergeCandidatesByFamily'].values() if v)}개 "
        "(2단계 정책: >=0.9 flag 진단 캐비앳, >=0.95 제외/병합 후보 — "
        "진단이며 클레임이 아닙니다).",
    )
    log_ko(
        _logger,
        "E-019 교차 패밀리 집계 완료: "
        f"families={len(seeds)}, policies={len(cross_block['policies'])}, "
        "Track L 조건 평가는 진단(DIAGNOSTIC)이며 클레임 잠금 해제는 "
        "클레임 래더의 문서화된 결정(G-009)으로만 이루어집니다.",
    )

    # 5. Inline replay audit: recompute the FIRST family and compare hashes.
    first_family = str(seeds[0])
    if replay_validate:
        recomputed = _run_family(
            int(seeds[0]), int(H), int(k), int(pool_size), int(n_permutations),
            bases, archive_cfg, policy_cfgs, probe_versions,
        )
        original_hash = canonical_hash(family_blocks[first_family])
        recomputed_hash = canonical_hash(recomputed)
        replayable = original_hash == recomputed_hash
        if not replayable:
            log_ko(
                _logger,
                "E-019 리플레이 감사 실패(LOUD): "
                f"family={first_family} 재계산 해시가 다릅니다 "
                f"(original={original_hash[:12]}, "
                f"recomputed={recomputed_hash[:12]}) — 재현 불가는 숨기지 "
                "않고 보고합니다",
            )
        else:
            log_ko(
                _logger,
                "E-019 리플레이 감사 통과: "
                f"family={first_family} 재계산이 비트 동일합니다 "
                f"(hash={original_hash[:12]}).",
            )
        replay_section: Dict[str, Any] = {
            "mode": "inline_recompute_first_family",
            "scope": "first_family_only",
            "scopeNote": (
                "The inline audit recomputes ONLY the first family and "
                "compares canonical hashes; families 2..N are not re-run "
                "(the cross-family aggregation layer is a pure function of "
                "the family blocks). replayable=true therefore attests the "
                "first family's bit-identity, not a whole-run re-execution; "
                "a full re-run replay is the replay-validator path."
            ),
            "family": first_family,
            "replayable": replayable,
            "familyBlockHash": original_hash,
            "recomputedBlockHash": recomputed_hash,
        }
    else:
        replay_section = {
            "mode": "inline_recompute_first_family",
            "scope": "skipped",
            "family": first_family,
            "replayable": None,
            "note": "replay_validate=False — 인라인 재계산을 건너뛰었습니다.",
        }

    # 6. Self-describing leakage metadata (PROXY framing travels with the
    #    data). nullPermutations is the validated run parameter threaded into
    #    every D-015/D-016 call (each row's own nullPermutations is read back
    #    from its computed block); probeVersions is the same mapping threaded
    #    into every family.
    leakage_meta = {
        # G-020: primary report label is probe_separability_proxy; the legacy
        # machine name is kept as legacyAlias (D-012 precedent). Per-family
        # table-row machine keys stay "leakage_proxy" byte-identically.
        "metric": "probe_separability_proxy",
        "legacyAlias": "leakage_proxy",
        "isProxy": IS_PROXY,
        "disclaimer": PROXY_DISCLAIMER,
        "deltaReference": E3_LEAKAGE_DELTA_REFERENCE,
        "nullPermutations": int(n_permutations),
        "saturationThreshold": SATURATION_UNIQUE_RATE_THRESHOLD,
        "overlapThreshold": PROBE_OVERLAP_THRESHOLD,
        "overlapExcludeThreshold": PROBE_OVERLAP_EXCLUDE_THRESHOLD,
        "probeVersions": dict(probe_versions),
    }

    # 7. Hash chain + seedBatchId. Family hashes aggregate the children's.
    seed_batch_id = canonical_hash(
        {
            "experiment": "E_LEAKAGE_DIAGNOSTIC",
            "base_seeds": run_params["base_seeds"],
            "H": int(H),
            "k": int(k),
            "pool_size": int(pool_size),
            "n_permutations": int(n_permutations),
            "probes": list(EXPANDED_PROBE_SET),
            "policies": sorted(E3_LEAKAGE_POLICIES),
        }
    )
    archive_hash = canonical_hash(
        {family: block["archiveHash"] for family, block in family_blocks.items()}
    )
    pool_hash = canonical_hash(
        {family: block["poolHash"] for family, block in family_blocks.items()}
    )
    slate_hash = canonical_hash(
        {family: block["slateHashes"] for family, block in family_blocks.items()}
    )
    trace_hash = canonical_hash(
        {family: block["traceHashes"] for family, block in family_blocks.items()}
    )

    per_family = {
        family: {
            key: value
            for key, value in block.items()
            if key not in ("slateHashes", "traceHashes")
        }
        for family, block in family_blocks.items()
    }

    results_body = {
        "config": run_params,
        "perFamily": per_family,
        "crossFamilyExcess": cross_block,
        "overlapCaveat": overlap_caveat,
        "trackLConditions": track_l,
        "leakageMeta": leakage_meta,
        "replayAudit": replay_section,
    }
    output_hash = canonical_hash(results_body)

    # 8. Report (reportHash computed over the report-minus-reportHash).
    report: Dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "schemaVersion": DIAGNOSTIC_SCHEMA_VERSION,
        "experiment": "E_LEAKAGE_DIAGNOSTIC",
        "phaseNote": (
            "E-019 (TRD E-019, Track L hardening): probe-separability "
            "diagnostic over the EXPANDED B-007 probe registry across "
            "multiple base-seed families. Reports null-corrected (D-015), "
            "channel-separated (D-016), saturation-diagnosed (D-017) excess "
            "NMI with a cross-family seeded-bootstrap CI (unit=seed_family), "
            "plus the B-007 probe-overlap audit under the B-009 two-tier "
            "overlap policy (>= 0.9 flagged as a diagnostic caveat; >= 0.95 "
            "exclude-or-merge candidate; see overlapCaveat). The Track L "
            "conditions "
            "block is a DIAGNOSTIC: unlocking the conditional rung is a "
            "documented ladder decision (G-009), never an automatic effect "
            "of this report. Every value is a PROXY — a system-level "
            "statistic over a controlled testbed, NOT a privacy guarantee, "
            "anonymity proof, identifiability bound, or legal/compliance "
            "claim. Strategy probes are controlled instrumented inputs, not "
            "synthetic users; no real-world generalization claim. E2/E3 "
            "remain on the frozen DEFAULT_PROBE_SET, untouched."
        ),
        "config": run_params,
        "configFreeze": freeze,
        "seedBatchId": seed_batch_id,
        "perFamily": per_family,
        "crossFamilyExcess": cross_block,
        "overlapCaveat": overlap_caveat,
        "trackLConditions": track_l,
        "leakageMeta": leakage_meta,
        "replayAudit": replay_section,
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
        "hashSemantics": (
            "Report-level archiveHash / poolHash / slateHash / traceHash are "
            "canonical hashes over the per-family hash maps (family -> "
            "hash(es)), mirroring the E-014 convention; they are NOT a "
            "re-hash of raw artifacts."
        ),
    }
    report_hash = canonical_hash(report)
    report["reportHash"] = report_hash

    # 9. Reproducibility pack.
    pack = ReproducibilityPack(
        configHash=config_hash,
        commitHash=_git_commit_hash(),
        archiveHash=archive_hash,
        poolHash=pool_hash,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=report_hash,
        seedBatchId=seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 10. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"leakage_diagnostic_{seed_batch_id[:12]}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "E-019 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={seed_batch_id[:12]}, families={len(seeds)}, "
        f"replayable={replay_section['replayable']}.",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run the diagnostic, log a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.e_leakage_diagnostic",
        description=(
            "ECHO-Bench E-019 expanded-probe leakage diagnostic runner."
        ),
    )
    parser.add_argument(
        "--base-seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_BASE_SEEDS),
        help="comma-separated base-seed families (default: 42,7,101,2025,31337)",
    )
    parser.add_argument(
        "--H", type=int, default=None, help="horizon (default: config default)"
    )
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=64, help="candidate pool size"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_NULL_PERMUTATIONS,
        help="D-015 permutation-null size per channel (default: 200)",
    )
    parser.add_argument(
        "--no-replay-validate",
        action="store_true",
        help="skip the inline first-family recompute replay audit",
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
    result = run_leakage_diagnostic(
        base_seeds=base_seeds,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        n_permutations=args.n_permutations,
        dry_run=args.dry_run,
        replay_validate=not args.no_replay_validate,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "E-019 드라이런 완료: "
            f"families={result['config']['base_seeds']}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "E-019 실행 완료: "
            f"families={result['config']['base_seeds']}, "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"replayable={result['replayAudit']['replayable']}.",
        )


if __name__ == "__main__":
    main()
