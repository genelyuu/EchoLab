"""S3 Coordinate Scramble runner for ECHO-Bench (Task E-006, Phase 3 supp.).

S3 tests whether trace-only utility depends on the **coordinate semantics** of
the cards (which coordinate vector belongs to which card) or only on the
**structure** of the coordinate set. It does so with a deterministic, seeded
coordinate-scramble transform and compares baseline vs. scrambled metrics,
quantifying the shift with :func:`echo_bench.metrics.robustness.robustness_score`.

The scramble transform (pure deterministic permutation)
-------------------------------------------------------
:func:`scramble_coordinates` is a pure function ``pool, seed -> new_pool`` that
applies a single seeded permutation of the ``coordinateContribution`` vectors
**across** the cards in the pool: card ``i`` keeps every other observable field
(``cardId``, ``basis``, ``complexityBand``, ``salienceScore``, ...) but receives
the ``coordinateContribution`` that card ``perm[i]`` had in the baseline pool.
Because it permutes the *existing* vectors, the multiset of coordinate vectors in
the pool is preserved exactly (structure unchanged) while the card->coordinate
association is shuffled (semantics broken). It is order-independent and
reproducible: the permutation is drawn from a *local* ``random.Random`` seeded
via :func:`canonical_hash` over the sorted cardIds + the scramble seed, never the
global RNG or wall-clock, so identical ``(pool, seed)`` always yields the same
scrambled pool.

What is recorded vs. what is asserted
-------------------------------------
The runner records, per policy and per seed batch, the full-episode metric shift
``robustness_score(baseline_metrics, scrambled_metrics)`` (a quantification of
how much the trace-only utility metrics move under the scramble) AND a
**round-0 selection-divergence rate** — the fraction of seeds for which the very
first selected ``cardId`` differs baseline vs. scrambled.

The round-0 signal is the clean, contamination-free direction probe: at round 0
the trace is empty, so every policy's RNG / context is seeded from an identical
(scramble-invariant) state, isolating *pure* coordinate dependence. ``RANDOM``
seeds its sampling from the (identical) ``candidatePoolHash`` and samples by
index, never reading ``coordinateContribution``, so its round-0 selection is
provably identical baseline vs. scrambled (divergence ``0``). Coordinate-driven
policies (``TRACE_GREEDY``, ``TRACE_LIN_UCB``) score candidates by their
coordinate vectors, so the scramble changes their round-0 pick. (In later rounds
the recorded coordinate vectors feed back through ``traceHash``, which contaminates
even ``RANDOM`` — hence the *direction* invariant is asserted on the clean round-0
signal, while the full-episode ``robustness_score`` is reported as the magnitude
quantification.)

Expectation (asserted as direction, not magnitude)
--------------------------------------------------
The report flag ``coordinatePoliciesShiftAtLeastControl`` asserts that each
coordinate-driven policy's round-0 selection-divergence rate is ``>=`` the
``RANDOM`` control's — i.e. coordinate policies shift at least as much as the
control. Direction only; no magnitude threshold.

Pipeline mirrors ``experiments/e1_horizon.py`` / ``experiments/e2_policy.py``:
load configs, build the archive + pool, scramble, run baseline & scrambled
episodes, compute metrics, hash everything (the permutation is logged into the
report), build a :class:`ReproducibilityPack`, and write
``outputs/reports/s3_coordinate_scramble_<id>.json``. ``--dry-run`` validates
config + computes the available hashes (incl. the scramble permutation) and
writes nothing.

Guardrails
----------
No user / persona / emotion / preference / user_model / free-text field enters
the pool, the traces, the metrics, or the report. The report carries only
system-level hashes, the controlled scramble permutation, and trace-only
utility/robustness metrics; it makes no claim about user preference, experience,
emotion, wellbeing, or legal compliance, and no real-world generalization claim.

All identifiers, metric names, config keys and file paths stay English; runtime
log messages are Korean per the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from echo_bench.archive.builder import build_archive
from echo_bench.basis.schema import load_bases
from echo_bench.env.horizon import default_h, load_horizon, validate_h
from echo_bench.env.round_runner import run_episode
from echo_bench.env.seed_batch import derive_child_seeds, seed_batch_id
from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.metrics.aggregate import aggregate_values
from echo_bench.metrics.robustness import ROBUSTNESS_DIRECTION, robustness_score
from echo_bench.metrics.utility import CORE_METRIC_KEYS, METRIC_KEYS, compute_all
from echo_bench.policies.random import RandomPolicy
from echo_bench.policies.trace_greedy import TraceGreedyPolicy
from echo_bench.policies.trace_lin_ucb import TraceLinUcbPolicy
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "run_s3_coordinate_scramble",
    "scramble_coordinates",
    "scramble_permutation",
    "main",
    "S3_POLICIES",
    "CONTROL_POLICY",
    "COORDINATE_POLICIES",
]

_logger = get_logger(__name__)

S3_SCHEMA = "echo_bench.s3_coordinate_scramble.report"
S3_SCHEMA_VERSION = "1"

# The coordinate-driven policies (read coordinateContribution) and the control
# policy (RANDOM, ignores coordinates). The control is the direction reference.
CONTROL_POLICY = "RANDOM"
COORDINATE_POLICIES = ("TRACE_GREEDY", "TRACE_LIN_UCB")

S3_POLICIES = {
    "RANDOM": (RandomPolicy, "random.yaml"),
    "TRACE_GREEDY": (TraceGreedyPolicy, "trace_greedy.yaml"),
    "TRACE_LIN_UCB": (TraceLinUcbPolicy, "trace_lin_ucb.yaml"),
}

# The trace-only metric keys compared baseline vs scrambled. Pinned to
# CORE_METRIC_KEYS (the original four utility keys) to preserve a stable
# denominator of four keys across the C-011 freeze boundary; the D-010
# distribution metrics (coordinate_entropy, cell_visit_gini,
# time_to_saturation) are excluded from the scramble-shift denominator.
# METRIC_KEYS is still imported for report completeness where needed.
S3_METRIC_KEYS = CORE_METRIC_KEYS

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


def scramble_permutation(
    pool: Sequence[Mapping[str, Any]], seed: Any
) -> List[int]:
    """Return the deterministic, seeded scramble permutation for ``pool``.

    The permutation is a list ``perm`` of length ``len(pool)`` that is a
    bijection over ``range(len(pool))``: under the scramble, card ``i`` receives
    the ``coordinateContribution`` of baseline card ``perm[i]``. The permutation
    is drawn from a *local* ``random.Random`` seeded by
    :func:`canonical_hash` over the pool's sorted cardIds plus ``seed``, so it is
    reproducible and independent of the input pool ordering. The global RNG /
    wall-clock are never touched.
    """
    n = len(pool)
    card_ids = sorted(str(c["cardId"]) for c in pool)
    seed_material = canonical_hash(
        {"cardIds": card_ids, "seed": seed, "transform": "coordinate_scramble"}
    )
    rng = random.Random(int(seed_material, 16))
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def scramble_coordinates(
    pool: Sequence[Mapping[str, Any]], seed: Any
) -> List[Dict[str, Any]]:
    """Permute ``coordinateContribution`` vectors across cards (pure function).

    Applies the single seeded permutation from :func:`scramble_permutation`:
    output card ``i`` is a shallow copy of input card ``i`` with its
    ``coordinateContribution`` replaced by that of input card ``perm[i]``. Every
    other observable field is left unchanged. The multiset of coordinate vectors
    is preserved exactly (structure unchanged); only the card->coordinate
    association is shuffled (semantics broken). Pure: the input pool is never
    mutated; identical ``(pool, seed)`` -> identical scrambled pool.
    """
    pool = list(pool)
    perm = scramble_permutation(pool, seed)
    out: List[Dict[str, Any]] = []
    for i, card in enumerate(pool):
        new_card = dict(card)
        donor = pool[perm[i]]
        donor_coord = donor.get("coordinateContribution")
        # Copy the donor's coordinate vector (list copy so the source is never
        # aliased into the scrambled card).
        new_card["coordinateContribution"] = (
            list(donor_coord) if donor_coord is not None else donor_coord
        )
        out.append(new_card)
    return out


def run_s3_coordinate_scramble(
    base_seed: int = 42,
    n: int = 10,
    H: int | None = None,
    k: int = 4,
    pool_size: int = 64,
    scramble_seed: int = 7,
    dry_run: bool = False,
) -> dict:
    """Run the S3 coordinate-scramble study and return a fully hashed report.

    For each policy in :data:`S3_POLICIES`, run a seed batch of ``n`` child seeds
    twice (baseline pool + coordinate-scrambled pool), compute the four
    trace-only metrics for each episode, and record per policy: the mean
    full-episode baseline->scrambled shift via
    :func:`echo_bench.metrics.robustness.robustness_score` (magnitude
    quantification) and the **round-0 selection-divergence rate** (the
    contamination-free direction signal). The report flag
    ``coordinatePoliciesShiftAtLeastControl`` asserts each coordinate policy's
    round-0 divergence is ``>=`` the RANDOM control's (direction, not magnitude).

    Args:
        base_seed: base integer seed; child seeds are derived from it.
        n: number of child seeds (episodes) per policy / pool variant.
        H: horizon (rounds); defaults to the horizon config's default when None.
        k: slate size (S3 fixes ``k=4``).
        pool_size: candidate pool size.
        scramble_seed: seed for the deterministic coordinate-scramble permutation.
        dry_run: when true, validate config + compute config/archive/pool/scramble
            hashes and the planned policy list, write no files, run no episodes.

    Returns:
        The report dict (a dry-run plan dict when ``dry_run`` is true).
    """
    # 1. Load configs and resolve the horizon.
    bases = load_bases(_BASES_CFG_PATH)
    archive_cfg = _load_yaml(_ARCHIVE_CFG_PATH)
    horizon_cfg = load_horizon(_HORIZON_CFG_PATH)
    H = default_h(horizon_cfg) if H is None else validate_h(int(H), horizon_cfg)

    policy_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, (_cls, cfg_file) in S3_POLICIES.items():
        cfg = dict(_load_yaml(_POLICY_CFG_DIR / cfg_file))
        cfg["k"] = int(k)
        policy_cfgs[name] = cfg

    run_params = {
        "base_seed": int(base_seed),
        "n": int(n),
        "H": int(H),
        "k": int(k),
        "pool_size": int(pool_size),
        "scramble_seed": int(scramble_seed),
        "policies": sorted(S3_POLICIES),
        "controlPolicy": CONTROL_POLICY,
        "coordinatePolicies": list(COORDINATE_POLICIES),
    }
    config_hash = canonical_hash(
        {
            "archive_cfg": archive_cfg,
            "horizon_cfg": horizon_cfg,
            "policy_cfgs": policy_cfgs,
            "run_params": run_params,
        }
    )

    # 2. Build the archive and take the deterministic baseline candidate pool.
    archive = build_archive(bases, archive_cfg, base_seed)
    archive_hash = archive["archiveHash"]
    pool = archive["cards"][:pool_size]
    if len(pool) < pool_size:
        raise ValueError(
            f"S3 실행: 아카이브 카드 수 {len(pool)} 가 pool_size={pool_size} "
            "보다 작습니다"
        )
    pool_hash = canonical_hash([c["cardId"] for c in pool])

    # The deterministic coordinate-scramble: permutation + scrambled pool + hash.
    permutation = scramble_permutation(pool, scramble_seed)
    scrambled_pool = scramble_coordinates(pool, scramble_seed)
    permutation_hash = canonical_hash(permutation)
    scrambled_pool_hash = canonical_hash(
        [c["cardId"] for c in scrambled_pool]
    )

    # 3. Dry run: emit the plan (incl. the scramble permutation), write nothing.
    if dry_run:
        log_ko(
            _logger,
            "S3 드라이런 요약: "
            f"policies={sorted(S3_POLICIES)}, control={CONTROL_POLICY}, "
            f"coordinatePolicies={list(COORDINATE_POLICIES)}, "
            f"n={n}, H={H}, k={k}, pool_size={pool_size}, "
            f"scramble_seed={scramble_seed}, "
            f"archiveHash={archive_hash[:12]}, poolHash={pool_hash[:12]}, "
            f"permutationHash={permutation_hash[:12]}, "
            f"configHash={config_hash[:12]} (파일 미작성)",
        )
        return {
            "dryRun": True,
            "config": run_params,
            "configHash": config_hash,
            "archiveHash": archive_hash,
            "poolHash": pool_hash,
            "scramblePermutation": permutation,
            "scramblePermutationHash": permutation_hash,
            "scrambledPoolHash": scrambled_pool_hash,
        }

    # 4. Run a baseline + scrambled seed batch per policy; record the mean
    #    full-episode shift (magnitude) and the round-0 selection-divergence rate
    #    (the clean direction signal).
    child_seeds = derive_child_seeds(base_seed, n)
    table: List[Dict[str, Any]] = []
    all_trace_hashes: List[str] = []
    all_slate_hashes: List[str] = []
    seed_batch_ids: Dict[str, str] = {}
    divergence_by_policy: Dict[str, float] = {}

    for name in sorted(S3_POLICIES):
        cls, _cfg_file = S3_POLICIES[name]
        cfg = policy_cfgs[name]
        batch_id = seed_batch_id(
            base_seed, n, cls(dict(cfg)).policy_version(), H
        )
        seed_batch_ids[name] = batch_id

        per_seed_shifts: List[float] = []
        round0_diffs = 0
        cell_baseline_hashes: List[str] = []
        cell_scrambled_hashes: List[str] = []

        for child_seed in child_seeds:
            # Baseline episode (original pool) + scrambled episode (scrambled
            # pool) for the SAME child seed -- an apples-to-apples pair.
            baseline_trace = run_episode(
                pool, cls(dict(cfg)), child_seed, H, int(k), bases
            )
            scrambled_trace = run_episode(
                scrambled_pool, cls(dict(cfg)), child_seed, H, int(k), bases
            )
            per_seed_shifts.append(
                robustness_score(
                    compute_all(baseline_trace),
                    compute_all(scrambled_trace),
                    keys=S3_METRIC_KEYS,
                )
            )
            # Round-0 selection divergence: clean of trace-hash feedback because
            # round 0 runs on an empty (scramble-invariant) trace.
            b0 = baseline_trace.rounds()[0]["selectedCardId"]
            s0 = scrambled_trace.rounds()[0]["selectedCardId"]
            if b0 != s0:
                round0_diffs += 1

            cell_baseline_hashes.append(baseline_trace.trace_hash())
            cell_scrambled_hashes.append(scrambled_trace.trace_hash())
            all_trace_hashes.append(baseline_trace.trace_hash())
            all_trace_hashes.append(scrambled_trace.trace_hash())
            all_slate_hashes.append(
                canonical_hash([r["slate"] for r in baseline_trace.rounds()])
            )
            all_slate_hashes.append(
                canonical_hash([r["slate"] for r in scrambled_trace.rounds()])
            )

        mean_shift = float(sum(per_seed_shifts) / len(per_seed_shifts))
        round0_divergence = float(round0_diffs / n) if n else 0.0
        divergence_by_policy[name] = round0_divergence

        row = {
            "policy": name,
            "policyVersion": cls(dict(cfg)).policy_version(),
            "isControl": name == CONTROL_POLICY,
            "isCoordinatePolicy": name in COORDINATE_POLICIES,
            "seedBatchId": batch_id,
            "n": n,
            "scramble_shift": mean_shift,
            "round0_selection_divergence": round0_divergence,
            "stats": {
                "scramble_shift": aggregate_values(per_seed_shifts, "scramble_shift")
            },
            "baselineTraceHashes": cell_baseline_hashes,
            "scrambledTraceHashes": cell_scrambled_hashes,
        }
        table.append(row)

        log_ko(
            _logger,
            "S3 정책 완료: "
            f"policy={name}, seedBatchId={batch_id[:12]}, "
            f"scramble_shift={mean_shift:.6f}, "
            f"round0_selection_divergence={round0_divergence:.4f}, "
            f"isCoordinatePolicy={name in COORDINATE_POLICIES}",
        )

    # Direction invariant (asserted on the clean round-0 signal): each
    # coordinate policy's round-0 selection-divergence >= the control's.
    control_divergence = divergence_by_policy.get(CONTROL_POLICY, 0.0)
    per_coordinate_direction = {
        name: bool(divergence_by_policy.get(name, 0.0) >= control_divergence)
        for name in COORDINATE_POLICIES
    }
    coordinate_policies_shift_at_least_control = all(
        per_coordinate_direction.values()
    )

    # An experiment-level seedBatchId summarizing the whole comparison.
    exp_seed_batch_id = canonical_hash(
        {
            "base_seed": base_seed,
            "n": n,
            "H": H,
            "k": k,
            "scramble_seed": scramble_seed,
            "policies": sorted(S3_POLICIES),
            "perPolicySeedBatchIds": seed_batch_ids,
        }
    )

    slate_hash = canonical_hash(all_slate_hashes)
    trace_hash = canonical_hash(all_trace_hashes)

    # 5. Results body + output hash.
    results_body = {
        "config": run_params,
        "metricKeys": list(S3_METRIC_KEYS),
        "table": table,
        "controlDivergence": control_divergence,
        "perCoordinateDirection": per_coordinate_direction,
        "coordinatePoliciesShiftAtLeastControl": (
            coordinate_policies_shift_at_least_control
        ),
    }
    output_hash = canonical_hash(results_body)

    # 6. Report (the scramble permutation is logged into the report).
    report: Dict[str, Any] = {
        "schema": S3_SCHEMA,
        "schemaVersion": S3_SCHEMA_VERSION,
        "experiment": "S3_COORDINATE_SCRAMBLE",
        "phaseNote": (
            "S3 applies a deterministic seeded permutation of the "
            "coordinateContribution vectors across pool cards (structure "
            "preserved, card->coordinate semantics broken). It reports the mean "
            "full-episode baseline-vs-scrambled metric shift (robustness_score) "
            "as the magnitude, and the round-0 selection-divergence rate as the "
            "clean direction signal. Coordinate-driven policies (TRACE_GREEDY, "
            "TRACE_LIN_UCB) are expected to diverge at least as much as the "
            "RANDOM control at round 0 (direction, not magnitude); the asserted "
            "invariant is on that round-0 signal. System-level metrics over a "
            "controlled testbed; no real-world generalization claim."
        ),
        "config": run_params,
        "metricKeys": list(S3_METRIC_KEYS),
        "scrambleShiftDirection": (
            "scramble_shift is the mean full-episode baseline-vs-scrambled "
            "sensitivity magnitude (robustness_score units): " + ROBUSTNESS_DIRECTION
        ),
        "seedBatchId": exp_seed_batch_id,
        "perPolicySeedBatchIds": seed_batch_ids,
        "scramblePermutation": permutation,
        "scramblePermutationHash": permutation_hash,
        "scrambledPoolHash": scrambled_pool_hash,
        "controlPolicy": CONTROL_POLICY,
        "controlDivergence": control_divergence,
        "perCoordinateDirection": per_coordinate_direction,
        "coordinatePoliciesShiftAtLeastControl": (
            coordinate_policies_shift_at_least_control
        ),
        "configHash": config_hash,
        "archiveHash": archive_hash,
        "poolHash": pool_hash,
        "slateHash": slate_hash,
        "traceHash": trace_hash,
        "outputHash": output_hash,
        "table": table,
    }
    report_hash = canonical_hash(report)
    report["reportHash"] = report_hash

    # 7. Reproducibility pack.
    pack = ReproducibilityPack(
        configHash=config_hash,
        commitHash=_git_commit_hash(),
        archiveHash=archive_hash,
        poolHash=pool_hash,
        slateHash=slate_hash,
        traceHash=trace_hash,
        outputHash=output_hash,
        reportHash=report_hash,
        seedBatchId=exp_seed_batch_id,
    )
    report["reproducibilityPack"] = pack.to_dict()
    report["packHash"] = pack.pack_hash()

    # 8. Write the report json.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        _REPORTS_DIR / f"s3_coordinate_scramble_{exp_seed_batch_id[:12]}.json"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)

    log_ko(
        _logger,
        "S3 보고서 작성 완료: "
        f"path={out_path}, reportHash={report_hash[:12]}, "
        f"seedBatchId={exp_seed_batch_id[:12]}, rows={len(table)}, "
        f"coordinatePoliciesShiftAtLeastControl="
        f"{coordinate_policies_shift_at_least_control}",
    )
    return report


def main() -> None:
    """CLI entry point. Parse args, run S3, print a Korean summary."""
    parser = argparse.ArgumentParser(
        prog="python -m echo_bench.experiments.s3_coordinate_scramble",
        description="ECHO-Bench S3 coordinate-scramble runner.",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed")
    parser.add_argument("--n", type=int, default=10, help="child seeds per policy")
    parser.add_argument(
        "--H", type=int, default=None, help="horizon (default: config default)"
    )
    parser.add_argument("--k", type=int, default=4, help="slate size")
    parser.add_argument(
        "--pool-size", type=int, default=64, help="candidate pool size"
    )
    parser.add_argument(
        "--scramble-seed", type=int, default=7, help="coordinate-scramble seed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + plan only; write no files",
    )
    args = parser.parse_args()

    result = run_s3_coordinate_scramble(
        base_seed=args.seed,
        n=args.n,
        H=args.H,
        k=args.k,
        pool_size=args.pool_size,
        scramble_seed=args.scramble_seed,
        dry_run=args.dry_run,
    )

    if result.get("dryRun"):
        log_ko(
            _logger,
            "S3 드라이런 완료: "
            f"policies={result['config']['policies']}, "
            f"archiveHash={result['archiveHash'][:12]}, "
            f"poolHash={result['poolHash'][:12]}, "
            f"permutationHash={result['scramblePermutationHash'][:12]}, "
            f"configHash={result['configHash'][:12]} (파일 미작성)",
        )
    else:
        log_ko(
            _logger,
            "S3 실행 완료: "
            f"seedBatchId={result['seedBatchId'][:12]}, "
            f"reportHash={result['reportHash'][:12]}, "
            f"rows={len(result['table'])}, "
            f"coordinatePoliciesShiftAtLeastControl="
            f"{result['coordinatePoliciesShiftAtLeastControl']}",
        )


if __name__ == "__main__":
    main()
