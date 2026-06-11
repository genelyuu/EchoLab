"""Tests for the E-011 supplement seed-batch scale-up + S2 confound fix.

Covers:
- Each supplement runner's default ``n`` is at least ``MIN_SUFFICIENT_N`` so a
  default run yields usable bootstrap confidence intervals.
- With ``n >= MIN_SUFFICIENT_N`` every supplement report row carries a ``stats``
  block whose aggregates are ``sufficient_n=True`` / ``ci_method="bootstrap"``.
- S2 holds the candidate-pool size constant across ablation arms (capped to the
  smallest arm's archive), so basis importance is not confounded with pool size.
"""

from __future__ import annotations

import inspect

from echo_bench.experiments.s1_k_sensitivity import run_s1_k_sensitivity
from echo_bench.experiments.s2_basis_ablation import run_s2_basis_ablation
from echo_bench.experiments.s3_coordinate_scramble import run_s3_coordinate_scramble
from echo_bench.experiments.s4_salience_audit import run_s4_salience_audit
from echo_bench.metrics.aggregate import MIN_SUFFICIENT_N

_RUNNERS = (
    run_s1_k_sensitivity,
    run_s2_basis_ablation,
    run_s3_coordinate_scramble,
    run_s4_salience_audit,
)


def test_default_n_is_at_least_min_sufficient():
    # Scaling exists so reports get usable CIs; the default must clear the
    # bootstrap minimum (n >= 3), not the old n=2.
    for runner in _RUNNERS:
        default_n = inspect.signature(runner).parameters["n"].default
        assert default_n >= MIN_SUFFICIENT_N, runner.__name__


def _has_bootstrap(agg: dict) -> bool:
    return agg.get("sufficient_n") is True and agg.get("ci_method") == "bootstrap"


def test_s1_rows_carry_sufficient_bootstrap_stats():
    report = run_s1_k_sensitivity(base_seed=42, n=3, H=4, pool_size=16, dry_run=False)
    for row in report["table"]:
        assert "stats" in row
        assert _has_bootstrap(row["stats"]["coordinate_coverage"])


def test_s2_rows_carry_sufficient_bootstrap_stats():
    report = run_s2_basis_ablation(
        base_seed=42, n=3, H=4, k=4, pool_size=16, dry_run=False
    )
    for row in report["table"]:
        assert "stats" in row
        assert _has_bootstrap(row["stats"]["coordinate_coverage"])


def test_s3_rows_carry_sufficient_bootstrap_stats():
    report = run_s3_coordinate_scramble(
        base_seed=42, n=3, H=4, k=4, pool_size=16, dry_run=False
    )
    for row in report["table"]:
        assert "stats" in row
        assert _has_bootstrap(row["stats"]["scramble_shift"])


def test_s4_rows_carry_sufficient_bootstrap_stats():
    report = run_s4_salience_audit(
        base_seed=42, n=3, H=4, k=4, pool_size=16, dry_run=False
    )
    for row in report["table"]:
        assert "stats" in row
        assert _has_bootstrap(row["stats"]["salience_outlier_rate"])


def test_s2_pool_size_constant_across_arms_when_capped():
    # Request a pool larger than any single ablation arm's archive: the runner
    # must cap to the smallest arm's size and apply that SAME size to every arm,
    # so the basis-ablation signal is not confounded with pool size.
    report = run_s2_basis_ablation(
        base_seed=42, n=2, H=4, k=4, pool_size=10_000, dry_run=False
    )
    pool_sizes = {row["poolSize"] for row in report["table"]}
    assert len(pool_sizes) == 1  # identical across every arm
    effective = report["config"]["effectivePoolSize"]
    assert effective == next(iter(pool_sizes))
    assert effective < 10_000  # genuinely capped to the smallest arm
