"""Tests for E-012 inline replay validation on E1 and E2.

E1/E2 previously asserted determinism-by-construction without an inline replay
result. These tests verify that a normal (non-dry-run) E1/E2 run now carries a
``replayAudit`` section reporting ``replayable=True`` / ``first_divergent=None``,
that the audit is NOT part of the hashed report body (reportHash unchanged), and
that the re-entrant guard (``replay_validate=False``) omits the audit so there is
no unbounded recursion.
"""

from __future__ import annotations

from echo_bench.experiments.e1_horizon import run_e1_horizon
from echo_bench.experiments.e2_policy import run_e2_policy

_E1_KW = dict(base_seed=42, n=2, k=4, pool_size=16)
_E2_KW = dict(base_seed=42, H=4, k=4, pool_size=16, n=2)


def test_e1_report_carries_replayable_audit():
    report = run_e1_horizon(dry_run=False, **_E1_KW)
    audit = report["replayAudit"]
    assert audit["replayable"] is True
    assert audit["first_divergent"] is None
    assert audit["runFn"].endswith("run_e1_horizon")


def test_e2_report_carries_replayable_audit():
    report = run_e2_policy(dry_run=False, **_E2_KW)
    audit = report["replayAudit"]
    assert audit["replayable"] is True
    assert audit["first_divergent"] is None
    assert audit["runFn"].endswith("run_e2_policy")


def test_replay_validate_false_omits_audit_and_keeps_reporthash():
    # The re-entrant guard run produces no audit (prevents recursion), and the
    # reportHash is identical with or without the audit (audit is not hashed).
    full = run_e1_horizon(dry_run=False, replay_validate=True, **_E1_KW)
    guarded = run_e1_horizon(dry_run=False, replay_validate=False, **_E1_KW)
    assert "replayAudit" not in guarded
    assert full["reportHash"] == guarded["reportHash"]


def test_e2_replay_validate_false_omits_audit_and_keeps_reporthash():
    full = run_e2_policy(dry_run=False, replay_validate=True, **_E2_KW)
    guarded = run_e2_policy(dry_run=False, replay_validate=False, **_E2_KW)
    assert "replayAudit" not in guarded
    assert full["reportHash"] == guarded["reportHash"]
