"""Tests for the C-014 oracle/reference display-name layer.

Covers:
- DISPLAY_NAMES mapping correctness for all three oracle policy names.
- REFERENCE_NOTE exact string.
- display_name() passthrough for unknown / non-oracle names.
- is_reference_policy() truth-table.
- Generated E2 report: oracle rows carry ``displayName`` and ``referenceNote``;
  non-oracle rows carry ``displayName`` equal to the policy name and no
  ``referenceNote``; the top-level ``oraclePolicyDisplayName`` and
  ``oracleNote`` are present and correct.
"""

from __future__ import annotations

import pytest

from echo_bench.policies.display_names import (
    DISPLAY_NAMES,
    REFERENCE_NOTE,
    display_name,
    is_reference_policy,
)


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------


def test_display_names_has_all_three_oracle_keys():
    assert set(DISPLAY_NAMES) == {
        "ORACLE_COVERAGE",
        "ORACLE_DIVERSITY",
        "ORACLE_STRATEGY",
    }


def test_display_names_values():
    assert DISPLAY_NAMES["ORACLE_COVERAGE"] == "COVERAGE_GREEDY_REFERENCE"
    assert DISPLAY_NAMES["ORACLE_DIVERSITY"] == "DIVERSITY_GREEDY_REFERENCE"
    assert DISPLAY_NAMES["ORACLE_STRATEGY"] == "STRATEGY_OBJECTIVE_REFERENCE"


def test_reference_note_exact_string():
    assert REFERENCE_NOTE == "objective-specific reference, not global optimum"


# ---------------------------------------------------------------------------
# display_name() tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_name,expected",
    [
        ("ORACLE_COVERAGE", "COVERAGE_GREEDY_REFERENCE"),
        ("ORACLE_DIVERSITY", "DIVERSITY_GREEDY_REFERENCE"),
        ("ORACLE_STRATEGY", "STRATEGY_OBJECTIVE_REFERENCE"),
    ],
)
def test_display_name_mapped_oracle(policy_name, expected):
    assert display_name(policy_name) == expected


@pytest.mark.parametrize(
    "policy_name",
    [
        "RANDOM",
        "FIXED_LOW_TO_HIGH",
        "FIXED_BALANCED",
        "TRACE_GREEDY",
        "TRACE_LIN_UCB",
        "PSEUDO_USER_MODEL",
        "PSEUDO_USER_MODEL_DIVERSITY_REG",
        "PSEUDO_USER_MODEL_SESSION_EMBEDDING",
        "SOME_UNKNOWN_POLICY",
        "",
    ],
)
def test_display_name_passthrough_for_non_oracle(policy_name):
    assert display_name(policy_name) == policy_name


# ---------------------------------------------------------------------------
# is_reference_policy() tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_name",
    ["ORACLE_COVERAGE", "ORACLE_DIVERSITY", "ORACLE_STRATEGY"],
)
def test_is_reference_policy_true_for_oracle(policy_name):
    assert is_reference_policy(policy_name) is True


@pytest.mark.parametrize(
    "policy_name",
    ["RANDOM", "TRACE_GREEDY", "PSEUDO_USER_MODEL", "SOME_UNKNOWN_POLICY", ""],
)
def test_is_reference_policy_false_for_non_oracle(policy_name):
    assert is_reference_policy(policy_name) is False


# ---------------------------------------------------------------------------
# E2 report integration tests
# ---------------------------------------------------------------------------


def _run_e2_small():
    """Run E2 with the smallest valid parameters (same as other E2 tests)."""
    from echo_bench.experiments.e2_policy import run_e2_policy

    return run_e2_policy(base_seed=42, H=4, k=4, pool_size=16, n=1,
                         replay_validate=False)


def test_e2_oracle_rows_have_display_name_and_reference_note():
    report = _run_e2_small()
    oracle_names = {"ORACLE_COVERAGE", "ORACLE_DIVERSITY", "ORACLE_STRATEGY"}
    oracle_rows = [r for r in report["table"] if r["policy"] in oracle_names]
    assert len(oracle_rows) == 3, "Expected three oracle rows"

    for row in oracle_rows:
        assert "displayName" in row, (
            f"Oracle row {row['policy']!r} is missing 'displayName'"
        )
        assert "referenceNote" in row, (
            f"Oracle row {row['policy']!r} is missing 'referenceNote'"
        )
        # displayName must equal the mapped value, not the internal name.
        assert row["displayName"] == display_name(row["policy"]), (
            f"displayName mismatch for {row['policy']!r}"
        )
        assert row["referenceNote"] == REFERENCE_NOTE, (
            f"referenceNote mismatch for {row['policy']!r}"
        )


def test_e2_non_oracle_rows_display_name_equals_policy_no_reference_note():
    report = _run_e2_small()
    oracle_names = {"ORACLE_COVERAGE", "ORACLE_DIVERSITY", "ORACLE_STRATEGY"}
    non_oracle_rows = [r for r in report["table"] if r["policy"] not in oracle_names]
    assert non_oracle_rows, "Expected non-oracle rows in E2 table"

    for row in non_oracle_rows:
        assert "displayName" in row, (
            f"Non-oracle row {row['policy']!r} is missing 'displayName'"
        )
        assert row["displayName"] == row["policy"], (
            f"Non-oracle displayName must equal policy name for {row['policy']!r}"
        )
        assert "referenceNote" not in row, (
            f"Non-oracle row {row['policy']!r} must not have 'referenceNote'"
        )


def test_e2_top_level_oracle_display_fields():
    report = _run_e2_small()
    assert "oraclePolicyDisplayName" in report, (
        "Report top-level must have 'oraclePolicyDisplayName'"
    )
    assert "oracleNote" in report, (
        "Report top-level must have 'oracleNote'"
    )
    assert report["oraclePolicyDisplayName"] == "STRATEGY_OBJECTIVE_REFERENCE"
    assert report["oracleNote"] == REFERENCE_NOTE


def test_e2_report_reference_note_exact_string_present():
    """The REFERENCE_NOTE string must appear verbatim somewhere in oracle rows."""
    report = _run_e2_small()
    oracle_names = {"ORACLE_COVERAGE", "ORACLE_DIVERSITY", "ORACLE_STRATEGY"}
    notes = [
        r.get("referenceNote", "")
        for r in report["table"]
        if r["policy"] in oracle_names
    ]
    assert all(n == REFERENCE_NOTE for n in notes), (
        f"Not all oracle rows have the exact REFERENCE_NOTE. Got: {notes}"
    )
