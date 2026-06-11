"""C-014: Oracle / reference display-name layer for ECHO-Bench.

Prevents oracle policy names from being misread as global upper bounds in paper
artifacts. Each oracle maximises only its own objective; the name ``ORACLE_*``
invites the wrong reading. This module provides a display-name layer so reports
label these policies as objective-specific references.

**Code aliases (internal policy names, config keys, hash inputs) are NEVER
changed by this module.** Only the human-facing ``displayName`` field in report
rows is affected.

Usage
-----
::

    from echo_bench.policies.display_names import display_name, REFERENCE_NOTE

    row["displayName"] = display_name(row["policy"])
    if is_reference_policy(row["policy"]):
        row["referenceNote"] = REFERENCE_NOTE

All identifiers stay English; no runtime log messages are emitted from this
module (it is a pure data/utility module).
"""

from __future__ import annotations

__all__ = [
    "DISPLAY_NAMES",
    "REFERENCE_NOTE",
    "display_name",
    "is_reference_policy",
]

# Maps internal policy names to their paper-artifact display names.
# Keys are the code aliases used in E2_POLICIES, configs, and hashes.
# Values are the display labels for report rows and figures.
DISPLAY_NAMES: dict[str, str] = {
    "ORACLE_COVERAGE": "COVERAGE_GREEDY_REFERENCE",
    "ORACLE_DIVERSITY": "DIVERSITY_GREEDY_REFERENCE",
    "ORACLE_STRATEGY": "STRATEGY_OBJECTIVE_REFERENCE",
}

# Exact note string attached to every oracle row's ``referenceNote`` field.
REFERENCE_NOTE: str = "objective-specific reference, not global optimum"


def display_name(policy_name: str) -> str:
    """Return the display name for *policy_name*.

    For oracle / reference policies this is the mapped label (e.g.
    ``"COVERAGE_GREEDY_REFERENCE"``). For all other policies the input is
    returned unchanged so non-oracle rows always carry their internal name as
    the display name.

    Args:
        policy_name: Internal policy code alias (e.g. ``"ORACLE_COVERAGE"``).

    Returns:
        Mapped display name for oracle policies; the input string otherwise.
    """
    return DISPLAY_NAMES.get(policy_name, policy_name)


def is_reference_policy(policy_name: str) -> bool:
    """Return ``True`` iff *policy_name* is one of the oracle reference policies.

    Args:
        policy_name: Internal policy code alias.

    Returns:
        ``True`` for ``ORACLE_COVERAGE``, ``ORACLE_DIVERSITY``, and
        ``ORACLE_STRATEGY``; ``False`` for every other name.
    """
    return policy_name in DISPLAY_NAMES
