"""Tests for the common policy interface (Task C-001, base.py).

These tests exercise the interface contract only — RANDOM and the other
concrete policies arrive in later waves. A tiny in-test ``_NullPolicy`` supplies
a minimal concrete :meth:`Policy.select` so the abstract base can be exercised;
``base.py`` itself exports only :class:`Policy`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from echo_bench.policies.base import Policy


class _NullPolicy(Policy):
    """Minimal concrete policy for interface testing only.

    Selects the first ``k`` (default 4) ``cardId`` values from the pool. Uses no
    user/persona/emotion/preference state — it reads only ``cardId`` and the
    optional ``k`` config key. Lives in the test module, never in ``base.py``.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any],
    ) -> List[Any]:
        k = int(config.get("k", self.config.get("k", 4)))
        return [card["cardId"] for card in pool[:k]]


def _make_pool(n: int = 6) -> List[Dict[str, Any]]:
    return [
        {
            "cardId": f"c{i}",
            "basis": f"B{(i % 4) + 1}",
            "complexityBand": "low",
            "salienceScore": 0.5,
            "coordinateContribution": [0.0, 0.0],
        }
        for i in range(n)
    ]


def test_minimal_subclass_can_be_instantiated_and_select() -> None:
    """A subclass implementing ``select`` can be instantiated and used."""
    policy = _NullPolicy()
    pool = _make_pool(6)
    slate = policy.select(pool, trace=None, seed=0, config={"k": 4})
    assert slate == ["c0", "c1", "c2", "c3"]


def test_policy_version_is_deterministic() -> None:
    """``policy_version`` is stable across calls and instances for same config."""
    p1 = _NullPolicy({"k": 4})
    p2 = _NullPolicy({"k": 4})
    assert p1.policy_version() == p1.policy_version()
    assert p1.policy_version() == p2.policy_version()


def test_policy_version_changes_with_config() -> None:
    """Changing the config changes the recorded policy identity."""
    p_a = _NullPolicy({"k": 4})
    p_b = _NullPolicy({"k": 6})
    assert p_a.policy_version() != p_b.policy_version()


def test_log_score_components_records_onto_attribute() -> None:
    """``log_score_components`` stores per-card components on the buffer."""
    policy = _NullPolicy()
    assert policy.last_score_components == {}
    components = {
        "c0": {"base": 1.0, "bonus": 0.2},
        "c1": {"base": 0.5, "bonus": 0.1},
    }
    policy.log_score_components(components)
    assert policy.last_score_components == components
    # A second call replaces the buffer with the most recent components.
    policy.log_score_components({"c2": {"base": 0.0}})
    assert policy.last_score_components == {"c2": {"base": 0.0}}


def test_policy_cannot_be_instantiated_directly() -> None:
    """``Policy`` is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Policy()  # type: ignore[abstract]
