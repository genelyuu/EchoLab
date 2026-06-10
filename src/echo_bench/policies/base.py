"""Common policy interface for ECHO-Bench (Task C-001, base.py).

This module defines :class:`Policy`, the single abstract interface every
ECHO-Bench policy (RANDOM, FIXED_*, TRACE_*, PSEUDO_USER_MODEL, ORACLE_STRATEGY)
implements. It deliberately contains *only* the interface: the concrete RANDOM
policy and all other policies are added in later waves and must implement this
contract without altering the signatures here.

Contract (stable — downstream policies and the round runner depend on it):

- ``select(self, pool, trace, seed, config) -> list`` returns a slate as a list
  of ``cardId`` values chosen from ``pool``. ``pool`` is a list of card dicts
  (each carrying at least ``cardId``, ``basis``, ``complexityBand``,
  ``salienceScore``, ``coordinateContribution``); ``trace`` is an
  ``echo_bench.env.trace_state.TraceState``; ``seed`` is an int; ``config`` is a
  dict. The selection must be a pure function of its inputs (no global RNG, no
  wall-clock) so runs replay deterministically.
- ``policy_version(self) -> str`` returns a stable string identifying the policy
  class together with its config, used in the reproducibility hash chain.
- ``log_score_components(self, components) -> None`` records the most recent
  per-card ``scoreComponents`` onto ``self.last_score_components`` so the round
  runner can log them.

Guardrail (non-negotiable): no method here reads or accepts any user, persona,
emotion, preference, demographic, or free-text field. The interface is defined
purely over the observable candidate pool, the observable trace, a seed, and a
config dict.

All identifiers stay English; any runtime log messages remain Korean per the
project logging convention.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List

from echo_bench.utils.hash import canonical_hash

__all__ = ["Policy"]


class Policy(abc.ABC):
    """Abstract base class for every ECHO-Bench slate-selection policy.

    Subclasses implement :meth:`select` to choose a constraint-satisfying slate
    of ``cardId`` values from a candidate pool, conditioned only on observable
    inputs (the pool, the trace, a seed, and a config). Latent user/persona/
    emotion/preference state must never enter a policy through this interface.
    """

    #: Most recent per-card ``scoreComponents``, keyed by ``cardId``. Populated
    #: by :meth:`log_score_components`; consumed by the round runner for logging.
    last_score_components: Dict[Any, Dict[str, Any]]

    def __init__(self) -> None:
        # Initialise the score-component buffer so the attribute always exists,
        # even before the first :meth:`select` call.
        self.last_score_components = {}

    @abc.abstractmethod
    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any],
    ) -> List[Any]:
        """Return a slate as a list of ``cardId`` values chosen from ``pool``.

        Parameters
        ----------
        pool:
            List of candidate card dicts. Each card carries at least
            ``cardId``, ``basis``, ``complexityBand``, ``salienceScore``, and
            ``coordinateContribution``. No card field describing a user/persona/
            emotion/preference exists or may be read.
        trace:
            An ``echo_bench.env.trace_state.TraceState`` holding the observable
            interaction history. Only observable trace fields may be used.
        seed:
            Integer seed for any policy-local RNG. Implementations must use a
            local, seeded RNG — never the global RNG or wall-clock — so that
            selection replays deterministically.
        config:
            Policy configuration dict (weights, schedule, exploration
            coefficients, etc.).

        Returns
        -------
        list
            The chosen slate, expressed as a list of ``cardId`` values drawn
            from ``pool``.

        Notes
        -----
        Implementations must be a pure function of ``(pool, trace, seed,
        config)`` and must satisfy the active ``k``-constraint (enforced by the
        environment). This abstract method defines the contract only.
        """
        raise NotImplementedError

    def policy_version(self) -> str:
        """Return a stable identifier for this policy and its config.

        The default implementation hashes the policy class name together with
        the policy's ``config`` (an empty dict if the subclass has not set one),
        producing a deterministic string suitable for the reproducibility hash
        chain. Changing the config changes the returned version. Subclasses with
        additional version-relevant state may override this.
        """
        return canonical_hash(
            {
                "policy": self.__class__.__name__,
                "config": getattr(self, "config", {}),
            }
        )

    def log_score_components(self, components: Dict[Any, Dict[str, Any]]) -> None:
        """Record the most recent per-card ``scoreComponents``.

        Stores ``components`` (a mapping of ``cardId`` to its score-component
        dict) on :attr:`last_score_components` so the round runner can log the
        per-card scoring breakdown for the most recent :meth:`select` call.
        Deterministic and side-effect free beyond setting the attribute.
        """
        self.last_score_components = dict(components)
