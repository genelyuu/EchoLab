"""RANDOM slate-selection policy for ECHO-Bench (Task C-001, RANDOM portion).

The RANDOM policy chooses a constraint-satisfying slate of size ``k`` uniformly
at random from the candidate pool, using a *seeded local* RNG so the selection
is a pure function of ``(poolHash, traceHash, seed, policyVersion)`` and replays
deterministically. It never touches the global RNG or the wall-clock.

Selection procedure
--------------------
1. Derive a single integer seed from a canonical hash of
   ``{poolHash, traceHash, seed, policyVersion}`` and build a
   ``random.Random`` from it.
2. Repeatedly sample ``k`` distinct cards from the pool and validate them with
   :func:`echo_bench.env.constraints.check_slate`. Accept the first sample that
   satisfies the active ``k``-constraint.
3. Bound the search by ``max_attempts`` (default 200); if no satisfying slate is
   found, raise :class:`ValueError` with a Korean message.

Per-card ``scoreComponents`` are logged for the accepted slate. For RANDOM the
score is a uniform random tiebreak value drawn from the same local RNG — there
is no trace-conditioned scoring.

Guardrail: no user/persona/emotion/preference/demographic/free-text field is
read or produced anywhere. The policy reads only the observable pool, the
observable trace hash, a seed, and its config.

All identifiers stay English; runtime log messages are Korean per the project
logging convention.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.env.constraints import check_slate
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = ["RandomPolicy"]

_logger = get_logger(__name__)

# Default cap on rejection-sampling attempts before failing closed.
DEFAULT_MAX_ATTEMPTS = 200

# Location of the optional basis config; passed to check_slate when present.
_BASES_CFG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "basis" / "bases.yaml"
)


def _load_bases_cfg() -> dict:
    """Load ``configs/basis/bases.yaml`` if present, else return ``{}``.

    ``check_slate`` accepts a dict (its k-rule mapping is the documented spec and
    is not overridden by this config), so an empty dict is a valid fallback.
    """
    try:
        with open(_BASES_CFG_PATH, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}


class RandomPolicy(Policy):
    """Uniform-random, constraint-satisfying slate selection.

    Stores ``config`` (so :meth:`Policy.policy_version` reflects it). ``config``
    carries at least ``k`` and an optional ``max_attempts``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate of ``cardId`` values.

        Samples ``k`` distinct cards uniformly at random (seeded local RNG) and
        accepts the first sample passing :func:`check_slate`. Bounded by
        ``max_attempts``; raises :class:`ValueError` (Korean message) if no
        satisfying slate is found.
        """
        effective = config if config is not None else self.config
        k = int(effective["k"])
        max_attempts = int(effective.get("max_attempts", DEFAULT_MAX_ATTEMPTS))

        if k > len(pool):
            raise ValueError(
                f"RANDOM 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        # Seed a local RNG purely from observable, hashable inputs so identical
        # (poolHash, traceHash, seed, policyVersion) always replay identically.
        pool_hash = canonical_hash([c["cardId"] for c in pool])
        seed_material = canonical_hash(
            {
                "poolHash": pool_hash,
                "traceHash": trace.trace_hash(),
                "seed": seed,
                "policyVersion": self.policy_version(),
            }
        )
        rng = random.Random(int(seed_material, 16))

        bases_cfg = _load_bases_cfg()

        for attempt in range(1, max_attempts + 1):
            sampled = rng.sample(pool, k)
            ok, reason, _slot_permutation = check_slate(
                sampled, k, bases_cfg, seed
            )
            if ok:
                # RANDOM scoring: a uniform random tiebreak value per card.
                components = {
                    card["cardId"]: {"randomScore": rng.random()}
                    for card in sampled
                }
                self.log_score_components(components)
                log_ko(
                    _logger,
                    f"RANDOM 슬레이트 확정: k={k}, 시도 {attempt}/{max_attempts}, "
                    f"poolHash={pool_hash}",
                )
                return [card["cardId"] for card in sampled]
            log_ko(
                _logger,
                f"RANDOM 샘플 거부 (attempt={attempt}, reason={reason}), 재시도",
            )

        raise ValueError(
            f"RANDOM 정책: max_attempts={max_attempts} 회 시도 내에 "
            f"k={k} 제약을 만족하는 슬레이트를 찾지 못했습니다"
        )
