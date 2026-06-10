"""FIXED_BALANCED slate-selection policy for ECHO-Bench (Task C-003).

A *non-adaptive* schedule that targets a configured balanced mix over bases and
complexity bands in **every** slate, independent of the trace. The policy reads
no trace content at all (not even the round index) — balance targets live in
config — so every slate is a pure function of (pool, seed, config). It always
satisfies the active ``k``-constraint, including the >=3-distinct-bases rule for
``k==4``.

Selection procedure
--------------------
1. Resolve target proportions over bases (``basis_targets``) and bands
   (``band_targets``) from config; absent targets default to a uniform mix over
   the bases / bands actually present in the pool.
2. Convert proportions into integer per-basis / per-band quotas summing to ``k``
   (largest-remainder rounding), guaranteeing at least the constraint's required
   number of distinct bases for ``k``.
3. Greedily fill the slate honoring basis quotas first (constraint-critical),
   breaking ties toward the band whose quota is least satisfied, with a seeded
   local RNG tiebreak. Validate with :func:`check_slate`; fall back to a
   diversity-first fill if the quota fill misses the constraint.

Per-card ``scoreComponents`` (chosen basis/band + quota fit) are logged.

Guardrail: NON-ADAPTIVE. No latent user/persona/emotion/preference field and no
trace content is read.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.cards.metrics import COMPLEXITY_BANDS
from echo_bench.env.constraints import check_slate, required_distinct_bases
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = ["FixedBalancedPolicy"]

_logger = get_logger(__name__)

BAND_ORDER: tuple[str, ...] = tuple(name for name, _lo, _hi in COMPLEXITY_BANDS)

_BASES_CFG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "basis" / "bases.yaml"
)


def _load_bases_cfg() -> dict:
    """Load ``configs/basis/bases.yaml`` if present, else return ``{}``."""
    try:
        with open(_BASES_CFG_PATH, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}


def _largest_remainder(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Apportion ``total`` integer units across keys by their weights.

    Largest-remainder (Hamilton) method: floor each share, then hand the leftover
    units to the largest fractional remainders (key-sorted for determinism).
    """
    keys = sorted(weights)
    wsum = sum(weights[k] for k in keys)
    if wsum <= 0 or total <= 0:
        return {k: 0 for k in keys}
    raw = {k: weights[k] / wsum * total for k in keys}
    floors = {k: int(raw[k]) for k in keys}
    assigned = sum(floors.values())
    leftover = total - assigned
    # Distribute leftovers by descending remainder, then ascending key.
    order = sorted(keys, key=lambda k: (-(raw[k] - floors[k]), k))
    for i in range(leftover):
        floors[order[i % len(order)]] += 1
    return floors


class FixedBalancedPolicy(Policy):
    """Non-adaptive balanced basis/band coverage policy.

    ``config`` carries at least ``k``; optional ``basis_targets`` and
    ``band_targets`` are proportion maps. Independent of the trace.
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
        """Return a constraint-satisfying slate with a balanced basis/band mix."""
        effective = config if config is not None else self.config
        k = int(effective["k"])

        if k > len(pool):
            raise ValueError(
                f"FIXED_BALANCED 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        # Seed material excludes any trace field: fully non-adaptive.
        pool_hash = canonical_hash([c["cardId"] for c in pool])
        seed_material = canonical_hash(
            {
                "poolHash": pool_hash,
                "seed": seed,
                "policyVersion": self.policy_version(),
            }
        )
        rng = random.Random(int(seed_material, 16))

        pool_bases = sorted({c["basis"] for c in pool})
        pool_bands = [b for b in BAND_ORDER if any(
            c["complexityBand"] == b for c in pool)] or BAND_ORDER

        basis_targets = effective.get("basis_targets") or {
            b: 1.0 for b in pool_bases
        }
        band_targets = effective.get("band_targets") or {
            b: 1.0 for b in pool_bands
        }
        # Restrict targets to what the pool can actually supply.
        basis_targets = {
            b: float(w) for b, w in basis_targets.items() if b in pool_bases
        } or {b: 1.0 for b in pool_bases}
        band_targets = {
            b: float(w) for b, w in band_targets.items()
        } or {b: 1.0 for b in pool_bands}

        basis_quota = _largest_remainder(basis_targets, k)
        band_quota = _largest_remainder(band_targets, k)

        min_required, _prefer_four, _ns = required_distinct_bases(k)
        # Ensure the basis quota spreads across at least the required # of bases.
        basis_quota = self._enforce_basis_spread(
            basis_quota, pool_bases, k, min_required
        )

        slate = self._fill(
            pool, k, basis_quota, band_quota, rng, _load_bases_cfg(), seed
        )
        if slate is None:
            raise ValueError(
                f"FIXED_BALANCED 정책: k={k} 제약을 만족하는 균형 슬레이트를 "
                f"구성하지 못했습니다 (poolHash={pool_hash})"
            )

        components = {
            c["cardId"]: {
                "chosenBasis": c["basis"],
                "chosenBand": c["complexityBand"],
                "basisQuotaTarget": basis_quota.get(c["basis"], 0),
                "bandQuotaTarget": band_quota.get(c["complexityBand"], 0),
            }
            for c in slate
        }
        self.log_score_components(components)
        log_ko(
            _logger,
            f"FIXED_BALANCED 슬레이트 확정: k={k}, basisQuota={basis_quota}, "
            f"bandQuota={band_quota}, poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    @staticmethod
    def _enforce_basis_spread(
        quota: Dict[str, int],
        pool_bases: List[str],
        k: int,
        min_required: int,
    ) -> Dict[str, int]:
        """Adjust quota so at least ``min_required`` bases get >=1 unit."""
        available = min(min_required, len(pool_bases), k)
        positive = [b for b in pool_bases if quota.get(b, 0) > 0]
        if len(positive) >= available:
            return quota
        quota = dict(quota)
        # Promote zero-quota bases by borrowing from the largest holders.
        zeros = [b for b in pool_bases if quota.get(b, 0) == 0]
        need = available - len(positive)
        for b in zeros[:need]:
            donor = max(quota, key=lambda x: quota[x])
            if quota[donor] <= 1:
                break
            quota[donor] -= 1
            quota[b] = quota.get(b, 0) + 1
        return quota

    @staticmethod
    def _fill(
        pool: List[Dict[str, Any]],
        k: int,
        basis_quota: Dict[str, int],
        band_quota: Dict[str, int],
        rng: random.Random,
        bases_cfg: dict,
        seed: int,
    ) -> List[Dict[str, Any]] | None:
        """Greedily fill k cards honoring basis quotas, then band balance."""
        # Stable per-card tiebreak so selection replays deterministically.
        tiebreak = {c["cardId"]: rng.random() for c in pool}

        remaining_basis = dict(basis_quota)
        remaining_band = dict(band_quota)
        chosen: List[Dict[str, Any]] = []
        chosen_ids: set = set()

        def candidate_key(card: Dict[str, Any]) -> tuple:
            # Prefer cards whose basis still has quota, then whose band still has
            # quota, then by stable tiebreak.
            b_need = remaining_basis.get(card["basis"], 0) > 0
            band_need = remaining_band.get(card["complexityBand"], 0) > 0
            return (
                0 if b_need else 1,
                0 if band_need else 1,
                tiebreak[card["cardId"]],
                card["cardId"],
            )

        while len(chosen) < k:
            pool_left = [c for c in pool if c["cardId"] not in chosen_ids]
            if not pool_left:
                break
            pool_left.sort(key=candidate_key)
            pick = pool_left[0]
            chosen.append(pick)
            chosen_ids.add(pick["cardId"])
            if remaining_basis.get(pick["basis"], 0) > 0:
                remaining_basis[pick["basis"]] -= 1
            if remaining_band.get(pick["complexityBand"], 0) > 0:
                remaining_band[pick["complexityBand"]] -= 1

        if len(chosen) == k:
            ok, _reason, _perm = check_slate(chosen, k, bases_cfg, seed)
            if ok:
                return chosen

        # Diversity-first fallback: one card per distinct basis, then backfill.
        by_basis: Dict[str, List[Dict[str, Any]]] = {}
        for card in sorted(pool, key=lambda c: (tiebreak[c["cardId"]], c["cardId"])):
            by_basis.setdefault(card["basis"], []).append(card)
        diverse: List[Dict[str, Any]] = []
        diverse_ids: set = set()
        for cards in by_basis.values():
            diverse.append(cards[0])
            diverse_ids.add(cards[0]["cardId"])
            if len(diverse) >= k:
                break
        for card in sorted(pool, key=lambda c: (tiebreak[c["cardId"]], c["cardId"])):
            if len(diverse) >= k:
                break
            if card["cardId"] not in diverse_ids:
                diverse.append(card)
                diverse_ids.add(card["cardId"])
        if len(diverse) != k:
            return None
        ok, _reason, _perm = check_slate(diverse, k, bases_cfg, seed)
        return diverse if ok else None
