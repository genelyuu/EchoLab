"""FIXED_LOW_TO_HIGH slate-selection policy for ECHO-Bench (Task C-002).

A *non-adaptive* schedule that walks the target complexity band from low to high
as the horizon advances. The target band for a round is a fixed function of the
**round index** read from a config schedule — the round index is taken purely as
a counter ``len(trace.rounds())`` and is NEVER used as an adaptive signal. The
policy reads no trace *content*: two different traces of the same length always
produce the same target band and (for the same pool/seed) the same slate.

Selection procedure
--------------------
1. Round index ``r = len(trace.rounds())`` (counter only).
2. Resolve the target complexity band from the configured ``schedule`` (a list of
   band names indexed by ``r``, clamped to the last entry past the horizon). If no
   schedule is given, derive an evenly-spaced low->high walk over ``horizon``.
3. Score every pool card by its band distance to the target (0 = exact band),
   with a seeded random tiebreak from a local RNG. Lower distance is preferred.
4. Greedily assemble a ``k``-slate from the band-sorted candidates such that the
   active ``k``-constraint (:func:`check_slate`) is satisfied, adding cards that
   introduce new bases first when diversity is still required.

Per-card ``scoreComponents`` (band distance + tiebreak) are logged for the slate.

Guardrail: NON-ADAPTIVE. No latent user/persona/emotion/preference field and no
trace *content* is read — only the trace *length* as a round counter.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.cards.metrics import COMPLEXITY_BANDS
from echo_bench.env.constraints import check_slate
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = ["FixedLowToHighPolicy", "BAND_ORDER", "band_index"]

_logger = get_logger(__name__)

# Ordinal complexity-band order derived from the documented bands in
# cards/metrics.py (low -> mid -> high). Never hard-coded independently.
BAND_ORDER: tuple[str, ...] = tuple(name for name, _lo, _hi in COMPLEXITY_BANDS)

DEFAULT_HORIZON = 6

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


def band_index(band: str) -> int:
    """Return the ordinal index of a complexity band (low=0 .. high=N-1)."""
    try:
        return BAND_ORDER.index(str(band))
    except ValueError:
        # Unknown band: treat as the highest so it sorts last (fail soft).
        return len(BAND_ORDER)


class FixedLowToHighPolicy(Policy):
    """Non-adaptive low->high complexity-band schedule.

    ``config`` carries at least ``k`` and an optional ``schedule`` (list of band
    names) and/or ``horizon`` (int). The schedule is a function of round index
    only.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    def _target_band(self, round_index: int, effective: Dict[str, Any]) -> str:
        """Resolve the scheduled target band for a round index (counter only)."""
        schedule = effective.get("schedule")
        if schedule:
            idx = min(round_index, len(schedule) - 1)
            return str(schedule[idx])
        # No explicit schedule: evenly-spaced low->high walk over the horizon.
        horizon = int(effective.get("horizon", DEFAULT_HORIZON))
        horizon = max(1, horizon)
        clamped = min(round_index, horizon - 1)
        # Map round position in [0, horizon-1] onto band index [0, len-1].
        if horizon == 1:
            band_pos = 0
        else:
            band_pos = round(clamped * (len(BAND_ORDER) - 1) / (horizon - 1))
        band_pos = min(band_pos, len(BAND_ORDER) - 1)
        return BAND_ORDER[band_pos]

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate targeting the scheduled band."""
        effective = config if config is not None else self.config
        k = int(effective["k"])

        if k > len(pool):
            raise ValueError(
                f"FIXED_LOW_TO_HIGH 정책: pool 크기 {len(pool)} 가 k={k} 보다 "
                "작아 슬레이트를 구성할 수 없습니다"
            )

        # Round index used ONLY as a counter (non-adaptive); never trace content.
        round_index = len(trace.rounds())
        target = self._target_band(round_index, effective)
        target_idx = band_index(target)

        # Seed material excludes trace content so identical (pool, seed, round
        # index, policyVersion) replay identically and remain non-adaptive.
        pool_hash = canonical_hash([c["cardId"] for c in pool])
        seed_material = canonical_hash(
            {
                "poolHash": pool_hash,
                "roundIndex": round_index,
                "seed": seed,
                "policyVersion": self.policy_version(),
            }
        )
        rng = random.Random(int(seed_material, 16))

        # Score every card: distance to target band + deterministic tiebreak.
        scored = []
        for card in pool:
            distance = abs(band_index(card["complexityBand"]) - target_idx)
            tiebreak = rng.random()
            scored.append((distance, tiebreak, card))
        # Ascending by (distance, tiebreak): closest-to-target first.
        scored.sort(key=lambda t: (t[0], t[1]))
        ranked = [t[2] for t in scored]
        components = {
            t[2]["cardId"]: {
                "targetBand": target,
                "bandDistance": t[0],
                "tiebreak": round(t[1], 12),
            }
            for t in scored
        }

        bases_cfg = _load_bases_cfg()
        slate = self._assemble(ranked, k, bases_cfg, seed)
        if slate is None:
            raise ValueError(
                f"FIXED_LOW_TO_HIGH 정책: k={k} 제약을 만족하는 슬레이트를 "
                f"구성하지 못했습니다 (targetBand={target}, poolHash={pool_hash})"
            )

        self.log_score_components(
            {c["cardId"]: components[c["cardId"]] for c in slate}
        )
        log_ko(
            _logger,
            f"FIXED_LOW_TO_HIGH 슬레이트 확정: k={k}, round={round_index}, "
            f"targetBand={target}, poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    def _assemble(
        self,
        ranked: List[Dict[str, Any]],
        k: int,
        bases_cfg: dict,
        seed: int,
    ) -> List[Dict[str, Any]] | None:
        """Greedily pick a band-prioritized, constraint-satisfying k-slate.

        Adds the highest-ranked cards while preferring ones that introduce a new
        basis whenever basis diversity is still required for the final check.
        """
        chosen: List[Dict[str, Any]] = []
        chosen_ids: set = set()
        chosen_bases: set = set()
        remaining = list(ranked)

        while len(chosen) < k and remaining:
            slots_left = k - len(chosen)
            # Prefer a card that adds a new basis if we may still need diversity.
            pick = None
            for card in remaining:
                if card["basis"] not in chosen_bases:
                    pick = card
                    break
            # If no new-basis card needed/available, or enough bases collected,
            # fall back to the best-ranked remaining card.
            if pick is None or len(chosen_bases) >= 3 or slots_left == 1 and len(chosen_bases) >= 2:
                pick = remaining[0]
            chosen.append(pick)
            chosen_ids.add(pick["cardId"])
            chosen_bases.add(pick["basis"])
            remaining = [c for c in remaining if c["cardId"] not in chosen_ids]

        if len(chosen) != k:
            return None
        ok, _reason, _perm = check_slate(chosen, k, bases_cfg, seed)
        if ok:
            return chosen

        # Diversity fallback: re-seed selection prioritizing distinct bases, then
        # backfill with closest-band cards.
        by_basis: Dict[str, List[Dict[str, Any]]] = {}
        for card in ranked:
            by_basis.setdefault(card["basis"], []).append(card)
        diverse: List[Dict[str, Any]] = []
        diverse_ids: set = set()
        for cards in by_basis.values():
            diverse.append(cards[0])
            diverse_ids.add(cards[0]["cardId"])
            if len(diverse) >= k:
                break
        for card in ranked:
            if len(diverse) >= k:
                break
            if card["cardId"] not in diverse_ids:
                diverse.append(card)
                diverse_ids.add(card["cardId"])
        if len(diverse) != k:
            return None
        ok, _reason, _perm = check_slate(diverse, k, bases_cfg, seed)
        return diverse if ok else None
