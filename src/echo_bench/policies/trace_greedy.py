"""TRACE_GREEDY slate-selection policy for ECHO-Bench (Task C-004).

A *trace-only* greedy policy. It scores candidate cards using **only observable
trace fields** — the coordinate contributions, complexity bands, and selected
cards already recorded in the trace — and never any latent / persona / emotion /
preference / demographic / free-text signal. From an empty trace it scores from
the in-slate context alone.

Per-card reward (weighted, config-driven):

- ``novelty``: L2 distance between the card's ``coordinateContribution`` and the
  accumulated trace coordinate centroid (cards far from what has been surfaced
  score higher) — rewards coordinate-gap coverage.
- ``progression``: alignment of the card's complexity band with the next band
  after the highest band seen so far in the trace — rewards low->high progress.
- ``redundancy`` (penalty): closeness of the card's coordinates to cards already
  chosen *within the slate being built* (greedy de-duplication).

Greedy argmax: repeatedly add the highest-scoring remaining card, recomputing the
in-slate redundancy each step; assemble a ``k``-slate satisfying the active
constraint. Ties broken by a seeded local RNG. The weighted components are logged
per chosen card.

Guardrail: TRACE-ONLY. Reads only ``coordinateContribution``, ``complexityBand``,
and ``selectedCardId`` from trace rounds. No latent field is touched anywhere.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.cards.metrics import COMPLEXITY_BANDS
from echo_bench.env.constraints import check_slate
from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.utility import coordinate_cell
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = ["TraceGreedyPolicy", "BAND_ORDER"]

_logger = get_logger(__name__)

BAND_ORDER: tuple[str, ...] = tuple(name for name, _lo, _hi in COMPLEXITY_BANDS)

# Per-card reward weights. The original three terms (novelty / progression /
# in-slate redundancy) are kept; the C-009 redesign ADDS session-level terms that
# fix the long-horizon coverage collapse: ``session_coverage`` rewards landing in
# a coordinate-grid cell the trace has NOT visited, ``cell_repulsion`` penalizes
# revisiting an already-visited cell (cross-round, not just in-slate), and
# ``exploration`` is a small deterministic epsilon (seeded jitter) so the greedy
# does not lock onto one neighborhood. All terms read OBSERVABLE trace fields only.
DEFAULT_WEIGHTS = {
    "novelty": 1.0,
    "progression": 0.5,
    "redundancy": 1.0,
    "session_coverage": 2.0,
    "cell_repulsion": 0.75,
    "exploration": 0.1,
}

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


def _band_index(band: str) -> int:
    try:
        return BAND_ORDER.index(str(band))
    except ValueError:
        return 0


def _l2(a: List[float], b: List[float]) -> float:
    """Euclidean distance over the shared prefix of two coordinate vectors."""
    n = min(len(a), len(b))
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


class TraceGreedyPolicy(Policy):
    """Greedy trace-only coordinate-novelty / progression policy.

    ``config`` carries at least ``k`` and an optional ``weights`` map
    (``novelty``, ``progression``, ``redundancy``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    def _trace_summary(self, trace: Any) -> tuple[List[float], int]:
        """Summarize observable trace coords -> (centroid, max band index).

        Reads ONLY ``coordinateContribution`` and ``complexityBand`` from trace
        rounds. Returns a coordinate centroid (empty if no rounds) and the
        highest complexity-band index seen so far (-1 if none).
        """
        coords: List[List[float]] = []
        max_band = -1
        for record in trace.rounds():
            cc = record.get("coordinateContribution")
            if cc is not None:
                coords.append([float(x) for x in cc])
            band = record.get("complexityBand")
            if band is not None:
                max_band = max(max_band, _band_index(band))
        if not coords:
            return ([], max_band)
        dim = max(len(c) for c in coords)
        centroid = [0.0] * dim
        for c in coords:
            for i in range(len(c)):
                centroid[i] += c[i]
        centroid = [x / len(coords) for x in centroid]
        return (centroid, max_band)

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate by greedy trace-only scoring."""
        # Merge: the round runner passes only ``{"k": k}`` to override the slate
        # size, but the policy's own config (``weights``) must still apply during
        # an episode — so self.config is the base and the call-time config
        # overrides it (rather than replacing it wholesale).
        effective = {**self.config, **(config or {})}
        k = int(effective["k"])
        weights = {**DEFAULT_WEIGHTS, **(effective.get("weights") or {})}

        if k > len(pool):
            raise ValueError(
                f"TRACE_GREEDY 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        centroid, max_band = self._trace_summary(trace)
        target_band_idx = min(max_band + 1, len(BAND_ORDER) - 1) if max_band >= 0 else 0

        # Session-level coverage memory (C-009): how many prior rounds landed in
        # each coordinate-grid cell. Reads only the observable
        # ``coordinateContribution`` of prior rounds — no latent field.
        visited_counts: Dict[tuple, int] = {}
        for record in trace.rounds():
            cc = record.get("coordinateContribution")
            if cc:
                cell = coordinate_cell(cc)
                visited_counts[cell] = visited_counts.get(cell, 0) + 1

        # Seed material includes traceHash: the policy IS trace-driven, but the
        # RNG is used only for deterministic tiebreaks among equal scores.
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
        tiebreak = {c["cardId"]: rng.random() for c in pool}

        bases_cfg = _load_bases_cfg()

        def novelty(card: Dict[str, Any]) -> float:
            cc = [float(x) for x in card["coordinateContribution"]]
            if not centroid:
                # No trace context: novelty proportional to vector magnitude.
                return _l2(cc, [0.0] * len(cc))
            return _l2(cc, centroid)

        def progression(card: Dict[str, Any]) -> float:
            # 1.0 when the card sits at the next target band, decaying with
            # ordinal distance from it.
            dist = abs(_band_index(card["complexityBand"]) - target_band_idx)
            return 1.0 / (1.0 + dist)

        def redundancy(card: Dict[str, Any], chosen: List[Dict[str, Any]]) -> float:
            # Penalty: inverse distance to the nearest already-chosen card.
            if not chosen:
                return 0.0
            cc = [float(x) for x in card["coordinateContribution"]]
            nearest = min(
                _l2(cc, [float(x) for x in o["coordinateContribution"]])
                for o in chosen
            )
            return 1.0 / (1.0 + nearest)

        def session_coverage_gain(card: Dict[str, Any]) -> float:
            # Reward (C-009): 1.0 if the card's coordinate cell is NOT yet visited
            # across the trace, else 0.0 — the marginal session coverage it adds.
            cc = card.get("coordinateContribution")
            if not cc:
                return 0.0
            return 0.0 if coordinate_cell(cc) in visited_counts else 1.0

        def cell_repulsion(card: Dict[str, Any]) -> float:
            # Penalty (C-009): how many prior rounds already sit in this card's
            # cell — pushes selection away from over-visited neighborhoods.
            cc = card.get("coordinateContribution")
            if not cc:
                return 0.0
            return float(visited_counts.get(coordinate_cell(cc), 0))

        def components_for(card: Dict[str, Any], chosen: List[Dict[str, Any]]) -> Dict[str, float]:
            nov = weights["novelty"] * novelty(card)
            prog = weights["progression"] * progression(card)
            red = weights["redundancy"] * redundancy(card, chosen)
            cov = weights["session_coverage"] * session_coverage_gain(card)
            rep = weights["cell_repulsion"] * cell_repulsion(card)
            # Deterministic epsilon exploration via the seeded tiebreak draw, so
            # the greedy occasionally breaks out of its preferred neighborhood
            # without using the global RNG or wall-clock.
            exp = weights["exploration"] * tiebreak[card["cardId"]]
            total = nov + prog - red + cov - rep + exp
            return {
                "novelty": round(nov, 12),
                "progression": round(prog, 12),
                "redundancy": round(-red, 12),  # logged as a (negative) penalty
                "session_coverage": round(cov, 12),
                "cell_repulsion": round(-rep, 12),  # logged as a (negative) penalty
                "exploration": round(exp, 12),
                "total": round(total, 12),
            }

        chosen: List[Dict[str, Any]] = []
        chosen_ids: set = set()
        chosen_bases: set = set()
        logged: Dict[Any, Dict[str, float]] = {}

        while len(chosen) < k:
            pool_left = [c for c in pool if c["cardId"] not in chosen_ids]
            if not pool_left:
                break
            slots_left = k - len(chosen)
            distinct_bases_left = len({c["basis"] for c in pool_left} - chosen_bases)
            need_new_basis = (
                len(chosen_bases) < 3
                and slots_left <= (3 - len(chosen_bases))
                and distinct_bases_left > 0
            )

            def rank_key(card: Dict[str, Any]) -> tuple:
                comp = components_for(card, chosen)
                # Argmax on total score; seeded tiebreak; constraint-aware boost
                # for new-basis cards when diversity is still required.
                new_basis_bonus = 1 if (need_new_basis and card["basis"] not in chosen_bases) else 0
                return (-new_basis_bonus, -comp["total"], tiebreak[card["cardId"]], card["cardId"])

            pool_left.sort(key=rank_key)
            pick = pool_left[0]
            logged[pick["cardId"]] = components_for(pick, chosen)
            chosen.append(pick)
            chosen_ids.add(pick["cardId"])
            chosen_bases.add(pick["basis"])

        slate = chosen if len(chosen) == k else None
        if slate is not None:
            ok, _reason, _perm = check_slate(slate, k, bases_cfg, seed)
            if not ok:
                slate = self._diversity_fallback(pool, k, tiebreak, bases_cfg, seed, components_for)
                if slate is not None:
                    logged = {c["cardId"]: components_for(c, []) for c in slate}

        if slate is None:
            raise ValueError(
                f"TRACE_GREEDY 정책: k={k} 제약을 만족하는 슬레이트를 구성하지 "
                f"못했습니다 (poolHash={pool_hash})"
            )

        self.log_score_components({c["cardId"]: logged[c["cardId"]] for c in slate})
        log_ko(
            _logger,
            f"TRACE_GREEDY 슬레이트 확정: k={k}, targetBandIdx={target_band_idx}, "
            f"traceLen={len(trace.rounds())}, poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    @staticmethod
    def _diversity_fallback(
        pool: List[Dict[str, Any]],
        k: int,
        tiebreak: Dict[Any, float],
        bases_cfg: dict,
        seed: int,
        components_for,
    ) -> List[Dict[str, Any]] | None:
        """Distinct-basis-first fill when the greedy slate misses the constraint."""
        ordered = sorted(
            pool,
            key=lambda c: (-components_for(c, [])["total"], tiebreak[c["cardId"]], c["cardId"]),
        )
        by_basis: Dict[str, List[Dict[str, Any]]] = {}
        for card in ordered:
            by_basis.setdefault(card["basis"], []).append(card)
        diverse: List[Dict[str, Any]] = []
        diverse_ids: set = set()
        for cards in by_basis.values():
            diverse.append(cards[0])
            diverse_ids.add(cards[0]["cardId"])
            if len(diverse) >= k:
                break
        for card in ordered:
            if len(diverse) >= k:
                break
            if card["cardId"] not in diverse_ids:
                diverse.append(card)
                diverse_ids.add(card["cardId"])
        if len(diverse) != k:
            return None
        ok, _reason, _perm = check_slate(diverse, k, bases_cfg, seed)
        return diverse if ok else None
