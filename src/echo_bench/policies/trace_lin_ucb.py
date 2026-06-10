"""TRACE_LIN_UCB slate-selection policy for ECHO-Bench (Task C-005).

A *trace-only* linear UCB contextual bandit. Every context feature is computed
**exclusively from observable trace fields and observable card fields** — the
accumulated coordinate contributions and complexity bands recorded in the trace,
and each candidate card's own observable ``coordinateContribution`` /
``complexityBand`` / ``salienceScore``. There is **no latent user vector** in
this policy of any kind: nothing here models a person, a persona, an emotion, a
preference, or a demographic. The ``A`` / ``b`` bandit state is a system-level
estimator over those observable features only.

Context features (per candidate card, all observable)
-----------------------------------------------------
- ``coordinate-gap``: per-dimension absolute gap between the card's
  ``coordinateContribution`` and the accumulated trace coordinate centroid
  (how much the card moves away from where the trace has already gone).
- ``complexity-band-progress`` (scalar): how well the card's complexity band
  aligns with the next band after the highest band seen so far in the trace.
- ``redundancy-indicator`` (scalar): closeness of the card's coordinates to the
  trace centroid (1 / (1 + distance)); high when the card duplicates prior
  coverage.
- a constant bias term.

Linear UCB
----------
The policy maintains ``A`` (``d x d``) and ``b`` (``d``) ridge-regression
matrices, seeded from ``A = lambda_reg * I`` and ``b = 0``. Across the rounds
already present in the *observable* trace it replays a deterministic update: for
each prior round it reconstructs that round's feature vector ``x`` from the
observable round record and folds in the documented system-level reward
(``coordinate novelty`` of the selected card — the L2 distance of the selected
card's ``coordinateContribution`` from the running coordinate centroid). The
update is ``A += x x^T`` and ``b += reward * x``. This is fully deterministic and
replays identically from ``(poolHash, traceHash, seed, policyVersion)``.

The per-card UCB score is ``mean + alpha * bonus`` where
``theta = A^{-1} b``, ``mean = theta . x``, and
``bonus = sqrt(x^T A^{-1} x)``. ``alpha`` and the active feature set come from
config. The k-slate is assembled by repeated argmax over the UCB score subject
to the active ``k``-constraint (basis-diversity aware), with a deterministic
seeded tie-break. Per-card ``scoreComponents`` ``{"mean": .., "bonus": ..}`` are
logged for the chosen cards.

Guardrail: TRACE-ONLY. This module never imports the isolated contrast-baseline
policy and never reads or constructs any latent / persona / emotion / preference
/ demographic / free-text field.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from echo_bench.cards.metrics import COMPLEXITY_BANDS
from echo_bench.env.constraints import check_slate
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = ["TraceLinUcbPolicy", "BAND_ORDER"]

_logger = get_logger(__name__)

BAND_ORDER: tuple[str, ...] = tuple(name for name, _lo, _hi in COMPLEXITY_BANDS)

DEFAULT_ALPHA = 1.0
DEFAULT_LAMBDA = 1.0
# Default feature set (observable, trace-only). Order is stable and load-bearing
# for the dimensionality of the A/b matrices.
DEFAULT_FEATURES = ("coordinate_gap", "band_progress", "redundancy", "bias")

# Number of coordinate dimensions used for the coordinate-gap feature block.
COORD_DIMS = 4

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


def _band_index(band: Any) -> int:
    try:
        return BAND_ORDER.index(str(band))
    except ValueError:
        return 0


class TraceLinUcbPolicy(Policy):
    """Trace-only linear-UCB contextual bandit (no latent user vector).

    ``config`` carries at least ``k`` and optional ``alpha`` (exploration
    coefficient), ``lambda_reg`` (ridge regularizer), and ``features`` (ordered
    list of active feature blocks).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    # -- feature extraction (observable only) ------------------------------

    def _trace_centroid_and_band(
        self, trace: Any
    ) -> tuple[List[float], int]:
        """Summarize observable trace coords -> (centroid, max band index).

        Reads ONLY ``coordinateContribution`` and ``complexityBand`` from trace
        rounds. Returns a zero-padded coordinate centroid (zeros if no rounds)
        and the highest complexity-band index seen so far (-1 if none).
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
            return ([0.0] * COORD_DIMS, max_band)
        dim = max(COORD_DIMS, max(len(c) for c in coords))
        centroid = [0.0] * dim
        for c in coords:
            for i in range(len(c)):
                centroid[i] += c[i]
        centroid = [x / len(coords) for x in centroid]
        return (centroid[:COORD_DIMS] if dim > COORD_DIMS else centroid, max_band)

    def _features(
        self,
        coord: List[float],
        band: Any,
        centroid: List[float],
        target_band_idx: int,
        active: tuple[str, ...],
    ) -> np.ndarray:
        """Build the observable context feature vector for one card.

        ``coord`` / ``band`` are the card's observable fields; ``centroid`` and
        ``target_band_idx`` summarize the observable trace. No latent field is
        consulted. The feature ordering follows ``active``.
        """
        cc = [float(x) for x in coord][:COORD_DIMS]
        cc = cc + [0.0] * (COORD_DIMS - len(cc))
        cen = list(centroid)[:COORD_DIMS]
        cen = cen + [0.0] * (COORD_DIMS - len(cen))

        gap = [abs(cc[i] - cen[i]) for i in range(COORD_DIMS)]
        dist = float(np.sqrt(sum((cc[i] - cen[i]) ** 2 for i in range(COORD_DIMS))))
        band_dist = abs(_band_index(band) - target_band_idx)
        band_progress = 1.0 / (1.0 + band_dist)
        redundancy = 1.0 / (1.0 + dist)

        parts: List[float] = []
        for feat in active:
            if feat == "coordinate_gap":
                parts.extend(gap)
            elif feat == "band_progress":
                parts.append(band_progress)
            elif feat == "redundancy":
                parts.append(redundancy)
            elif feat == "bias":
                parts.append(1.0)
            else:
                raise ValueError(
                    f"TRACE_LIN_UCB 정책: 알 수 없는 feature 이름 {feat!r} "
                    f"(허용: {sorted(set(DEFAULT_FEATURES))})"
                )
        return np.asarray(parts, dtype=float)

    def _feature_dim(self, active: tuple[str, ...]) -> int:
        """Total feature dimensionality for the active feature blocks."""
        dim = 0
        for feat in active:
            if feat == "coordinate_gap":
                dim += COORD_DIMS
            elif feat in ("band_progress", "redundancy", "bias"):
                dim += 1
            else:
                raise ValueError(
                    f"TRACE_LIN_UCB 정책: 알 수 없는 feature 이름 {feat!r}"
                )
        return dim

    def _replay_bandit_state(
        self,
        trace: Any,
        active: tuple[str, ...],
        lambda_reg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministically rebuild (A, b) from the observable trace.

        For each prior round, reconstruct that round's feature vector from the
        observable record and fold in the documented system-level reward
        (coordinate novelty of the selected card relative to the running
        coordinate centroid). Update is ``A += x x^T``; ``b += reward * x``.
        """
        dim = self._feature_dim(active)
        A = lambda_reg * np.eye(dim)
        b = np.zeros(dim)

        running: List[List[float]] = []
        max_band = -1
        for record in trace.rounds():
            cc = record.get("coordinateContribution")
            band = record.get("complexityBand")
            if cc is None:
                continue
            coord = [float(x) for x in cc]
            # Centroid / target band as of BEFORE this round was observed.
            if running:
                rdim = max(COORD_DIMS, max(len(c) for c in running))
                centroid = [0.0] * rdim
                for c in running:
                    for i in range(len(c)):
                        centroid[i] += c[i]
                centroid = [x / len(running) for x in centroid][:COORD_DIMS]
            else:
                centroid = [0.0] * COORD_DIMS
            target_band_idx = (
                min(max_band + 1, len(BAND_ORDER) - 1) if max_band >= 0 else 0
            )

            x = self._features(coord, band, centroid, target_band_idx, active)
            # Documented system-level reward: coordinate novelty of the selected
            # card (L2 distance from the running centroid).
            cpad = (coord + [0.0] * COORD_DIMS)[:COORD_DIMS]
            reward = float(
                np.sqrt(sum((cpad[i] - centroid[i]) ** 2 for i in range(COORD_DIMS)))
            )
            A = A + np.outer(x, x)
            b = b + reward * x

            running.append(coord)
            if band is not None:
                max_band = max(max_band, _band_index(band))
        return A, b

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate by trace-only linear UCB."""
        effective = config if config is not None else self.config
        k = int(effective["k"])
        alpha = float(effective.get("alpha", DEFAULT_ALPHA))
        lambda_reg = float(effective.get("lambda_reg", DEFAULT_LAMBDA))
        active = tuple(effective.get("features") or DEFAULT_FEATURES)

        if k > len(pool):
            raise ValueError(
                f"TRACE_LIN_UCB 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        A, b = self._replay_bandit_state(trace, active, lambda_reg)
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b

        centroid, max_band = self._trace_centroid_and_band(trace)
        target_band_idx = (
            min(max_band + 1, len(BAND_ORDER) - 1) if max_band >= 0 else 0
        )

        # Seed material includes traceHash: the bandit IS trace-driven; the RNG
        # is used only for deterministic tiebreaks among equal UCB scores.
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

        def ucb_components(card: Dict[str, Any]) -> Dict[str, float]:
            x = self._features(
                card["coordinateContribution"],
                card["complexityBand"],
                centroid,
                target_band_idx,
                active,
            )
            mean = float(theta @ x)
            bonus = float(np.sqrt(max(0.0, x @ (A_inv @ x))))
            return {
                "mean": round(mean, 12),
                "bonus": round(bonus, 12),
                "ucb": round(mean + alpha * bonus, 12),
            }

        comp_cache = {c["cardId"]: ucb_components(c) for c in pool}

        chosen: List[Dict[str, Any]] = []
        chosen_ids: set = set()
        chosen_bases: set = set()

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
                comp = comp_cache[card["cardId"]]
                new_basis_bonus = (
                    1 if (need_new_basis and card["basis"] not in chosen_bases) else 0
                )
                return (
                    -new_basis_bonus,
                    -comp["ucb"],
                    tiebreak[card["cardId"]],
                    card["cardId"],
                )

            pool_left.sort(key=rank_key)
            pick = pool_left[0]
            chosen.append(pick)
            chosen_ids.add(pick["cardId"])
            chosen_bases.add(pick["basis"])

        slate = chosen if len(chosen) == k else None
        if slate is not None:
            ok, _reason, _perm = check_slate(slate, k, bases_cfg, seed)
            if not ok:
                slate = self._diversity_fallback(
                    pool, k, comp_cache, tiebreak, bases_cfg, seed
                )

        if slate is None:
            raise ValueError(
                f"TRACE_LIN_UCB 정책: k={k} 제약을 만족하는 슬레이트를 구성하지 "
                f"못했습니다 (poolHash={pool_hash})"
            )

        self.log_score_components(
            {
                c["cardId"]: {
                    "mean": comp_cache[c["cardId"]]["mean"],
                    "bonus": comp_cache[c["cardId"]]["bonus"],
                }
                for c in slate
            }
        )
        log_ko(
            _logger,
            f"TRACE_LIN_UCB 슬레이트 확정: k={k}, alpha={alpha}, "
            f"featureDim={self._feature_dim(active)}, traceLen={len(trace.rounds())}, "
            f"poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    @staticmethod
    def _diversity_fallback(
        pool: List[Dict[str, Any]],
        k: int,
        comp_cache: Dict[Any, Dict[str, float]],
        tiebreak: Dict[Any, float],
        bases_cfg: dict,
        seed: int,
    ) -> List[Dict[str, Any]] | None:
        """Distinct-basis-first fill when the greedy slate misses the constraint."""
        ordered = sorted(
            pool,
            key=lambda c: (
                -comp_cache[c["cardId"]]["ucb"],
                tiebreak[c["cardId"]],
                c["cardId"],
            ),
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
