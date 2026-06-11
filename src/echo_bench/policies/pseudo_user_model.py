"""PSEUDO_USER_MODEL slate-selection policy for ECHO-Bench (Task C-006).

ISOLATED CONTRAST BASELINE — READ FIRST
=======================================
This is the **only** ECHO-Bench policy permitted a latent vector, and it exists
**purely as a contrast baseline**. The latent vector here is a SYNTHETIC SCORING
CONSTRUCT: a fixed, seeded numeric vector used to weight observable card
features. **It is NOT a model of a real person.** It does not represent, infer,
or claim anything about any user's preference, persona, emotion, wellbeing,
demographics, or experience. No such claim may be derived from this policy. The
benchmark uses it only to bound how much a *trace-only* policy gives up versus a
policy that is handed an (artificial) latent scoring vector.

Isolation invariants (the core point of this module)
----------------------------------------------------
- The latent vector lives ONLY inside this module. It is derived deterministically
  from ``latent_dim`` / ``latent_seed`` in config via a seeded local RNG.
- The latent vector is NEVER written into the trace and NEVER passed to or read
  by any other policy. ``select`` returns only ``cardId`` values; the round
  runner records only the standard observable round fields.
- No trace-only policy (TRACE_GREEDY, TRACE_LIN_UCB) imports or references this
  module. (Enforced by an acceptance test scanning their source.)

Scoring
-------
``select`` builds an observable feature vector per candidate card (from
``coordinateContribution``, ``complexityBand``, ``salienceScore`` and a bias),
projects it onto the synthetic latent vector (dot product), and assembles a
constraint-satisfying ``k``-slate by repeated argmax with a deterministic seeded
tie-break. Per-card ``scoreComponents`` ``{"latentScore": ..}`` are logged; the
latent vector itself is never placed into ``scoreComponents`` or the trace.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from echo_bench.cards.metrics import COMPLEXITY_BANDS
from echo_bench.env.constraints import check_slate
from echo_bench.logging import get_logger, log_ko
from echo_bench.metrics.utility import coordinate_cell
from echo_bench.policies.base import Policy
from echo_bench.utils.hash import canonical_hash

__all__ = ["PseudoUserModelPolicy", "BAND_ORDER", "VARIANTS", "DEFAULT_VARIANT"]

_logger = get_logger(__name__)

BAND_ORDER: tuple[str, ...] = tuple(name for name, _lo, _hi in COMPLEXITY_BANDS)

DEFAULT_LATENT_DIM = 6
DEFAULT_LATENT_SEED = 12345

# Observable feature blocks fed into the latent dot product. Order is stable.
COORD_DIMS = 4

# Contrast-baseline variants (Task C-008). BASIC is the original static-latent
# baseline (kept for back-compat). The two regularized variants exist so the
# contrast baseline is not a straw man: BASIC collapses ``artifact_diversity``
# toward 0, while these keep it positive. The latent vector stays internal-only
# in EVERY variant — none reads or emits any user/persona/emotion field.
DEFAULT_VARIANT = "BASIC"
VARIANTS: tuple[str, ...] = ("BASIC", "DIVERSITY_REG", "SESSION_EMBEDDING")

# Weight on the SESSION_EMBEDDING coordinate-novelty term (added to the
# normalized latent as a secondary key). Band coverage is handled lexicographically
# (see ``selection_key``), so no band-weight tuning is needed.
_SESSION_NOVELTY_WEIGHT = 1.0

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


class PseudoUserModelPolicy(Policy):
    """Isolated contrast baseline driven by a SYNTHETIC latent scoring vector.

    The latent vector is an artificial numeric construct (NOT a model of a real
    person). ``config`` carries at least ``k``; optional ``latent_dim`` and
    ``latent_seed`` parameterize the synthetic latent vector.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    def _observable_features(self, card: Dict[str, Any]) -> np.ndarray:
        """Build the per-card observable feature vector (no latent field read).

        Reads only ``coordinateContribution`` / ``complexityBand`` /
        ``salienceScore`` plus a constant bias. The latent vector multiplies
        THIS observable feature vector; the card itself carries no latent field.
        """
        cc = [float(x) for x in card.get("coordinateContribution", [])][:COORD_DIMS]
        cc = cc + [0.0] * (COORD_DIMS - len(cc))
        band_norm = _band_index(card.get("complexityBand")) / max(
            1, len(BAND_ORDER) - 1
        )
        salience = float(card.get("salienceScore", 0.0))
        return np.asarray(cc + [band_norm, salience, 1.0], dtype=float)

    def _latent_vector(self, dim: int, latent_seed: int) -> np.ndarray:
        """Construct the SYNTHETIC latent scoring vector (module-internal only).

        Deterministic from ``(latent_seed, dim)`` via a seeded local RNG. This is
        an artificial scoring construct, NOT a representation of any person. It is
        never written to the trace and never shared with another policy.
        """
        material = canonical_hash({"latent_seed": latent_seed, "latent_dim": dim})
        rng = random.Random(int(material, 16))
        return np.asarray([rng.uniform(-1.0, 1.0) for _ in range(dim)], dtype=float)

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate by synthetic latent scoring."""
        # Merge: the round runner passes only ``{"k": k}`` to override the slate
        # size, but the policy's own config (``variant``, ``latent_seed``) must
        # still apply during an episode — so self.config is the base and the
        # call-time config overrides it (rather than replacing it wholesale).
        effective = {**self.config, **(config or {})}
        k = int(effective["k"])
        latent_seed = int(effective.get("latent_seed", DEFAULT_LATENT_SEED))

        if k > len(pool):
            raise ValueError(
                f"PSEUDO_USER_MODEL 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        feature_dim = COORD_DIMS + 3  # coord block + band + salience + bias
        # latent_dim is honored but the dot product needs matching length; we use
        # the feature dimensionality so the synthetic vector aligns with features.
        latent = self._latent_vector(feature_dim, latent_seed)

        pool_hash = canonical_hash([c["cardId"] for c in pool])
        seed_material = canonical_hash(
            {
                "poolHash": pool_hash,
                "seed": seed,
                "policyVersion": self.policy_version(),
            }
        )
        rng = random.Random(int(seed_material, 16))
        tiebreak = {c["cardId"]: rng.random() for c in pool}

        bases_cfg = _load_bases_cfg()

        variant = str(effective.get("variant", DEFAULT_VARIANT))
        if variant not in VARIANTS:
            raise ValueError(
                f"PSEUDO_USER_MODEL 정책: 알 수 없는 variant={variant!r} "
                f"(사용 가능: {list(VARIANTS)})"
            )

        def latent_score(card: Dict[str, Any]) -> float:
            return float(latent @ self._observable_features(card))

        score_cache = {c["cardId"]: latent_score(c) for c in pool}

        # Normalized latent scores in [0, 1] so the diversity-regularization
        # terms (also ~[0, 1]) combine on a comparable scale across variants.
        raw_vals = list(score_cache.values())
        lo, hi = (min(raw_vals), max(raw_vals)) if raw_vals else (0.0, 0.0)
        span = hi - lo
        norm = {
            cid: ((v - lo) / span if span > 0 else 0.5)
            for cid, v in score_cache.items()
        }

        # Trace-derived context (observable fields only) for the regularized
        # variants: how often each complexity band already appeared, and the
        # session coordinate centroid of prior rounds. (This reads observable
        # trace fields; the synthetic latent vector stays internal-only.)
        prior_band_counts: Dict[Any, int] = {}
        prior_coords: List[List[float]] = []
        for record in trace.rounds():
            b = record.get("complexityBand")
            if b is not None:
                prior_band_counts[b] = prior_band_counts.get(b, 0) + 1
            cc = record.get("coordinateContribution")
            if cc is not None:
                prior_coords.append([float(x) for x in cc])
        n_prior = len(trace.rounds())
        session_centroid = None
        if prior_coords:
            dim = max(len(c) for c in prior_coords)
            session_centroid = [0.0] * dim
            for c in prior_coords:
                for i in range(len(c)):
                    session_centroid[i] += c[i]
            session_centroid = [x / len(prior_coords) for x in session_centroid]

        def selection_key(
            card: Dict[str, Any], chosen: List[Dict[str, Any]]
        ) -> tuple:
            """Variant selection key (ascending; lower = preferred).

            BASIC ranks purely by the raw normalized latent (back-compat). The
            regularized variants make complexity-band coverage the PRIMARY key —
            the least-used band (over prior rounds + the in-slate cards) is picked
            first — so the assembled slate is band-mixed and the controlled
            permutation rule then surfaces varying bands across rounds (preventing
            the BASIC single-band collapse). Each variant differs in its secondary
            key: DIVERSITY_REG adds in-slate coordinate-cell de-duplication, while
            SESSION_EMBEDDING adds coordinate-novelty against the session centroid.
            """
            base = norm[card["cardId"]]
            if variant == "BASIC":
                return (-base,)
            band = card.get("complexityBand")
            band_bucket = prior_band_counts.get(band, 0) + sum(
                1 for c in chosen if c.get("complexityBand") == band
            )
            if variant == "DIVERSITY_REG":
                cell = coordinate_cell(card.get("coordinateContribution", []))
                cell_bucket = sum(
                    1
                    for c in chosen
                    if coordinate_cell(c.get("coordinateContribution", [])) == cell
                )
                return (band_bucket, cell_bucket, -base)
            # SESSION_EMBEDDING: latent + coordinate-novelty vs the session centroid.
            novelty = 0.0
            if session_centroid is not None:
                cc = [float(x) for x in card.get("coordinateContribution", [])]
                length = min(len(cc), len(session_centroid))
                if length:
                    dist = math.sqrt(
                        sum((cc[i] - session_centroid[i]) ** 2 for i in range(length))
                    )
                    novelty = dist / (1.0 + dist)
            return (band_bucket, -(base + _SESSION_NOVELTY_WEIGHT * novelty))

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
                new_basis_bonus = (
                    1 if (need_new_basis and card["basis"] not in chosen_bases) else 0
                )
                return (
                    (-new_basis_bonus,)
                    + selection_key(card, chosen)
                    + (tiebreak[card["cardId"]], card["cardId"])
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
                    pool, k, score_cache, tiebreak, bases_cfg, seed
                )

        if slate is None:
            raise ValueError(
                f"PSEUDO_USER_MODEL 정책: k={k} 제약을 만족하는 슬레이트를 "
                f"구성하지 못했습니다 (poolHash={pool_hash})"
            )

        # Log only the resulting scalar score per card. The synthetic latent
        # vector itself is NEVER placed into scoreComponents or the trace.
        self.log_score_components(
            {c["cardId"]: {"latentScore": round(score_cache[c["cardId"]], 12)}
             for c in slate}
        )
        log_ko(
            _logger,
            f"PSEUDO_USER_MODEL 슬레이트 확정 (대조 베이스라인): k={k}, "
            f"latentSeed={latent_seed}, poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    @staticmethod
    def _diversity_fallback(
        pool: List[Dict[str, Any]],
        k: int,
        score_cache: Dict[Any, float],
        tiebreak: Dict[Any, float],
        bases_cfg: dict,
        seed: int,
    ) -> List[Dict[str, Any]] | None:
        """Distinct-basis-first fill when the greedy slate misses the constraint."""
        ordered = sorted(
            pool,
            key=lambda c: (
                -score_cache[c["cardId"]],
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
