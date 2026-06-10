"""Card record schema for ECHO-Bench (Tasks A-002 / A-006).

A card is a seed-generated stimulus object produced from ``basis / seed /
params`` only. It carries EXACTLY the eleven fields fixed by the Card / Basis
Invariants in CLAUDE.md and docs/04_BASIS_CARD_GENERATION.md:

    cardId, basis, seed, params, visualMetrics, coordinateContribution,
    complexityScore, complexityBand, salienceScore, renderHash, rendererVersion

No personal / user / persona / emotion / preference / semantic field may exist
anywhere in the record. :meth:`Card.from_dict` fails closed on any extra key.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

# The exact, ordered set of allowed card fields. Used both for construction and
# for the forbidden-field rejection in from_dict.
CARD_FIELDS = (
    "cardId",
    "basis",
    "seed",
    "params",
    "visualMetrics",
    "coordinateContribution",
    "complexityScore",
    "complexityBand",
    "salienceScore",
    "renderHash",
    "rendererVersion",
)


@dataclass(frozen=True)
class Card:
    """Immutable card record. Contains exactly the eleven invariant fields."""

    cardId: str
    basis: str
    seed: int
    params: dict[str, float]
    visualMetrics: dict[str, float]
    coordinateContribution: tuple[float, ...]
    complexityScore: float
    complexityBand: str
    salienceScore: float
    renderHash: str
    rendererVersion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view with the canonical field ordering.

        ``coordinateContribution`` is normalized to a list so the record is
        JSON / hash friendly.
        """
        d = asdict(self)
        d["coordinateContribution"] = list(self.coordinateContribution)
        return {key: d[key] for key in CARD_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Card":
        """Reconstruct a :class:`Card`, rejecting any extra / forbidden key.

        Raises:
            ValueError: if ``data`` contains any key outside :data:`CARD_FIELDS`
                (e.g. ``persona``, ``user_id``, ``emotion``) or is missing a
                required key.
        """
        keys = set(data.keys())
        allowed = set(CARD_FIELDS)
        extra = keys - allowed
        if extra:
            raise ValueError(
                f"Card.from_dict rejected forbidden/extra field(s): "
                f"{sorted(extra)}"
            )
        missing = allowed - keys
        if missing:
            raise ValueError(
                f"Card.from_dict missing required field(s): {sorted(missing)}"
            )
        coord = data["coordinateContribution"]
        return cls(
            cardId=str(data["cardId"]),
            basis=str(data["basis"]),
            seed=int(data["seed"]),
            params=dict(data["params"]),
            visualMetrics=dict(data["visualMetrics"]),
            coordinateContribution=tuple(float(x) for x in coord),
            complexityScore=float(data["complexityScore"]),
            complexityBand=str(data["complexityBand"]),
            salienceScore=float(data["salienceScore"]),
            renderHash=str(data["renderHash"]),
            rendererVersion=str(data["rendererVersion"]),
        )


# Sanity guard: keep the dataclass and the canonical field tuple in lockstep.
assert tuple(f.name for f in fields(Card)) == CARD_FIELDS, (
    "Card dataclass fields drifted from CARD_FIELDS"
)
