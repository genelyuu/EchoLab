"""Strategy probes for ECHO-Bench (Tasks B-004, B-007 / TRD B-008).

GUARDRAIL — READ FIRST
======================
**Strategy probes are controlled, instrumented INPUT policies, not synthetic
users.** A probe is a fixed, documented rule that, given an observable slate and
the observable trace, selects exactly one card from the slate. Probes carry no
latent user state of any kind: there is *no* user/persona/emotion/preference/
demographic vector anywhere in this module, and a probe never reads such a field
because the trace and card schemas do not contain one (see
:mod:`echo_bench.env.trace_state`). Probes select purely from *observable card
fields* (``complexityBand``, ``salienceScore``, ``coordinateContribution``,
``visualMetrics``, ``cardId``, ``basis``).

    "Probes are controlled instrumented inputs, not models of people; the
    testbed is intentionally controlled, not ecologically realistic."

Probes drive the ``strategy_sensitivity`` measurement in D-001. They let us ask
"how does this policy's behaviour change under different *controlled input
strategies*?" — never "how would a real person behave?".

Determinism
===========
Each probe is a pure deterministic function of ``(slate, trace, seed,
probeVersion)``. Selection is identical across processes and platforms for the
same inputs. Ties (and only ties) are broken by a *seeded local*
``random.Random`` whose seed is derived from
``canonical_hash((sorted slate cardIds, seed, probeVersion))`` — never the
global RNG, wall-clock, or process entropy. Changing a probe's ``probeVersion``
changes its derived RNG seed *and* the recorded probe identity, so a different
version is distinguishable in the trace/manifest.

Probe set expansion (B-007 / TRD B-008)
=======================================
The registry carries seven probes so that probe-separability measurement has a
sufficiently diverse separation target. Each probe maps one TRD diagnostic axis
onto a REAL observable card field:

- complexity  -> ``complexityBand``               (PREFER_HIGH_COMPLEXITY)
- salience    -> ``salienceScore``                (PREFER_LOW_SALIENCE)
- novelty     -> ``coordinateContribution`` vs trace (PREFER_COORD_NOVELTY)
- boundary    -> ``visualMetrics.edgeDensity``    (PREFER_HIGH_EDGE_DENSITY)
- fragment    -> ``visualMetrics.spatialFrequency`` (PREFER_HIGH_SPATIAL_FREQUENCY)
- gravity     -> L2 magnitude of ``coordinateContribution`` (PREFER_LOW_COORD_MAGNITUDE)
- residue     -> nearness to the trace coordinate centroid  (PREFER_COORD_RESIDUE)

The TRD's suggested "basis-switching" axis is intentionally NOT registered: the
trace round record stores only ``selectedCardId`` (no ``basis``), so the basis
of the previously selected card is not an observable trace field and a probe
scoring it would violate the observable-fields-only contract.

Experiments E2/E3 consume the frozen :data:`DEFAULT_PROBE_SET` (the original
B-004 trio), NOT the full registry — growing the registry must never change
existing experiment behaviour or report hashes. E-019 opts experiments into the
expanded set explicitly. Pairwise overlap diagnostics for the expanded set live
in :mod:`echo_bench.probes.probe_overlap`.

All identifiers, keys, and version strings stay English; runtime log messages
are Korean per the project logging convention (see :mod:`echo_bench.logging`).
"""

from __future__ import annotations

import random
from typing import Any, List, Mapping, Optional, Sequence

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "StrategyProbe",
    "PreferHighComplexityProbe",
    "PreferLowSalienceProbe",
    "PreferCoordNoveltyProbe",
    "PreferHighEdgeDensityProbe",
    "PreferHighSpatialFrequencyProbe",
    "PreferLowCoordMagnitudeProbe",
    "PreferCoordResidueProbe",
    "PROBES",
    "DEFAULT_PROBE_SET",
    "get_probe",
    "COMPLEXITY_BAND_ORDER",
]

_logger = get_logger(__name__)

# Ordinal ranking for the observable ``complexityBand`` field. Higher value =
# higher complexity. Unknown bands sort lowest (defensive, deterministic).
COMPLEXITY_BAND_ORDER = {"low": 0, "mid": 1, "high": 2}


def _band_rank(card: Mapping[str, Any]) -> int:
    """Return the ordinal rank of a card's observable ``complexityBand``."""
    return COMPLEXITY_BAND_ORDER.get(card.get("complexityBand"), -1)


def _card_id(card: Mapping[str, Any]) -> str:
    """Return a card's ``cardId``, failing closed (Korean message) if absent."""
    if "cardId" not in card:
        raise ValueError("카드에 cardId 필드가 없습니다")
    return card["cardId"]


def _seeded_rng(
    slate: Sequence[Mapping[str, Any]], seed: Any, probe_version: str
) -> random.Random:
    """Return a local ``random.Random`` seeded deterministically.

    The seed is ``int(canonical_hash((sorted cardIds, seed, probeVersion)), 16)``
    so it depends only on the observable slate identities, the caller seed, and
    the probe version — and never on the global RNG or wall-clock.
    """
    sorted_ids = sorted(_card_id(c) for c in slate)
    digest = canonical_hash(
        {
            "slate_card_ids": sorted_ids,
            "seed": seed,
            "probeVersion": probe_version,
        }
    )
    return random.Random(int(digest, 16))


def _accumulated_coordinates(trace: Any) -> Optional[List[float]]:
    """Sum the observable ``coordinateContribution`` over all recorded rounds.

    Returns ``None`` if the trace has no rounds (no accumulated coordinates yet)
    or exposes no readable round history. Only the observable
    ``coordinateContribution`` field is read — never any user/latent field.
    """
    if trace is None:
        return None
    rounds_fn = getattr(trace, "rounds", None)
    if not callable(rounds_fn):
        return None
    rounds = rounds_fn()
    acc: Optional[List[float]] = None
    for record in rounds:
        contrib = record.get("coordinateContribution")
        if contrib is None:
            continue
        vec = [float(x) for x in contrib]
        if acc is None:
            acc = list(vec)
        else:
            # Sum component-wise over the shared prefix length (defensive).
            length = min(len(acc), len(vec))
            acc = [acc[i] + vec[i] for i in range(length)]
    return acc


def _l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance over the shared-length prefix of a and b."""
    length = min(len(a), len(b))
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(length)) ** 0.5


def _visual_metric(card: Mapping[str, Any], key: str) -> float:
    """Return an observable ``visualMetrics`` scalar, defaulting to 0.0.

    Reads only the card's own ``visualMetrics`` dict (real image statistics from
    A-006) — never any user/latent/semantic field. Missing metrics score 0.0 so
    selection on degenerate inputs falls through to the deterministic tie-break.
    """
    metrics = card.get("visualMetrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _coordinate_centroid(trace: Any) -> Optional[List[float]]:
    """Mean of the observable per-round ``coordinateContribution`` vectors.

    Returns ``None`` when the trace has no rounds with a readable
    ``coordinateContribution`` (no centroid yet). Only the observable
    coordinate field is read — never any user/latent field.
    """
    if trace is None:
        return None
    rounds_fn = getattr(trace, "rounds", None)
    if not callable(rounds_fn):
        return None
    acc: Optional[List[float]] = None
    count = 0
    for record in rounds_fn():
        contrib = record.get("coordinateContribution")
        if contrib is None:
            continue
        vec = [float(x) for x in contrib]
        if acc is None:
            acc = list(vec)
        else:
            length = min(len(acc), len(vec))
            acc = [acc[i] + vec[i] for i in range(length)]
        count += 1
    if acc is None or count == 0:
        return None
    return [x / count for x in acc]


class StrategyProbe:
    """Base class for a controlled, deterministic strategy probe.

    A probe is an instrumented *input policy*, not a synthetic user. Subclasses
    implement :meth:`_score` (higher score = selected first) over observable
    card fields only. The base :meth:`select` finds the max-scoring card(s) and
    breaks ties with a seeded local RNG, guaranteeing a deterministic, in-slate
    selection. No subclass may introduce any latent user/persona/preference
    state.
    """

    #: English machine-read probe name (set by each subclass).
    name: str = "BASE"
    #: English version tag; bumping it changes the recorded probe identity.
    version: str = "p0"

    def probe_version(self) -> str:
        """Return this probe's ``probeVersion`` identity string."""
        return self.version

    def _score(
        self, card: Mapping[str, Any], trace: Any
    ) -> float:
        """Return a selection score for a card (higher = selected first).

        Must read only observable card fields (and observable trace fields via
        ``trace``). Subclasses override this.
        """
        raise NotImplementedError

    def select(
        self,
        slate: Sequence[Mapping[str, Any]],
        trace: Any,
        seed: Any,
    ) -> str:
        """Select exactly one ``cardId`` from ``slate`` by this probe's rule.

        Pure deterministic function of ``(slate, trace, seed, probeVersion)``.
        Returns the ``cardId`` of the single max-scoring card; exact ties are
        broken by a seeded local RNG (uniform choice among the tied cardIds,
        sorted for stable ordering). Raises :class:`ValueError` (Korean message)
        on an empty slate.
        """
        if not slate:
            raise ValueError("빈 슬레이트에서는 카드를 선택할 수 없습니다")

        scored = [(self._score(card, trace), _card_id(card)) for card in slate]
        best_score = max(score for score, _ in scored)
        # Tie set, sorted by cardId for a stable, platform-independent order.
        tied = sorted(cid for score, cid in scored if score == best_score)

        if len(tied) == 1:
            selected = tied[0]
        else:
            rng = _seeded_rng(slate, seed, self.version)
            selected = rng.choice(tied)

        _logger.info(
            "전략 프로브 선택: probe=%s, probeVersion=%s, selectedCardId=%s, "
            "동점수=%d",
            self.name,
            self.version,
            selected,
            len(tied),
        )
        return selected


class PreferHighComplexityProbe(StrategyProbe):
    """Pick the card with the highest observable ``complexityBand``.

    Controlled input strategy: "always surface the most structurally complex
    option available". Reads only ``complexityBand``. Not a model of any person.
    """

    name = "PREFER_HIGH_COMPLEXITY"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        return float(_band_rank(card))


class PreferLowSalienceProbe(StrategyProbe):
    """Pick the card with the lowest observable ``salienceScore``.

    Controlled input strategy: "always surface the least salient option".
    Negates ``salienceScore`` so that lowest salience scores highest. Reads only
    ``salienceScore``. Not a model of any person.
    """

    name = "PREFER_LOW_SALIENCE"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        return -float(card.get("salienceScore", 0.0))


class PreferCoordNoveltyProbe(StrategyProbe):
    """Pick the card whose ``coordinateContribution`` is farthest from the trace.

    Controlled input strategy: "always surface the option that moves farthest
    from where the trace has already gone". Distance is the Euclidean distance
    between a card's observable ``coordinateContribution`` and the accumulated
    (summed) coordinate contributions of all prior rounds in the observable
    trace. When the trace is empty there is no accumulated coordinate, so every
    card scores ``0.0`` and selection falls through to the deterministic seeded
    tie-break. Reads only ``coordinateContribution``. Not a model of any person.
    """

    name = "PREFER_COORD_NOVELTY"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        accumulated = _accumulated_coordinates(trace)
        if accumulated is None:
            return 0.0
        contrib = card.get("coordinateContribution")
        if contrib is None:
            return 0.0
        return _l2_distance(contrib, accumulated)


class PreferHighEdgeDensityProbe(StrategyProbe):
    """Pick the card with the highest observable ``visualMetrics.edgeDensity``.

    Controlled input strategy for the TRD "boundary" axis: "always surface the
    option with the most boundary/edge structure". Scores the real A-006
    gradient-magnitude edge-density statistic of the rendered raster. Reads only
    ``visualMetrics.edgeDensity``. Not a model of any person.
    """

    name = "PREFER_HIGH_EDGE_DENSITY"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        return _visual_metric(card, "edgeDensity")


class PreferHighSpatialFrequencyProbe(StrategyProbe):
    """Pick the card with the highest observable ``visualMetrics.spatialFrequency``.

    Controlled input strategy for the TRD "fragment" axis: "always surface the
    option with the most fine-grained / fragmented structure", measured by the
    real A-006 high-frequency FFT energy ratio of the rendered raster. Reads
    only ``visualMetrics.spatialFrequency``. Not a model of any person.
    """

    name = "PREFER_HIGH_SPATIAL_FREQUENCY"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        return _visual_metric(card, "spatialFrequency")


class PreferLowCoordMagnitudeProbe(StrategyProbe):
    """Pick the card whose ``coordinateContribution`` has the smallest L2 norm.

    Controlled input strategy for the TRD "gravity" axis: "always surface the
    option that pulls the accumulated trace coordinates toward the origin".
    Scores the negated Euclidean magnitude of the card's observable
    ``coordinateContribution`` (missing vectors score as magnitude 0.0, i.e.
    maximal pull). Reads only ``coordinateContribution``. Not a model of any
    person.
    """

    name = "PREFER_LOW_COORD_MAGNITUDE"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        contrib = card.get("coordinateContribution")
        if contrib is None:
            return 0.0
        return -sum(float(x) ** 2 for x in contrib) ** 0.5


class PreferCoordResidueProbe(StrategyProbe):
    """Pick the card nearest the trace's accumulated coordinate centroid.

    Controlled input strategy for the TRD "residue" axis — the anti-novelty
    counterpart of :class:`PreferCoordNoveltyProbe`: "always surface the option
    that stays closest to where the trace has already gone". The score is the
    negated Euclidean distance between the card's observable
    ``coordinateContribution`` and the centroid (mean) of the per-round
    ``coordinateContribution`` vectors recorded in the observable trace. When
    the trace is empty there is no centroid, so every card scores ``0.0`` and
    selection falls through to the deterministic seeded tie-break. Reads only
    ``coordinateContribution``. Not a model of any person.
    """

    name = "PREFER_COORD_RESIDUE"
    version = "p1"

    def _score(self, card: Mapping[str, Any], trace: Any) -> float:
        centroid = _coordinate_centroid(trace)
        if centroid is None:
            return 0.0
        contrib = card.get("coordinateContribution")
        if contrib is None:
            return 0.0
        return -_l2_distance(contrib, centroid)


# Registry of named probes. Names are English machine-read identifiers.
PROBES = {
    PreferHighComplexityProbe.name: PreferHighComplexityProbe(),
    PreferLowSalienceProbe.name: PreferLowSalienceProbe(),
    PreferCoordNoveltyProbe.name: PreferCoordNoveltyProbe(),
    PreferHighEdgeDensityProbe.name: PreferHighEdgeDensityProbe(),
    PreferHighSpatialFrequencyProbe.name: PreferHighSpatialFrequencyProbe(),
    PreferLowCoordMagnitudeProbe.name: PreferLowCoordMagnitudeProbe(),
    PreferCoordResidueProbe.name: PreferCoordResidueProbe(),
}

# Frozen default subset consumed by the existing experiments (E2 / E3): exactly
# the original B-004 trio, in registration order. Growing PROBES (B-007 / TRD
# B-008) must be bit-identical for those runs — same probe list, same report
# hashes — so the experiments iterate THIS constant, never the full registry.
# E-019 opts experiments into the expanded set explicitly.
DEFAULT_PROBE_SET = (
    PreferHighComplexityProbe.name,
    PreferLowSalienceProbe.name,
    PreferCoordNoveltyProbe.name,
)


def get_probe(name: str) -> StrategyProbe:
    """Return the registered probe for ``name``.

    Raises :class:`KeyError` (Korean message) if no probe is registered under
    ``name``.
    """
    if name not in PROBES:
        raise KeyError(
            f"등록되지 않은 전략 프로브 이름입니다: {name!r} "
            f"(사용 가능: {sorted(PROBES)})"
        )
    return PROBES[name]
