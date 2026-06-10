"""System-level utility metrics over observable traces (Task D-001).

This module implements the **trace-only** subset of the ECHO-Bench utility
metrics (Phase 1, Wave 3). Every metric here is a deterministic function of an
:class:`echo_bench.env.trace_state.TraceState` and nothing else: identical
traces yield identical values, and each metric is bounded to ``[0.0, 1.0]``.

What these metrics are
----------------------
They quantify *system-level* properties of the surfaced/selected sequence —
how much of the coordinate space the trace touched, how spread the complexity
bands are, how often selections repeat, and how smooth the round-to-round
progression is. They are **not** measures of user satisfaction, emotion,
wellbeing, preference, privacy, or legal compliance, and they make **no**
ecological / real-world generalization claim. Interpretation stays strictly
inside the controlled testbed (per the project guardrails).

Phase scope
-----------
The four trace-only metrics (need no strategy probe, no oracle reference) plus
the probe-driven :func:`strategy_sensitivity` (B-004 probes, Phase 2) are
implemented:

- :func:`coordinate_coverage`
- :func:`artifact_diversity`
- :func:`redundancy_rate`
- :func:`round_coherence`
- :func:`strategy_sensitivity`  (over multiple traces keyed by probe)

:func:`compute_all` returns the four *single-trace* metrics plus the
``traceHash`` they were computed over; it never invokes
:func:`strategy_sensitivity` (which inherently needs more than one trace).
:func:`compute_all_with_strategy` additionally folds in
:func:`strategy_sensitivity` for the experiment runner. The oracle-dependent
metric :func:`regret_to_oracle` (Phase 3, C-007) is now implemented: it measures
the normalized mean shortfall of the trace's achieved system-level objective
against the C-007 ``ORACLE_STRATEGY`` per-round reference. It still requires an
explicit ``oracle_ref`` and is excluded from both ``compute_all`` variants;
:func:`compute_all_with_oracle` is the variant (for E2/E3) that folds in
``strategy_sensitivity`` *and* ``regret_to_oracle``.

Identifiers, keys, and metric names stay English; runtime log lines are Korean
per the project logging convention. Hashing is delegated to
:func:`echo_bench.utils.hash.canonical_hash`; this module never reimplements it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash

__all__ = [
    "coordinate_coverage",
    "artifact_diversity",
    "redundancy_rate",
    "round_coherence",
    "compute_all",
    "compute_all_with_strategy",
    "compute_all_with_oracle",
    "strategy_sensitivity",
    "regret_to_oracle",
    "oracle_reference_from_objectives",
    "METRIC_KEYS",
    "COORDINATE_GRID_BINS",
    "REDUNDANCY_NEAR_DUP_EPS",
    "STRATEGY_SENSITIVITY_COORD_BINS",
    "REGRET_SCALE",
]

_logger = get_logger(__name__)

# The four trace-only metric keys returned by :func:`compute_all` (alongside
# ``traceHash``). Kept as an explicit constant so callers/tests can assert the
# exact key set.
METRIC_KEYS = (
    "coordinate_coverage",
    "artifact_diversity",
    "redundancy_rate",
    "round_coherence",
)

# Number of bins per coordinate dimension used to discretize the coordinate
# space for :func:`coordinate_coverage`. Deterministic, fixed constant.
COORDINATE_GRID_BINS = 8

# L2 distance below which two coordinate-contribution vectors are treated as
# near-duplicates in :func:`redundancy_rate`.
REDUNDANCY_NEAR_DUP_EPS = 1e-6

# Canonical complexity-band ordering used to map bands onto an ordinal scale for
# :func:`round_coherence`. Unknown bands fall back to a stable position derived
# from their sorted order, so the metric stays deterministic for any band set.
_KNOWN_BAND_ORDER = ("low", "mid", "high")

# Fixed ordinal scale for the observable ``complexityBand`` field used by
# :func:`strategy_sensitivity`. Higher value = higher complexity. Unknown bands
# map to ``0`` (defensive, deterministic). The denominator below normalizes the
# band ordinal into ``[0, 1]`` so it is commensurate with coordinate components.
_STRATEGY_BAND_ORDINAL = {"low": 0.0, "mid": 1.0, "high": 2.0}
_STRATEGY_BAND_MAX = 2.0

# Bins per coordinate dimension used to discretize the mean coordinate vector
# into a normalized selected-region distribution for :func:`strategy_sensitivity`.
STRATEGY_SENSITIVITY_COORD_BINS = 8

# Normalization scale for :func:`regret_to_oracle`. The per-round objective is
# the coordinate-novelty of the selected card — the L2 distance between the
# round's ``coordinateContribution`` (components clamped to ``[0, 1]``) and the
# summed contributions of all prior rounds, matching
# ``PreferCoordNoveltyProbe._score`` in oracle_strategy.py. The raw
# oracle-minus-achieved shortfall is averaged over rounds, divided by this fixed
# positive scale, and then clipped into ``[0, 1]`` (the final clip guarantees the
# bound regardless of the raw magnitude). The scale is a fixed, documented
# constant so the metric is deterministic and config-independent; ``1.0`` keeps
# the regret in the same coordinate-novelty units as the oracle reference (a mean
# shortfall of one full novelty unit saturates to ``1.0``).
REGRET_SCALE = 1.0


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval ``[0.0, 1.0]``."""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _coordinate_vectors(trace) -> List[List[float]]:
    """Return the per-round ``coordinateContribution`` vectors as float lists.

    Each round's contribution is coerced to a list of floats. Rounds whose
    contribution is missing or empty contribute an empty vector (skipped by the
    metrics that consume this helper).
    """
    vectors: List[List[float]] = []
    for record in trace.rounds():
        contribution = record.get("coordinateContribution")
        if not contribution:
            vectors.append([])
            continue
        vectors.append([float(x) for x in contribution])
    return vectors


def _band_to_ordinal(bands: Sequence[str]) -> Dict[str, float]:
    """Map complexity-band labels to ordinals normalized to ``[0.0, 1.0]``.

    Known bands (``low`` < ``mid`` < ``high``) get fixed positions; any band
    outside that set is appended in sorted order after the known ones. With a
    single distinct band every position is ``0.0``. The mapping is a pure
    function of the band set, so it is fully deterministic.
    """
    distinct = set(bands)
    known = [b for b in _KNOWN_BAND_ORDER if b in distinct]
    extra = sorted(distinct - set(_KNOWN_BAND_ORDER))
    ordered = known + extra
    n = len(ordered)
    if n <= 1:
        return {b: 0.0 for b in ordered}
    return {band: idx / (n - 1) for idx, band in enumerate(ordered)}


def coordinate_coverage(trace) -> float:
    """Fraction of the discretized coordinate space the trace touched.

    Definition: each round's ``coordinateContribution`` vector is mapped to a
    cell of a uniform grid with :data:`COORDINATE_GRID_BINS` bins per dimension.
    Coordinate components are assumed to lie in ``[0.0, 1.0]`` and are clamped
    into that range before binning, so the metric is robust to small overshoot.
    Coverage is the number of distinct grid cells visited divided by the number
    of rounds, i.e. ``distinct_cells / n_rounds`` — the fraction of decision
    steps that explored a *new* region of the coordinate space.

    The result is ``1.0`` when every round lands in a distinct cell (maximal
    spread) and approaches ``0.0`` when rounds collapse onto one cell (no
    spread). An empty trace returns ``0.0``. Deterministic and bounded to
    ``[0.0, 1.0]``.
    """
    vectors = [v for v in _coordinate_vectors(trace) if v]
    n_rounds = len(vectors)
    if n_rounds == 0:
        return 0.0

    bins = COORDINATE_GRID_BINS
    cells = set()
    for vec in vectors:
        cell = []
        for component in vec:
            c = _clamp01(component)
            idx = int(c * bins)
            if idx >= bins:  # c == 1.0 edge maps into the last bin
                idx = bins - 1
            cell.append(idx)
        cells.add(tuple(cell))

    return _clamp01(len(cells) / n_rounds)


def artifact_diversity(trace) -> float:
    """Normalized Shannon entropy of the complexity-band distribution.

    Definition: over all rounds, compute the empirical distribution of
    ``complexityBand`` values and its Shannon entropy ``H`` (natural log), then
    normalize by ``log(B)`` where ``B`` is the number of distinct bands present.
    A perfectly uniform spread across the observed bands gives ``1.0``; a trace
    concentrated on a single band gives ``0.0``.

    This is intentionally **trace-only**: it uses only the ``complexityBand``
    field already recorded in each round and does not consult any external
    pool/card lookup. An empty trace, or one with a single distinct band,
    returns ``0.0``. Deterministic and bounded to ``[0.0, 1.0]``.
    """
    bands = [r.get("complexityBand") for r in trace.rounds()]
    bands = [b for b in bands if b is not None]
    n = len(bands)
    if n == 0:
        return 0.0

    counts: Dict[Any, int] = {}
    for band in bands:
        counts[band] = counts.get(band, 0) + 1

    distinct = len(counts)
    if distinct <= 1:
        return 0.0

    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log(p)

    return _clamp01(entropy / math.log(distinct))


def redundancy_rate(trace) -> float:
    """Fraction of rounds that repeat an earlier selection or coordinate.

    Definition: walking the trace in order, a round is *redundant* if either
    its ``selectedCardId`` equals the ``selectedCardId`` of any earlier round,
    or its ``coordinateContribution`` is within :data:`REDUNDANCY_NEAR_DUP_EPS`
    (L2 distance) of any earlier round's contribution. The first occurrence of a
    card/coordinate is never counted as redundant; only later repeats are. The
    rate is ``redundant_rounds / n_rounds``.

    A trace whose selections and coordinates are all distinct returns ``0.0``;
    a trace where every round after the first repeats returns a value
    approaching ``1.0``. An empty trace returns ``0.0``. Deterministic and
    bounded to ``[0.0, 1.0]``.
    """
    rounds = trace.rounds()
    n = len(rounds)
    if n == 0:
        return 0.0

    seen_cards = set()
    seen_vectors: List[List[float]] = []
    redundant = 0

    for record in rounds:
        is_redundant = False

        card = record.get("selectedCardId")
        if card in seen_cards:
            is_redundant = True

        vec = record.get("coordinateContribution")
        vec = [float(x) for x in vec] if vec else []
        if vec:
            for prev in seen_vectors:
                if len(prev) != len(vec):
                    continue
                dist_sq = sum((a - b) ** 2 for a, b in zip(prev, vec))
                if math.sqrt(dist_sq) <= REDUNDANCY_NEAR_DUP_EPS:
                    is_redundant = True
                    break

        if is_redundant:
            redundant += 1

        if card is not None:
            seen_cards.add(card)
        if vec:
            seen_vectors.append(vec)

    return _clamp01(redundant / n)


def round_coherence(trace) -> float:
    """Smoothness of band / coordinate progression across consecutive rounds.

    Definition: for each adjacent pair of rounds we measure two step sizes,
    each normalized to ``[0.0, 1.0]``:

    - the absolute change in the ordinal position of ``complexityBand``
      (bands mapped to an evenly-spaced ordinal scale via the observed band
      set), and
    - the mean per-component absolute change of ``coordinateContribution``
      (components clamped to ``[0.0, 1.0]``; mismatched/empty vectors contribute
      a step of ``0.0`` for that pair).

    The per-pair step is the mean of the two normalized step sizes. Coherence is
    ``1.0 - mean(step)`` over all adjacent pairs: a perfectly flat progression
    (no change) scores ``1.0``, and maximally jumpy progression approaches
    ``0.0``. A trace with fewer than two rounds has no transitions and returns
    ``1.0`` (vacuously coherent). Deterministic and bounded to ``[0.0, 1.0]``.
    """
    rounds = trace.rounds()
    if len(rounds) < 2:
        return 1.0

    bands = [r.get("complexityBand") for r in rounds]
    band_ord = _band_to_ordinal([b for b in bands if b is not None])
    vectors = _coordinate_vectors(trace)

    steps: List[float] = []
    for i in range(1, len(rounds)):
        # Complexity-band ordinal step (already normalized to [0, 1]).
        prev_band = bands[i - 1]
        cur_band = bands[i]
        if prev_band is None or cur_band is None:
            band_step = 0.0
        else:
            band_step = abs(band_ord.get(cur_band, 0.0) - band_ord.get(prev_band, 0.0))

        # Coordinate step: mean per-component absolute change, clamped inputs.
        prev_vec = vectors[i - 1]
        cur_vec = vectors[i]
        if prev_vec and cur_vec and len(prev_vec) == len(cur_vec):
            diffs = [
                abs(_clamp01(a) - _clamp01(b))
                for a, b in zip(prev_vec, cur_vec)
            ]
            coord_step = sum(diffs) / len(diffs)
        else:
            coord_step = 0.0

        steps.append((band_step + coord_step) / 2.0)

    mean_step = sum(steps) / len(steps)
    return _clamp01(1.0 - mean_step)


def compute_all(trace) -> Dict[str, Any]:
    """Compute every trace-only utility metric over ``trace``.

    Returns a dict containing exactly the four trace-only metric values
    (:data:`METRIC_KEYS`) plus the ``traceHash`` they were computed over::

        {
            "traceHash": <trace.trace_hash()>,
            "coordinate_coverage": float in [0, 1],
            "artifact_diversity": float in [0, 1],
            "redundancy_rate":    float in [0, 1],
            "round_coherence":    float in [0, 1],
        }

    The two probe/oracle-dependent metrics are deferred and are **never**
    invoked here. Deterministic: identical traces produce an identical dict.
    """
    trace_hash = trace.trace_hash()
    result: Dict[str, Any] = {
        "traceHash": trace_hash,
        "coordinate_coverage": coordinate_coverage(trace),
        "artifact_diversity": artifact_diversity(trace),
        "redundancy_rate": redundancy_rate(trace),
        "round_coherence": round_coherence(trace),
    }
    _logger.info(
        "유틸리티 지표를 계산했습니다 (traceHash=%s, metrics=%d)",
        trace_hash,
        len(METRIC_KEYS),
    )
    return result


def _strategy_feature_vector(trace) -> List[float]:
    """Build a fixed-length observable feature vector for one trace.

    The vector concatenates three observable, system-level descriptors of the
    trace — all read from the trace's own round records, never from any latent /
    user / persona field (the trace schema has none):

    1. **Mean complexity-band ordinal** — the ``complexityBand`` of every round
       mapped onto :data:`_STRATEGY_BAND_ORDINAL` and averaged, then normalized
       to ``[0, 1]`` by :data:`_STRATEGY_BAND_MAX`. Captures *what complexity
       regime* the trace's selections occupy.
    2. **Selected-region distribution** — each round's
       ``coordinateContribution`` is discretized to a per-dimension bin
       (:data:`STRATEGY_SENSITIVITY_COORD_BINS` bins per dim, components clamped
       to ``[0, 1]``) and the bin indices are summed into a fixed-size histogram
       over the first coordinate dimension, normalized to sum to 1. Captures
       *where in the coordinate space* selections concentrate. (Using the first
       dimension's bins gives a fixed-length, dimension-count-independent
       descriptor.)
    3. **Mean coordinate-contribution vector** — the per-component mean of all
       rounds' ``coordinateContribution`` (clamped to ``[0, 1]``), padded /
       truncated to a fixed width equal to the histogram size. Captures the
       *average coordinate direction* the trace moved in.

    Every component lies in ``[0, 1]``. An empty trace yields an all-zero
    vector. The vector is a pure deterministic function of the trace's
    observable rounds.
    """
    bins = STRATEGY_SENSITIVITY_COORD_BINS
    rounds = trace.rounds()

    # (1) mean complexity-band ordinal, normalized to [0, 1].
    band_ords = [
        _STRATEGY_BAND_ORDINAL.get(r.get("complexityBand"), 0.0)
        for r in rounds
        if r.get("complexityBand") is not None
    ]
    mean_band = (sum(band_ords) / len(band_ords) / _STRATEGY_BAND_MAX) if band_ords else 0.0

    vectors = [v for v in _coordinate_vectors(trace) if v]

    # (2) selected-region distribution over the first-dimension bins.
    histogram = [0.0] * bins
    for vec in vectors:
        c = _clamp01(vec[0])
        idx = int(c * bins)
        if idx >= bins:
            idx = bins - 1
        histogram[idx] += 1.0
    total = sum(histogram)
    if total > 0.0:
        histogram = [h / total for h in histogram]

    # (3) mean coordinate-contribution vector, clamped, padded/truncated to bins.
    if vectors:
        width = max(len(v) for v in vectors)
        sums = [0.0] * width
        counts = [0] * width
        for vec in vectors:
            for i, comp in enumerate(vec):
                sums[i] += _clamp01(comp)
                counts[i] += 1
        mean_vec = [
            (sums[i] / counts[i]) if counts[i] else 0.0 for i in range(width)
        ]
    else:
        mean_vec = []
    # Pad / truncate to a fixed width so every trace yields the same length.
    mean_vec = (mean_vec + [0.0] * bins)[:bins]

    return [mean_band] + histogram + mean_vec


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def strategy_sensitivity(traces_by_probe: Dict[str, Any]) -> float:
    """Normalized spread of trace feature vectors *across controlled probes*.

    A policy's output is "strategy-sensitive" when the trace it produces changes
    a lot as the controlled input strategy (B-004 strategy probe) changes. This
    metric quantifies that as the normalized divergence of the traces' observable
    feature vectors across probes.

    Parameters
    ----------
    traces_by_probe
        Maps ``probeName`` (str) -> ``TraceState``. Each value is the trace a
        policy produced under that controlled probe.

    Definition
    ----------
    For each probe's trace we build the observable feature vector described in
    :func:`_strategy_feature_vector` (mean complexity-band ordinal, selected
    coordinate-region distribution, and mean coordinate-contribution vector —
    all in ``[0, 1]``, all read from observable trace fields only). The metric
    is the **mean pairwise L2 distance** between these feature vectors over all
    distinct probe pairs, normalized by ``sqrt(d)`` where ``d`` is the feature
    dimension (the maximum possible L2 distance between two points in the unit
    hypercube ``[0, 1]^d``), then clamped to ``[0, 1]``.

    - Identical traces under every probe -> all pairwise distances ``0`` -> the
      metric is ``0.0`` (no sensitivity).
    - Traces whose observable behaviour diverges across probes -> ``> 0``.
    - With fewer than two probes there are no pairs, so the metric is ``0.0``.

    This is a *system-level* statistic over observable traces and the controlled
    input probes; it is **not** a measure of user preference, emotion, wellbeing,
    privacy, or any real-world effect, and makes no ecological generalization
    claim. Deterministic: identical ``traces_by_probe`` -> identical value.
    """
    if not traces_by_probe or len(traces_by_probe) < 2:
        return 0.0

    # Iterate in a stable, name-sorted order so the result is independent of the
    # dict's insertion order.
    names = sorted(traces_by_probe)
    features = [_strategy_feature_vector(traces_by_probe[name]) for name in names]

    dim = len(features[0])
    max_dist = math.sqrt(dim) if dim > 0 else 1.0

    distances: List[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distances.append(_l2(features[i], features[j]))

    mean_dist = sum(distances) / len(distances)
    value = _clamp01(mean_dist / max_dist if max_dist > 0 else 0.0)

    _logger.info(
        "전략 민감도를 계산했습니다 (probes=%d, pairs=%d, value=%.6f)",
        len(names),
        len(distances),
        value,
    )
    return value


def compute_all_with_strategy(
    trace, traces_by_probe: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute the four trace-only metrics + ``strategy_sensitivity`` + hash.

    This is the variant the experiment runner (E2) uses when it has, for one
    policy/seed, the family of traces produced under each controlled probe.
    Returns the dict from :func:`compute_all` (the four trace-only metric values
    plus ``traceHash`` computed over ``trace``) with an additional
    ``strategy_sensitivity`` key computed over ``traces_by_probe``::

        {
            "traceHash": ...,
            "coordinate_coverage": ...,
            "artifact_diversity": ...,
            "redundancy_rate": ...,
            "round_coherence": ...,
            "strategy_sensitivity": float in [0, 1],
        }

    ``regret_to_oracle`` is **excluded** here too — it stays deferred until the
    C-007 oracle (Phase 3). Deterministic: identical inputs -> identical dict.
    """
    result = compute_all(trace)
    result["strategy_sensitivity"] = strategy_sensitivity(traces_by_probe)
    return result


def oracle_reference_from_objectives(values: List[float]) -> List[float]:
    """Coerce a list of per-round oracle objective values into a clean ref.

    Convenience helper for callers/tests that have produced the oracle's
    per-round objective values (e.g. the ``probeObjective`` of the
    oracle-selected card per round, from the C-007 ``ORACLE_STRATEGY`` policy).
    Returns a new list of floats aligned 1:1 with the rounds it describes, in the
    same order. It does not pad, truncate, or otherwise alter the count — it only
    coerces each value to ``float`` — so a length mismatch against the trace is
    surfaced by :func:`regret_to_oracle` rather than silently hidden here.

    :raises ValueError: if ``values`` is not a sequence of numbers (Korean
        message).
    """
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(
            "oracle_ref 목적값은 숫자들의 시퀀스여야 합니다 (values=%r)" % (values,)
        )
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:  # non-numeric element
        raise ValueError(
            "oracle_ref 목적값을 float 로 변환할 수 없습니다: %r" % (values,)
        ) from exc


def _achieved_coordinate_novelty(trace) -> List[float]:
    """Per-round achieved objective = coordinate-novelty of the selected card.

    This computes, for each round ``i`` in order, the SAME system-level objective
    the C-007 oracle maximizes under the default ``PREFER_COORD_NOVELTY`` probe
    (see ``oracle_strategy.py``): the Euclidean distance between round ``i``'s
    observable ``coordinateContribution`` and the *accumulated* (component-wise
    summed) ``coordinateContribution`` of all rounds strictly before ``i``. This
    mirrors ``PreferCoordNoveltyProbe._score`` evaluated on the round's selected
    card against the trace prefix:

    - The first round (no prior rounds) has no accumulated coordinate, so its
      achieved novelty is ``0.0`` — exactly the probe's empty-trace behaviour.
    - Components are clamped to ``[0, 1]`` before differencing so the objective is
      robust to small overshoot, matching the coordinate-space convention used by
      the other metrics in this module.
    - Distance is taken over the shared-length prefix of the two vectors
      (defensive against ragged dimensions), matching the probe's ``_l2_distance``.

    The result is one objective value per round, aligned with ``trace.rounds()``.
    A round with a missing/empty ``coordinateContribution`` contributes ``0.0``.
    Pure deterministic function of the observable trace.
    """
    novelties: List[float] = []
    accumulated: Optional[List[float]] = None
    for record in trace.rounds():
        contrib = record.get("coordinateContribution")
        vec = [_clamp01(float(x)) for x in contrib] if contrib else []

        if accumulated is None or not vec:
            novelties.append(0.0)
        else:
            length = min(len(accumulated), len(vec))
            dist_sq = sum((accumulated[i] - vec[i]) ** 2 for i in range(length))
            novelties.append(math.sqrt(dist_sq))

        # Accumulate this round's (clamped) contribution for later rounds.
        if vec:
            if accumulated is None:
                accumulated = list(vec)
            else:
                length = min(len(accumulated), len(vec))
                accumulated = [accumulated[i] + vec[i] for i in range(length)] + (
                    accumulated[length:] if len(accumulated) > length else vec[length:]
                )
    return novelties


def regret_to_oracle(trace, oracle_ref: Sequence[float]) -> float:
    """Normalized mean shortfall of the trace's objective vs. the C-007 oracle.

    The system-level objective is the **coordinate-novelty of the selected
    card** — the L2 distance between a round's observable
    ``coordinateContribution`` and the accumulated contributions of all prior
    rounds. This is the SAME objective the C-007 ``ORACLE_STRATEGY`` policy
    maximizes under its default ``PREFER_COORD_NOVELTY`` probe (see
    ``oracle_strategy.py``, which scores each candidate with
    ``PreferCoordNoveltyProbe._score``), so the oracle reference and the achieved
    trace objective are measured in identical units — an apples-to-apples
    comparison.

    Parameters
    ----------
    trace
        The realized :class:`~echo_bench.env.trace_state.TraceState`.
    oracle_ref
        A sequence of per-round oracle objective values — the best achievable
        documented system-level objective per round, produced by the C-007
        oracle — aligned 1:1 with ``trace.rounds()`` (round ``i`` of the trace
        corresponds to ``oracle_ref[i]``). :func:`oracle_reference_from_objectives`
        can clean a raw value list for this argument.

    Definition
    ----------
    Let ``achieved_i`` be the coordinate-novelty objective of trace round ``i``
    (see :func:`_achieved_coordinate_novelty`) and ``oracle_i = oracle_ref[i]``.
    The regret is::

        mean_i( clip(oracle_i - achieved_i, 0, None) ) / REGRET_SCALE

    clipped to ``[0, 1]``. Per-round shortfalls are clipped at ``0`` (the oracle
    is an upper reference, so a round where the trace meets or beats it
    contributes no regret), averaged over rounds, divided by the fixed
    :data:`REGRET_SCALE`, and clamped into ``[0, 1]``.

    - ``oracle_ref == achieved`` for every round -> regret ``0.0``.
    - ``oracle_ref`` strictly above achieved on some rounds -> regret ``> 0``,
      always bounded to ``[0, 1]``.
    - An empty trace (zero rounds) with an empty ``oracle_ref`` -> ``0.0``.

    This is a *system-level* metric over observable trace fields and the
    controlled oracle reference; it is **not** a measure of user preference,
    emotion, wellbeing, privacy, or any real-world effect, and makes no
    ecological generalization claim. Deterministic: identical ``(trace,
    oracle_ref)`` -> identical value.

    :raises ValueError: (Korean message) if ``oracle_ref`` is ``None`` or its
        length does not match the number of trace rounds. The reference is never
        silently padded or truncated.
    """
    achieved = _achieved_coordinate_novelty(trace)
    n = len(achieved)

    if oracle_ref is None:
        raise ValueError(
            "regret_to_oracle: oracle_ref 가 None 입니다 — C-007 오라클 기준값이 "
            "필요합니다 (라운드 수=%d)" % (n,)
        )

    ref = oracle_reference_from_objectives(list(oracle_ref))
    if len(ref) != n:
        raise ValueError(
            "regret_to_oracle: oracle_ref 길이(%d)가 트레이스 라운드 수(%d)와 "
            "일치하지 않습니다 — 기준값을 임의로 패딩하지 않습니다"
            % (len(ref), n)
        )

    if n == 0:
        value = 0.0
    else:
        shortfall = sum(max(ref[i] - achieved[i], 0.0) for i in range(n))
        value = _clamp01((shortfall / n) / REGRET_SCALE)

    trace_hash = trace.trace_hash()
    _logger.info(
        "오라클 대비 후회를 계산했습니다 (traceHash=%s, rounds=%d, value=%.6f)",
        trace_hash,
        n,
        value,
    )
    return value


def compute_all_with_oracle(
    trace, traces_by_probe: Dict[str, Any], oracle_ref: Sequence[float]
) -> Dict[str, Any]:
    """Four trace-only metrics + ``strategy_sensitivity`` + ``regret_to_oracle``.

    The variant the experiment runner (E2/E3) uses when it has, for one
    policy/seed, both the probe-keyed trace family AND the C-007 oracle per-round
    reference. Returns the dict from :func:`compute_all_with_strategy` plus a
    ``regret_to_oracle`` key computed over ``(trace, oracle_ref)``::

        {
            "traceHash": ...,
            "coordinate_coverage": ...,
            "artifact_diversity": ...,
            "redundancy_rate": ...,
            "round_coherence": ...,
            "strategy_sensitivity": float in [0, 1],
            "regret_to_oracle": float in [0, 1],
        }

    Deterministic: identical inputs -> identical dict. Raises the same
    :class:`ValueError` as :func:`regret_to_oracle` on a ``None``/length-mismatch
    ``oracle_ref``.
    """
    result = compute_all_with_strategy(trace, traces_by_probe)
    result["regret_to_oracle"] = regret_to_oracle(trace, oracle_ref)
    return result
