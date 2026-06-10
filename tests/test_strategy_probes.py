"""Tests for echo_bench.probes.strategy_probes (Task B-004).

These verify that strategy probes are deterministic, in-slate, version-aware,
controlled instrumented inputs that carry NO latent user/persona/preference
field and select purely from observable card fields.
"""

from __future__ import annotations

import pytest

from echo_bench.env.trace_state import FORBIDDEN_FIELDS, TraceState
from echo_bench.probes.strategy_probes import (
    PROBES,
    PreferCoordNoveltyProbe,
    PreferHighComplexityProbe,
    PreferLowSalienceProbe,
    StrategyProbe,
    get_probe,
)


def _card(card_id, basis, band, salience, coord):
    return {
        "cardId": card_id,
        "basis": basis,
        "complexityBand": band,
        "salienceScore": salience,
        "coordinateContribution": coord,
    }


def _crafted_slate():
    # Designed so each probe prefers a DIFFERENT card:
    #  - highest complexity -> c-high (band "high")
    #  - lowest salience     -> c-lowsal (salience 0.01)
    #  - coord novelty       -> c-far (coord far from accumulated trace coords)
    return [
        _card("c-high", "B1", "high", 0.5, [0.0, 0.0, 0.0]),
        _card("c-lowsal", "B2", "low", 0.01, [0.1, 0.1, 0.1]),
        _card("c-far", "B3", "mid", 0.5, [50.0, 50.0, 50.0]),
        _card("c-plain", "B4", "low", 0.5, [0.2, 0.2, 0.2]),
    ]


def _trace_with_history():
    trace = TraceState()
    trace.append_round(
        {
            "candidatePoolHash": "pool-1",
            "slate": ["a", "b"],
            "selectedCardId": "a",
            "coordinateContribution": [1.0, 1.0, 1.0],
            "complexityBand": "mid",
            "salienceScore": 0.3,
            "slotPermutation": [0, 1],
        }
    )
    return trace


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PROBES))
def test_probe_deterministic_same_inputs(name):
    probe = get_probe(name)
    slate = _crafted_slate()
    trace = _trace_with_history()
    first = probe.select(slate, trace, seed=42)
    for _ in range(5):
        assert probe.select(slate, trace, seed=42) == first


def test_probe_deterministic_across_fresh_instances():
    # Same selection from a freshly constructed probe instance (no hidden state
    # carried between calls / instances => stable across processes).
    slate = _crafted_slate()
    trace = _trace_with_history()
    a = PreferHighComplexityProbe().select(slate, trace, seed=7)
    b = PreferHighComplexityProbe().select(slate, trace, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# Selection is always a cardId in the slate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PROBES))
@pytest.mark.parametrize("seed", [0, 1, 99, 123456])
def test_selection_is_in_slate(name, seed):
    probe = get_probe(name)
    slate = _crafted_slate()
    slate_ids = {c["cardId"] for c in slate}
    selected = probe.select(slate, _trace_with_history(), seed=seed)
    assert selected in slate_ids


def test_empty_slate_raises():
    probe = get_probe("PREFER_HIGH_COMPLEXITY")
    with pytest.raises(ValueError):
        probe.select([], _trace_with_history(), seed=1)


# --------------------------------------------------------------------------- #
# probeVersion is reflected and changing it changes recorded identity
# --------------------------------------------------------------------------- #
def test_probe_version_exposed():
    for probe in PROBES.values():
        assert isinstance(probe.probe_version(), str)
        assert probe.probe_version() == probe.version


def test_changing_probe_version_changes_identity_and_tiebreak():
    # Construct two probes identical except for version. The recorded identity
    # (probe_version) differs, and on a fully-tied slate the seeded tie-break
    # uses the version, so selections can diverge.
    class _TiedProbeA(StrategyProbe):
        name = "TIED_TEST"
        version = "pA"

        def _score(self, card, trace):
            return 0.0  # everything tied -> seeded tie-break decides

    class _TiedProbeB(_TiedProbeA):
        version = "pB"

    a, b = _TiedProbeA(), _TiedProbeB()
    assert a.probe_version() == "pA"
    assert b.probe_version() == "pB"
    assert a.probe_version() != b.probe_version()

    # Build a slate large enough that the two version-seeded RNGs are very
    # likely to choose different cards; assert the seeded choices are each
    # deterministic and at least one version differs from the other.
    slate = [_card(f"c-{i}", "B1", "low", 0.5, [0.0]) for i in range(10)]
    sel_a = a.select(slate, None, seed=5)
    sel_b = b.select(slate, None, seed=5)
    # Deterministic per version:
    assert a.select(slate, None, seed=5) == sel_a
    assert b.select(slate, None, seed=5) == sel_b
    # The version genuinely participates in the seed -> different selection.
    assert sel_a != sel_b


# --------------------------------------------------------------------------- #
# GUARDRAIL: probes carry no latent/user field; select from observable fields
# --------------------------------------------------------------------------- #
def test_probes_carry_no_latent_user_state():
    latent_markers = set(FORBIDDEN_FIELDS) | {
        "demographic",
        "demographics",
        "persona",
        "emotion",
        "preference",
        "user_model",
        "user_id",
        "personality",
    }
    for probe in PROBES.values():
        attrs = set(vars(probe).keys()) | set(dir(probe))
        leaked = {a for a in attrs if a.lower() in latent_markers}
        assert not leaked, f"probe {probe.name} leaks latent state: {leaked}"


def test_probes_select_only_from_observable_card_fields():
    # A card carrying an extra forbidden/user field must not change selection:
    # probes read only observable fields, so the selection is identical whether
    # or not the latent field is present.
    observable = _crafted_slate()
    contaminated = []
    for c in observable:
        cc = dict(c)
        cc["persona"] = "SHOULD_BE_IGNORED"
        cc["user_id"] = "u-123"
        contaminated.append(cc)

    for name in PROBES:
        probe = get_probe(name)
        trace = _trace_with_history()
        clean = probe.select(observable, trace, seed=11)
        dirty = probe.select(contaminated, _trace_with_history(), seed=11)
        assert clean == dirty


# --------------------------------------------------------------------------- #
# Different probes can pick different cards on a crafted slate
# --------------------------------------------------------------------------- #
def test_different_probes_pick_different_cards():
    slate = _crafted_slate()
    trace = _trace_with_history()
    high = PreferHighComplexityProbe().select(slate, trace, seed=3)
    low_sal = PreferLowSalienceProbe().select(slate, trace, seed=3)
    novelty = PreferCoordNoveltyProbe().select(slate, trace, seed=3)

    assert high == "c-high"
    assert low_sal == "c-lowsal"
    assert novelty == "c-far"
    # All three differ on this crafted slate.
    assert len({high, low_sal, novelty}) == 3


def test_coord_novelty_empty_trace_is_deterministic_tiebreak():
    # With an empty trace there is no accumulated coordinate, so all cards tie
    # at score 0.0 and the seeded tie-break must still be deterministic.
    probe = PreferCoordNoveltyProbe()
    slate = _crafted_slate()
    sel1 = probe.select(slate, TraceState(), seed=8)
    sel2 = probe.select(slate, TraceState(), seed=8)
    assert sel1 == sel2
    assert sel1 in {c["cardId"] for c in slate}


def test_get_probe_unknown_raises():
    with pytest.raises(KeyError):
        get_probe("NOT_A_PROBE")
