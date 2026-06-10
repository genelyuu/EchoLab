"""Tests for echo_bench.cards.schema (card record schema)."""

from __future__ import annotations

import pytest

from echo_bench.cards.schema import CARD_FIELDS, Card


def _sample_card() -> Card:
    return Card(
        cardId="abc123",
        basis="B1",
        seed=7,
        params={"depth": 5.0, "jitter": 0.1},
        visualMetrics={"meanIntensity": 0.5, "entropy": 0.4},
        coordinateContribution=(0.1, 0.2, 0.3, 0.4),
        complexityScore=0.5,
        complexityBand="mid",
        salienceScore=0.3,
        renderHash="deadbeef",
        rendererVersion="stub-r1",
    )


def test_exactly_eleven_fields():
    assert len(CARD_FIELDS) == 11


def test_round_trip():
    card = _sample_card()
    d = card.to_dict()
    assert set(d.keys()) == set(CARD_FIELDS)
    restored = Card.from_dict(d)
    assert restored == card


def test_to_dict_coordinate_is_list():
    card = _sample_card()
    d = card.to_dict()
    assert isinstance(d["coordinateContribution"], list)


def test_from_dict_rejects_extra_persona_key():
    card = _sample_card()
    d = card.to_dict()
    d["persona"] = "forbidden"
    with pytest.raises(ValueError):
        Card.from_dict(d)


def test_from_dict_rejects_user_id():
    card = _sample_card()
    d = card.to_dict()
    d["user_id"] = 42
    with pytest.raises(ValueError):
        Card.from_dict(d)


def test_from_dict_rejects_missing_field():
    card = _sample_card()
    d = card.to_dict()
    del d["salienceScore"]
    with pytest.raises(ValueError):
        Card.from_dict(d)
