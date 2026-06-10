"""Tests for echo_bench.cards.filter (Task A-007)."""

from __future__ import annotations

from echo_bench.cards.filter import (
    REASON_BLANK_LOW_STD,
    REASON_SATURATED,
    is_degenerate,
)
from echo_bench.cards.schema import Card

THRESHOLDS = {
    "min_std_intensity": 0.02,
    "min_dynamic_range": 0.05,
    "max_saturation_fraction": 0.85,
    "min_complexity_score": 0.05,
    "max_complexity_score": 0.99,
    "min_salience_score": 0.01,
}


def _card(vm, complexity=0.5, salience=0.3) -> Card:
    return Card(
        cardId="x",
        basis="B1",
        seed=1,
        params={"depth": 5.0},
        visualMetrics=vm,
        coordinateContribution=(0.1, 0.2, 0.3, 0.4),
        complexityScore=complexity,
        complexityBand="mid",
        salienceScore=salience,
        renderHash="h",
        rendererVersion="stub-r1",
    )


def _good_vm():
    return {
        "stdIntensity": 0.3,
        "dynamicRange": 0.8,
        "saturationFraction": 0.1,
    }


def test_accept_good_card():
    card = _card(_good_vm())
    degenerate, reason = is_degenerate(card, THRESHOLDS)
    assert degenerate is False
    assert reason is None


def test_reject_blank_low_std():
    vm = _good_vm()
    vm["stdIntensity"] = 0.0
    degenerate, reason = is_degenerate(_card(vm), THRESHOLDS)
    assert degenerate is True
    assert reason == REASON_BLANK_LOW_STD


def test_reject_saturated():
    vm = _good_vm()
    vm["saturationFraction"] = 0.95
    degenerate, reason = is_degenerate(_card(vm), THRESHOLDS)
    assert degenerate is True
    assert reason == REASON_SATURATED


def test_deterministic_reason():
    vm = _good_vm()
    vm["stdIntensity"] = 0.0
    r1 = is_degenerate(_card(vm), THRESHOLDS)
    r2 = is_degenerate(_card(vm), THRESHOLDS)
    assert r1 == r2
