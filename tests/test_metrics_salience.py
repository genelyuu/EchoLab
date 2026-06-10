"""Tests for echo_bench.metrics.salience (Task D-005).

Covers: thresholds loaded from config (not hard-coded); salience_outlier_rate
bounded [0, 1] and deterministic; salience_control bounded [0, 1] and
deterministic, zero when observed matches target; metadata carries traceHash.
"""

from __future__ import annotations

from echo_bench.env.trace_state import TraceState
from echo_bench.metrics.salience import (
    SALIENCE_AUDIT_CONFIG_PATH,
    load_salience_config,
    salience_control,
    salience_control_with_metadata,
    salience_outlier_rate,
    salience_outlier_rate_with_metadata,
)


def _record(salience: float) -> dict:
    return {
        "candidatePoolHash": "poolhash-abc",
        "slate": ["c1", "c2", "c3", "c4"],
        "selectedCardId": "c1",
        "coordinateContribution": [0.1, 0.2, 0.3, 0.4],
        "complexityBand": "mid",
        "salienceScore": salience,
        "slotPermutation": [0, 1, 2, 3],
    }


def _trace(saliences: list) -> TraceState:
    t = TraceState()
    for s in saliences:
        t.append_round(_record(s))
    return t


def test_config_file_loads():
    cfg = load_salience_config()
    assert SALIENCE_AUDIT_CONFIG_PATH.exists()
    assert "outlier_threshold" in cfg
    assert "target_distribution" in cfg


def test_outlier_rate_bounded_and_uses_config_threshold():
    cfg = load_salience_config()  # outlier_threshold == 0.85
    # 2 of 4 strictly exceed 0.85.
    trace = _trace([0.90, 0.95, 0.50, 0.10])
    v = salience_outlier_rate(trace, cfg)
    assert 0.0 <= v <= 1.0
    assert abs(v - 0.5) < 1e-12


def test_outlier_rate_threshold_from_config_not_hardcoded():
    trace = _trace([0.40, 0.60, 0.80])
    # With a low configured threshold, all three count.
    assert salience_outlier_rate(trace, {"outlier_threshold": 0.30}) == 1.0
    # With a high configured threshold, none count.
    assert salience_outlier_rate(trace, {"outlier_threshold": 0.95}) == 0.0


def test_outlier_rate_empty_trace_is_zero():
    assert salience_outlier_rate(_trace([]), {"outlier_threshold": 0.5}) == 0.0


def test_outlier_rate_deterministic():
    cfg = load_salience_config()
    trace = _trace([0.9, 0.1, 0.95])
    assert salience_outlier_rate(trace, cfg) == salience_outlier_rate(trace, cfg)


def test_salience_control_bounded():
    cfg = load_salience_config()
    trace = _trace([0.9, 0.95, 0.92, 0.88])  # concentrated in high band
    v = salience_control(trace, cfg)
    assert 0.0 <= v <= 1.0
    assert v > 0.0  # diverges from the balanced target


def test_salience_control_zero_when_matching_target():
    # Target distribution from config: 3 balanced bands [0,.33),[.33,.66),[.66,1].
    cfg = load_salience_config()
    # 1 per band, equally -> observed [1/3,1/3,1/3] vs target ~[.34,.33,.33].
    trace = _trace([0.10, 0.50, 0.90])
    v = salience_control(trace, cfg)
    # Very close to zero (target weights are ~balanced).
    assert v < 0.02


def test_salience_control_perfect_match_is_zero():
    # Construct a target that exactly matches a single-band observation.
    target_cfg = {
        "target_distribution": {
            "bands": [
                {"lo": 0.0, "hi": 0.5},
                {"lo": 0.5, "hi": 1.0},
            ],
            "weights": [1.0, 0.0],
        }
    }
    trace = _trace([0.1, 0.2, 0.3])  # all in band 0
    assert salience_control(trace, target_cfg) == 0.0


def test_salience_control_empty_trace_is_zero():
    cfg = load_salience_config()
    assert salience_control(_trace([]), cfg) == 0.0


def test_salience_control_deterministic():
    cfg = load_salience_config()
    trace = _trace([0.2, 0.9, 0.5, 0.95])
    assert salience_control(trace, cfg) == salience_control(trace, cfg)


def test_metadata_carries_trace_hash():
    cfg = load_salience_config()
    trace = _trace([0.9, 0.1])
    md_out = salience_outlier_rate_with_metadata(trace, cfg)
    md_ctl = salience_control_with_metadata(trace, cfg)
    assert md_out["traceHash"] == trace.trace_hash()
    assert md_ctl["traceHash"] == trace.trace_hash()
    assert md_out["metric"] == "salience_outlier_rate"
    assert md_ctl["metric"] == "salience_control"
