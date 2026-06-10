"""Tests for the S3 coordinate-scramble runner (Task E-006)."""

from __future__ import annotations

import json

from echo_bench.experiments.s3_coordinate_scramble import (
    COORDINATE_POLICIES,
    S3_METRIC_KEYS,
    _REPORTS_DIR,
    run_s3_coordinate_scramble,
    scramble_coordinates,
    scramble_permutation,
)
from echo_bench.metrics.utility import CORE_METRIC_KEYS

# Small but valid parameters: n=4 seeds (so the round-0 divergence signal is
# stable), short H, 16-card pool, k=4. H=4 is in the horizon allowed set.
_KW = dict(base_seed=42, n=4, H=4, k=4, pool_size=16, scramble_seed=7)

_REQUIRED_HASHES = (
    "configHash",
    "archiveHash",
    "poolHash",
    "slateHash",
    "traceHash",
    "outputHash",
    "reportHash",
)

_FORBIDDEN_KEY_TOKENS = (
    "user_id",
    "persona",
    "emotion",
    "preference",
    "user_model",
)

_FORBIDDEN_CLAIM_PHRASES = (
    "wellbeing",
    "gdpr",
    "users prefer",
    "user satisfaction",
    "emotion",
    "personality",
    "diagnosis",
)


def _iter_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _iter_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_keys(item)


def _iter_string_values(obj, skip_keys=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            yield from _iter_string_values(v, skip_keys)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_string_values(item, skip_keys)
    elif isinstance(obj, str):
        yield obj


def _toy_pool(n=8):
    """A small pool with distinct cardIds and coordinate vectors."""
    return [
        {
            "cardId": f"C{i}",
            "basis": "B1",
            "complexityBand": "mid",
            "salienceScore": 0.5,
            "coordinateContribution": [float(i) / 10.0, float(i) / 20.0],
        }
        for i in range(n)
    ]


def test_s3_scramble_is_pure_deterministic_permutation():
    pool = _toy_pool(8)
    original = json.loads(json.dumps(pool))  # deep snapshot

    scr1 = scramble_coordinates(pool, 7)
    scr2 = scramble_coordinates(pool, 7)

    # Pure: input pool untouched.
    assert pool == original
    # Deterministic: identical (pool, seed) -> identical scrambled pool.
    assert scr1 == scr2

    # Every non-coordinate field is preserved exactly, in order.
    for src, out in zip(pool, scr1):
        assert out["cardId"] == src["cardId"]
        assert out["basis"] == src["basis"]
        assert out["complexityBand"] == src["complexityBand"]
        assert out["salienceScore"] == src["salienceScore"]

    # It is a true permutation: the MULTISET of coordinate vectors is preserved
    # exactly (structure unchanged), only the card->coordinate map is shuffled.
    src_coords = sorted(tuple(c["coordinateContribution"]) for c in pool)
    out_coords = sorted(tuple(c["coordinateContribution"]) for c in scr1)
    assert src_coords == out_coords

    # The permutation index list is a bijection over range(len(pool)).
    perm = scramble_permutation(pool, 7)
    assert sorted(perm) == list(range(len(pool)))
    # And it actually moves something (not the identity for this seed/pool).
    assert perm != list(range(len(pool)))

    # A different seed yields a different permutation (deterministic, seeded).
    assert scramble_permutation(pool, 7) != scramble_permutation(pool, 8)


def test_s3_dry_run_returns_hashes_writes_nothing():
    before = (
        set(_REPORTS_DIR.glob("s3_coordinate_scramble_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    result = run_s3_coordinate_scramble(dry_run=True, **_KW)

    assert result["dryRun"] is True
    for key in ("configHash", "archiveHash", "poolHash"):
        assert isinstance(result[key], str) and result[key]
    # The scramble permutation is logged into the (dry-run) plan.
    assert isinstance(result["scramblePermutation"], list)
    assert result["scramblePermutationHash"]
    after = (
        set(_REPORTS_DIR.glob("s3_coordinate_scramble_*.json"))
        if _REPORTS_DIR.exists()
        else set()
    )
    assert after == before


def test_s3_real_run_writes_report_with_hashes():
    report = run_s3_coordinate_scramble(dry_run=False, **_KW)

    for key in _REQUIRED_HASHES:
        assert isinstance(report[key], str) and report[key], key
    assert isinstance(report["seedBatchId"], str) and report["seedBatchId"]
    assert "reproducibilityPack" in report and "packHash" in report
    # The permutation is logged into the manifest/report (E-006 requirement).
    assert isinstance(report["scramblePermutation"], list)
    assert report["scramblePermutationHash"]

    out_path = (
        _REPORTS_DIR / f"s3_coordinate_scramble_{report['seedBatchId'][:12]}.json"
    )
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["reportHash"] == report["reportHash"]


def test_s3_metric_keys_pinned_to_core_and_recorded_in_report():
    """S3_METRIC_KEYS is pinned to CORE_METRIC_KEYS and recorded in the report.

    D-010 review: the scramble_shift denominator must be the original four
    utility keys (CORE_METRIC_KEYS), not the full seven after D-010 added the
    distribution metrics. The report is self-describing: its 'metricKeys' field
    records which keys were used.
    """
    # S3_METRIC_KEYS must equal CORE_METRIC_KEYS (not the full METRIC_KEYS).
    assert S3_METRIC_KEYS == CORE_METRIC_KEYS, (
        f"S3_METRIC_KEYS must be pinned to CORE_METRIC_KEYS={CORE_METRIC_KEYS!r}, "
        f"got {S3_METRIC_KEYS!r}"
    )
    assert len(S3_METRIC_KEYS) == 4

    report = run_s3_coordinate_scramble(dry_run=False, **_KW)
    # The report records which keys were used (self-describing).
    assert report["metricKeys"] == list(CORE_METRIC_KEYS), (
        f"S3 report['metricKeys'] must be {list(CORE_METRIC_KEYS)!r}, "
        f"got {report.get('metricKeys')!r}"
    )


def test_s3_coordinate_policies_shift_at_least_control():
    report = run_s3_coordinate_scramble(dry_run=False, **_KW)

    by_policy = {row["policy"]: row for row in report["table"]}
    control_div = report["controlDivergence"]

    # Direction (not magnitude): each coordinate-driven policy diverges at least
    # as much as the RANDOM control at round 0.
    for name in COORDINATE_POLICIES:
        assert by_policy[name]["round0_selection_divergence"] >= control_div
        assert report["perCoordinateDirection"][name] is True
    assert report["coordinatePoliciesShiftAtLeastControl"] is True

    # The control (RANDOM) is provably coordinate-invariant at round 0: it never
    # reads coordinateContribution and seeds from the (identical) pool hash.
    assert control_div == 0.0

    # Every metric value reported is bounded.
    for row in report["table"]:
        assert 0.0 <= row["scramble_shift"] <= 1.0
        assert 0.0 <= row["round0_selection_divergence"] <= 1.0


def test_s3_replay_identical_report_hash():
    r1 = run_s3_coordinate_scramble(dry_run=False, **_KW)
    r2 = run_s3_coordinate_scramble(dry_run=False, **_KW)
    assert r1["reportHash"] == r2["reportHash"]
    assert r1["traceHash"] == r2["traceHash"]
    assert r1["seedBatchId"] == r2["seedBatchId"]
    assert r1["scramblePermutationHash"] == r2["scramblePermutationHash"]


def test_s3_no_forbidden_fields_in_report():
    report = run_s3_coordinate_scramble(dry_run=False, **_KW)

    keys_lower = [str(k).lower() for k in _iter_keys(report)]
    for token in _FORBIDDEN_KEY_TOKENS:
        for key in keys_lower:
            assert token not in key, f"forbidden field key leaked: {key} ({token})"

    values_lower = [
        v.lower() for v in _iter_string_values(report, skip_keys=("phaseNote",))
    ]
    for token in _FORBIDDEN_CLAIM_PHRASES:
        for value in values_lower:
            assert token not in value, f"forbidden claim phrase: {value!r} ({token})"
