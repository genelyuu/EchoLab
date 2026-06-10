"""Candidate archive builder for ECHO-Bench (Task A-008).

Builds a small candidate archive deterministically from a basis config plus a
base seed. Pipeline order (fixed): sample params -> render -> extract metrics ->
filter -> store -> hash. The archive is reproducible: the same config + seed
always yields the same ``archiveHash``. Only card records plus the archive hash
are stored — no personal / user / semantic metadata.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from echo_bench.basis.schema import BasisSpec
from echo_bench.cards.filter import is_degenerate
from echo_bench.cards.metrics import build_card
from echo_bench.cards.schema import Card
from echo_bench.logging import get_logger, log_ko
from echo_bench.utils.hash import canonical_hash

_logger = get_logger(__name__)


def _load_filter_thresholds(archive_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Load filter thresholds referenced by the archive config.

    Falls back to an empty-but-complete default set only if the referenced file
    is missing; normally the thresholds come from ``configs/archive/filter.yaml``.
    """
    rel = archive_cfg.get("filter_config", "configs/archive/filter.yaml")
    path = Path(rel)
    if not path.exists():
        raise FileNotFoundError(f"filter_config not found: {rel}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        doc = yaml.safe_load(text)
    except ModuleNotFoundError:
        from echo_bench.basis.schema import _minimal_yaml

        doc = _minimal_yaml(text)
    thresholds = doc.get("thresholds", doc) if isinstance(doc, Mapping) else {}
    return dict(thresholds)


def build_archive(
    bases: Mapping[str, BasisSpec],
    archive_cfg: Mapping[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    """Generate a reproducible candidate archive.

    Args:
        bases: Mapping ``basis_id -> BasisSpec`` (from ``load_bases``).
        archive_cfg: Parsed ``configs/archive/archive.yaml``
            (``pool_size``, ``candidates_to_generate``, ``filter_config``).
        base_seed: Base seed driving deterministic generation. Overrides the
            config default.

    Returns:
        ``{"cards": [card.to_dict(), ...], "archiveHash": <hex>}``. Rebuilding
        with the same config + seed yields an identical ``archiveHash``.
    """
    pool_size = int(archive_cfg.get("pool_size", 64))
    candidates = int(archive_cfg.get("candidates_to_generate", pool_size + 16))
    if candidates < pool_size:
        candidates = pool_size
    thresholds = _load_filter_thresholds(archive_cfg)

    basis_ids = sorted(bases.keys())  # deterministic, canonical basis order
    accepted: list[Card] = []
    rejected = 0

    # Deterministic generation: iterate candidate index in order, round-robin
    # across bases, deriving each card's param-RNG seed from the base seed.
    for idx in range(candidates):
        if len(accepted) >= pool_size:
            break
        basis = basis_ids[idx % len(basis_ids)]
        spec = bases[basis]
        # Per-candidate deterministic seeds derived from base_seed + idx.
        param_seed = canonical_hash({"base_seed": int(base_seed), "idx": idx,
                                     "basis": basis, "role": "params"})
        card_seed_h = canonical_hash({"base_seed": int(base_seed), "idx": idx,
                                      "basis": basis, "role": "seed"})
        rng = random.Random(int(param_seed[:16], 16))
        card_seed = int(card_seed_h[:12], 16)

        card = build_card(spec, basis, card_seed, rng)
        degenerate, reason = is_degenerate(card, thresholds)
        if degenerate:
            rejected += 1
            continue
        accepted.append(card)

    card_dicts = [c.to_dict() for c in accepted]
    archive_hash = canonical_hash(card_dicts)

    log_ko(
        _logger,
        f"아카이브 생성 완료: 채택={len(card_dicts)} 거부={rejected} "
        f"pool_size={pool_size} archiveHash={archive_hash[:12]}",
    )

    return {"cards": card_dicts, "archiveHash": archive_hash}
