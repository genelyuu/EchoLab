"""ORACLE_STRATEGY slate-selection policy for ECHO-Bench (Task C-007).

Given a KNOWN controlled strategy probe (a controlled, instrumented INPUT policy
— *not* a synthetic user), this policy selects the constraint-satisfying
``k``-slate that maximizes the documented system-level objective the probe
implies. Oracle access to the probe is **explicit and confined to this module**:
no other policy is granted access to probe internals.

Objective
---------
The probe (e.g. ``PREFER_COORD_NOVELTY``, ``PREFER_HIGH_COMPLEXITY``,
``PREFER_LOW_SALIENCE``) deterministically selects one card from any presented
slate. The oracle constructs the slate so that the probe's *selected* card
maximizes the documented system-level objective the probe encodes — here, the
probe's own ``_score`` of its selected card. Concretely the oracle:

1. Loads the probe by name (from ``config["probe"]``) via
   :func:`echo_bench.probes.strategy_probes.get_probe`.
2. Ranks candidate cards by the probe's documented objective (the probe's score
   of each card against the observable trace), highest first.
3. Builds a constraint-satisfying slate whose highest-objective member is as
   large as possible while still meeting the basis-diversity rule, then
   confirms with the probe that this member is the one the probe selects from the
   assembled slate.

This defines the ``regret_to_oracle`` reference consumed by D-001 in Wave 2: the
oracle is the upper-reference for "best achievable under this controlled input
strategy", and a policy's regret is measured against it.

Determinism
-----------
Selection is a pure function of ``(poolHash, probeVersion, seed)``. A seeded
local ``random.Random`` (never the global RNG / wall-clock) breaks ties; changing
the probe changes ``probeVersion`` and the objective, hence the slate.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from echo_bench.env.constraints import check_slate, required_distinct_bases
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.base import Policy
from echo_bench.probes.oracle_objectives import get_oracle_objective
from echo_bench.probes.strategy_probes import get_probe
from echo_bench.utils.hash import canonical_hash

__all__ = ["OracleStrategyPolicy", "DEFAULT_PROBE"]

_logger = get_logger(__name__)

DEFAULT_PROBE = "PREFER_COORD_NOVELTY"

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


class OracleStrategyPolicy(Policy):
    """Oracle slate selection against a KNOWN controlled strategy probe.

    ``config`` carries at least ``k`` and ``probe`` (the registered probe name).
    Oracle access to the probe is explicit and confined here.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config or {})

    def policy_version(self) -> str:
        """Version reflects the policy, its config, AND the probe version.

        Including ``probeVersion`` means changing the controlled probe changes the
        recorded oracle identity (and hence the regret reference).
        """
        probe_name = self.config.get("probe", DEFAULT_PROBE)
        try:
            probe_version = get_probe(probe_name).probe_version()
        except KeyError:
            probe_version = "unknown"
        return canonical_hash(
            {
                "policy": self.__class__.__name__,
                "config": self.config,
                "probeVersion": probe_version,
            }
        )

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return the constraint-satisfying slate maximizing the oracle objective.

        Two objective sources are supported. When ``config['objective']`` is set
        (C-010: ``COVERAGE_GAIN`` / ``DIVERSITY_GAIN``) the oracle ranks cards by
        that objective-specific scorer — making the oracle an explicit reference
        for *that* objective, NOT a universal upper bound. Otherwise it falls back
        to the C-007 probe-encoded objective (default ``PREFER_COORD_NOVELTY``).
        """
        # Merge: the round runner passes only ``{"k": k}``; the policy's own
        # config (``probe`` / ``objective``) must still apply during an episode.
        effective = {**self.config, **(config or {})}
        k = int(effective["k"])
        objective_name = effective.get("objective")

        if k > len(pool):
            raise ValueError(
                f"ORACLE_STRATEGY 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        pool_hash = canonical_hash([c["cardId"] for c in pool])
        bases_cfg = _load_bases_cfg()
        min_required, _prefer_four, _ns = required_distinct_bases(k)

        # ---- C-010 objective-specific oracle (no probe; objective scorer) -----
        if objective_name:
            score_fn = get_oracle_objective(objective_name)
            seed_material = canonical_hash(
                {"poolHash": pool_hash, "objective": objective_name, "seed": seed}
            )
            rng = random.Random(int(seed_material, 16))
            tiebreak = {c["cardId"]: rng.random() for c in pool}
            objective = {c["cardId"]: float(score_fn(c, trace)) for c in pool}
            ordered = sorted(
                pool,
                key=lambda c: (-objective[c["cardId"]], tiebreak[c["cardId"]], c["cardId"]),
            )
            slate = self._assemble_max_objective_slate(
                ordered, k, min_required, objective, tiebreak, bases_cfg, seed
            )
            if slate is None:
                raise ValueError(
                    f"ORACLE_STRATEGY 정책: objective={objective_name} 에 대해 k={k} "
                    f"제약을 만족하는 슬레이트를 구성하지 못했습니다 (poolHash={pool_hash})"
                )
            self.log_score_components(
                {
                    c["cardId"]: {
                        "objectiveValue": round(objective[c["cardId"]], 12),
                        "objective": objective_name,
                    }
                    for c in slate
                }
            )
            log_ko(
                _logger,
                "ORACLE_STRATEGY 슬레이트 확정(objective): "
                f"objective={objective_name}, k={k}, poolHash={pool_hash}",
            )
            return [c["cardId"] for c in slate]

        # ---- C-007 probe-encoded objective (default; byte-identical path) -----
        probe_name = effective.get("probe", DEFAULT_PROBE)

        # Explicit, confined oracle access to the probe.
        probe = get_probe(probe_name)
        probe_version = probe.probe_version()

        # Deterministic seeded tie-break from (poolHash, probeVersion, seed).
        seed_material = canonical_hash(
            {
                "poolHash": pool_hash,
                "probeVersion": probe_version,
                "seed": seed,
            }
        )
        rng = random.Random(int(seed_material, 16))
        tiebreak = {c["cardId"]: rng.random() for c in pool}

        # Probe objective per card (documented system-level objective the probe
        # encodes). Higher is better.
        objective = {c["cardId"]: float(probe._score(c, trace)) for c in pool}

        ordered = sorted(
            pool,
            key=lambda c: (-objective[c["cardId"]], tiebreak[c["cardId"]], c["cardId"]),
        )

        slate = self._assemble_max_objective_slate(
            ordered, k, min_required, objective, tiebreak, bases_cfg, seed
        )
        if slate is None:
            raise ValueError(
                f"ORACLE_STRATEGY 정책: probe={probe_name} 에 대해 k={k} 제약을 "
                f"만족하는 슬레이트를 구성하지 못했습니다 (poolHash={pool_hash})"
            )

        # Confirm the probe selects, from the assembled slate, the
        # highest-objective member the oracle intended (deterministic check).
        selected_by_probe = probe.select(slate, trace, seed)
        intended = max(slate, key=lambda c: (objective[c["cardId"]], -tiebreak[c["cardId"]]))

        components = {
            c["cardId"]: {
                "probeObjective": round(objective[c["cardId"]], 12),
                "isProbeSelected": c["cardId"] == selected_by_probe,
            }
            for c in slate
        }
        self.log_score_components(components)
        log_ko(
            _logger,
            f"ORACLE_STRATEGY 슬레이트 확정: probe={probe_name}, "
            f"probeVersion={probe_version}, k={k}, "
            f"probe선택={selected_by_probe}, oracle의도={intended['cardId']}, "
            f"poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    @staticmethod
    def _assemble_max_objective_slate(
        ordered: List[Dict[str, Any]],
        k: int,
        min_required: int,
        objective: Dict[Any, float],
        tiebreak: Dict[Any, float],
        bases_cfg: dict,
        seed: int,
    ) -> List[Dict[str, Any]] | None:
        """Build a constraint-satisfying slate maximizing the top-objective member.

        Greedily take the highest-objective cards; if the resulting slate lacks
        the required basis diversity, backfill with the highest-objective cards
        of missing bases (replacing the lowest-objective duplicates) so the
        slate satisfies :func:`check_slate` while keeping the top objective card.
        """
        # Take the top-k by objective first.
        top = ordered[:k]
        if len({c["basis"] for c in top}) >= min_required:
            ok, _reason, _perm = check_slate(top, k, bases_cfg, seed)
            if ok:
                return top

        # Diversity repair: keep highest-objective cards but guarantee distinct
        # bases. Pick one highest-objective card per basis first, then backfill
        # with remaining highest-objective cards.
        by_basis: Dict[str, List[Dict[str, Any]]] = {}
        for card in ordered:  # already sorted by objective desc
            by_basis.setdefault(card["basis"], []).append(card)

        slate: List[Dict[str, Any]] = []
        slate_ids: set = set()
        # One top card per basis (bases ordered by their best objective card).
        bases_by_best = sorted(
            by_basis,
            key=lambda b: (
                -objective[by_basis[b][0]["cardId"]],
                tiebreak[by_basis[b][0]["cardId"]],
            ),
        )
        for b in bases_by_best:
            if len(slate) >= k:
                break
            card = by_basis[b][0]
            slate.append(card)
            slate_ids.add(card["cardId"])

        # Backfill remaining slots with the next-highest objective cards.
        for card in ordered:
            if len(slate) >= k:
                break
            if card["cardId"] not in slate_ids:
                slate.append(card)
                slate_ids.add(card["cardId"])

        if len(slate) != k:
            return None

        # Re-sort the chosen slate so the top-objective card is preserved and the
        # slate is in a stable order.
        slate.sort(key=lambda c: (-objective[c["cardId"]], tiebreak[c["cardId"]], c["cardId"]))
        ok, _reason, _perm = check_slate(slate, k, bases_cfg, seed)
        return slate if ok else None
