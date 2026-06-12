"""AXS_UCB policy family for ECHO-Bench (Task AXS-P0 T1).

Preregistered mechanism-experiment policy arms. This module contains:

- :class:`TraceView`: lightweight wrapper that lets a truncated list of round
  records satisfy the same ``.rounds()`` / ``.trace_hash()`` interface as
  :class:`~echo_bench.env.trace_state.TraceState`.
- :class:`AxsUcbPolicy`: subclass of
  :class:`~echo_bench.policies.trace_lin_ucb.TraceLinUcbPolicy` with three
  independently-controllable experimental seams: ``alpha``, ``freeze_round``,
  and ``tie_break_order``.
- :class:`AxsYokedBonusPolicy`: extends ``AxsUcbPolicy`` with a schedule-driven
  exploration bonus whose direction vector carries zero history information
  (trace-independent seed material).

All context features are computed EXCLUSIVELY from observable trace fields and
observable card fields. No latent user vector, persona, emotion, preference,
demographic, or free-text field is read or stored by any class in this module.

Guardrail: TRACE-ONLY. This module never imports
:mod:`echo_bench.policies.pseudo_user_model` and never reads or constructs any
latent / persona / emotion / preference / demographic / free-text field.

All identifiers stay English; runtime log messages are Korean per convention.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from echo_bench.env.constraints import check_slate
from echo_bench.logging import get_logger, log_ko
from echo_bench.policies.trace_lin_ucb import (
    BAND_ORDER,
    DEFAULT_ALPHA,
    DEFAULT_FEATURES,
    DEFAULT_LAMBDA,
    TraceLinUcbPolicy,
    _load_bases_cfg,
)
from echo_bench.utils.hash import canonical_hash

__all__ = ["TraceView", "AxsUcbPolicy", "AxsYokedBonusPolicy"]

_logger = get_logger(__name__)

# Allowed tie_break_order values (string identifiers — English).
_ALLOWED_TIE_BREAK_ORDERS = frozenset(
    {
        "canonical",
        "reverse",
        "hash_seeded",
        "feature_lexicographic",
        # v2 trace-free tie-break variants (AXS-ENTRY-001).
        # Seed material: poolHash, seed, roundIndex, policyVersion, salt.
        # No traceHash — the tiebreak carries zero history information.
        "trace_free",
        "trace_free_alt1",
    }
)

# Repo root: src/echo_bench/policies/axs_ucb.py -> parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _validate_freeze_round(freeze_round: Any, policy_label: str) -> None:
    """Raise Korean ValueError if freeze_round is not None or a non-negative int.

    Booleans are rejected (True/False are int subclasses but not valid rounds).
    Float values such as 2.5 are rejected. Only None or a plain non-negative int
    is allowed.
    """
    if freeze_round is None:
        return
    if (
        not isinstance(freeze_round, int)
        or isinstance(freeze_round, bool)
        or freeze_round < 0
    ):
        raise ValueError(
            f"{policy_label}: freeze_round 값이 유효하지 않습니다 "
            f"({freeze_round!r}). None 또는 0 이상의 정수만 허용합니다."
        )


def _validate_ctx_freeze_round(ctx_freeze_round: Any, policy_label: str) -> None:
    """Raise Korean ValueError if ctx_freeze_round is not None or a non-negative int.

    Booleans are rejected (True/False are int subclasses but not valid rounds).
    Float values such as 2.5 are rejected. Only None or a plain non-negative int
    is allowed. This validator is analogous to _validate_freeze_round but applies
    to the ctx_freeze_round seam (context-feature freeze) and emits a distinct
    message naming the key.
    """
    if ctx_freeze_round is None:
        return
    if (
        not isinstance(ctx_freeze_round, int)
        or isinstance(ctx_freeze_round, bool)
        or ctx_freeze_round < 0
    ):
        raise ValueError(
            f"{policy_label}: ctx_freeze_round 값이 유효하지 않습니다 "
            f"({ctx_freeze_round!r}). None 또는 0 이상의 정수만 허용합니다."
        )


# ---------------------------------------------------------------------------
# TraceView
# ---------------------------------------------------------------------------


class TraceView:
    """Lightweight wrapper exposing a list of round-record dicts as a trace.

    Satisfies the same ``.rounds()`` / ``.trace_hash()`` interface as
    :class:`~echo_bench.env.trace_state.TraceState` so that a truncated history
    slice can be passed to ``_replay_bandit_state`` without modifying the parent
    class.
    """

    def __init__(self, records: List[dict]) -> None:
        self._records = list(records)

    def rounds(self) -> List[dict]:
        """Return a copied list of the stored round records."""
        return [dict(r) for r in self._records]

    def trace_hash(self) -> str:
        """Return a deterministic hash over the stored round records."""
        return canonical_hash(self._records)


# ---------------------------------------------------------------------------
# AxsUcbPolicy
# ---------------------------------------------------------------------------


class AxsUcbPolicy(TraceLinUcbPolicy):
    """Trace-only linear-UCB contextual bandit with AXS experimental seams.

    Inherits feature extraction and bandit-state replay from
    :class:`TraceLinUcbPolicy` but implements its own ``select()`` and
    diversity-fallback so that its behaviour is frozen independently of the
    parent class (C-011 FREEZE).

    Config keys
    -----------
    k : int
        Slate size.
    alpha : float, default 1.0
        UCB exploration coefficient. ``alpha=0.0`` reduces to mean-only ranking.
    lambda_reg : float, default 1.0
        Ridge regularizer seeding ``A = lambda_reg * I``.
    features : list[str], optional
        Ordered active feature blocks (defaults to ``DEFAULT_FEATURES``).
    freeze_round : int or None, default None
        When set, the bandit state ``(A, b)`` is built from only the first
        ``freeze_round`` rounds; live context features and tiebreak seed
        material still use the full trace.
    ctx_freeze_round : int or None, default None
        When set to ``f``, the live-context computation (centroid and
        ``max_band``, used to derive ``target_band_idx``) uses only the first
        ``f`` rounds: ``TraceView(trace.rounds()[:f])`` is passed to
        ``_trace_centroid_and_band`` instead of the live trace. All other
        computations — bandit-state replay (governed by ``freeze_round``
        independently) and canonical tie-break seed material (traceHash is
        always taken from the LIVE trace) — remain unchanged. The two seams
        are **orthogonal** and may be set independently. ``f=0`` produces a
        probe-blind context: the empty-trace default centroid/band is used
        every round.
    tie_break_order : str, default "canonical"
        Alters only the tie components (positions 3–4) of the rank-key tuple.
        Allowed: ``"canonical"``, ``"reverse"``, ``"hash_seeded"``,
        ``"feature_lexicographic"``, ``"trace_free"``, ``"trace_free_alt1"``.
        The ``trace_free`` and ``trace_free_alt1`` variants seed the tiebreak
        RNG from ``{poolHash, seed, roundIndex, policyVersion, salt}`` with no
        ``traceHash``, so the tie-break carries zero history information.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate by trace-only linear UCB (AXS variant).

        TraceLinUcbPolicy.select() 사본, 기준 커밋 a1a8b6f
        """
        # SEAM 1: effective config resolution
        # run_round (round_runner.py:96) 는 policy.select(pool, trace, seed, {"k": k}) 형태로
        # 호출한다. per-call config 를 그대로 대입하면 생성자 config 의 alpha/freeze_round 등
        # 실험 심(seam) 값이 모두 소실되어 코드 기본값으로 폴백된다(config-drop 버그).
        # 수정: 생성자 config 를 베이스로 두고 per-call 키만 덮어써서 심을 보존한다.
        effective = {**self.config, **(config or {})}
        k = int(effective["k"])
        alpha = float(effective.get("alpha", DEFAULT_ALPHA))
        lambda_reg = float(effective.get("lambda_reg", DEFAULT_LAMBDA))
        active = tuple(effective.get("features") or DEFAULT_FEATURES)
        freeze_round: Optional[int] = effective.get("freeze_round", None)
        ctx_freeze_round: Optional[int] = effective.get("ctx_freeze_round", None)
        tie_break_order: str = str(effective.get("tie_break_order", "canonical"))

        # SEAM 2: freeze_round and ctx_freeze_round validation
        _validate_freeze_round(freeze_round, "AXS_UCB 정책")
        _validate_ctx_freeze_round(ctx_freeze_round, "AXS_UCB 정책")

        if tie_break_order not in _ALLOWED_TIE_BREAK_ORDERS:
            raise ValueError(
                f"AXS_UCB 정책: 알 수 없는 tie_break_order 값 {tie_break_order!r} "
                f"(허용: {sorted(_ALLOWED_TIE_BREAK_ORDERS)})"
            )

        if k > len(pool):
            raise ValueError(
                f"AXS_UCB 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        # Build bandit state — optionally frozen to first freeze_round rounds.
        if freeze_round is not None:
            bandit_trace: Any = TraceView(trace.rounds()[:freeze_round])
        else:
            bandit_trace = trace

        A, b = self._replay_bandit_state(bandit_trace, active, lambda_reg)
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b

        # Context features (centroid/band) — Seam: ctx_freeze_round.
        # When ctx_freeze_round is set, the context view is frozen to the first
        # ctx_freeze_round rounds; the bandit state (above) is governed
        # independently by freeze_round. The two seams are orthogonal.
        # IMPORTANT: canonical tie-break seed material always uses the LIVE
        # traceHash regardless of ctx_freeze_round (the seams stay orthogonal).
        if ctx_freeze_round is not None:
            context_trace: Any = TraceView(trace.rounds()[:ctx_freeze_round])
        else:
            context_trace = trace
        centroid, max_band = self._trace_centroid_and_band(context_trace)
        target_band_idx = (
            min(max_band + 1, len(BAND_ORDER) - 1) if max_band >= 0 else 0
        )

        # Tiebreak RNG — seed material depends on tie_break_order mode:
        # - canonical/reverse/hash_seeded: includes traceHash from LIVE trace.
        # - trace_free/trace_free_alt1: no traceHash; uses roundIndex (=len of
        #   LIVE trace.rounds()) so floats vary per round.
        # - feature_lexicographic: no RNG needed (uses feat_key + cid).
        # NOTE: traceHash for tie-break seed is ALWAYS from the live trace,
        # even when ctx_freeze_round is set. ctx_freeze_round only affects
        # context features, not the tie-break seed material.
        pool_hash = canonical_hash([c["cardId"] for c in pool])
        tiebreak: Dict[str, float] = {}
        if tie_break_order == "trace_free":
            tf_seed_material: Dict[str, Any] = {
                "poolHash": pool_hash,
                "seed": seed,
                "roundIndex": len(trace.rounds()),
                "policyVersion": self.policy_version(),
                "salt": "axs-v2-trace-free-tb-v1",
            }
            tf_seed_hex = canonical_hash(tf_seed_material)
            rng = random.Random(int(tf_seed_hex, 16))
            tiebreak = {c["cardId"]: rng.random() for c in pool}
        elif tie_break_order == "trace_free_alt1":
            tf_alt1_seed_material: Dict[str, Any] = {
                "poolHash": pool_hash,
                "seed": seed,
                "roundIndex": len(trace.rounds()),
                "policyVersion": self.policy_version(),
                "salt": "axs-v2-trace-free-tb-alt1",
            }
            tf_alt1_seed_hex = canonical_hash(tf_alt1_seed_material)
            rng = random.Random(int(tf_alt1_seed_hex, 16))
            tiebreak = {c["cardId"]: rng.random() for c in pool}
        elif tie_break_order != "feature_lexicographic":
            base_seed_material: Dict[str, Any] = {
                "poolHash": pool_hash,
                "traceHash": trace.trace_hash(),
                "seed": seed,
                "policyVersion": self.policy_version(),
            }
            if tie_break_order == "hash_seeded":
                base_seed_material["salt"] = "axs010-hash-seeded-v1"
            seed_hex = canonical_hash(base_seed_material)
            rng = random.Random(int(seed_hex, 16))
            tiebreak = {c["cardId"]: rng.random() for c in pool}

        bases_cfg = _load_bases_cfg()

        # Compute UCB components for every card; cache feature vectors for
        # feature_lexicographic mode.
        feature_cache: Dict[str, np.ndarray] = {}

        def ucb_components(card: Dict[str, Any]) -> Dict[str, float]:
            x = self._features(
                card["coordinateContribution"],
                card["complexityBand"],
                centroid,
                target_band_idx,
                active,
            )
            feature_cache[card["cardId"]] = x
            mean = float(theta @ x)
            bonus = float(np.sqrt(max(0.0, x @ (A_inv @ x))))
            return {
                "mean": round(mean, 12),
                "bonus": round(bonus, 12),
                "ucb": round(mean + alpha * bonus, 12),
            }

        comp_cache = {c["cardId"]: ucb_components(c) for c in pool}

        # SEAM 3: tie_break_order rank-key dispatch
        def _rank_key(card: Dict[str, Any], need_new_basis: bool, chosen_bases: set) -> tuple:
            comp = comp_cache[card["cardId"]]
            cid = card["cardId"]
            new_basis_bonus = (
                1 if (need_new_basis and card["basis"] not in chosen_bases) else 0
            )
            primary = (-new_basis_bonus, -comp["ucb"])
            if tie_break_order == "canonical":
                return primary + (tiebreak[cid], cid)
            elif tie_break_order == "reverse":
                # NOTE: desc_key uses -ord(ch) per character. For very short IDs
                # that share all characters except length (e.g. "ab" vs "abc"),
                # the ordering is identical in both directions — this is a known
                # prefix-length ambiguity; document here so it is not silently
                # relied upon.
                neg_tb = -tiebreak[cid]
                desc_key = tuple(-ord(ch) for ch in cid)
                return primary + (neg_tb, desc_key)
            elif tie_break_order == "hash_seeded":
                return primary + (tiebreak[cid], cid)
            elif tie_break_order == "feature_lexicographic":
                x_card = feature_cache[cid]
                feat_key = tuple(round(float(v), 12) for v in x_card)
                return primary + (feat_key, cid)
            elif tie_break_order in ("trace_free", "trace_free_alt1"):
                # trace_free / trace_free_alt1: same rank-key structure as
                # canonical (primary + float + cid) but tiebreak floats are
                # seeded without traceHash (computed above).
                return primary + (tiebreak[cid], cid)
            else:
                # Should not reach here — checked above.
                raise ValueError(
                    f"AXS_UCB 정책: 알 수 없는 tie_break_order 값 {tie_break_order!r}"
                )

        slate = self._assemble_slate(
            pool, comp_cache, feature_cache, tiebreak, _rank_key,
            k, bases_cfg, seed, tie_break_order, pool_hash,
        )

        if slate is None:
            raise ValueError(
                f"AXS_UCB 정책: k={k} 제약을 만족하는 슬레이트를 구성하지 "
                f"못했습니다 (poolHash={pool_hash})"
            )

        self.log_score_components(
            {
                c["cardId"]: {
                    "mean": comp_cache[c["cardId"]]["mean"],
                    "bonus": comp_cache[c["cardId"]]["bonus"],
                }
                for c in slate
            }
        )
        log_ko(
            _logger,
            f"AXS_UCB 슬레이트 확정: k={k}, alpha={alpha}, "
            f"tie_break_order={tie_break_order}, "
            f"freeze_round={freeze_round}, ctx_freeze_round={ctx_freeze_round}, "
            f"traceLen={len(trace.rounds())}, poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]

    def _assemble_slate(
        self,
        pool: List[Dict[str, Any]],
        comp_cache: Dict[str, Dict[str, float]],
        feature_cache: Dict[str, np.ndarray],
        tiebreak: Dict[str, float],
        rank_key_fn: Callable,
        k: int,
        bases_cfg: dict,
        seed: int,
        tie_break_order: str,
        pool_hash: str,
    ) -> List[Dict[str, Any]] | None:
        """Greedy slate assembly shared by AxsUcbPolicy and AxsYokedBonusPolicy.

        Runs the greedy loop with constraint check and diversity fallback.
        Returns the assembled list of card dicts, or None if no valid slate can
        be constructed.

        Parameters
        ----------
        pool:
            Full candidate card list.
        comp_cache:
            Pre-computed score components (mean/bonus/ucb) keyed by cardId.
        feature_cache:
            Pre-computed feature vectors keyed by cardId (used by fallback).
        tiebreak:
            Per-card deterministic tiebreak floats (empty for
            feature_lexicographic mode).
        rank_key_fn:
            Callable ``(card, need_new_basis, chosen_bases) -> tuple`` producing
            the sort key for the greedy step.
        k:
            Target slate size.
        bases_cfg:
            Loaded bases configuration dict (passed to check_slate).
        seed:
            Round seed (passed to check_slate).
        tie_break_order:
            Active tiebreak mode (passed to fallback).
        pool_hash:
            Pre-computed pool hash (used in error message).
        """
        chosen: List[Dict[str, Any]] = []
        chosen_ids: set = set()
        chosen_bases: set = set()

        while len(chosen) < k:
            pool_left = [c for c in pool if c["cardId"] not in chosen_ids]
            if not pool_left:
                break
            slots_left = k - len(chosen)
            distinct_bases_left = len({c["basis"] for c in pool_left} - chosen_bases)
            need_new_basis = (
                len(chosen_bases) < 3
                and slots_left <= (3 - len(chosen_bases))
                and distinct_bases_left > 0
            )

            pool_left.sort(key=lambda c: rank_key_fn(c, need_new_basis, chosen_bases))
            pick = pool_left[0]
            chosen.append(pick)
            chosen_ids.add(pick["cardId"])
            chosen_bases.add(pick["basis"])

        slate = chosen if len(chosen) == k else None
        if slate is not None:
            ok, _reason, _perm = check_slate(slate, k, bases_cfg, seed)
            if not ok:
                slate = self._axs_diversity_fallback(
                    pool, k, comp_cache, tiebreak, feature_cache, bases_cfg, seed,
                    tie_break_order,
                )

        return slate

    def _axs_diversity_fallback(
        self,
        pool: List[Dict[str, Any]],
        k: int,
        comp_cache: Dict[Any, Dict[str, float]],
        tiebreak: Dict[Any, float],
        feature_cache: Dict[str, np.ndarray],
        bases_cfg: dict,
        seed: int,
        tie_break_order: str,
    ) -> List[Dict[str, Any]] | None:
        """AXS-local diversity fallback (frozen copy of parent logic with AXS tie-break)."""

        def fallback_key(c: Dict[str, Any]) -> tuple:
            cid = c["cardId"]
            ucb_neg = -comp_cache[cid]["ucb"]
            if tie_break_order in ("canonical", "hash_seeded", "trace_free", "trace_free_alt1"):
                return (ucb_neg, tiebreak[cid], cid)
            elif tie_break_order == "reverse":
                neg_tb = -tiebreak[cid]
                desc_key = tuple(-ord(ch) for ch in cid)
                return (ucb_neg, neg_tb, desc_key)
            elif tie_break_order == "feature_lexicographic":
                # Use direct index — feature_cache is always populated for all
                # pool cards before _assemble_slate is called.
                x_card = feature_cache[cid]
                feat_key = tuple(round(float(v), 12) for v in x_card)
                return (ucb_neg, feat_key, cid)
            else:
                return (ucb_neg, tiebreak[cid], cid)

        ordered = sorted(pool, key=fallback_key)
        by_basis: Dict[str, List[Dict[str, Any]]] = {}
        for card in ordered:
            by_basis.setdefault(card["basis"], []).append(card)
        diverse: List[Dict[str, Any]] = []
        diverse_ids: set = set()
        for cards in by_basis.values():
            diverse.append(cards[0])
            diverse_ids.add(cards[0]["cardId"])
            if len(diverse) >= k:
                break
        for card in ordered:
            if len(diverse) >= k:
                break
            if card["cardId"] not in diverse_ids:
                diverse.append(card)
                diverse_ids.add(card["cardId"])
        if len(diverse) != k:
            return None
        ok, _reason, _perm = check_slate(diverse, k, bases_cfg, seed)
        return diverse if ok else None


# ---------------------------------------------------------------------------
# AxsYokedBonusPolicy
# ---------------------------------------------------------------------------


class AxsYokedBonusPolicy(AxsUcbPolicy):
    """AXS_UCB variant with a schedule-driven, history-free exploration bonus.

    The bonus direction ``u(card) ∈ [0,1)`` is seeded from pool+seed+round-index
    only — **no traceHash** — so the exploration direction carries zero history
    information. The bonus scale ``B_t`` comes from ``perRoundBonus[t]`` (clamped
    to the last entry when ``t >= len(perRoundBonus)``).

    Scoring: ``score = mean + alpha * B_t * u(card)`` replacing the parent bonus
    term. The mean term remains history-dependent (deliberate — the contrast
    isolates the bonus coupling).

    Config keys (in addition to AxsUcbPolicy's keys)
    -------------------------------------------------
    schedule_path : str
        Path to the schedule JSON. Relative paths are resolved against the repo
        root (``Path(__file__).resolve().parents[3]``). Empty/blank paths are
        explicitly rejected before filesystem access.
    schedule_hash : str
        Expected ``scheduleHash`` string. Must match the recomputed hash of the
        schedule body (with ``"scheduleHash"`` key excluded).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._schedule: Optional[dict] = None  # loaded lazily
        tbo = config.get("tie_break_order", "canonical")
        if tbo not in (None, "canonical"):
            raise ValueError(
                f"AXS_YOKED 정책: tie_break_order='{tbo}' 은 허용되지 않습니다. "
                "사전등록된 yoked arm은 'canonical' 만 지원합니다."
            )
        # ctx_freeze_round is v2-only; AxsYokedBonusPolicy is a v1-only arm.
        # Reject it at init to fail closed.
        if "ctx_freeze_round" in config:
            raise ValueError(
                "AXS_YOKED 정책: ctx_freeze_round 는 v1 yoked arm에서 허용되지 않습니다. "
                "v2 실험에는 AxsUcbPolicy 를 사용하십시오."
            )

    def policy_version(self) -> str:
        """Override: hash config with schedule_path REPLACED by schedule_hash.

        Path-independence fix: identity is content (schedule_hash), not location.
        Same schedule content at two different paths yields identical policy_version
        and identical slate (AXS-P0 T4 리뷰 반영).
        """
        config_for_version = dict(self.config)
        # Replace schedule_path with schedule_hash in the version hash material.
        # This ensures the version is location-independent: same content -> same version.
        config_for_version.pop("schedule_path", None)
        # schedule_hash already in config (required) — it pins content identity
        return canonical_hash(
            {
                "policy": self.__class__.__name__,
                "config": config_for_version,
            }
        )

    def _load_and_verify_schedule(self) -> dict:
        """Load, verify, and cache the yoked schedule JSON.

        Raises
        ------
        ValueError
            If the schedule_path is empty/blank, the file is missing, not valid
            JSON, the hash does not match, or the ``perRoundBonus`` field is
            missing / empty / contains invalid values.
        """
        if self._schedule is not None:
            return self._schedule

        raw_path: str = self.config.get("schedule_path", "")

        # Explicit rejection of empty/blank path before filesystem access.
        if not raw_path or not raw_path.strip():
            raise ValueError(
                "AXS_YOKED 정책: schedule_path 가 비어 있습니다. "
                "유효한 파일 경로를 지정해야 합니다."
            )

        schedule_path = Path(raw_path)
        if not schedule_path.is_absolute():
            schedule_path = _REPO_ROOT / schedule_path

        expected_from_config: str = str(self.config.get("schedule_hash", ""))

        try:
            text = schedule_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ValueError(
                f"AXS_YOKED 정책: 스케줄 파일을 찾을 수 없습니다: {schedule_path}"
            )
        except OSError as exc:
            raise ValueError(
                f"AXS_YOKED 정책: 스케줄 파일 읽기 오류: {schedule_path} — {exc}"
            )

        try:
            body: dict = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"AXS_YOKED 정책: 스케줄 파일 JSON 파싱 오류: {schedule_path} — {exc}"
            )

        # Extract and remove embedded scheduleHash for hash recomputation.
        embedded_hash: str = str(body.get("scheduleHash", ""))
        body_without_hash = {k: v for k, v in body.items() if k != "scheduleHash"}

        recomputed = canonical_hash(body_without_hash)

        if recomputed != embedded_hash:
            raise ValueError(
                f"AXS_YOKED 정책: 스케줄 파일 내장 scheduleHash 불일치 "
                f"(재계산={recomputed!r}, 파일내={embedded_hash!r})"
            )
        if recomputed != expected_from_config:
            raise ValueError(
                f"AXS_YOKED 정책: config schedule_hash 와 파일 scheduleHash 불일치 "
                f"(config={expected_from_config!r}, 파일={recomputed!r})"
            )

        # Validate perRoundBonus structure after hash check.
        per_round_bonus = body.get("perRoundBonus")
        if per_round_bonus is None:
            raise ValueError(
                f"AXS_YOKED 정책: 스케줄 파일에 'perRoundBonus' 키가 없습니다: "
                f"{schedule_path}"
            )
        if not isinstance(per_round_bonus, list) or len(per_round_bonus) == 0:
            raise ValueError(
                f"AXS_YOKED 정책: 'perRoundBonus' 가 비어 있거나 리스트가 아닙니다: "
                f"{schedule_path}"
            )
        for idx, val in enumerate(per_round_bonus):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"AXS_YOKED 정책: 'perRoundBonus[{idx}]' 값이 유효한 숫자가 "
                    f"아닙니다 ({val!r}): {schedule_path}"
                )
            if not math.isfinite(float(val)):
                raise ValueError(
                    f"AXS_YOKED 정책: 'perRoundBonus[{idx}]' 값이 유한하지 않습니다 "
                    f"({val!r}): {schedule_path}"
                )

        self._schedule = body
        return body

    def select(
        self,
        pool: List[Dict[str, Any]],
        trace: Any,
        seed: int,
        config: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return a constraint-satisfying slate using a yoked bonus schedule.

        TraceLinUcbPolicy.select() 사본, 기준 커밋 a1a8b6f
        """
        # SEAM 1: effective config resolution + per-call consistency check
        # run_round (round_runner.py:96) 는 policy.select(pool, trace, seed, {"k": k}) 형태로
        # 호출한다. per-call config 를 그대로 대입하면 생성자 config 의 alpha/freeze_round 등
        # 실험 심(seam) 값이 모두 소실되어 코드 기본값으로 폴백된다(config-drop 버그).
        # 수정: 생성자 config 를 베이스로 두고 per-call 키만 덮어써서 심을 보존한다.
        effective = {**self.config, **(config or {})}

        # Explicit rejection of per-call config keys that would silently alter
        # preregistered yoked-arm behavior.
        # 수정(merge 이후): per-call 에 동일 값이 들어오면 무해하므로 허용;
        # self.config 값과 다른 경우에만 거부한다(의도 변경 봉쇄).
        if config is not None:
            for forbidden_key in ("schedule_path", "schedule_hash"):
                if forbidden_key in config:
                    # per-call 값이 생성자 값과 다르면 거부.
                    if config[forbidden_key] != self.config.get(forbidden_key):
                        raise ValueError(
                            f"AXS_YOKED 정책: per-call config 에 '{forbidden_key}' 키가 "
                            "포함되어 있습니다. yoked arm은 생성자 config에서만 "
                            "스케줄을 읽습니다."
                        )
            per_call_tbo = config.get("tie_break_order")
            if per_call_tbo is not None and per_call_tbo != self.config.get("tie_break_order", "canonical"):
                raise ValueError(
                    f"AXS_YOKED 정책: per-call config의 tie_break_order="
                    f"{per_call_tbo!r} 는 허용되지 않습니다. "
                    "yoked arm은 'canonical' 만 지원합니다."
                )
            if "ctx_freeze_round" in config:
                raise ValueError(
                    "AXS_YOKED 정책: per-call config 에 ctx_freeze_round 키가 "
                    "포함되어 있습니다. v1 yoked arm은 ctx_freeze_round 를 "
                    "지원하지 않습니다."
                )

        k = int(effective["k"])
        alpha = float(effective.get("alpha", DEFAULT_ALPHA))
        lambda_reg = float(effective.get("lambda_reg", DEFAULT_LAMBDA))
        active = tuple(effective.get("features") or DEFAULT_FEATURES)
        freeze_round: Optional[int] = effective.get("freeze_round", None)

        # SEAM 2: freeze_round validation
        _validate_freeze_round(freeze_round, "AXS_YOKED 정책")

        # tie_break_order is always canonical for yoked (spec: "Tie-break stays canonical")
        tie_break_order = "canonical"

        if k > len(pool):
            raise ValueError(
                f"AXS_YOKED 정책: pool 크기 {len(pool)} 가 k={k} 보다 작아 "
                "슬레이트를 구성할 수 없습니다"
            )

        schedule = self._load_and_verify_schedule()
        per_round_bonus: List[float] = [float(v) for v in schedule["perRoundBonus"]]

        # t = number of prior rounds (trace length at selection time)
        t = len(trace.rounds())
        B_t = per_round_bonus[min(t, len(per_round_bonus) - 1)]

        # Build bandit state (optionally frozen).
        if freeze_round is not None:
            bandit_trace: Any = TraceView(trace.rounds()[:freeze_round])
        else:
            bandit_trace = trace

        A, b = self._replay_bandit_state(bandit_trace, active, lambda_reg)
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b

        # Live context (centroid/band) always uses the full trace.
        centroid, max_band = self._trace_centroid_and_band(trace)
        target_band_idx = (
            min(max_band + 1, len(BAND_ORDER) - 1) if max_band >= 0 else 0
        )

        # Canonical tiebreak (includes traceHash — the mean is trace-dependent).
        pool_hash = canonical_hash([c["cardId"] for c in pool])
        seed_hex = canonical_hash(
            {
                "poolHash": pool_hash,
                "traceHash": trace.trace_hash(),
                "seed": seed,
                "policyVersion": self.policy_version(),
            }
        )
        rng = random.Random(int(seed_hex, 16))
        tiebreak = {c["cardId"]: rng.random() for c in pool}

        # u(card): per-card floats from a trace-INDEPENDENT RNG.
        # Seed material: poolHash, seed, roundIndex (t), policyVersion, salt.
        # No traceHash in u-map seed (that is the point).
        u_seed_hex = canonical_hash(
            {
                "poolHash": pool_hash,
                "seed": seed,
                "roundIndex": t,
                "policyVersion": self.policy_version(),
                "salt": "axs004c-yoked-direction-v1",
            }
        )
        u_rng = random.Random(int(u_seed_hex, 16))
        sorted_card_ids = sorted(c["cardId"] for c in pool)
        u_map = {cid: u_rng.random() for cid in sorted_card_ids}

        bases_cfg = _load_bases_cfg()

        feature_cache: Dict[str, np.ndarray] = {}

        def yoked_components(card: Dict[str, Any]) -> Dict[str, float]:
            x = self._features(
                card["coordinateContribution"],
                card["complexityBand"],
                centroid,
                target_band_idx,
                active,
            )
            feature_cache[card["cardId"]] = x
            mean = float(theta @ x)
            u = u_map[card["cardId"]]
            yoked_bonus = B_t * u
            ucb = mean + alpha * yoked_bonus
            # For compatibility with log_score_components, store mean and bonus.
            return {
                "mean": round(mean, 12),
                "bonus": round(yoked_bonus, 12),
                "ucb": round(ucb, 12),
            }

        comp_cache = {c["cardId"]: yoked_components(c) for c in pool}

        # SEAM 3: yoked rank key (canonical only)
        def _rank_key(card: Dict[str, Any], need_new_basis: bool, chosen_bases: set) -> tuple:
            comp = comp_cache[card["cardId"]]
            cid = card["cardId"]
            nb = 1 if (need_new_basis and card["basis"] not in chosen_bases) else 0
            return (-nb, -comp["ucb"], tiebreak[cid], cid)

        slate = self._assemble_slate(
            pool, comp_cache, feature_cache, tiebreak, _rank_key,
            k, bases_cfg, seed, tie_break_order, pool_hash,
        )

        if slate is None:
            raise ValueError(
                f"AXS_YOKED 정책: k={k} 제약을 만족하는 슬레이트를 구성하지 "
                f"못했습니다 (poolHash={pool_hash})"
            )

        self.log_score_components(
            {
                c["cardId"]: {
                    "mean": comp_cache[c["cardId"]]["mean"],
                    "bonus": comp_cache[c["cardId"]]["bonus"],
                }
                for c in slate
            }
        )
        log_ko(
            _logger,
            f"AXS_YOKED 슬레이트 확정: k={k}, alpha={alpha}, B_t={B_t}, t={t}, "
            f"poolHash={pool_hash}",
        )
        return [c["cardId"] for c in slate]
