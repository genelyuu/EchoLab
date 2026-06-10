"""Experiment horizon configuration loader for ECHO-Bench (Task B-005).

The horizon ``H`` is the number of adaptation decision steps (rounds) in an
episode. It is loaded deterministically from a versioned config
(``configs/experiments/horizon.yaml``) that declares a bounded allowed set and a
default; horizons are never unbounded or open-ended, and ``H`` is never
hard-coded outside the config.

Contract:

    load_horizon(path) -> dict
        Parse the horizon config into ``{"H_allowed": [...], "H_default": int}``.
        Round-trips deterministically (parsing the same file always yields the
        same dict).

    validate_h(h, cfg) -> int
        Return ``h`` if it is in the config's allowed set, else raise
        :class:`ValueError` with a Korean message.

    default_h(cfg) -> int
        Return the config's default ``H`` (itself validated against the allowed
        set so a misconfigured default fails closed).

The selected ``H`` is a plain int and is therefore directly recordable in the
run manifest (F-003).

Config keys (``H_allowed``, ``H_default``) stay English; runtime log messages
are Korean per the project logging convention (see :mod:`echo_bench.logging`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import yaml

from echo_bench.logging import get_logger

__all__ = ["load_horizon", "validate_h", "default_h"]

_logger = get_logger(__name__)


def _coerce_allowed(raw: Any) -> List[int]:
    """Coerce the raw ``H_allowed`` value into a list of ints, failing closed."""
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"H_allowed 는 정수 리스트여야 합니다 (type={type(raw).__name__})"
        )
    allowed: List[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"H_allowed 의 모든 값은 정수여야 합니다 (잘못된 값={item!r})"
            )
        allowed.append(item)
    if not allowed:
        raise ValueError("H_allowed 가 비어 있을 수 없습니다")
    return allowed


def load_horizon(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the horizon config from ``path`` into a normalized dict.

    Returns ``{"H_allowed": list[int], "H_default": int}``. Raises
    :class:`ValueError` (Korean message) if the file is missing required keys,
    if ``H_allowed`` is not a non-empty integer list, or if ``H_default`` is not
    an integer inside ``H_allowed``. Parsing the same file is deterministic.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)

    if not isinstance(doc, Mapping):
        raise ValueError(
            f"horizon 설정 파일이 매핑이 아닙니다: path={config_path}"
        )

    if "H_allowed" not in doc:
        raise ValueError("horizon 설정에 H_allowed 키가 없습니다")
    if "H_default" not in doc:
        raise ValueError("horizon 설정에 H_default 키가 없습니다")

    allowed = _coerce_allowed(doc["H_allowed"])

    h_default = doc["H_default"]
    if isinstance(h_default, bool) or not isinstance(h_default, int):
        raise ValueError(
            f"H_default 는 정수여야 합니다 (잘못된 값={h_default!r})"
        )
    if h_default not in allowed:
        raise ValueError(
            f"H_default={h_default} 가 H_allowed={allowed} 에 포함되어 있지 "
            "않습니다"
        )

    cfg = {"H_allowed": allowed, "H_default": h_default}
    _logger.info(
        "horizon 설정을 로드했습니다: H_allowed=%s, H_default=%d",
        allowed,
        h_default,
    )
    return cfg


def validate_h(h: int, cfg: Mapping[str, Any]) -> int:
    """Return ``h`` if it is in ``cfg['H_allowed']``, else raise ``ValueError``.

    The error message is Korean per project convention. Booleans are rejected
    explicitly (``True``/``False`` are not valid horizons).
    """
    allowed = cfg.get("H_allowed")
    if not isinstance(allowed, (list, tuple)):
        raise ValueError("cfg 에 유효한 H_allowed 가 없습니다")
    if isinstance(h, bool) or not isinstance(h, int):
        raise ValueError(f"H 는 정수여야 합니다 (잘못된 값={h!r})")
    if h not in allowed:
        raise ValueError(
            f"H={h} 가 허용된 집합 {list(allowed)} 에 포함되어 있지 않습니다"
        )
    return h


def default_h(cfg: Mapping[str, Any]) -> int:
    """Return the config's default ``H``, validated against the allowed set."""
    if "H_default" not in cfg:
        raise ValueError("cfg 에 H_default 가 없습니다")
    return validate_h(cfg["H_default"], cfg)
