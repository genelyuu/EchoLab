"""Basis configuration schema for ECHO-Bench (Task A-001).

Defines the bounded, versioned configuration schema for the four procedural
bases:

- ``B1`` Branching
- ``B2`` Reaction-Diffusion
- ``B3`` Topographic fBM
- ``B4`` Flow-Attractor

The schema is the single source of truth for all four renderer tasks
(A-002..A-005). It enforces the project guardrails: every parameter has an
explicit ``[min, max]`` bound, and no personal / user / semantic field
(``user_id``, ``persona``, ``emotion``, ``preference``, free text, ...) may
appear anywhere in a basis config.

All identifiers stay English. Any runtime log messages are written in Korean
per project convention (see :mod:`echo_bench.logging`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Forbidden-field guardrail
# ---------------------------------------------------------------------------

# Any config key whose lowercase form contains one of these tokens is rejected
# on load. This is a fail-closed defence against personal / semantic fields
# leaking into the controlled action space (CLAUDE.md guardrail 1).
FORBIDDEN_FIELD_TOKENS = (
    "user_id",
    "user",
    "persona",
    "emotion",
    "preference",
    "demographic",
    "diagnosis",
    "free_text",
    "freetext",
    "text",
    "label",
    "note",
    "comment",
)

# The exact set of basis ids the schema must declare.
REQUIRED_BASIS_IDS = ("B1", "B2", "B3", "B4")


@dataclass(frozen=True)
class BasisSpec:
    """Bounded, versioned specification for a single procedural basis.

    Attributes:
        basis: Basis id, one of ``B1``..``B4``.
        param_ranges: Mapping ``name -> (min, max)`` of inclusive numeric
            bounds for each sampled parameter. Every range satisfies
            ``min <= max``.
        rendererVersion: Version string for the renderer that consumes this
            basis (e.g. ``"stub-r1"``). Mixed into the render hash, so bumping
            it deterministically changes downstream hashes.
    """

    basis: str
    param_ranges: dict[str, tuple[float, float]]
    rendererVersion: str


def _is_forbidden_key(key: str) -> bool:
    """Return ``True`` if ``key`` names a forbidden personal/semantic field."""
    low = str(key).lower()
    return any(tok in low for tok in FORBIDDEN_FIELD_TOKENS)


def _check_no_forbidden(obj: Any, where: str) -> None:
    """Recursively reject forbidden keys / free-text-ish content.

    Walks dicts and lists. Raises :class:`ValueError` on any forbidden key.
    """
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if _is_forbidden_key(str(k)):
                raise ValueError(
                    f"basis config rejected: forbidden field {k!r} at {where}"
                )
            _check_no_forbidden(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_no_forbidden(v, f"{where}[{i}]")


def _coerce_range(name: str, raw: Any, where: str) -> tuple[float, float]:
    """Validate and coerce a single ``[min, max]`` range entry."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"basis config rejected: param {name!r} at {where} must be a "
            f"[min, max] pair, got {raw!r}"
        )
    lo, hi = raw
    if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(
        lo, (int, float)
    ) or not isinstance(hi, (int, float)):
        raise ValueError(
            f"basis config rejected: param {name!r} bounds must be numeric, "
            f"got {raw!r}"
        )
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > hi_f:
        raise ValueError(
            f"basis config rejected: param {name!r} has min {lo_f} > max {hi_f}"
        )
    return (lo_f, hi_f)


def _spec_from_entry(basis_id: str, entry: Any) -> BasisSpec:
    """Build a :class:`BasisSpec` from one parsed config entry."""
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"basis config rejected: entry for {basis_id!r} must be a mapping"
        )
    _check_no_forbidden(entry, basis_id)

    renderer_version = entry.get("rendererVersion")
    if not isinstance(renderer_version, str) or not renderer_version:
        raise ValueError(
            f"basis config rejected: {basis_id!r} missing non-empty "
            f"'rendererVersion'"
        )

    params = entry.get("param_ranges")
    if not isinstance(params, Mapping) or not params:
        raise ValueError(
            f"basis config rejected: {basis_id!r} missing non-empty "
            f"'param_ranges'"
        )

    ranges: dict[str, tuple[float, float]] = {}
    for name, raw in params.items():
        if _is_forbidden_key(str(name)):
            raise ValueError(
                f"basis config rejected: forbidden param {name!r} in {basis_id!r}"
            )
        ranges[str(name)] = _coerce_range(str(name), raw, basis_id)

    return BasisSpec(
        basis=basis_id, param_ranges=ranges, rendererVersion=renderer_version
    )


def _parse_yaml(text: str) -> Any:
    """Parse YAML using pyyaml if available, else a minimal stdlib fallback.

    The fallback understands only the restricted structure used by
    ``configs/basis/bases.yaml`` (a two-level mapping with ``[min, max]`` list
    values and string scalars). It exists purely so the schema remains
    importable and verifiable when pyyaml is not installed.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _minimal_yaml(text)


def _scalar(token: str) -> Any:
    """Coerce a bare YAML scalar token to int / float / str."""
    token = token.strip()
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Very small indentation-based YAML subset parser (fallback only)."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if ":" not in content:
            raise ValueError(f"minimal yaml: cannot parse line {raw_line!r}")
        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        elif rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            items = [_scalar(p) for p in inner.split(",")] if inner else []
            parent[key] = items
        else:
            parent[key] = _scalar(rest)
    return root


def load_bases(path: str | Path) -> dict[str, BasisSpec]:
    """Load and validate ``bases.yaml`` into ``{basis_id: BasisSpec}``.

    The config must declare exactly ``B1``..``B4`` under a top-level ``bases``
    mapping (or at the document root). Loading is deterministic and rejects any
    forbidden personal/semantic field or unbounded param.

    Raises:
        ValueError: on missing/extra bases, forbidden fields, or invalid ranges.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    doc = _parse_yaml(text)
    if not isinstance(doc, Mapping):
        raise ValueError("basis config rejected: top-level document is not a mapping")

    _check_no_forbidden(doc, "<root>")

    bases = doc.get("bases", doc)
    if not isinstance(bases, Mapping):
        raise ValueError("basis config rejected: 'bases' is not a mapping")

    declared = tuple(str(k) for k in bases.keys())
    if set(declared) != set(REQUIRED_BASIS_IDS):
        raise ValueError(
            f"basis config rejected: must declare exactly {REQUIRED_BASIS_IDS}, "
            f"got {declared}"
        )

    specs: dict[str, BasisSpec] = {}
    for basis_id in REQUIRED_BASIS_IDS:  # deterministic, canonical order
        specs[basis_id] = _spec_from_entry(basis_id, bases[basis_id])
    return specs


def sample_params(spec: BasisSpec, rng: random.Random) -> dict[str, float]:
    """Sample each parameter uniformly within its bounds using ``rng``.

    Args:
        spec: The basis specification whose ``param_ranges`` drive sampling.
        rng: A seeded :class:`random.Random` (stdlib-safe). A numpy ``Generator``
            also works as long as it exposes a ``uniform(lo, hi)`` method; this
            function relies only on that interface.

    Returns:
        Mapping ``name -> sampled value`` with each value inside ``[min, max]``.

    Raises:
        ValueError: if any sampled value falls outside its declared bounds
            (defence against a misbehaving rng).
    """
    out: dict[str, float] = {}
    for name in sorted(spec.param_ranges):  # deterministic draw order
        lo, hi = spec.param_ranges[name]
        value = float(rng.uniform(lo, hi))
        if value < lo or value > hi:
            raise ValueError(
                f"sample_params: value {value} for {name!r} out of bounds "
                f"[{lo}, {hi}]"
            )
        out[name] = value
    return out


def validate_params(spec: BasisSpec, params: Mapping[str, Any]) -> None:
    """Validate an externally supplied params mapping against ``spec`` bounds.

    Raises:
        ValueError: on unknown params, missing params, or out-of-bounds values.
    """
    expected = set(spec.param_ranges)
    given = set(str(k) for k in params)
    if given != expected:
        raise ValueError(
            f"validate_params: param set {sorted(given)} != expected "
            f"{sorted(expected)} for basis {spec.basis!r}"
        )
    for name, (lo, hi) in spec.param_ranges.items():
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"validate_params: {name!r} must be numeric")
        if float(value) < lo or float(value) > hi:
            raise ValueError(
                f"validate_params: {name!r}={value} out of bounds [{lo}, {hi}]"
            )
