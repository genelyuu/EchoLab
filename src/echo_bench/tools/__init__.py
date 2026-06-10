"""ECHO-Bench tooling subpackage (Task G-005).

Exposes the context-aware forbidden-claim scanner
(:mod:`echo_bench.tools.claim_check`), the automated guardrail aid that scans
``docs/`` and ``outputs/reports/`` for forbidden user-facing claim language.

Names are re-exported lazily via :func:`__getattr__` so that running the scanner
as ``python -m echo_bench.tools.claim_check`` does not import the module twice
(which would emit a ``runpy`` RuntimeWarning).

All identifiers and file paths stay English; runtime summary log lines are
Korean per the project logging convention.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FORBIDDEN_PHRASES",
    "Finding",
    "main",
    "scan_path",
    "scan_paths",
    "scan_text",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from echo_bench.tools import claim_check

        return getattr(claim_check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
