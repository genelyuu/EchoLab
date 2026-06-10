"""ECHO-Bench runtime logging package.

Logging convention (project-wide): all runtime log messages — logger output,
console prints, progress/status text — are written in **Korean**. Machine-read
identifiers stay in English: task IDs, hash keys/values, metric names, config
keys, file paths, versions, and exception *type* names.

This module exposes a single configured stdlib logger factory so every call
site shares the same handler/format setup. Callers pass Korean text as the
message body; :func:`log_ko` is a thin convenience wrapper that makes that
intent explicit at the call site.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger", "log_ko"]


def get_logger(name: str) -> logging.Logger:
    """Return a configured stdlib logger.

    Attaches a single ``StreamHandler`` at INFO level with a simple formatter
    if the logger has no handlers yet. Idempotent: repeated calls for the same
    name do not stack handlers. Runtime messages passed by callers are expected
    to be Korean; machine-read identifiers stay English.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Avoid duplicate emission via the root logger.
        logger.propagate = False
    return logger


def log_ko(logger: logging.Logger, msg: str) -> None:
    """Emit an INFO-level Korean runtime message.

    Convenience wrapper around ``logger.info(msg)``. Callers pass Korean text;
    this exists to make the Korean-logging convention explicit at call sites.
    """
    logger.info(msg)
