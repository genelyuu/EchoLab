"""Stable hashing primitives for ECHO-Bench (Task F-002).

This module provides deterministic, cross-platform hashing for the
reproducibility hash chain. Identical logical content must always hash
identically, regardless of process, platform, or in-memory key ordering.

Design rules enforced here:
- Canonical JSON serialization with sorted keys and compact separators.
- Floats are serialized with a fixed ``format(x, ".12g")`` representation to
  avoid platform-dependent ``repr`` drift.
- The hash algorithm version (:data:`HASH_VERSION`) is mixed into every digest
  input, so changing the version explicitly changes all hashes.
- No use of the salted builtin ``hash()``, wall-clock time, process id, or
  memory addresses ever enters a hash.

All identifiers (function names, the version string) stay English; any runtime
log messages produced by callers are written in Korean per project convention.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# Hash algorithm/version tag. Mixed into every digest input so that bumping
# this constant deterministically changes every produced hash.
HASH_VERSION = "h1"


def _json_default(obj: Any) -> Any:
    """Fallback encoder for objects not natively JSON-serializable.

    Only used for types that the canonical encoder cannot otherwise handle.
    Raises ``TypeError`` for unsupported types to fail closed rather than
    silently producing an unstable hash.
    """
    raise TypeError(
        f"canonical_hash: unsupported type for deterministic encoding: "
        f"{type(obj).__name__!r}"
    )


def _stabilize(obj: Any) -> Any:
    """Recursively rewrite an object into a canonically-encodable form.

    Floats are converted to a fixed-precision string so their serialization is
    stable across platforms (avoiding ``repr`` drift). Dicts and lists/tuples
    are walked recursively. Tuples are normalized to lists so that logically
    identical sequences encode identically.
    """
    if isinstance(obj, bool):
        # bool is a subclass of int; keep it as a real boolean.
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        # Fixed-precision, platform-stable float text.
        return "f:" + format(obj, ".12g")
    if isinstance(obj, dict):
        return {str(k): _stabilize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stabilize(v) for v in obj]
    return obj


def canonical_hash(obj: Any) -> str:
    """Return a deterministic sha256 hex digest of a JSON-serializable object.

    Accepts any nested structure of dict/list/str/int/float/bool/None. Keys are
    sorted, separators are compact, and floats are encoded with a fixed
    precision so the digest is stable across processes and platforms. Reordering
    the keys of a dict does not change the result.

    The serialized payload is prefixed with :data:`HASH_VERSION`, so changing
    the version changes every hash.
    """
    stable = _stabilize(obj)
    payload = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    digest_input = (HASH_VERSION + "\x00" + payload).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def raster_hash(data: Any) -> str:
    """Return a deterministic sha256 hex digest of raw raster bytes.

    Accepts ``bytes``/``bytearray`` directly, or any object exposing a
    ``.tobytes()`` method (e.g. a numpy array, duck-typed — numpy is never
    imported here). The :data:`HASH_VERSION` tag is mixed into the digest input.
    """
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    elif hasattr(data, "tobytes"):
        raw = data.tobytes()
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(
                "raster_hash: .tobytes() must return bytes-like, got "
                f"{type(raw).__name__!r}"
            )
        raw = bytes(raw)
    else:
        raise TypeError(
            "raster_hash: expected bytes/bytearray or an object with "
            f".tobytes(), got {type(data).__name__!r}"
        )
    digest = hashlib.sha256()
    digest.update((HASH_VERSION + "\x00").encode("utf-8"))
    digest.update(raw)
    return digest.hexdigest()
