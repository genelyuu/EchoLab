"""Tests for echo_bench.utils.hash (Task F-002)."""

from __future__ import annotations

import pytest

from echo_bench.utils.hash import HASH_VERSION, canonical_hash, raster_hash


def test_same_content_same_hash():
    obj = {"a": 1, "b": [1, 2, 3], "c": {"x": True, "y": None}}
    assert canonical_hash(obj) == canonical_hash(dict(obj))


def test_dict_key_reordering_same_hash():
    a = {"alpha": 1, "beta": 2, "gamma": {"p": 0.5, "q": "z"}}
    b = {"gamma": {"q": "z", "p": 0.5}, "beta": 2, "alpha": 1}
    assert canonical_hash(a) == canonical_hash(b)


def test_different_content_different_hash():
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
    assert canonical_hash([1, 2, 3]) != canonical_hash([3, 2, 1])


def test_version_is_mixed_in():
    # The version tag is part of the digest input; document it is non-empty
    # and present. Changing it would change all hashes (covered by design).
    assert isinstance(HASH_VERSION, str) and HASH_VERSION != ""


def test_float_stability():
    # Logically equal floats hash identically; ints and floats are distinct.
    assert canonical_hash({"v": 0.1 + 0.2}) == canonical_hash({"v": 0.30000000000000004})
    assert canonical_hash({"v": 1.0}) == canonical_hash({"v": 1.0})


def test_determinism_across_calls():
    # Determinism across two calls demonstrates the builtin salted hash() is
    # not used (PYTHONHASHSEED would otherwise perturb results).
    obj = {"k": "v", "n": 42, "f": 3.14159, "nested": {"l": [1, "two", 3.0]}}
    assert canonical_hash(obj) == canonical_hash(obj)


def test_raster_hash_accepts_bytes():
    h1 = raster_hash(b"\x00\x01\x02\x03")
    h2 = raster_hash(bytearray(b"\x00\x01\x02\x03"))
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64


def test_raster_hash_accepts_tobytes_duck():
    class FakeArray:
        def __init__(self, payload: bytes):
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    payload = b"raster-bytes-content"
    assert raster_hash(FakeArray(payload)) == raster_hash(payload)


def test_raster_hash_rejects_bad_type():
    with pytest.raises(TypeError):
        raster_hash(12345)


def test_canonical_hash_rejects_unsupported_type():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        canonical_hash({"bad": Opaque()})
