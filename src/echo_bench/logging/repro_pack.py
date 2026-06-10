"""ReproducibilityPack schema (Task F-001).

The ReproducibilityPack is the canonical container bundling the full hash set
for a single run: config, commit, candidate-archive, candidate-pool, selected
slate, trace, output, and report hashes, plus the seed-batch id.

Invariants:
- Exactly the declared fields exist; the dataclass is frozen and fixed-field,
  so no user/persona/emotion/preference/free-text metadata can be attached.
- Every field is required and validated as a non-empty string.
- The pack round-trips deterministically via :meth:`to_dict` / :meth:`from_dict`.
- :meth:`pack_hash` derives a stable digest through ``utils.hash.canonical_hash``.

Field names stay English (machine-read identifiers); validation messages raised
to operators are written in Korean per project convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from echo_bench.utils.hash import canonical_hash

__all__ = ["ReproducibilityPack"]


@dataclass(frozen=True)
class ReproducibilityPack:
    """Frozen container for the full per-run hash chain.

    All fields are required, non-empty hash/id strings. The fixed-field frozen
    dataclass design structurally forbids any non-run, user, or semantic
    metadata from entering the pack.
    """

    configHash: str
    commitHash: str
    archiveHash: str
    poolHash: str
    slateHash: str
    traceHash: str
    outputHash: str
    reportHash: str
    seedBatchId: str

    def __post_init__(self) -> None:
        """Validate every field is a non-empty string (fail closed)."""
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, str):
                raise ValueError(
                    f"ReproducibilityPack 필드 '{f.name}' 는 문자열이어야 합니다 "
                    f"(현재 타입: {type(value).__name__})."
                )
            if value == "" or value.strip() == "":
                raise ValueError(
                    f"ReproducibilityPack 필드 '{f.name}' 는 비어 있을 수 없습니다."
                )

    def to_dict(self) -> dict:
        """Return a deterministic dict representation of the pack."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReproducibilityPack":
        """Construct a pack from a dict, rejecting unknown/forbidden keys.

        Raises ``ValueError`` if ``d`` contains any key outside the declared
        field set (this covers rejection of user/persona/emotion/preference/
        free-text fields) or if required fields are missing/empty.
        """
        if not isinstance(d, dict):
            raise ValueError(
                f"ReproducibilityPack.from_dict 입력은 dict 여야 합니다 "
                f"(현재 타입: {type(d).__name__})."
            )
        allowed = {f.name for f in fields(cls)}
        extra = set(d.keys()) - allowed
        if extra:
            raise ValueError(
                "ReproducibilityPack 에 허용되지 않은 필드가 포함되어 있습니다: "
                f"{sorted(extra)}. 허용 필드: {sorted(allowed)}."
            )
        missing = allowed - set(d.keys())
        if missing:
            raise ValueError(
                "ReproducibilityPack 필수 필드가 누락되었습니다: "
                f"{sorted(missing)}."
            )
        return cls(**d)

    def pack_hash(self) -> str:
        """Return the deterministic canonical hash of this pack."""
        return canonical_hash(self.to_dict())
