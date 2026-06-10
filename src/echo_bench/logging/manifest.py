"""Run manifest schema (Task F-003).

A :class:`RunManifest` captures the full per-run hash chain (embedded as a
:class:`~echo_bench.logging.repro_pack.ReproducibilityPack`) together with every
component version that participated in the run: renderer, policy, probe, metric,
and report versions, plus the seed-batch id.

Invariants:
- Embeds a ReproducibilityPack (all 9 hash fields) plus exactly the five version
  fields ``rendererVersion, policyVersion, probeVersion, metricVersion,
  reportVersion`` and ``seedBatchId``.
- The dataclass is frozen and fixed-field, so no user/persona/emotion/preference/
  free-text metadata can be attached. ``from_dict`` additionally rejects any
  unknown key.
- A manifest missing any version or any pack hash fails validation
  (``ValueError`` with a Korean operator message).
- The manifest round-trips deterministically via :meth:`to_dict` / :meth:`from_dict`
  and carries its own :meth:`manifest_hash`.

Field names, versions, and paths stay English (machine-read identifiers); the
runtime log line emitted on build and any validation messages are Korean per
project convention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from echo_bench.logging import get_logger, log_ko
from echo_bench.logging.repro_pack import ReproducibilityPack
from echo_bench.utils.hash import canonical_hash

__all__ = ["RunManifest", "VERSION_FIELDS"]

_logger = get_logger("echo_bench.logging.manifest")

# The five component-version fields every run must record. The seedBatchId is
# carried both on the embedded pack and on the manifest; it is validated here as
# a required, non-empty identifier as well.
VERSION_FIELDS = (
    "rendererVersion",
    "policyVersion",
    "probeVersion",
    "metricVersion",
    "reportVersion",
)


@dataclass(frozen=True)
class RunManifest:
    """Frozen per-run manifest: a ReproducibilityPack plus component versions.

    All fields are required, non-empty strings (or a valid ``ReproducibilityPack``
    for ``pack``). The fixed-field frozen design structurally forbids any
    non-run, user, or semantic metadata from entering the manifest.
    """

    pack: ReproducibilityPack
    rendererVersion: str
    policyVersion: str
    probeVersion: str
    metricVersion: str
    reportVersion: str
    seedBatchId: str

    def __post_init__(self) -> None:
        """Validate the embedded pack, every version, and the seed-batch id.

        Fails closed: a non-pack ``pack``, a missing/empty version, or a
        missing/empty ``seedBatchId`` raises ``ValueError`` with a Korean
        operator message. Pack hash validation is delegated to
        :class:`ReproducibilityPack` (re-run here to surface a Korean manifest
        message if a malformed pack was constructed by other means).
        """
        if not isinstance(self.pack, ReproducibilityPack):
            raise ValueError(
                "RunManifest.pack 는 ReproducibilityPack 이어야 합니다 "
                f"(현재 타입: {type(self.pack).__name__})."
            )
        # Re-validate the embedded pack's hashes (fail closed even if the pack
        # was built bypassing its own __post_init__).
        for f in fields(ReproducibilityPack):
            value = getattr(self.pack, f.name)
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(
                    f"RunManifest 에 포함된 pack 의 해시 필드 '{f.name}' 가 "
                    "누락되었거나 비어 있습니다."
                )
        # Validate the five component versions.
        for name in VERSION_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(
                    f"RunManifest 버전 필드 '{name}' 는 문자열이어야 합니다 "
                    f"(현재 타입: {type(value).__name__})."
                )
            if value.strip() == "":
                raise ValueError(
                    f"RunManifest 버전 필드 '{name}' 는 비어 있을 수 없습니다."
                )
        # Validate the seed-batch id on the manifest itself.
        if not isinstance(self.seedBatchId, str) or self.seedBatchId.strip() == "":
            raise ValueError(
                "RunManifest 필드 'seedBatchId' 는 비어 있을 수 없습니다."
            )

        log_ko(
            _logger,
            f"실행 매니페스트 생성됨: seedBatchId={self.seedBatchId}, "
            f"manifestHash={self.manifest_hash()}.",
        )

    def to_dict(self) -> dict:
        """Return a deterministic dict representation of the manifest.

        The embedded pack is serialized via its own ``to_dict`` so the nested
        hash chain is preserved exactly.
        """
        d = {"pack": self.pack.to_dict()}
        for name in VERSION_FIELDS:
            d[name] = getattr(self, name)
        d["seedBatchId"] = self.seedBatchId
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        """Construct a manifest from a dict, rejecting unknown/forbidden keys.

        Raises ``ValueError`` if ``d`` contains any key outside the declared
        manifest field set (covering rejection of user/persona/emotion/
        preference/free-text fields) or if any required field is missing. The
        embedded ``pack`` is rebuilt through :meth:`ReproducibilityPack.from_dict`,
        which itself rejects unknown keys and missing hashes.
        """
        if not isinstance(d, dict):
            raise ValueError(
                f"RunManifest.from_dict 입력은 dict 여야 합니다 "
                f"(현재 타입: {type(d).__name__})."
            )
        allowed = {"pack", "seedBatchId", *VERSION_FIELDS}
        extra = set(d.keys()) - allowed
        if extra:
            raise ValueError(
                "RunManifest 에 허용되지 않은 필드가 포함되어 있습니다: "
                f"{sorted(extra)}. 허용 필드: {sorted(allowed)}."
            )
        missing = allowed - set(d.keys())
        if missing:
            raise ValueError(
                "RunManifest 필수 필드가 누락되었습니다: "
                f"{sorted(missing)}."
            )
        pack_val = d["pack"]
        pack = (
            pack_val
            if isinstance(pack_val, ReproducibilityPack)
            else ReproducibilityPack.from_dict(pack_val)
        )
        return cls(
            pack=pack,
            rendererVersion=d["rendererVersion"],
            policyVersion=d["policyVersion"],
            probeVersion=d["probeVersion"],
            metricVersion=d["metricVersion"],
            reportVersion=d["reportVersion"],
            seedBatchId=d["seedBatchId"],
        )

    def manifest_hash(self) -> str:
        """Return the deterministic canonical hash of this manifest."""
        return canonical_hash(self.to_dict())
