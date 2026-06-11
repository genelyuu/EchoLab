"""Drift guard for the frozen TRACE_GREEDY effective config (Task C-011).

Asserts that the live policy's ``policy_version()`` hash matches the hash
recorded in ``configs/policies/frozen/trace_greedy_frozen.json``.  The test
fails immediately if anyone mutates a weight in ``trace_greedy.yaml``,
``DEFAULT_WEIGHTS``, or the class name without explicitly updating the freeze
manifest via a documented decision.

Construction path mirrors ``src/echo_bench/experiments/e2_policy.py`` exactly:
  1. Load ``configs/policies/trace_greedy.yaml`` with ``yaml.safe_load``.
  2. Override ``k`` with the E2 run-level default (4).
  3. Instantiate ``TraceGreedyPolicy(cfg)`` and call ``.policy_version()``.

No user/persona/emotion/preference field is used anywhere.  All identifiers and
assertion messages stay English; this file is a test artifact, not a log source.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from echo_bench.policies.trace_greedy import TraceGreedyPolicy

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FROZEN_MANIFEST = (
    _REPO_ROOT / "configs" / "policies" / "frozen" / "trace_greedy_frozen.json"
)
_POLICY_CFG = _REPO_ROOT / "configs" / "policies" / "trace_greedy.yaml"

# Required top-level keys in the manifest.
_REQUIRED_KEYS = {
    "policyName",
    "effectiveConfig",
    "policyEffectiveConfigHash",
    "frozenAt",
    "taskId",
    "note",
}


def _load_manifest() -> dict:
    with open(_FROZEN_MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_live_policy() -> TraceGreedyPolicy:
    """Instantiate TraceGreedyPolicy the same way e2_policy.py does."""
    with open(_POLICY_CFG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = dict(cfg) if isinstance(cfg, dict) else {}
    cfg["k"] = 4  # E2 run-level k override (fixed for the E2 comparison)
    return TraceGreedyPolicy(cfg)


# ---------------------------------------------------------------------------
# Manifest presence and structure
# ---------------------------------------------------------------------------


def test_manifest_file_exists():
    assert _FROZEN_MANIFEST.exists(), (
        f"Frozen manifest not found: {_FROZEN_MANIFEST}\n"
        "Run the C-011 freeze generation step to create it."
    )


def test_manifest_has_required_keys():
    manifest = _load_manifest()
    missing = _REQUIRED_KEYS - manifest.keys()
    assert not missing, (
        f"Frozen manifest is missing required keys: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Hash match
# ---------------------------------------------------------------------------


def test_policy_version_matches_manifest():
    manifest = _load_manifest()
    frozen_hash = manifest["policyEffectiveConfigHash"]

    policy = _build_live_policy()
    live_hash = policy.policy_version()

    assert live_hash == frozen_hash, (
        "TRACE_GREEDY effective config has drifted from the C-011 freeze!\n"
        f"  frozen hash : {frozen_hash}\n"
        f"  live hash   : {live_hash}\n"
        "If you intentionally changed a weight or the class name, update "
        "configs/policies/frozen/trace_greedy_frozen.json via an explicit "
        "decision (do NOT silently regenerate the manifest)."
    )


def test_manifest_task_id_is_c011():
    manifest = _load_manifest()
    assert manifest["taskId"] == "C-011", (
        f"Expected taskId 'C-011', got {manifest['taskId']!r}"
    )


def test_manifest_policy_name():
    manifest = _load_manifest()
    assert manifest["policyName"] == "TRACE_GREEDY", (
        f"Expected policyName 'TRACE_GREEDY', got {manifest['policyName']!r}"
    )


# ---------------------------------------------------------------------------
# Manifest-config validation: guards against in-file weight tampering
# ---------------------------------------------------------------------------


def test_manifest_effective_config_matches_live_policy():
    """The manifest's ``effectiveConfig`` must equal the live policy's config.

    This test catches the failure mode where someone edits a weight value
    inside ``effectiveConfig`` in the manifest JSON while the
    ``policyEffectiveConfigHash`` (which is computed from the live policy, not
    from this file) still matches.  On real drift it pinpoints exactly which
    key changed.
    """
    manifest = _load_manifest()
    policy = _build_live_policy()

    # policy_version() hashes {"policy": class_name, "config": self.config}.
    # The live config is self.config, set directly from the e2-style construction.
    live_config = getattr(policy, "config", {})

    assert manifest["effectiveConfig"] == live_config, (
        "manifest['effectiveConfig'] does not match the live policy config!\n"
        f"  manifest effectiveConfig : {manifest['effectiveConfig']}\n"
        f"  live policy config       : {live_config}\n"
        "If you intentionally changed a weight, update both effectiveConfig "
        "and policyEffectiveConfigHash in the frozen manifest via an explicit "
        "decision."
    )
