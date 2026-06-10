"""Tests for F-007 optional GPU backend + CPU-equivalence gate."""
import numpy as np
import pytest

from echo_bench.utils.compute import (
    GPU_ELIGIBLE_TASKS, ComputeBackend, assert_cpu_equivalence, get_backend,
    load_hardware_config,
)

_HW = {
    "prefer_gpu": True,
    "allow_gpu_for": list(GPU_ELIGIBLE_TASKS),
    "require_cpu_equivalence": True,
    "gpu_backend": "cupy",
}


def test_prefer_gpu_false_is_cpu():
    b = get_backend(prefer_gpu=False, task="batch_card_generation", hw_cfg=_HW)
    assert isinstance(b, ComputeBackend)
    assert b.name == "cpu"
    assert b.xp is np


def test_non_whitelisted_task_is_cpu():
    b = get_backend(prefer_gpu=True, task="round_runner", hw_cfg=_HW)
    assert b.name == "cpu"  # replay-chain task is never GPU-eligible


def test_gpu_eligible_falls_back_to_cpu_without_cupy():
    # On the CPU-only test env cupy is absent -> must fall back, never raise.
    b = get_backend(prefer_gpu=True, task="batch_card_generation", hw_cfg=_HW)
    assert b.name in {"cpu", "gpu"}  # cpu on CI; gpu only if cupy present
    if b.name == "cpu":
        assert "cupy" in b.reason  # import-failure path


def test_equivalence_passes_on_match():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert_cpu_equivalence(a.copy(), a.copy(), "raster") is None


def test_equivalence_raises_on_mismatch():
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.ones((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        assert_cpu_equivalence(a, b, "raster")


def test_load_hardware_config_defaults(tmp_path):
    cfg = load_hardware_config(tmp_path / "missing.yaml")
    assert cfg["prefer_gpu"] is False  # safe default when file absent


def test_load_hardware_config_partial_override(tmp_path):
    yaml_file = tmp_path / "hw.yaml"
    yaml_file.write_text("prefer_gpu: true\n", encoding="utf-8")
    cfg = load_hardware_config(yaml_file)
    # Overridden key
    assert cfg["prefer_gpu"] is True
    # Missing keys keep their defaults
    assert cfg["require_cpu_equivalence"] is True
    assert cfg["gpu_backend"] == "cupy"
