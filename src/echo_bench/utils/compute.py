"""Optional GPU compute backend + CPU-equivalence gate (Task F-007).

The ECHO-Bench core is CPU-replayable. GPU is OPT-IN and permitted ONLY for the
whitelisted tasks in :data:`GPU_ELIGIBLE_TASKS` (large-scale batch card
generation and neural baselines) — never for any main-claim trace producer
(``run_episode`` and the E*/S* experiments do not appear in the whitelist).

:func:`get_backend` returns a CPU backend unless ALL hold: the caller opted in,
the loaded hardware config opted in, the task is whitelisted, the task is
GPU-eligible, and ``cupy`` imports. Otherwise it returns CPU with a Korean
``reason``. cupy is NEVER imported at module top level.

:func:`assert_cpu_equivalence` is the fail-closed gate: any GPU array destined for
the hash chain must bit-match a CPU reference (compared via the project hashers)
or it raises, forcing CPU fallback. This is what lets "optional GPU" coexist
with the replay-or-no-claim invariant.

Identifiers / keys stay English; runtime log lines are Korean per convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml

from echo_bench.logging import get_logger
from echo_bench.utils.hash import canonical_hash, raster_hash

__all__ = [
    "ComputeBackend",
    "GPU_ELIGIBLE_TASKS",
    "get_backend",
    "assert_cpu_equivalence",
    "load_hardware_config",
]

_logger = get_logger(__name__)

# The ONLY tasks GPU may ever accelerate. No main-claim trace producer is here.
GPU_ELIGIBLE_TASKS = ("batch_card_generation", "neural_baseline")

# Safe defaults used when the hardware config file is absent or empty.
_DEFAULT_HW = {
    "prefer_gpu": False,
    "allow_gpu_for": list(GPU_ELIGIBLE_TASKS),
    "require_cpu_equivalence": True,
    "gpu_backend": "cupy",
}


@dataclass(frozen=True)
class ComputeBackend:
    """Frozen selected backend: ``name`` ('cpu'|'gpu'), ``xp`` module, ``reason``."""

    name: str
    xp: Any
    reason: str


def load_hardware_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the hardware compute config, falling back to safe CPU defaults."""
    p = Path(path)
    if not p.exists():
        log_msg = f"하드웨어 설정 파일이 없어 CPU 기본값을 사용합니다: path={p}"
        _logger.info(log_msg)
        return dict(_DEFAULT_HW)
    with open(p, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    cfg = dict(_DEFAULT_HW)
    if isinstance(doc, dict):
        cfg.update(doc)
    return cfg


def get_backend(prefer_gpu: bool, task: str, hw_cfg: Dict[str, Any]) -> ComputeBackend:
    """Return a CPU backend unless GPU is opted-in, whitelisted, and importable.

    GPU requires ALL of: ``prefer_gpu`` (caller) and ``hw_cfg['prefer_gpu']``,
    ``task in hw_cfg['allow_gpu_for']``, ``task in GPU_ELIGIBLE_TASKS``, and a
    successful ``cupy`` import. Any failure yields a CPU backend (never raises).
    """
    allow = list(hw_cfg.get("allow_gpu_for", GPU_ELIGIBLE_TASKS))
    if not (prefer_gpu and bool(hw_cfg.get("prefer_gpu", False))):
        return ComputeBackend("cpu", np, "GPU 미요청: CPU 백엔드 사용")
    if task not in GPU_ELIGIBLE_TASKS or task not in allow:
        return ComputeBackend(
            "cpu", np, f"task={task} 는 GPU 화이트리스트에 없어 CPU 사용"
        )
    try:
        import cupy as cp  # noqa: WPS433 (intentional lazy import)
    except Exception as exc:  # cupy absent / no CUDA -> fail closed to CPU
        return ComputeBackend(
            "cpu", np, f"cupy 임포트 실패로 CPU 폴백: {type(exc).__name__}"
        )
    _logger.info("GPU 백엔드를 선택했습니다 (task=%s)", task)
    return ComputeBackend("gpu", cp, f"GPU 허용: task={task}")


def _to_cpu_bytes(arr: Any) -> Any:
    """Materialize an array to CPU numpy (handles cupy via ``.get()``)."""
    if hasattr(arr, "get") and not isinstance(arr, np.ndarray):
        return np.asarray(arr.get())
    return np.asarray(arr)


def assert_cpu_equivalence(gpu_array: Any, cpu_array: Any, label: str) -> None:
    """Raise (Korean) unless the two arrays hash identically on CPU.

    Fail-closed gate for any GPU artifact entering the hash chain.  The check
    proceeds in three guards:

    1. **dtype guard** — if ``gpu_array.dtype != cpu_array.dtype`` raise a
       Korean :class:`ValueError` describing the dtype mismatch.
    2. **shape guard** — if the shapes differ raise a Korean
       :class:`ValueError` describing the shape mismatch.
    3. **byte-hash compare** — materialize both arrays to CPU bytes and
       compare ``raster_hash(g.tobytes())`` vs ``raster_hash(c.tobytes())``.
       This path is uniform for ALL dtypes (uint8, float32, float64, …).
       On mismatch raise a Korean :class:`ValueError` noting the value
       mismatch.
    """
    g = _to_cpu_bytes(gpu_array)
    c = _to_cpu_bytes(cpu_array)
    if g.dtype != c.dtype:
        raise ValueError(
            f"CPU 동등성 검증 실패(label={label}): dtype 불일치 "
            f"(gpu={g.dtype}, cpu={c.dtype}). "
            "재현성 보장을 위해 CPU 로 폴백해야 합니다."
        )
    if g.shape != c.shape:
        raise ValueError(
            f"CPU 동등성 검증 실패(label={label}): shape 불일치 "
            f"(gpu={g.shape}, cpu={c.shape}). "
            "재현성 보장을 위해 CPU 로 폴백해야 합니다."
        )
    gh = raster_hash(g.tobytes())
    ch = raster_hash(c.tobytes())
    if gh != ch:
        raise ValueError(
            f"CPU 동등성 검증 실패(label={label}): 값 불일치 — GPU 결과가 CPU 기준과 "
            f"일치하지 않습니다 (gpuHash={gh[:12]}, cpuHash={ch[:12]}). "
            "재현성 보장을 위해 CPU 로 폴백해야 합니다."
        )
