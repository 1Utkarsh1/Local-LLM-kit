"""Measured host memory and approximate model budgets (GiB).

Estimates depend on architecture, backend, batch size and cache allocation.
They are planning guidance and cannot guarantee that a model will fit.
"""

import math
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Optional

QUANT_BPW: Dict[str, float] = {
    "FP16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.0,
    "Q5_0": 5.0,
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.25,
    "Q4_0": 4.5,
    "Q3_K_M": 3.4,
    "Q3_K_S": 3.0,
    "Q2_K": 2.6,
}


def _command(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=3).stdout


def get_system_memory() -> Dict[str, Any]:
    """Return measured memory, with zero and notes for unavailable measurements.

    Dedicated GPUs are reported separately: their memory is never summed into
    a fictitious single GPU. Apple unified memory uses reclaimable host RAM.
    """
    total = available = gpu_total = gpu_available = 0.0
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    notes = []
    gpus = []
    try:
        if sys.platform == "darwin":
            total = int(_command("sysctl", "-n", "hw.memsize").strip()) / 1024**3
            stats = _command("vm_stat")
            page_size = int(re.search(r"page size of (\d+) bytes", stats).group(1))
            pages = sum(
                int(value)
                for value in re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)", stats)
            )
            available = pages * page_size / 1024**3
            notes.append("macOS available RAM approximates free, inactive and speculative pages.")
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as file:
                info = dict(re.findall(r"^(MemTotal|MemAvailable):\s+(\d+)", file.read(), re.M))
            total = int(info["MemTotal"]) / 1024**2
            available = int(info.get("MemAvailable", 0)) / 1024**2
        elif sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "total",
                        "available",
                        "page_total",
                        "page_available",
                        "virtual_total",
                        "virtual_available",
                        "extended_available",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            total, available = status.total / 1024**3, status.available / 1024**3
    except (OSError, ValueError, KeyError, AttributeError, subprocess.SubprocessError) as exc:
        notes.append("RAM measurement incomplete: %s" % type(exc).__name__)
    if apple:
        gpu_total, gpu_available = total, available
    else:
        try:
            output = _command(
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            )
            for line in output.strip().splitlines():
                size, free = (float(part.strip()) / 1024 for part in line.split(","))
                gpus.append({"total_gb": round(size, 2), "available_gb": round(free, 2)})
            if gpus:
                largest = max(gpus, key=lambda gpu: gpu["available_gb"])
                gpu_total, gpu_available = largest["total_gb"], largest["available_gb"]
        except (OSError, ValueError, subprocess.SubprocessError):
            notes.append("Dedicated GPU memory unavailable; no supported nvidia-smi measurement.")
    if available <= 0:
        notes.append("Available RAM is unknown; provide available_memory_gb for a recommendation.")
    return {
        "total_ram_gb": round(total, 2),
        "available_ram_gb": round(available, 2),
        "gpu_vram_gb": round(gpu_total, 2),
        "available_gpu_vram_gb": round(gpu_available, 2),
        "gpus": gpus,
        "platform": platform.platform(),
        "is_apple_silicon": apple,
        "notes": notes,
    }


def _positive(value: Any, name: str, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            "%s must be a finite positive %s." % (name, "integer" if integer else "number")
        )


def estimate_model_vram(
    params_b: float,
    quant: str = "Q4_K_M",
    context_length: int = 4096,
    kv_quant: str = "fp16",
    *,
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate weights, KV cache and runtime overhead for one sequence.

    Supply all three architecture dimensions for the model's KV cache formula.
    Otherwise a Llama-like geometry is assumed. Quantization metadata and runtime
    overhead remain approximate even with the exact architecture dimensions.
    """
    _positive(params_b, "params_b")
    _positive(context_length, "context_length", integer=True)
    if not isinstance(quant, str) or quant.upper() not in QUANT_BPW:
        raise ValueError("Unknown weight quantization: %r" % quant)
    precisions = {"fp16": 2.0, "q8_0": 1.0, "q4_0": 0.5}
    if not isinstance(kv_quant, str) or kv_quant.lower() not in precisions:
        raise ValueError("Unknown KV cache quantization: %r" % kv_quant)
    geometry = (num_layers, num_kv_heads, head_dim)
    heuristic = all(value is None for value in geometry)
    if not heuristic:
        for value, name in zip(geometry, ("num_layers", "num_kv_heads", "head_dim")):
            _positive(value, name, integer=True)
    else:
        num_layers, num_kv_heads, head_dim = max(16, int(32 * params_b / 8)), 8, 128
    weights = params_b * 1e9 * QUANT_BPW[quant.upper()] / 8 / 1024**3
    kv = (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * precisions[kv_quant.lower()]
        * context_length
        / 1024**3
    )
    overhead = weights * 0.15 + 0.5
    return {
        "weights_gb": round(weights, 2),
        "kv_cache_gb": round(kv, 2),
        "overhead_gb": round(overhead, 2),
        "total_gb": round(weights + kv + overhead, 2),
        "architecture_assumed": heuristic,
        "assumptions": "One sequence; approximate weight sizes and runtime overhead. "
        + (
            "Llama-like KV geometry inferred from parameter count."
            if heuristic
            else "KV geometry supplied by caller."
        ),
    }


def recommend_quantization(
    params_b: float,
    available_memory_gb: Optional[float] = None,
    context_length: int = 4096,
) -> Dict[str, Any]:
    """Suggest a quantization within 85% of measured or supplied available memory."""
    notes = []
    if available_memory_gb is None:
        memory = get_system_memory()
        notes = memory["notes"]
        available_memory_gb = memory["available_gpu_vram_gb"] or memory["available_ram_gb"]
        if available_memory_gb <= 0:
            raise ValueError("Available memory is unknown. Pass available_memory_gb explicitly.")
    _positive(available_memory_gb, "available_memory_gb")
    candidates = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    best = None
    for quant in candidates:
        estimate = estimate_model_vram(params_b, quant, context_length)
        if estimate["total_gb"] <= available_memory_gb * 0.85:
            best = quant, estimate
            break
    quant, estimate = best or ("Q2_K", estimate_model_vram(params_b, "Q2_K", context_length))
    return {
        "recommended_quant": quant if best else None,
        "status": "fits_comfortably" if best else "exceeds_budget",
        "available_memory_gb": round(available_memory_gb, 2),
        "estimate": estimate,
        "notes": notes,
        "recommendation": (
            "Recommended estimate: %s" % quant
            if best
            else "Warning: no candidate fits the estimated budget"
        )
        + ". Estimates cannot guarantee a fit; verify actual peak memory with your backend.",
    }
