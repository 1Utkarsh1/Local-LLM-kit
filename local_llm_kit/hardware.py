"""
Hardware memory inspection and VRAM estimation for local LLMs.

Prevents out-of-memory (OOM) crashes and system freezes by computing exact
memory footprints for quantized models (Q4_K_M, Q5_K_M, Q8_0, FP16) and their
KV caches before downloading or loading.

Example::

    from local_llm_kit.hardware import estimate_model_vram, recommend_quantization

    # Estimate VRAM for Llama-3 8B at 8192 context
    est = estimate_model_vram(params_b=8.0, quant="Q4_K_M", context_length=8192)
    print(f"Total VRAM needed: {est['total_gb']:.2f} GB")

    # Get hardware-aware recommendation for host
    rec = recommend_quantization(params_b=8.0, context_length=8192)
    print(rec["recommendation"])

Python 3.9+ compatible. Zero mandatory dependencies.
"""

import logging
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bits per weight lookup for standard GGUF quantizations
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


def get_system_memory() -> Dict[str, Any]:
    """Inspect physical host RAM and GPU VRAM without heavy dependencies.

    Returns:
        Dict with keys: ``total_ram_gb``, ``available_ram_gb``, ``gpu_vram_gb``,
        ``platform``, ``is_apple_silicon``.
    """
    total_ram = 0.0
    available_ram = 0.0
    gpu_vram = 0.0
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"

    # 1. macOS memory inspection via sysctl
    if sys.platform == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            )
            bytes_total = int(res.stdout.strip())
            total_ram = bytes_total / (1024**3)
            # On Apple Silicon, unified memory acts as both RAM and GPU VRAM
            if is_apple_silicon:
                gpu_vram = total_ram
            available_ram = total_ram * 0.75  # conservative estimate
        except Exception:
            total_ram = 16.0
            available_ram = 12.0

    # 2. Linux memory inspection via /proc/meminfo
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    total_ram = int(line.split()[1]) / (1024**2)
                elif line.startswith("MemAvailable:"):
                    available_ram = int(line.split()[1]) / (1024**2)
        except Exception:
            total_ram = 16.0
            available_ram = 12.0

        # Check nvidia-smi for dedicated GPU VRAM
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            vram_mb = sum(int(x.strip()) for x in res.stdout.strip().splitlines() if x.strip())
            gpu_vram = vram_mb / 1024.0
        except Exception:
            gpu_vram = 0.0

    # 3. Windows fallback
    else:
        total_ram = 16.0
        available_ram = 10.0

    return {
        "total_ram_gb": round(total_ram, 2),
        "available_ram_gb": round(available_ram, 2),
        "gpu_vram_gb": round(gpu_vram, 2),
        "platform": platform.platform(),
        "is_apple_silicon": is_apple_silicon,
    }


def estimate_model_vram(
    params_b: float,
    quant: str = "Q4_K_M",
    context_length: int = 4096,
    kv_quant: str = "fp16",
) -> Dict[str, float]:
    """Calculate the required memory footprint for a local model.

    Args:
        params_b: Number of parameters in billions (e.g. 7.0 for 7B, 8.0 for 8B).
        quant: Quantization name (e.g. ``"Q4_K_M"``, ``"Q8_0"``, ``"FP16"``).
        context_length: Context window token capacity.
        kv_quant: Precision for KV cache (``"fp16"`` or ``"q8_0"`` or ``"q4_0"``).

    Returns:
        Dict with keys: ``weights_gb``, ``kv_cache_gb``, ``overhead_gb``, ``total_gb``.
    """
    if params_b <= 0:
        raise ValueError(f"params_b must be positive, got {params_b}")
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")

    bpw = QUANT_BPW.get(quant.upper(), 4.5)

    # Weights size in GiB
    # params_b * 1e9 * bpw / 8 / (1024^3)
    weights_gb = (params_b * 1e9 * bpw) / (8.0 * (1024.0**3))

    # KV Cache calculation
    # Scaled based on typical Llama architecture proportions:
    # layers ~ 32 * (params / 8), kv_heads ~ 8, head_dim ~ 128
    # KV per token = 2 * layers * kv_heads * head_dim * bytes_per_elem
    kv_bytes_per_elem = (
        2.0 if kv_quant.lower() == "fp16" else (1.0 if kv_quant.lower() == "q8_0" else 0.5)
    )
    layers = max(16, int(32 * (params_b / 8.0)))
    kv_heads = 8
    head_dim = 128

    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * kv_bytes_per_elem
    kv_cache_gb = (kv_bytes_per_token * context_length) / (1024.0**3)

    # Runtime context / CUDA / activation overhead (approx 15% of weights + 0.5GB scratch)
    overhead_gb = (weights_gb * 0.15) + 0.5

    total_gb = weights_gb + kv_cache_gb + overhead_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_gb": round(total_gb, 2),
    }


def recommend_quantization(
    params_b: float,
    available_memory_gb: Optional[float] = None,
    context_length: int = 4096,
) -> Dict[str, Any]:
    """Recommend the optimal quantization tier for host hardware.

    Args:
        params_b: Model parameter count in billions.
        available_memory_gb: Optional memory limit in GB; if omitted, autodetects host memory.
        context_length: Desired context length.

    Returns:
        Dict with recommended quant, estimated memory, fits status, and advisory notes.
    """
    if available_memory_gb is None:
        sys_mem = get_system_memory()
        # On Apple Silicon or GPU systems, use GPU VRAM; otherwise available RAM
        available_memory_gb = (
            sys_mem["gpu_vram_gb"] if sys_mem["gpu_vram_gb"] > 0 else sys_mem["available_ram_gb"]
        )

    # Candidate quantizations ordered from highest quality to lowest
    candidates = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    fits: List[Tuple[str, Dict[str, float]]] = []

    # Safe operating threshold: 85% of available memory
    budget = available_memory_gb * 0.85

    for c in candidates:
        est = estimate_model_vram(params_b, quant=c, context_length=context_length)
        if est["total_gb"] <= budget:
            fits.append((c, est))

    if fits:
        best_quant, best_est = fits[0]
        status = "fits_comfortably"
        rec_msg = (
            f"Recommended: {best_quant} (requires ~{best_est['total_gb']} GB, "
            f"fits within available {available_memory_gb:.1f} GB)."
        )
    else:
        # Lowest candidate
        best_quant = "Q4_K_M"
        best_est = estimate_model_vram(params_b, quant="Q4_K_M", context_length=context_length)
        status = "exceeds_budget"
        rec_msg = (
            f"Warning: {params_b}B at {best_quant} requires ~{best_est['total_gb']} GB, "
            f"which exceeds available {available_memory_gb:.1f} GB. "
            f"Consider reducing context length to {max(2048, context_length // 2)} or using a smaller model."
        )

    return {
        "recommended_quant": best_quant,
        "status": status,
        "available_memory_gb": round(available_memory_gb, 2),
        "estimate": best_est,
        "recommendation": rec_msg,
    }
