#!/usr/bin/env python3
"""Example: Hardware-Aware VRAM Estimator and Model Sizing Advisor.

Demonstrates:
1. Querying host physical RAM and GPU VRAM.
2. Estimating approximate memory footprints for quantized models and KV caches.
3. Receiving hardware-aware quantization recommendations before downloading.

Run:
    python examples/vram_advisor.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_llm_kit.hardware import estimate_model_vram, get_system_memory, recommend_quantization


def main():
    print("=== Local LLM Kit: Hardware VRAM Advisor ===\n")

    # 1. System memory detection
    mem = get_system_memory()
    print("Host Environment:")
    print(f"  Platform:    {mem['platform']}")
    print(
        f"  Total RAM:   {mem['total_ram_gb']:.1f} GB ({mem['available_ram_gb']:.1f} GB available)"
    )
    if mem["gpu_vram_gb"] > 0:
        desc = "Apple Silicon Unified Memory" if mem["is_apple_silicon"] else "Dedicated GPU VRAM"
        print(f"  GPU Memory:  {mem['gpu_vram_gb']:.1f} GB ({desc})")

    # 2. Compare memory for popular model sizes
    print("\nModel Memory Footprints at 4096 Context:")
    for params in [3.0, 7.0, 8.0, 14.0, 70.0]:
        q4 = estimate_model_vram(params, quant="Q4_K_M", context_length=4096)
        q8 = estimate_model_vram(params, quant="Q8_0", context_length=4096)
        fp16 = estimate_model_vram(params, quant="FP16", context_length=4096)
        print(
            f"  {params:4.1f}B Model: Q4_K_M ~{q4['total_gb']:5.2f} GB | Q8_0 ~{q8['total_gb']:5.2f} GB | FP16 ~{fp16['total_gb']:5.2f} GB"
        )

    # 3. Recommendation for 8B model
    print("\nRecommendation for 8B Model on Current Machine:")
    try:
        rec = recommend_quantization(8.0, context_length=4096)
        print(f"  {rec['recommendation']}")
    except ValueError as exc:
        print(f"  {exc}")


if __name__ == "__main__":
    main()
