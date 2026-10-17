"""Tests for hardware inspection and VRAM estimation."""

import unittest

from local_llm_kit.hardware import estimate_model_vram, get_system_memory, recommend_quantization


class TestHardware(unittest.TestCase):
    def test_get_system_memory(self):
        mem = get_system_memory()
        self.assertIn("total_ram_gb", mem)
        self.assertIn("available_ram_gb", mem)
        self.assertIn("gpu_vram_gb", mem)
        self.assertGreater(mem["total_ram_gb"], 0.0)

    def test_estimate_model_vram_formula(self):
        # 8B model at Q4_K_M (4.5 bpw) @ 4096 context
        est = estimate_model_vram(params_b=8.0, quant="Q4_K_M", context_length=4096)
        self.assertGreater(est["weights_gb"], 3.5)
        self.assertLess(est["weights_gb"], 5.0)
        self.assertGreater(est["kv_cache_gb"], 0.1)
        self.assertGreater(est["total_gb"], est["weights_gb"])

    def test_estimate_model_vram_quant_scaling(self):
        # Q8_0 weights must be larger than Q4_K_M
        est_q4 = estimate_model_vram(params_b=7.0, quant="Q4_K_M", context_length=4096)
        est_q8 = estimate_model_vram(params_b=7.0, quant="Q8_0", context_length=4096)
        est_fp16 = estimate_model_vram(params_b=7.0, quant="FP16", context_length=4096)

        self.assertLess(est_q4["weights_gb"], est_q8["weights_gb"])
        self.assertLess(est_q8["weights_gb"], est_fp16["weights_gb"])

    def test_recommend_quantization_fit(self):
        # With 24GB available, an 8B model should comfortably fit Q8_0 or FP16
        rec = recommend_quantization(params_b=8.0, available_memory_gb=24.0, context_length=4096)
        self.assertEqual(rec["status"], "fits_comfortably")
        self.assertIn("Recommended", rec["recommendation"])

        # With only 3GB available, an 8B model will exceed budget
        rec_tight = recommend_quantization(
            params_b=8.0, available_memory_gb=3.0, context_length=4096
        )
        self.assertEqual(rec_tight["status"], "exceeds_budget")
        self.assertIn("Warning", rec_tight["recommendation"])


if __name__ == "__main__":
    unittest.main()
