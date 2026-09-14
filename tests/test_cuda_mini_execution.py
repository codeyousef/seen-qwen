import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaMiniExecutionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.native = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.harness = (ROOT / "native/tests/mini_model_execution_test.cu").read_text()
        cls.seen = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.contract = " ".join((
            ROOT / "docs/qwen-cuda-mini-execution-contract.md"
        ).read_text().lower().split())

    def test_missing_composition_kernels_are_bounded_adapters(self) -> None:
        for symbol in (
            "seen_qwen_linear_f32",
            "seen_qwen_gdn_prepare_f32",
            "seen_qwen_gdn_parameters_f32",
        ):
            self.assertIn(symbol, self.header)
            self.assertIn(symbol, self.native)
            self.assertIn(symbol, self.seen)
        for required in (
            "validate_token", "validate_view", "checked_multiply",
            "cudaPeekAtLastError", "FP32 linear output must not overlap inputs",
            "GDN preparation buffers must be disjoint",
            "GDN parameter buffers must be disjoint",
        ):
            self.assertIn(required, self.native)

    def test_harness_executes_the_frozen_hybrid_schedule(self) -> None:
        for required in (
            "model.language_model.embed_tokens.weight",
            "layer_index == 3 || layer_index == 7",
            "seen_qwen_gdn_recurrent_prefill_f32",
            "seen_qwen_gdn_recurrent_decode_f32",
            "seen_qwen_attention_prefill_f32",
            "seen_qwen_attention_decode_f32",
            "model.language_model.norm.weight", "lm_head.weight",
            "seen_qwen_greedy_argmax_f32",
        ):
            self.assertIn(required, self.harness)

    def test_execution_is_resident_bounded_and_seen_stream_ordered(self) -> None:
        for required in (
            "kContext = 128", "kScratchSlots = 20",
            "kPersistentFloats", "seen_cuda_stream_borrow_launch_token",
            "seen_cuda_memcpy_async", "seen_cuda_event_synchronize",
            "seen_cuda_free", "seen_cuda_stream_destroy",
        ):
            self.assertIn(required, self.harness)
        for forbidden in (
            "cudaDeviceSynchronize", "cudaStreamCreate", "cudaMalloc(",
            "cudaFree(", "PyTorch", "LibTorch",
        ):
            self.assertNotIn(forbidden, self.harness)

    def test_acceptance_covers_differential_and_lifecycle(self) -> None:
        for required in (
            "max_abs_error", "greedy_token_ids", "decode_logits",
            "!engine.decode", "engine.cancel", "engine.reset",
            "engine.close", "engine.close", "fail_allocation_at",
            "certify_allocation_failures", "failure <= 4",
        ):
            self.assertIn(required, self.harness)
        for required in (
            "no silent fallback", "one seen-owned stream",
            "all allocations occur before execution",
            "conformance harness", "not a production policy owner",
            "required local verification",
        ):
            self.assertIn(required, self.contract)


if __name__ == "__main__":
    unittest.main()
