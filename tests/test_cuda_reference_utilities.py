import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaReferenceUtilityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.wrapper = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_040b.sh").read_text()

    def test_complete_qwn_040b_surface(self) -> None:
        symbols = (
            "seen_qwen_rms_norm_f32",
            "seen_qwen_l2_norm_f32",
            "seen_qwen_silu_f32",
            "seen_qwen_swiglu_f32",
            "seen_qwen_sigmoid_gate_f32",
            "seen_qwen_partial_rope_f32",
            "seen_qwen_kv_append_f32",
            "seen_qwen_greedy_argmax_f32",
            "seen_qwen_top_k_f32",
        )
        for symbol in symbols:
            self.assertIn(symbol, self.header)
            self.assertIn(symbol, self.source)
            self.assertIn(symbol, self.wrapper)

    def test_adapter_preserves_seen_stream_ownership(self) -> None:
        for forbidden in (
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaStreamCreate",
            "cudaStreamDestroy",
            "cudaMalloc",
            "cudaFree",
            "seen_cuda_stream_borrow_launch_token",
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertEqual(self.source.count("<<<"), 22)
        self.assertEqual(self.source.count(", 0, stream>>>"), 22)
        self.assertNotIn("/usr/local/bin/seen", self.runner)
        self.assertNotIn("sudo", self.runner)

    def test_runner_is_bounded_and_audited(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh",
            "QWN_TASKS_MAX=32",
            "--parallel 1",
            "compute-sanitizer",
            "--leak-check full",
            "nvidia-smi",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
        ):
            self.assertIn(required, self.runner)


if __name__ == "__main__":
    unittest.main()
