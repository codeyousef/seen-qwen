import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaAttentionQkContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.wrapper = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.test = (ROOT / "native/tests/attention_qk_geometry_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_042a.sh").read_text()
        cls.contract = (ROOT / "docs/qwen-attention-qk-cuda-reference-contract.md").read_text()

    def test_surface_preserves_projection_and_gate_layout(self) -> None:
        symbol = "seen_qwen_attention_qk_rope_f32"
        for text in (self.header, self.source, self.wrapper):
            self.assertIn(symbol, text)
        for required in (
            "query_gate_projection", "key_projection", "query_norm_weight",
            "key_norm_weight", "query_output", "key_output", "gate_output",
            "query_heads", "kv_heads", "rotary_dim", "position_offset",
        ):
            self.assertIn(required, self.header)

    def test_kernels_use_only_the_borrowed_seen_stream(self) -> None:
        self.assertIn("attention_query_rope_f32<<<", self.source)
        self.assertIn("attention_key_rope_f32<<<", self.source)
        self.assertEqual(self.source.count("<<<"), 25)
        self.assertEqual(self.source.count(", 0, stream>>>"), 25)
        for forbidden in (
            "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaStreamCreate",
            "cudaStreamDestroy", "cudaMalloc", "cudaFree",
            "seen_cuda_stream_borrow_launch_token",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_official_geometry_and_fail_closed_paths_are_covered(self) -> None:
        for required in (
            "kQueryHeads = 24", "kKvHeads = 4", "kHeadDim = 256",
            "kRotaryDim = 64", "kTheta = 10000000.0f",
            "kMaxPosition = 262144", "Partial RoPE leaves dimensions 64..255",
            "Invalid bounds, geometry, overlap, device identity",
            "std::numeric_limits<uint64_t>::max()",
            "seen_cuda_graph_begin_capture", "SEEN_CUDA_INCOMPATIBLE",
        ):
            self.assertIn(required, self.test)
        for required in (
            "1 + weight", "Normalization precedes RoPE", "no aliasing",
            "no retry", "no fallback",
        ):
            self.assertIn(required.lower(), self.contract.lower())

    def test_runner_is_hard_scoped_and_sanitized(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "run_qwn_041d.sh", "qwn_042a_cuda_test", "compute-sanitizer",
            "--leak-check full", "nvidia-smi",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
