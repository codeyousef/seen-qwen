import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaGdnStateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.wrapper = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.test = (ROOT / "native/tests/gdn_causal_conv_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_041a.sh").read_text()

    def test_surface_is_model_owned_and_fixed_width(self) -> None:
        symbol = "seen_qwen_causal_conv_silu_f32"
        for text in (self.header, self.source, self.wrapper):
            self.assertIn(symbol, text)
        self.assertIn("uint64_t processed_position", self.header)
        self.assertIn("SeenQwenCudaBufferView history", self.header)

    def test_state_is_borrowed_and_launch_is_on_seen_stream(self) -> None:
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
        self.assertEqual(self.source.count("<<<"), 21)
        self.assertEqual(self.source.count(", 0, stream>>>"), 21)
        self.assertIn("start_position != processed_position", self.source)
        self.assertIn("kernel_length != 4", self.source)

    def test_corpus_covers_frozen_geometry_and_fail_closed_paths(self) -> None:
        for required in (
            "kMaximumChannels = 10240",
            "Arbitrary chunk boundaries",
            "Exact input/output aliasing",
            "Host-side rejection occurs before state mutation",
            "std::numeric_limits<uint64_t>::max()",
            "seen_cuda_graph_begin_capture",
            "SEEN_CUDA_INCOMPATIBLE",
        ):
            self.assertIn(required, self.test)

    def test_runner_is_hard_scoped_and_sanitized(self) -> None:
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
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
