import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaAttentionPrefillContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.seen = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.test = (ROOT / "native/tests/attention_prefill_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_042d.sh").read_text()
        cls.contract = (ROOT / "docs/qwen-attention-prefill-cuda-integration-contract.md").read_text()

    def test_fixed_width_integrated_surface(self) -> None:
        symbol = "seen_qwen_attention_prefill_f32"
        self.assertIn(symbol, self.header)
        self.assertIn(f'extern "C" fun {symbol}', self.seen)
        self.assertIn(f'extern "C" SeenCudaStatus {symbol}', self.source)
        self.assertIn("fun prefill(token: *const CudaStreamLaunchToken", self.seen)
        self.assertIn("this.used = this.used + tokenCount", self.seen)

    def test_single_enqueue_is_causal_stable_and_borrowed(self) -> None:
        for required in (
            "One correctness-oracle block integrates cache mutation",
            "const uint64_t causal_length = start_position + token_index + 1",
            "query_head / (query_heads / kv_heads)",
            "running_max", "denominator", "fmaxf", "expf",
            "attention_prefill_f32<<<1, kThreads, 0, stream>>>",
            "token_count > cache_capacity - start_position",
        ):
            self.assertIn(required, self.source)
        for forbidden in (
            "cudaDeviceSynchronize", "cudaStreamSynchronize",
            "cudaStreamCreate", "cudaMalloc", "cudaFree",
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertEqual(self.source.count("<<<"), 25)
        self.assertEqual(self.source.count(", 0, stream>>>"), 25)

    def test_hardware_corpus_covers_chunking_and_failures(self) -> None:
        for required in (
            "kQueryHeads = 24", "kKvHeads = 4", "kHeadDim = 256",
            "Arbitrary chunking preserves causal outputs",
            "reference(query, keys, values", "std::isfinite",
            "numeric_limits<uint64_t>::max()", "incompatible.generation = 0",
            "seen_cuda_graph_begin_capture", "seen_cuda_graph_launch",
            "seen_cuda_free",
        ):
            self.assertIn(required, self.test)
        for required in (
            "arbitrary chunk boundaries", "one accepted launch",
            "does not allocate", "silently fall back", "correctness oracle",
        ):
            self.assertIn(required, self.contract.lower())

    def test_runner_is_hard_scoped_and_sanitized(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "run_qwn_042c.sh", "qwn_042d_cuda_test", "compute-sanitizer",
            "--leak-check full", "nvidia-smi",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
