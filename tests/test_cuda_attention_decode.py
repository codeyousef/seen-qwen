import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaAttentionDecodeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.seen = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.test = (ROOT / "native/tests/attention_decode_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_042c.sh").read_text()
        cls.contract = (
            ROOT / "docs/qwen-attention-decode-cuda-reference-contract.md"
        ).read_text()

    def test_fixed_width_seen_and_c_abi_surface(self) -> None:
        symbol = "seen_qwen_attention_decode_f32"
        self.assertIn(symbol, self.header)
        self.assertIn(f'extern "C" fun {symbol}', self.seen)
        self.assertIn(f'extern "C" SeenCudaStatus {symbol}', self.source)
        self.assertIn("fun decode(token: *const CudaStreamLaunchToken", self.seen)
        self.assertIn("this.keyView(), this.valueView()", self.seen)

    def test_kernel_is_stable_bounded_and_on_borrowed_stream(self) -> None:
        for required in (
            "const uint64_t kv_head = query_head / (query_heads / kv_heads)",
            "rsqrtf(static_cast<float>(head_dim))",
            "running_max", "denominator", "fmaxf", "expf",
            "cache_length > cache_capacity", "checked_multiply",
            "attention_decode_f32<<<static_cast<uint32_t>(query_heads), kThreads, 0, stream>>>",
        ):
            self.assertIn(required, self.source)
        for forbidden in (
            "cudaDeviceSynchronize", "cudaStreamSynchronize",
            "cudaStreamCreate", "cudaMalloc", "cudaFree",
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertEqual(self.source.count("<<<"), 25)
        self.assertEqual(self.source.count(", 0, stream>>>"), 25)

    def test_hardware_corpus_is_official_and_negative_complete(self) -> None:
        for required in (
            "kQueryHeads = 24", "kKvHeads = 4", "kHeadDim = 256",
            "reference(query, keys, values", "std::isfinite",
            "kCacheCapacity + 1", "numeric_limits<uint64_t>::max()",
            "incompatible.generation = 0", "seen_cuda_graph_begin_capture",
            "seen_cuda_graph_launch", "seen_cuda_free",
        ):
            self.assertIn(required, self.test)
        for required in (
            "overflow-safe online softmax", "exact ledgered seen stream",
            "no allocation", "no silent fallback", "correctness oracle",
        ):
            self.assertIn(required, self.contract.lower())

    def test_runner_is_hard_scoped_and_sanitized(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "run_qwn_042b.sh", "qwn_042c_cuda_test", "compute-sanitizer",
            "--leak-check full", "nvidia-smi",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
