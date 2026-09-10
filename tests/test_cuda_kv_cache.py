import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaKvCacheContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.native = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.test = (ROOT / "native/tests/kv_cache_ownership_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_042b.sh").read_text()
        cls.contract = (ROOT / "docs/qwen-kv-cache-cuda-ownership-contract.md").read_text()

    def test_seen_owner_controls_allocations_and_length(self) -> None:
        for required in (
            "@move\npub class QwenCudaKvCache",
            "keyAllocation: CudaAllocation", "valueAllocation: CudaAllocation",
            "createQwenCudaKvCache", "CudaAllocation.allocate",
            "startPosition != this.used", "this.used = this.used + tokenCount",
            "this.valueAllocation.close()", "this.keyAllocation.close()",
        ):
            self.assertIn(required, self.source)
        self.assertLess(
            self.source.index("if not cudaSucceeded(status)"),
            self.source.index("this.used = this.used + tokenCount"),
        )

    def test_native_boundary_remains_borrowed_and_allocation_free(self) -> None:
        self.assertEqual(self.native.count("<<<"), 21)
        self.assertEqual(self.native.count(", 0, stream>>>"), 21)
        for forbidden in (
            "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaStreamCreate",
            "cudaStreamDestroy", "cudaMalloc", "cudaFree",
            "seen_cuda_stream_borrow_launch_token",
        ):
            self.assertNotIn(forbidden, self.native)

    def test_hardware_gate_covers_state_transitions_and_cleanup(self) -> None:
        for required in (
            "kKvHeads = 4", "kHeadDim = 256", "start_position != used",
            "cache.used == 3", "cache.used == 5", "cache.reset()",
            "original_keys", "seen_cuda_graph_begin_capture",
            "incompatible.generation = 0", "cache.close()",
        ):
            self.assertIn(required, self.test)
        for required in (
            "authoritative used length", "stable addresses", "idempotent",
            "never retain", "never own", "no fallback",
        ):
            self.assertIn(required.lower(), self.contract.lower())

    def test_runner_is_hard_scoped_and_sanitized(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "run_qwn_042a.sh", "qwn_042b_cuda_test", "compute-sanitizer",
            "--leak-check full", "nvidia-smi",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
