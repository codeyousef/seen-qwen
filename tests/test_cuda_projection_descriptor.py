import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaProjectionDescriptorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seen = (ROOT / "src/backend/cuda/projection.seen").read_text()
        cls.native = (ROOT / "native/tests/projection_descriptor_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_043a.sh").read_text()
        cls.contract = " ".join((
            ROOT / "docs/qwen-cublaslt-projection-selection-contract.md"
        ).read_text().lower().split())

    def test_seen_layer_owns_only_qwen_projection_policy(self) -> None:
        for required in (
            "qwenCudaProjectionDescriptor", "selectQwenCudaProjection",
            "transposeA: 1 as Int32", "m: outputWidth", "n: tokenCount",
            "k: inputWidth", "handle.selectAlgorithm(descriptor)",
            "algorithm.cacheIdentity != (0 as UInt64)", "fun close() r: Bool",
        ):
            self.assertIn(required, self.seen)

    def test_contract_is_exact_and_non_owning(self) -> None:
        for required in (
            "y^t = w * x^t", "only f16 and bf16", "there is no allocation",
            "never stores, closes, or decodes that handle", "never serialized",
            "qwn-043b", "idempotent `close`",
        ):
            self.assertIn(required, self.contract)

    def test_hardware_gate_covers_identity_bounds_and_cleanup(self) -> None:
        for required in (
            "5120", "17408", "6144", "seen_cublaslt_select_algorithm",
            "same_algorithm", "different.cache_identity",
            "SEEN_CUDA_BF16", "SEEN_CUDA_F16", "seen_cublaslt_destroy",
        ):
            self.assertIn(required, self.native)

    def test_runner_is_contained_and_audits_toolchain(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "qwn_043a_cuda_test", "compute-sanitizer", "--leak-check full",
            "nvidia-smi", "03a06cc002355251b7aeea3539a3ceb466d447733a66e5b0ee3ab8c184672124",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
