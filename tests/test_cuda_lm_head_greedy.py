import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaLmHeadGreedyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seen = (ROOT / "src/backend/cuda/projection.seen").read_text()
        cls.wrapper = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.native = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.hardware = (ROOT / "native/tests/lm_head_greedy_test.cu").read_text()
        cls.contract = " ".join((
            ROOT / "docs/qwen-lm-head-greedy-contract.md"
        ).read_text().lower().split())
        cls.stubs = (
            ROOT / "tests/qwn_044a_project/seen_cuda_link_stubs.c"
        ).read_text() + (
            ROOT / "tests/qwn_044a_project/seen_qwen_cuda_link_stubs.c"
        ).read_text()

    def test_seen_owns_resident_schedule_and_resources(self) -> None:
        for required in (
            "QwenCudaLmHeadExecutor", "createQwenCudaLmHeadExecutor",
            "seen_cublaslt_matmul", "seen_qwen_greedy_argmax_low_precision",
            "this.stream.borrowLaunchToken()", "this.selection.close()",
            "this.handle.close()", "this.stream.close()",
        ):
            self.assertIn(required, self.seen)

    def test_native_boundary_is_low_precision_and_checked(self) -> None:
        for source in (self.wrapper, self.header, self.native):
            self.assertIn("seen_qwen_greedy_argmax_low_precision", source)
        for required in (
            "SEEN_CUDA_F16", "SEEN_CUDA_BF16", "validate_token",
            "validate_view", "checked_multiply(rows, width",
            "overlapping low-precision greedy buffers", "cudaPeekAtLastError",
        ):
            self.assertIn(required, self.native)

    def test_hardware_gate_covers_contract(self) -> None:
        for required in (
            "kOfficialVocabulary", "run_type<__half>",
            "run_type<__nv_bfloat16>", "expected_tokens[2] = {3, 7}",
            "seen_cuda_graph_begin_capture", "iteration < 1000",
            "SEEN_CUDA_INVALID_ARGUMENT", "SEEN_CUDA_INCOMPATIBLE",
        ):
            self.assertIn(required, self.hardware)

    def test_contract_is_explicit_and_fail_closed(self) -> None:
        for required in (
            "[tokens,5120] * [248320,5120]^t -> [tokens,248320]",
            "equal finite logits select the lower token id",
            "performs no allocation", "no performance-selection claim",
            "never silently replaced with greedy selection",
        ):
            self.assertIn(required, self.contract)

    def test_cpu_link_fixture_cannot_fallback(self) -> None:
        for required in (
            "seen_cublaslt_matmul", "seen_qwen_greedy_argmax_low_precision",
            "__builtin_trap()",
        ):
            self.assertIn(required, self.stubs)
        for forbidden in ("cuda_runtime", "cudaSuccess", "malloc", "return 0"):
            self.assertNotIn(forbidden, self.stubs)


if __name__ == "__main__":
    unittest.main()
