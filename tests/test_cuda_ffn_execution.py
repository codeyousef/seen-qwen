import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaFfnExecutionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seen = (ROOT / "src/backend/cuda/projection.seen").read_text()
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.native = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.hardware = (ROOT / "native/tests/ffn_execution_test.cu").read_text()
        cls.runtime_stub = (
            ROOT / "tests/qwn_043b_project/seen_cuda_link_stubs.c"
        ).read_text()
        cls.qwen_stub = (
            ROOT / "tests/qwn_043b_project/seen_qwen_cuda_link_stubs.c"
        ).read_text()
        cls.contract = " ".join((
            ROOT / "docs/qwen-ffn-cuda-execution-contract.md"
        ).read_text().lower().split())

    def test_seen_owns_exact_schedule_and_resources(self) -> None:
        for required in (
            "QwenCudaFfnExecutor", "createQwenCudaFfnExecutor",
            "seen_cublaslt_matmul", "seen_qwen_swiglu_low_precision",
            "this.stream.borrowLaunchToken()", "gate.n * gate.m",
            "this.gateSelection.close()", "this.handle.close()",
            "this.stream.close()", "if not this.active",
        ):
            self.assertIn(required, self.seen)
        self.assertNotIn("/usr/local/bin/seen", self.seen)

    def test_native_boundary_is_narrow_and_checked(self) -> None:
        self.assertIn("seen_qwen_swiglu_low_precision", self.header)
        for required in (
            "SEEN_CUDA_F16", "SEEN_CUDA_BF16", "validate_token",
            "validate_view", "checked_multiply(count, 2",
            "gate_output_alias", "cudaPeekAtLastError",
        ):
            self.assertIn(required, self.native)

    def test_hardware_gate_covers_required_contract(self) -> None:
        for required in (
            "seen_cublaslt_matmul", "seen_cuda_graph_begin_capture",
            "seen_cuda_graph_launch", "run_type<__half>",
            "run_type<__nv_bfloat16>", "iteration < 1000",
            "SEEN_CUDA_INVALID_ARGUMENT", "SEEN_CUDA_INCOMPATIBLE",
            "seen_cublaslt_destroy(&cublas)",
            "seen_cuda_stream_destroy(&stream)",
        ):
            self.assertIn(required, self.hardware)

    def test_contract_prohibits_policy_and_fallback(self) -> None:
        for required in (
            "down_proj(silu(gate_proj(x)) * up_proj(x))",
            "performs no allocation", "default-stream", "no performance-selection claim",
            "does not trigger cpu execution", "close` is idempotent",
        ):
            self.assertIn(required, self.contract)

    def test_cpu_ci_link_fixture_is_fail_closed(self) -> None:
        stubs = self.runtime_stub + self.qwen_stub
        for required in (
            "seen_cuda_stream_create", "seen_cuda_stream_destroy",
            "seen_cublaslt_create", "seen_cublaslt_destroy",
            "seen_cublaslt_select_algorithm", "seen_cublaslt_matmul",
            "seen_qwen_swiglu_low_precision", "__builtin_trap()",
        ):
            self.assertIn(required, stubs)
        for forbidden in (
            "cuda_runtime", "cudaSuccess", "malloc", "calloc", "realloc",
            "cpu fallback", "return 0",
        ):
            self.assertNotIn(forbidden, stubs)


if __name__ == "__main__":
    unittest.main()
