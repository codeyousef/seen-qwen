import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaReferencePrimitiveContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "native/include/seen_qwen_cuda.h").read_text()
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.wrapper = (
            ROOT / "src/backend/cuda/reference_primitives.seen"
        ).read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_040a.sh").read_text()

    def test_public_boundary_is_fixed_width_and_cuda_free(self) -> None:
        self.assertIn('#include "seen_cuda.h"', self.header)
        self.assertNotIn("cuda_runtime", self.header)
        self.assertNotIn("cudaStream_t", self.header)
        self.assertNotIn("SeenCudaHandle", self.header)
        for field in ("abi_version", "device_ordinal", "address", "byte_length"):
            self.assertIn(field, self.header)

    def test_reference_surface_is_complete_and_qwen_owned(self) -> None:
        symbols = (
            "seen_qwen_fill_f32",
            "seen_qwen_add_f32",
            "seen_qwen_row_sum_f32",
            "seen_qwen_transpose_2d_f32",
            "seen_qwen_copy_f32",
            "seen_qwen_embedding_gather_f32",
        )
        for symbol in symbols:
            self.assertIn(symbol, self.header)
            self.assertIn(symbol, self.source)
            self.assertIn(symbol, self.wrapper)
        self.assertNotIn("seen_cuda_", self.header.replace("seen_cuda.h", ""))

    def test_adapter_only_enqueues_on_the_borrowed_stream(self) -> None:
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
        self.assertEqual(self.source.count("<<<"), 16)
        self.assertEqual(self.source.count(", 0, stream>>>"), 16)
        self.assertIn("token->generation == 0", self.source)
        self.assertIn("cudaPointerGetAttributes", self.source)

    def test_hardware_gate_is_exact_bounded_and_sanitized(self) -> None:
        for required in (
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
            "bfed49cea60c983751c26cef81b21e3374360f3a43de677e8134c14a3c30a158",
            "69441bbf20755f0bbf12a4241fffadf3ad20df6f9d1155a6b0b5ab92992e9e2c",
            "scripts/oracle/run_bounded.sh",
            "QWN_TASKS_MAX=32",
            "--parallel 1",
            "compute-sanitizer",
            "--leak-check full",
            "nvidia-smi",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
