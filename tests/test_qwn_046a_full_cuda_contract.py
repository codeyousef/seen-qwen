#!/usr/bin/env python3
"""Static ownership contract for QWN-046A complete-model CUDA residency."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/runtime/full_model_cuda.seen"
NATIVE = ROOT / "native/tests/full_model_residency_test.cu"
RUNNER = ROOT / "scripts/cuda/run_qwn_046a.sh"


class FullCudaContractTests(unittest.TestCase):
    def test_seen_owns_coarse_allocations_and_cleanup(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        for required in (
            "createQwenFullCudaResidency", "QwenFullCudaResidency",
            "Array<CudaAllocation>", "CudaStream.create", "CudaDevice.open",
            "device.freeMemoryBytes < plan.allocationBytes",
            "closeQwenFullAllocations", "if this.closed",
        ):
            self.assertIn(required, source)
        for forbidden in ("cudaMalloc", "cudaFree", "cudaDeviceSynchronize",
                          "fallback", "offload"):
            self.assertNotIn(forbidden, source)

    def test_hardware_gate_uses_actual_sqw_and_seen_runtime(self) -> None:
        native = NATIVE.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        for required in (
            "kTensorCount = 866", "kWeightBytes = 14515042384ULL",
            "seen_cuda_malloc", "seen_cuda_memcpy_async",
            "seen_cuda_stream_synchronize", "seen_cuda_free",
            "allocated_bytes == kAllocationBytes",
            "resident.total_memory_bytes - allocated_bytes",
            "allocations[index - 1].handle == 0", "host == 0 && stream == 0",
        ):
            self.assertIn(required, native)
        self.assertNotIn("cudaDeviceSynchronize", native)
        for required in ("run_bounded.sh", "QWN_TASKS_MAX=32",
                         "compute-sanitizer",
                         "NVIDIA GeForce RTX 4090",
                         "qwn_046a_full_cuda_hardware_test.seen"):
            self.assertIn(required, runner)
        self.assertNotIn("/usr/local/bin/seen", runner)


if __name__ == "__main__":
    unittest.main()
