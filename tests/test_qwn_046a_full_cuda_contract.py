#!/usr/bin/env python3
"""Static ownership contract for QWN-046A complete-model CUDA residency."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/runtime/full_model_cuda.seen"
MEMORY = ROOT / "src/runtime/full_model_memory.seen"
NATIVE = ROOT / "native/tests/full_model_residency_test.cu"
RUNNER = ROOT / "scripts/cuda/run_qwn_046a.sh"
ORACLE = ROOT / "tests/fixtures/qwn_025b_full_model_oracles.json"
SEQUENCE_TEST = ROOT / "tests/qwn_046b_full_cuda_oracle_test.seen"
SEQUENCE_RUNNER = ROOT / "scripts/cuda/run_qwn_046b.sh"


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
        self.assertIn("let capacity = this.plan.contextTokens", source)
        self.assertIn("cacheCapacity: capacity", source)
        self.assertNotIn("cacheCapacity: 128 as UInt64", source)
        self.assertIn("Result<QwenFullStepResult, CudaError>", source)
        self.assertIn("selectedLogit: selectedLogits[0]", source)

        native = (ROOT / "native/src/reference_primitives.cu").read_text(
            encoding="utf-8"
        )
        self.assertIn("constexpr uint64_t kFullContext = 188", native)
        self.assertIn("seen_qwen_top_k_f32(token, logits, token_ids", native)
        self.assertNotIn("constexpr uint64_t kFullContext = 128", native)

    def test_capacity_covers_frozen_generated_token_sequences(self) -> None:
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        required_positions = max(
            prompt["input_token_count"] + len(prompt["greedy_token_ids"])
            for prompt in oracle["prompts"]
            if prompt["greedy_sequence_role"] == "128-step"
        )
        self.assertEqual(required_positions, 188)
        memory = MEMORY.read_text(encoding="utf-8")
        self.assertIn(
            "QWEN_FULL_MAX_CONTEXT: UInt64 = 188 as UInt64", memory
        )

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

    def test_two_prompt_oracle_gate_is_bounded_and_gpu_polite(self) -> None:
        test = SEQUENCE_TEST.read_text(encoding="utf-8")
        runner = SEQUENCE_RUNNER.read_text(encoding="utf-8")
        for required in (
            'role == "128-step"', "while step < 128",
            "firstTokenDivergence", "firstLogitDivergence",
            "maximumSelectedLogitError",
            "(inputValues.length() + 128) as UInt64",
            "commit every generated token in the frozen sequence",
            "residency.position == QWEN_FULL_MAX_CONTEXT",
            "comparedPrompts == 2", "allocatorAfter.usedBytes ==",
        ):
            self.assertIn(required, test)
        for required in (
            "run_bounded.sh", "QWN_TASKS_MAX=24", "--parallel 1",
            "--release --lto=thin", "--sanitize undefined",
            "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1",
            "compute-sanitizer",
            'QWEN_GPU_UUID=$("$ROOT_DIR/scripts/cuda/require_idle_gpu.sh")',
            'export CUDA_VISIBLE_DEVICES="$QWEN_GPU_UUID"',
            'nvidia-smi --id="$QWEN_GPU_UUID"',
        ):
            self.assertIn(required, runner)
        for forbidden in ("sudo", "pkexec", "kill ", "pkill", "killall"):
            self.assertNotIn(forbidden, runner)


if __name__ == "__main__":
    unittest.main()
