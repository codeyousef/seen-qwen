import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CudaAttentionOutputGateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "native/src/reference_primitives.cu").read_text()
        cls.seen = (ROOT / "src/backend/cuda/reference_primitives.seen").read_text()
        cls.test = (ROOT / "native/tests/attention_output_gate_test.cu").read_text()
        cls.runner = (ROOT / "scripts/cuda/run_qwn_042e.sh").read_text()
        cls.contract = " ".join((
            ROOT / "docs/qwen-attention-output-gate-cuda-contract.md"
        ).read_text().lower().split())

    def test_seen_integration_reuses_the_ledgered_primitive(self) -> None:
        self.assertIn("pub fun qwenCudaAttentionOutputGate", self.seen)
        self.assertIn("tokenCount > limit / queryHeads", self.seen)
        self.assertIn("rows > limit / headDim", self.seen)
        self.assertIn("seen_qwen_sigmoid_gate_f32(token, attended, gate, output", self.seen)
        self.assertEqual(self.source.count("<<<"), 23)
        self.assertEqual(self.source.count(", 0, stream>>>"), 23)

    def test_contract_preserves_the_projection_boundary(self) -> None:
        for required in (
            "[tokens, query_heads, head_dim]",
            "[tokens, query_heads * head_dim]",
            "attended * sigmoid(gate)",
            "separately owned `o_proj`", "partial overlap is rejected",
            "does not allocate", "silently fall back", "correctness oracle",
        ):
            self.assertIn(required, self.contract)

    def test_hardware_corpus_is_official_and_complete(self) -> None:
        for required in (
            "kQueryHeads = 24", "kHeadDim = 256", "cpu_reference",
            "std::isfinite", "input in-place", "gate in-place",
            "numeric_limits<uint64_t>::max()", "incompatible.generation = 0",
            "seen_cuda_graph_begin_capture", "seen_cuda_graph_launch",
            "seen_cuda_free",
        ):
            self.assertIn(required, self.test)

    def test_runner_is_bounded_and_sanitized(self) -> None:
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "run_qwn_042d.sh", "qwn_042e_cuda_test", "compute-sanitizer",
            "--leak-check full", "nvidia-smi",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner)
        self.assertNotIn("/usr/local/bin/seen", self.runner)


if __name__ == "__main__":
    unittest.main()
