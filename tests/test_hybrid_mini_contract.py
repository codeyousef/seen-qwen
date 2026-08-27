#!/usr/bin/env python3
"""Independent structural oracle for the QWN-023A mini-model contract."""

from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_contract.json"
CONTRACT_SHA256 = "f29839615771e344bf89329f2195e5921fc8ea371849249be55722ab1999dddf"


class HybridMiniContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = CONTRACT.read_bytes()
        cls.contract = json.loads(cls.raw)

    def test_file_identity_is_frozen(self) -> None:
        self.assertEqual(sha256(self.raw).hexdigest(), CONTRACT_SHA256)
        self.assertEqual(self.contract["schema"], "seen-qwen-hybrid-mini-v1")
        self.assertEqual(self.contract["fixture_id"], "qwen3_8_hybrid_mini_v1")
        self.assertFalse(self.contract["qwen_compatible"])
        self.assertEqual(self.contract["deterministic_seed"], 20260827)
        self.assertEqual(self.contract["weights_dtype"], "float32")
        self.assertEqual(len(self.contract), 39)

    def test_hybrid_schedule_is_exact(self) -> None:
        expected = [
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(8)
        ]
        self.assertEqual(self.contract["layer_types"], expected)
        self.assertEqual(self.contract["num_hidden_layers"], 8)
        self.assertEqual(self.contract["linear_attention_layers"], 6)
        self.assertEqual(self.contract["full_attention_layers"], 2)

    def test_geometry_preserves_required_relationships(self) -> None:
        c = self.contract
        self.assertEqual(c["num_attention_heads"] // c["num_key_value_heads"], 6)
        self.assertEqual(c["linear_num_value_heads"] // c["linear_num_key_heads"], 3)
        self.assertEqual(c["rotary_dim"], c["head_dim"] * c["partial_rotary_factor"])
        self.assertEqual(sum(c["mrope_section"]), c["rotary_dim"] // 2)
        self.assertEqual(c["mtp_num_hidden_layers"], 1)
        self.assertFalse(c["mtp_use_dedicated_embeddings"])
        self.assertFalse(c["tie_word_embeddings"])

    def test_namespace_policy_excludes_vision(self) -> None:
        self.assertEqual(
            self.contract["required_namespaces"],
            ["model.language_model.*", "lm_head.weight", "mtp.*"],
        )
        self.assertEqual(self.contract["excluded_namespaces"], ["model.visual.*"])


if __name__ == "__main__":
    unittest.main()
