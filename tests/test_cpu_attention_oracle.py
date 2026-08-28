#!/usr/bin/env python3

from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_024b_cpu_attention_oracle.json"
EXTRACTOR = ROOT / "tools/extract_cpu_attention_oracle.py"
ORACLE_SHA256 = "f7d1daf7e1d1bd5cf919ef7f1d8b866350d3632b999dfe731ad23cbdc28f9797"
SOURCE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"


class CpuAttentionOracleTests(unittest.TestCase):
    def test_locked_identity_and_geometry(self) -> None:
        self.assertEqual(sha256(ORACLE.read_bytes()).hexdigest(), ORACLE_SHA256)
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        self.assertEqual(oracle["schema"], "seen-qwen-cpu-attention-oracle-v1")
        self.assertEqual(oracle["source"]["expected_safetensors_sha256"], SOURCE_SHA256)
        self.assertEqual(oracle["source"]["layer"], 3)
        self.assertEqual(oracle["geometry"], {
            "head_dim": 24,
            "kv_heads": 1,
            "query_heads": 2,
            "query_start": 0,
            "sequence": 3,
        })
        self.assertEqual(len(oracle["query"]), 144)
        self.assertEqual(len(oracle["key"]), 72)
        self.assertEqual(len(oracle["value"]), 72)
        self.assertEqual(len(oracle["gate"]), 144)
        self.assertEqual(len(oracle["probabilities"]), 18)
        self.assertEqual(len(oracle["attended"]), 144)

    def test_causal_probabilities_are_normalized(self) -> None:
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        probabilities = oracle["probabilities"]
        for position in range(3):
            for head in range(2):
                row = probabilities[(position * 2 + head) * 3:(position * 2 + head + 1) * 3]
                self.assertAlmostEqual(sum(row), 1.0, places=6)
                self.assertTrue(all(value == 0.0 for value in row[position + 1:]))

    def test_extractor_pins_the_source_oracle(self) -> None:
        source = EXTRACTOR.read_text(encoding="utf-8")
        self.assertIn(SOURCE_SHA256, source)
        self.assertIn('prefix = "base.layer_3"', source)
        self.assertIn('f"{prefix}.probabilities"', source)
        self.assertNotIn("import torch", source)


if __name__ == "__main__":
    unittest.main()
