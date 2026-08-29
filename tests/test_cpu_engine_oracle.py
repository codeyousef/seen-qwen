#!/usr/bin/env python3

from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_024e_cpu_engine_oracle.json"
EXTRACTOR = ROOT / "tools/extract_cpu_engine_oracle.py"
ORACLE_SHA256 = "350bc70fb4c7dc010643cfd0c44f93fb60cb75cacfcddda08bc02eff26906857"
SOURCE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
MANIFEST_SHA256 = "da4ead2d07206e9f091ac180802da58c770d58c993d72a8cb70c15098fe51baa"


class CpuEngineOracleTests(unittest.TestCase):
    def test_locked_execution_oracle(self) -> None:
        self.assertEqual(sha256(ORACLE.read_bytes()).hexdigest(), ORACLE_SHA256)
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        self.assertEqual(oracle["schema"], "seen-qwen-cpu-engine-oracle-v1")
        self.assertEqual(oracle["source"]["expected_safetensors_sha256"], SOURCE_SHA256)
        self.assertEqual(oracle["source"]["manifest_sha256"], MANIFEST_SHA256)
        self.assertEqual(oracle["prompt_ids"], [83, 101, 101, 110, 32, 81, 119, 101, 110])
        self.assertEqual(oracle["greedy_token_ids"], [170, 129, 34, 239])
        self.assertEqual([len(row) for row in oracle["decode_logits"]], [256] * 4)

    def test_extractor_is_pinned_and_framework_free(self) -> None:
        source = EXTRACTOR.read_text(encoding="utf-8")
        self.assertIn(SOURCE_SHA256, source)
        self.assertIn(MANIFEST_SHA256, source)
        self.assertNotIn("import torch", source)


if __name__ == "__main__":
    unittest.main()
