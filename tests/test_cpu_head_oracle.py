#!/usr/bin/env python3

from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_024d_cpu_head_oracle.json"
EXTRACTOR = ROOT / "tools/extract_cpu_head_oracle.py"
ORACLE_SHA256 = "98dcd9ff0ca710d9f21579cfb4222698777b12ae2338541a3eb943642eddd56b"
MODEL_SHA256 = "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9"
SOURCE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"


class CpuHeadOracleTests(unittest.TestCase):
    def test_locked_identity_geometry_and_tokens(self) -> None:
        self.assertEqual(sha256(ORACLE.read_bytes()).hexdigest(), ORACLE_SHA256)
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        self.assertEqual(oracle["schema"], "seen-qwen-cpu-head-oracle-v1")
        self.assertEqual(oracle["source"]["model_safetensors_sha256"], MODEL_SHA256)
        self.assertEqual(oracle["source"]["expected_safetensors_sha256"], SOURCE_SHA256)
        self.assertEqual(oracle["geometry"], {
            "hidden": 48, "reduced_intermediate": 8,
            "reduced_output": 6, "vocabulary": 256,
        })
        self.assertEqual(oracle["lm_head_token_rows"], [0, 1, 166, 170, 255])
        self.assertEqual(oracle["base_greedy_token"], 170)
        self.assertEqual(oracle["mtp_greedy_token"], 166)
        self.assertEqual(len(oracle["mtp_fusion_weight"]), 48 * 96)

    def test_extractor_is_pinned_and_framework_free(self) -> None:
        source = EXTRACTOR.read_text(encoding="utf-8")
        self.assertIn(MODEL_SHA256, source)
        self.assertIn(SOURCE_SHA256, source)
        self.assertIn('tensor(oh, op, "mtp.stem")', source)
        self.assertNotIn("import torch", source)


if __name__ == "__main__":
    unittest.main()
