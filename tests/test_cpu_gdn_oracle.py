#!/usr/bin/env python3

from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_024c_cpu_gdn_oracle.json"
EXTRACTOR = ROOT / "tools/extract_cpu_gdn_oracle.py"
MODEL_SHA256 = "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9"
SOURCE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
ORACLE_SHA256 = "a323c959cfa1c419b640ed52ef5025f043eddb57de6d4f157085e021f8c97fb1"


class CpuGdnOracleTests(unittest.TestCase):
    def test_locked_identity_and_geometry(self) -> None:
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        self.assertEqual(oracle["schema"], "seen-qwen-cpu-gdn-oracle-v1")
        self.assertEqual(oracle["source"]["model_safetensors_sha256"], MODEL_SHA256)
        self.assertEqual(oracle["source"]["expected_safetensors_sha256"], SOURCE_SHA256)
        self.assertEqual(oracle["source"]["layer"], 0)
        self.assertEqual(oracle["geometry"], {
            "channels": 80, "kernel": 4, "key_dim": 8,
            "sequence": 9, "value_dim": 8, "value_heads": 6,
        })
        lengths = {
            "mixed": 720, "conv_weight": 320, "convolution": 720,
            "query": 432, "key": 432, "value": 432,
            "beta": 54, "log_decay": 54, "gate": 432,
            "norm_weight": 8, "gated_norm": 432, "recurrent_final": 384,
        }
        for name, expected in lengths.items():
            self.assertEqual(len(oracle[name]), expected, name)
        self.assertEqual(sha256(ORACLE.read_bytes()).hexdigest(), ORACLE_SHA256)

    def test_extractor_is_pinned_and_framework_free(self) -> None:
        source = EXTRACTOR.read_text(encoding="utf-8")
        self.assertIn(MODEL_SHA256, source)
        self.assertIn(SOURCE_SHA256, source)
        self.assertIn('oracle_prefix = "base.layer_0"', source)
        self.assertNotIn("import torch", source)


if __name__ == "__main__":
    unittest.main()
