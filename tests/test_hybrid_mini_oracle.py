#!/usr/bin/env python3
"""Independent structural and numerical checks for the QWN-023C oracle."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle"
EXPECTED = ORACLE / "expected.safetensors"
MANIFEST = ORACLE / "manifest.json"
GENERATOR = ROOT / "tools/generate_hybrid_mini_oracle.py"
EXPECTED_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
GENERATOR_SHA256 = "fea88c9dd92e1e5ffa12699be84b52293557e91234917191083f4bcc41c858ab"
CONTRACT_SHA256 = "f29839615771e344bf89329f2195e5921fc8ea371849249be55722ab1999dddf"
MODEL_SHA256 = "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def decode(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("truncated Safetensors file")
    header_length = struct.unpack_from("<Q", raw)[0]
    if header_length == 0 or header_length > 1 << 20:
        raise ValueError("invalid Safetensors header length")
    header_end = 8 + header_length
    if header_end > len(raw):
        raise ValueError("Safetensors header exceeds file")
    return json.loads(raw[8:header_end].decode("utf-8").rstrip(" ")), raw[header_end:]


class HybridMiniOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.header, cls.payload = decode(EXPECTED)
        cls.entries = {name: value for name, value in cls.header.items() if name != "__metadata__"}

    def values(self, name: str) -> tuple[float | int, ...]:
        entry = self.entries[name]
        start, end = entry["data_offsets"]
        formats = {"F32": "f", "I64": "q"}
        unit = {"F32": 4, "I64": 8}[entry["dtype"]]
        self.assertEqual((end - start) // unit, math.prod(entry["shape"]))
        return struct.unpack(f"<{(end - start) // unit}{formats[entry['dtype']]}", self.payload[start:end])

    def test_manifest_locks_sources_dependencies_and_environment(self) -> None:
        self.assertEqual(self.manifest["schema"], "seen-qwen-hybrid-mini-oracle-v1")
        self.assertEqual(self.manifest["maturity"], "verified")
        self.assertEqual(self.manifest["dependencies"]["contract_sha256"], CONTRACT_SHA256)
        self.assertEqual(self.manifest["dependencies"]["model_safetensors_sha256"], MODEL_SHA256)
        self.assertEqual(
            self.manifest["sources"]["transformers_commit"],
            "562cfd944ee1f20702cfb0f4404014ee27c24813",
        )
        self.assertEqual(
            self.manifest["sources"]["qwen_mtp_commit"],
            "22283d7a1b4eff75fb0d63fb2e862ade39df7b73",
        )
        self.assertEqual(
            self.manifest["sources"]["qwen_mtp_source_sha256"],
            "cf97664e82371425df14e412c9d351405d05ae8db3622fc817813ddda6858622",
        )
        self.assertEqual(self.manifest["environment"]["torch"], "2.11.0+cpu")
        self.assertEqual(self.manifest["environment"]["threads"], 1)
        self.assertTrue(self.manifest["environment"]["deterministic_algorithms"])
        self.assertEqual(self.manifest["comparison"], {"atol": 1e-5, "dtype": "F32", "rtol": 1e-5})

    def test_assets_and_generator_are_content_addressed(self) -> None:
        self.assertEqual(digest(GENERATOR), GENERATOR_SHA256)
        self.assertEqual(self.manifest["generator_sha256"], GENERATOR_SHA256)
        self.assertEqual(digest(EXPECTED), EXPECTED_SHA256)
        self.assertEqual(self.manifest["assets"]["expected.safetensors"]["sha256"], EXPECTED_SHA256)
        self.assertEqual(self.manifest["assets"]["expected.safetensors"]["bytes"], EXPECTED.stat().st_size)

    def test_tensor_table_is_bounded_contiguous_and_finite(self) -> None:
        self.assertEqual(len(self.entries), 147)
        self.assertEqual(self.manifest["coverage"]["tensor_count"], 147)
        cursor = 0
        for name in sorted(self.entries):
            entry = self.entries[name]
            self.assertIn(entry["dtype"], ("F32", "I64"), name)
            self.assertTrue(entry["shape"], name)
            self.assertTrue(all(isinstance(dimension, int) and dimension > 0 for dimension in entry["shape"]), name)
            start, end = entry["data_offsets"]
            self.assertEqual(start, cursor, name)
            self.assertGreater(end, start, name)
            self.assertLessEqual(end, len(self.payload), name)
            cursor = end
            if entry["dtype"] == "F32":
                self.assertTrue(all(math.isfinite(value) for value in self.values(name)), name)
        self.assertEqual(cursor, len(self.payload))
        self.assertEqual(self.manifest["coverage"]["non_finite_values"], 0)

    def test_first_middle_last_and_mtp_checkpoints_are_complete(self) -> None:
        expected_shapes = {
            "base.embeddings": [1, 9, 48],
            "base.layer_0.recurrent_final": [1, 6, 8, 8],
            "base.layer_0.output": [1, 9, 48],
            "base.layer_3.probabilities": [1, 6, 9, 9],
            "base.layer_4.recurrent_final": [1, 6, 8, 8],
            "base.layer_7.probabilities": [1, 6, 9, 9],
            "base.layer_7.output": [1, 9, 48],
            "base.final_hidden": [1, 9, 48],
            "base.logits": [1, 9, 256],
            "mtp.stem": [1, 1, 48],
            "mtp.layer_0.probabilities": [1, 6, 1, 1],
            "mtp.final_hidden": [1, 1, 48],
            "mtp.logits": [1, 1, 256],
        }
        for name, shape in expected_shapes.items():
            self.assertIn(name, self.entries)
            self.assertEqual(self.entries[name]["shape"], shape, name)
        for layer in range(8):
            for point in ("input", "mixer_input", "mixer_output", "post_mixer", "mlp_input", "mlp_output", "output"):
                self.assertIn(f"base.layer_{layer}.{point}", self.entries)

    def test_tokens_match_each_recorded_logit_argmax(self) -> None:
        self.assertEqual(self.values("base.input_ids"), (83, 101, 101, 110, 32, 81, 119, 101, 110))
        generated = tuple(self.manifest["outputs"]["greedy_token_ids"])
        self.assertEqual(generated, self.values("base.generated_ids"))
        self.assertEqual(generated, (170, 129, 34, 239))
        for step, token in enumerate(generated):
            logits = self.values(f"base.decode_step_{step}.logits")
            self.assertEqual(max(range(len(logits)), key=logits.__getitem__), token)
        self.assertEqual(self.values("base.greedy_token"), (generated[0],))
        mtp_logits = self.values("mtp.logits")
        mtp_token = max(range(len(mtp_logits)), key=mtp_logits.__getitem__)
        self.assertEqual(mtp_token, 166)
        self.assertEqual(self.values("mtp.greedy_token"), (mtp_token,))
        self.assertEqual(self.manifest["outputs"]["mtp_input_token_id"], generated[0])
        self.assertEqual(self.values("mtp.position"), (9,))

    def test_attention_probabilities_are_causal_and_normalized(self) -> None:
        for layer in (3, 7):
            values = self.values(f"base.layer_{layer}.probabilities")
            for head in range(6):
                for query in range(9):
                    base = (head * 9 + query) * 9
                    row = values[base : base + 9]
                    self.assertAlmostEqual(sum(row), 1.0, places=6)
                    self.assertTrue(all(value == 0.0 for value in row[query + 1 :]))
        self.assertEqual(self.values("mtp.layer_0.probabilities"), (1.0,) * 6)


if __name__ == "__main__":
    unittest.main()
