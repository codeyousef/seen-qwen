#!/usr/bin/env python3

from hashlib import sha256
import json
import math
from pathlib import Path
import unittest

from tools.fetch_official_layer_ranges import select_layer_entries


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_025a_operator_layer_oracles.json"
CAPTURE = ROOT / "tools/capture_official_operator_layer_oracles.py"
FETCH = ROOT / "tools/fetch_official_layer_ranges.py"
ORACLE_SHA256 = "a85770501619699d8af36f09f928272a0bcbcc036c58554d26993b52b13b50d6"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"
PACKS = {
    0: ("linear_attention", "22ad61f0f4ffebe43c41ae85311635b952165bdeb7248314b4a3e21ad8954eb2"),
    3: ("full_attention", "c43c2b3867748358cfc9233b30d395a5ed6d437d70c26493e41798d9d8fc7288"),
    31: ("full_attention", "c4bcffaebd09d727cf017848a12669f1e7e04d5685e469a00b79e7570d9ea41a"),
    32: ("linear_attention", "fffae639e3e35f6be62b119df7d6f032a305e05971b5f14e316092f5ce92985b"),
    60: ("linear_attention", "663d7f2fac180323e705052346d11ab3b2c6691d74c04a9f80dfdb80e857312c"),
    63: ("full_attention", "20bd3a48018bcc86d7dbb12f38fadb6c714a1183c9286baf9ab07616cd48dcfa"),
}
LFS = {
    "model-00001-of-00018.safetensors": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
    "model-00009-of-00018.safetensors": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
    "model-00010-of-00018.safetensors": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
    "model-00016-of-00018.safetensors": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
    "model-00017-of-00018.safetensors": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
}


class OfficialOperatorLayerOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = ORACLE.read_bytes()
        cls.document = json.loads(cls.raw)

    def test_content_and_source_identities_are_exact(self) -> None:
        self.assertEqual(sha256(self.raw).hexdigest(), ORACLE_SHA256)
        self.assertEqual(self.document["schema"], "seen-qwen-official-operator-layer-oracles-v1")
        self.assertEqual(self.document["classification"], "verified-cpu-reference")
        source = self.document["source"]
        self.assertEqual(source["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(source["model_revision"], MODEL_REVISION)
        self.assertEqual(source["transformers_commit"], TRANSFORMERS_COMMIT)
        self.assertEqual(source["modeling_source_sha256"], MODELING_SHA256)
        self.assertEqual(source["torch"], "2.11.0+cpu")
        self.assertEqual(source["transformers"], "5.16.0.dev0")
        self.assertEqual(source["input_manifest_sha256"], "efa24d8afb46a4ddda2509705504fa634dcdc90e14ade5f2b938afd55912e57a")
        self.assertEqual(source["capture_tool_sha256"], sha256(CAPTURE.read_bytes()).hexdigest())
        self.assertEqual(source["fetch_tool_sha256"], sha256(FETCH.read_bytes()).hexdigest())

    def test_first_middle_last_layer_classification_and_provenance(self) -> None:
        self.assertEqual(
            self.document["selection"],
            {
                "first_linear_attention": 0,
                "first_full_attention": 3,
                "middle_full_attention": 31,
                "middle_linear_attention": 32,
                "last_linear_attention": 60,
                "last_full_attention": 63,
            },
        )
        layers = {entry["layer"]: entry for entry in self.document["layers"]}
        self.assertEqual(set(layers), set(PACKS))
        for layer, (kind, pack_hash) in PACKS.items():
            entry = layers[layer]
            self.assertEqual(entry["kind"], kind)
            self.assertEqual(entry["source"]["kind"], kind)
            self.assertEqual(entry["source"]["sha256"], pack_hash)
            self.assertEqual(
                entry["source"]["source_shard_lfs_sha256"],
                LFS[entry["source"]["source_shard"]],
            )
            first, last = entry["source"]["source_absolute_range"]
            self.assertGreaterEqual(first, 8)
            self.assertGreaterEqual(last, first)
            self.assertLess(last, entry["source"]["source_shard_bytes"])
            self.assertEqual(entry["positions"], [0, 31])

    def test_operator_vectors_are_bounded_finite_and_content_addressed(self) -> None:
        common = {
            "layer_input",
            "input_norm",
            "token_mixer",
            "post_attention_norm",
            "mlp_output",
            "layer_output",
        }
        for layer in self.document["layers"]:
            operators = layer["operators"]
            self.assertTrue(common.issubset(operators))
            if layer["kind"] == "linear_attention":
                self.assertTrue(
                    {"qkv_projection", "recurrent_state", "gated_recurrent_output"}.issubset(operators)
                )
                self.assertEqual(operators["recurrent_state"]["shape"], [1, 48, 128, 128])
                self.assertEqual(operators["qkv_projection"]["shape"], [1, 2, 10240])
            else:
                self.assertIn("query_gate_projection", operators)
                self.assertEqual(operators["query_gate_projection"]["shape"], [1, 2, 12288])
            self.assertEqual(operators["layer_output"]["shape"], [1, 2, 5120])
            for record in operators.values():
                self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
                indices = record["sample_indices"]
                values = record["sample_values_f32"]
                self.assertEqual(indices, sorted(set(indices)))
                self.assertEqual(len(indices), len(values))
                self.assertLessEqual(len(values), 48)
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertTrue(math.isfinite(record["minimum_f32"]))
                self.assertTrue(math.isfinite(record["maximum_f32"]))
                self.assertLessEqual(record["minimum_f32"], record["maximum_f32"])

    def test_capture_path_is_pinned_bounded_and_does_not_execute_remote_code(self) -> None:
        source = CAPTURE.read_text(encoding="utf-8") + FETCH.read_text(encoding="utf-8")
        for identity in (MODEL_REVISION, TRANSFORMERS_COMMIT, MODELING_SHA256, *LFS.values()):
            self.assertIn(identity, source)
        self.assertIn('headers={"Range":', source)
        self.assertIn("torch.set_num_threads(1)", source)
        self.assertIn("torch.use_deterministic_algorithms(True)", source)
        self.assertNotIn("trust_remote_code", source)
        self.assertNotIn("AutoModel", source)
        self.assertNotIn("from_pretrained", source)

    def test_fetcher_rejects_missing_misindexed_and_noncontiguous_tensors(self) -> None:
        prefix = "model.language_model.layers.0."
        header = {
            prefix + "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            prefix + "b": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
        }
        index = {
            prefix + "a": "model-00001-of-00018.safetensors",
            prefix + "b": "model-00001-of-00018.safetensors",
        }
        self.assertEqual(len(select_layer_entries(header, 0, 1, index)), 2)
        with self.assertRaisesRegex(ValueError, "index disagrees"):
            select_layer_entries({prefix + "a": header[prefix + "a"]}, 0, 1, index)
        wrong_index = dict(index)
        wrong_index[prefix + "b"] = "model-00002-of-00018.safetensors"
        with self.assertRaisesRegex(ValueError, "index disagrees"):
            select_layer_entries(header, 0, 1, wrong_index)
        noncontiguous = dict(header)
        noncontiguous[prefix + "b"] = {
            "dtype": "BF16", "shape": [1], "data_offsets": [4, 6]
        }
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            select_layer_entries(noncontiguous, 0, 1, index)
        with self.assertRaisesRegex(ValueError, "absent"):
            select_layer_entries({}, 0, 1, index)


if __name__ == "__main__":
    unittest.main()
