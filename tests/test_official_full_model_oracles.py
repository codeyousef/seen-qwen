#!/usr/bin/env python3

from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import tempfile
import tomllib
import unittest

from tools.fetch_official_full_model import (
    INDEX_SHA256,
    MODEL_REVISION,
    SHARDS,
    validate_index,
    validate_output_root,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_025b_full_model_oracles.json"
LOGITS = ROOT / "tests/fixtures/qwn_025b_full_model_logits.safetensors"
TOLERANCES = ROOT / "tests/tolerances.toml"
CAPTURE = ROOT / "tools/capture_official_full_model_oracles.py"
FETCH = ROOT / "tools/fetch_official_full_model.py"
ORACLE_SHA256 = "42ba8e0b9d5adec7c5ce1708bfc60d7b9911c126af9e6703b35efd108fedc921"
LOGITS_SHA256 = "87a63ff88f93dded6846533221cdc489bacf844d8260c9ce4f79ca070f586b4e"
PROMPTS = {
    "minimal", "english", "arabic", "code", "long_repeated", "tool_chat",
    "thinking_on", "thinking_off",
}
GREEDY_PROMPTS = {"thinking_on", "thinking_off"}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise ValueError("short Safetensors length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length == 0 or header_length > 1024 * 1024:
            raise ValueError("invalid Safetensors header length")
        raw_header = source.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError("short Safetensors header")
    return json.loads(raw_header.decode("utf-8").rstrip(" ")), 8 + header_length


class OfficialFullModelOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = ORACLE.read_bytes()
        cls.document = json.loads(cls.raw)

    def test_content_and_source_identities_are_exact(self) -> None:
        self.assertEqual(digest(ORACLE), ORACLE_SHA256)
        self.assertEqual(digest(LOGITS), LOGITS_SHA256)
        self.assertEqual(self.document["schema"], "seen-qwen-full-model-oracles-v1")
        self.assertEqual(self.document["classification"], "verified-cpu-reference")
        source = self.document["source"]
        self.assertEqual(source["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(source["model_revision"], MODEL_REVISION)
        self.assertEqual(source["tensor_index_sha256"], INDEX_SHA256)
        self.assertEqual(source["input_manifest_sha256"], "5222ae63faa0e62c33abf0166f32b033cd8175b8f6c83870bd95c20fa6c1720f")
        self.assertEqual(source["transformers_commit"], "562cfd944ee1f20702cfb0f4404014ee27c24813")
        self.assertEqual(source["modeling_source_sha256"], "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8")
        self.assertEqual(source["torch"], "2.11.0+cpu")
        self.assertEqual(source["transformers"], "5.16.0.dev0")
        self.assertEqual(source["capture_tool_sha256"], digest(CAPTURE))
        self.assertEqual(source["fetch_tool_sha256"], digest(FETCH))

    def test_prompt_logits_and_greedy_sequences_are_complete_and_bounded(self) -> None:
        prompts = self.document["prompts"]
        self.assertEqual({entry["id"] for entry in prompts}, PROMPTS)
        self.assertEqual(len(prompts), len(PROMPTS))
        for prompt in prompts:
            self.assertGreater(prompt["input_token_count"], 0)
            self.assertLessEqual(prompt["input_token_count"], 256)
            self.assertEqual(prompt["input_token_count"], len(prompt["input_token_ids"]))
            self.assertTrue(all(0 <= token < 248_320 for token in prompt["input_token_ids"]))
            self.assertRegex(prompt["rendered_utf8_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(prompt["initial_top5_token_ids"]), 5)
            self.assertEqual(len(set(prompt["initial_top5_token_ids"])), 5)
            self.assertTrue(all(0 <= token < 248_320 for token in prompt["initial_top5_token_ids"]))
            self.assertTrue(all(math.isfinite(value) for value in prompt["initial_top5_logits_f32"]))
            if prompt["id"] in GREEDY_PROMPTS:
                self.assertEqual(prompt["greedy_sequence_role"], "128-step")
                self.assertEqual(len(prompt["greedy_token_ids"]), 128)
                self.assertEqual(len(prompt["greedy_selected_logits_f32"]), 128)
                self.assertTrue(all(0 <= token < 248_320 for token in prompt["greedy_token_ids"]))
                self.assertTrue(all(math.isfinite(value) for value in prompt["greedy_selected_logits_f32"]))
                self.assertRegex(prompt["greedy_decoded_utf8_sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(prompt["first_eos_step"] is None or 0 <= prompt["first_eos_step"] < 128)
            else:
                self.assertEqual(prompt["greedy_sequence_role"], "logits-only")
                self.assertNotIn("greedy_token_ids", prompt)
                self.assertNotIn("greedy_selected_logits_f32", prompt)

    def test_complete_bf16_logit_vectors_are_safe_and_exactly_shaped(self) -> None:
        header, payload_start = safetensors_header(LOGITS)
        metadata = header.pop("__metadata__")
        self.assertEqual(metadata["schema"], "seen-qwen-full-model-logits-v1")
        self.assertEqual(metadata["model_revision"], MODEL_REVISION)
        self.assertEqual(set(header), {f"{name}.next_logits_bf16" for name in PROMPTS})
        ranges = []
        for record in header.values():
            self.assertEqual(record["dtype"], "BF16")
            self.assertEqual(record["shape"], [248_320])
            first, last = record["data_offsets"]
            self.assertEqual(last - first, 248_320 * 2)
            ranges.append((first, last))
        self.assertEqual(sorted(ranges), [(index * 496_640, (index + 1) * 496_640) for index in range(8)])
        self.assertEqual(LOGITS.stat().st_size, payload_start + 8 * 496_640)
        logits = self.document["logits"]
        self.assertEqual(logits["sha256"], LOGITS_SHA256)
        self.assertEqual(logits["shape_per_prompt"], [248_320])
        self.assertEqual(logits["tensor_count"], 8)

    def test_tolerance_contract_is_versioned_and_exact(self) -> None:
        policy = tomllib.loads(TOLERANCES.read_text(encoding="utf-8"))
        self.assertEqual(policy["schema"], "seen-qwen-tolerances-v1")
        self.assertEqual(policy["fp32_cpu"], {"atol_max": 1e-5, "rtol_max": 1e-5, "allow_nan_inf": False})
        bf16 = policy["bf16_reference_full_model"]
        self.assertEqual(bf16["mean_logit_cosine_min"], 0.9995)
        self.assertEqual(bf16["top5_set_overlap_min"], 0.99)
        self.assertEqual(bf16["greedy_compare_steps"], 128)
        self.assertTrue(bf16["capture_first_divergence"])
        self.assertFalse(bf16["allow_nan_inf"])
        self.assertEqual(self.document["tolerance_contract"]["path"], "tests/tolerances.toml")
        self.assertEqual(
            self.document["determinism"]["greedy_prompt_ids"],
            ["thinking_on", "thinking_off"],
        )

    def test_fetch_and_capture_paths_are_pinned_bounded_and_fail_closed(self) -> None:
        self.assertTrue(validate_index())
        fetch_source = FETCH.read_text(encoding="utf-8")
        capture_source = CAPTURE.read_text(encoding="utf-8")
        for _shard, (size, identity) in SHARDS.items():
            self.assertIn(f"{size:_}", fetch_source)
            self.assertIn(identity, fetch_source)
        for identity in (MODEL_REVISION, INDEX_SHA256, "torch.use_deterministic_algorithms(True)"):
            self.assertIn(identity, capture_source + fetch_source)
        self.assertIn("MAX_PROMPT_TOKENS = 256", capture_source)
        self.assertIn("GREEDY_STEPS = 128", capture_source)
        self.assertIn("local_files_only=True, trust_remote_code=False", capture_source)
        self.assertNotIn("AutoModel", capture_source)
        self.assertNotIn("urllib", capture_source)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "ignored project .seen root"):
                validate_output_root(Path(temporary))


if __name__ == "__main__":
    unittest.main()
