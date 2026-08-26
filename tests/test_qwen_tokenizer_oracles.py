#!/usr/bin/env python3
"""Validate the committed Qwen tokenizer oracle without third-party imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


FIXTURE = Path(__file__).parent / "fixtures/qwen3_8_tokenizer_oracles.json"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"


class QwenTokenizerOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_source_and_payload_are_pinned(self) -> None:
        document = dict(self.document)
        recorded_hash = document.pop("payload_sha256")
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertEqual(document["schema"], "seen-qwen-tokenizer-oracle-v1")
        self.assertEqual(document["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(document["model_revision"], MODEL_REVISION)
        self.assertEqual(document["source"]["commit"], TRANSFORMERS_COMMIT)
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), recorded_hash)

    def test_vector_sets_are_bounded_and_unique(self) -> None:
        expected_names = {
            "text_vectors": {
                "ascii",
                "arabic",
                "unicode_combining",
                "seen_code",
                "whitespace",
                "special_literal",
                "bounded_repetition_1024",
            },
            "chat_vectors": {
                "user_xhigh_generation_prompt",
                "system_user_low",
                "thinking_disabled",
                "preserve_thinking_false",
                "tool_definition_call_response",
            },
            "rejected_chat_vectors": {
                "no_messages",
                "invalid_reasoning_effort",
                "late_system_message",
            },
        }
        for key, expected in expected_names.items():
            actual = [vector["name"] for vector in self.document[key]]
            self.assertEqual(set(actual), expected)
            self.assertEqual(len(actual), len(set(actual)))

        for vector in self.document["text_vectors"] + self.document["chat_vectors"]:
            self.assertEqual(vector["token_count"], len(vector["token_ids"]))
            self.assertLessEqual(vector["token_count"], 4096)

    def test_contract_and_chat_round_trips(self) -> None:
        contract = self.document["tokenizer_contract"]
        self.assertEqual(contract["vocab_size"], 248044)
        self.assertEqual(contract["model_max_length"], 262144)
        self.assertEqual(contract["eos_token_id"], 248046)
        self.assertEqual(contract["pad_token_id"], 248044)
        special_tokens = contract["special_tokens"]
        self.assertEqual(len(special_tokens), 33)
        self.assertEqual(special_tokens[0], {"id": 248044, "token": "<|endoftext|>"})
        self.assertEqual(special_tokens[-1], {"id": 248076, "token": "<|audio_pad|>"})

        for vector in self.document["chat_vectors"]:
            self.assertEqual(vector["decoded"], vector["rendered"])

    def test_rejected_template_cases_are_exact(self) -> None:
        rejected = {vector["name"]: vector for vector in self.document["rejected_chat_vectors"]}
        self.assertEqual(rejected["no_messages"]["error_type"], "ValueError")
        self.assertEqual(
            rejected["no_messages"]["error"],
            "Cannot apply chat template to an empty conversation. Provide at least one message.",
        )
        self.assertEqual(rejected["invalid_reasoning_effort"]["error_type"], "TemplateError")
        self.assertIn("Supported types are xhigh (default), medium, and low", rejected["invalid_reasoning_effort"]["error"])
        self.assertEqual(rejected["late_system_message"]["error_type"], "TemplateError")
        self.assertEqual(rejected["late_system_message"]["error"], "System message must be at the beginning.")


if __name__ == "__main__":
    unittest.main()
