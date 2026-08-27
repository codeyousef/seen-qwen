#!/usr/bin/env python3
"""Independent policy check for the committed Qwen sampling profile lock."""

from pathlib import Path
import hashlib
import json
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/sampling.toml"
GENERATION = ROOT / ".seen/oracle-assets-qwen38/generation_config.json"


class SamplingProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile_bytes = PROFILE.read_bytes()
        cls.document = tomllib.loads(cls.profile_bytes.decode("utf-8"))

    def test_source_identity_is_exact(self) -> None:
        self.assertEqual(self.document["schema_version"], 1)
        self.assertEqual(self.document["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(
            self.document["model_revision"],
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        )
        self.assertEqual(
            self.document["model_card_sha256"],
            "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
        )
        self.assertEqual(
            hashlib.sha256(GENERATION.read_bytes()).hexdigest(),
            self.document["generation_config_sha256"],
        )

    def test_official_profiles_are_exact(self) -> None:
        thinking = self.document["presets"]["thinking"]
        instruct = self.document["presets"]["instruct"]
        self.assertEqual(
            {key: thinking[key] for key in (
                "temperature", "top_p", "top_k", "min_p",
                "presence_penalty", "repetition_penalty")},
            {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
             "min_p": 0.0, "presence_penalty": 0.0,
             "repetition_penalty": 1.0},
        )
        self.assertEqual(
            {key: instruct[key] for key in (
                "temperature", "top_p", "top_k", "min_p",
                "presence_penalty", "repetition_penalty")},
            {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
             "min_p": 0.0, "presence_penalty": 1.5,
             "repetition_penalty": 1.0},
        )
        generation = json.loads(GENERATION.read_text(encoding="utf-8"))
        self.assertEqual(generation["temperature"], thinking["temperature"])
        self.assertEqual(generation["top_p"], thinking["top_p"])
        self.assertEqual(generation["top_k"], thinking["top_k"])

    def test_exact_named_profile_set(self) -> None:
        self.assertEqual(
            set(self.document["presets"]),
            {"thinking", "instruct", "greedy", "custom"},
        )


if __name__ == "__main__":
    unittest.main()
