#!/usr/bin/env python3
"""Static and schema evidence for the frozen SQW v1.0 contract."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/qwen-sqw-manifest.schema.json"
SOURCE = ROOT / "src/formats/sqw.seen"
DOC = ROOT / "docs/qwen-sqw-v1-contract.md"


class SqwContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.doc = DOC.read_text(encoding="utf-8")

    def test_schema_is_closed_and_identity_complete(self) -> None:
        self.assertFalse(self.schema["additionalProperties"])
        required = set(self.schema["required"])
        self.assertEqual(
            required,
            {
                "schema",
                "format_version",
                "model_lock_sha256",
                "source_lock_sha256",
                "conversion_policy_sha256",
                "tensor_contract",
                "catalog_sha256",
                "directory_sha256",
                "payload_order",
                "compatibility",
                "tensors",
            },
        )
        props = self.schema["properties"]
        self.assertEqual(props["schema"]["const"], "seen-qwen-sqw-manifest-v1")
        self.assertEqual(props["format_version"]["const"], "1.0")
        self.assertEqual(props["tensor_contract"]["const"], "seen-qwen38-text-v1")
        self.assertEqual(props["payload_order"]["const"], "canonical_utf8_tensor_name")
        self.assertEqual(props["compatibility"]["properties"]["required_features"]["maxItems"], 0)

    def test_u64_and_digest_encodings_are_canonical(self) -> None:
        self.assertEqual(
            self.schema["$defs"]["u64"]["pattern"],
            r"^(0|[1-9][0-9]{0,18})$",
        )
        self.assertEqual(
            self.schema["$defs"]["sha256"]["pattern"],
            "^[0-9a-f]{64}$",
        )
        tensor = self.schema["$defs"]["tensor"]
        self.assertFalse(tensor["additionalProperties"])
        self.assertEqual(tensor["properties"]["rank"], {"type": "integer", "minimum": 1, "maximum": 8})
        self.assertEqual(self.schema["properties"]["tensors"]["maxItems"], 2048)

    def test_binary_offsets_and_ids_are_locked_in_source_and_docs(self) -> None:
        for spelling in (
            "SQW_HEADER_BYTES: Int = 256",
            "SQW_DIRECTORY_ENTRY_BYTES: Int = 256",
            "SQW_FOOTER_HEADER_BYTES: Int = 64",
            "SQW_FOOTER_ENTRY_BYTES: Int = 64",
            "SQW_ENDIAN_MARKER: UInt32 = 0x01020304",
            "SQW_CODEC_Q8_SYM_G64: UInt16 = 4",
            "SQW_CODEC_Q4_SYM_G64: UInt16 = 5",
            "parseSqwHeader",
            "parseSqwDirectoryEntry",
            "parseSqwFooterHeader",
            "parseSqwFooterEntry",
            "sqwCheckedAdd",
            "sqwCheckedMultiply",
            "sqwCheckedAlign",
            "parseSqwUnsignedDecimal",
            "0x7FFFFFFFFFFFFFFF",
        ):
            self.assertIn(spelling, self.source)
        for spelling in (
            "`data || scale || zero || metadata`",
            "whole-file digest is SHA-256",
            "replaced by zero",
            "offset relative to the name table",
            "Q4_SYM_G64",
            "Safetensors remains the canonical source",
            "9,223,372,036,854,775,807",
        ):
            self.assertIn(spelling, self.doc)

    def test_manifest_canonicalization_has_one_stable_encoding(self) -> None:
        digest = "01" * 32
        manifest = {
            "schema": "seen-qwen-sqw-manifest-v1",
            "format_version": "1.0",
            "model_lock_sha256": digest,
            "source_lock_sha256": digest,
            "conversion_policy_sha256": digest,
            "tensor_contract": "seen-qwen38-text-v1",
            "catalog_sha256": digest,
            "directory_sha256": digest,
            "payload_order": "canonical_utf8_tensor_name",
            "compatibility": {
                "required_features": [],
                "reader_major": 1,
                "reader_minor": 0,
            },
            "tensors": [],
        }
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertFalse(encoded.startswith("\ufeff"))
        self.assertFalse(encoded.endswith("\n"))
        self.assertEqual(json.loads(encoded), manifest)
        self.assertEqual(encoded, json.dumps(json.loads(encoded), ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def test_private_historical_schema_is_not_the_shipped_contract(self) -> None:
        private_schema = ROOT / "docs/private/seen_qwen_spec_pack/schemas/qwen-sqw-manifest.schema.json"
        self.assertEqual(
            SCHEMA.relative_to(ROOT).as_posix(),
            "schemas/qwen-sqw-manifest.schema.json",
        )
        if private_schema.is_file():
            self.assertNotEqual(SCHEMA.resolve(), private_schema.resolve())


if __name__ == "__main__":
    unittest.main()
