#!/usr/bin/env python3
"""Static and filesystem-policy regression checks for QWN-032C."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/converter/conversion_evidence.seen"
OUTPUT = ROOT / ".seen/ci/output/qwn_032c"


class ConversionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        OUTPUT.mkdir(parents=True, exist_ok=True)

    def test_source_uses_streaming_hash_and_indexed_strict_json(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        for text in (
            "Sha256.new()",
            "updateBytes(bytes)",
            "QWEN_CONVERT_MAX_CHUNK_BYTES",
            "parseStrictJson",
            "keyAt(index)",
            "valueAt(index)",
            "destroyJsonParseResult(this.document, true)",
            "writeTextAtomically(path, text)",
            "canonical_included_catalog_prefix",
            "toolchain_compatibility_sha256",
            '"output_format", "SQW1"',
            "catalog.fingerprint != QWEN_CONVERT_CATALOG_SHA256",
        ):
            self.assertIn(text, source)
        self.assertNotIn("python", source.lower())
        self.assertNotIn("cuda", source.lower())

    def test_failed_atomic_write_left_no_temporary_file(self) -> None:
        leftovers = list(OUTPUT.rglob("*.seen-tmp-*"))
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
