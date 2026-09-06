#!/usr/bin/env python3
"""Static and staging-cleanup contract for the QWN-030C SQW writer."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/formats/sqw_writer.seen"
SEEN_TEST = ROOT / "tests/qwn_030c_sqw_writer_test.seen"
MANIFEST = ROOT / "Seen.toml"
INNER = ROOT / "scripts/ci/required_inner.sh"
OUTPUT = ROOT / ".seen/ci/output/qwn_030c"


class SqwWriterContractTests(unittest.TestCase):
    def test_writer_is_manifested_and_required(self) -> None:
        manifest = MANIFEST.read_text(encoding="utf-8")
        inner = INNER.read_text(encoding="utf-8")
        self.assertIn('"src/formats/sqw_writer.seen"', manifest)
        self.assertGreaterEqual(inner.count("qwn_030c_sqw_writer_test"), 3)

    def test_writer_uses_bounded_staging_validation_and_atomic_rename(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        for spelling in (
            "SQW_WRITER_MAX_CHUNK_BYTES",
            "@move\npub class SqwWriter",
            "createTemporaryBeside",
            "writeAllAt",
            "SqwReader.open",
            "renamePath",
            "syncPromotedDirectory",
            "removeSingle",
            "sqwCheckedAdd",
            "isPromoted",
        ):
            self.assertIn(spelling, source)
        self.assertLess(
            source.index("SqwReader.open"),
            source.index("renamePath(this.staging"),
        )
        self.assertNotIn("CUDA", source)
        self.assertNotRegex(source, r"(?<![A-Za-z])(/tmp|/usr/local)")

    def test_focused_test_covers_commit_boundaries(self) -> None:
        source = SEEN_TEST.read_text(encoding="utf-8")
        for spelling in (
            "testPromotionAndDeterminism",
            "requireSameArtifact",
            "testFailedValidationPreservesDestination",
            "testBoundsCancellationAndNoReplace",
            "sqw.incomplete",
            "sqw.cancelled",
            "sqw.promote",
            "sqw.magic",
        ):
            self.assertIn(spelling, source)

    def test_no_rejected_staging_artifact_remains(self) -> None:
        if not OUTPUT.exists():
            return
        leftovers = [
            path.name
            for path in OUTPUT.iterdir()
            if re.search(r"\.seen-tmp-", path.name)
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
