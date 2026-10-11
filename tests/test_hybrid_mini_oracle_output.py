#!/usr/bin/env python3
"""Atomic-promotion regression tests that do not import PyTorch."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from hybrid_mini_oracle_output import promote_oracle_output


class HybridMiniOracleOutputTest(unittest.TestCase):
    def test_rejects_filesystem_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe output"):
            promote_oracle_output(Path("/"), {"new": b"replacement"})

    def test_promotes_fresh_output_without_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "oracle"

            promote_oracle_output(output, {"expected.bin": b"oracle", "manifest.json": b"{}\n"})

            self.assertEqual((output / "expected.bin").read_bytes(), b"oracle")
            self.assertEqual((output / "manifest.json").read_bytes(), b"{}\n")
            self.assertEqual(sorted(parent.iterdir()), [output])

    def test_replaces_existing_directory_without_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "oracle"
            output.mkdir()
            (output / "old").write_bytes(b"preserve only until commit")

            promote_oracle_output(output, {"new": b"replacement"})

            self.assertFalse((output / "old").exists())
            self.assertEqual((output / "new").read_bytes(), b"replacement")
            self.assertEqual(sorted(parent.iterdir()), [output])

    def test_rejects_existing_file_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "oracle"
            output.write_bytes(b"original")

            with self.assertRaisesRegex(ValueError, "must be a directory"):
                promote_oracle_output(output, {"new": b"replacement"})

            self.assertEqual(output.read_bytes(), b"original")
            self.assertEqual(sorted(parent.iterdir()), [output])

    def test_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "target"
            target.mkdir()
            (target / "old").write_bytes(b"original")
            output = parent / "oracle"
            output.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "unsafe output"):
                promote_oracle_output(output, {"new": b"replacement"})

            self.assertEqual((target / "old").read_bytes(), b"original")
            self.assertTrue(output.is_symlink())

    def test_restores_existing_directory_when_final_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "oracle"
            output.mkdir()
            (output / "old").write_bytes(b"original")
            calls = 0

            def fail_final_replace(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected final promotion failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(OSError, "injected final promotion failure"):
                promote_oracle_output(
                    output,
                    {"new": b"replacement"},
                    replace=fail_final_replace,
                )

            self.assertEqual(calls, 3)
            self.assertEqual((output / "old").read_bytes(), b"original")
            self.assertFalse((output / "new").exists())
            self.assertEqual(sorted(parent.iterdir()), [output])

    def test_rejects_nested_asset_name_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle"

            with self.assertRaisesRegex(ValueError, "unsafe oracle asset"):
                promote_oracle_output(output, {"../escape": b"no"})

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
