#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from atomic_output_file import write_atomic_file


class AtomicOutputFileTests(unittest.TestCase):
    def test_fresh_and_replacement_writes_leave_no_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "oracle.json"
            write_atomic_file(output, b"first\n", maximum_bytes=64)
            write_atomic_file(output, b"second\n", maximum_bytes=64)
            self.assertEqual(output.read_bytes(), b"second\n")
            self.assertEqual(sorted(root.iterdir()), [output])

    def test_symlink_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_bytes(b"original\n")
            output = root / "oracle.json"
            output.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "unsafe output"):
                write_atomic_file(output, b"replacement\n", maximum_bytes=64)

            self.assertEqual(target.read_bytes(), b"original\n")
            self.assertTrue(output.is_symlink())

    def test_directory_and_symlinked_parent_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "oracle.json"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                write_atomic_file(output, b"replacement\n", maximum_bytes=64)

            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "parent directory"):
                write_atomic_file(
                    linked_parent / "new" / "oracle.json",
                    b"replacement\n",
                    maximum_bytes=64,
                )
            self.assertFalse((real_parent / "new").exists())

    def test_byte_bound_is_checked_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle.json"
            with self.assertRaisesRegex(ValueError, "byte bound"):
                write_atomic_file(output, b"too large", maximum_bytes=3)
            self.assertFalse(output.exists())

    def test_replace_failure_preserves_prior_file_and_cleans_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "oracle.json"
            output.write_bytes(b"original\n")

            def fail_replace(_source: str | Path, _destination: str | Path) -> None:
                raise OSError("injected replace failure")

            with self.assertRaisesRegex(OSError, "injected replace failure"):
                write_atomic_file(
                    output,
                    b"replacement\n",
                    maximum_bytes=64,
                    replace=fail_replace,
                )

            self.assertEqual(output.read_bytes(), b"original\n")
            self.assertEqual(sorted(root.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
