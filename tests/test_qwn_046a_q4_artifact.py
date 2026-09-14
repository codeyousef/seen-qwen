#!/usr/bin/env python3
"""Independent bounded contracts for the QWN-046A Q4 artifact builder."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/oracle/build_qwn_046a_q4_artifact.py"
POLICY = ROOT / "configs/q4-bringup-quantization.toml"


def load_builder():
    spec = importlib.util.spec_from_file_location("qwn_046a_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load QWN-046A builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Q4ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def test_policy_is_explicitly_experimental_without_fallback(self) -> None:
        digest = self.builder.validate_policy(POLICY)
        self.assertEqual(len(digest), 64)

    def test_normative_ties_even_and_zero_vectors(self) -> None:
        ties = np.zeros((1, 64), dtype=np.float32)
        ties[0, :8] = (7.0, -7.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5)
        data, scales = self.builder._encode_rows(ties)
        self.assertEqual(data[:4], bytes((0x97, 0x20, 0x02, 0xEE)))
        self.assertEqual(scales, bytes((0x00, 0x3C)))
        zero_data, zero_scales = self.builder._encode_rows(
            np.zeros((1, 3), dtype=np.float32))
        self.assertEqual(zero_data, b"\0\0")
        self.assertEqual(zero_scales, b"\0\0")

    def test_odd_row_tails_are_independent(self) -> None:
        values = np.zeros((2, 65), dtype=np.float32)
        values[0, 0] = 7.0
        values[0, 64] = 2.0
        values[1, 0] = 4.0
        values[1, 64] = -3.0
        first_data, first_scales = self.builder._encode_rows(values)
        second_data, second_scales = self.builder._encode_rows(values)
        self.assertEqual((len(first_data), len(first_scales)), (66, 8))
        self.assertEqual(first_data[32], 7)
        self.assertEqual(first_data[65], 9)
        self.assertEqual((first_data, first_scales), (second_data, second_scales))

    def test_builder_is_bounded_and_fully_reads_back_before_promotion(self) -> None:
        source = BUILDER.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import transformers",
                          "trust_remote_code", "/usr/local/bin/seen"):
            self.assertNotIn(forbidden, source.lower())
        for required in ("MAX_BATCH_BYTES", "MAX_ARTIFACT_BYTES",
                         "tempfile.mkdtemp", "validate_sqw(weights",
                         "quality_approved\": False", "fallback\": \"prohibited",
                         "allocation_bytes_at_max_context\": \"16627529808"):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
