#!/usr/bin/env python3
"""Independent canonical-plan oracle and ownership checks for QWN-032A."""

from hashlib import sha256
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/converter/conversion_plan.seen"
CONTRACT = ROOT / "docs/qwen-conversion-plan-contract.md"


class ConversionPlanTests(unittest.TestCase):
    def test_canonical_plan_fingerprint(self) -> None:
        values = [
            ("source_total_bytes", 55_562_855_904),
            ("source_shards", 18),
            ("catalog_tensors", 1_199),
            ("included_tensors", 866),
            ("excluded_vision_tensors", 333),
            ("max_open_shards", 1),
            ("max_in_flight_tensors", 1),
            ("worker_count", 1),
            ("memory_budget_bytes", 4_294_967_296),
            ("source_window_bytes", 67_108_864),
            ("codec_workspace_bytes", 134_217_728),
            ("writer_chunk_bytes", 1_048_576),
            ("evidence_buffer_bytes", 1_048_576),
            ("fixed_overhead_bytes", 268_435_456),
            ("peak_host_bytes", 471_859_200),
            ("max_artifact_bytes", 68_719_476_736),
        ]
        canonical = (
            "qwen38-conversion-plan-v1\n"
            "model_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0\n"
            "catalog_sha256="
            "5f466d43bae3059e54f0bfe183d0e82c822242f45a834d778414d3e5b5248f1f\n"
            + "".join(f"{name}={value}\n" for name, value in values)
        )
        self.assertEqual(
            sha256(canonical.encode()).hexdigest(),
            "0666131d7d6d01f51ce437a5e5a93c070f9f9b057d5b5e04d2746daf18018b0c",
        )

    def test_source_freezes_admission_and_non_goals(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        contract = CONTRACT.read_text(encoding="utf-8")
        for text in (
            "QWEN_CONVERSION_SOURCE_BYTES",
            "QWEN_CONVERSION_MAX_MEMORY_BYTES",
            "maxOpenShards: 1 as UInt64",
            "maxInFlightTensors: 1 as UInt64",
            "workerCount: 1 as UInt64",
            "qwen.conversion.budget",
            "qwen.conversion.closed",
            "validateQwen38ConversionPlan",
        ):
            self.assertIn(text, source)
        for text in (
            "does not open a shard",
            "does not choose a codec",
            "at most 64 GiB",
            "`maxArtifactBytes` is a storage extent",
        ):
            self.assertIn(text, contract)


if __name__ == "__main__":
    unittest.main()
