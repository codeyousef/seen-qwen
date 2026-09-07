#!/usr/bin/env python3
"""Independent deterministic full-catalog SQW fixture for QWN-032D."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".seen/ci/output/qwn_032d"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
READER_ORACLE = ROOT / "tests/test_sqw_reader.py"

MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_LOCK = "a" * 64
SOURCE_LOCK = "b" * 64
CONVERSION_POLICY = "c" * 64
CATALOG = "5f466d43bae3059e54f0bfe183d0e82c822242f45a834d778414d3e5b5248f1f"
COMPATIBILITY = "f39aebda1fdafc20d04b6c6a0072a491afea3f8f9ab836fd173ead0d1fe1ea33"
EXPECTED_METADATA = {
    "file_bytes": 1066304,
    "journal_bytes": 252725,
    "journal_sha256": "ced7b10e39919e81659eb36c92561ad93194ed679ef156e759416eb72f6d3b2a",
    "tensor_count": 866,
    "whole_sha256": "4de515a72a029f7e03847485d40fc6a0a8975aa07d81f984a4cbb5f14da4c9a1",
}
EXPECTED_FILE_SHA256 = "2de4916051bf1a48a952c55ea4af43c3e3707b7d86b9a3306f53e753441a687b"

PLAN_VALUES = {
    "source_total_bytes": 55562855904,
    "source_shards": 18,
    "catalog_tensors": 1199,
    "included_tensors": 866,
    "excluded_vision_tensors": 333,
    "max_open_shards": 1,
    "max_in_flight_tensors": 1,
    "worker_count": 1,
    "memory_budget_bytes": 4294967296,
    "source_window_bytes": 67108864,
    "codec_workspace_bytes": 134217728,
    "writer_chunk_bytes": 1048576,
    "evidence_buffer_bytes": 1048576,
    "fixed_overhead_bytes": 268435456,
    "peak_host_bytes": 471859200,
    "max_artifact_bytes": 68719476736,
}


def plan_fingerprint() -> str:
    lines = [
        "qwen38-conversion-plan-v1",
        f"model_revision={MODEL_REVISION}",
        f"catalog_sha256={CATALOG}",
    ]
    lines.extend(f"{key}={value}" for key, value in PLAN_VALUES.items())
    return sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def load_reader_oracle():
    spec = importlib.util.spec_from_file_location("qwn_030b_oracle", READER_ORACLE)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load the local SQW oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def included_names() -> tuple[str, ...]:
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    names = tuple(
        sorted(
            (
                name
                for name in document["weight_map"]
                if name == "lm_head.weight"
                or name.startswith("model.language_model.")
                or name.startswith("mtp.")
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(names) != 866:
        raise AssertionError(f"expected 866 included tensors, found {len(names)}")
    return names


def journal_and_tensors(oracle):
    entries = []
    tensors = []
    for ordinal, name in enumerate(included_names()):
        source = name.encode("utf-8")
        converted = ordinal.to_bytes(2, "little")
        entries.append(
            {
                "ordinal": ordinal,
                "name": name,
                "source_bytes": str(len(source)),
                "converted_bytes": str(len(converted)),
                "source_sha256": sha256(source).hexdigest(),
                "converted_sha256": sha256(converted).hexdigest(),
            }
        )
        tensors.append(
            oracle.TensorSpec(
                name=name,
                role_id=12,
                role_name="norm",
                dtype_id=7,
                dtype_name="BF16",
                codec_id=1,
                codec_name="BF16",
                shape=(1,),
                row_elements=1,
                group_elements=0,
                alignment=64,
                source_bytes=source,
                data=converted,
            )
        )
    journal = {
        "schema": "seen-qwen-conversion-journal-v1",
        "version": "1",
        "output_format": "SQW1",
        "model_revision": MODEL_REVISION,
        "model_lock_sha256": MODEL_LOCK,
        "source_lock_sha256": SOURCE_LOCK,
        "conversion_policy_sha256": CONVERSION_POLICY,
        "catalog_sha256": CATALOG,
        "plan_sha256": plan_fingerprint(),
        "toolchain_compatibility_sha256": COMPATIBILITY,
        "tensor_order": "canonical_included_catalog_prefix",
        "completed_count": len(entries),
        "entries": entries,
    }
    encoded = json.dumps(journal, ensure_ascii=False, separators=(",", ":")).encode()
    return encoded, tuple(tensors)


def build_fixture() -> dict[str, object]:
    oracle = load_reader_oracle()
    oracle.MODEL_LOCK = bytes.fromhex(MODEL_LOCK)
    oracle.SOURCE_LOCK = bytes.fromhex(SOURCE_LOCK)
    oracle.CONVERSION_POLICY = bytes.fromhex(CONVERSION_POLICY)
    oracle.CATALOG_DIGEST = bytes.fromhex(CATALOG)
    journal, tensors = journal_and_tensors(oracle)
    built = oracle.build_sqw(tensors=tensors, evidence=journal)
    corrupt = bytearray(built.data)
    payload_offset, _ = built.sections["payload"]
    corrupt[payload_offset] ^= 1
    reordered = json.loads(journal)
    reordered["entries"][0], reordered["entries"][1] = (
        reordered["entries"][1],
        reordered["entries"][0],
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "valid.sqw").write_bytes(built.data)
    (OUTPUT / "corrupt_payload.sqw").write_bytes(corrupt)
    (OUTPUT / "truncated.sqw").write_bytes(built.data[:-1])
    (OUTPUT / "reordered_journal.json").write_text(
        json.dumps(reordered, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    metadata = {
        "file_bytes": len(built.data),
        "whole_sha256": built.data[
            built.whole_digest_offset : built.whole_digest_offset + 32
        ].hex(),
        "journal_sha256": sha256(journal).hexdigest(),
        "journal_bytes": len(journal),
        "tensor_count": len(tensors),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return metadata


class ConversionFinalizerFixtureTests(unittest.TestCase):
    def test_full_catalog_fixture_is_deterministic_and_bounded(self) -> None:
        first = build_fixture()
        first_bytes = (OUTPUT / "valid.sqw").read_bytes()
        second = build_fixture()
        self.assertEqual(first, second)
        self.assertEqual(first, EXPECTED_METADATA)
        self.assertEqual(first_bytes, (OUTPUT / "valid.sqw").read_bytes())
        self.assertEqual(sha256(first_bytes).hexdigest(), EXPECTED_FILE_SHA256)
        self.assertEqual(
            (OUTPUT / "truncated.sqw").read_bytes(), first_bytes[:-1]
        )
        reordered = json.loads(
            (OUTPUT / "reordered_journal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(reordered["entries"][0]["ordinal"], 1)
        self.assertEqual(reordered["entries"][1]["ordinal"], 0)
        self.assertEqual(first["tensor_count"], 866)
        self.assertLess(first["file_bytes"], 2 * 1024 * 1024)
        self.assertLess(first["journal_bytes"], 1024 * 1024)

    def test_production_finalizer_has_no_python_or_gpu_dependency(self) -> None:
        source = (ROOT / "src/converter/conversion_finalizer.seen").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("python", source.lower())
        self.assertNotIn("cuda", source.lower())
        for spelling in (
            "writer.seal()",
            "SqwReader.open",
            "reader.evidenceText",
            "qwenFinalizeJournalMatches",
            "writer.commit()",
        ):
            self.assertIn(spelling, source)


if __name__ == "__main__":
    unittest.main()
