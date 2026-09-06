#!/usr/bin/env python3
"""Sparse full-catalog fixture and source contract checks for QWN-032B."""

import json
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
OUTPUT = ROOT / ".seen/ci/output"
SOURCE_ROOT = OUTPUT / "qwn_032b_source"
MISMATCH_ROOT = OUTPUT / "qwn_032b_mismatch"
UNKNOWN_ROOT = OUTPUT / "qwn_032b_unknown"
ZERO_ROOT = OUTPUT / "qwn_032b_zero"
SOURCE = ROOT / "src/converter/shard_stream.seen"
TOTAL_TENSOR_BYTES = 55_562_855_904


def write_sparse_safetensors(path: Path, entries: list[tuple[str, int]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    for name, length in entries:
        header[name] = {
            "dtype": "U8",
            "shape": [length],
            "data_offsets": [offset, offset + length],
        }
        offset += length
    encoded = json.dumps(
        header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        if offset:
            handle.seek(8 + len(encoded) + offset - 1)
            handle.write(b"\0")


def prepare_sparse_source() -> None:
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = document["weight_map"]
    names = sorted(weight_map)
    quotient, remainder = divmod(TOTAL_TENSOR_BYTES, len(names))
    by_shard: dict[str, list[tuple[str, int]]] = {}
    for index, name in enumerate(names):
        length = quotient + (1 if index < remainder else 0)
        by_shard.setdefault(weight_map[name], []).append((name, length))
    for number in range(1, 19):
        shard = f"model-{number:05d}-of-00018.safetensors"
        write_sparse_safetensors(SOURCE_ROOT / shard, by_shard[shard])

    shard_two_name = next(
        name for name in names
        if weight_map[name] == "model-00002-of-00018.safetensors"
    )
    write_sparse_safetensors(
        MISMATCH_ROOT / "model-00001-of-00018.safetensors",
        [(shard_two_name, 1)],
    )
    write_sparse_safetensors(
        UNKNOWN_ROOT / "model-00001-of-00018.safetensors",
        [("unknown.weight", 1)],
    )
    shard_one_name = next(
        name for name in names
        if weight_map[name] == "model-00001-of-00018.safetensors"
    )
    write_sparse_safetensors(
        ZERO_ROOT / "model-00001-of-00018.safetensors",
        [(shard_one_name, 0)],
    )


class ShardStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prepare_sparse_source()

    def test_sparse_fixture_has_exact_full_catalog_geometry(self) -> None:
        document = json.loads(INDEX.read_text(encoding="utf-8"))
        self.assertEqual(document["metadata"]["total_size"], TOTAL_TENSOR_BYTES)
        self.assertEqual(len(document["weight_map"]), 1_199)
        shards = sorted(SOURCE_ROOT.glob("model-*.safetensors"))
        self.assertEqual(len(shards), 18)
        logical = sum(path.stat().st_size for path in shards)
        allocated = sum(path.stat().st_blocks * 512 for path in shards)
        self.assertGreater(logical, TOTAL_TENSOR_BYTES)
        self.assertLess(allocated, 16 * 1024 * 1024)

    def test_source_enforces_one_window_and_vision_exclusion(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        for text in (
            "QWEN_STREAM_MAX_WINDOWS",
            "qwenStreamWindowAdmission",
            "mappedWindowCount > admittedWindows.unwrap()",
            'if category.unwrap() == "vision"',
            "visionPayloadWindows: 0 as UInt64",
            "maximumOpenShards: 1",
            "maximumInFlightTensors: 1",
            "window.validate()",
            "ptr_deref_i8(window.data as Int)",
            "window.discardPages()",
            "window.close()",
            "qwen.stream.shard-mismatch",
            "qwen.stream.unknown-tensor",
            "qwen.stream.geometry",
        ):
            self.assertIn(text, source)
        vision_branch = source.split('if category.unwrap() == "vision"', 1)[1]
        vision_branch = vision_branch.split("} else {", 1)[0]
        self.assertNotIn("file.file.window", vision_branch)
        self.assertNotIn("seen_string_view_bytes", vision_branch)


if __name__ == "__main__":
    unittest.main()
