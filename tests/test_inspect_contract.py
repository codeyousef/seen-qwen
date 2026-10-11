"""Independent contract checks for FEL-1458 / QWN-047B."""

import json
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
SPARSE_ROOT = ROOT / ".seen/verification/output/qwn_047b_source"
CATEGORY_BYTES = {
    "text": 53_791_996_928,
    "mtp": 849_398_784,
    "vision": 921_460_192,
}


def tensor_category(name: str) -> str:
    if name == "lm_head.weight" or name.startswith("model.language_model."):
        return "text"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual."):
        return "vision"
    raise ValueError(f"tensor is outside the frozen catalog: {name}")


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


def prepare_sparse_inspect_source() -> None:
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = document["weight_map"]
    categorized: dict[str, list[str]] = {
        category: [] for category in CATEGORY_BYTES
    }
    for name in sorted(weight_map):
        categorized[tensor_category(name)].append(name)

    lengths: dict[str, int] = {}
    for category, names in categorized.items():
        quotient, remainder = divmod(CATEGORY_BYTES[category], len(names))
        for index, name in enumerate(names):
            lengths[name] = quotient + (1 if index < remainder else 0)

    by_shard: dict[str, list[tuple[str, int]]] = {}
    for name in sorted(weight_map):
        by_shard.setdefault(weight_map[name], []).append((name, lengths[name]))
    for number in range(1, 19):
        shard = f"model-{number:05d}-of-00018.safetensors"
        write_sparse_safetensors(SPARSE_ROOT / shard, by_shard[shard])


class QwenInspectContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "src/cli/inspect.seen").read_text(encoding="utf-8")
        cls.main = (ROOT / "src/cli/main.seen").read_text(encoding="utf-8")
        cls.contract = (ROOT / "docs/qwen-cli-contract.md").read_text(
            encoding="utf-8"
        )
        cls.manifest = (ROOT / "Seen.toml").read_text(encoding="utf-8")
        cls.header_fixture = (
            ROOT / "scripts/oracle/prepare_qwn_047b_sparse_headers.py"
        ).read_text(encoding="utf-8")

    def test_inspect_is_installed_as_an_atomic_leaf(self) -> None:
        self.assertIn('"src/cli/inspect.seen"', self.manifest)
        self.assertIn("invocation.kind == QWEN_CLI_INSPECT", self.main)
        self.assertIn("inspectQwenModel(invocation.modelLock", self.main)
        self.assertIn("seen-qwen-inspect-v1", self.source)

    def test_source_identity_is_exact_and_bounded(self) -> None:
        for value in (
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
            "399b6f25f8cd20ae5d233040005f0ebb82b7e5b396e0eb91214dbe94b828c3be",
            "f43b83cd9174a91a2ccb2669dc1751c83b9c6730593abe6004bc185855cc78bb",
            "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
            "5f466d43bae3059e54f0bfe183d0e82c822242f45a834d778414d3e5b5248f1f",
            "QWEN_INSPECT_TOTAL_BYTES: UInt64 = 55562855904",
            "while fill < 1199",
            "while shard < 18",
            "textCount != 851",
            "mtpCount != 15",
            "visionCount != 333",
        ):
            self.assertIn(value, self.source)

    def test_metadata_only_and_cpu_default_are_structural(self) -> None:
        self.assertIn("file.metadata.tensors[metadata].dataLength", self.source)
        self.assertNotIn(".window(", self.source)
        self.assertNotIn("mapTensor", self.source)
        guard = 'if deviceOrdinal >= (0 as Int32) {'
        self.assertIn(guard, self.source)
        self.assertGreater(
            self.source.index("CudaDevice.open(deviceOrdinal)"),
            self.source.index(guard),
        )
        self.assertIn('"device_probe"', self.source)
        self.assertIn('"not-requested"', self.source)

    def test_output_and_cleanup_are_explicit(self) -> None:
        for value in (
            "writeTextAtomically(outputPath, rendered)",
            "file.close()",
            "catalog.close(); catalog.shardIndex.free(); catalog.free()",
            "report.close(); report.free()",
            "releaseSha256DigestResult",
        ):
            self.assertIn(value, self.source + self.main)
        self.assertIn("only file mutation", self.contract)

    def test_sparse_fixture_preserves_exact_inspect_aggregates(self) -> None:
        prepare_sparse_inspect_source()
        document = json.loads(INDEX.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = document["weight_map"]
        counts = {category: 0 for category in CATEGORY_BYTES}
        totals = {category: 0 for category in CATEGORY_BYTES}
        for name in sorted(weight_map):
            category = tensor_category(name)
            counts[category] += 1

        for path in sorted(SPARSE_ROOT.glob("model-*.safetensors")):
            with path.open("rb") as handle:
                header_length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_length))
            for name, metadata in header.items():
                category = tensor_category(name)
                start, end = metadata["data_offsets"]
                totals[category] += end - start

        self.assertEqual(counts, {"text": 851, "mtp": 15, "vision": 333})
        self.assertEqual(totals, CATEGORY_BYTES)
        shards = sorted(SPARSE_ROOT.glob("model-*.safetensors"))
        self.assertEqual(len(shards), 18)
        self.assertLess(sum(path.stat().st_blocks * 512 for path in shards),
                        16 * 1024 * 1024)

    def test_official_header_fixture_is_pinned_and_range_bounded(self) -> None:
        for value in (
            '"--range", f"0-{end}"',
            'header_value(raw_headers, "x-repo-commit")',
            'header_value(raw_headers, "x-linked-size")',
            'header_value(raw_headers, "x-linked-etag")',
            'header_value(raw_headers, "content-range")',
            "official headers do not contain the complete frozen tensor index",
            "official header category bytes differ",
        ):
            self.assertIn(value, self.header_fixture)


if __name__ == "__main__":
    unittest.main()
