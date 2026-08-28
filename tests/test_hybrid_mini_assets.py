#!/usr/bin/env python3
"""Independent reproducibility and format oracle for QWN-023B assets."""

from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests/fixtures/qwen3_8_hybrid_mini"
GENERATOR = ROOT / "tools/generate_hybrid_mini_fixture.py"
EXPECTED = {
    "manifest.json": "98fa563ccd9036804832749844d1327ad6d20c3e12b0ace219ef53b5ae35e884",
    "merges.txt": "215a6aba00d27bcd42b8ad1dccc4b4d23f40decc150bdbf0d5ce6bb2410708df",
    "model.safetensors": "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9",
    "tokenizer_config.json": "17b53cbb51444a64f73a18201605cbae2321153fd7eefba3c66be25c6555baa6",
    "vocab.json": "6f5112365ce8a9d83ed809cfb7d7749f2283f05696848fe7d687f1049623bb11",
}


def decode_safetensors(raw: bytes) -> tuple[dict[str, object], bytes]:
    if len(raw) < 8:
        raise ValueError("missing header length")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if not 0 < header_length <= 1_048_576 or 8 + header_length > len(raw):
        raise ValueError("invalid header length")
    header = json.loads(raw[8 : 8 + header_length])
    payload = raw[8 + header_length :]
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not name or not isinstance(entry, dict) or entry.get("dtype") != "F32":
            raise ValueError("invalid tensor entry")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(shape, list) or not shape or any(
            not isinstance(value, int) or value <= 0 for value in shape
        ):
            raise ValueError("invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid data offsets")
        start, end = offsets
        elements = 1
        for dimension in shape:
            elements *= dimension
        if not 0 <= start <= end <= len(payload) or end - start != elements * 4:
            raise ValueError("invalid tensor range")
        ranges.append((start, end, name))
    ranges.sort()
    if ranges[0][0] != 0 or ranges[-1][1] != len(payload):
        raise ValueError("payload is not fully covered")
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] != current[0]:
            raise ValueError("tensor ranges overlap or contain gaps")
    return header, payload


class HybridMiniAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = (ASSETS / "model.safetensors").read_bytes()
        cls.header, cls.payload = decode_safetensors(cls.raw)
        cls.manifest = json.loads((ASSETS / "manifest.json").read_text())
        output = ROOT / ".seen/ci/output"
        output.mkdir(parents=True, exist_ok=True)
        minimal_header = json.dumps({
            "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        }, separators=(",", ":")).encode()
        minimal_header += b" " * ((-len(minimal_header)) % 8)
        (output / "qwn_023b_minimal.safetensors").write_bytes(
            struct.pack("<Q", len(minimal_header)) + minimal_header + struct.pack("<f", 1.0)
        )
        (output / "qwn_023b_truncated.safetensors").write_bytes(cls.raw[:7])
        unsupported = json.loads(cls.raw[8 : 8 + struct.unpack("<Q", cls.raw[:8])[0]])
        first = min(name for name in unsupported if name != "__metadata__")
        unsupported[first]["dtype"] = "BAD"
        encoded = json.dumps(unsupported, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        (output / "qwn_023b_unsupported_dtype.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + cls.payload
        )

    def test_asset_hashes_and_manifest_are_exact(self) -> None:
        self.assertEqual(set(EXPECTED), {path.name for path in ASSETS.iterdir()})
        for name, expected in EXPECTED.items():
            self.assertEqual(sha256((ASSETS / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(self.manifest["safetensors"]["tensor_count"], 124)
        self.assertEqual(self.manifest["safetensors"]["payload_bytes"], len(self.payload))
        self.assertEqual(self.manifest["tokenizer"]["vocabulary_size"], 256)
        self.assertEqual(
            self.manifest["sources"]["model_revision"],
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        )
        self.assertEqual(
            self.manifest["sources"]["transformers_commit"],
            "562cfd944ee1f20702cfb0f4404014ee27c24813",
        )

    def test_safetensors_names_shapes_and_ranges_are_canonical(self) -> None:
        tensors = {name: value for name, value in self.header.items() if name != "__metadata__"}
        self.assertEqual(len(tensors), 124)
        self.assertEqual(list(tensors), sorted(tensors))
        self.assertEqual(tensors["model.language_model.embed_tokens.weight"]["shape"], [256, 48])
        self.assertEqual(tensors["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]["shape"], [80, 48])
        self.assertEqual(tensors["model.language_model.layers.3.self_attn.q_proj.weight"]["shape"], [288, 48])
        self.assertEqual(tensors["model.language_model.layers.7.self_attn.o_proj.weight"]["shape"], [48, 144])
        self.assertEqual(tensors["mtp.fc.weight"]["shape"], [48, 96])
        self.assertEqual(tensors["mtp.layers.0.self_attn.q_proj.weight"]["shape"], [288, 48])
        self.assertFalse(any(name.startswith("model.visual.") for name in tensors))

    def test_tokenizer_is_a_complete_reversible_byte_fixture(self) -> None:
        vocab = json.loads((ASSETS / "vocab.json").read_text())
        self.assertEqual(len(vocab), 256)
        self.assertEqual(set(vocab.values()), set(range(256)))
        self.assertEqual((ASSETS / "merges.txt").read_text(), "#version: 0.2\n")
        config = json.loads((ASSETS / "tokenizer_config.json").read_text())
        self.assertEqual(config["model_max_length"], 128)
        self.assertEqual(config["special_tokens"], [])

    def test_generator_reproduces_every_byte(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".seen/ci") as directory:
            output = Path(directory) / "assets"
            subprocess.run(
                [sys.executable, str(GENERATOR), "--output-dir", str(output)],
                cwd=ROOT,
                check=True,
                timeout=30,
            )
            for name in EXPECTED:
                self.assertEqual((output / name).read_bytes(), (ASSETS / name).read_bytes())

    def test_independent_reader_rejects_malformed_ranges(self) -> None:
        with self.assertRaises(ValueError):
            decode_safetensors(self.raw[:7])
        header = json.loads(self.raw[8 : 8 + struct.unpack("<Q", self.raw[:8])[0]])
        names = sorted(name for name in header if name != "__metadata__")
        header[names[1]]["data_offsets"][0] = header[names[0]]["data_offsets"][0]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        with self.assertRaises(ValueError):
            decode_safetensors(struct.pack("<Q", len(encoded)) + encoded + self.payload)


if __name__ == "__main__":
    unittest.main()
