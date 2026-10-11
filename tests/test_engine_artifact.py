#!/usr/bin/env python3
"""Independent bounded contracts for QWN-034A artifact construction."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/oracle/build_qwn_034a_bf16_artifact.py"
POLICY = ROOT / "configs/quantization.toml"
SCHEMA = ROOT / "schemas/qwen-engine-artifact.schema.json"
FIXTURE = ROOT / "tests/fixtures/qwn_034a_engine_artifact.json"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("qwn_034a_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load QWN-034A builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EngineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def test_policy_schema_and_fixture_are_closed(self) -> None:
        policy_sha = self.builder.validate_policy(POLICY)
        self.assertEqual(
            policy_sha,
            "ae8ab061f558f57ada0a026ca03a85be95d719e014d07126d33f040563638028",
        )
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        fixture_bytes = FIXTURE.read_bytes()
        fixture = self.builder.strict_json_bytes(fixture_bytes, maximum=1024 * 1024)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["tokenizer"]["minItems"], 7)
        self.assertEqual(schema["properties"]["tokenizer"]["maxItems"], 7)
        self.assertEqual(len(fixture["tokenizer"]), 7)
        self.assertEqual(fixture["quantization"]["policy_sha256"], policy_sha)
        expected = self.builder.engine_id(
            fixture["model_lock_sha256"], fixture["source_lock_sha256"],
            fixture["format"]["weights_sha256"], policy_sha,
            fixture["created_by"]["compiler_sha256"],
        )
        self.assertEqual(fixture["engine_id"], expected)
        self.assertEqual(fixture_bytes, self.builder.canonical_json(fixture) + b"\n")

    def test_strict_json_and_semantic_catalog_fail_closed(self) -> None:
        with self.assertRaises(self.builder.DuplicateKey):
            self.builder.strict_json_bytes(b'{"a":1,"a":2}', maximum=100)
        with self.assertRaises(ValueError):
            self.builder.strict_json_bytes(b'{"x":NaN}', maximum=100)
        index, _ = self.builder.read_index(INDEX)
        names = sorted((name for name in index if self.builder.is_included(name)), key=lambda value: value.encode("utf-8"))
        self.assertEqual(len(names), 866)
        roles = [self.builder.semantic_role(name) for name in names]
        self.assertEqual(names[0], "lm_head.weight")
        self.assertEqual(names[-1], "mtp.pre_fc_norm_hidden.weight")
        self.assertIn((14, "mtp"), roles)
        with self.assertRaises(ValueError):
            self.builder.semantic_role("model.language_model.unowned.weight")

    def test_small_sqw_is_sealed_and_fully_read_back(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwn-034a-unit-") as directory:
            root = Path(directory)
            tensors = []
            specifications = (
                ("lm_head.weight", b"\x00\x01\x02\x03", (1, 2)),
                ("model.language_model.embed_tokens.weight", b"\x04\x05\x06\x07", (1, 2)),
                ("mtp.pre_fc_norm_hidden.weight", b"\x08\x09\x0a\x0b", (2,)),
            )
            for ordinal, (name, data, shape) in enumerate(specifications):
                shard = root / f"source-{ordinal}.bin"
                shard.write_bytes(data)
                role_id, role_name = self.builder.semantic_role(name)
                tensor = self.builder.SourceTensor(name, shard, 0, len(data), shape, role_id, role_name)
                tensor.source_sha256 = sha256(data).hexdigest()
                tensors.append(tensor)
            tensors.sort(key=lambda tensor: tensor.name.encode("utf-8"))
            model_lock = "1" * 64
            source_lock = "2" * 64
            policy = "3" * 64
            evidence = self.builder.conversion_evidence(tensors, model_lock, source_lock, policy, 12)
            layout = self.builder.compute_layout(tensors, model_lock, source_lock, policy, evidence)
            output = root / "weights.sqw"
            whole, conventional = self.builder.write_sqw(output, tensors, layout, model_lock, policy, 257)
            self.builder.validate_sqw(output, tensors, layout, model_lock, policy, whole, conventional, 257)
            with output.open("r+b") as sink:
                sink.seek(layout.sections["payload"][0])
                original = sink.read(1)
                sink.seek(layout.sections["payload"][0])
                sink.write(bytes([original[0] ^ 1]))
            with self.assertRaises(ValueError):
                self.builder.validate_sqw(output, tensors, layout, model_lock, policy, whole, conventional, 257)

    def make_existing_artifact(self, root: Path) -> tuple[Path, bytes]:
        destination = root / "artifact"
        destination.mkdir()
        payload = b"bounded artifact payload"
        (destination / "payload.bin").write_bytes(payload)
        checksums = f"{sha256(payload).hexdigest()}  payload.bin\n".encode("ascii")
        (destination / "checksums.sha256").write_bytes(checksums)
        return destination, checksums

    def validate_existing(self, destination: Path, checksums: bytes) -> None:
        self.builder.validate_existing_artifact(destination, checksums, 7, 1024)

    def test_existing_artifact_requires_every_exact_regular_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, checksums = self.make_existing_artifact(root)
            self.validate_existing(destination, checksums)

            (destination / "payload.bin").write_bytes(b"corrupt")
            with self.assertRaisesRegex(FileExistsError, "hash differs"):
                self.validate_existing(destination, checksums)

            (destination / "payload.bin").unlink()
            with self.assertRaisesRegex(FileExistsError, "members differ"):
                self.validate_existing(destination, checksums)

    def test_existing_artifact_rejects_extra_and_non_regular_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, checksums = self.make_existing_artifact(root)
            (destination / "extra").write_bytes(b"unexpected")
            with self.assertRaisesRegex(FileExistsError, "members differ"):
                self.validate_existing(destination, checksums)
            (destination / "extra").unlink()
            (destination / "nested").mkdir()
            with self.assertRaisesRegex(FileExistsError, "non-regular"):
                self.validate_existing(destination, checksums)

    def test_existing_artifact_rejects_destination_and_member_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, checksums = self.make_existing_artifact(root)
            link = root / "linked-artifact"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(FileExistsError, "safe directory"):
                self.validate_existing(link, checksums)
            regular = root / "regular-artifact"
            regular.write_bytes(b"not a directory")
            with self.assertRaisesRegex(FileExistsError, "safe directory"):
                self.validate_existing(regular, checksums)

            (target / "payload.bin").unlink()
            (target / "payload.bin").symlink_to(root / "outside")
            with self.assertRaisesRegex(FileExistsError, "non-regular"):
                self.validate_existing(target, checksums)

    def test_existing_artifact_rejects_malformed_or_unsafe_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination, _ = self.make_existing_artifact(Path(directory))
            with self.assertRaisesRegex(ValueError, "malformed"):
                self.validate_existing(destination, b"not-a-checksum\n")
            line = f"{'0' * 64}  payload.bin\n".encode("ascii")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.validate_existing(destination, line + line)
            unsafe = f"{'0' * 64}  ../escape\n".encode("ascii")
            with self.assertRaisesRegex(ValueError, "unsafe member"):
                self.validate_existing(destination, unsafe)

    def test_builder_is_bounded_local_and_runtime_independent(self) -> None:
        source = BUILDER.read_text(encoding="utf-8")
        lowered = source.lower()
        for forbidden in ("import torch", "import transformers", "trust_remote_code", "/usr/local/bin/seen"):
            self.assertNotIn(forbidden, lowered)
        for required in (
            "MAX_CHUNK_BYTES = 1024 * 1024", "MAX_ARTIFACT_BYTES = 68_719_476_736",
            "tempfile.mkdtemp", "os.rename", "os.fsync", "validate_sqw(",
            "SOURCE_SHARDS = 18", "INCLUDED_TENSORS = 866",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
