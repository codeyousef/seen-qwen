#!/usr/bin/env python3
"""Build the first complete, lossless BF16 Qwen text+MTP SQW artifact.

This is bounded offline conversion/certification tooling.  It never imports
model code, torch, transformers, or safetensors.  It treats the locked index,
Safetensors headers, shard bytes, policy, and assets as hostile inputs and
streams one bounded chunk at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import tomllib
from typing import BinaryIO, Iterable


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_CONTRACT = "seen-qwen38-text-v1"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
CATALOG_SHA256 = "5f466d43bae3059e54f0bfe183d0e82c822242f45a834d778414d3e5b5248f1f"
COMPILER_SHA256 = "44a90884a9ff188718ed839aed4c8d692c8c520216c6a7b072ca6d0d35aa7fbc"
COMPATIBILITY_SHA256 = "f39aebda1fdafc20d04b6c6a0072a491afea3f8f9ab836fd173ead0d1fe1ea33"
SEEN_COMMIT = "a05932e231e33c7512be3432e9d88a938466f820"
SEEN_VERSION = "0.19.4"
POLICY_ID = "bf16-bringup-v1"
INCLUDED_TENSORS = 866
TEXT_TENSORS = 851
MTP_TENSORS = 15
VISION_TENSORS = 333
SOURCE_TENSORS = 1199
SOURCE_SHARDS = 18
SOURCE_TENSOR_BYTES = 55_562_855_904
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 68_719_476_736
HEADER_BYTES = 256
DIRECTORY_ENTRY_BYTES = 256
FOOTER_HEADER_BYTES = 64
FOOTER_ENTRY_BYTES = 64
WHOLE_DIGEST_BYTES = 32
SECTION_IDS = {"manifest": 1, "directory": 2, "names": 3, "payload": 4, "evidence": 5}
ASSET_IDENTITIES = {
    "README.md": (65012, "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"),
    "chat_template.jinja": (8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "config.json": (4312, "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"),
    "generation_config.json": (202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
    "merges.txt": (3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    "tokenizer_config.json": (17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    "vocab.json": (6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
}


class DuplicateKey(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(key)
        result[key] = value
    return result


def strict_json_bytes(data: bytes, *, maximum: int) -> object:
    if not data or len(data) > maximum:
        raise ValueError("JSON input is empty or exceeds its bound")
    return json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def align_up(value: int, alignment: int) -> int:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    result = (value + alignment - 1) & ~(alignment - 1)
    if result > 0x7FFF_FFFF_FFFF_FFFF:
        raise OverflowError("aligned extent is outside the signed-safe range")
    return result


def digest_file(path: Path, chunk_bytes: int) -> tuple[int, str]:
    if chunk_bytes < 1 or chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError("chunk bound must be between 1 byte and 1 MiB")
    total = 0
    digest = sha256()
    with path.open("rb", buffering=0) as source:
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            total += len(block)
            digest.update(block)
    return total, digest.hexdigest()


def is_included(name: str) -> bool:
    return name == "lm_head.weight" or name.startswith("model.language_model.") or name.startswith("mtp.")


def semantic_role(name: str) -> tuple[int, str]:
    if name == "model.language_model.embed_tokens.weight":
        return 1, "embedding"
    if name == "lm_head.weight":
        return 2, "lm_head"
    if name.startswith("mtp."):
        return 14, "mtp"
    if ".self_attn.q_proj." in name:
        return 3, "attention_q"
    if ".self_attn.k_proj." in name:
        return 4, "attention_k"
    if ".self_attn.v_proj." in name:
        return 5, "attention_v"
    if ".self_attn.o_proj." in name:
        return 6, "attention_o"
    if ".linear_attn." in name:
        if name.endswith(".A_log") or name.endswith(".dt_bias") or ".conv1d." in name:
            return 8, "gdn_state_parameter"
        return 7, "gdn_projection"
    if ".mlp.gate_proj." in name:
        return 9, "mlp_gate"
    if ".mlp.up_proj." in name:
        return 10, "mlp_up"
    if ".mlp.down_proj." in name:
        return 11, "mlp_down"
    if "norm" in name:
        return 12, "norm"
    if name.endswith(".bias") or name.endswith(".scalar"):
        return 13, "bias_scalar"
    raise ValueError(f"required tensor has no frozen semantic role: {name}")


@dataclass
class SourceTensor:
    name: str
    shard: Path
    data_offset: int
    data_length: int
    shape: tuple[int, ...]
    role_id: int
    role_name: str
    source_sha256: str = ""
    data_output_offset: int = 0
    name_offset: int = 0

    @property
    def logical_elements(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
            if product > 0x7FFF_FFFF_FFFF_FFFF:
                raise OverflowError(f"tensor shape overflows signed-safe geometry: {self.name}")
        return product

    @property
    def row_elements(self) -> int:
        return self.shape[-1]


def read_index(path: Path) -> tuple[dict[str, str], bytes]:
    data = path.read_bytes()
    if sha256(data).hexdigest() != INDEX_SHA256:
        raise ValueError("tensor index digest differs from the immutable lock")
    document = strict_json_bytes(data, maximum=1024 * 1024)
    if not isinstance(document, dict) or set(document) != {"metadata", "weight_map"}:
        raise ValueError("tensor index root differs from the closed contract")
    weight_map = document["weight_map"]
    if not isinstance(weight_map, dict) or len(weight_map) != SOURCE_TENSORS:
        raise ValueError("tensor index must contain exactly 1199 entries")
    result: dict[str, str] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("tensor index entries must be strings")
        if "/" in shard or "\\" in shard or not shard.startswith("model-") or not shard.endswith(".safetensors"):
            raise ValueError("tensor index contains an unsafe shard name")
        result[name] = shard
    return result, data


def read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    size = path.stat().st_size
    with path.open("rb", buffering=0) as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated Safetensors header: {path.name}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes < 2 or header_bytes > MAX_HEADER_BYTES or 8 + header_bytes > size:
            raise ValueError(f"Safetensors header length is outside bounds: {path.name}")
        encoded = source.read(header_bytes)
        if len(encoded) != header_bytes:
            raise ValueError(f"truncated Safetensors JSON header: {path.name}")
    document = strict_json_bytes(encoded, maximum=MAX_HEADER_BYTES)
    if not isinstance(document, dict):
        raise ValueError(f"Safetensors header root is not an object: {path.name}")
    return 8 + header_bytes, document


def load_source_tensors(source_root: Path, index: dict[str, str]) -> tuple[list[SourceTensor], int]:
    encountered: set[str] = set()
    tensors: list[SourceTensor] = []
    total_bytes = 0
    for ordinal in range(1, SOURCE_SHARDS + 1):
        shard_name = f"model-{ordinal:05d}-of-00018.safetensors"
        shard_path = source_root / shard_name
        data_start, header = read_safetensors_header(shard_path)
        shard_size = shard_path.stat().st_size
        ranges: list[tuple[int, int, str]] = []
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in encountered or name not in index or index[name] != shard_name:
                raise ValueError(f"duplicate, unknown, or misassigned tensor: {name}")
            if not isinstance(metadata, dict) or set(metadata) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"tensor metadata differs from the closed contract: {name}")
            dtype = metadata["dtype"]
            shape = metadata["shape"]
            offsets = metadata["data_offsets"]
            if dtype != "BF16" or not isinstance(shape, list) or not 1 <= len(shape) <= 8:
                raise ValueError(f"required source dtype/rank is unsupported: {name}")
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(value, int) for value in offsets):
                raise ValueError(f"tensor data offsets are invalid: {name}")
            start, end = offsets
            if start < 0 or end <= start or data_start + end > shard_size:
                raise ValueError(f"tensor extent is outside its shard: {name}")
            dimensions = tuple(shape)
            if any(not isinstance(value, int) or value <= 0 for value in dimensions):
                raise ValueError(f"tensor shape is invalid: {name}")
            logical = 1
            for value in dimensions:
                logical *= value
                if logical > 0x7FFF_FFFF_FFFF_FFFF:
                    raise OverflowError(f"tensor geometry overflows: {name}")
            if logical * 2 != end - start:
                raise ValueError(f"BF16 tensor byte geometry differs from shape: {name}")
            ranges.append((start, end, name))
            encountered.add(name)
            total_bytes += end - start
            if is_included(name):
                role_id, role_name = semantic_role(name)
                tensors.append(SourceTensor(name, shard_path, data_start + start, end - start, dimensions, role_id, role_name))
        ordered = sorted(ranges)
        for left, right in zip(ordered, ordered[1:]):
            if left[1] > right[0]:
                raise ValueError(f"overlapping tensor extents in {shard_name}")
    if encountered != set(index):
        raise ValueError("source shards do not exactly cover the immutable tensor index")
    if len(encountered) != SOURCE_TENSORS or total_bytes != SOURCE_TENSOR_BYTES:
        raise ValueError("source tensor count or aggregate bytes differ from the lock")
    tensors.sort(key=lambda tensor: tensor.name.encode("utf-8"))
    if len(tensors) != INCLUDED_TENSORS:
        raise ValueError("required text+MTP tensor count is not 866")
    text = sum(not tensor.name.startswith("mtp.") for tensor in tensors)
    mtp = sum(tensor.name.startswith("mtp.") for tensor in tensors)
    if (text, mtp) != (TEXT_TENSORS, MTP_TENSORS):
        raise ValueError("required text/MTP class counts differ from the lock")
    return tensors, total_bytes


def copy_range(source: BinaryIO, offset: int, length: int, sink: BinaryIO | None, digest, chunk_bytes: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        block = source.read(min(remaining, chunk_bytes))
        if not block:
            raise ValueError("source tensor became truncated during streaming")
        if sink is not None:
            sink.write(block)
        digest.update(block)
        remaining -= len(block)


def hash_tensors(tensors: Iterable[SourceTensor], chunk_bytes: int) -> None:
    for tensor in tensors:
        digest = sha256()
        with tensor.shard.open("rb", buffering=0) as source:
            copy_range(source, tensor.data_offset, tensor.data_length, None, digest, chunk_bytes)
        tensor.source_sha256 = digest.hexdigest()


def validate_policy(path: Path) -> str:
    data = path.read_bytes()
    document = tomllib.loads(data.decode("utf-8", errors="strict"))
    expected = {
        "schema": "seen-qwen-quantization-policy-v1",
        "policy": {
            POLICY_ID: {
                "maturity": "offline-correctness",
                "tensor_scope": "required-text-and-mtp",
                "source_dtype": "BF16",
                "runtime_codec": "BF16",
                "ordering": "canonical-utf8-tensor-name",
                "lossy": False,
                "quality_approved": False,
                "runtime_profile": False,
                "fallback": "prohibited",
            }
        },
    }
    if document != expected:
        raise ValueError("bring-up policy differs from the closed lossless contract")
    return sha256(data).hexdigest()


def validate_source_manifest(source_root: Path, chunk_bytes: int) -> bytes:
    path = source_root / "manifest.json"
    data = path.read_bytes()
    document = strict_json_bytes(data, maximum=65536)
    if not isinstance(document, dict) or document.get("schema") != "seen-qwen-official-full-model-inputs-v1":
        raise ValueError("official source manifest schema is invalid")
    if document.get("model_id") != MODEL_ID or document.get("model_revision") != MODEL_REVISION:
        raise ValueError("official source manifest identity changed")
    if document.get("index_sha256") != INDEX_SHA256 or document.get("tensor_bytes") != SOURCE_TENSOR_BYTES:
        raise ValueError("official source manifest geometry changed")
    shards = document.get("shards")
    if not isinstance(shards, list) or len(shards) != SOURCE_SHARDS:
        raise ValueError("official source manifest must contain 18 shards")
    for ordinal, entry in enumerate(shards, 1):
        expected_name = f"model-{ordinal:05d}-of-00018.safetensors"
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "lfs_sha256"} or entry.get("path") != expected_name:
            raise ValueError("official source shard manifest is reordered or malformed")
        size, digest = digest_file(source_root / expected_name, chunk_bytes)
        if size != entry["bytes"] or digest != entry["lfs_sha256"]:
            raise ValueError(f"official source shard failed its immutable hash: {expected_name}")
    return data


def validate_assets(asset_root: Path, chunk_bytes: int) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for name in sorted(ASSET_IDENTITIES):
        expected_size, expected_digest = ASSET_IDENTITIES[name]
        size, digest = digest_file(asset_root / name, chunk_bytes)
        if (size, digest) != (expected_size, expected_digest):
            raise ValueError(f"locked tokenizer/configuration asset changed: {name}")
        result.append({"path": name, "bytes": str(size), "sha256": digest})
    return result


def _write_u16(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", target, offset, value)


def _write_u32(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", target, offset, value)


def _write_u64(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", target, offset, value)


def build_names(tensors: list[SourceTensor]) -> bytes:
    result = bytearray()
    for tensor in tensors:
        encoded = tensor.name.encode("utf-8")
        if not encoded or len(encoded) > 1024:
            raise ValueError("tensor name is empty or outside the SQW bound")
        tensor.name_offset = len(result)
        result.extend(encoded)
    return bytes(result)


def assign_payload_offsets(tensors: list[SourceTensor], payload_offset: int) -> int:
    cursor = payload_offset
    for tensor in tensors:
        cursor = align_up(cursor, 64)
        tensor.data_output_offset = cursor
        cursor += tensor.data_length
        if cursor > MAX_ARTIFACT_BYTES:
            raise ValueError("lossless artifact exceeds its explicit 64 GiB bound")
    return cursor - payload_offset


def build_directory(tensors: list[SourceTensor]) -> bytes:
    directory = bytearray(len(tensors) * DIRECTORY_ENTRY_BYTES)
    for index, tensor in enumerate(tensors):
        base = index * DIRECTORY_ENTRY_BYTES
        encoded = tensor.name.encode("utf-8")
        _write_u64(directory, base, tensor.name_offset)
        _write_u32(directory, base + 8, len(encoded))
        _write_u16(directory, base + 12, tensor.role_id)
        _write_u16(directory, base + 14, 7)
        _write_u16(directory, base + 16, 1)
        _write_u16(directory, base + 18, len(tensor.shape))
        for dimension_index, dimension in enumerate(tensor.shape):
            _write_u64(directory, base + 24 + dimension_index * 8, dimension)
        _write_u64(directory, base + 88, tensor.logical_elements)
        _write_u64(directory, base + 96, tensor.data_output_offset)
        _write_u64(directory, base + 104, tensor.data_length)
        _write_u64(directory, base + 160, tensor.row_elements)
        _write_u32(directory, base + 168, 0)
        _write_u32(directory, base + 172, 64)
        digest = bytes.fromhex(tensor.source_sha256)
        directory[base + 176 : base + 208] = digest
        directory[base + 208 : base + 240] = digest
    return bytes(directory)


def tensor_manifest(tensor: SourceTensor) -> dict[str, object]:
    empty = {"offset": "0", "length": "0"}
    return {
        "name": tensor.name,
        "semantic_role": tensor.role_name,
        "source_dtype": "BF16",
        "runtime_codec": "BF16",
        "rank": len(tensor.shape),
        "shape": [str(value) for value in tensor.shape],
        "logical_elements": str(tensor.logical_elements),
        "data": {"offset": str(tensor.data_output_offset), "length": str(tensor.data_length)},
        "scale": dict(empty), "zero": dict(empty), "metadata": dict(empty),
        "row_elements": str(tensor.row_elements),
        "group_elements": 0,
        "required_alignment": 64,
        "source_sha256": tensor.source_sha256,
        "converted_sha256": tensor.source_sha256,
    }


def sqw_manifest(tensors: list[SourceTensor], model_lock: str, source_lock: str, policy: str, directory_digest: str) -> dict[str, object]:
    return {
        "schema": "seen-qwen-sqw-manifest-v1", "format_version": "1.0",
        "model_lock_sha256": model_lock, "source_lock_sha256": source_lock,
        "conversion_policy_sha256": policy, "tensor_contract": MODEL_CONTRACT,
        "catalog_sha256": CATALOG_SHA256, "directory_sha256": directory_digest,
        "payload_order": "canonical_utf8_tensor_name",
        "compatibility": {"required_features": [], "reader_major": 1, "reader_minor": 0},
        "tensors": [tensor_manifest(tensor) for tensor in tensors],
    }


def plan_fingerprint(policy_digest: str, included_bytes: int) -> str:
    material = (
        "qwen38-bf16-bringup-plan-v1\n"
        f"model_revision={MODEL_REVISION}\n"
        f"catalog_sha256={CATALOG_SHA256}\n"
        f"policy_sha256={policy_digest}\n"
        f"included_tensors={INCLUDED_TENSORS}\n"
        f"included_bytes={included_bytes}\n"
        "source_window_bytes=1048576\nwriter_chunk_bytes=1048576\nworkers=1\n"
    )
    return sha256(material.encode()).hexdigest()


def conversion_evidence(tensors: list[SourceTensor], model_lock: str, source_lock: str, policy: str, included_bytes: int) -> bytes:
    entries = [
        {"ordinal": index, "name": tensor.name, "source_bytes": str(tensor.data_length),
         "converted_bytes": str(tensor.data_length), "source_sha256": tensor.source_sha256,
         "converted_sha256": tensor.source_sha256}
        for index, tensor in enumerate(tensors)
    ]
    return canonical_json({
        "schema": "seen-qwen-conversion-journal-v1", "version": "1", "output_format": "SQW1",
        "model_revision": MODEL_REVISION, "model_lock_sha256": model_lock,
        "source_lock_sha256": source_lock, "conversion_policy_sha256": policy,
        "catalog_sha256": CATALOG_SHA256, "plan_sha256": plan_fingerprint(policy, included_bytes),
        "toolchain_compatibility_sha256": COMPATIBILITY_SHA256,
        "tensor_order": "canonical_included_catalog_prefix", "completed_count": len(entries),
        "entries": entries,
    })


@dataclass(frozen=True)
class Layout:
    manifest: bytes
    directory: bytes
    names: bytes
    evidence: bytes
    sections: dict[str, tuple[int, int]]
    footer_offset: int
    footer_length: int
    whole_digest_offset: int
    file_bytes: int


def compute_layout(tensors: list[SourceTensor], model_lock: str, source_lock: str, policy: str, evidence: bytes) -> Layout:
    names = build_names(tensors)
    payload_offset = 4096
    stable: tuple[int, int, int, int] | None = None
    for _ in range(32):
        payload_length = assign_payload_offsets(tensors, payload_offset)
        directory = build_directory(tensors)
        manifest = canonical_json(sqw_manifest(tensors, model_lock, source_lock, policy, sha256(directory).hexdigest()))
        directory_offset = align_up(HEADER_BYTES + len(manifest), 64)
        names_offset = align_up(directory_offset + len(directory), 64)
        next_payload = align_up(names_offset + len(names), 4096)
        state = (next_payload, len(manifest), directory_offset, names_offset)
        if state == stable:
            break
        stable = state
        payload_offset = next_payload
    else:
        raise ValueError("SQW layout did not converge")
    payload_length = assign_payload_offsets(tensors, payload_offset)
    directory = build_directory(tensors)
    manifest = canonical_json(sqw_manifest(tensors, model_lock, source_lock, policy, sha256(directory).hexdigest()))
    directory_offset = align_up(HEADER_BYTES + len(manifest), 64)
    names_offset = align_up(directory_offset + len(directory), 64)
    if payload_offset != align_up(names_offset + len(names), 4096):
        raise ValueError("SQW layout changed after convergence")
    evidence_offset = align_up(payload_offset + payload_length, 64)
    footer_offset = align_up(evidence_offset + len(evidence), 64)
    footer_length = FOOTER_HEADER_BYTES + 5 * FOOTER_ENTRY_BYTES
    file_bytes = footer_offset + footer_length
    if file_bytes > MAX_ARTIFACT_BYTES:
        raise ValueError("complete SQW exceeds its signed-safe artifact bound")
    sections = {
        "manifest": (HEADER_BYTES, len(manifest)), "directory": (directory_offset, len(directory)),
        "names": (names_offset, len(names)), "payload": (payload_offset, payload_length),
        "evidence": (evidence_offset, len(evidence)),
    }
    return Layout(manifest, directory, names, evidence, sections, footer_offset, footer_length, footer_offset + 16, file_bytes)


def write_zeros(sink: BinaryIO, count: int, digest, chunk_bytes: int) -> None:
    block = b"\0" * min(chunk_bytes, 65536)
    remaining = count
    while remaining:
        piece = block[: min(remaining, len(block))]
        sink.write(piece); digest.update(piece); remaining -= len(piece)


def write_sqw(path: Path, tensors: list[SourceTensor], layout: Layout, model_lock: str, policy: str, chunk_bytes: int) -> tuple[str, str]:
    header = bytearray(HEADER_BYTES)
    header[:4] = b"SQW1"; _write_u32(header, 4, 0x01020304); _write_u16(header, 8, 1)
    _write_u16(header, 10, 0); _write_u32(header, 12, HEADER_BYTES); _write_u32(header, 16, 1)
    _write_u64(header, 24, HEADER_BYTES); _write_u64(header, 32, len(layout.manifest))
    _write_u64(header, 40, layout.sections["directory"][0]); _write_u32(header, 48, DIRECTORY_ENTRY_BYTES)
    _write_u32(header, 52, len(tensors)); _write_u64(header, 56, layout.sections["names"][0])
    _write_u64(header, 64, len(layout.names)); _write_u64(header, 72, layout.sections["payload"][0])
    _write_u64(header, 80, layout.sections["payload"][1]); _write_u64(header, 88, layout.sections["evidence"][0])
    _write_u64(header, 96, len(layout.evidence)); _write_u64(header, 104, layout.footer_offset)
    _write_u64(header, 112, layout.footer_length); _write_u64(header, 120, layout.whole_digest_offset)
    header[128:160] = bytes.fromhex(model_lock); header[160:192] = bytes.fromhex(policy)
    footer = bytearray(layout.footer_length)
    footer[:4] = b"SQWF"; _write_u16(footer, 4, 1); _write_u16(footer, 6, FOOTER_HEADER_BYTES)
    _write_u16(footer, 8, FOOTER_ENTRY_BYTES); _write_u16(footer, 10, 1); _write_u32(footer, 12, 5)
    section_hashes = {"manifest": sha256(layout.manifest).digest(), "directory": sha256(layout.directory).digest(), "names": sha256(layout.names).digest(), "evidence": sha256(layout.evidence).digest()}
    payload_hash = sha256()
    with path.open("w+b", buffering=0) as sink:
        sink.truncate(layout.file_bytes)
        sink.seek(0); sink.write(header)
        for name, data in (("manifest", layout.manifest), ("directory", layout.directory), ("names", layout.names)):
            sink.seek(layout.sections[name][0]); sink.write(data)
        payload_start, payload_length = layout.sections["payload"]
        sink.seek(payload_start); cursor = payload_start
        for tensor in tensors:
            if tensor.data_output_offset < cursor:
                raise ValueError("tensor payload order regressed")
            write_zeros(sink, tensor.data_output_offset - cursor, payload_hash, chunk_bytes)
            with tensor.shard.open("rb", buffering=0) as source:
                copy_range(source, tensor.data_offset, tensor.data_length, sink, payload_hash, chunk_bytes)
            cursor = tensor.data_output_offset + tensor.data_length
        if cursor != payload_start + payload_length:
            raise ValueError("payload extent changed during streaming")
        section_hashes["payload"] = payload_hash.digest()
        sink.seek(layout.sections["evidence"][0]); sink.write(layout.evidence)
        for index, name in enumerate(("manifest", "directory", "names", "payload", "evidence")):
            entry = FOOTER_HEADER_BYTES + index * FOOTER_ENTRY_BYTES
            offset, length = layout.sections[name]
            _write_u32(footer, entry, SECTION_IDS[name]); _write_u64(footer, entry + 8, offset)
            _write_u64(footer, entry + 16, length); footer[entry + 24 : entry + 56] = section_hashes[name]
        sink.seek(layout.footer_offset); sink.write(footer); sink.flush(); os.fsync(sink.fileno())
    _, whole_digest = digest_file(path, chunk_bytes)
    with path.open("r+b", buffering=0) as sink:
        sink.seek(layout.whole_digest_offset); sink.write(bytes.fromhex(whole_digest)); sink.flush(); os.fsync(sink.fileno())
    size, file_digest = digest_file(path, chunk_bytes)
    if size != layout.file_bytes:
        raise ValueError("final SQW extent changed")
    return whole_digest, file_digest


def validate_sqw(
    path: Path,
    tensors: list[SourceTensor],
    layout: Layout,
    expected_model_lock: str,
    expected_policy: str,
    expected_whole: str,
    expected_file: str,
    chunk_bytes: int,
) -> None:
    """Read back the complete sealed SQW before it can be promoted."""
    if path.stat().st_size != layout.file_bytes:
        raise ValueError("read-back SQW extent differs from the sealed layout")
    with path.open("rb", buffering=0) as source:
        header = source.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:4] != b"SQW1":
            raise ValueError("read-back SQW header is truncated or has wrong magic")
        expected_header = {
            4: ("<I", 0x01020304), 8: ("<H", 1), 10: ("<H", 0),
            12: ("<I", HEADER_BYTES), 16: ("<I", 1),
            24: ("<Q", HEADER_BYTES), 32: ("<Q", len(layout.manifest)),
            40: ("<Q", layout.sections["directory"][0]),
            48: ("<I", DIRECTORY_ENTRY_BYTES), 52: ("<I", len(tensors)),
            56: ("<Q", layout.sections["names"][0]),
            64: ("<Q", len(layout.names)),
            72: ("<Q", layout.sections["payload"][0]),
            80: ("<Q", layout.sections["payload"][1]),
            88: ("<Q", layout.sections["evidence"][0]),
            96: ("<Q", len(layout.evidence)), 104: ("<Q", layout.footer_offset),
            112: ("<Q", layout.footer_length), 120: ("<Q", layout.whole_digest_offset),
        }
        for offset, (format_string, expected) in expected_header.items():
            if struct.unpack_from(format_string, header, offset)[0] != expected:
                raise ValueError(f"read-back SQW header field at {offset} changed")
        if header[128:160].hex() != expected_model_lock or header[160:192].hex() != expected_policy:
            raise ValueError("read-back SQW lock identities changed")
        if any(header[20:24]) or any(header[192:]):
            raise ValueError("read-back SQW reserved header bytes are nonzero")

        source.seek(layout.footer_offset)
        footer = source.read(layout.footer_length)
        if len(footer) != layout.footer_length or footer[:4] != b"SQWF":
            raise ValueError("read-back SQW footer is truncated or has wrong magic")
        if struct.unpack_from("<HHHHI", footer, 4) != (1, FOOTER_HEADER_BYTES, FOOTER_ENTRY_BYTES, 1, 5):
            raise ValueError("read-back SQW footer contract changed")
        if footer[16:48].hex() != expected_whole or any(footer[48:64]):
            raise ValueError("read-back SQW embedded whole digest or reserved bytes changed")

        section_digests: dict[str, bytes] = {}
        for ordinal, name in enumerate(("manifest", "directory", "names", "payload", "evidence")):
            base = FOOTER_HEADER_BYTES + ordinal * FOOTER_ENTRY_BYTES
            identifier = struct.unpack_from("<I", footer, base)[0]
            offset = struct.unpack_from("<Q", footer, base + 8)[0]
            length = struct.unpack_from("<Q", footer, base + 16)[0]
            if identifier != SECTION_IDS[name] or (offset, length) != layout.sections[name]:
                raise ValueError(f"read-back SQW {name} footer entry changed")
            if any(footer[base + 4 : base + 8]) or any(footer[base + 56 : base + 64]):
                raise ValueError(f"read-back SQW {name} footer reserved bytes are nonzero")
            section_digests[name] = footer[base + 24 : base + 56]

        source.seek(layout.sections["directory"][0])
        directory = source.read(len(layout.directory))
        source.seek(layout.sections["names"][0])
        names = source.read(len(layout.names))
        if directory != layout.directory or names != layout.names:
            raise ValueError("read-back SQW directory or name table differs from the plan")
        for ordinal in (0, len(tensors) // 2, len(tensors) - 1):
            tensor = tensors[ordinal]
            base = ordinal * DIRECTORY_ENTRY_BYTES
            name_offset = struct.unpack_from("<Q", directory, base)[0]
            name_length = struct.unpack_from("<I", directory, base + 8)[0]
            if names[name_offset : name_offset + name_length].decode("utf-8", errors="strict") != tensor.name:
                raise ValueError(f"read-back SQW boundary tensor name changed at {ordinal}")
            if directory[base + 176 : base + 208].hex() != tensor.source_sha256 or directory[base + 208 : base + 240].hex() != tensor.source_sha256:
                raise ValueError(f"read-back SQW boundary tensor digest changed at {ordinal}")

    final_digest = sha256()
    logical_digest = sha256()
    section_hashers = {name: sha256() for name in layout.sections}
    cursor = 0
    with path.open("rb", buffering=0) as source:
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            final_digest.update(block)
            logical = bytearray(block)
            zero_start = max(layout.whole_digest_offset, cursor)
            zero_end = min(layout.whole_digest_offset + WHOLE_DIGEST_BYTES, cursor + len(block))
            if zero_start < zero_end:
                logical[zero_start - cursor : zero_end - cursor] = b"\0" * (zero_end - zero_start)
            logical_digest.update(logical)
            block_end = cursor + len(block)
            for name, (section_start, section_length) in layout.sections.items():
                section_end = section_start + section_length
                overlap_start = max(cursor, section_start)
                overlap_end = min(block_end, section_end)
                if overlap_start < overlap_end:
                    section_hashers[name].update(block[overlap_start - cursor : overlap_end - cursor])
            cursor = block_end
    if cursor != layout.file_bytes or final_digest.hexdigest() != expected_file:
        raise ValueError("read-back SQW conventional file digest changed")
    if logical_digest.hexdigest() != expected_whole:
        raise ValueError("read-back SQW logical whole digest changed")
    for name, digest in section_hashers.items():
        if digest.digest() != section_digests[name]:
            raise ValueError(f"read-back SQW {name} section digest changed")


def source_lock_document(source_manifest_sha: str, policy_sha: str, assets: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema": "seen-qwen-source-lock-v1", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "model_input_manifest_sha256": source_manifest_sha,
        "tensor_index_sha256": INDEX_SHA256, "catalog_sha256": CATALOG_SHA256,
        "quantization_policy_sha256": policy_sha,
        "seen": {"version": SEEN_VERSION, "commit": SEEN_COMMIT, "compiler_sha256": COMPILER_SHA256,
                 "compatibility_sha256": COMPATIBILITY_SHA256, "cpu_baseline": "x86-64"},
        "assets": assets,
    }


def engine_id(model_lock: str, source_lock: str, weights: str, policy: str) -> str:
    material = (
        "seen-qwen-engine-artifact-v1\n" f"model_lock_sha256={model_lock}\n"
        f"source_lock_sha256={source_lock}\n" f"weights_sha256={weights}\n"
        f"policy_sha256={policy}\n" f"compiler_sha256={COMPILER_SHA256}\n"
    )
    return sha256(material.encode()).hexdigest()


def engine_document(model_lock: str, source_lock: str, policy: str, weights_file_sha: str, whole_sha: str, weights_bytes: int, assets: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema": "seen-qwen-engine-artifact-v1", "version": 1, "maturity": "offline-validated",
        "engine_id": engine_id(model_lock, source_lock, weights_file_sha, policy),
        "model_lock_sha256": model_lock, "source_lock_sha256": source_lock,
        "format": {"sqw_version": "1.0", "weights_file": "weights.sqw", "weights_sha256": weights_file_sha,
                   "sqw_whole_sha256": whole_sha, "tensor_count": INCLUDED_TENSORS, "weights_bytes": str(weights_bytes)},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "contract": MODEL_CONTRACT,
                  "layers": 64, "vocab_size": 248320, "native_context": 262144,
                  "text_tensors": TEXT_TENSORS, "mtp_tensors": MTP_TENSORS, "vision_tensors": 0},
        "quantization": {"policy_id": POLICY_ID, "policy_sha256": policy, "source_dtype": "BF16",
                         "runtime_codec": "BF16", "tensor_count": INCLUDED_TENSORS, "lossy": False,
                         "quality_approved": False, "fallback": "prohibited"},
        "tokenizer": assets, "intended_backend": {"kind": "cuda", "target": "sm_89", "bound": False},
        "memory": {"weight_bytes": str(weights_bytes), "persistent_state_bytes": "0",
                   "max_profile_bytes": "0", "status": "not-runtime-sized"},
        "compatible_profiles": [],
        "created_by": {"seen_version": SEEN_VERSION, "seen_commit": SEEN_COMMIT,
                       "compiler_sha256": COMPILER_SHA256, "compatibility_sha256": COMPATIBILITY_SHA256,
                       "cpu_baseline": "x86-64"},
        "checksums_file": "checksums.sha256",
    }


def write_file(path: Path, data: bytes) -> None:
    with path.open("xb", buffering=0) as sink:
        sink.write(data); sink.flush(); os.fsync(sink.fileno())


def build(args: argparse.Namespace) -> Path:
    if args.chunk_bytes < 1 or args.chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError("chunk-bytes must be in 1..1048576")
    source_root = args.source_root.resolve(); asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    index, _ = read_index(args.index.resolve())
    policy_digest = validate_policy(args.policy.resolve())
    source_manifest = validate_source_manifest(source_root, args.chunk_bytes)
    assets = validate_assets(asset_root, args.chunk_bytes)
    model_lock_bytes = args.model_lock.resolve().read_bytes()
    model_lock_doc = strict_json_bytes(model_lock_bytes, maximum=65536)
    if not isinstance(model_lock_doc, dict) or model_lock_doc.get("schema") != "seen-qwen-model-lock-v1" or model_lock_doc.get("revision") != MODEL_REVISION:
        raise ValueError("model lock identity is incompatible")
    model_lock_digest = sha256(model_lock_bytes).hexdigest()
    tensors, _ = load_source_tensors(source_root, index)
    hash_tensors(tensors, args.chunk_bytes)
    included_bytes = sum(tensor.data_length for tensor in tensors)
    source_lock_bytes = canonical_json(source_lock_document(sha256(source_manifest).hexdigest(), policy_digest, assets))
    source_lock_digest = sha256(source_lock_bytes).hexdigest()
    evidence = conversion_evidence(tensors, model_lock_digest, source_lock_digest, policy_digest, included_bytes)
    layout = compute_layout(tensors, model_lock_digest, source_lock_digest, policy_digest, evidence)
    staging = Path(tempfile.mkdtemp(prefix=".qwn-034a-stage-", dir=output_root))
    try:
        weights = staging / "weights.sqw"
        whole_digest, weights_digest = write_sqw(weights, tensors, layout, model_lock_digest, policy_digest, args.chunk_bytes)
        validate_sqw(weights, tensors, layout, model_lock_digest, policy_digest,
                     whole_digest, weights_digest, args.chunk_bytes)
        write_file(staging / "conversion-evidence.json", evidence)
        write_file(staging / "source-lock.json", source_lock_bytes)
        write_file(staging / "model-lock.json", model_lock_bytes)
        for asset in assets:
            shutil.copyfile(asset_root / asset["path"], staging / asset["path"])
        engine = canonical_json(engine_document(model_lock_digest, source_lock_digest, policy_digest, weights_digest, whole_digest, layout.file_bytes, assets))
        write_file(staging / "engine.json", engine)
        checksummed = ["README.md", "chat_template.jinja", "config.json", "conversion-evidence.json", "engine.json",
                       "generation_config.json", "merges.txt", "model-lock.json", "source-lock.json",
                       "tokenizer_config.json", "vocab.json", "weights.sqw"]
        lines = []
        for name in checksummed:
            digest = weights_digest if name == "weights.sqw" else digest_file(staging / name, args.chunk_bytes)[1]
            lines.append(f"{digest}  {name}\n")
        write_file(staging / "checksums.sha256", "".join(lines).encode("ascii"))
        engine_doc = strict_json_bytes(engine, maximum=1024 * 1024)
        artifact_id = engine_doc["engine_id"]
        destination = output_root / artifact_id
        if destination.exists():
            existing = (destination / "checksums.sha256").read_bytes()
            if existing != (staging / "checksums.sha256").read_bytes():
                raise FileExistsError("content-addressed destination exists with different checksums")
            shutil.rmtree(staging)
            return destination
        os.rename(staging, destination)
        directory = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--asset-root", type=Path, required=True)
    value.add_argument("--index", type=Path, required=True)
    value.add_argument("--model-lock", type=Path, required=True)
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--chunk-bytes", type=int, default=MAX_CHUNK_BYTES)
    return value


def main() -> int:
    destination = build(parser().parse_args())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
