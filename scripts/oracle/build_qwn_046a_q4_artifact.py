#!/usr/bin/env python3
"""Build the complete experimental Q4 Qwen text+MTP SQW artifact.

This bounded offline converter consumes only the immutable official BF16
checkpoint bytes.  It uses the normative Q4_SYM_G64 reference rules, stages
one bounded row batch at a time, and atomically promotes a fully sealed SQW.
It does not import model code, torch, transformers, or safetensors.
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

import numpy as np

from scripts.oracle import build_qwn_034a_bf16_artifact as base


POLICY_ID = "q4-bringup-v1"
SEEN_VERSION = "0.20.10"
SEEN_COMMIT = "f714d85d0e53ca907ada16b0c58149fd93250211"
COMPILER_SHA256 = "7b71bba386641ce6655f1d7e5a124dbd5828709709af498b9f8b03fa1e7bda30"
COMPATIBILITY_SHA256 = "af7209cc9407c3933f969cdcf74164956ede888993eba021e15519ff25fcf53e"
Q4_CODEC_ID = 5
GROUP_ELEMENTS = 64
MAX_BATCH_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 24 * 1024 * 1024 * 1024


@dataclass
class Q4Tensor:
    source: base.SourceTensor
    data_path: Path
    scale_path: Path
    data_length: int
    scale_length: int
    converted_sha256: str
    name_offset: int = 0
    data_output_offset: int = 0
    scale_output_offset: int = 0

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    @property
    def logical_elements(self) -> int:
        return self.source.logical_elements

    @property
    def row_elements(self) -> int:
        return self.source.row_elements

    @property
    def role_id(self) -> int:
        return self.source.role_id

    @property
    def role_name(self) -> str:
        return self.source.role_name

    @property
    def source_sha256(self) -> str:
        return self.source.source_sha256


def validate_policy(path: Path) -> str:
    data = path.read_bytes()
    expected = {
        "schema": "seen-qwen-quantization-policy-v1",
        "policy": {POLICY_ID: {
            "maturity": "experimental-hardware",
            "tensor_scope": "required-text-and-mtp",
            "source_dtype": "BF16",
            "runtime_codec": "Q4_SYM_G64",
            "ordering": "canonical-utf8-tensor-name",
            "lossy": True,
            "quality_approved": False,
            "runtime_profile": True,
            "fallback": "prohibited",
        }},
    }
    if tomllib.loads(data.decode("utf-8", errors="strict")) != expected:
        raise ValueError("Q4 bring-up policy differs from its closed contract")
    return sha256(data).hexdigest()


def q4_geometry(tensor: base.SourceTensor) -> tuple[int, int]:
    rows = tensor.logical_elements // tensor.row_elements
    groups = (tensor.row_elements + GROUP_ELEMENTS - 1) // GROUP_ELEMENTS
    return rows * ((tensor.row_elements + 1) // 2), rows * groups * 2


def _bf16_rows(raw: bytes, rows: int, row_elements: int) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != rows * row_elements:
        raise ValueError("source tensor row batch became truncated")
    bits = words.astype(np.uint32) << np.uint32(16)
    values = bits.view(np.float32).reshape(rows, row_elements)
    if not np.isfinite(values).all():
        raise ValueError("Q4 source contains NaN or infinity")
    return values


def _encode_rows(values: np.ndarray) -> tuple[bytes, bytes]:
    rows, row_elements = values.shape
    groups = (row_elements + GROUP_ELEMENTS - 1) // GROUP_ELEMENTS
    padded_elements = groups * GROUP_ELEMENTS
    padded = np.zeros((rows, padded_elements), dtype=np.float32)
    padded[:, :row_elements] = values
    grouped = padded.reshape(rows, groups, GROUP_ELEMENTS)
    maxima = np.max(np.abs(grouped), axis=2).astype(np.float32, copy=False)
    scales = (maxima / np.float32(7.0)).astype(np.float32, copy=False)
    stored_scales = scales.astype("<f2")
    nonzero = maxima != np.float32(0.0)
    if (not np.isfinite(stored_scales).all() or
            np.any(stored_scales[nonzero] == np.float16(0.0))):
        raise ValueError("nonzero Q4 scale is not finite FP16")
    divisors = np.where(nonzero, scales, np.float32(1.0))
    quantized = np.rint(grouped / divisors[:, :, None])
    quantized = np.clip(quantized, -8, 7).astype(np.int8)
    quantized[~nonzero, :] = 0
    flat = quantized.reshape(rows, padded_elements)[:, :row_elements]
    if row_elements & 1:
        flat = np.pad(flat, ((0, 0), (0, 1)), constant_values=0)
    codes = flat.astype(np.int16) & 0x0F
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(order="C"), stored_scales.tobytes(order="C")


def quantize_tensor(tensor: base.SourceTensor, root: Path, ordinal: int) -> Q4Tensor:
    rows = tensor.logical_elements // tensor.row_elements
    rows_per_batch = max(1, MAX_BATCH_BYTES // (tensor.row_elements * 2))
    data_path = root / f"{ordinal:04d}.data"
    scale_path = data_path.with_suffix(".scale")
    with tensor.shard.open("rb", buffering=0) as source, \
            data_path.open("xb", buffering=0) as data_sink, \
            scale_path.open("xb", buffering=0) as scale_sink:
        source.seek(tensor.data_offset)
        remaining = rows
        while remaining:
            batch_rows = min(remaining, rows_per_batch)
            expected = batch_rows * tensor.row_elements * 2
            raw = source.read(expected)
            if len(raw) != expected:
                raise ValueError(f"source tensor became truncated: {tensor.name}")
            data, scales = _encode_rows(
                _bf16_rows(raw, batch_rows, tensor.row_elements))
            data_sink.write(data)
            scale_sink.write(scales)
            remaining -= batch_rows
        data_sink.flush(); os.fsync(data_sink.fileno())
        scale_sink.flush(); os.fsync(scale_sink.fileno())
    data_length, scale_length = q4_geometry(tensor)
    if data_path.stat().st_size != data_length or scale_path.stat().st_size != scale_length:
        raise ValueError(f"Q4 component geometry changed: {tensor.name}")
    converted = sha256()
    for path in (data_path, scale_path):
        with path.open("rb", buffering=0) as source:
            while block := source.read(base.MAX_CHUNK_BYTES):
                converted.update(block)
    return Q4Tensor(tensor, data_path, scale_path, data_length, scale_length,
                    converted.hexdigest())


def build_names(tensors: list[Q4Tensor]) -> bytes:
    result = bytearray()
    for tensor in tensors:
        encoded = tensor.name.encode("utf-8")
        if not encoded or len(encoded) > 1024:
            raise ValueError("tensor name is outside the SQW bound")
        tensor.name_offset = len(result)
        result.extend(encoded)
    return bytes(result)


def assign_payload_offsets(tensors: list[Q4Tensor], payload_offset: int) -> int:
    cursor = payload_offset
    for tensor in tensors:
        cursor = base.align_up(cursor, 64)
        tensor.data_output_offset = cursor
        cursor += tensor.data_length
        cursor = base.align_up(cursor, 64)
        tensor.scale_output_offset = cursor
        cursor += tensor.scale_length
        if cursor > MAX_ARTIFACT_BYTES:
            raise ValueError("Q4 artifact exceeds its 24 GiB bound")
    return cursor - payload_offset


def build_directory(tensors: list[Q4Tensor]) -> bytes:
    directory = bytearray(len(tensors) * base.DIRECTORY_ENTRY_BYTES)
    for index, tensor in enumerate(tensors):
        offset = index * base.DIRECTORY_ENTRY_BYTES
        encoded = tensor.name.encode("utf-8")
        base._write_u64(directory, offset, tensor.name_offset)
        base._write_u32(directory, offset + 8, len(encoded))
        base._write_u16(directory, offset + 12, tensor.role_id)
        base._write_u16(directory, offset + 14, 7)
        base._write_u16(directory, offset + 16, Q4_CODEC_ID)
        base._write_u16(directory, offset + 18, len(tensor.shape))
        for dimension_index, dimension in enumerate(tensor.shape):
            base._write_u64(directory, offset + 24 + dimension_index * 8,
                            dimension)
        base._write_u64(directory, offset + 88, tensor.logical_elements)
        base._write_u64(directory, offset + 96, tensor.data_output_offset)
        base._write_u64(directory, offset + 104, tensor.data_length)
        base._write_u64(directory, offset + 112, tensor.scale_output_offset)
        base._write_u64(directory, offset + 120, tensor.scale_length)
        base._write_u64(directory, offset + 160, tensor.row_elements)
        base._write_u32(directory, offset + 168, GROUP_ELEMENTS)
        base._write_u32(directory, offset + 172, 64)
        directory[offset + 176:offset + 208] = bytes.fromhex(tensor.source_sha256)
        directory[offset + 208:offset + 240] = bytes.fromhex(tensor.converted_sha256)
    return bytes(directory)


def tensor_manifest(tensor: Q4Tensor) -> dict[str, object]:
    empty = {"offset": "0", "length": "0"}
    return {
        "name": tensor.name, "semantic_role": tensor.role_name,
        "source_dtype": "BF16", "runtime_codec": "Q4_SYM_G64",
        "rank": len(tensor.shape), "shape": [str(v) for v in tensor.shape],
        "logical_elements": str(tensor.logical_elements),
        "data": {"offset": str(tensor.data_output_offset),
                 "length": str(tensor.data_length)},
        "scale": {"offset": str(tensor.scale_output_offset),
                  "length": str(tensor.scale_length)},
        "zero": dict(empty), "metadata": dict(empty),
        "row_elements": str(tensor.row_elements),
        "group_elements": GROUP_ELEMENTS, "required_alignment": 64,
        "source_sha256": tensor.source_sha256,
        "converted_sha256": tensor.converted_sha256,
    }


def manifest(tensors: list[Q4Tensor], model_lock: str, source_lock: str,
             policy: str, directory_digest: str) -> dict[str, object]:
    return {
        "schema": "seen-qwen-sqw-manifest-v1", "format_version": "1.0",
        "model_lock_sha256": model_lock, "source_lock_sha256": source_lock,
        "conversion_policy_sha256": policy,
        "tensor_contract": base.MODEL_CONTRACT,
        "catalog_sha256": base.CATALOG_SHA256,
        "directory_sha256": directory_digest,
        "payload_order": "canonical_utf8_tensor_name",
        "compatibility": {"required_features": [], "reader_major": 1,
                          "reader_minor": 0},
        "tensors": [tensor_manifest(tensor) for tensor in tensors],
    }


def conversion_evidence(tensors: list[Q4Tensor], model_lock: str,
                        source_lock: str, policy: str) -> bytes:
    entries = [{
        "ordinal": index, "name": tensor.name,
        "source_bytes": str(tensor.source.data_length),
        "converted_bytes": str(tensor.data_length + tensor.scale_length),
        "source_sha256": tensor.source_sha256,
        "converted_sha256": tensor.converted_sha256,
    } for index, tensor in enumerate(tensors)]
    return base.canonical_json({
        "schema": "seen-qwen-conversion-journal-v1", "version": "1",
        "output_format": "SQW1", "model_revision": base.MODEL_REVISION,
        "model_lock_sha256": model_lock, "source_lock_sha256": source_lock,
        "conversion_policy_sha256": policy,
        "catalog_sha256": base.CATALOG_SHA256,
        "plan_sha256": sha256((POLICY_ID + policy + "\nworkers=1\n").encode()).hexdigest(),
        "toolchain_compatibility_sha256": COMPATIBILITY_SHA256,
        "tensor_order": "canonical_included_catalog_prefix",
        "completed_count": len(entries), "entries": entries,
    })


def compute_layout(tensors: list[Q4Tensor], model_lock: str,
                   source_lock: str, policy: str, evidence: bytes) -> base.Layout:
    names = build_names(tensors)
    payload_offset = 4096
    stable = None
    for _ in range(32):
        payload_length = assign_payload_offsets(tensors, payload_offset)
        directory = build_directory(tensors)
        encoded_manifest = base.canonical_json(manifest(
            tensors, model_lock, source_lock, policy,
            sha256(directory).hexdigest()))
        directory_offset = base.align_up(base.HEADER_BYTES + len(encoded_manifest), 64)
        names_offset = base.align_up(directory_offset + len(directory), 64)
        next_payload = base.align_up(names_offset + len(names), 4096)
        state = (next_payload, len(encoded_manifest), directory_offset, names_offset)
        if state == stable:
            break
        stable = state; payload_offset = next_payload
    else:
        raise ValueError("Q4 SQW layout did not converge")
    payload_length = assign_payload_offsets(tensors, payload_offset)
    directory = build_directory(tensors)
    encoded_manifest = base.canonical_json(manifest(
        tensors, model_lock, source_lock, policy, sha256(directory).hexdigest()))
    directory_offset = base.align_up(base.HEADER_BYTES + len(encoded_manifest), 64)
    names_offset = base.align_up(directory_offset + len(directory), 64)
    evidence_offset = base.align_up(payload_offset + payload_length, 64)
    footer_offset = base.align_up(evidence_offset + len(evidence), 64)
    footer_length = base.FOOTER_HEADER_BYTES + 5 * base.FOOTER_ENTRY_BYTES
    file_bytes = footer_offset + footer_length
    sections = {
        "manifest": (base.HEADER_BYTES, len(encoded_manifest)),
        "directory": (directory_offset, len(directory)),
        "names": (names_offset, len(names)),
        "payload": (payload_offset, payload_length),
        "evidence": (evidence_offset, len(evidence)),
    }
    return base.Layout(encoded_manifest, directory, names, evidence, sections,
                       footer_offset, footer_length, footer_offset + 16,
                       file_bytes)


def _copy_file(source_path: Path, sink, digest) -> None:
    with source_path.open("rb", buffering=0) as source:
        while block := source.read(base.MAX_CHUNK_BYTES):
            sink.write(block); digest.update(block)


def write_sqw(path: Path, tensors: list[Q4Tensor], layout: base.Layout,
              model_lock: str, policy: str) -> tuple[str, str]:
    header = bytearray(base.HEADER_BYTES)
    header[:4] = b"SQW1"; base._write_u32(header, 4, 0x01020304)
    base._write_u16(header, 8, 1); base._write_u16(header, 10, 0)
    base._write_u32(header, 12, base.HEADER_BYTES); base._write_u32(header, 16, 1)
    base._write_u64(header, 24, base.HEADER_BYTES)
    base._write_u64(header, 32, len(layout.manifest))
    base._write_u64(header, 40, layout.sections["directory"][0])
    base._write_u32(header, 48, base.DIRECTORY_ENTRY_BYTES)
    base._write_u32(header, 52, len(tensors))
    base._write_u64(header, 56, layout.sections["names"][0])
    base._write_u64(header, 64, len(layout.names))
    base._write_u64(header, 72, layout.sections["payload"][0])
    base._write_u64(header, 80, layout.sections["payload"][1])
    base._write_u64(header, 88, layout.sections["evidence"][0])
    base._write_u64(header, 96, len(layout.evidence))
    base._write_u64(header, 104, layout.footer_offset)
    base._write_u64(header, 112, layout.footer_length)
    base._write_u64(header, 120, layout.whole_digest_offset)
    header[128:160] = bytes.fromhex(model_lock)
    header[160:192] = bytes.fromhex(policy)
    footer = bytearray(layout.footer_length)
    footer[:4] = b"SQWF"; base._write_u16(footer, 4, 1)
    base._write_u16(footer, 6, base.FOOTER_HEADER_BYTES)
    base._write_u16(footer, 8, base.FOOTER_ENTRY_BYTES)
    base._write_u16(footer, 10, 1); base._write_u32(footer, 12, 5)
    hashes = {"manifest": sha256(layout.manifest).digest(),
              "directory": sha256(layout.directory).digest(),
              "names": sha256(layout.names).digest(),
              "evidence": sha256(layout.evidence).digest()}
    payload_hash = sha256()
    with path.open("w+b", buffering=0) as sink:
        sink.truncate(layout.file_bytes); sink.seek(0); sink.write(header)
        for name, data in (("manifest", layout.manifest),
                           ("directory", layout.directory),
                           ("names", layout.names)):
            sink.seek(layout.sections[name][0]); sink.write(data)
        cursor = layout.sections["payload"][0]; sink.seek(cursor)
        for tensor in tensors:
            base.write_zeros(sink, tensor.data_output_offset - cursor,
                             payload_hash, base.MAX_CHUNK_BYTES)
            _copy_file(tensor.data_path, sink, payload_hash)
            cursor = tensor.data_output_offset + tensor.data_length
            base.write_zeros(sink, tensor.scale_output_offset - cursor,
                             payload_hash, base.MAX_CHUNK_BYTES)
            _copy_file(tensor.scale_path, sink, payload_hash)
            cursor = tensor.scale_output_offset + tensor.scale_length
        payload_start, payload_length = layout.sections["payload"]
        base.write_zeros(sink, payload_start + payload_length - cursor,
                         payload_hash, base.MAX_CHUNK_BYTES)
        hashes["payload"] = payload_hash.digest()
        sink.seek(layout.sections["evidence"][0]); sink.write(layout.evidence)
        for index, name in enumerate(("manifest", "directory", "names",
                                      "payload", "evidence")):
            entry = base.FOOTER_HEADER_BYTES + index * base.FOOTER_ENTRY_BYTES
            offset, length = layout.sections[name]
            base._write_u32(footer, entry, base.SECTION_IDS[name])
            base._write_u64(footer, entry + 8, offset)
            base._write_u64(footer, entry + 16, length)
            footer[entry + 24:entry + 56] = hashes[name]
        sink.seek(layout.footer_offset); sink.write(footer)
        sink.flush(); os.fsync(sink.fileno())
    _, whole = base.digest_file(path, base.MAX_CHUNK_BYTES)
    with path.open("r+b", buffering=0) as sink:
        sink.seek(layout.whole_digest_offset); sink.write(bytes.fromhex(whole))
        sink.flush(); os.fsync(sink.fileno())
    size, conventional = base.digest_file(path, base.MAX_CHUNK_BYTES)
    if size != layout.file_bytes:
        raise ValueError("Q4 SQW extent changed")
    return whole, conventional


def validate_sqw(path: Path, tensors: list[Q4Tensor], layout: base.Layout,
                 expected_model_lock: str, expected_policy: str,
                 expected_whole: str, expected_file: str) -> None:
    """Independently read back every sealed Q4 section before promotion."""
    if path.stat().st_size != layout.file_bytes:
        raise ValueError("read-back Q4 SQW extent changed")
    with path.open("rb", buffering=0) as source:
        header = source.read(base.HEADER_BYTES)
        if header[:4] != b"SQW1" or len(header) != base.HEADER_BYTES:
            raise ValueError("read-back Q4 SQW header is invalid")
        expected_header = {
            4: ("<I", 0x01020304), 8: ("<H", 1), 10: ("<H", 0),
            12: ("<I", base.HEADER_BYTES), 16: ("<I", 1),
            24: ("<Q", base.HEADER_BYTES),
            32: ("<Q", len(layout.manifest)),
            40: ("<Q", layout.sections["directory"][0]),
            48: ("<I", base.DIRECTORY_ENTRY_BYTES),
            52: ("<I", len(tensors)),
            56: ("<Q", layout.sections["names"][0]),
            64: ("<Q", len(layout.names)),
            72: ("<Q", layout.sections["payload"][0]),
            80: ("<Q", layout.sections["payload"][1]),
            88: ("<Q", layout.sections["evidence"][0]),
            96: ("<Q", len(layout.evidence)),
            104: ("<Q", layout.footer_offset),
            112: ("<Q", layout.footer_length),
            120: ("<Q", layout.whole_digest_offset),
        }
        for offset, (format_string, expected) in expected_header.items():
            if struct.unpack_from(format_string, header, offset)[0] != expected:
                raise ValueError(f"read-back Q4 header field changed at {offset}")
        if (header[128:160].hex() != expected_model_lock or
                header[160:192].hex() != expected_policy or
                any(header[20:24]) or any(header[192:])):
            raise ValueError("read-back Q4 lock or reserved header changed")

        source.seek(layout.footer_offset)
        footer = source.read(layout.footer_length)
        if (len(footer) != layout.footer_length or footer[:4] != b"SQWF" or
                struct.unpack_from("<HHHHI", footer, 4) !=
                (1, base.FOOTER_HEADER_BYTES, base.FOOTER_ENTRY_BYTES, 1, 5) or
                footer[16:48].hex() != expected_whole or any(footer[48:64])):
            raise ValueError("read-back Q4 footer changed")
        section_digests: dict[str, bytes] = {}
        for ordinal, name in enumerate(("manifest", "directory", "names",
                                        "payload", "evidence")):
            entry = base.FOOTER_HEADER_BYTES + ordinal * base.FOOTER_ENTRY_BYTES
            identifier = struct.unpack_from("<I", footer, entry)[0]
            offset = struct.unpack_from("<Q", footer, entry + 8)[0]
            length = struct.unpack_from("<Q", footer, entry + 16)[0]
            if (identifier != base.SECTION_IDS[name] or
                    (offset, length) != layout.sections[name] or
                    any(footer[entry + 4:entry + 8]) or
                    any(footer[entry + 56:entry + 64])):
                raise ValueError(f"read-back Q4 {name} footer entry changed")
            section_digests[name] = footer[entry + 24:entry + 56]

        source.seek(layout.sections["directory"][0])
        directory = source.read(len(layout.directory))
        source.seek(layout.sections["names"][0])
        names = source.read(len(layout.names))
        if directory != layout.directory or names != layout.names:
            raise ValueError("read-back Q4 directory or names changed")
        for ordinal, tensor in enumerate(tensors):
            entry = ordinal * base.DIRECTORY_ENTRY_BYTES
            name_offset = struct.unpack_from("<Q", directory, entry)[0]
            name_length = struct.unpack_from("<I", directory, entry + 8)[0]
            actual_name = names[name_offset:name_offset + name_length].decode(
                "utf-8", errors="strict")
            actual = (
                struct.unpack_from("<H", directory, entry + 14)[0],
                struct.unpack_from("<H", directory, entry + 16)[0],
                struct.unpack_from("<Q", directory, entry + 88)[0],
                struct.unpack_from("<Q", directory, entry + 96)[0],
                struct.unpack_from("<Q", directory, entry + 104)[0],
                struct.unpack_from("<Q", directory, entry + 112)[0],
                struct.unpack_from("<Q", directory, entry + 120)[0],
                struct.unpack_from("<Q", directory, entry + 160)[0],
                struct.unpack_from("<I", directory, entry + 168)[0],
            )
            expected = (7, Q4_CODEC_ID, tensor.logical_elements,
                        tensor.data_output_offset, tensor.data_length,
                        tensor.scale_output_offset, tensor.scale_length,
                        tensor.row_elements, GROUP_ELEMENTS)
            if (actual_name != tensor.name or actual != expected or
                    directory[entry + 176:entry + 208].hex() != tensor.source_sha256 or
                    directory[entry + 208:entry + 240].hex() != tensor.converted_sha256):
                raise ValueError(f"read-back Q4 tensor entry changed: {tensor.name}")

    final_digest = sha256()
    logical_digest = sha256()
    section_hashers = {name: sha256() for name in layout.sections}
    cursor = 0
    with path.open("rb", buffering=0) as source:
        while block := source.read(base.MAX_CHUNK_BYTES):
            final_digest.update(block)
            logical = bytearray(block)
            zero_start = max(layout.whole_digest_offset, cursor)
            zero_end = min(layout.whole_digest_offset + base.WHOLE_DIGEST_BYTES,
                           cursor + len(block))
            if zero_start < zero_end:
                logical[zero_start - cursor:zero_end - cursor] = \
                    b"\0" * (zero_end - zero_start)
            logical_digest.update(logical)
            block_end = cursor + len(block)
            for name, (section_start, section_length) in layout.sections.items():
                overlap_start = max(cursor, section_start)
                overlap_end = min(block_end, section_start + section_length)
                if overlap_start < overlap_end:
                    section_hashers[name].update(
                        block[overlap_start - cursor:overlap_end - cursor])
            cursor = block_end
    if (cursor != layout.file_bytes or
            final_digest.hexdigest() != expected_file or
            logical_digest.hexdigest() != expected_whole):
        raise ValueError("read-back Q4 whole-file digest changed")
    for name, digest in section_hashers.items():
        if digest.digest() != section_digests[name]:
            raise ValueError(f"read-back Q4 {name} digest changed")


def source_lock(source_manifest_sha: str, policy_sha: str,
                assets: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema": "seen-qwen-source-lock-v1", "model_id": base.MODEL_ID,
        "model_revision": base.MODEL_REVISION,
        "model_input_manifest_sha256": source_manifest_sha,
        "tensor_index_sha256": base.INDEX_SHA256,
        "catalog_sha256": base.CATALOG_SHA256,
        "quantization_policy_sha256": policy_sha,
        "seen": {"version": SEEN_VERSION, "commit": SEEN_COMMIT,
                 "compiler_sha256": COMPILER_SHA256,
                 "compatibility_sha256": COMPATIBILITY_SHA256,
                 "cpu_baseline": "x86-64"},
        "assets": assets,
    }


def engine_document(model_lock: str, source_lock_sha: str, policy: str,
                    weights_sha: str, whole_sha: str, weights_bytes: int,
                    component_bytes: int,
                    assets: list[dict[str, str]]) -> dict[str, object]:
    identity = sha256(("seen-qwen-engine-artifact-v1\n" +
        f"model_lock_sha256={model_lock}\nsource_lock_sha256={source_lock_sha}\n" +
        f"weights_sha256={weights_sha}\npolicy_sha256={policy}\n" +
        f"compiler_sha256={COMPILER_SHA256}\n").encode()).hexdigest()
    return {
        "schema": "seen-qwen-engine-artifact-v1", "version": 1,
        "maturity": "experimental-hardware", "engine_id": identity,
        "model_lock_sha256": model_lock, "source_lock_sha256": source_lock_sha,
        "format": {"sqw_version": "1.0", "weights_file": "weights.sqw",
                   "weights_sha256": weights_sha,
                   "sqw_whole_sha256": whole_sha,
                   "tensor_count": base.INCLUDED_TENSORS,
                   "weights_bytes": str(weights_bytes)},
        "model": {"id": base.MODEL_ID, "revision": base.MODEL_REVISION,
                  "contract": base.MODEL_CONTRACT, "layers": 64,
                  "vocab_size": 248320, "native_context": 262144,
                  "text_tensors": base.TEXT_TENSORS,
                  "mtp_tensors": base.MTP_TENSORS, "vision_tensors": 0},
        "quantization": {"policy_id": POLICY_ID, "policy_sha256": policy,
                         "source_dtype": "BF16",
                         "runtime_codec": "Q4_SYM_G64",
                         "tensor_count": base.INCLUDED_TENSORS,
                         "lossy": True, "quality_approved": False,
                         "fallback": "prohibited"},
        "tokenizer": assets,
        "intended_backend": {"kind": "cuda", "target": "sm_89",
                             "bound": False},
        "memory": {"weight_bytes": str(component_bytes),
                   "persistent_state_bytes": "156893184",
                   "kv_bytes_at_max_context": "8388608",
                   "transient_workspace_bytes": "1947205632",
                   "allocation_bytes_at_max_context": "16627529808",
                   "minimum_safety_reserve_bytes": "536870912",
                   "minimum_admitted_device_bytes": "17164400720",
                   "max_context_tokens": "128",
                   "status": "checked-experimental-q4-profile"},
        "compatible_profiles": [POLICY_ID],
        "created_by": {"seen_version": SEEN_VERSION,
                       "seen_commit": SEEN_COMMIT,
                       "compiler_sha256": COMPILER_SHA256,
                       "compatibility_sha256": COMPATIBILITY_SHA256,
                       "cpu_baseline": "x86-64"},
        "checksums_file": "checksums.sha256",
    }


def build(args: argparse.Namespace) -> Path:
    source_root = args.source_root.resolve(); asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    index, _ = base.read_index(args.index.resolve())
    policy_sha = validate_policy(args.policy.resolve())
    source_manifest = base.validate_source_manifest(source_root, base.MAX_CHUNK_BYTES)
    assets = base.validate_assets(asset_root, base.MAX_CHUNK_BYTES)
    model_lock_bytes = args.model_lock.resolve().read_bytes()
    model_lock_doc = base.strict_json_bytes(model_lock_bytes, maximum=65536)
    if not isinstance(model_lock_doc, dict) or model_lock_doc.get("revision") != base.MODEL_REVISION:
        raise ValueError("model lock identity is incompatible")
    model_lock_sha = sha256(model_lock_bytes).hexdigest()
    source_tensors, _ = base.load_source_tensors(source_root, index)
    base.hash_tensors(source_tensors, base.MAX_CHUNK_BYTES)
    source_lock_bytes = base.canonical_json(source_lock(
        sha256(source_manifest).hexdigest(), policy_sha, assets))
    source_lock_sha = sha256(source_lock_bytes).hexdigest()
    stage = Path(tempfile.mkdtemp(prefix=".qwn-046a-stage-", dir=output_root))
    components = stage / "components"; components.mkdir()
    try:
        tensors = [quantize_tensor(tensor, components, ordinal)
                   for ordinal, tensor in enumerate(source_tensors)]
        evidence = conversion_evidence(tensors, model_lock_sha,
                                       source_lock_sha, policy_sha)
        layout = compute_layout(tensors, model_lock_sha, source_lock_sha,
                                policy_sha, evidence)
        weights = stage / "weights.sqw"
        whole_sha, weights_sha = write_sqw(weights, tensors, layout,
                                           model_lock_sha, policy_sha)
        validate_sqw(weights, tensors, layout, model_lock_sha, policy_sha,
                     whole_sha, weights_sha)
        shutil.rmtree(components)
        base.write_file(stage / "conversion-evidence.json", evidence)
        base.write_file(stage / "source-lock.json", source_lock_bytes)
        base.write_file(stage / "model-lock.json", model_lock_bytes)
        for asset in assets:
            shutil.copyfile(asset_root / asset["path"], stage / asset["path"])
        component_bytes = sum(t.data_length + t.scale_length for t in tensors)
        engine = base.canonical_json(engine_document(
            model_lock_sha, source_lock_sha, policy_sha, weights_sha,
            whole_sha, layout.file_bytes, component_bytes, assets))
        base.write_file(stage / "engine.json", engine)
        names = ["README.md", "chat_template.jinja", "config.json",
                 "conversion-evidence.json", "engine.json",
                 "generation_config.json", "merges.txt", "model-lock.json",
                 "source-lock.json", "tokenizer_config.json", "vocab.json",
                 "weights.sqw"]
        checksums = "".join(
            f"{base.digest_file(stage / name, base.MAX_CHUNK_BYTES)[1]}  {name}\n"
            for name in names)
        base.write_file(stage / "checksums.sha256", checksums.encode("ascii"))
        artifact_id = json.loads(engine)["engine_id"]
        destination = output_root / artifact_id
        if destination.exists():
            if (destination / "checksums.sha256").read_bytes() != (stage / "checksums.sha256").read_bytes():
                raise FileExistsError("content-addressed destination differs")
            shutil.rmtree(stage); return destination
        os.rename(stage, destination)
        directory = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--asset-root", type=Path, required=True)
    value.add_argument("--index", type=Path, required=True)
    value.add_argument("--model-lock", type=Path, required=True)
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(build(parser().parse_args()))
