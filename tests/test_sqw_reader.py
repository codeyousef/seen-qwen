#!/usr/bin/env python3
"""Independent deterministic SQW v1 fixture and source oracle for QWN-030B.

The binary fixtures produced here are deliberately disposable.  They live only
under the repository's ignored ``.seen`` tree and are rebuilt before the Seen
reader regression consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
READER_SOURCE = ROOT / "src/formats/sqw_reader.seen"
SEEN_READER_TEST = ROOT / "tests/qwn_030b_sqw_reader_test.seen"
OUTPUT = ROOT / ".seen/ci/output/qwn_030b"

HEADER_BYTES = 256
DIRECTORY_ENTRY_BYTES = 256
FOOTER_HEADER_BYTES = 64
FOOTER_ENTRY_BYTES = 64
WHOLE_DIGEST_BYTES = 32
SECTION_IDS = {
    "manifest": 1,
    "directory": 2,
    "names": 3,
    "payload": 4,
    "evidence": 5,
}
BASE_SECTION_NAMES = ("manifest", "directory", "names", "payload")
EVIDENCE_BYTES = (
    b'{"records":[{"gate":"QWN-030B","result":"verified"}],'
    b'"schema":"seen-qwen-sqw-evidence-v1"}'
)

MODEL_LOCK = sha256(b"qwn-030b-model-lock-v1").digest()
SOURCE_LOCK = sha256(b"qwn-030b-source-lock-v1").digest()
CONVERSION_POLICY = sha256(b"qwn-030b-conversion-policy-v1").digest()
CATALOG_DIGEST = sha256(b"qwn-030b-catalog-v1").digest()


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a non-zero power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class DuplicateJsonKey(ValueError):
    pass


def strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result


def parse_strict_json(data: bytes) -> object:
    text = data.decode("utf-8", errors="strict")
    return json.loads(
        text,
        object_pairs_hook=strict_json_object,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


@dataclass(frozen=True)
class TensorSpec:
    name: str
    role_id: int
    role_name: str
    dtype_id: int
    dtype_name: str
    codec_id: int
    codec_name: str
    shape: tuple[int, ...]
    row_elements: int
    group_elements: int
    alignment: int
    source_bytes: bytes
    data: bytes
    scale: bytes = b""
    zero: bytes = b""
    metadata: bytes = b""

    @property
    def logical_elements(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    @property
    def converted_digest(self) -> bytes:
        return sha256(self.data + self.scale + self.zero + self.metadata).digest()


@dataclass(frozen=True)
class ComponentRange:
    offset: int
    length: int


@dataclass(frozen=True)
class TensorLayout:
    name_offset: int
    name_length: int
    data: ComponentRange
    scale: ComponentRange
    zero: ComponentRange
    metadata: ComponentRange


@dataclass(frozen=True)
class BuiltSqw:
    data: bytes
    manifest: dict[str, object]
    tensors: tuple[TensorSpec, ...]
    layouts: tuple[TensorLayout, ...]
    sections: dict[str, tuple[int, int]]
    whole_digest_offset: int


TENSORS = (
    TensorSpec(
        name="lm_head.weight",
        role_id=2,
        role_name="lm_head",
        dtype_id=7,
        dtype_name="BF16",
        codec_id=1,
        codec_name="BF16",
        shape=(2, 2),
        row_elements=2,
        group_elements=0,
        alignment=64,
        source_bytes=b"qwn-030b-source-lm-head",
        data=bytes.fromhex("0000803f00400040"),
    ),
    TensorSpec(
        name="model.language_model.embed_tokens.weight",
        role_id=1,
        role_name="embedding",
        dtype_id=7,
        dtype_name="BF16",
        codec_id=4,
        codec_name="Q8_SYM_G64",
        shape=(2, 64),
        row_elements=64,
        group_elements=64,
        alignment=64,
        source_bytes=b"qwn-030b-source-embedding",
        data=bytes((index * 37 + 11) & 0xFF for index in range(128)),
        scale=bytes.fromhex("003c0040"),
    ),
)

Q4_TENSORS = (
    TensorSpec(
        name="mtp.odd_q4.weight",
        role_id=14,
        role_name="mtp",
        dtype_id=7,
        dtype_name="BF16",
        codec_id=5,
        codec_name="Q4_SYM_G64",
        shape=(1, 65),
        row_elements=65,
        group_elements=64,
        alignment=64,
        source_bytes=b"qwn-030b-source-q4-odd-row",
        data=bytes((index * 13 + 5) & 0xFF for index in range(32)) + b"\x07",
        scale=bytes.fromhex("003c0040"),
    ),
)


def _write_u16(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", target, offset, value)


def _write_u32(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", target, offset, value)


def _write_u64(target: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", target, offset, value)


def _manifest_bytes(manifest: dict[str, object], mode: str) -> bytes:
    encoded = canonical_json(manifest)
    if mode == "canonical":
        return encoded
    if mode == "spaced":
        return json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(", ", ": "),
        ).encode("utf-8")
    if mode == "duplicate-schema":
        duplicate = b'"schema":"seen-qwen-sqw-manifest-v1",'
        return b"{" + duplicate + encoded[1:]
    raise ValueError(f"unsupported manifest mode: {mode}")


def _lay_out_payload(
    payload_offset: int,
    name_offsets: tuple[tuple[int, int], ...],
    tensors: tuple[TensorSpec, ...],
) -> tuple[bytes, tuple[TensorLayout, ...]]:
    payload = bytearray()
    layouts: list[TensorLayout] = []
    for tensor, (name_offset, name_length) in zip(tensors, name_offsets, strict=True):
        ranges: list[ComponentRange] = []
        for component in (tensor.data, tensor.scale, tensor.zero, tensor.metadata):
            if not component:
                ranges.append(ComponentRange(0, 0))
                continue
            absolute = align_up(payload_offset + len(payload), tensor.alignment)
            payload.extend(b"\0" * (absolute - payload_offset - len(payload)))
            payload.extend(component)
            ranges.append(ComponentRange(absolute, len(component)))
        layouts.append(
            TensorLayout(
                name_offset=name_offset,
                name_length=name_length,
                data=ranges[0],
                scale=ranges[1],
                zero=ranges[2],
                metadata=ranges[3],
            )
        )
    return bytes(payload), tuple(layouts)


def _build_directory(
    layouts: tuple[TensorLayout, ...],
    tensors: tuple[TensorSpec, ...],
) -> bytes:
    directory = bytearray(len(tensors) * DIRECTORY_ENTRY_BYTES)
    for index, (tensor, layout) in enumerate(zip(tensors, layouts, strict=True)):
        base = index * DIRECTORY_ENTRY_BYTES
        _write_u64(directory, base + 0, layout.name_offset)
        _write_u32(directory, base + 8, layout.name_length)
        _write_u16(directory, base + 12, tensor.role_id)
        _write_u16(directory, base + 14, tensor.dtype_id)
        _write_u16(directory, base + 16, tensor.codec_id)
        _write_u16(directory, base + 18, len(tensor.shape))
        for dimension_index, dimension in enumerate(tensor.shape):
            _write_u64(directory, base + 24 + dimension_index * 8, dimension)
        _write_u64(directory, base + 88, tensor.logical_elements)
        for field_offset, component in zip(
            (96, 112, 128, 144),
            (layout.data, layout.scale, layout.zero, layout.metadata),
            strict=True,
        ):
            _write_u64(directory, base + field_offset, component.offset)
            _write_u64(directory, base + field_offset + 8, component.length)
        _write_u64(directory, base + 160, tensor.row_elements)
        _write_u32(directory, base + 168, tensor.group_elements)
        _write_u32(directory, base + 172, tensor.alignment)
        directory[base + 176 : base + 208] = sha256(tensor.source_bytes).digest()
        directory[base + 208 : base + 240] = tensor.converted_digest
    return bytes(directory)


def _manifest(
    directory_digest: bytes,
    layouts: tuple[TensorLayout, ...],
    tensors: tuple[TensorSpec, ...],
) -> dict[str, object]:
    tensor_values: list[dict[str, object]] = []
    for tensor, layout in zip(tensors, layouts, strict=True):
        component_values = {}
        for name, component in (
            ("data", layout.data),
            ("scale", layout.scale),
            ("zero", layout.zero),
            ("metadata", layout.metadata),
        ):
            component_values[name] = {
                "offset": str(component.offset),
                "length": str(component.length),
            }
        tensor_values.append(
            {
                "name": tensor.name,
                "semantic_role": tensor.role_name,
                "source_dtype": tensor.dtype_name,
                "runtime_codec": tensor.codec_name,
                "rank": len(tensor.shape),
                "shape": [str(value) for value in tensor.shape],
                "logical_elements": str(tensor.logical_elements),
                **component_values,
                "row_elements": str(tensor.row_elements),
                "group_elements": tensor.group_elements,
                "required_alignment": tensor.alignment,
                "source_sha256": sha256(tensor.source_bytes).hexdigest(),
                "converted_sha256": tensor.converted_digest.hex(),
            }
        )
    return {
        "schema": "seen-qwen-sqw-manifest-v1",
        "format_version": "1.0",
        "model_lock_sha256": MODEL_LOCK.hex(),
        "source_lock_sha256": SOURCE_LOCK.hex(),
        "conversion_policy_sha256": CONVERSION_POLICY.hex(),
        "tensor_contract": "seen-qwen38-text-v1",
        "catalog_sha256": CATALOG_DIGEST.hex(),
        "directory_sha256": directory_digest.hex(),
        "payload_order": "canonical_utf8_tensor_name",
        "compatibility": {
            "required_features": [],
            "reader_major": 1,
            "reader_minor": 0,
        },
        "tensors": tensor_values,
    }


def build_sqw(
    manifest_mode: str = "canonical",
    tensors: tuple[TensorSpec, ...] = TENSORS,
    evidence: bytes | None = None,
) -> BuiltSqw:
    if not tensors:
        raise AssertionError("SQW fixture must contain at least one tensor")
    if evidence == b"":
        raise AssertionError("present SQW evidence must contain at least one byte")
    if tuple(sorted(tensors, key=lambda item: item.name.encode("utf-8"))) != tensors:
        raise AssertionError("fixture tensors must be in canonical UTF-8 name order")
    name_table = b"".join(tensor.name.encode("utf-8") for tensor in tensors)
    name_offsets: list[tuple[int, int]] = []
    cursor = 0
    for tensor in tensors:
        encoded = tensor.name.encode("utf-8")
        name_offsets.append((cursor, len(encoded)))
        cursor += len(encoded)
    name_offset_tuple = tuple(name_offsets)

    payload_offset = 4096
    stable: tuple[object, ...] | None = None
    for _ in range(16):
        payload, layouts = _lay_out_payload(
            payload_offset,
            name_offset_tuple,
            tensors,
        )
        directory = _build_directory(layouts, tensors)
        manifest = _manifest(sha256(directory).digest(), layouts, tensors)
        manifest_data = _manifest_bytes(manifest, manifest_mode)
        directory_offset = align_up(HEADER_BYTES + len(manifest_data), 64)
        names_offset = align_up(
            directory_offset + len(directory),
            64,
        )
        next_payload_offset = align_up(names_offset + len(name_table), 4096)
        state = (
            next_payload_offset,
            len(manifest_data),
            directory_offset,
            names_offset,
            sha256(directory).digest(),
        )
        if state == stable:
            break
        stable = state
        payload_offset = next_payload_offset
    else:
        raise AssertionError("SQW layout did not reach a deterministic fixed point")

    payload, layouts = _lay_out_payload(payload_offset, name_offset_tuple, tensors)
    directory = _build_directory(layouts, tensors)
    manifest = _manifest(sha256(directory).digest(), layouts, tensors)
    manifest_data = _manifest_bytes(manifest, manifest_mode)
    directory_offset = align_up(HEADER_BYTES + len(manifest_data), 64)
    names_offset = align_up(directory_offset + len(directory), 64)
    if payload_offset != align_up(names_offset + len(name_table), 4096):
        raise AssertionError("final SQW layout differs from its fixed point")

    evidence_offset = 0
    footer_minimum = payload_offset + len(payload)
    if evidence is not None:
        evidence_offset = align_up(footer_minimum, 64)
        footer_minimum = evidence_offset + len(evidence)
    footer_offset = align_up(footer_minimum, 64)
    section_count = len(BASE_SECTION_NAMES) + (1 if evidence is not None else 0)
    footer_length = FOOTER_HEADER_BYTES + section_count * FOOTER_ENTRY_BYTES
    whole_digest_offset = footer_offset + 16
    file_bytes = bytearray(footer_offset + footer_length)

    header = memoryview(file_bytes)[:HEADER_BYTES]
    header[0:4] = b"SQW1"
    struct.pack_into("<I", header, 4, 0x01020304)
    struct.pack_into("<H", header, 8, 1)
    struct.pack_into("<H", header, 10, 0)
    struct.pack_into("<I", header, 12, HEADER_BYTES)
    struct.pack_into("<I", header, 16, 1)
    struct.pack_into("<Q", header, 24, HEADER_BYTES)
    struct.pack_into("<Q", header, 32, len(manifest_data))
    struct.pack_into("<Q", header, 40, directory_offset)
    struct.pack_into("<I", header, 48, DIRECTORY_ENTRY_BYTES)
    struct.pack_into("<I", header, 52, len(tensors))
    struct.pack_into("<Q", header, 56, names_offset)
    struct.pack_into("<Q", header, 64, len(name_table))
    struct.pack_into("<Q", header, 72, payload_offset)
    struct.pack_into("<Q", header, 80, len(payload))
    struct.pack_into("<Q", header, 88, evidence_offset)
    struct.pack_into("<Q", header, 96, 0 if evidence is None else len(evidence))
    struct.pack_into("<Q", header, 104, footer_offset)
    struct.pack_into("<Q", header, 112, footer_length)
    struct.pack_into("<Q", header, 120, whole_digest_offset)
    header[128:160] = MODEL_LOCK
    header[160:192] = CONVERSION_POLICY
    header.release()

    sections = {
        "manifest": (HEADER_BYTES, len(manifest_data)),
        "directory": (directory_offset, len(directory)),
        "names": (names_offset, len(name_table)),
        "payload": (payload_offset, len(payload)),
    }
    section_data = {
        "manifest": manifest_data,
        "directory": directory,
        "names": name_table,
        "payload": payload,
    }
    if evidence is not None:
        sections["evidence"] = (evidence_offset, len(evidence))
        section_data["evidence"] = evidence
    for name, (offset, length) in sections.items():
        data = section_data[name]
        if len(data) != length:
            raise AssertionError("section length changed during assembly")
        file_bytes[offset : offset + length] = data

    footer = footer_offset
    file_bytes[footer : footer + 4] = b"SQWF"
    _write_u16(file_bytes, footer + 4, 1)
    _write_u16(file_bytes, footer + 6, FOOTER_HEADER_BYTES)
    _write_u16(file_bytes, footer + 8, FOOTER_ENTRY_BYTES)
    _write_u16(file_bytes, footer + 10, 1)
    _write_u32(file_bytes, footer + 12, len(sections))
    for index, name in enumerate(sections):
        section_id = SECTION_IDS[name]
        entry = footer + FOOTER_HEADER_BYTES + index * FOOTER_ENTRY_BYTES
        offset, length = sections[name]
        _write_u32(file_bytes, entry, section_id)
        _write_u64(file_bytes, entry + 8, offset)
        _write_u64(file_bytes, entry + 16, length)
        file_bytes[entry + 24 : entry + 56] = sha256(
            file_bytes[offset : offset + length]
        ).digest()

    whole_digest = sha256(file_bytes).digest()
    file_bytes[whole_digest_offset : whole_digest_offset + WHOLE_DIGEST_BYTES] = (
        whole_digest
    )
    return BuiltSqw(
        data=bytes(file_bytes),
        manifest=manifest,
        tensors=tensors,
        layouts=layouts,
        sections=sections,
        whole_digest_offset=whole_digest_offset,
    )


def _header_ranges(data: bytes | bytearray) -> dict[str, tuple[int, int]]:
    tensor_count = struct.unpack_from("<I", data, 52)[0]
    ranges = {
        "manifest": struct.unpack_from("<QQ", data, 24),
        "directory": (
            struct.unpack_from("<Q", data, 40)[0],
            tensor_count * DIRECTORY_ENTRY_BYTES,
        ),
        "names": struct.unpack_from("<QQ", data, 56),
        "payload": struct.unpack_from("<QQ", data, 72),
    }
    evidence = struct.unpack_from("<QQ", data, 88)
    if evidence[1] > 0:
        ranges["evidence"] = evidence
    return ranges


def _footer_entry_offset(data: bytes | bytearray, section_id: int) -> int:
    footer = struct.unpack_from("<Q", data, 104)[0]
    count = struct.unpack_from("<I", data, footer + 12)[0]
    for index in range(count):
        entry = footer + FOOTER_HEADER_BYTES + index * FOOTER_ENTRY_BYTES
        if struct.unpack_from("<I", data, entry)[0] == section_id:
            return entry
    raise AssertionError(f"missing footer section {section_id}")


def _resign_whole(data: bytearray) -> None:
    offset = struct.unpack_from("<Q", data, 120)[0]
    data[offset : offset + WHOLE_DIGEST_BYTES] = b"\0" * WHOLE_DIGEST_BYTES
    data[offset : offset + WHOLE_DIGEST_BYTES] = sha256(data).digest()


def _resign_section(data: bytearray, name: str) -> None:
    offset, length = _header_ranges(data)[name]
    entry = _footer_entry_offset(data, SECTION_IDS[name])
    data[entry + 24 : entry + 56] = sha256(data[offset : offset + length]).digest()


def _set_manifest_directory_digest(data: bytearray, digest: bytes) -> None:
    offset, length = _header_ranges(data)["manifest"]
    manifest_data = bytes(data[offset : offset + length])
    manifest = parse_strict_json(manifest_data)
    if not isinstance(manifest, dict):
        raise AssertionError("manifest is not an object")
    old = str(manifest["directory_sha256"]).encode("ascii")
    replacement = digest.hex().encode("ascii")
    if len(old) != len(replacement) or manifest_data.count(old) != 1:
        raise AssertionError("directory digest is not uniquely replaceable")
    data[offset : offset + length] = manifest_data.replace(old, replacement)


def _section_digest_valid(data: bytes | bytearray, name: str) -> bool:
    try:
        offset, length = _header_ranges(data)[name]
        entry = _footer_entry_offset(data, SECTION_IDS[name])
        if struct.unpack_from("<Q", data, entry + 8)[0] != offset:
            return False
        if struct.unpack_from("<Q", data, entry + 16)[0] != length:
            return False
        return bytes(data[entry + 24 : entry + 56]) == sha256(
            data[offset : offset + length]
        ).digest()
    except (IndexError, struct.error, AssertionError):
        return False


def _all_section_digests_valid(data: bytes | bytearray) -> bool:
    try:
        return all(_section_digest_valid(data, name) for name in _header_ranges(data))
    except (IndexError, struct.error):
        return False


def _whole_digest_valid(data: bytes | bytearray) -> bool:
    try:
        offset = struct.unpack_from("<Q", data, 120)[0]
        stored = bytes(data[offset : offset + WHOLE_DIGEST_BYTES])
        if len(stored) != WHOLE_DIGEST_BYTES:
            return False
        copy = bytearray(data)
        copy[offset : offset + WHOLE_DIGEST_BYTES] = b"\0" * WHOLE_DIGEST_BYTES
        return stored == sha256(copy).digest()
    except (IndexError, struct.error):
        return False


def build_corruptions(valid: BuiltSqw) -> dict[str, bytes]:
    corruptions: dict[str, bytes] = {}

    wrong_magic = bytearray(valid.data)
    wrong_magic[0] ^= 0xFF
    _resign_whole(wrong_magic)
    corruptions["wrong_magic.sqw"] = bytes(wrong_magic)

    wrong_whole = bytearray(valid.data)
    wrong_whole[valid.whole_digest_offset] ^= 0x01
    corruptions["wrong_whole_digest.sqw"] = bytes(wrong_whole)

    wrong_section = bytearray(valid.data)
    payload_entry = _footer_entry_offset(wrong_section, SECTION_IDS["payload"])
    wrong_section[payload_entry + 24] ^= 0x01
    _resign_whole(wrong_section)
    corruptions["wrong_section_digest.sqw"] = bytes(wrong_section)

    wrong_tensor = bytearray(valid.data)
    wrong_tensor[valid.layouts[0].data.offset] ^= 0x01
    _resign_section(wrong_tensor, "payload")
    _resign_whole(wrong_tensor)
    corruptions["wrong_tensor_digest.sqw"] = bytes(wrong_tensor)

    nonzero_padding = bytearray(valid.data)
    names_offset, names_length = valid.sections["names"]
    padding_offset = names_offset + names_length
    if padding_offset >= valid.sections["payload"][0]:
        raise AssertionError("fixture has no name-to-payload padding")
    nonzero_padding[padding_offset] = 1
    _resign_whole(nonzero_padding)
    corruptions["nonzero_padding.sqw"] = bytes(nonzero_padding)

    unsupported_codec = bytearray(valid.data)
    directory_offset = valid.sections["directory"][0]
    _write_u16(unsupported_codec, directory_offset + 16, 0)
    directory_digest = sha256(
        unsupported_codec[
            directory_offset : directory_offset + valid.sections["directory"][1]
        ]
    ).digest()
    _set_manifest_directory_digest(unsupported_codec, directory_digest)
    _resign_section(unsupported_codec, "manifest")
    _resign_section(unsupported_codec, "directory")
    _resign_whole(unsupported_codec)
    corruptions["unsupported_codec.sqw"] = bytes(unsupported_codec)

    bad_name = bytearray(valid.data)
    bad_name[valid.sections["names"][0]] = 0xFF
    _resign_section(bad_name, "names")
    _resign_whole(bad_name)
    corruptions["invalid_utf8_name.sqw"] = bytes(bad_name)

    corruptions["noncanonical_manifest.sqw"] = build_sqw("spaced").data
    corruptions["duplicate_manifest_key.sqw"] = build_sqw(
        "duplicate-schema"
    ).data
    corruptions["truncated.sqw"] = valid.data[:-1]
    return corruptions


def build_q4_tail_fixtures() -> tuple[BuiltSqw, bytes]:
    valid = build_sqw(tensors=Q4_TENSORS)
    tail_offset = valid.layouts[0].data.offset + valid.layouts[0].data.length - 1
    if valid.data[tail_offset] & 0xF0:
        raise AssertionError("valid Q4 odd-row tail must have a zero high nibble")
    corrupt = bytearray(valid.data)
    corrupt[tail_offset] |= 0xF0
    _resign_section(corrupt, "payload")
    _resign_whole(corrupt)
    return valid, bytes(corrupt)


def build_evidence_fixtures() -> tuple[BuiltSqw, bytes]:
    valid = build_sqw(evidence=EVIDENCE_BYTES)
    evidence_offset, evidence_length = valid.sections["evidence"]
    corruption_offset = evidence_offset + evidence_length // 2
    corrupt = bytearray(valid.data)
    corrupt[corruption_offset] ^= 0x01
    _resign_whole(corrupt)
    return valid, bytes(corrupt)


def _balanced_block(source: str, declaration: str) -> str:
    match = re.search(declaration, source)
    if match is None:
        raise AssertionError(f"missing declaration matching {declaration!r}")
    opening = source.find("{", match.end())
    if opening < 0:
        raise AssertionError("declaration has no body")
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError("declaration body is not balanced")


class SqwReaderOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = build_sqw()
        cls.corruptions = build_corruptions(cls.valid)
        cls.q4_valid, cls.q4_nonzero_tail = build_q4_tail_fixtures()
        cls.evidence_valid, cls.evidence_corrupt = build_evidence_fixtures()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "valid.sqw").write_bytes(cls.valid.data)
        (OUTPUT / "q4_odd_row_valid.sqw").write_bytes(cls.q4_valid.data)
        (OUTPUT / "q4_odd_row_nonzero_tail.sqw").write_bytes(
            cls.q4_nonzero_tail
        )
        (OUTPUT / "evidence_valid.sqw").write_bytes(cls.evidence_valid.data)
        (OUTPUT / "evidence_corrupt.sqw").write_bytes(cls.evidence_corrupt)
        for name, data in cls.corruptions.items():
            (OUTPUT / name).write_bytes(data)
        oracle = {
            "schema": "seen-qwen-qwn-030b-fixture-oracle-v1",
            "valid_file_sha256": sha256(cls.valid.data).hexdigest(),
            "whole_file_sha256": cls.valid.data[
                cls.valid.whole_digest_offset : cls.valid.whole_digest_offset
                + WHOLE_DIGEST_BYTES
            ].hex(),
            "policy": {
                "model_lock_sha256": MODEL_LOCK.hex(),
                "source_lock_sha256": SOURCE_LOCK.hex(),
                "conversion_policy_sha256": CONVERSION_POLICY.hex(),
                "catalog_sha256": CATALOG_DIGEST.hex(),
                "tensor_count": len(cls.valid.tensors),
            },
            "sections": {
                name: {
                    "offset": offset,
                    "length": length,
                    "sha256": sha256(cls.valid.data[offset : offset + length]).hexdigest(),
                }
                for name, (offset, length) in cls.valid.sections.items()
            },
            "tensors": [
                {
                    "name": tensor.name,
                    "source_sha256": sha256(tensor.source_bytes).hexdigest(),
                    "converted_sha256": tensor.converted_digest.hex(),
                }
                for tensor in cls.valid.tensors
            ],
            "corruptions": sorted(cls.corruptions),
            "q4_odd_row": {
                "name": cls.q4_valid.tensors[0].name,
                "tensor_count": len(cls.q4_valid.tensors),
                "row_elements": cls.q4_valid.tensors[0].row_elements,
                "data_length": cls.q4_valid.layouts[0].data.length,
                "tail_offset": cls.q4_valid.layouts[0].data.offset
                + cls.q4_valid.layouts[0].data.length
                - 1,
                "valid_file_sha256": sha256(cls.q4_valid.data).hexdigest(),
                "valid_whole_file_sha256": cls.q4_valid.data[
                    cls.q4_valid.whole_digest_offset :
                    cls.q4_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
                ].hex(),
                "corrupt_file_sha256": sha256(cls.q4_nonzero_tail).hexdigest(),
                "corrupt_whole_file_sha256": cls.q4_nonzero_tail[
                    cls.q4_valid.whole_digest_offset :
                    cls.q4_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
                ].hex(),
                "expected_error": "sqw.q4_tail",
            },
            "evidence": {
                "section_length": cls.evidence_valid.sections["evidence"][1],
                "valid_file_sha256": sha256(cls.evidence_valid.data).hexdigest(),
                "valid_whole_file_sha256": cls.evidence_valid.data[
                    cls.evidence_valid.whole_digest_offset :
                    cls.evidence_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
                ].hex(),
                "corrupt_file_sha256": sha256(cls.evidence_corrupt).hexdigest(),
                "corrupt_whole_file_sha256": cls.evidence_corrupt[
                    cls.evidence_valid.whole_digest_offset :
                    cls.evidence_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
                ].hex(),
                "expected_error": "sqw.section_digest",
            },
        }
        (OUTPUT / "oracle.json").write_text(
            json.dumps(oracle, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_generator_is_deterministic_and_output_is_ignored(self) -> None:
        self.assertEqual(build_sqw().data, self.valid.data)
        self.assertEqual(build_sqw().manifest, self.valid.manifest)
        self.assertEqual(build_corruptions(build_sqw()), self.corruptions)
        self.assertEqual(build_q4_tail_fixtures(), (
            self.q4_valid,
            self.q4_nonzero_tail,
        ))
        self.assertEqual(build_evidence_fixtures(), (
            self.evidence_valid,
            self.evidence_corrupt,
        ))
        self.assertEqual(
            OUTPUT.relative_to(ROOT).parts,
            (".seen", "ci", "output", "qwn_030b"),
        )
        self.assertIn("/.seen/", (ROOT / ".gitignore").read_text(encoding="utf-8"))
        self.assertFalse(list((ROOT / "tests/fixtures").rglob("*.sqw")))
        self.assertEqual((OUTPUT / "valid.sqw").read_bytes(), self.valid.data)
        self.assertEqual(
            (OUTPUT / "evidence_valid.sqw").read_bytes(),
            self.evidence_valid.data,
        )
        self.assertEqual(
            (OUTPUT / "evidence_corrupt.sqw").read_bytes(),
            self.evidence_corrupt,
        )

    def test_optional_evidence_footer_and_corruption_are_targeted(self) -> None:
        valid = self.evidence_valid
        corrupt = self.evidence_corrupt
        self.assertEqual(tuple(valid.sections), (*BASE_SECTION_NAMES, "evidence"))
        evidence_offset, evidence_length = valid.sections["evidence"]
        self.assertEqual(
            valid.data[evidence_offset : evidence_offset + evidence_length],
            EVIDENCE_BYTES,
        )
        footer = struct.unpack_from("<Q", valid.data, 104)[0]
        self.assertEqual(struct.unpack_from("<I", valid.data, footer + 12)[0], 5)
        self.assertTrue(_all_section_digests_valid(valid.data))
        self.assertTrue(_whole_digest_valid(valid.data))

        for name in BASE_SECTION_NAMES:
            with self.subTest(section=name):
                self.assertTrue(_section_digest_valid(corrupt, name))
        self.assertFalse(_section_digest_valid(corrupt, "evidence"))
        self.assertTrue(_whole_digest_valid(corrupt))

        evidence_differences = [
            index
            for index in range(evidence_offset, evidence_offset + evidence_length)
            if valid.data[index] != corrupt[index]
        ]
        self.assertEqual(
            evidence_differences,
            [evidence_offset + evidence_length // 2],
        )
        allowed_differences = set(evidence_differences)
        allowed_differences.update(range(
            valid.whole_digest_offset,
            valid.whole_digest_offset + WHOLE_DIGEST_BYTES,
        ))
        self.assertFalse(any(
            valid.data[index] != corrupt[index] and index not in allowed_differences
            for index in range(len(valid.data))
        ))
        evidence_entry = _footer_entry_offset(valid.data, SECTION_IDS["evidence"])
        self.assertEqual(
            corrupt[evidence_entry + 24 : evidence_entry + 56],
            valid.data[evidence_entry + 24 : evidence_entry + 56],
        )

    def test_q4_odd_row_tail_is_zero_and_corruption_is_targeted(self) -> None:
        valid = self.q4_valid
        tensor = valid.tensors[0]
        layout = valid.layouts[0]
        tail_offset = layout.data.offset + layout.data.length - 1
        self.assertEqual(tensor.codec_name, "Q4_SYM_G64")
        self.assertEqual(tensor.codec_id, 5)
        self.assertEqual(tensor.shape, (1, 65))
        self.assertEqual(tensor.row_elements, 65)
        self.assertEqual(tensor.group_elements, 64)
        self.assertEqual(layout.data.length, 33)
        self.assertEqual(layout.scale.length, 4)
        self.assertEqual(valid.data[tail_offset], 0x07)
        self.assertEqual(valid.data[tail_offset] & 0xF0, 0)
        self.assertTrue(_all_section_digests_valid(valid.data))
        self.assertTrue(_whole_digest_valid(valid.data))

        corrupt = self.q4_nonzero_tail
        self.assertEqual(corrupt[tail_offset], 0xF7)
        self.assertEqual(corrupt[tail_offset] & 0x0F, valid.data[tail_offset])
        self.assertTrue(_all_section_digests_valid(corrupt))
        self.assertTrue(_whole_digest_valid(corrupt))
        directory_offset = valid.sections["directory"][0]
        self.assertEqual(
            corrupt[directory_offset + 208 : directory_offset + 240],
            valid.data[directory_offset + 208 : directory_offset + 240],
        )
        self.assertEqual(
            (OUTPUT / "q4_odd_row_valid.sqw").read_bytes(),
            valid.data,
        )
        self.assertEqual(
            (OUTPUT / "q4_odd_row_nonzero_tail.sqw").read_bytes(),
            corrupt,
        )

    def test_valid_fixture_has_canonical_sections_and_digests(self) -> None:
        data = self.valid.data
        self.assertEqual(data[0:4], b"SQW1")
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], 0x01020304)
        self.assertEqual(struct.unpack_from("<HH", data, 8), (1, 0))
        self.assertEqual(struct.unpack_from("<I", data, 12)[0], HEADER_BYTES)
        self.assertEqual(_header_ranges(data), self.valid.sections)
        self.assertEqual(self.valid.sections["directory"][0] % 64, 0)
        self.assertEqual(self.valid.sections["names"][0] % 64, 0)
        self.assertEqual(self.valid.sections["payload"][0] % 4096, 0)
        self.assertTrue(_all_section_digests_valid(data))
        self.assertTrue(_whole_digest_valid(data))
        footer = struct.unpack_from("<Q", data, 104)[0]
        footer_length = struct.unpack_from("<Q", data, 112)[0]
        self.assertEqual(footer + footer_length, len(data))
        self.assertEqual(struct.unpack_from("<Q", data, 120)[0], footer + 16)
        self.assertEqual(data[footer : footer + 4], b"SQWF")
        self.assertEqual(struct.unpack_from("<HHHHI", data, footer + 4), (1, 64, 64, 1, 4))
        self.assertEqual(data[20:24], b"\0" * 4)
        self.assertEqual(data[192:256], b"\0" * 64)

        manifest_offset, manifest_length = self.valid.sections["manifest"]
        manifest_data = data[manifest_offset : manifest_offset + manifest_length]
        parsed = parse_strict_json(manifest_data)
        self.assertEqual(parsed, self.valid.manifest)
        self.assertEqual(canonical_json(parsed), manifest_data)
        directory_offset, directory_length = self.valid.sections["directory"]
        self.assertEqual(
            parsed["directory_sha256"],
            sha256(data[directory_offset : directory_offset + directory_length]).hexdigest(),
        )

    def test_directory_names_and_tensor_bundle_digests_match_manifest(self) -> None:
        data = self.valid.data
        directory_offset = self.valid.sections["directory"][0]
        names_offset = self.valid.sections["names"][0]
        manifest_tensors = self.valid.manifest["tensors"]
        decoded_names: list[bytes] = []
        for index, (tensor, layout) in enumerate(
            zip(self.valid.tensors, self.valid.layouts, strict=True)
        ):
            entry = directory_offset + index * DIRECTORY_ENTRY_BYTES
            name_relative, name_length = struct.unpack_from("<QI", data, entry)
            name_bytes = data[
                names_offset + name_relative : names_offset + name_relative + name_length
            ]
            self.assertEqual(name_bytes.decode("utf-8", errors="strict"), tensor.name)
            decoded_names.append(name_bytes)
            self.assertEqual(struct.unpack_from("<HHHH", data, entry + 12), (
                tensor.role_id,
                tensor.dtype_id,
                tensor.codec_id,
                len(tensor.shape),
            ))
            self.assertEqual(struct.unpack_from("<I", data, entry + 20)[0], 0)
            shape = struct.unpack_from("<8Q", data, entry + 24)
            self.assertEqual(shape[: len(tensor.shape)], tensor.shape)
            self.assertEqual(shape[len(tensor.shape) :], (0,) * (8 - len(tensor.shape)))
            self.assertEqual(struct.unpack_from("<Q", data, entry + 88)[0], tensor.logical_elements)
            for field_offset, component in zip(
                (96, 112, 128, 144),
                (layout.data, layout.scale, layout.zero, layout.metadata),
                strict=True,
            ):
                self.assertEqual(
                    struct.unpack_from("<QQ", data, entry + field_offset),
                    (component.offset, component.length),
                )
            self.assertEqual(struct.unpack_from("<Q", data, entry + 160)[0], tensor.row_elements)
            self.assertEqual(struct.unpack_from("<I", data, entry + 168)[0], tensor.group_elements)
            self.assertEqual(struct.unpack_from("<I", data, entry + 172)[0], tensor.alignment)
            self.assertEqual(data[entry + 176 : entry + 208], sha256(tensor.source_bytes).digest())
            components = b"".join(
                data[component.offset : component.offset + component.length]
                for component in (layout.data, layout.scale, layout.zero, layout.metadata)
            )
            self.assertEqual(sha256(components).digest(), tensor.converted_digest)
            self.assertEqual(data[entry + 208 : entry + 240], tensor.converted_digest)
            manifest_tensor = manifest_tensors[index]
            self.assertEqual(manifest_tensor["name"], tensor.name)
            self.assertEqual(manifest_tensor["source_sha256"], sha256(tensor.source_bytes).hexdigest())
            self.assertEqual(manifest_tensor["converted_sha256"], tensor.converted_digest.hex())
            self.assertEqual(data[entry + 240 : entry + 256], b"\0" * 16)
        self.assertEqual(decoded_names, sorted(decoded_names))
        self.assertEqual(len(decoded_names), len(set(decoded_names)))

    def test_corruptions_are_independent_and_targeted(self) -> None:
        for name, data in self.corruptions.items():
            with self.subTest(name=name):
                self.assertNotEqual(data, self.valid.data)
                self.assertEqual((OUTPUT / name).read_bytes(), data)

        self.assertTrue(_whole_digest_valid(self.corruptions["wrong_magic.sqw"]))
        self.assertTrue(_all_section_digests_valid(self.corruptions["wrong_magic.sqw"]))
        self.assertFalse(_whole_digest_valid(self.corruptions["wrong_whole_digest.sqw"]))
        self.assertTrue(_whole_digest_valid(self.corruptions["wrong_section_digest.sqw"]))
        self.assertFalse(_all_section_digests_valid(self.corruptions["wrong_section_digest.sqw"]))
        self.assertTrue(_whole_digest_valid(self.corruptions["wrong_tensor_digest.sqw"]))
        self.assertTrue(_all_section_digests_valid(self.corruptions["wrong_tensor_digest.sqw"]))
        self.assertTrue(_whole_digest_valid(self.corruptions["nonzero_padding.sqw"]))
        self.assertTrue(_all_section_digests_valid(self.corruptions["nonzero_padding.sqw"]))
        self.assertTrue(_whole_digest_valid(self.corruptions["unsupported_codec.sqw"]))
        self.assertTrue(_all_section_digests_valid(self.corruptions["unsupported_codec.sqw"]))
        self.assertEqual(
            struct.unpack_from(
                "<H",
                self.corruptions["unsupported_codec.sqw"],
                self.valid.sections["directory"][0] + 16,
            )[0],
            0,
        )
        self.assertTrue(_whole_digest_valid(self.corruptions["invalid_utf8_name.sqw"]))
        self.assertTrue(_all_section_digests_valid(self.corruptions["invalid_utf8_name.sqw"]))
        with self.assertRaises(UnicodeDecodeError):
            offset, length = self.valid.sections["names"]
            self.corruptions["invalid_utf8_name.sqw"][offset : offset + length].decode("utf-8")

        noncanonical = self.corruptions["noncanonical_manifest.sqw"]
        offset, length = _header_ranges(noncanonical)["manifest"]
        parsed = parse_strict_json(noncanonical[offset : offset + length])
        self.assertNotEqual(canonical_json(parsed), noncanonical[offset : offset + length])
        self.assertTrue(_whole_digest_valid(noncanonical))
        self.assertTrue(_all_section_digests_valid(noncanonical))

        duplicate = self.corruptions["duplicate_manifest_key.sqw"]
        offset, length = _header_ranges(duplicate)["manifest"]
        with self.assertRaises(DuplicateJsonKey):
            parse_strict_json(duplicate[offset : offset + length])
        self.assertTrue(_whole_digest_valid(duplicate))
        self.assertTrue(_all_section_digests_valid(duplicate))
        self.assertEqual(len(self.corruptions["truncated.sqw"]), len(self.valid.data) - 1)

    def test_reader_source_has_mapped_indexed_strict_json_and_ownership(self) -> None:
        self.assertTrue(READER_SOURCE.is_file(), READER_SOURCE)
        source = READER_SOURCE.read_text(encoding="utf-8")
        for spelling in (
            "SqwReader",
            "SqwReaderPolicy",
            "SqwTensorView",
            "MappedFile",
            "MappedWindow",
            "parseJsonWithStrictLimits",
            "validateJsonUtf8",
            ".length()",
            ".keyAt(",
            ".valueAt(",
        ):
            self.assertIn(spelling, source)
        self.assertNotIn(".keys()", source)
        self.assertNotIn(".values()", source)
        self.assertNotRegex(source, re.compile(r"\bparseStrictJson\s*\("))
        self.assertRegex(source, re.compile(r"canonical", re.IGNORECASE))
        self.assertRegex(
            source,
            re.compile(r"@noinline\s+fun\s+sqwParseManifestJson\b"),
        )

        validate_file = _balanced_block(source, r"fun\s+sqwValidateFile\b")
        self.assertIn("sqwParseManifestJson", validate_file)
        self.assertNotIn("parseJsonWithStrictLimits", validate_file)
        parse_failure = _balanced_block(
            validate_file, r"if\s+parsed\.success\s*!=\s*1"
        )
        self.assertIn("FEL-1561", validate_file)
        self.assertIn("destroyJsonParseResult(parsed, false)", parse_failure)
        self.assertLess(
            parse_failure.index("destroyJsonParseResult(parsed, false)"),
            parse_failure.index("return Err"),
        )

        validate_directory = _balanced_block(
            source, r"fun\s+sqwValidateDirectoryAndNames\b"
        )
        bundle_error = _balanced_block(
            validate_directory, r"if\s+bundleDigest\.isErr\(\)"
        )
        self.assertLess(
            validate_directory.index("if bundleDigest.isErr()"),
            validate_directory.index(
                "if bundleDigest.unwrap() != convertedSha256"
            ),
        )
        self.assertIn("let error = bundleDigest.unwrapErr()", bundle_error)
        self.assertRegex(bundle_error, re.compile(r"return\s+Err[^\n]*\(error\)"))

        validate_footer = _balanced_block(
            source, r"fun\s+sqwValidateFooterAndHashes\b"
        )
        for owner, result_name in (
            (validate_footer, "namesResult"),
            (validate_footer, "payloadResult"),
            (validate_footer, "evidence"),
            (validate_footer, "wholeResult"),
            (validate_file, "directoryResult"),
            (validate_file, "manifestDigestResult"),
        ):
            failure = _balanced_block(
                owner, rf"if\s+{result_name}\.isErr\(\)"
            )
            self.assertIn(f"{result_name}.unwrapErr()", failure)
            self.assertNotIn("sqwReaderError", failure)

        stable_codes = set(re.findall(r'"(sqw\.[a-z0-9_]+)"', source))
        self.assertGreaterEqual(len(stable_codes), 4, sorted(stable_codes))
        self.assertNotRegex(source, re.compile(r'"sqw\."\s*\+'))

        reader = _balanced_block(source, r"(?:pub\s+)?class\s+SqwReader\b")
        view = _balanced_block(source, r"(?:pub\s+)?class\s+SqwTensorView\b")
        close = _balanced_block(reader, r"fun\s+close\s*\(")
        self.assertIn("MappedFile", reader)
        self.assertRegex(reader, re.compile(r"\bclosed\b"))
        self.assertRegex(reader, re.compile(r"\bclosing\b"))
        self.assertRegex(reader, re.compile(r"\bmetadataReleased\b"))
        self.assertIn("close()", reader)
        self.assertRegex(
            reader,
            re.compile(r"if\s+this\.closed\s*\{[^}]*\breturn\b"),
        )
        self.assertRegex(reader, re.compile(r"this\.closed\s*=\s*true"))
        self.assertIn("this.closing = true", close)
        self.assertIn("if not this.metadataReleased", close)
        self.assertIn("this.metadataReleased = true", close)
        self.assertIn("retry reader close explicitly", close)
        self.assertLess(close.index("this.closing = true"), close.index(".window.close()"))
        self.assertLess(close.index(".window.close()"), close.index("this.windows.free()"))
        self.assertLess(close.index("this.windows.free()"), close.index("sqwDestroyRecords"))
        self.assertLess(close.index("sqwDestroyRecords"), close.index("this.file.close()"))
        self.assertLess(close.index("this.file.close()"), close.index("this.closed = true"))
        self.assertNotRegex(source, re.compile(r"let\s+_\s*=\s*[^\n]*\.close\(\)"))
        self.assertIn("sqwCloseTemporaryWindow", source)
        self.assertIn("sqwCloseRejectedFile", source)
        self.assertRegex(source, re.compile(r"(?:window|windows).*\.close\(\)", re.IGNORECASE))
        self.assertRegex(source, re.compile(r"(?:file|mappedFile).*\.close\(\)", re.IGNORECASE))
        self.assertIn("destroyJsonParseResult", source)
        self.assertIn("seen_string_clone_owned", source)
        self.assertIn("seen_string_release_owned", source)
        self.assertRegex(view, re.compile(r"\*const\s+UInt8"))
        self.assertEqual(
            re.findall(r"\bvar\s+([A-Za-z][A-Za-z0-9]*):", view),
            ["data", "byteLength", "logicalElements", "codec"],
        )

    def test_public_seen_regression_consumes_generated_oracles(self) -> None:
        self.assertTrue(SEEN_READER_TEST.is_file(), SEEN_READER_TEST)
        source = SEEN_READER_TEST.read_text(encoding="utf-8")
        self.assertIn(".seen/ci/output/qwn_030b", source)
        for spelling in ("SqwReader", "SqwReaderPolicy", "SqwTensorView", "valid.sqw", "sqw."):
            self.assertIn(spelling, source)
        self.assertGreaterEqual(source.count(".close()"), 2)
        for spelling in (
            MODEL_LOCK.hex(),
            SOURCE_LOCK.hex(),
            CONVERSION_POLICY.hex(),
            CATALOG_DIGEST.hex(),
            self.valid.data[
                self.valid.whole_digest_offset :
                self.valid.whole_digest_offset + WHOLE_DIGEST_BYTES
            ].hex(),
            self.q4_valid.data[
                self.q4_valid.whole_digest_offset :
                self.q4_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
            ].hex(),
            self.q4_nonzero_tail[
                self.q4_valid.whole_digest_offset :
                self.q4_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
            ].hex(),
            self.evidence_valid.data[
                self.evidence_valid.whole_digest_offset :
                self.evidence_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
            ].hex(),
            self.evidence_corrupt[
                self.evidence_valid.whole_digest_offset :
                self.evidence_valid.whole_digest_offset + WHOLE_DIGEST_BYTES
            ].hex(),
            "q4_odd_row_valid.sqw",
            "q4_odd_row_nonzero_tail.sqw",
            "sqw.q4_tail",
            "evidence_valid.sqw",
            "evidence_corrupt.sqw",
            "sqw.section_digest",
        ):
            self.assertIn(spelling, source)


if __name__ == "__main__":
    unittest.main()
