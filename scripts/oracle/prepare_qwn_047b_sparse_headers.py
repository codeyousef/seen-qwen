#!/usr/bin/env python3
"""Materialize pinned official safetensors headers as sparse local files."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile


MODEL_ID = "Qwen/Qwen3.8-27B"
CATEGORY_BYTES = {
    "text": 53_791_996_928,
    "mtp": 849_398_784,
    "vision": 921_460_192,
}


def fail(message: str) -> None:
    raise SystemExit(f"qwn-047b-header-fixture: {message}")


def category(name: str) -> str:
    if name == "lm_head.weight" or name.startswith("model.language_model."):
        return "text"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual."):
        return "vision"
    fail(f"tensor is outside the frozen catalog: {name}")


def curl_range(url: str, end: int, output: Path, headers: Path) -> None:
    subprocess.run(
        [
            "curl", "--proto", "=https", "--tlsv1.2", "--fail", "--silent",
            "--show-error", "--location", "--max-time", "120", "--retry", "2",
            "--range", f"0-{end}", "--max-filesize", "4194304", "-D",
            str(headers), "-o", str(output), url,
        ],
        check=True,
    )


def header_value(text: str, name: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(name)}:\s*(.+?)\r?$", re.IGNORECASE | re.MULTILINE)
    return [match.group(1).strip() for match in pattern.finditer(text)]


def parse_unique_json(data: bytes) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate safetensors header key: {key}")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        fail("safetensors header is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--tensor-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    index = json.loads(args.tensor_index.read_text(encoding="utf-8"))
    if lock.get("model_id") != MODEL_ID:
        fail("model lock identifies a different repository")
    revision = lock.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        fail("model lock revision is not a full commit")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 1_199:
        fail("tensor index does not contain the frozen 1,199-entry catalog")

    shard_locks = {
        entry["path"]: entry
        for entry in lock.get("files", [])
        if entry.get("purpose") == "safetensors-shard"
    }
    if len(shard_locks) != 18:
        fail("model lock does not contain exactly eighteen shards")
    if args.output.exists():
        fail(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    totals = {name: 0 for name in CATEGORY_BYTES}
    counts = {name: 0 for name in CATEGORY_BYTES}
    observed_names: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="qwn-047b-headers.", dir=args.output.parent) as temp:
        staging = Path(temp)
        for shard_name in sorted(shard_locks):
            entry = shard_locks[shard_name]
            expected_size = int(entry["bytes"])
            expected_digest = str(entry["sha256"])
            url = (
                f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/"
                f"{shard_name}?download=true"
            )
            first = staging / f"{shard_name}.first"
            first_headers = staging / f"{shard_name}.first.headers"
            curl_range(url, 7, first, first_headers)
            if first.stat().st_size != 8:
                fail(f"{shard_name}: first range was not eight bytes")
            header_length = struct.unpack("<Q", first.read_bytes())[0]
            if header_length == 0 or header_length > 4_000_000:
                fail(f"{shard_name}: header length is outside the bounded policy")

            prefix = staging / f"{shard_name}.prefix"
            prefix_headers = staging / f"{shard_name}.prefix.headers"
            curl_range(url, 7 + header_length, prefix, prefix_headers)
            if prefix.stat().st_size != 8 + header_length:
                fail(f"{shard_name}: header range length differs from its prefix")
            raw_headers = (
                first_headers.read_text(encoding="latin-1") + "\n" +
                prefix_headers.read_text(encoding="latin-1")
            )
            if revision not in header_value(raw_headers, "x-repo-commit"):
                fail(f"{shard_name}: resolver commit differs from the model lock")
            if str(expected_size) not in header_value(raw_headers, "x-linked-size"):
                fail(f"{shard_name}: resolver byte length differs from the model lock")
            linked_etags = [value.strip('"') for value in header_value(raw_headers, "x-linked-etag")]
            if expected_digest not in linked_etags:
                fail(f"{shard_name}: resolver SHA-256 differs from the model lock")
            expected_range = f"bytes 0-{7 + header_length}/{expected_size}"
            if expected_range not in header_value(raw_headers, "content-range"):
                fail(f"{shard_name}: CDN range identity is incomplete")

            prefix_bytes = prefix.read_bytes()
            header = parse_unique_json(prefix_bytes[8:])
            previous_end = 0
            for tensor_name, metadata in header.items():
                if tensor_name == "__metadata__":
                    continue
                if not isinstance(metadata, dict):
                    fail(f"{shard_name}: invalid metadata for {tensor_name}")
                offsets = metadata.get("data_offsets")
                if (not isinstance(offsets, list) or len(offsets) != 2 or
                        not all(isinstance(value, int) for value in offsets)):
                    fail(f"{shard_name}: invalid offsets for {tensor_name}")
                start, end = offsets
                if start != previous_end or end <= start:
                    fail(f"{shard_name}: non-contiguous or empty tensor {tensor_name}")
                if weight_map.get(tensor_name) != shard_name or tensor_name in observed_names:
                    fail(f"{shard_name}: unknown, duplicate, or misplaced tensor {tensor_name}")
                observed_names.add(tensor_name)
                tensor_category = category(tensor_name)
                totals[tensor_category] += end - start
                counts[tensor_category] += 1
                previous_end = end
            if 8 + header_length + previous_end != expected_size:
                fail(f"{shard_name}: header geometry differs from the locked file size")

            destination = staging / shard_name
            destination.write_bytes(prefix_bytes)
            with destination.open("r+b") as handle:
                handle.truncate(expected_size)
            print(
                f"{shard_name} bytes={expected_size} header={header_length} "
                f"header_sha256={hashlib.sha256(prefix_bytes).hexdigest()}"
            )

        if observed_names != set(weight_map):
            fail("official headers do not contain the complete frozen tensor index")
        if counts != {"text": 851, "mtp": 15, "vision": 333}:
            fail(f"official header category counts differ: {counts}")
        if totals != CATEGORY_BYTES:
            fail(f"official header category bytes differ: {totals}")
        staging.rename(args.output)

    allocated = sum(path.stat().st_blocks * 512 for path in args.output.iterdir())
    print(f"PASS: official sparse headers counts={counts} bytes={totals} allocated={allocated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
