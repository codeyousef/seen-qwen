#!/usr/bin/env python3
"""Fetch only the immutable official Qwen layer ranges required by QWN-025A.

The output is ignored project-local input for the oracle generator.  No model
repository code is downloaded or executed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
DEFAULT_OUTPUT = ROOT / ".seen/oracle-official/layers"
HEADER_WINDOW = 262_144

# LFS object identities and sizes returned by the immutable HF revision.
SHARDS = {
    1: (3_966_730_552, "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"),
    9: (2_108_759_344, "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"),
    10: (3_979_553_696, "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"),
    16: (3_979_564_040, "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"),
    17: (2_108_759_344, "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"),
}
LAYERS = {0: 1, 3: 1, 31: 9, 32: 10, 60: 16, 63: 17}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def shard_name(shard: int) -> str:
    return f"model-{shard:05d}-of-00018.safetensors"


def url(shard: int) -> str:
    return (
        f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"
        f"{shard_name(shard)}?download=true"
    )


def ranged_request(shard: int, first: int, last: int):
    request = urllib.request.Request(url(shard), headers={"Range": f"bytes={first}-{last}"})
    response = urllib.request.urlopen(request, timeout=120)
    if response.status != 206:
        response.close()
        raise ValueError(f"server did not honor byte range for shard {shard}: {response.status}")
    content_range = response.headers.get("Content-Range", "")
    expected = f"bytes {first}-{last}/{SHARDS[shard][0]}"
    if content_range != expected:
        response.close()
        raise ValueError(f"unexpected Content-Range for shard {shard}: {content_range!r}")
    return response


def read_header(shard: int) -> tuple[int, dict[str, dict[str, object]]]:
    with ranged_request(shard, 0, HEADER_WINDOW - 1) as response:
        prefix = response.read(HEADER_WINDOW)
    if len(prefix) != HEADER_WINDOW:
        raise ValueError(f"short header window for shard {shard}")
    header_length = struct.unpack_from("<Q", prefix)[0]
    if header_length <= 0 or 8 + header_length > len(prefix):
        raise ValueError(f"invalid Safetensors header length for shard {shard}")
    header = json.loads(prefix[8 : 8 + header_length].decode("utf-8").rstrip(" "))
    return header_length, header


def canonical_header(entries: list[tuple[str, dict[str, object]]], metadata: dict[str, str]) -> bytes:
    offset = 0
    document: dict[str, object] = {"__metadata__": metadata}
    for name, entry in entries:
        start, end = entry["data_offsets"]
        length = end - start
        document[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length
    raw = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    padding = (-len(raw)) % 8
    return struct.pack("<Q", len(raw) + padding) + raw + b" " * padding


def select_layer_entries(
    source_header: dict[str, dict[str, object]],
    layer: int,
    shard: int,
    index_map: dict[str, str],
) -> list[tuple[str, dict[str, object]]]:
    prefix = f"model.language_model.layers.{layer}."
    entries = sorted(
        ((name, entry) for name, entry in source_header.items() if name.startswith(prefix)),
        key=lambda item: item[1]["data_offsets"][0],
    )
    if not entries:
        raise ValueError(f"layer {layer} is absent from shard {shard}")
    actual_names = {name for name, _ in entries}
    expected_names = {
        name
        for name, filename in index_map.items()
        if name.startswith(prefix) and filename == shard_name(shard)
    }
    if actual_names != expected_names:
        raise ValueError(f"checked-in index disagrees with shard {shard} for layer {layer}")
    for previous, current in zip(entries, entries[1:]):
        if previous[1]["data_offsets"][1] != current[1]["data_offsets"][0]:
            raise ValueError(f"layer {layer} tensor payload is not contiguous")
    return entries


def fetch_layer(layer: int, shard: int, output_root: Path, index_map: dict[str, str]) -> dict[str, object]:
    header_length, source_header = read_header(shard)
    entries = select_layer_entries(source_header, layer, shard, index_map)

    payload_first = entries[0][1]["data_offsets"][0]
    payload_last = entries[-1][1]["data_offsets"][1]
    absolute_first = 8 + header_length + payload_first
    absolute_last = 8 + header_length + payload_last - 1
    expected_bytes = payload_last - payload_first
    output = output_root / f"layer-{layer:02d}.safetensors"
    output.parent.mkdir(parents=True, exist_ok=True)
    header = canonical_header(
        entries,
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_shard": shard_name(shard),
            "source_shard_lfs_sha256": SHARDS[shard][1],
            "source_absolute_range": f"{absolute_first}-{absolute_last}",
        },
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    received = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(header)
            with ranged_request(shard, absolute_first, absolute_last) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    received += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if received != expected_bytes:
            raise ValueError(f"short layer {layer} payload: {received} != {expected_bytes}")
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "layer": layer,
        "kind": "full_attention" if layer % 4 == 3 else "linear_attention",
        "path": output.name,
        "sha256": digest(output),
        "bytes": output.stat().st_size,
        "source_shard": shard_name(shard),
        "source_shard_bytes": SHARDS[shard][0],
        "source_shard_lfs_sha256": SHARDS[shard][1],
        "source_absolute_range": [absolute_first, absolute_last],
        "tensor_count": len(entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    index_document = json.loads(INDEX.read_text(encoding="utf-8"))
    index_map = index_document["weight_map"]
    records = [fetch_layer(layer, shard, output_root, index_map) for layer, shard in LAYERS.items()]
    manifest = {
        "schema": "seen-qwen-official-layer-inputs-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_sha256": digest(INDEX),
        "layers": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} sha256={digest(manifest_path)}")


if __name__ == "__main__":
    main()
