#!/usr/bin/env python3
"""Fetch and verify the immutable official Qwen3.8-27B shard set for QWN-025B.

The complete upstream shards stay below the ignored project-local `.seen/`
root.  The capture tool reads only text-model tensors; repository code is
never downloaded or executed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
DEFAULT_OUTPUT = ROOT / ".seen/oracle-official/full-model"
CHUNK_BYTES = 8 * 1024 * 1024
FREE_SPACE_RESERVE_BYTES = 8 * 1024 * 1024 * 1024

# Immutable Hugging Face LFS identities returned for MODEL_REVISION.
SHARDS = {
    1: (3_966_730_552, "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"),
    2: (3_043_080_328, "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"),
    3: (2_542_796_952, "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"),
    4: (3_988_973_152, "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"),
    5: (2_099_339_864, "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"),
    6: (3_979_553_696, "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"),
    7: (2_108_759_344, "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"),
    8: (3_979_553_696, "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"),
    9: (2_108_759_344, "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"),
    10: (3_979_553_696, "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"),
    11: (2_108_759_344, "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"),
    12: (3_979_553_696, "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"),
    13: (2_108_759_344, "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"),
    14: (3_979_553_696, "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"),
    15: (2_108_759_344, "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"),
    16: (3_979_564_040, "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"),
    17: (2_108_759_344, "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"),
    18: (3_392_197_344, "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"),
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            value.update(chunk)
    return value.hexdigest()


def shard_name(shard: int) -> str:
    return f"model-{shard:05d}-of-00018.safetensors"


def shard_url(shard: int) -> str:
    return (
        f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"
        f"{shard_name(shard)}?download=true"
    )


def validate_index() -> dict[str, str]:
    if digest(INDEX) != INDEX_SHA256:
        raise ValueError("checked-in model index digest mismatch")
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index weight_map is absent or empty")
    expected_names = {shard_name(shard) for shard in SHARDS}
    actual_names = set(weight_map.values())
    if actual_names != expected_names:
        raise ValueError("model index does not reference exactly the locked shard set")
    if document.get("metadata", {}).get("total_size") != 55_562_855_904.0:
        raise ValueError("model index tensor byte total changed")
    return weight_map


def validate_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("output root must be a real directory")
    resolved = output_root.resolve()
    ignored_root = (ROOT / ".seen").resolve()
    if resolved != ignored_root and ignored_root not in resolved.parents:
        raise ValueError("full-model shards must remain under the ignored project .seen root")


def validate_existing(path: Path, expected_bytes: int, expected_sha: str) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe shard path: {path.name}")
    if path.stat().st_size != expected_bytes:
        return False
    return digest(path) == expected_sha


def download_shard(shard: int, output_root: Path) -> dict[str, object]:
    expected_bytes, expected_sha = SHARDS[shard]
    output = output_root / shard_name(shard)
    if validate_existing(output, expected_bytes, expected_sha):
        return {"path": output.name, "bytes": expected_bytes, "lfs_sha256": expected_sha}

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output_root)
    received = 0
    try:
        request = urllib.request.Request(shard_url(shard), headers={"Accept-Encoding": "identity"})
        with os.fdopen(descriptor, "wb") as target, urllib.request.urlopen(request, timeout=120) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != expected_bytes:
                raise ValueError(f"unexpected Content-Length for shard {shard}: {length}")
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                target.write(chunk)
                received += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        temporary = Path(temporary_name)
        if received != expected_bytes or temporary.stat().st_size != expected_bytes:
            raise ValueError(f"short shard {shard}: {received} != {expected_bytes}")
        actual_sha = digest(temporary)
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for shard {shard}: {actual_sha}")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {"path": output.name, "bytes": expected_bytes, "lfs_sha256": expected_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    validate_index()
    output_root = arguments.output_root.resolve()
    validate_output_root(output_root)
    missing_bytes = sum(
        size
        for shard, (size, identity) in SHARDS.items()
        if not validate_existing(output_root / shard_name(shard), size, identity)
    )
    available = shutil.disk_usage(output_root).free
    if available - missing_bytes < FREE_SPACE_RESERVE_BYTES:
        raise ValueError(
            f"insufficient free space: available={available} missing={missing_bytes} "
            f"reserve={FREE_SPACE_RESERVE_BYTES}"
        )
    records = []
    for shard in SHARDS:
        print(f"fetching/validating {shard_name(shard)}", flush=True)
        records.append(download_shard(shard, output_root))
    manifest = {
        "schema": "seen-qwen-official-full-model-inputs-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_sha256": INDEX_SHA256,
        "tensor_bytes": 55_562_855_904,
        "shards": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} sha256={digest(manifest_path)}")


if __name__ == "__main__":
    main()
