#!/usr/bin/env python3
"""Extract the bounded QWN-024E prefill/decode oracle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle/expected.safetensors"
MANIFEST = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle/manifest.json"
OUTPUT = ROOT / "tests/fixtures/qwn_024e_cpu_engine_oracle.json"
SOURCE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
MANIFEST_SHA256 = "da4ead2d07206e9f091ac180802da58c770d58c993d72a8cb70c15098fe51baa"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert digest(SOURCE) == SOURCE_SHA256 and digest(MANIFEST) == MANIFEST_SHA256
    raw = SOURCE.read_bytes()
    header_length = struct.unpack_from("<Q", raw)[0]
    payload_start = 8 + header_length
    header = json.loads(raw[8:payload_start].decode("utf-8").rstrip(" "))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    logits: list[list[float]] = []
    for step in range(4):
        entry = header[f"base.decode_step_{step}.logits"]
        start, end = entry["data_offsets"]
        chunk = raw[payload_start + start:payload_start + end]
        logits.append(list(struct.unpack("<256f", chunk)))
    document = {
        "schema": "seen-qwen-cpu-engine-oracle-v1",
        "source": {
            "expected_safetensors_sha256": SOURCE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
        },
        "prompt_ids": manifest["input"]["token_ids"],
        "greedy_token_ids": manifest["outputs"]["greedy_token_ids"],
        "decode_logits": logits,
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={digest(OUTPUT)}")


if __name__ == "__main__":
    main()
