#!/usr/bin/env python3
"""Extract a bounded QWN-024B prefix from the locked QWN-023C oracle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle/expected.safetensors"
OUTPUT = ROOT / "tests/fixtures/qwn_024b_cpu_attention_oracle.json"
SOURCE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SOURCE_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"
SEQUENCE = 3
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 24


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict[str, dict[str, object]], bytes]:
    raw = SOURCE.read_bytes()
    header_length = struct.unpack_from("<Q", raw)[0]
    header_end = 8 + header_length
    return json.loads(raw[8:header_end].decode("utf-8").rstrip(" ")), raw[header_end:]


def tensor(header: dict[str, dict[str, object]], payload: bytes, name: str) -> tuple[list[int], list[float]]:
    entry = header[name]
    assert entry["dtype"] == "F32"
    start, end = entry["data_offsets"]
    shape = entry["shape"]
    count = 1
    for dimension in shape:
        count *= dimension
    assert end - start == count * 4
    return shape, list(struct.unpack(f"<{count}f", payload[start:end]))


def main() -> None:
    assert digest(SOURCE) == SOURCE_SHA256
    header, payload = load()
    prefix = "base.layer_3"
    query_shape, query_source = tensor(header, payload, f"{prefix}.query")
    key_shape, key_source = tensor(header, payload, f"{prefix}.key")
    value_shape, value_source = tensor(header, payload, f"{prefix}.value")
    gate_shape, gate_source = tensor(header, payload, f"{prefix}.gate")
    probability_shape, probability_source = tensor(header, payload, f"{prefix}.probabilities")
    attended_shape, attended_source = tensor(header, payload, f"{prefix}.attended")
    assert query_shape == [1, 6, 9, HEAD_DIM]
    assert key_shape == value_shape == [1, KV_HEADS, 9, HEAD_DIM]
    assert gate_shape == attended_shape == [1, 9, 6 * HEAD_DIM]
    assert probability_shape == [1, 6, 9, 9]

    query = [
        query_source[((head * 9 + position) * HEAD_DIM) + dimension]
        for position in range(SEQUENCE)
        for head in range(QUERY_HEADS)
        for dimension in range(HEAD_DIM)
    ]
    key = [
        key_source[(position * HEAD_DIM) + dimension]
        for position in range(SEQUENCE)
        for dimension in range(HEAD_DIM)
    ]
    value = [
        value_source[(position * HEAD_DIM) + dimension]
        for position in range(SEQUENCE)
        for dimension in range(HEAD_DIM)
    ]
    gate = [
        gate_source[(position * 6 * HEAD_DIM) + (head * HEAD_DIM) + dimension]
        for position in range(SEQUENCE)
        for head in range(QUERY_HEADS)
        for dimension in range(HEAD_DIM)
    ]
    probabilities = [
        probability_source[((head * 9 + position) * 9) + key_position]
        for position in range(SEQUENCE)
        for head in range(QUERY_HEADS)
        for key_position in range(SEQUENCE)
    ]
    attended = [
        attended_source[(position * 6 * HEAD_DIM) + (head * HEAD_DIM) + dimension]
        for position in range(SEQUENCE)
        for head in range(QUERY_HEADS)
        for dimension in range(HEAD_DIM)
    ]
    document = {
        "schema": "seen-qwen-cpu-attention-oracle-v1",
        "source": {
            "expected_safetensors_sha256": SOURCE_SHA256,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "modeling_source_sha256": MODELING_SOURCE_SHA256,
            "layer": 3,
        },
        "comparison": {"atol": 1.0e-5, "rtol": 1.0e-5},
        "geometry": {
            "sequence": SEQUENCE,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "query_start": 0,
        },
        "query": query,
        "key": key,
        "value": value,
        "gate": gate,
        "probabilities": probabilities,
        "attended": attended,
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={digest(OUTPUT)}")


if __name__ == "__main__":
    main()
