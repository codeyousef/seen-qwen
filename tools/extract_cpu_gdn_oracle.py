#!/usr/bin/env python3
"""Extract bounded QWN-024C GDN vectors from immutable model/oracle assets."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tests/fixtures/qwen3_8_hybrid_mini/model.safetensors"
ORACLE = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle/expected.safetensors"
OUTPUT = ROOT / "tests/fixtures/qwn_024c_cpu_gdn_oracle.json"
MODEL_SHA256 = "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9"
ORACLE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SOURCE_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> tuple[dict[str, dict[str, object]], bytes]:
    raw = path.read_bytes()
    header_length = struct.unpack_from("<Q", raw)[0]
    header_end = 8 + header_length
    return json.loads(raw[8:header_end].decode("utf-8").rstrip(" ")), raw[header_end:]


def tensor(
    header: dict[str, dict[str, object]], payload: bytes, name: str
) -> tuple[list[int], list[float]]:
    entry = header[name]
    assert entry["dtype"] == "F32"
    start, end = entry["data_offsets"]
    shape = entry["shape"]
    count = 1
    for dimension in shape:
        count *= dimension
    assert end - start == count * 4
    return shape, list(struct.unpack(f"<{count}f", payload[start:end]))


def linear(rows: list[float], row_count: int, width: int, weights: list[float], outputs: int) -> list[float]:
    result: list[float] = []
    for row in range(row_count):
        for output in range(outputs):
            total = 0.0
            for column in range(width):
                total += rows[row * width + column] * weights[output * width + column]
            result.append(total)
    return result


def main() -> None:
    assert digest(MODEL) == MODEL_SHA256
    assert digest(ORACLE) == ORACLE_SHA256
    model_header, model_payload = load(MODEL)
    oracle_header, oracle_payload = load(ORACLE)
    model_prefix = "model.language_model.layers.0.linear_attn"
    oracle_prefix = "base.layer_0"

    mixer_shape, mixer = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.mixer_input")
    qkv_shape, qkv_weight = tensor(model_header, model_payload, f"{model_prefix}.in_proj_qkv.weight")
    conv_weight_shape, conv_weight = tensor(model_header, model_payload, f"{model_prefix}.conv1d.weight")
    gate_weight_shape, gate_weight = tensor(model_header, model_payload, f"{model_prefix}.in_proj_z.weight")
    norm_weight_shape, norm_weight = tensor(model_header, model_payload, f"{model_prefix}.norm.weight")
    convolution_shape, convolution = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.convolution")
    query_shape, query = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.query")
    key_shape, key = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.key")
    value_shape, value = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.value")
    beta_shape, beta = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.beta")
    decay_shape, log_decay = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.log_decay")
    recurrent_shape, recurrent_final = tensor(
        oracle_header, oracle_payload, f"{oracle_prefix}.recurrent_final"
    )
    gated_shape, gated_norm = tensor(oracle_header, oracle_payload, f"{oracle_prefix}.gated_norm")

    assert mixer_shape == [1, 9, 48]
    assert qkv_shape == [80, 48]
    assert conv_weight_shape == [80, 1, 4]
    assert gate_weight_shape == [48, 48]
    assert norm_weight_shape == [8]
    assert convolution_shape == [1, 9, 80]
    assert query_shape == key_shape == value_shape == gated_shape == [1, 9, 6, 8]
    assert beta_shape == decay_shape == [1, 9, 6]
    assert recurrent_shape == [1, 6, 8, 8]

    mixed = linear(mixer, 9, 48, qkv_weight, 80)
    gate = linear(mixer, 9, 48, gate_weight, 48)
    document = {
        "schema": "seen-qwen-cpu-gdn-oracle-v1",
        "source": {
            "model_safetensors_sha256": MODEL_SHA256,
            "expected_safetensors_sha256": ORACLE_SHA256,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "modeling_source_sha256": MODELING_SOURCE_SHA256,
            "layer": 0,
        },
        "comparison": {"atol": 1.0e-5, "rtol": 1.0e-5},
        "geometry": {
            "sequence": 9,
            "channels": 80,
            "kernel": 4,
            "value_heads": 6,
            "key_dim": 8,
            "value_dim": 8,
        },
        "rms_norm_epsilon": 1.0e-6,
        "mixed": mixed,
        "conv_weight": conv_weight,
        "convolution": convolution,
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "log_decay": log_decay,
        "gate": gate,
        "norm_weight": norm_weight,
        "gated_norm": gated_norm,
        "recurrent_final": recurrent_final,
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={digest(OUTPUT)}")


if __name__ == "__main__":
    main()
