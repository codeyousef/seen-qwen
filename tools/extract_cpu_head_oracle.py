#!/usr/bin/env python3
"""Extract bounded QWN-024D FFN, MTP, and logits vectors."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tests/fixtures/qwen3_8_hybrid_mini/model.safetensors"
ORACLE = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle/expected.safetensors"
OUTPUT = ROOT / "tests/fixtures/qwn_024d_cpu_head_oracle.json"
MODEL_SHA256 = "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9"
ORACLE_SHA256 = "dfa4e8eb7550e7e694c9044d63f602e406fea09153a849274250b046db350096"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MTP_COMMIT = "22283d7a1b4eff75fb0d63fb2e862ade39df7b73"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def load(path: Path) -> tuple[dict[str, dict[str, object]], bytes]:
    raw = path.read_bytes(); header_length = struct.unpack_from("<Q", raw)[0]
    end = 8 + header_length
    return json.loads(raw[8:end].decode("utf-8").rstrip(" ")), raw[end:]


def tensor(header: dict[str, dict[str, object]], payload: bytes, name: str) -> tuple[list[int], list[float]]:
    entry = header[name]; assert entry["dtype"] == "F32"
    start, end = entry["data_offsets"]; shape = entry["shape"]
    count = math.prod(shape); assert end - start == count * 4
    return shape, list(struct.unpack(f"<{count}f", payload[start:end]))


def linear(row: list[float], weights: list[float], outputs: int, width: int) -> list[float]:
    result: list[float] = []
    for output in range(outputs):
        total = 0.0
        for column in range(width):
            total = f32(total + f32(row[column] * weights[output * width + column]))
        result.append(total)
    return result


def main() -> None:
    assert digest(MODEL) == MODEL_SHA256 and digest(ORACLE) == ORACLE_SHA256
    mh, mp = load(MODEL); oh, op = load(ORACLE)
    _, mlp_input_all = tensor(oh, op, "base.layer_0.mlp_input")
    mlp_input = mlp_input_all[:48]
    _, gate_all = tensor(mh, mp, "model.language_model.layers.0.mlp.gate_proj.weight")
    _, up_all = tensor(mh, mp, "model.language_model.layers.0.mlp.up_proj.weight")
    _, down_all = tensor(mh, mp, "model.language_model.layers.0.mlp.down_proj.weight")
    gate_weight = gate_all[:8 * 48]; up_weight = up_all[:8 * 48]
    down_weight = [down_all[row * 160 + column] for row in range(6) for column in range(8)]
    gate = linear(mlp_input, gate_weight, 8, 48)
    up = linear(mlp_input, up_weight, 8, 48)
    swiglu = [f32(f32(value / f32(1.0 + f32(math.exp(f32(-value))))) * up[i])
              for i, value in enumerate(gate)]
    ffn_output = linear(swiglu, down_weight, 6, 8)

    _, input_embedding = tensor(oh, op, "mtp.input_embedding")
    _, base_hidden_all = tensor(oh, op, "base.final_hidden")
    base_hidden = base_hidden_all[-48:]
    _, embedding_norm_weight = tensor(mh, mp, "mtp.pre_fc_norm_embedding.weight")
    _, hidden_norm_weight = tensor(mh, mp, "mtp.pre_fc_norm_hidden.weight")
    _, fusion_weight = tensor(mh, mp, "mtp.fc.weight")
    _, mtp_stem = tensor(oh, op, "mtp.stem")

    _, lm_head_all = tensor(mh, mp, "lm_head.weight")
    token_rows = [0, 1, 166, 170, 255]
    lm_head_rows = [lm_head_all[token * 48 + column] for token in token_rows for column in range(48)]
    _, base_logits_all = tensor(oh, op, "base.logits")
    base_logits = base_logits_all[-256:]
    _, mtp_hidden = tensor(oh, op, "mtp.final_hidden")
    _, mtp_logits = tensor(oh, op, "mtp.logits")
    document = {
        "schema": "seen-qwen-cpu-head-oracle-v1",
        "source": {"model_safetensors_sha256": MODEL_SHA256,
                   "expected_safetensors_sha256": ORACLE_SHA256,
                   "transformers_commit": TRANSFORMERS_COMMIT,
                   "mtp_commit": MTP_COMMIT},
        "geometry": {"hidden": 48, "reduced_intermediate": 8,
                     "reduced_output": 6, "vocabulary": 256},
        "mlp_input": mlp_input, "gate_weight": gate_weight,
        "up_weight": up_weight, "down_weight": down_weight,
        "ffn_output": ffn_output,
        "mtp_input_embedding": input_embedding, "mtp_base_hidden": base_hidden,
        "mtp_embedding_norm_weight": embedding_norm_weight,
        "mtp_hidden_norm_weight": hidden_norm_weight,
        "mtp_fusion_weight": fusion_weight, "mtp_stem": mtp_stem,
        "lm_head_token_rows": token_rows, "lm_head_rows": lm_head_rows,
        "base_hidden": base_hidden, "base_logits": base_logits,
        "mtp_hidden": mtp_hidden, "mtp_logits": mtp_logits,
        "base_greedy_token": 170, "mtp_greedy_token": 166,
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} sha256={digest(OUTPUT)}")


if __name__ == "__main__":
    main()
