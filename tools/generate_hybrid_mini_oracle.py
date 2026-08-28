#!/usr/bin/env python3
"""Generate deterministic QWN-023C reference states with pinned PyTorch semantics."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "tests/fixtures/qwen3_8_hybrid_mini"
MODEL_PATH = ASSET_ROOT / "model.safetensors"
CONTRACT_PATH = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_contract.json"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_oracle"
CONTRACT_SHA256 = "f29839615771e344bf89329f2195e5921fc8ea371849249be55722ab1999dddf"
MODEL_SHA256 = "16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SOURCE_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"
MTP_COMMIT = "22283d7a1b4eff75fb0d63fb2e862ade39df7b73"
MTP_SOURCE_SHA256 = "cf97664e82371425df14e412c9d351405d05ae8db3622fc817813ddda6858622"
PROMPT_TEXT = "Seen Qwen"
PROMPT_IDS = list(PROMPT_TEXT.encode("utf-8"))
GENERATED_TOKEN_COUNT = 4
ORACLE_ATOL = 1.0e-5
ORACLE_RTOL = 1.0e-5


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    raw = bytearray(path.read_bytes())
    if len(raw) < 8:
        raise ValueError("truncated Safetensors input")
    header_length = struct.unpack_from("<Q", raw)[0]
    header_end = 8 + header_length
    if header_end > len(raw):
        raise ValueError("Safetensors header exceeds file")
    header = json.loads(raw[8:header_end].decode("utf-8").rstrip(" "))
    payload = memoryview(raw)[header_end:]
    tensors: dict[str, torch.Tensor] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] != "F32":
            raise ValueError(f"unsupported fixture dtype for {name}")
        start, end = entry["data_offsets"]
        view = payload[start:end]
        expected = math.prod(entry["shape"])
        if len(view) != expected * 4:
            raise ValueError(f"invalid payload extent for {name}")
        tensors[name] = torch.frombuffer(view, dtype=torch.float32).clone().reshape(entry["shape"])
    return tensors


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * (1.0 + weight.float())).to(x.dtype)


def gated_rms_norm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (weight * normalized.to(x.dtype) * F.silu(gate.float())).to(x.dtype)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotary_embeddings(
    sequence_length: int,
    rotary_dim: int,
    theta: float,
    dtype: torch.dtype,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(
        position_offset, position_offset + sequence_length, dtype=torch.float32
    )
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rotary(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_out = q_rot * cos + rotate_half(q_rot) * sin
    k_out = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def full_attention(
    hidden: torch.Tensor,
    prefix: str,
    weights: dict[str, torch.Tensor],
    contract: dict[str, object],
    checkpoints: dict[str, torch.Tensor],
    position_offset: int = 0,
) -> torch.Tensor:
    heads = int(contract["num_attention_heads"])
    kv_heads = int(contract["num_key_value_heads"])
    head_dim = int(contract["head_dim"])
    eps = float(contract["rms_norm_eps"])
    batch, sequence, _ = hidden.shape
    projected = linear(hidden, weights[f"{prefix}.q_proj.weight"])
    projected = projected.reshape(batch, sequence, heads, head_dim * 2)
    query, gate = projected.chunk(2, dim=-1)
    gate = gate.reshape(batch, sequence, heads * head_dim)
    query = rms_norm(query, weights[f"{prefix}.q_norm.weight"], eps).transpose(1, 2)
    key = linear(hidden, weights[f"{prefix}.k_proj.weight"]).reshape(
        batch, sequence, kv_heads, head_dim
    )
    key = rms_norm(key, weights[f"{prefix}.k_norm.weight"], eps).transpose(1, 2)
    value = linear(hidden, weights[f"{prefix}.v_proj.weight"]).reshape(
        batch, sequence, kv_heads, head_dim
    ).transpose(1, 2)
    cos, sin = rotary_embeddings(
        sequence,
        int(contract["rotary_dim"]),
        float(contract["rope_theta"]),
        hidden.dtype,
        position_offset,
    )
    query, key = apply_rotary(query, key, cos, sin)
    repeat = heads // kv_heads
    repeated_key = key.repeat_interleave(repeat, dim=1)
    repeated_value = value.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * (head_dim**-0.5)
    causal = torch.triu(torch.full((sequence, sequence), float("-inf")), diagonal=1)
    probabilities = torch.softmax(scores + causal[None, None, :, :], dim=-1, dtype=torch.float32)
    attended = torch.matmul(probabilities.to(query.dtype), repeated_value)
    attended = attended.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
    gated = attended * torch.sigmoid(gate)
    checkpoints.update({
        "query": query,
        "key": key,
        "value": value,
        "probabilities": probabilities,
        "gate": gate,
        "attended": attended,
    })
    return linear(gated, weights[f"{prefix}.o_proj.weight"])


def gated_delta(
    hidden: torch.Tensor,
    prefix: str,
    weights: dict[str, torch.Tensor],
    contract: dict[str, object],
    checkpoints: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch, sequence, _ = hidden.shape
    key_heads = int(contract["linear_num_key_heads"])
    value_heads = int(contract["linear_num_value_heads"])
    key_head_dim = int(contract["linear_key_head_dim"])
    value_head_dim = int(contract["linear_value_head_dim"])
    key_width = key_heads * key_head_dim
    value_width = value_heads * value_head_dim
    eps = float(contract["rms_norm_eps"])
    mixed = linear(hidden, weights[f"{prefix}.in_proj_qkv.weight"]).transpose(1, 2)
    convolution = F.conv1d(
        mixed,
        weights[f"{prefix}.conv1d.weight"],
        padding=int(contract["linear_conv_kernel_dim"]) - 1,
        groups=mixed.shape[1],
    )[:, :, :sequence]
    convolution = F.silu(convolution).transpose(1, 2)
    query, key, value = torch.split(convolution, [key_width, key_width, value_width], dim=-1)
    query = query.reshape(batch, sequence, key_heads, key_head_dim)
    key = key.reshape(batch, sequence, key_heads, key_head_dim)
    value = value.reshape(batch, sequence, value_heads, value_head_dim)
    beta = torch.sigmoid(linear(hidden, weights[f"{prefix}.in_proj_b.weight"]))
    decay = -weights[f"{prefix}.A_log"].float().exp() * F.softplus(
        linear(hidden, weights[f"{prefix}.in_proj_a.weight"]).float()
        + weights[f"{prefix}.dt_bias"]
    )
    repeat = value_heads // key_heads
    query = query.repeat_interleave(repeat, dim=2)
    key = key.repeat_interleave(repeat, dim=2)
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1.0e-6)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1.0e-6)
    query = query * (key_head_dim**-0.5)
    state = torch.zeros(batch, value_heads, key_head_dim, value_head_dim, dtype=torch.float32)
    outputs: list[torch.Tensor] = []
    for position in range(sequence):
        q_t = query[:, position].float()
        k_t = key[:, position].float()
        v_t = value[:, position].float()
        state = state * decay[:, position].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - memory) * beta[:, position].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    core = torch.stack(outputs, dim=1).to(hidden.dtype)
    gate = linear(hidden, weights[f"{prefix}.in_proj_z.weight"]).reshape(
        batch, sequence, value_heads, value_head_dim
    )
    normalized = gated_rms_norm(core, gate, weights[f"{prefix}.norm.weight"], eps)
    checkpoints.update({
        "convolution": convolution,
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "log_decay": decay,
        "recurrent_final": state,
        "gated_norm": normalized,
    })
    return linear(normalized.reshape(batch, sequence, value_width), weights[f"{prefix}.out_proj.weight"])


def mlp(hidden: torch.Tensor, prefix: str, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = F.silu(linear(hidden, weights[f"{prefix}.gate_proj.weight"]))
    up = linear(hidden, weights[f"{prefix}.up_proj.weight"])
    return linear(gate * up, weights[f"{prefix}.down_proj.weight"])


def decoder_layer(
    hidden: torch.Tensor,
    layer_prefix: str,
    mixer_prefix: str,
    layer_type: str,
    weights: dict[str, torch.Tensor],
    contract: dict[str, object],
    position_offset: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = float(contract["rms_norm_eps"])
    checkpoints: dict[str, torch.Tensor] = {"input": hidden}
    mixer_input = rms_norm(hidden, weights[f"{layer_prefix}.input_layernorm.weight"], eps)
    checkpoints["mixer_input"] = mixer_input
    if layer_type == "linear_attention":
        mixer_output = gated_delta(mixer_input, mixer_prefix, weights, contract, checkpoints)
    else:
        mixer_output = full_attention(
            mixer_input, mixer_prefix, weights, contract, checkpoints, position_offset
        )
    hidden = hidden + mixer_output
    checkpoints["mixer_output"] = mixer_output
    checkpoints["post_mixer"] = hidden
    mlp_input = rms_norm(hidden, weights[f"{layer_prefix}.post_attention_layernorm.weight"], eps)
    mlp_output = mlp(mlp_input, f"{layer_prefix}.mlp", weights)
    hidden = hidden + mlp_output
    checkpoints.update({"mlp_input": mlp_input, "mlp_output": mlp_output, "output": hidden})
    return hidden, checkpoints


def base_forward(
    token_ids: list[int], weights: dict[str, torch.Tensor], contract: dict[str, object], capture: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    ids = torch.tensor(token_ids, dtype=torch.long)[None, :]
    hidden = F.embedding(ids, weights["model.language_model.embed_tokens.weight"])
    captured: dict[str, torch.Tensor] = {"base.embeddings": hidden}
    for index, layer_type in enumerate(contract["layer_types"]):
        layer_prefix = f"model.language_model.layers.{index}"
        mixer_name = "linear_attn" if layer_type == "linear_attention" else "self_attn"
        hidden, layer_points = decoder_layer(
            hidden,
            layer_prefix,
            f"{layer_prefix}.{mixer_name}",
            layer_type,
            weights,
            contract,
        )
        if capture:
            for name, tensor in layer_points.items():
                captured[f"base.layer_{index}.{name}"] = tensor
    hidden = rms_norm(hidden, weights["model.language_model.norm.weight"], float(contract["rms_norm_eps"]))
    logits = linear(hidden, weights["lm_head.weight"])
    captured["base.final_hidden"] = hidden
    captured["base.logits"] = logits
    return hidden, logits, captured


def mtp_forward(
    input_id: int,
    position: int,
    base_hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    contract: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    eps = float(contract["rms_norm_eps"])
    embedding = weights["model.language_model.embed_tokens.weight"][input_id].reshape(1, 1, -1)
    base_hidden = base_hidden[:, -1:, :]
    embedding_norm = rms_norm(embedding, weights["mtp.pre_fc_norm_embedding.weight"], eps)
    hidden_norm = rms_norm(base_hidden, weights["mtp.pre_fc_norm_hidden.weight"], eps)
    stem = linear(torch.cat((embedding_norm, hidden_norm), dim=-1), weights["mtp.fc.weight"])
    # A single-token MTP step has no visible earlier key/value state in this fixture.
    hidden, points = decoder_layer(
        stem,
        "mtp.layers.0",
        "mtp.layers.0.self_attn",
        "full_attention",
        weights,
        contract,
        position,
    )
    hidden = rms_norm(hidden, weights["mtp.norm.weight"], eps)
    logits = linear(hidden, weights["lm_head.weight"])
    captured = {
        "mtp.input_embedding": embedding,
        "mtp.embedding_norm": embedding_norm,
        "mtp.hidden_norm": hidden_norm,
        "mtp.stem": stem,
        "mtp.position": torch.tensor([position], dtype=torch.int64),
        "mtp.final_hidden": hidden,
        "mtp.logits": logits,
    }
    for name, tensor in points.items():
        captured[f"mtp.layer_0.{name}"] = tensor
    return hidden, logits, captured


def safetensors_bytes(tensors: dict[str, torch.Tensor]) -> bytes:
    header: dict[str, object] = {
        "__metadata__": {"fixture": "qwen3_8_hybrid_mini_v1", "oracle": "qwn-023c-v1"}
    }
    chunks: list[bytes] = []
    offset = 0
    dtype_names = {torch.float32: "F32", torch.int64: "I64"}
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.dtype not in dtype_names:
            raise ValueError(f"unsupported output dtype for {name}: {tensor.dtype}")
        chunk = bytes(tensor.view(torch.uint8).flatten().tolist())
        header[name] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(chunk)],
        }
        chunks.append(chunk)
        offset += len(chunk)
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)


def write_atomic(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def generate(output: Path) -> None:
    if digest(CONTRACT_PATH) != CONTRACT_SHA256 or digest(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("QWN-023A/B dependency hash mismatch")
    if torch.__version__ != "2.11.0+cpu":
        raise ValueError(f"oracle requires pinned torch 2.11.0+cpu, got {torch.__version__}")
    if sys.version.split()[0] != "3.14.7":
        raise ValueError(f"oracle requires pinned Python 3.14.7, got {sys.version.split()[0]}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    weights = load_safetensors(MODEL_PATH)
    with torch.inference_mode():
        hidden, logits, captured = base_forward(PROMPT_IDS, weights, contract, capture=True)
        generated: list[int] = []
        sequence = PROMPT_IDS[:]
        for step in range(GENERATED_TOKEN_COUNT):
            _, step_logits, _ = base_forward(sequence, weights, contract, capture=False)
            captured[f"base.decode_step_{step}.logits"] = step_logits[:, -1:, :]
            token = int(torch.argmax(step_logits[0, -1]).item())
            generated.append(token)
            sequence.append(token)
        mtp_input = generated[0]
        _, mtp_logits, mtp_points = mtp_forward(
            mtp_input, len(PROMPT_IDS), hidden, weights, contract
        )
        captured.update(mtp_points)
        captured["base.input_ids"] = torch.tensor(PROMPT_IDS, dtype=torch.int64)
        captured["base.generated_ids"] = torch.tensor(generated, dtype=torch.int64)
        captured["base.greedy_token"] = torch.tensor(
            [int(torch.argmax(logits[0, -1]).item())], dtype=torch.int64
        )
        captured["mtp.greedy_token"] = torch.tensor(
            [int(torch.argmax(mtp_logits[0, -1]).item())], dtype=torch.int64
        )
    for name, tensor in captured.items():
        if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite oracle tensor: {name}")
    oracle = safetensors_bytes(captured)
    manifest = {
        "schema": "seen-qwen-hybrid-mini-oracle-v1",
        "fixture_id": contract["fixture_id"],
        "maturity": "verified",
        "dependencies": {
            "contract_sha256": CONTRACT_SHA256,
            "model_safetensors_sha256": MODEL_SHA256,
        },
        "sources": {
            "transformers_commit": TRANSFORMERS_COMMIT,
            "modeling_source_sha256": MODELING_SOURCE_SHA256,
            "qwen_mtp_repository": "QwenLM/Confident-Decoding",
            "qwen_mtp_commit": MTP_COMMIT,
            "qwen_mtp_source_sha256": MTP_SOURCE_SHA256,
        },
        "environment": {
            "python": "3.14.7",
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "deterministic_algorithms": True,
        },
        "input": {"text": PROMPT_TEXT, "token_ids": PROMPT_IDS},
        "outputs": {
            "greedy_token_ids": generated,
            "mtp_input_token_id": mtp_input,
            "mtp_greedy_token_id": int(torch.argmax(mtp_logits[0, -1]).item()),
        },
        "coverage": {
            "base_layers": len(contract["layer_types"]),
            "linear_attention_layers": contract["layer_types"].count("linear_attention"),
            "full_attention_layers": contract["layer_types"].count("full_attention"),
            "mtp_layers": int(contract["mtp_num_hidden_layers"]),
            "tensor_count": len(captured),
            "non_finite_values": 0,
        },
        "comparison": {"dtype": "F32", "atol": ORACLE_ATOL, "rtol": ORACLE_RTOL},
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": {
            "expected.safetensors": {"bytes": len(oracle), "sha256": sha256(oracle).hexdigest()}
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if output in (Path("/"), ROOT, ROOT.parent) or output.is_symlink():
        raise ValueError("unsafe output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    backup: Path | None = None
    try:
        write_atomic(temporary / "expected.safetensors", oracle)
        write_atomic(temporary / "manifest.json", manifest_bytes)
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        os.replace(temporary, output)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.output_dir.absolute())


if __name__ == "__main__":
    main()
