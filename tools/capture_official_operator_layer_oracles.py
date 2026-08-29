#!/usr/bin/env python3
"""Capture bounded O1 vectors from exact official Qwen3.8 layer weights/source."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"
CONFIG = ROOT / "tests/fixtures/qwen3_8_config.json"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
DEFAULT_INPUT = ROOT / ".seen/oracle-official/layers"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/qwn_025a_operator_layer_oracles.json"
LAYERS = (0, 3, 31, 32, 60, 63)
POSITIONS = (0, 31)
SAMPLE_WIDTH = 16


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return sha256(raw).hexdigest()


def tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    flat = value.float().flatten()
    if flat.numel() == 0:
        raise ValueError("oracle tensors must not be empty")
    if not torch.isfinite(flat).all():
        raise ValueError("oracle tensor contains NaN or infinity")
    middle = flat.numel() // 2
    indices = sorted(
        set(
            list(range(min(SAMPLE_WIDTH, flat.numel())))
            + list(range(max(0, middle - SAMPLE_WIDTH // 2), min(flat.numel(), middle + SAMPLE_WIDTH // 2)))
            + list(range(max(0, flat.numel() - SAMPLE_WIDTH), flat.numel()))
        )
    )
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": tensor_digest(value),
        "sample_indices": indices,
        "sample_values_f32": [float(flat[index].item()) for index in indices],
        "minimum_f32": float(flat.min().item()),
        "maximum_f32": float(flat.max().item()),
    }


def load_layer(path: Path, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer}."
    result: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in source.keys():
            if not name.startswith(prefix):
                raise ValueError(f"unexpected tensor in layer pack: {name}")
            result[name[len(prefix) :]] = source.get_tensor(name)
    return result


def deterministic_hidden(layer: int, hidden_size: int) -> torch.Tensor:
    positions = torch.arange(len(POSITIONS), dtype=torch.int64)[:, None]
    dimensions = torch.arange(hidden_size, dtype=torch.int64)[None, :]
    integers = ((positions + 3) * (dimensions + 11) + layer * 19) % 257 - 128
    return (integers.to(torch.float32) / 128.0).to(torch.bfloat16).unsqueeze(0)


def capture_gdn_state(module, normalized: torch.Tensor, modeling) -> tuple[torch.Tensor, torch.Tensor]:
    mixer = module.linear_attn
    batch, sequence, _ = normalized.shape
    mixed = mixer.in_proj_qkv(normalized).transpose(1, 2)
    z = mixer.in_proj_z(normalized).reshape(batch, sequence, -1, mixer.head_v_dim)
    b = mixer.in_proj_b(normalized)
    a = mixer.in_proj_a(normalized)
    mixed = modeling.causal_conv1d_fn(
        mixed,
        mixer.conv1d.weight.squeeze(1),
        mixer.conv1d.bias,
        activation=mixer.activation,
    ).transpose(1, 2)
    query, key, value = torch.split(mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1)
    query = query.reshape(batch, sequence, -1, mixer.head_k_dim)
    key = key.reshape(batch, sequence, -1, mixer.head_k_dim)
    value = value.reshape(batch, sequence, -1, mixer.head_v_dim)
    beta = b.sigmoid()
    decay = -mixer.A_log.float().exp() * torch.nn.functional.softplus(a.float() + mixer.dt_bias)
    repeat = mixer.num_v_heads // mixer.num_k_heads
    query = query.repeat_interleave(repeat, dim=2)
    key = key.repeat_interleave(repeat, dim=2)
    core, state = modeling.torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=decay,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    gated = mixer.norm(core.reshape(-1, mixer.head_v_dim), z.reshape(-1, mixer.head_v_dim))
    return state, gated.reshape(batch, sequence, -1)


def capture_layer(layer: int, input_root: Path, config, rotary, modeling, manifest_entry) -> dict[str, object]:
    path = input_root / f"layer-{layer:02d}.safetensors"
    if digest(path) != manifest_entry["sha256"]:
        raise ValueError(f"layer {layer} pack digest mismatch")
    state = load_layer(path, layer)
    with torch.device("meta"):
        module = modeling.Qwen3_5DecoderLayer(config, layer)
    missing, unexpected = module.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise ValueError(f"layer {layer} state mismatch: missing={missing}, unexpected={unexpected}")
    module.eval()
    hidden = deterministic_hidden(layer, config.hidden_size)
    position_ids = torch.tensor(POSITIONS, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)
    causal = torch.full((1, 1, len(POSITIONS), len(POSITIONS)), float("-inf"), dtype=hidden.dtype)
    causal = torch.triu(causal, diagonal=1)
    captured: dict[str, torch.Tensor] = {"layer_input": hidden}
    hooks = []

    def save(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output
        return hook

    hooks.append(module.input_layernorm.register_forward_hook(save("input_norm")))
    hooks.append(module.post_attention_layernorm.register_forward_hook(save("post_attention_norm")))
    hooks.append(module.mlp.register_forward_hook(save("mlp_output")))
    if module.block_type == "linear_attention":
        hooks.append(module.linear_attn.register_forward_hook(save("token_mixer")))
        hooks.append(module.linear_attn.in_proj_qkv.register_forward_hook(save("qkv_projection")))
    else:
        hooks.append(module.self_attn.register_forward_hook(save("token_mixer")))
        hooks.append(module.self_attn.q_proj.register_forward_hook(save("query_gate_projection")))
    with torch.inference_mode():
        captured["layer_output"] = module(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal if module.block_type == "full_attention" else None,
            position_ids=position_ids,
            past_key_values=None,
        )
        if module.block_type == "linear_attention":
            recurrent, gated = capture_gdn_state(module, captured["input_norm"], modeling)
            captured["recurrent_state"] = recurrent
            captured["gated_recurrent_output"] = gated
    for hook in hooks:
        hook.remove()
    return {
        "layer": layer,
        "kind": module.block_type,
        "source": manifest_entry,
        "positions": list(POSITIONS),
        "operators": {name: tensor_record(value) for name, value in sorted(captured.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if torch.__version__ != "2.11.0+cpu":
        raise ValueError(f"oracle requires torch 2.11.0+cpu, got {torch.__version__}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    import transformers

    if transformers.__version__ != "5.16.0.dev0":
        raise ValueError(f"unexpected Transformers version: {transformers.__version__}")
    source_path = Path(modeling.__file__).resolve()
    if digest(source_path) != MODELING_SHA256:
        raise ValueError("official modeling source digest mismatch")
    config_document = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = Qwen3_5TextConfig.from_dict(config_document["text_config"])
    config._attn_implementation = "eager"
    rotary = modeling.Qwen3_5TextRotaryEmbedding(config)
    rotary.eval()
    input_root = arguments.input_root.resolve()
    input_manifest_path = input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    entries = {entry["layer"]: entry for entry in input_manifest["layers"]}
    if tuple(sorted(entries)) != LAYERS:
        raise ValueError("official layer input manifest has the wrong selected layers")
    layers = [capture_layer(layer, input_root, config, rotary, modeling, entries[layer]) for layer in LAYERS]
    output = {
        "schema": "seen-qwen-official-operator-layer-oracles-v1",
        "classification": "verified-cpu-reference",
        "source": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": digest(CONFIG),
            "tensor_index_sha256": digest(INDEX),
            "capture_tool_sha256": digest(Path(__file__).resolve()),
            "fetch_tool_sha256": digest(ROOT / "tools/fetch_official_layer_ranges.py"),
            "transformers_commit": TRANSFORMERS_COMMIT,
            "modeling_source_sha256": MODELING_SHA256,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "input_manifest_sha256": digest(input_manifest_path),
        },
        "determinism": {
            "device": "cpu",
            "workers": 1,
            "algorithms": "torch.use_deterministic_algorithms(true)",
            "input_formula": "bf16((((position_index+3)*(dimension+11)+layer*19)%257-128)/128)",
        },
        "selection": {
            "first_linear_attention": 0,
            "first_full_attention": 3,
            "middle_full_attention": 31,
            "middle_linear_attention": 32,
            "last_linear_attention": 60,
            "last_full_attention": 63,
        },
        "layers": layers,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output} sha256={digest(arguments.output)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"qwn-025a capture failed: {error}", file=sys.stderr)
        raise
