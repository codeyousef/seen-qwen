#!/usr/bin/env python3
"""Generate the deterministic QWN-023B Safetensors and tokenizer fixtures."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests/fixtures/qwen3_8_hybrid_mini_contract.json"
CONTRACT_SHA256 = "f29839615771e344bf89329f2195e5921fc8ea371849249be55722ab1999dddf"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/qwen3_8_hybrid_mini"
MASK64 = (1 << 64) - 1
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SOURCE_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"
SHARD_01_HEADER_SHA256 = "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0"
SHARD_18_HEADER_SHA256 = "2e6b71178b57a98de637fa669dedfaf35cbed1388914ffc93238bddd169eab54"


def product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension <= 0 or result > (1 << 63) // dimension:
            raise ValueError(f"invalid or overflowing shape: {shape}")
        result *= dimension
    return result


def tensor_shapes(c: dict[str, object]) -> dict[str, tuple[int, ...]]:
    hidden = int(c["hidden_size"])
    intermediate = int(c["intermediate_size"])
    vocab = int(c["vocab_size"])
    attn_heads = int(c["num_attention_heads"])
    kv_heads = int(c["num_key_value_heads"])
    head_dim = int(c["head_dim"])
    key_heads = int(c["linear_num_key_heads"])
    value_heads = int(c["linear_num_value_heads"])
    key_head_dim = int(c["linear_key_head_dim"])
    value_head_dim = int(c["linear_value_head_dim"])
    conv_kernel = int(c["linear_conv_kernel_dim"])
    key_dim = key_heads * key_head_dim
    value_dim = value_heads * value_head_dim
    conv_dim = key_dim * 2 + value_dim

    shapes: dict[str, tuple[int, ...]] = {
        "lm_head.weight": (vocab, hidden),
        "model.language_model.embed_tokens.weight": (vocab, hidden),
        "model.language_model.norm.weight": (hidden,),
    }
    layer_types = c["layer_types"]
    if not isinstance(layer_types, list) or len(layer_types) != int(c["num_hidden_layers"]):
        raise ValueError("contract layer_types does not match num_hidden_layers")
    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (intermediate, hidden)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (intermediate, hidden)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, intermediate)
        if layer_type == "linear_attention":
            linear = f"{prefix}.linear_attn"
            shapes[f"{linear}.A_log"] = (value_heads,)
            shapes[f"{linear}.conv1d.weight"] = (conv_dim, 1, conv_kernel)
            shapes[f"{linear}.dt_bias"] = (value_heads,)
            shapes[f"{linear}.in_proj_a.weight"] = (value_heads, hidden)
            shapes[f"{linear}.in_proj_b.weight"] = (value_heads, hidden)
            shapes[f"{linear}.in_proj_qkv.weight"] = (conv_dim, hidden)
            shapes[f"{linear}.in_proj_z.weight"] = (value_dim, hidden)
            shapes[f"{linear}.norm.weight"] = (value_head_dim,)
            shapes[f"{linear}.out_proj.weight"] = (hidden, value_dim)
        elif layer_type == "full_attention":
            attention = f"{prefix}.self_attn"
            # attn_output_gate=true stores query and output-gate projections
            # together, hence twice the logical query width.
            shapes[f"{attention}.q_proj.weight"] = (attn_heads * head_dim * 2, hidden)
            shapes[f"{attention}.k_proj.weight"] = (kv_heads * head_dim, hidden)
            shapes[f"{attention}.v_proj.weight"] = (kv_heads * head_dim, hidden)
            shapes[f"{attention}.o_proj.weight"] = (hidden, attn_heads * head_dim)
            shapes[f"{attention}.q_norm.weight"] = (head_dim,)
            shapes[f"{attention}.k_norm.weight"] = (head_dim,)
        else:
            raise ValueError(f"unsupported layer type: {layer_type!r}")

    mtp = "mtp.layers.0"
    shapes.update({
        "mtp.fc.weight": (hidden, hidden * 2),
        "mtp.norm.weight": (hidden,),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
        f"{mtp}.input_layernorm.weight": (hidden,),
        f"{mtp}.post_attention_layernorm.weight": (hidden,),
        f"{mtp}.mlp.gate_proj.weight": (intermediate, hidden),
        f"{mtp}.mlp.up_proj.weight": (intermediate, hidden),
        f"{mtp}.mlp.down_proj.weight": (hidden, intermediate),
        f"{mtp}.self_attn.q_proj.weight": (attn_heads * head_dim * 2, hidden),
        f"{mtp}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
        f"{mtp}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
        f"{mtp}.self_attn.o_proj.weight": (hidden, attn_heads * head_dim),
        f"{mtp}.self_attn.q_norm.weight": (head_dim,),
        f"{mtp}.self_attn.k_norm.weight": (head_dim,),
    })
    return shapes


class XorShift64Star:
    def __init__(self, seed: int) -> None:
        if seed <= 0 or seed > MASK64:
            raise ValueError("seed must be a non-zero UInt64")
        self.state = seed

    def next_f32(self) -> bytes:
        value = self.state
        value ^= value >> 12
        value ^= (value << 25) & MASK64
        value ^= value >> 27
        self.state = value & MASK64
        bits = ((self.state * 0x2545F4914F6CDD1D) & MASK64) >> 40
        number = (bits / float(1 << 23) - 1.0) * 0.05
        return struct.pack("<f", number)


def safetensors_bytes(shapes: dict[str, tuple[int, ...]], seed: int) -> bytes:
    header: dict[str, object] = {
        "__metadata__": {
            "fixture": "qwen3_8_hybrid_mini_v1",
            "generator": "seen-qwen-qwn-023b-v1",
            "seed": str(seed),
        }
    }
    offset = 0
    for name in sorted(shapes):
        length = product(shapes[name]) * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(shapes[name]),
            "data_offsets": [offset, offset + length],
        }
        offset += length
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    rng = XorShift64Star(seed)
    payload = bytearray(offset)
    cursor = 0
    while cursor < offset:
        payload[cursor : cursor + 4] = rng.next_f32()
        cursor += 4
    return struct.pack("<Q", len(encoded)) + encoded + payload


def bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(visible, map(chr, codepoints), strict=True))


def write_atomic(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def generate(output: Path) -> None:
    contract_bytes = CONTRACT_PATH.read_bytes()
    if sha256(contract_bytes).hexdigest() != CONTRACT_SHA256:
        raise ValueError("QWN-023A contract hash does not match the generator lock")
    contract = json.loads(contract_bytes)
    shapes = tensor_shapes(contract)
    if len(shapes) != 124:
        raise ValueError(f"expected exactly 124 tensors, got {len(shapes)}")

    mapping = bytes_to_unicode()
    vocab = {mapping[byte]: byte for byte in range(256)}
    assets = {
        "model.safetensors": safetensors_bytes(shapes, int(contract["deterministic_seed"])),
        "vocab.json": (json.dumps(vocab, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        "merges.txt": b"#version: 0.2\n",
        "tokenizer_config.json": (json.dumps({
            "add_prefix_space": False,
            "clean_up_tokenization_spaces": False,
            "fixture_id": contract["fixture_id"],
            "model_max_length": contract["max_position_embeddings"],
            "model_type": "byte_level_bpe",
            "special_tokens": [],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    }
    manifest = {
        "schema": "seen-qwen-hybrid-mini-assets-v1",
        "fixture_id": contract["fixture_id"],
        "contract_sha256": CONTRACT_SHA256,
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {
            "model_revision": MODEL_REVISION,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "modeling_source_sha256": MODELING_SOURCE_SHA256,
            "official_shard_01_header_sha256": SHARD_01_HEADER_SHA256,
            "official_shard_18_header_sha256": SHARD_18_HEADER_SHA256,
        },
        "safetensors": {
            "dtype": "F32",
            "tensor_count": len(shapes),
            "payload_bytes": sum(product(shape) * 4 for shape in shapes.values()),
        },
        "tokenizer": {"vocabulary_size": 256, "merge_count": 0},
        "assets": {
            name: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in sorted(assets.items())
        },
    }
    assets["manifest.json"] = (json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode()

    if output in (Path("/"), ROOT, ROOT.parent) or output.is_symlink():
        raise ValueError("unsafe output directory")
    if output.exists() and not output.is_dir():
        raise ValueError("output path must be a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    backup: Path | None = None
    try:
        for name, data in assets.items():
            write_atomic(temporary / name, data)
        temporary_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        if output.exists():
            backup = Path(tempfile.mkdtemp(
                prefix=f".{output.name}.previous.", dir=output.parent
            ))
            backup.rmdir()
            os.replace(output, backup)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except BaseException:
        if backup is not None and not output.exists():
            os.replace(backup, output)
            backup = None
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir.absolute())


if __name__ == "__main__":
    main()
