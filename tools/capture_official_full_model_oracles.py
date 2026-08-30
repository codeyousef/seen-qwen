#!/usr/bin/env python3
"""Capture QWN-025B full-model logits and 128-step greedy sequences.

This is a bounded CPU O1 reference runner.  It constructs one exact official
decoder layer at a time from verified local Safetensors shards, retaining only
the hybrid decode cache between layers.  It never imports model-repository
code and never makes a network request.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.fetch_official_full_model import INDEX_SHA256, MODEL_ID, MODEL_REVISION, SHARDS, shard_name


TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
MODELING_SHA256 = "25c4912dc14dda47b14a1c24efe36ec055be4a2f150c64c9a29860aebe42aff8"
TORCH_VERSION = "2.11.0+cpu"
TRANSFORMERS_VERSION = "5.16.0.dev0"
CONFIG = ROOT / "tests/fixtures/qwen3_8_config.json"
CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
ASSETS = ROOT / ".seen/oracle-assets-qwen38"
DEFAULT_INPUT = ROOT / ".seen/oracle-official/full-model"
DEFAULT_JSON = ROOT / "tests/fixtures/qwn_025b_full_model_oracles.json"
DEFAULT_LOGITS = ROOT / "tests/fixtures/qwn_025b_full_model_logits.safetensors"
MAX_PROMPT_TOKENS = 256
GREEDY_STEPS = 128
GREEDY_PROMPT_IDS = ("thinking_on", "thinking_off")
EXPECTED_PAD_EOS_ID = 248_044

ASSET_LOCKS = {
    "vocab.json": (6_722_759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
    "merges.txt": (3_353_259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    "tokenizer_config.json": (17_928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    "chat_template.jinja": (8_952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "generation_config.json": (202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
    "config.json": (4_312, CONFIG_SHA256),
}

PROMPTS: tuple[dict[str, Any], ...] = (
    {"id": "minimal", "messages": [{"role": "user", "content": ""}], "enable_thinking": False},
    {
        "id": "english",
        "messages": [{"role": "user", "content": "Explain why deterministic tests matter in one paragraph."}],
        "enable_thinking": False,
    },
    {
        "id": "arabic",
        "messages": [{"role": "user", "content": "اشرح بإيجاز لماذا الاختبارات الحتمية مهمة."}],
        "enable_thinking": False,
    },
    {
        "id": "code",
        "messages": [{"role": "user", "content": "Write a Seen function that returns the larger of two Int values."}],
        "enable_thinking": False,
    },
    {
        "id": "long_repeated",
        "messages": [{
            "role": "user",
            "content": "Summarize this ordered structure without dropping an item:\n"
            + "\n".join(f"{index}: alpha" for index in range(1, 13)),
        }],
        "enable_thinking": False,
    },
    {
        "id": "tool_chat",
        "messages": [
            {"role": "user", "content": "What is the weather in Riyadh? Use the tool."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "weather", "arguments": {"city": "Riyadh"}},
                }],
            },
            {"role": "tool", "content": '{"temperature_c": 34, "condition": "clear"}'},
        ],
        "enable_thinking": False,
        "has_tool_context": True,
    },
    {
        "id": "thinking_on",
        "messages": [{"role": "user", "content": "Which is larger: 17 times 19 or 18 times 18?"}],
        "enable_thinking": True,
        "reasoning_effort": "low",
    },
    {
        "id": "thinking_off",
        "messages": [{"role": "user", "content": "Which is larger: 17 times 19 or 18 times 18?"}],
        "enable_thinking": False,
    },
)


class CaptureError(RuntimeError):
    """Stable fail-closed QWN-025B capture diagnostic."""


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_deadline(deadline: float, operation: str) -> None:
    if time.monotonic() >= deadline:
        raise CaptureError(f"deadline exceeded before {operation}")


def validate_inputs(input_root: Path, deadline: float) -> tuple[dict[str, str], str]:
    if digest(CONFIG) != CONFIG_SHA256 or digest(INDEX) != INDEX_SHA256:
        raise CaptureError("checked-in config or tensor index digest mismatch")
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CaptureError("verified full-model input manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "seen-qwen-official-full-model-inputs-v1":
        raise CaptureError("full-model input manifest schema mismatch")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise CaptureError("full-model input identity mismatch")
    expected = {
        shard_name(shard): {"bytes": size, "lfs_sha256": identity}
        for shard, (size, identity) in SHARDS.items()
    }
    actual = {entry.get("path"): {"bytes": entry.get("bytes"), "lfs_sha256": entry.get("lfs_sha256")} for entry in manifest.get("shards", [])}
    if actual != expected:
        raise CaptureError("full-model input manifest shard set mismatch")
    for name, record in expected.items():
        check_deadline(deadline, f"input verification {name}")
        print(f"verifying {name}", flush=True)
        path = input_root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise CaptureError(f"missing or unsafe locked shard: {name}")
        if digest(path) != record["lfs_sha256"]:
            raise CaptureError(f"locked shard digest mismatch: {name}")
    for name, (size, identity) in ASSET_LOCKS.items():
        path = ASSETS / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size or digest(path) != identity:
            raise CaptureError(f"tokenizer asset identity mismatch: {name}")
    index = json.loads(INDEX.read_text(encoding="utf-8"))["weight_map"]
    return index, digest(manifest_path)


def get_tensor(input_root: Path, index: dict[str, str], name: str) -> torch.Tensor:
    filename = index.get(name)
    if filename is None:
        raise CaptureError(f"tensor is absent from checked index: {name}")
    with safe_open(input_root / filename, framework="pt", device="cpu") as source:
        return source.get_tensor(name)


def embedding_rows(
    input_root: Path, index: dict[str, str], token_ids: torch.Tensor
) -> torch.Tensor:
    name = "model.language_model.embed_tokens.weight"
    filename = index.get(name)
    if filename is None:
        raise CaptureError("embedding tensor is absent from checked index")
    rows = []
    with safe_open(input_root / filename, framework="pt", device="cpu") as source:
        weights = source.get_slice(name)
        for token_id in token_ids.reshape(-1).tolist():
            if token_id < 0 or token_id >= 248_320:
                raise CaptureError(f"token ID outside embedding bounds: {token_id}")
            rows.append(weights[token_id : token_id + 1])
    return torch.cat(rows, dim=0).reshape(*token_ids.shape, 5120)


def load_layer(
    input_root: Path, index: dict[str, str], layer: int
) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer}."
    names = sorted(name for name in index if name.startswith(prefix))
    if not names:
        raise CaptureError(f"layer {layer} is absent from checked index")
    state: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(index[name], []).append(name)
    for filename, shard_names in sorted(by_file.items()):
        with safe_open(input_root / filename, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            for name in shard_names:
                if name not in keys:
                    raise CaptureError(f"checked tensor missing from shard: {name}")
                state[name[len(prefix) :]] = source.get_tensor(name)
    return state


def build_prompts(tokenizer) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    records = []
    sequences = []
    for prompt in PROMPTS:
        options = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": prompt["enable_thinking"],
        }
        if "reasoning_effort" in prompt:
            options["reasoning_effort"] = prompt["reasoning_effort"]
        if "tools" in prompt:
            options["tools"] = prompt["tools"]
        token_ids = tokenizer.apply_chat_template(prompt["messages"], **options)
        if hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]
        if not isinstance(token_ids, list) or not token_ids or len(token_ids) > MAX_PROMPT_TOKENS:
            raise CaptureError(f"prompt token bound failed: {prompt['id']}")
        rendered_options = dict(options)
        rendered_options["tokenize"] = False
        rendered = tokenizer.apply_chat_template(prompt["messages"], **rendered_options)
        sequences.append(token_ids)
        records.append({
            "id": prompt["id"],
            "category": prompt["id"],
            "enable_thinking": prompt["enable_thinking"],
            "reasoning_effort": prompt.get("reasoning_effort"),
            "has_tools": "tools" in prompt or prompt.get("has_tool_context", False),
            "rendered_utf8_sha256": sha256(rendered.encode("utf-8")).hexdigest(),
            "input_token_ids": token_ids,
            "input_token_count": len(token_ids),
        })
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), EXPECTED_PAD_EOS_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, -len(sequence) :] = 1
    return records, input_ids, attention_mask


def stream_layers(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
    cache,
    config,
    modeling,
    input_root: Path,
    index: dict[str, str],
    deadline: float,
    phase: str,
) -> torch.Tensor:
    mask_kwargs = {
        "config": config,
        "inputs_embeds": hidden,
        "attention_mask": attention_mask,
        "past_key_values": cache,
        "position_ids": positions,
    }
    masks = {
        "full_attention": modeling.create_causal_mask(**mask_kwargs),
        "linear_attention": modeling.create_recurrent_attention_mask(**mask_kwargs),
    }
    rotary = modeling.Qwen3_5TextRotaryEmbedding(config)
    rotary.eval()
    rope_positions = positions.unsqueeze(0).expand(3, *positions.shape)
    position_embeddings = rotary(hidden, rope_positions)
    del rotary
    for layer in range(config.num_hidden_layers):
        check_deadline(deadline, f"layer {layer}")
        if layer % 16 == 0 or layer + 1 == config.num_hidden_layers:
            print(f"{phase}: layer {layer + 1}/{config.num_hidden_layers}", flush=True)
        state = load_layer(input_root, index, layer)
        with torch.device("meta"):
            module = modeling.Qwen3_5DecoderLayer(config, layer)
        missing, unexpected = module.load_state_dict(state, strict=True, assign=True)
        if missing or unexpected:
            raise CaptureError(f"layer {layer} state mismatch: missing={missing}, unexpected={unexpected}")
        module.eval()
        hidden = module(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=masks[config.layer_types[layer]],
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
        del module, state
        gc.collect()
    return hidden


def capture(
    input_root: Path,
    output_json: Path,
    output_logits: Path,
    deadline_seconds: int,
) -> None:
    started = time.monotonic()
    deadline = started + deadline_seconds
    index, input_manifest_sha = validate_inputs(input_root, deadline)
    check_deadline(deadline, "source import")

    import transformers
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    if torch.__version__ != TORCH_VERSION or transformers.__version__ != TRANSFORMERS_VERSION:
        raise CaptureError(
            f"oracle environment mismatch: torch={torch.__version__} transformers={transformers.__version__}"
        )
    if digest(Path(modeling.__file__).resolve()) != MODELING_SHA256:
        raise CaptureError("official modeling source digest mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise CaptureError("QWN-025B O1 capture must remain CPU-only")

    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(ASSETS), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.convert_tokens_to_ids("<|endoftext|>") != EXPECTED_PAD_EOS_ID:
        raise CaptureError("locked padding/EOS token identity mismatch")
    prompt_records, input_ids, attention_mask = build_prompts(tokenizer)
    config_document = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = Qwen3_5TextConfig.from_dict(config_document["text_config"])
    config._attn_implementation = "eager"
    if config.num_hidden_layers != 64 or config.vocab_size != 248_320:
        raise CaptureError("full-model geometry changed")
    cache = DynamicCache(config=config)

    with torch.inference_mode():
        hidden = embedding_rows(input_root, index, input_ids)
        positions = attention_mask.cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        hidden = stream_layers(
            hidden, attention_mask, positions, cache, config, modeling, input_root, index, deadline, "prefill"
        )
        final_norm = modeling.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        final_norm.load_state_dict(
            {"weight": get_tensor(input_root, index, "model.language_model.norm.weight")},
            strict=True,
            assign=True,
        )
        hidden = final_norm(hidden[:, -1:, :])[:, 0, :]
        del final_norm
        lm_head = get_tensor(input_root, index, "lm_head.weight")
        if tuple(lm_head.shape) != (248_320, 5120) or lm_head.dtype != torch.bfloat16:
            raise CaptureError("LM head geometry or dtype mismatch")
        logits = F.linear(hidden, lm_head)
        initial_logits = {f"{record['id']}.next_logits_bf16": logits[row].contiguous() for row, record in enumerate(prompt_records)}
        next_values, next_ids = torch.topk(logits.float(), k=5, dim=-1, largest=True, sorted=True)

        for row, record in enumerate(prompt_records):
            record["initial_top5_token_ids"] = next_ids[row].tolist()
            record["initial_top5_logits_f32"] = next_values[row].tolist()
            record["greedy_sequence_role"] = (
                "128-step" if record["id"] in GREEDY_PROMPT_IDS else "logits-only"
            )

        greedy_rows = [
            row for row, record in enumerate(prompt_records) if record["id"] in GREEDY_PROMPT_IDS
        ]
        if [prompt_records[row]["id"] for row in greedy_rows] != list(GREEDY_PROMPT_IDS):
            raise CaptureError("greedy prompt selection changed")
        greedy_index = torch.tensor(greedy_rows, dtype=torch.long)
        cache.reorder_cache(greedy_index)
        attention_mask = attention_mask.index_select(0, greedy_index)
        logits = logits.index_select(0, greedy_index)
        greedy_records = [prompt_records[row] for row in greedy_rows]
        generated: list[list[int]] = [[] for _ in greedy_records]
        selected_logits: list[list[float]] = [[] for _ in greedy_records]

        for step in range(GREEDY_STEPS):
            check_deadline(deadline, f"greedy step {step}")
            print(f"greedy: step {step + 1}/{GREEDY_STEPS}", flush=True)
            token_ids = torch.argmax(logits, dim=-1)
            chosen = logits.gather(1, token_ids[:, None]).float()[:, 0]
            for row, token_id in enumerate(token_ids.tolist()):
                generated[row].append(token_id)
                selected_logits[row].append(float(chosen[row].item()))
            if step + 1 == GREEDY_STEPS:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)], dim=1
            )
            positions = attention_mask.sum(dim=-1, keepdim=True) - 1
            hidden = embedding_rows(input_root, index, token_ids[:, None])
            hidden = stream_layers(
                hidden,
                attention_mask,
                positions,
                cache,
                config,
                modeling,
                input_root,
                index,
                deadline,
                f"decode {step + 2}/{GREEDY_STEPS}",
            )
            final_norm = modeling.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            final_norm.load_state_dict(
                {"weight": get_tensor(input_root, index, "model.language_model.norm.weight")},
                strict=True,
                assign=True,
            )
            hidden = final_norm(hidden)[:, 0, :]
            logits = F.linear(hidden, lm_head)
            del final_norm
        for row, record in enumerate(greedy_records):
            record["greedy_token_ids"] = generated[row]
            record["greedy_selected_logits_f32"] = selected_logits[row]
            decoded = tokenizer.decode(generated[row], skip_special_tokens=False)
            record["greedy_decoded_utf8_sha256"] = sha256(decoded.encode("utf-8")).hexdigest()
            eos_steps = [index for index, token_id in enumerate(generated[row]) if token_id == EXPECTED_PAD_EOS_ID]
            record["first_eos_step"] = eos_steps[0] if eos_steps else None

    output_logits.parent.mkdir(parents=True, exist_ok=True)
    temporary_logits = output_logits.with_name(f".{output_logits.name}.{os.getpid()}.tmp")
    save_file(initial_logits, temporary_logits, metadata={
        "schema": "seen-qwen-full-model-logits-v1",
        "model_revision": MODEL_REVISION,
        "dtype": "BF16",
    })
    os.replace(temporary_logits, output_logits)
    document = {
        "schema": "seen-qwen-full-model-oracles-v1",
        "classification": "verified-cpu-reference",
        "source": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": CONFIG_SHA256,
            "tensor_index_sha256": INDEX_SHA256,
            "input_manifest_sha256": input_manifest_sha,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "modeling_source_sha256": MODELING_SHA256,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "capture_tool_sha256": digest(Path(__file__).resolve()),
            "fetch_tool_sha256": digest(ROOT / "tools/fetch_official_full_model.py"),
        },
        "determinism": {
            "device": "cpu",
            "workers": 1,
            "algorithms": "torch.use_deterministic_algorithms(true)",
            "attention": "eager",
            "weight_loading": "one exact official decoder layer at a time",
            "greedy_steps": GREEDY_STEPS,
            "greedy_prompt_ids": list(GREEDY_PROMPT_IDS),
            "padding_side": "left",
            "padding_token_id": EXPECTED_PAD_EOS_ID,
        },
        "logits": {
            "path": output_logits.name,
            "sha256": digest(output_logits),
            "dtype": "BF16",
            "shape_per_prompt": [248_320],
            "tensor_count": len(initial_logits),
        },
        "tolerance_contract": {
            "path": "tests/tolerances.toml",
            "higher_precision_mean_logit_cosine_min": 0.9995,
            "higher_precision_top5_set_overlap_min": 0.99,
            "greedy_compare_steps": GREEDY_STEPS,
            "capture_first_divergence": True,
            "nan_inf_allowed": False,
        },
        "prompts": prompt_records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_name(f".{output_json.name}.{os.getpid()}.tmp")
    temporary_json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_json, output_json)
    print(
        f"wrote {output_json} sha256={digest(output_json)} "
        f"logits_sha256={digest(output_logits)} elapsed_seconds={time.monotonic() - started:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--deadline-seconds", type=int, default=28_800)
    arguments = parser.parse_args()
    if arguments.deadline_seconds <= 0 or arguments.deadline_seconds > 28_800:
        raise CaptureError("deadline-seconds must be in 1..28800")
    signal.signal(signal.SIGTERM, lambda _signal, _frame: (_ for _ in ()).throw(CaptureError("cancelled")))
    capture(
        arguments.input_root.resolve(),
        arguments.output_json.resolve(),
        arguments.output_logits.resolve(),
        arguments.deadline_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"qwn-025b capture failed: {error}", file=sys.stderr)
        raise
