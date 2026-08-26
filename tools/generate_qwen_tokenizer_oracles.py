#!/usr/bin/env python3
"""Generate deterministic Qwen3.8 tokenizer and chat-template oracle vectors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRANSFORMERS_COMMIT = "562cfd944ee1f20702cfb0f4404014ee27c24813"
TRANSFORMERS_ARCHIVE_SHA256 = (
    "7a8311b63affb125e81e3f4c52e2376590f7c20824df95c128bd4a536fc22438"
)
QWEN2_TOKENIZER_SOURCE_SHA256 = (
    "fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628"
)
MAX_ASSET_BYTES = 8 * 1024 * 1024
TRANSFORMERS_ARCHIVE_BYTES = 20_763_604
MAX_TRANSFORMERS_ARCHIVE_BYTES = 32 * 1024 * 1024
QWEN2_TOKENIZER_SOURCE_BYTES = 3_323
MAX_TRANSFORMERS_SOURCE_BYTES = 1024 * 1024

ASSET_LOCKS = {
    "vocab.json": (6_722_759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
    "merges.txt": (3_353_259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    "tokenizer_config.json": (17_928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    "chat_template.jinja": (8_952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "generation_config.json": (202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
    "config.json": (4_312, "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locked_assets(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, (expected_size, expected_sha) in ASSET_LOCKS.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe locked asset: {name}")
        size = path.stat().st_size
        if size > MAX_ASSET_BYTES or size != expected_size:
            raise ValueError(f"locked asset size mismatch: {name}")
        digest = sha256_bytes(path.read_bytes())
        if digest != expected_sha:
            raise ValueError(f"locked asset SHA-256 mismatch: {name}")
        result[name] = {"bytes": size, "sha256": digest}
    return result


def verify_transformers_source(source_root: Path) -> dict[str, Any]:
    tokenizer_source = source_root / "src/transformers/models/qwen2/tokenization_qwen2.py"
    archive = source_root.parent / "source.tar.gz"
    if not tokenizer_source.is_file() or tokenizer_source.is_symlink():
        raise ValueError("missing or unsafe pinned Qwen2 tokenizer source")
    source_size = tokenizer_source.stat().st_size
    if source_size > MAX_TRANSFORMERS_SOURCE_BYTES or source_size != QWEN2_TOKENIZER_SOURCE_BYTES:
        raise ValueError("pinned Qwen2 tokenizer source size mismatch")
    source_sha = sha256_bytes(tokenizer_source.read_bytes())
    if source_sha != QWEN2_TOKENIZER_SOURCE_SHA256:
        raise ValueError("pinned Qwen2 tokenizer source SHA-256 mismatch")
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("missing or unsafe pinned Transformers archive")
    archive_size = archive.stat().st_size
    if archive_size > MAX_TRANSFORMERS_ARCHIVE_BYTES or archive_size != TRANSFORMERS_ARCHIVE_BYTES:
        raise ValueError("pinned Transformers archive size mismatch")
    archive_sha = sha256_bytes(archive.read_bytes())
    if archive_sha != TRANSFORMERS_ARCHIVE_SHA256:
        raise ValueError("pinned Transformers archive SHA-256 mismatch")
    return {
        "commit": TRANSFORMERS_COMMIT,
        "archive_bytes": archive_size,
        "archive_sha256": archive_sha,
        "qwen2_tokenizer_source_bytes": source_size,
        "qwen2_tokenizer_source_sha256": source_sha,
    }


def text_vector(tokenizer: Any, name: str, text: str) -> dict[str, Any]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "name": name,
        "text": text,
        "utf8_sha256": sha256_bytes(text.encode("utf-8")),
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded": tokenizer.decode(token_ids, skip_special_tokens=False),
    }


def chat_vector(
    tokenizer: Any,
    name: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return {
        "name": name,
        "messages": messages,
        "options": kwargs,
        "rendered": rendered,
        "rendered_utf8_sha256": sha256_bytes(rendered.encode("utf-8")),
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "decoded": tokenizer.decode(token_ids, skip_special_tokens=False),
    }


def rejected_chat_vector(
    tokenizer: Any,
    name: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    except Exception as error:  # The oracle records the pinned template type/text.
        return {
            "name": name,
            "messages": messages,
            "options": kwargs,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    raise ValueError(f"negative chat oracle unexpectedly succeeded: {name}")


def build_document(assets_root: Path, transformers_source_root: Path) -> dict[str, Any]:
    assets = locked_assets(assets_root)
    source = verify_transformers_source(transformers_source_root)
    tokenizer = AutoTokenizer.from_pretrained(
        str(assets_root),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    if type(tokenizer).__name__ != "Qwen2Tokenizer":
        raise ValueError("unexpected tokenizer implementation")

    texts = [
        text_vector(tokenizer, "ascii", "Hello, world!"),
        text_vector(tokenizer, "arabic", "مرحبًا بالعالم"),
        text_vector(tokenizer, "unicode_combining", "café e\u0301 中文 🚀"),
        text_vector(tokenizer, "seen_code", "fun main() r: Int {\n    return 0\n}\n"),
        text_vector(tokenizer, "whitespace", " \t\n\n  "),
        text_vector(tokenizer, "special_literal", "<|im_start|>user\nhello<|im_end|>\n"),
        text_vector(tokenizer, "bounded_repetition_1024", "a" * 1024),
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    chats = [
        chat_vector(
            tokenizer,
            "user_xhigh_generation_prompt",
            [{"role": "user", "content": "Explain checked addition."}],
            add_generation_prompt=True,
        ),
        chat_vector(
            tokenizer,
            "system_user_low",
            [
                {"role": "system", "content": "Answer in Arabic."},
                {"role": "user", "content": "ما معنى الذاكرة؟"},
            ],
            add_generation_prompt=True,
            reasoning_effort="low",
        ),
        chat_vector(
            tokenizer,
            "thinking_disabled",
            [{"role": "user", "content": "Return 2 + 2."}],
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        chat_vector(
            tokenizer,
            "preserve_thinking_false",
            [
                {"role": "user", "content": "First question"},
                {
                    "role": "assistant",
                    "reasoning_content": "private prior reasoning",
                    "content": "First answer",
                },
                {"role": "user", "content": "Second question"},
            ],
            add_generation_prompt=True,
            preserve_thinking=False,
        ),
        chat_vector(
            tokenizer,
            "tool_definition_call_response",
            [
                {"role": "user", "content": "Weather in Riyadh?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Need current weather.",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "Riyadh"}},
                    }],
                },
                {"role": "tool", "content": '{"temperature_c": 38}'},
            ],
            add_generation_prompt=True,
            tools=tools,
        ),
    ]
    rejected = [
        rejected_chat_vector(tokenizer, "no_messages", [], add_generation_prompt=True),
        rejected_chat_vector(
            tokenizer,
            "invalid_reasoning_effort",
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
            reasoning_effort="invalid",
        ),
        rejected_chat_vector(
            tokenizer,
            "late_system_message",
            [
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Too late"},
            ],
            add_generation_prompt=True,
        ),
    ]

    special_tokens = [
        {"token": token, "id": token_id}
        for token, token_id in sorted(tokenizer.get_added_vocab().items(), key=lambda item: item[1])
    ]
    document: dict[str, Any] = {
        "schema": "seen-qwen-tokenizer-oracle-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source": source,
        "runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "jinja2": importlib.metadata.version("jinja2"),
            "implementation": type(tokenizer).__name__,
        },
        "assets": assets,
        "tokenizer_contract": {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token": tokenizer.unk_token,
            "unk_token_id": tokenizer.unk_token_id,
            "special_tokens": special_tokens,
        },
        "text_vectors": texts,
        "chat_vectors": chats,
        "rejected_chat_vectors": rejected,
    }
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document["payload_sha256"] = sha256_bytes(canonical.encode("utf-8"))
    return document


def atomic_write_json(output: Path, document: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > 1024 * 1024:
        raise ValueError("oracle output exceeds the 1 MiB bound")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--transformers-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_document(args.assets_root.resolve(), args.transformers_source_root.resolve())
    atomic_write_json(args.output.resolve(), document)
    print(f"wrote {args.output}: {document['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
