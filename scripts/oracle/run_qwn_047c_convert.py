#!/usr/bin/env python3
"""Exact-request bridge for the native Seen QWN-047C convert command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import build_qwn_034a_bf16_artifact as builder


FIELDS = {
    "schema",
    "model_lock",
    "output_root",
    "quant_profile",
    "quantization_config",
    "calibration_lock",
    "jobs",
    "max_host_memory",
    "resume",
    "verify_only",
}


def load_request(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    document = builder.strict_json_bytes(raw, maximum=65536)
    if not isinstance(document, dict) or set(document) != FIELDS:
        raise ValueError("convert request differs from the closed QWN-047C schema")
    if document["schema"] != "seen-qwen-convert-request-v1":
        raise ValueError("convert request schema is incompatible")
    for field in ("model_lock", "output_root", "quant_profile", "quantization_config", "calibration_lock"):
        if not isinstance(document[field], str):
            raise ValueError(f"convert request {field} must be a string")
    if not isinstance(document["jobs"], int) or isinstance(document["jobs"], bool):
        raise ValueError("convert request jobs must be an integer")
    if not isinstance(document["max_host_memory"], int) or isinstance(document["max_host_memory"], bool):
        raise ValueError("convert request max_host_memory must be an integer")
    if not isinstance(document["resume"], bool) or not isinstance(document["verify_only"], bool):
        raise ValueError("convert request flags must be booleans")
    return document


def execute(request: dict[str, object]) -> Path:
    if request["quant_profile"] != "bf16-bringup-v1":
        raise ValueError("QWN-047C supports only the frozen bf16-bringup-v1 conversion profile")
    if request["calibration_lock"]:
        raise ValueError("lossless BF16 conversion does not accept a calibration lock")
    if request["jobs"] != 1:
        raise ValueError("conversion is serial; --jobs must be exactly 1")
    if request["resume"]:
        raise ValueError("conversion resume is unsupported until a journaled restart contract is certified")
    maximum = request["max_host_memory"]
    if maximum < 0 or maximum > 64 * 1024 * 1024 * 1024:
        raise ValueError("max host memory must be zero or no more than 64 GiB")

    model_lock = Path(request["model_lock"]).resolve()
    root = model_lock.parent
    arguments = SimpleNamespace(
        source_root=root,
        asset_root=root,
        index=root / "model.safetensors.index.json",
        model_lock=model_lock,
        policy=Path(request["quantization_config"]).resolve(),
        output_root=Path(request["output_root"]).resolve(),
        chunk_bytes=builder.MAX_CHUNK_BYTES,
        verify_only=request["verify_only"],
    )
    return builder.build(arguments)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    request = load_request(parser.parse_args().request)
    print(execute(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
