#!/usr/bin/env python3
"""Generate a small complete 866-entry SQW for QWN-047C smoke verification."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts/oracle/build_qwn_034a_bf16_artifact.py"
INDEX = ROOT / "tests/fixtures/qwen3_8_model.safetensors.index.json"
POLICY = ROOT / "configs/quantization.toml"
OUTPUT = ROOT / ".seen/evidence/FEL-1455/smoke"


def load_builder():
    spec = importlib.util.spec_from_file_location("qwn_034a_smoke_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    builder = load_builder()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = OUTPUT / "source.bf16"
    source.write_bytes(b"\0\0")
    digest = sha256(b"\0\0").hexdigest()
    index, _ = builder.read_index(INDEX)
    tensors = []
    for name in sorted(
        (candidate for candidate in index if builder.is_included(candidate)),
        key=lambda candidate: candidate.encode("utf-8"),
    ):
        role_id, role_name = builder.semantic_role(name)
        tensor = builder.SourceTensor(name, source, 0, 2, (1,), role_id, role_name)
        tensor.source_sha256 = digest
        tensors.append(tensor)
    if len(tensors) != builder.INCLUDED_TENSORS:
        raise AssertionError("smoke catalog does not contain exactly 866 entries")

    model_lock_bytes = b'{"schema":"qwn-047c-smoke-model-lock"}\n'
    source_lock_bytes = b'{"schema":"qwn-047c-smoke-source-lock"}\n'
    (OUTPUT / "model-lock.json").write_bytes(model_lock_bytes)
    (OUTPUT / "source-lock.json").write_bytes(source_lock_bytes)
    model_lock = sha256(model_lock_bytes).hexdigest()
    source_lock = sha256(source_lock_bytes).hexdigest()
    policy = builder.validate_policy(POLICY)
    evidence = builder.conversion_evidence(
        tensors, model_lock, source_lock, policy, len(tensors) * 2
    )
    layout = builder.compute_layout(tensors, model_lock, source_lock, policy, evidence)
    weights = OUTPUT / "weights.sqw"
    whole, conventional = builder.write_sqw(
        weights, tensors, layout, model_lock, policy, builder.MAX_CHUNK_BYTES
    )
    builder.validate_sqw(
        weights,
        tensors,
        layout,
        model_lock,
        policy,
        whole,
        conventional,
        builder.MAX_CHUNK_BYTES,
    )
    assets = [
        {"path": name, "bytes": str(identity[0]), "sha256": identity[1]}
        for name, identity in builder.ASSET_IDENTITIES.items()
    ]
    document = builder.engine_document(
        model_lock,
        source_lock,
        policy,
        conventional,
        whole,
        layout.file_bytes,
        assets,
    )
    (OUTPUT / "engine.json").write_bytes(builder.canonical_json(document) + b"\n")
    print(OUTPUT / "engine.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
