# Qwen hybrid mini-model architecture contract

The deterministic CPU/CUDA integration fixture is frozen by
`tests/fixtures/qwen3_8_hybrid_mini_contract.json`. It is deliberately not a
Qwen3.8-compatible model and must never be accepted by the full-model loader.
It preserves the required architecture features while remaining small enough
for every-commit evidence:

- eight layers in the exact 3:1 GDN/full-attention schedule;
- six query heads to one KV head and six GDN value heads to two key heads;
- partial, interleaved RoPE; gated attention; causal convolution; float32 GDN
  recurrent state; SwiGLU/SiLU FFN; untied LM head; and one MTP layer;
- the text, LM-head, MTP, and excluded-vision namespace policy;
- deterministic float32 fixture generation with seed `20260827`.

The dimensions are fixture policy, not a claim about the production model:
hidden size 48, intermediate size 160, vocabulary 256, native context 128,
attention head dimension 24, and GDN head dimensions 8. QWN-023B must derive
all generated tensor shapes from this contract and QWN-023C must bind expected
states/logits/tokens to its exact file hash.

The canonical fixture file SHA-256 is
`f29839615771e344bf89329f2195e5921fc8ea371849249be55722ab1999dddf`.

## Deterministic assets

`tools/generate_hybrid_mini_fixture.py` is the bounded, dependency-free oracle
and conversion tool for QWN-023B. It derives all 124 canonical tensor names and
shapes from the frozen contract, sorts names by Unicode code point, emits one
little-endian FP32 Safetensors file, and creates a reversible 256-entry
byte-level BPE vocabulary with an empty merge table. It uses xorshift64* with
the locked seed; no Python, PyTorch, or foreign runtime is part of production.

The checked assets live under `tests/fixtures/qwen3_8_hybrid_mini/` and are
locked by `manifest.json`. The model is 1,466,672 bytes with SHA-256
`16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9`.
Shape derivation is audited against Transformers commit
`562cfd944ee1f20702cfb0f4404014ee27c24813` and the metadata-only headers of
official pinned model shards 1 and 18; their exact hashes are recorded in the
asset manifest. In particular, the gated attention `q_proj` stores twice the
logical query width while `o_proj` consumes the ungated query width.
Generation writes and fsyncs a private temporary directory before atomically
promoting the complete asset directory. CI regenerates every file, compares every byte,
validates header geometry independently, and opens the result through Seen's
released Safetensors reader with deterministic mapped-file cleanup.
