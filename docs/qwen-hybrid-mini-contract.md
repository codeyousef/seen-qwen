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
