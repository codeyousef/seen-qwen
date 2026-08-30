# Full-model oracle and tolerance contract

FEL-1411 / QWN-025B fixes the official text-model prompt, next-token logit,
greedy-sequence, and numerical-tolerance contract. The source is the immutable
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, exact Transformers commit
`562cfd944ee1f20702cfb0f4404014ee27c24813`, and CPU-only PyTorch
`2.11.0+cpu`. Model-repository code is never downloaded or executed.

The prompt corpus covers minimal, English, Arabic, Seen code, repeated ordered
structure, prior tool-call/tool-response chat, thinking-on, and thinking-off
inputs. Every rendered prompt is bounded to 256 tokens. The checked JSON
manifest records exact rendered hashes and token IDs and initial top-5 values
for all prompts. The matched thinking-on/thinking-off pair additionally records
128 greedy token transitions, selected-token logits, EOS position, and decoded
output hashes. The companion Safetensors corpus stores one complete BF16
248,320-element initial next-token logit vector per prompt. Both artifacts are
content addressed.

Official shards are never checked in. `tools/fetch_official_full_model.py`
downloads the eighteen complete files serially into ignored
`.seen/oracle-official/full-model/`, validates the checked tensor index, exact
file sizes, and every immutable LFS SHA-256, then atomically promotes them.
Complete shards are acquired because tensors cross shard boundaries; the
capture rejects visual tensors and reads only `model.language_model.*` plus
`lm_head.weight`.

`tools/capture_official_full_model_oracles.py` re-hashes all inputs and locked
tokenizer assets, uses deterministic eager CPU math with one worker, and
constructs only one decoder layer at a time. The parsed source documents,
tokenizer, hybrid attention cache, and LM head remain owned until the last
borrowed tensor/token result is consumed. Layer modules and mapped tensor views
are released deterministically after every layer; output promotion is atomic.
Cancellation and the eight-hour deadline fail closed and cannot promote a
partial oracle.

`tests/tolerances.toml` is authoritative. FP32 CPU operator comparison is at
most `1e-5` absolute and relative error. A BF16-semantics full-model path needs
mean logit cosine at least `0.9995`, top-5 set overlap at least `0.99`, and 128
greedy transitions with the first divergence recorded. NaN or infinity always
fails. Quantized paths use their separately named thresholds and never inherit
the BF16 classification silently.

This corpus is `verified-cpu-reference`. It is not CUDA evidence, a performance
claim, a quantization waiver, or release-consumer certification.
