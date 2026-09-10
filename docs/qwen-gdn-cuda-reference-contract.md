# Qwen GDN CUDA state, convolution, and recurrent-decode reference contract

FEL-1433 / QWN-041A adds the first Gated DeltaNet CUDA state operation. The
reference path is deliberately narrow: FP32 causal depthwise convolution with
the official Qwen convolution length of four, followed by SiLU. Recurrent
delta-state update, chunk prefill, and gated output belong to later QWN-041
leaves. FEL-1436 / QWN-041B adds the single-token recurrent DeltaNet decode
step. FEL-1435 / QWN-041C extends the same transition to bounded multi-token
chunk prefill; gated output remains a later leaf.

## Layout and semantics

- `input` and `output` are contiguous `[token_count, channels]` FP32 views.
- `weights` is contiguous `[channels, 4]` in oldest-history-to-current order.
- `history` is caller-owned contiguous `[3, channels]` FP32 storage, ordered
  oldest to newest. A successful launch mutates it to the last three raw input
  rows.
- `start_position` must equal the caller's `processed_position`. Position
  addition is checked before launch. The caller advances its host metadata only
  after accepting the launch status.
- Arbitrary sequential chunking is bit-identical to one launch. Exact
  input/output aliasing is supported; every other overlap is rejected.
- The official model boundary is 10,240 channels: two 2,048-wide Q/K
  projections plus one 6,144-wide value projection.

## Recurrent decode semantics

- `query` and `key` are contiguous `[value_heads, key_dim]` FP32 views;
  `value` and `output` are `[value_heads, value_dim]`; `beta` and `log_decay`
  are `[value_heads]`.
- `state` is caller-owned `[value_heads, key_dim, value_dim]` FP32 storage.
  For each head, a successful decode multiplies state by `exp(log_decay)`,
  computes the key-weighted memory, applies `(value - memory) * beta` as the
  rank-one delta, and emits the updated state's query projection.
- `start_position` must equal `processed_position`, and advancing by the one
  decoded token must not overflow. All seven views must be distinct.
- The official boundary is 48 value heads with key and value dimensions 128;
  smaller positive geometries are supported for differential evidence.
- Prefill adds a leading `token_count` dimension to every input and output
  view except state. It processes tokens strictly in sequence per head, so one
  prefill launch, arbitrary sequential prefill chunks, and repeated decode
  launches produce bit-identical output and final state.
- Prefill requires a positive token count, exact position continuity, and
  checked `processed_position + token_count`. Its seven views are pairwise
  disjoint and remain caller-owned.

## Ownership and execution

Seen owns every allocation and the stream. The adapter borrows the ledgered
`SeenCudaStreamLaunchToken`, validates fixed-width device views, and enqueues
one model-owned `seen_qwen_*` kernel on that exact stream. It neither allocates
nor retains state, creates or destroys streams, synchronizes, decodes an opaque
handle, applies fallback, or owns policy. History, recurrent state, and outputs
remain valid only under the caller's normal Seen allocation lifetime.

## Errors and maturity

Missing or incompatible tokens, wrong device/allocation identity, invalid or
overflowing geometry, stale positions, undersized views, and unsupported
overlap fail before enqueue with stable Seen CUDA status categories. Device
data is never copied to the host for validation. There is no fallback. This
corpus is
`experimental-hardware` until the complete QWN-041 corpus is certified.

The focused gates are `scripts/cuda/run_qwn_041a.sh` and
`scripts/cuda/run_qwn_041b.sh`, and `scripts/cuda/run_qwn_041c.sh`. They use audited Seen
v0.20.4, a current-memory-derived swap-disabled hard scope, one build worker,
RTX 4090 CPU/CUDA differential and state-continuity checks, CUDA graph capture,
deterministic teardown, and Compute Sanitizer memcheck, initcheck, racecheck,
and synccheck.
