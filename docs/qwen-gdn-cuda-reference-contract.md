# Qwen GDN CUDA state and causal-convolution reference contract

FEL-1433 / QWN-041A adds the first Gated DeltaNet CUDA state operation. The
reference path is deliberately narrow: FP32 causal depthwise convolution with
the official Qwen convolution length of four, followed by SiLU. Recurrent
delta-state update, chunk prefill, and gated output belong to later QWN-041
leaves.

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

## Ownership and execution

Seen owns every allocation and the stream. The adapter borrows the ledgered
`SeenCudaStreamLaunchToken`, validates fixed-width device views, and enqueues
one model-owned `seen_qwen_*` kernel on that exact stream. It neither allocates
nor retains state, creates or destroys streams, synchronizes, decodes an opaque
handle, applies fallback, or owns policy. History and output remain valid only
under the caller's normal Seen allocation lifetime.

## Errors and maturity

Missing or incompatible tokens, wrong device/allocation identity, invalid or
overflowing geometry, stale positions, undersized views, and unsupported
overlap fail before enqueue with stable Seen CUDA status categories. Device
data is never copied to the host for validation. This leaf is
`experimental-hardware` until the complete QWN-041 corpus is certified.

The focused gate is `scripts/cuda/run_qwn_041a.sh`. It uses audited Seen
v0.20.4, a current-memory-derived swap-disabled hard scope, one build worker,
RTX 4090 CPU/CUDA differential and state-continuity checks, CUDA graph capture,
deterministic teardown, and Compute Sanitizer memcheck, initcheck, racecheck,
and synccheck.
