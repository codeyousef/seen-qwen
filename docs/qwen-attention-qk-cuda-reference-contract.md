# Qwen full-attention Q/K CUDA reference contract

FEL-1439 / QWN-042A adds the bounded Q/K preparation step used by the later
cache, decode, and prefill leaves. The caller supplies completed linear
projection results; quantized/general matrix multiplication remains owned by
QWN-043. This operation validates and transforms projection layout only.

## Layout and semantics

- `query_gate_projection` is contiguous token-major
  `[tokens, query_heads, 2, head_dim]`, with each head storing its query row
  immediately followed by its gate row. `key_projection` is contiguous
  `[tokens, kv_heads, head_dim]`.
- `query_output`, `key_output`, and `gate_output` are caller-owned token-major
  buffers. The gate is copied exactly so QWN-042E can apply it after attention.
- Each query/key row uses RMS normalization with positive finite epsilon and
  the official learned scale `1 + weight`. Normalization precedes RoPE.
- Rotate-half RoPE applies only to the first positive even `rotary_dim`
  elements. Text-only interleaving gives every axis the same scalar position,
  so this matches the frozen Qwen3.5 modeling source and CPU oracle. Remaining
  dimensions retain their normalized learned scale.
- The official Qwen3.8-27B geometry is 24 query heads, 4 KV heads, head
  dimension 256, rotary dimension 64, theta 10,000,000, maximum position
  262,144, and epsilon 1e-6. Smaller positive grouped-query geometries remain
  available for deterministic differential tests; query heads must be exactly
  divisible by KV heads.

All shape products and the position interval are checked in 64 bits before
launch. Every input and output view must be a distinct, sufficiently large
device allocation on the token's current device. There is no aliasing or
allocation, and there is no fallback, repair, host inspection of device data, or hidden
synchronization.

## Ownership and execution

Seen owns all allocations and the stream. The adapter borrows one ledgered
`SeenCudaStreamLaunchToken` and enqueues the query and key kernels, in order,
on that exact stream. It never retains the token, decodes an opaque handle,
creates or destroys a stream, or changes caller ownership. CUDA graph capture
and replay preserve the same ordering. Caller teardown remains deterministic
and idempotent under the Seen allocation/stream ledger.

Missing or incompatible tokens, wrong device/allocation identity, invalid or
overflowing grouped-query geometry, out-of-range positions, undersized views,
and overlap fail before enqueue with stable Seen CUDA status categories. This
surface has no retry or cancellation wait: the caller's bounded command owns
its deadline before launch. The maturity is `experimental-hardware` until the
complete QWN-042 corpus is certified.

The focused gate is `scripts/cuda/run_qwn_042a.sh`. It uses the audited Seen
v0.20.4 payload, a current-memory-derived swap-disabled hard scope, serial
builds, official-geometry CPU/CUDA differential checks, end-of-context and
malformed geometry cases, CUDA graph capture, deterministic teardown, and all
four Compute Sanitizer tools.
