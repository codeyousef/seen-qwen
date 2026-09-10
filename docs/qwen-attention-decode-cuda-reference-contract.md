# Qwen grouped-query attention decode CUDA reference contract

FEL-1440 / QWN-042C adds the correctness-reference decode operation for one
query token. It consumes normalized, rotary-positioned query rows from
QWN-042A and borrows the committed key/value prefix owned by QWN-042B. Linear
projection, KV append, output gating, and output projection remain separate.

The layouts are contiguous FP32: query and output are
`[query_heads, head_dim]`; key and value cache are
`[cache_capacity, kv_heads, head_dim]`. Query heads are divided evenly into
KV-head groups. Each head computes scaled dot products with scale
`1 / sqrt(head_dim)`, an overflow-safe online softmax across exactly
`cache_length` positions, and the probability-weighted value row. Capacity
bytes beyond the committed prefix are never read.

The operation rejects zero or overflowing geometry, an empty or over-capacity
live range, non-divisible grouped-query heads, undersized or overlapping
views, wrong-device allocations, and missing or incompatible stream tokens
before launch. Every product is checked in 64 bits. It has no allocation,
workspace, retry, fallback, host inspection of device data, default stream,
device change, or hidden synchronization.

Seen owns the live KV-cache allocations, query/output allocations, and stream.
`QwenCudaKvCache.decode` borrows cache views for one immediately nested call
and never advances cache length. The native adapter validates those views and
enqueues one kernel on the exact ledgered Seen stream without retaining the
token or any view. The caller owns its deadline and deterministic cleanup.
CUDA graph capture/replay preserves the same ordering and ownership.

This is a K1 correctness oracle, not the production full-attention policy.
QWN-042D owns prefill integration, and a later approved FlashAttention,
FlashInfer, or vendor selection may replace production execution only after
its own numerical and performance certification. There is no silent fallback.

Run `scripts/cuda/run_qwn_042c.sh` for the complete gate. It verifies the exact
Seen v0.20.4 payload, a fixed-width Seen surface, serial SM89 construction,
official 24-query-head/4-KV-head/256-dimension CPU/CUDA differential results,
stable softmax, negative and boundary cases, exact-stream graph capture,
deterministic teardown, and all four Compute Sanitizer tools. Generated
artifacts remain in ignored project-local `.seen` paths.
