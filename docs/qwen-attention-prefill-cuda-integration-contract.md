# Qwen causal attention prefill CUDA integration contract

FEL-1443 / QWN-042D integrates bounded causal prefill with the contiguous
Seen-owned KV cache. Inputs are normalized, rotary-positioned query and key
rows plus value rows produced by earlier steps. The FP32 layouts are query and
output `[tokens, query_heads, head_dim]`, key/value input
`[tokens, kv_heads, head_dim]`, and cache
`[capacity, kv_heads, head_dim]`.

`QwenCudaKvCache.prefill` accepts only an append position exactly equal to the
authoritative cache length. One model-specific native enqueue first writes each
token's key/value rows and then evaluates that token against positions
`0..start_position+token`, inclusive. The single correctness-oracle block makes
cache mutation and causal attention one accepted launch; only then does Seen
advance the authoritative cache length. Arbitrary chunk boundaries preserve
the same output and final cache as one whole prompt.

The launch uses overflow-safe online softmax with scale `1 / sqrt(head_dim)`
and grouped-query head mapping. All geometry, extents, devices, aliases, cache
bounds, and the ledgered stream token are validated before enqueue. The adapter
does not allocate, retain borrows, inspect device data on the host, decode an
opaque handle, use the default stream, synchronize the device, retry, or
silently fall back. Seen owns query/output buffers, both cache allocations, the
stream, deadlines, cancellation points between chunks, and deterministic
cleanup.

This is the K1 causal correctness oracle, not a production prefill policy.
Selection of FlashAttention, FlashInfer, or a vendor path remains a later
benchmark- and owner-approved decision. Run `scripts/cuda/run_qwn_042d.sh` for
the Seen surface, official 24/4/256 CPU/CUDA differential, whole/chunked
equivalence, tail bounds, graph replay, negative cases, cleanup, and all four
CUDA sanitizer gates under the audited Seen v0.20.4 payload.
