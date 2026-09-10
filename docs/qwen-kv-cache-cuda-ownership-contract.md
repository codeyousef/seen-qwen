# Qwen CUDA KV-cache ownership contract

QWN-042B adds a move-only Seen owner for the contiguous FP32 attention KV
cache. The owner allocates exactly two persistent device regions through
`CudaAllocation`, records the immutable capacity and `[kv_heads, head_dim]`
geometry, and alone advances the authoritative used length.

Append is sequential: `start_position` must equal the current length, and the
new range must fit the fixed capacity. The owner passes borrowed input and
cache views to `seen_qwen_kv_append_f32` for one immediately nested call on the
exact borrowed Seen stream. It advances length only after all validation and
kernel enqueue succeed. A failed append leaves both ownership metadata and
previously committed cache contents unchanged.

Reset sets length to zero without reallocating, preserving stable addresses for
subsequent stream or graph work. Old bytes are outside the logical cache and
are overwritten before becoming readable again. Close releases both Seen
allocations, attempts the second release even if the first reports an error,
zeros logical state, and is idempotent. Native Qwen code never owns, retains,
decodes, allocates, frees, synchronizes, changes device, or chooses a stream.

The reference layout is token-major contiguous FP32
`[capacity, kv_heads, head_dim]`. QWN-042C may borrow `keyView()`, `valueView()`,
and `length()` while the owner remains open; it must never retain those views
past the owner lifetime. Paging, compression, offload, fallback, and cache
policy remain outside this reference leaf.
There is no fallback or retry path.

Run `scripts/cuda/run_qwn_042b.sh` for the complete gate. It verifies the exact
Seen v0.20.4 payload, checks and compiles the Seen ownership surface, builds
serially for SM89, exercises official 4x256 geometry, sequential and rejected
updates, capacity bounds, reset/reuse, stable addresses, exact-stream ordering,
graph capture/replay, invalid tokens, deterministic cleanup, and all four CUDA
sanitizers. Generated outputs remain under ignored project-local `.seen` paths.
