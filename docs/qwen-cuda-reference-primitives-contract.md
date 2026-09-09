# Qwen CUDA reference primitive contract

QWN-040A introduces the first Qwen-owned CUDA kernels as an
`experimental-hardware` correctness surface. It does not make a performance or
production-certification claim.

The implementation consumes `seen-cuda-stream-launch-token-v1` from the
published Seen v0.20.2 CUDA runtime. A live Seen-owned stream lends a token for
one immediately nested `seen_qwen_*` call. The adapter validates the token,
device, fixed-width buffer views, byte extents, launch geometry, and CUDA
allocation domain before enqueueing work on the token's exact native stream.
It never stores or decodes a Seen handle, owns or destroys a stream, changes
device, synchronizes work, selects the default stream, allocates memory, or
implements fallback or scheduling policy.

The bounded FP32 reference surface covers fill, elementwise/residual add,
deterministic row sum, two-dimensional transpose, copy, and embedding gather.
The row sum uses one thread per row and fixed increasing-column order; this is
deliberately a correctness reference rather than an optimized reduction.
Out-of-range embedding IDs produce a deterministic NaN sentinel for downstream
validation. Exact in-place add and copy are supported; partially overlapping
views and overlapping reduction, transpose, or embedding output are rejected
before launch because their parallel semantics would be ambiguous. Norms,
activation, RoPE, cache mutation, sampling, and optimized
kernels belong to later QWN-040 leaves.

The public C header contains no CUDA types. It exposes only the ledgered Seen
token, fixed-width scalars, a fixed-width borrowed device-buffer view, and
`SeenCudaStatus`. The implementation is a separately built Qwen native library;
the packaged Seen CUDA runtime remains the sole resource owner.

Run `scripts/cuda/run_qwn_040a.sh` for the complete local gate. It verifies the
published v0.20.2 payload, checks and compiles the Seen declaration surface, builds serially
for SM89, exercises correctness, same-stream ordering, capture/replay, malformed
tokens and views, stale/closed handles, and deterministic teardown, then runs
CUDA memcheck with full leak reporting plus initcheck, racecheck, and synccheck.
The entrypoint always delegates to the repository's current-memory-derived,
swap-disabled hard scope and writes generated files only below ignored `.seen`.
