# Qwen full-attention output-gate CUDA contract

FEL-1441 / QWN-042E completes the full-attention reference path after decode
or prefill. The attended FP32 rows and the gate branch copied from `q_proj` are
contiguous `[tokens, query_heads, head_dim]` buffers. The result is flattened
without reordering to `[tokens, query_heads * head_dim]`, then each element is
`attended * sigmoid(gate)`, exactly matching the pinned Qwen3.5 modeling
source. This caller-owned result is the input to the separately owned `o_proj`
linear path in QWN-043; this leaf does not duplicate GEMM or weight policy.

The Seen integration checks every non-zero geometry product before calling the
existing model-specific sigmoid-gate primitive. Native validation checks the
ledgered stream token, allocation identity, device, byte extent, and overlap.
Exact input or exact gate in-place output is supported because every element
is read before its corresponding output write; partial overlap is rejected.

The primitive issues one accepted launch on the exact borrowed Seen stream. It
does not allocate, retain a token or view, decode an opaque handle, create or
destroy a stream, select the default stream, synchronize, retry, or silently
fall back. The caller owns buffers, graph lifetime, deadline, and deterministic
cleanup. This path is a correctness oracle and makes no performance claim.

Run `scripts/cuda/run_qwn_042e.sh` for exact Seen surface checks, official
24-head by 256-dimension CPU/CUDA differential results, flattened-layout and
in-place equivalence, boundary and negative cases, graph capture/replay,
deterministic teardown, and all four CUDA sanitizer gates under audited Seen
v0.20.4.
