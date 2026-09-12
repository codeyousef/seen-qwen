# Qwen FFN CUDA execution contract

FEL-1444 implements the pinned Transformers Qwen3.8 text MLP from
`modeling_qwen3_5.py`, symbol `Qwen3_5MLP.forward`, at commit
`562cfd944ee1f20702cfb0f4404014ee27c24813`:

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

All projections are bias-free. Gate and up map `[tokens,5120]` to
`[tokens,17408]`; down maps `[tokens,17408]` to `[tokens,5120]`. Each uses the
QWN-043A row-major descriptor and a process-local selected cuBLASLt algorithm.
F16 and BF16 are the only admitted execution dtypes. FP32 accumulation is the
packaged cuBLASLt reference policy; output and activation storage retain the
selected low-precision dtype.

`QwenCudaFfnExecutor` owns one Seen cuBLASLt handle, one Seen stream, and the
three process-local selections. Execution borrows caller-owned bounded device
views for input, weights, gate/up intermediates, activation, output, and
workspace. It performs no allocation, handle creation, algorithm selection,
fallback, default-stream work, or synchronization. Gate storage may be reused
exactly as activation storage; all partial or incompatible aliases fail.

The model-specific native adapter implements only elementwise
`SiLU(gate) * up` for F16/BF16. It consumes one borrowed launch token, verifies
the token, device allocation identity and extent, dtype, count, and aliasing,
and enqueues on the exact Seen-owned stream. It never retains or decodes a
handle, owns no allocation or policy, and is CUDA Graph capture compatible.

Errors are typed and fail closed. A failed asynchronous stage does not trigger
CPU execution, another stream, a lower precision, or a retry. The executor
remains deterministically destructible after partial execution; `close` is
idempotent and invalidates all selections before closing the handle and stream.

RTX 4090 qualification covers F16/BF16 CPU differentials, exact operation
ordering without intermediate synchronization, graph capture/replay, invalid
tokens/dtypes/counts/extents/aliases, all four Compute Sanitizer modes, and
1,000 fixed-allocation repetitions. QWN-043C separately owns production
algorithm and layout certification; FEL-1444 makes no performance-selection
claim.
