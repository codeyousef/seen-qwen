# Qwen resident LM-head and greedy CUDA contract

FEL-1442 projects caller-owned resident hidden states through the distinct,
untied `lm_head.weight` from the pinned Qwen3.8-27B checkpoint, then selects
one deterministic greedy token for each projected row. The exact operation is
`logits = hidden * lm_head.weight^T`; the official dimensions are
`[tokens,5120] * [248320,5120]^T -> [tokens,248320]`.

`QwenCudaLmHeadExecutor` owns one Seen cuBLASLt handle, one Seen stream, and
one process-local projection selection. Hidden states, the resident LM-head
weight, logits, token IDs, and optional workspace remain caller-owned bounded
device views. `executeGreedy` performs no allocation, algorithm selection,
synchronization, transfer, offload, fallback, retry, or precision change. It
enqueues projection and argmax on the same Seen-owned stream.

F16 and BF16 are the only admitted storage dtypes; cuBLASLt accumulates in
FP32 under the packaged runtime contract. Greedy selection compares the stored
low-precision logits directly. Equal finite logits select the lower token ID.
If a row contains both NaN and finite values, finite values win; an all-NaN row
deterministically selects token zero. Seeded/non-greedy sampling remains owned
by FEL-1451 and is never silently replaced with greedy selection.

All dimensions and byte extents use checked arithmetic. Views must be live,
device-matched, sufficiently large, and pairwise disjoint, including workspace
when required. Errors are typed and fail closed. `close` invalidates selection
metadata before deterministically closing the handle and stream and is
idempotent.

RTX 4090 qualification covers official geometry selection, F16/BF16 CPU
differentials, tie behavior, same-stream ordering, graph capture/replay,
invalid token/dtype/geometry/extent/alias cases, four Compute Sanitizer modes,
and 1,000 fixed-allocation repetitions. This reference path makes no
performance-selection claim.
