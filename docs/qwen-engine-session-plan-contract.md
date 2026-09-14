# Qwen engine, session, and execution-plan ownership contract

QWN-045A introduces only the model-specific ownership and lifecycle boundary
used by the later end-to-end CUDA mini-model executor. `QwenEngine` move-owns
one validated `QwenExecutionPlan`. The plan owns copied artifact identities and
bounded numeric prefill/decode step arrays; it contains no string/reflection dispatch.
It fixes the eight-layer hybrid 3:1 GDN/full-attention schedule,
explicit dependencies, ping/pong slots, graph eligibility, final norm, LM head,
host sampling, and optional MTP step before any session can be admitted.

`QwenSession` move-owns its sampler and scalar sequence, cancellation, deadline,
backend, and fallback counters. It retains no borrowed engine, stream, handle,
allocation, or tensor view after a call. A session is identified by a copied
engine digest and must be returned to that exact engine for release. The engine
refuses teardown while sessions remain live, and plan/session cleanup is
deterministic and idempotent. Partial plan or engine construction failure frees
everything already owned.

CUDA is always the initial backend. An error never changes it. Fallback is a
numeric plan policy, never runtime string dispatch: the default is prohibited;
the only representable alternative is an explicitly admitted, previously
validated CPU reference. A fallback request must provide a nonzero reason,
semantic-equivalence assertion, and explicit approval, is counted, and changes
the declared active backend. The session retains the last nonzero reason code
for diagnostics and rejects a second transition away from CUDA. Missing evidence
fails closed without changing backend or counters. The first released engine artifact records
`fallback = prohibited`.

Generation admission is bounded by the hybrid mini-model's 128-token context,
an explicit generated-token limit, at most 1,024 live sessions, and a future
deadline epoch. Cancellation invalidates active work before commit. Late, cancelled,
over-context, or over-generation commits fail without mutating visible token
counters. QWN-045B owns actual operator execution as specified by
`qwen-cuda-mini-execution-contract.md`.
QWN-045A does not add a second CUDA resource stack, allocation policy, synchronization,
offload, precision change, or default-stream path.
