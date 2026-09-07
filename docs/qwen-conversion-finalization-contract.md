# Independent Qwen conversion finalization contract

QWN-032D finalizes one complete QWN-032C conversion journal and one complete
SQW staging artifact. It does not convert weights, choose codecs, repair input,
resume partial output, select a fallback, or initialize Python or CUDA.

## Admission and ownership

`finalizeQwenConversion(writer, journalText, identity, plan, catalog)` borrows
the live `SqwWriter`. The caller retains that owner for explicit retry or
cleanup. The identity, conversion plan, and 1,199-entry catalog must satisfy
their existing pinned contracts. The writer must contain exactly its declared
positive extent, remain within the plan's artifact bound, use a chunk bound no
larger than the plan's writer bound, and carry the exact model, source,
conversion-policy, catalog, and 866-tensor reader policy.

The journal is parsed with bounded strict JSON. Its parsed document remains
alive while entry names and digest values are borrowed through indexed access.
The finalizer closes that document exactly once on every accepted or rejected
path. A complete journal contains evidence for all 866 canonical text and MTP
tensors; a prefix is valid resume evidence but is not promotable.

## Independent readback

The finalizer first calls `writer.seal()`, which fully syncs and closes the
complete staging file without exposing it at the destination. It then opens the
sealed pathname through a new `SqwReader`. The reader independently validates
the complete SQW header, canonical manifest, directory, names, component
geometry, payload and padding, section digests, whole-file digest, identity,
compatibility, tensor count, and evidence range.

The embedded evidence bytes must equal the validated completed journal exactly.
For every index from 0 through 865, the reader-validated tensor name, source
SHA-256, converted SHA-256, and converted component extent must equal the
corresponding journal entry. Access is bounded; closing the reader invalidates
all borrowed names, digests, component metadata, and evidence views.

## Atomic promotion, retry, and cleanup

Only after the independent readback succeeds does the finalizer call
`writer.commit()`. Commit performs a second full `SqwReader` reopen, then uses
the existing same-directory atomic rename and destination-directory sync. A
pre-promotion failure leaves any existing destination unchanged and leaves the
writer as the explicit staging cleanup owner.

A directory-sync failure after rename is the only retryable finalization
failure. The writer records durability-pending state, and a bounded explicit
retry performs only the missing directory sync. Repeating a durable completed
promotion returns `Ok(false)`. Abort is deterministic and idempotent and never
removes a promoted destination.

Stable finalizer diagnostics use the `qwen.finalize.*` namespace:

- `journal`, `incomplete`, and `compatibility` reject invalid admission;
- `seal`, `readback`, and `evidence` reject staging or independent validation;
- `promote` reports rename or durability failures while retaining the exact
  underlying `sqw.*` cause.

There is no silent repair, alternate backend, precision change, host/GPU
fallback, or unbounded retry.

## Evidence and scope

The focused oracle builds a deterministic 866-tensor mixed-name SQW artifact
smaller than 2 MiB and embeds the exact completed canonical journal. Native
evidence verifies exact first, middle, and last tensor identities, all entry
digests and extents, caller bounds, ownership invalidation, positive promotion,
idempotency, incomplete and valid-but-mismatched evidence, payload corruption,
destination preservation, and deterministic cleanup.

FEL-1423 is CPU-only. Its Python fixture generator is test/oracle tooling only;
the production finalizer is native Seen. All checks, release/ThinLTO builds, and
native executions use an explicit compiler, serial workers, an 8 MiB stack,
current-memory-derived hard memory and task limits, zero swap, bounded timeouts,
and ignored project-local `.seen` artifacts.
