# Qwen conversion evidence and resume-journal contract

QWN-032C computes source and converted SHA-256 digests incrementally with
chunks no larger than 1 MiB. Source bytes must finish before converted bytes
begin, both declared extents must be consumed exactly, and incomplete or closed
checksum owners cannot produce evidence. Cleanup of unfinished hash state is
deterministic and idempotent.

The `seen-qwen-conversion-journal-v1` JSON schema binds every run to the pinned
model revision, model and source locks, conversion policy, QWN-032A plan,
QWN-021B catalog, Seen toolchain compatibility manifest, SQW v1 contract, and
canonical included-tensor order. Completed entries form one gap-free prefix of
the 866 text/MTP tensors. Each entry records canonical ordinal and name,
source/converted byte extents, and both SHA-256 digests. Geometry is encoded as
canonical decimal strings so no 64-bit value passes through floating point.

Journal parsing uses bounded strict JSON and indexed `length()`, `keyAt()`, and
`valueAt()` traversal. The parsed document remains owned by the live journal;
borrowed JSON values never outlive it. Unknown fields, duplicates, truncation,
invalid UTF-8, stale identity, reordered entries, bad checksums, overflow, and
plan-limit violations fail closed with stable non-retryable `qwen.convert.*`
diagnostics.

Persistence first validates the complete journal and then uses the Seen
runtime's same-directory atomic text replacement. A validation or write failure
preserves an existing destination and removes the operation-owned temporary
file. Journal cleanup is idempotent.

The journal proves only compatibility and completed evidence. It never makes a
partial SQW artifact trusted or promotable. Independent full-file readback,
durability, and atomic artifact promotion are defined by
`docs/qwen-conversion-finalization-contract.md`. This CPU-only contract
initializes neither Python nor CUDA.
