# Deterministic atomic SQW writer contract

This document defines QWN-030C / FEL-1415. The binary format remains frozen by
`docs/qwen-sqw-v1-contract.md`, and complete pre-exposure validation remains
defined by `docs/qwen-sqw-reader-contract.md`.

`SqwWriter` is a bounded persistence owner for one already-planned canonical
SQW v1 byte stream. It does not classify tensors, choose codecs, convert model
weights, repair malformed data, infer an identity, or select a fallback.
Those policies belong to later converter leaves. Safetensors remains the
canonical source and exchange format.

## Begin and staging ownership

`SqwWriter.begin(destination, expectedBytes, maxChunkBytes,
replaceExisting, context, readerPolicy)` requires:

- a valid bounded destination path;
- an exact positive signed-safe final extent;
- a chunk limit from 1 through 1 MiB;
- an active bounded `OperationContext`; and
- the exact `SqwReaderPolicy` that the final artifact must satisfy.

Begin creates one exclusive unpredictable temporary file beside the
destination. For this single-file format that file is the unique temporary
artifact; no sidecar, journal, payload spill, or second output is created.
Same-directory placement preserves the filesystem boundary required by the
single final rename. The writer owns the open file and staging pathname until
successful promotion or successful abort.

Creation never follows the final staging component. Sixteen bounded collision
attempts are provided by the released Seen filesystem contract. A failure
returns `sqw.stage` without opening a writer.

## Bounded writes

`write(bytes)` accepts a non-empty array no larger than the configured limit.
Every element must be in 0 through 255. Writes are positional and sequential;
checked 64-bit addition rejects a chunk that would exceed the declared final
extent. The writer retains no caller buffer after the call.

There is no queue, asynchronous task, implicit retry, sparse seek, overwrite,
or out-of-order write. Cancellation is checked before I/O and the filesystem
operation receives the same context. A failed native write moves the owner to
a failed state; the caller must explicitly call `abort()`.

## Validation and promotion

QWN-032D may call `seal()` before `commit()` to durably sync and close a
complete staging file for an independent readback pass. `sealedPath()` is
available only while that owner remains sealed and still owns the staging
pathname. Sealing is idempotent, accepts no more writes, and never promotes the
artifact. `commit()` accepts either an active complete writer or a sealed
writer; in both cases it still performs its own full `SqwReader` reopen before
the atomic rename.

`commit()` succeeds only after exactly `expectedBytes` have been written. It:

1. fully syncs the staging file;
2. closes the staging file;
3. reopens the staged pathname with `SqwReader.open` and the caller's exact
   policy;
4. closes the validated reader;
5. performs one same-filesystem atomic rename; and
6. fully syncs the destination directory.

The destination is not renamed or modified when the stream is incomplete or
when reader validation rejects any header, manifest, directory, name, payload,
footer, identity, compatibility, tensor digest, section digest, or whole-file
digest. Reader `sqw.*` errors propagate unchanged.

`replaceExisting` is explicit. When false, an existing destination causes
`sqw.promote` and remains unchanged. When true, the final rename atomically
replaces it. No partial staging file is ever treated as resumable input.

A directory-sync failure can occur after the rename is visible. In that case
`isPromoted()` is true, `commit()` returns `sqw.durability`, and a later
explicit `commit()` retries only the directory sync. No write, revalidation,
or second rename occurs. Once durable, another commit returns `Ok(false)`.

## Cleanup and stable errors

`abort()` closes the owned staging file before removing its pathname. A close
or removal failure retains the corresponding owner/state for an explicit
bounded caller retry. A successful abort is idempotent; abort after promotion
is a no-op and never removes the destination.

Temporary validation-reader and destination-directory handles cannot be
transferred through `SqwError`. A close anomaly on either handle therefore
fails the process deterministically instead of returning after losing its
cleanup owner. This follows the existing SQW reader rule for temporary mapped
resources; it does not affect retryable staging ownership or directory-sync
errors.

Stable writer diagnostics are:

- `sqw.path`, `sqw.cancelled`, and `sqw.writer_limit` for admission;
- `sqw.stage`, `sqw.chunk`, `sqw.write`, and `sqw.writer_state` for staging;
- `sqw.incomplete`, `sqw.sync`, and `sqw.close` before validation;
- the exact reader diagnostics for failed reopen validation; and
- `sqw.promote`, `sqw.durability`, and `sqw.cleanup` for finalization.

Errors do not expose an absolute staging pathname, alter the artifact bytes,
change precision, perform host/GPU fallback, or retry without an explicit
caller action.

## Evidence and scope

The focused regression streams the same valid artifact with different chunk
boundaries and proves identical reader-verified results. It covers exact first
and last tensor identity, incomplete input, non-byte input, signed-safe extent,
cancellation before staging, no-replace collision, corruption before rename,
destination preservation, cleanup, and idempotent commit/abort.

FEL-1415 is CPU-only. It must not locate or link CUDA. Focused and aggregate
native evidence uses an explicit exact Seen toolchain, serial workers, an
8 MiB stack, a current-memory-derived hard memory limit, disabled swap, bounded
tasks, and ignored project-local `.seen` artifacts.

QWN-030D owns extended malformed/fuzz, sparse/large-file, cross-filesystem,
compatibility-matrix, leak, soak, and reproducibility certification.
