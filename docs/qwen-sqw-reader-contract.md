# Bounded zero-copy SQW reader contract

This document defines the public QWN-030B / FEL-1416 reader contract for
Seen Quantized Weights v1.0. The binary format remains defined by
`docs/qwen-sqw-v1-contract.md`; this document defines how the native Seen
reader validates, exposes, releases, and closes that format.

SQW remains a deterministic derived runtime format. Safetensors remains the
canonical source and exchange format. Opening an SQW file does not authorize
remote code, repair malformed input, select a fallback, or replace the model,
source, conversion-policy, catalog, or engine-artifact locks.

## Reader policy and identity

`SqwReader.open(path, policy)` is read-only and requires an explicit
`SqwReaderPolicy`. The policy contains:

- the expected model-lock SHA-256;
- the expected source-lock SHA-256;
- the expected conversion-policy SHA-256;
- the expected tensor-catalog SHA-256;
- the expected external whole-file SHA-256 from the engine artifact;
- the exact expected tensor count; and
- the hard limit on simultaneously cached component windows.

All five digests are exactly 64 lowercase hexadecimal characters. The expected
tensor count is in the frozen parser range 1 through 2,048. The window limit is
1 through 64. Both `SqwReaderPolicy.exact` and `SqwReaderPolicy.qwen38`
default the window limit to 16. No expected identity is optional or inferred
from a path or the local machine.

`SqwReaderPolicy.exact` accepts an exact catalog and tensor count. It supports
small conformance fixtures and other artifacts that obey the same closed SQW
v1 contract, but it does not establish production-model compatibility merely
because validation succeeds.

`SqwReaderPolicy.qwen38` pins the production Qwen3.8-27B identity to exactly
866 converted tensors—851 required text tensors and 15 required MTP
tensors—and catalog SHA-256
`5f466d43bae3059e54f0bfe183d0e82c822242f45a834d778414d3e5b5248f1f`.
Vision and unknown tensors are rejected. A small exact-policy fixture cannot be
silently relabeled as a production artifact.

SQW v1.0 has no optional reader features. `required_features` is empty,
`reader_major` is 1, and `reader_minor` is 0. An unknown version, flag,
identifier, codec, feature, or compatibility value is rejected rather than
ignored.

## Validation before exposure

`SqwReader.open` returns a reader only after the complete artifact has passed
every applicable check below. A failed open returns no partially valid reader
or tensor pointer.

1. Open the file read-only, obtain its exact extent, and reject an extent above
   the SQW signed-safe `2^63 - 1` ceiling.
2. Map and decode exactly the 256-byte fixed header. Validate magic, endian,
   version, sizes, flags, identities, section bounds, ordering, alignment, EOF,
   and every reserved byte.
3. Verify that every gap between fixed sections contains only zero bytes.
4. Parse the bounded manifest as strict UTF-8 JSON with duplicate keys
   rejected. Traverse JSON objects only through indexed `length()`,
   `keyAt()`, and `valueAt()` access. Reject missing, unknown, mistyped, or
   non-canonical fields, and require the parsed value to serialize byte-for-byte
   to the stored canonical JSON.
5. Keep the owning strict-JSON parse document alive while any borrowed key,
   value, or string from its tree is being used. Borrowed JSON children are
   never destroyed independently. Once validation has consumed those borrows
   and cloned the tensor names retained by the reader, destroy the parse
   document before returning from `open`.
6. Decode every fixed directory entry and validate its name-table range. Names
   are strict UTF-8, non-empty, at most 1,024 bytes, contain no NUL, belong to
   the allowed text/MTP namespaces, and are strictly increasing and unique by
   canonical UTF-8 bytes.
7. Cross-check every manifest tensor field against the corresponding directory
   entry and name: role, dtype, codec, rank, shape, logical count, component
   ranges, row and group geometry, alignment, source digest, and converted
   digest. Tensor count and manifest order match the directory exactly.
8. Use checked 64-bit addition, multiplication, alignment, and range arithmetic
   before mapping or allocation. Validate active and unused shape slots,
   logical products, codec sizes, group and row geometry, and the high-bit
   prohibition.
9. Require each present data, scale, zero-point, and metadata component to obey
   its entry alignment. Components occur in canonical
   `data || scale || zero || metadata` order, never overlap or alias within or
   across tensors, and follow canonical tensor-name payload order. Every
   inter-component and inter-tensor padding byte is zero. The unused high
   nibble of an odd-length Q4 row is zero.
10. Validate the footer header and ordered checksum entries for manifest,
    directory, names, payload, and optional evidence. Verify section SHA-256,
    the manifest directory digest, and each converted-bundle digest over the
    exact component concatenation excluding padding.
11. Verify the whole-file SHA-256 over every byte with its own 32-byte footer
    slot replaced by zero. It matches both the footer value and the policy's
    expected external engine-artifact digest.
12. Apply the exact identity, count, and compatibility policy. Only then return
    an active reader.

File-sized validation work is streamed through bounded mappings. Opening may
copy bounded header, directory, name, manifest, checksum, and hash scratch
data, but it does not buffer or retain a copy of the SQW payload. All section
and tensor digests are verified before the first component window can be
requested; verification is not lazy.

## Lookup and the exact view ABI

The exported view is exactly this four-field, trivially copyable value:

```seen
@repr(C)
@trivially_copyable
pub class SqwTensorView {
    var data: *const UInt8
    var byteLength: UInt64
    var logicalElements: UInt64
    var codec: UInt16
}
```

The view has no owner handle, generation, component side pointers, or
`close()` method. `data` points directly into one read-only mapped component;
`byteLength` is that selected component's byte length;
`logicalElements` is the tensor's logical element count; and `codec` is its
runtime codec. Copying the structure copies the borrow, not the bytes or the
mapping.

The public lookup surface is:

- `length()` and `nameAt(index)` for the validated directory;
- `tensor(name)` and `tensorAt(index)` for the data component; and
- `component(name, selector)` and `componentAt(index, selector)` for
  `SQW_COMPONENT_DATA`, `SQW_COMPONENT_SCALE`, `SQW_COMPONENT_ZERO`, or
  `SQW_COMPONENT_METADATA`.

Lookup uses exact canonical names or validated indices. It never repairs an
alias, guesses by shape, substitutes a similarly named tensor, changes a codec,
or dequantizes data. Requesting an absent optional component is
`sqw.component_absent`; it does not return a synthetic null view.

## Reader-owned windows and invalidation

The reader owns a cache keyed by tensor index and component selector. The first
lookup of a component creates one bounded read-only mapped window. Repeated
lookups of that same component reuse the cached window and do not consume
another window slot, even though each returned `SqwTensorView` is a distinct
trivially copyable value. A different tensor/component key consumes another
slot. Reaching `policy.maxOpenWindows` returns `sqw.window_limit` before a
new mapping is created.

The caller releases cached mappings through the reader:

- `releaseTensor(name)` releases the data-component window; and
- `releaseComponent(name, selector)` releases the selected component window.

A successful release invalidates every copied view that points into that cached
window. A later lookup may create a new window; old copies remain invalid.
Releasing an existing but uncached component, releasing after reader close, or
after cleanup has begun, or closing an already closed reader returns
`Ok(false)`. A successful first release or close returns `Ok(true)`. Because
the four-field view carries no owner or generation, callers must not
dereference it after the corresponding release or after reader cleanup begins.

`length()` returns zero after close. Lookup and indexed-name operations reject
a closed reader. No pointer or reader-owned name may be retained past reader
close.

## Ownership and deterministic cleanup

While active, `SqwReader` owns the read-only mapped file handle, cloned tensor
names and decoded directory records, and its cache of component windows. It
does not retain the strict-JSON parse document or temporary header, manifest,
directory, name-table, footer, or hashing mappings after successful open.

`SqwReader.close` first enters a cleanup-pending state that rejects every
lookup, reports length zero, and prevents new mappings. It then closes every
active component window. A failed close remains owned in its cache and returns
`sqw.close`; an explicit later `close()` retries only the remaining owners.
Once all windows close, the reader frees the window cache, destroys owned names
and directory records exactly once, and closes the file last. A failed file
close likewise remains owned for an explicit later `close()` retry. Only a
successful file close marks cleanup complete. Repeated close after completion
is idempotent and returns `Ok(false)`.

Open failure destroys the validation-owned parse tree and bounded temporary
resources, then closes the mapped file; no reader escapes. In the released Seen
v0.19.2 strict-JSON wrapper, syntax-error conversion does not release its raw
failed parse result (FEL-1561). The reader therefore composes the same stdlib
UTF-8 validator and strict bounded parser behind an explicit `@noinline` frame
barrier, releasing the raw failed result before returning `sqw.manifest_json`.
Successful parsed-document ownership remains unchanged.

In the released Seen v0.19.2 mapped-resource contract, a native close failure retains its opaque
handle but cannot transfer that cleanup owner through `SqwError`. Therefore an
anomalous failure while closing a temporary validation mapping or rejected
file is fail-stop: process teardown reclaims the operating-system resources
instead of orphaning an unreachable handle. Reader-owned mappings do not use
this path and remain explicitly retryable as described above.

The reader is synchronous and creates no background task, queue, retry, or
asynchronous completion. Cancellation and deadline propagation are therefore
not applicable to this leaf. Retrying a rejected artifact means an explicit
new `open`; retrying reader cleanup means an explicit bounded call to `close()`.

## Stable diagnostics

Failures use `SqwError` with a literal, non-concatenated `sqw.*` code, a
bounded field identifier, and an actionable message. The reader adds these
stable names:

- input and policy: `sqw.path`, `sqw.reader_policy`, `sqw.reader_limit`,
  `sqw.open`;
- state and lookup: `sqw.closed`, `sqw.bounds`, `sqw.missing`,
  `sqw.component`, `sqw.component_absent`, `sqw.window_limit`;
- mapping and cleanup: `sqw.map`, `sqw.close`;
- manifest and compatibility: `sqw.manifest_json`,
  `sqw.manifest_canonical`, `sqw.manifest_schema`,
  `sqw.manifest_type`, `sqw.manifest_tensor_count`,
  `sqw.manifest_tensor`, `sqw.compatibility`, `sqw.identity`,
  `sqw.tensor_count`;
- names and layout: `sqw.name_utf8`, `sqw.name_overlap`,
  `sqw.name_order`, `sqw.padding`, `sqw.tensor_overlap`,
  `sqw.q4_tail`, `sqw.alignment`, `sqw.section_order`; and
- hashes: `sqw.digest`, `sqw.section_digest`, `sqw.tensor_digest`,
  `sqw.file_digest`.

Errors returned by the frozen QWN-030A header, directory, decimal, and footer
parsers retain their documented codes from `docs/qwen-sqw-v1-contract.md`,
including `sqw.magic`, `sqw.endian`, `sqw.version`, `sqw.flags`, `sqw.id`,
`sqw.truncated`, `sqw.overflow`, `sqw.range_overflow`,
`sqw.component_range`, `sqw.component_overlap`, `sqw.codec_geometry`,
`sqw.name_range`, `sqw.reserved`, and the `sqw.footer_*` family.

The first validation mismatch is returned. A diagnostic never triggers repair,
retry, reinterpretation, a precision change, fallback, host offload, or GPU
work. Mapping failures identify the operation without exposing the input's
absolute path.

## Platform, evidence, and scope

FEL-1416 is CPU-only and requires no GPU. The reader and its focused tests may
not locate or link CUDA or another GPU SDK. The implementation uses the
released Seen mapped-file, strict-JSON, and streaming SHA-256 contracts; it
adds no foreign allocation or scheduling policy.

Focused evidence uses a deterministic two-tensor `SqwReaderPolicy.exact`
fixture for positive indexed and exact-name lookup at both fixture boundaries,
individual component access, repeated cache hits, release invalidation, window
limits, bounds, missing and absent components, policy and identity rejection,
manifest/name/directory/payload/footer corruption, padding, digests, live-window
cleanup, and idempotent close. A separate one-tensor Q4 fixture proves that an
odd 65-element row accepts a zero unused high nibble and rejects a nonzero one
as `sqw.q4_tail`. These small fixtures prove the generic mechanism; they do not
replace the production `SqwReaderPolicy.qwen38` requirement of 866 tensors.
Affected frontend and native CPU regressions run under the repository's serial,
current-memory-derived, swap-disabled hard scope and required CI.

QWN-030B does not write or mutate SQW files. Unique temporary creation,
conversion journaling, full reopen-before-promotion, `fsync`, and atomic
promotion belong to QWN-030C.

QWN-030B contains the focused hostile and boundary cases needed for this
reader. Exhaustive fuzz certification, minimized fuzz corpora, reproducibility
certification, compatibility-matrix review, sparse/large-file certification,
and extended leak/soak evidence belong to QWN-030D.
