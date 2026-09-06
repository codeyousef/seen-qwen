# Bounded Qwen source-shard stream contract

QWN-032B traverses only the pinned `Qwen/Qwen3.8-27B` source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It requires all 18 canonical
Safetensors shard names and the QWN-021B catalog identity before payload work.

The scanner opens shards in numeric order. Exactly one shard and one mapped
payload window may be live at a time, with one synchronous worker and no task
queue. Every mapped window is no larger than the QWN-032A source-window
reservation and is closed before the next window is created. A preflight bound
rejects policies that could require more than 131,072 mapping operations.
Consumed pages are discarded before each window is closed so sequential scans
do not retain the model payload in the process working set. Boundary probes use
the mapped byte pointer directly and never materialize tensor bytes as strings.

Every shard header is parsed by the strict bounded Safetensors reader. Tensor
names, shard assignments, dtypes, shapes, byte geometry, non-overlap, and file
ranges are validated before a payload window is created. Traversal must observe
exactly 1,199 unique catalog tensors: 851 text, 15 MTP, and 333 vision.

Text and MTP payloads are visited in bounded slices. Vision entries participate
only in header, catalog, assignment, and byte-accounting validation. No vision
payload window is created and no vision payload byte is read. Missing, extra,
duplicate, misassigned, malformed, or unbounded inputs fail closed with stable,
non-retryable `qwen.stream.*` diagnostics.

The returned summary owns no source mapping. It records exact counts, mapped and
excluded byte totals, observed limits, and zero vision payload windows. Cleanup
clears it deterministically and is idempotent. The plan and catalog remain owned
by the caller and must stay live for the synchronous audit call.

This leaf does not convert or encode tensor data, choose a codec, hash tensor
payloads, write per-tensor evidence or a resume journal, build SQW bytes, or
promote an output artifact. It initializes neither Python nor CUDA and executes
no remote model code.
