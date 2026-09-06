# Deterministic Qwen conversion-plan contract

QWN-032A admits conversion of only the pinned Qwen3.8-27B text/MTP source. The
plan freezes source revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, 55,562,855,904 source bytes,
18 shards, the 1,199-entry catalog fingerprint, 866 included text/MTP tensors,
and 333 excluded vision tensors. A different config or catalog fails closed.

The caller explicitly supplies the host-memory budget and five simultaneous
reservations: source window, codec workspace, writer chunk, evidence buffer,
and fixed overhead. Checked unsigned addition computes `peakHostBytes` before
work begins. The peak must fit the supplied budget, and the budget is capped at
at most 64 GiB. The writer chunk is capped at 1 MiB and the optional evidence
buffer at the SQW 64 MiB limit. Zero evidence or fixed-overhead reservations
are permitted and remain explicit.

Conversion concurrency is fixed to one worker, one open source shard, and one
in-flight tensor. `maxArtifactBytes` is a storage extent, not resident memory;
it is positive and within the SQW signed-safe 64-bit bound but is not added to
the host peak. This allows an output larger than RAM without treating it as
resident.

The SHA-256 plan fingerprint covers the exact source/catalog identities,
counts, concurrency, budget, reservations, derived peak, and artifact bound in
a versioned canonical line format. Equal inputs produce the same fingerprint;
any resource-policy change produces a different identity.

The plan owns only its fingerprint. `close` releases it, clears the scalar
geometry, and is idempotent. Access after close returns the stable,
non-retryable `qwen.conversion.closed` diagnostic. Invalid source identity,
limits, arithmetic, state, or aggregate admission return stable
`qwen.conversion.*` errors without repair, retry, or fallback.

Later consumers call `validate` immediately before opening source or output
resources. It recomputes identity, concurrency, all bounds, peak geometry, and
the fingerprint, rejecting any caller-modified public field before I/O.

This leaf does not open a shard, read or convert tensor bytes, classify a
payload, exclude data during I/O, write a journal, build SQW bytes, promote an
artifact, or select conversion data. It does not choose a codec. Those
operations belong to later QWN-032 and QWN-033 leaves. It initializes neither
Python nor CUDA and executes no remote model code.
