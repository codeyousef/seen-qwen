# Qwen tensor-catalog contract

The `seen_qwen.model.tensor_catalog` module consumes only the strict
safetensors index for `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Input is bounded to 1 MiB and
2,048 entries before catalog allocation.

The shared Seen safetensors index parser validates and owns cloned tensor and
local shard names, then sorts them lexicographically. Qwen classification is
exclusive:

- `model.language_model.*` and `lm_head.weight` are required text tensors;
- `mtp.*` tensors are required MTP tensors;
- `model.visual.*` tensors are excluded vision tensors and must not be
  extracted by project zero;
- every other name fails closed as unsupported.

The stable category labels are `text`, `mtp`, and `vision`. Extraction policy
is queried separately so the `vision` category remains visible while returning
false from `shouldExtractAt`.

The pinned catalog contains 851 text, 15 MTP, and 333 excluded vision tensors.
Its canonical `category<TAB>name<TAB>shard<LF>` representation has SHA-256
`5f466d43bae3059e54f0bfe183d0e82c822242f45a834d778414d3e5b5248f1f`.
This locks all 1,199 names and shard assignments rather than trusting counts or
prefixes alone.

Catalog strings returned by accessors are borrowed and remain valid only until
`close`. The catalog owns the shared shard index, category array, and
fingerprint; `close` releases them deterministically and is idempotent. Closed
or out-of-range access returns a stable typed error.

This leaf does not inspect shard payloads, derive dtype/shape geometry, load
weights, or convert SQW data. Those operations belong to later work packages.
