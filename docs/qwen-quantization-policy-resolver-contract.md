# Qwen bounded quantization-policy resolver contract

FEL-1427 / QWN-033C resolves explicit, already-validated quantization policy
rules. It does not select or approve a production profile, derive a codec from
statistics, or claim model quality. Those decisions require later evidence.

Each tensor supplies a canonical name, layer index (`-1` for an unlayered
tensor), layer type, semantic role, and outlier bucket. Each rule supplies an
identifier, codec identifier, and any combination of exact tensor name, layer
range, layer type, semantic role, and outlier-bucket selectors. Empty string
selectors are wildcards; layer range is a wildcard only as `-1/-1`.

Rules are validated completely and evaluated in declared array order. The
first matching rule wins. There is no hidden specificity ranking, inferred
codec, repair, retry, or implicit fallback. Policy configuration therefore
owns ordering and should put narrow rules before broad rules. A tensor with no
matching declared rule fails with `qwen.policy.unmatched`.

The resolver rejects empty inputs, excessive caller or hard limits, layer
indexes outside the frozen 64-layer model, non-canonical tokens, duplicate
rule IDs, duplicate tensor names, and identical rule predicates. Hard caps are
2,048 tensors and 4,096 rules; callers must provide equal or tighter positive
limits. Matching is a bounded sequential scan and creates exactly one result
per input tensor.

The resolved array stores the input tensor index, matched rule index, tensor
name, rule ID, and codec. String fields borrow from the caller-owned profile,
tensors, and rules, which must remain alive and unchanged until the resolved
policy is closed. The result owns only its resolution array. `close()` releases
that array deterministically and idempotently; access after close fails.

The checked-in oracle is deliberately named `synthetic-resolver-test`. Its
codec assignments test mechanics only and are not a shipped `speed`,
`balanced`, `native-long`, or `extreme` profile. `configs/quantization.toml`
remains the eventual non-executable owner of evidence-approved named policies;
this leaf does not invent those choices or a TOML parser.
