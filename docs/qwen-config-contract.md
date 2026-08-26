# Qwen configuration contract

The `seen_qwen.model.config` module accepts only the pinned text architecture
of `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Parsing is bounded to 1 MiB of strict UTF-8 JSON, 24 levels, 8,192 values, and
64 KiB strings. Duplicate keys and malformed JSON fail before model fields are
used. The parser validates the model identity, 64-layer 3:1 linear/full
attention schedule, full-attention and RoPE geometry, Gated DeltaNet geometry,
FFN/vocabulary sizes, native context, and MTP layer count. It never repairs or
defaults an absent or incompatible semantic field.

The returned `Qwen38Config` owns scalar geometry only; it retains no view into
the parsed JSON. The parsed document is destroyed on every success and failure
path. Layer lookup accepts only indexes 0 through 63 and returns a typed error
outside that range.

This contract does not classify checkpoint tensors, calculate allocation byte
plans, load weights, execute remote model code, or provide a compatibility
fallback. Those capabilities belong to later QWN-021 leaves.
