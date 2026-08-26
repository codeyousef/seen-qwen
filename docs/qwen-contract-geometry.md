# Qwen contract geometry

`seen_qwen.model.contract_geometry` validates that parsed configuration and
catalog objects still represent the pinned Qwen3.8-27B text contract before it
derives any memory extent. Forged geometry, catalog drift, and use-after-close
fail with typed, stable, non-retryable diagnostics.

The calculation uses Seen's shared checked 64-bit geometry operations. It
reports:

- 32,768 full-attention K/V scalar values per token;
- context and batch-scaled K/V payload bytes for an explicit scalar width;
- 48 GDN layers of recurrent state shaped as 48 value heads by 128 key
  elements by 128 value elements, fixed to the required float32 contract;
- causal-convolution state over two key projections plus one value projection,
  four positions, 48 GDN layers, and an explicit storage width;
- checked persistent-state and K/V-plus-state sums.

For batch one at the native 262,144-token limit with two-byte K/V and
convolution scalars, the ideal scalar payload is 17,334,796,288 bytes:
17,179,869,184 K/V bytes plus 150,994,944 recurrent-state bytes and 3,932,160
convolution-state bytes.

These are contract scalar extents, not a runtime memory plan. They deliberately
exclude weight payloads, codec metadata/scales, padding and alignment, page
tables, activation buffers, workspaces, graph/runtime overhead, telemetry, and
the required safety reserve. Later planner work must add those measured terms
and reject any checked sum beyond its configured budget.

The result owns only scalar values. Configuration and catalog inputs are
borrowed for the duration of the call and are not retained. No allocation,
fallback, retry, CUDA discovery, or device execution occurs.
