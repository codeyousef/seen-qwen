# Qwen sampling profile contract

`profiles/sampling.toml` is the immutable sampling-policy source for the
pinned `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The thinking and instruct
values are copied from that revision's official model card. The thinking
values also agree with the pinned `generation_config.json`.

The Seen loader accepts only the four named profiles, rejects duplicate or
unknown fields, and validates every value before returning a profile. Sampling
profiles require an explicit non-negative signed 64-bit seed. Greedy mode does
not consume a seed. The custom profile is a neutral starting point; callers may
replace its values only through the same bounded validator.

Custom sampling bounds are `(0, 2]` for temperature, `(0, 1]` for top-p,
`[1, 248320]` for top-k, `[0, 1]` for min-p, `(0, 2]` for repetition penalty,
`[-2, 2]` for presence penalty, and `[0, 2^63-1]` for the seed. NaN and
infinity are rejected. The profile file is an artifact input: changing it
changes the engine input identity and must never happen implicitly after an
artifact is built.

The first correct runtime sampler is Seen-owned and operates on the final
host-visible logits row. It applies distinct-token repetition penalty, then
distinct-token presence penalty, temperature, top-k, top-p, and min-p. The
top-k boundary retains every token tied with the kth score. Ranking ties use
the lower token ID; categorical accumulation uses ascending token ID. Top-p
retains the boundary token that crosses the requested mass and every filter
retains at least the best token.

Sampling uses the project-frozen xorshift64* transition with a 24-bit uniform
numerator. Seed zero maps to `0x9E3779B97F4A7C15`; all other valid seeds are
used exactly. Each successful sampling selection consumes one draw. Validation
errors consume none, reset reproduces the initial stream, greedy consumes none,
and close is deterministic and idempotent. Inputs are borrowed; sampler state
owns only scalar seed/state/counter values. GPU-resident sampling remains a
later measured optimization, so this contract introduces no CPU fallback: the
host sampler is the declared QWN-044B execution path.

The execution boundary revalidates copied policy values because public Seen
objects remain mutable after profile parsing. It caps vocabulary geometry at
248,320 and admitted history at 262,144 tokens before temporary allocation.
Operation-owned score, rank, membership, and distinct-history arrays are freed
on every success or error path; no borrowed input is retained after return.
