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
