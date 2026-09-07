# First complete Qwen text and MTP artifact

FEL-1429 / QWN-034A defines the first complete offline-validated engine
artifact. It contains all 851 required text tensors and all 15 MTP tensors in
canonical UTF-8 order and excludes all 333 vision tensors. Safetensors remains
the canonical source; `weights.sqw` is a deterministic derived artifact.

The bring-up policy is `bf16-bringup-v1` from
`configs/quantization.toml`. Every admitted source tensor must be BF16 and is
copied bit-for-bit into the BF16 SQW codec. This policy is deliberately
lossless and correctness-only. It is not a production/default profile, a
quality waiver, a speed claim, or evidence that the artifact fits an RTX 4090.
There is no fallback: an absent, extra, reordered, non-BF16, corrupt, or
incompatible input fails before promotion.

The ignored artifact directory contains `engine.json`, `weights.sqw`, the six
locked tokenizer/configuration assets (seven files, including the tokenizer
configuration), `conversion-evidence.json`,
`source-lock.json`, `model-lock.json`, and `checksums.sha256`. Construction is
serial and streaming. At most one shard and one bounded byte chunk are active.
The staging directory is independently validated and atomically promoted only
after all file identities, the complete 866-entry SQW catalog, the embedded
conversion evidence, and checksums agree. Existing destinations are preserved
on failure and cleanup is deterministic.

`engine.json` uses `seen-qwen-engine-artifact-v1` and the maturity
`offline-validated`. Its intended CUDA target is SM89 but `bound` is false and
`compatible_profiles` is empty because Q3 precedes CUDA bundle construction,
runtime memory sizing, hardware execution, and profile certification. Later
issues must create those identities rather than interpreting zero or absent
runtime evidence as success.

The offline streaming builder is certification/conversion tooling and may use
Python under the frozen project contract. Runtime validation remains native
Seen and never initializes Python, PyTorch, LibTorch, CUDA, or remote model
code.
