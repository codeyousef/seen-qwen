# Official operator and layer oracle contract

`tests/fixtures/qwn_025a_operator_layer_oracles.json` is the bounded O1 CPU
reference corpus for FEL-1408 / QWN-025A. It is generated from the immutable
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and exact Transformers commit
`562cfd944ee1f20702cfb0f4404014ee27c24813` with CPU-only PyTorch
`2.11.0+cpu`.

The corpus covers the first, middle, and last Gated DeltaNet layers (0, 32,
60) and full-attention layers (3, 31, 63). Each layer records bounded sample
vectors plus a digest of the complete captured tensor for its deterministic
input, normalization, token mixer, MLP, and layer output. GDN entries also
record the complete recurrent-state digest and bounded state samples. The
input positions are 0 and 31, and the deterministic BF16 input formula is
recorded in the corpus.

Source weights are never checked in. `tools/fetch_official_layer_ranges.py`
validates the checked-in tensor index, immutable HF revision, shard sizes, and
LFS SHA-256 identities, then downloads only six contiguous layer byte ranges
into ignored `.seen/oracle-official/layers/`. It does not download or execute
model-repository code. `tools/capture_official_operator_layer_oracles.py`
imports the separately audited local Transformers checkout, verifies the
modeling source hash, uses one CPU worker and deterministic algorithms, and
writes the small checked-in corpus.

The corpus maturity is `verified-cpu-reference`. It is an operator and layer
correctness oracle, not full-model logits, CUDA evidence, a performance claim,
or release-consumer certification. FEL-1411 owns full-model prompt/logit and
tolerance contracts.
