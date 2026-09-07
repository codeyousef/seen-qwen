# Qwen calibration corpus and provenance lock

FEL-1424 / QWN-033A freezes the inputs admitted to calibration. The canonical
lock is `configs/calibration.lock.json`; its closed schema is
`schemas/qwen-calibration-lock.schema.json`. The lock is an input identity, not
a quality result, quantization decision, or permission to run model-repository
code.

The corpus combines the already certified QWN-025B full-model prompt corpus
with six small project-authored token recipes. The QWN-025B source supplies
natural English, Arabic, Seen code, ordered structure, tool context, and paired
thinking-on/thinking-off prompts. The supplement supplies a multilingual
probe, bounded high- and low-variance activation probe candidates, a 32K-token
long-document recipe, and deterministic 128K- and native-262K retrieval
recipes. “High” and “low” name input candidates only; FEL-1426 must measure and
report activation statistics before either label can become an empirical
claim.

Every source has a repository-relative path, exact byte length, SHA-256
revision, license, and explicit not-executed-at-runtime/non-private classification.
Model and tokenizer revisions are the immutable official
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Remote model code, pickle,
network locations, executable expressions, environment-dependent paths, and
private user data are not admitted.

Samples and sources are sorted by ID. Sample identities bind either the exact
rendered UTF-8 digest from QWN-025B or the SHA-256 of a canonical token-recipe
record. Recipes use only token IDs proven by the locked QWN-022B tokenizer
oracle, validate every token against vocabulary size 248,320, and cap any one
sample at the native 262,144-token context. The complete lock contains 14
samples and 426,873 tokens; recipe materialization is serial and bounded to
8 MiB of encoded token storage.

`seen_qwen.quant.calibration.parseQwenCalibrationLock` accepts strict JSON with
duplicate keys prohibited, indexed object traversal, closed field sets, fixed
cardinality, exact ordering, exact identities, and bounded geometry. It keeps
the parsed document alive for as long as borrowed source or sample IDs may be
returned. `close()` releases that document and the owned lock digest exactly
once; access after close and out-of-range indexes fail with stable
`qwen.calibration.*` diagnostics.

The Python fixture oracle independently checks source bytes and hashes, recipe
canonicalization, token bounds, insertion bounds, non-overlap, coverage, and
closed-schema behavior. Calibration statistics, per-tensor errors, policy
selection, GPU execution, and quality claims belong to later QWN-033 leaves.
