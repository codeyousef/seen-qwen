# Qwen deterministic sensitivity statistics contract

FEL-1426 / QWN-033B supplies bounded CPU reference statistics for the frozen
QWN-033A calibration inputs. It does not choose a quantization policy or claim
that any codec meets a model-quality threshold.

`measureActivationStatistics` consumes a non-empty caller-owned finite vector
and an explicit positive element limit. It reports count, minimum, maximum,
maximum absolute value, arithmetic mean, mean absolute value, and root mean
square. `measureErrorStatistics` consumes equal-length caller-owned reference
and candidate vectors and reports maximum/mean absolute error, RMS error,
relative L2 error, cosine similarity, and exact equality.

Both reductions are strictly sequential in input order. There is no parallel
reduction, reassociation, random sampling, implicit dtype change, retry, repair,
or fallback. Inputs exceeding their bound, empty or mismatched vectors,
non-finite values, and non-finite intermediate products or sums fail with stable
`qwen.statistics.*` diagnostics before a result is returned.

Relative L2 is undefined when the reference norm is zero. Cosine similarity is
undefined when either norm is zero. The result retains explicit
`relativeL2Defined` and `cosineDefined` booleans; the numeric field is zero only
as a non-measurement sentinel and must not be interpreted unless its flag is
true. This avoids inventing perfect similarity for two zero vectors.

Inputs remain owned by the caller and may be released immediately after the
function returns. Result objects own no borrowed storage. `close()` clears every
scalar and flag and is deterministic and idempotent.

The checked-in `seen-qwen-sensitivity-oracle-v1` fixture pins the model revision,
QWN-033A calibration-lock digest, record ordering, explicit element bound, and
small activation/tensor/layer golden vectors. Its closed schema rejects unknown
fields and bounds record/vector cardinality. Model-level perplexity/logit
quality, memory estimation, kernel compatibility, profile resolution, and GPU
evidence belong to later QWN-033/QWN-034 leaves.
