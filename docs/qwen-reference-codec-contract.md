# Qwen BF16 and F16 reference codec contract

`seen_qwen.quant.reference_codec` is the deterministic CPU reference for the
two initial 16-bit weight codecs. It is conversion and correctness machinery,
not an optimized compute kernel or a production codec-selection decision.

Both encoders first round the Seen `Float` input to IEEE binary32 and then use
round-to-nearest, ties-to-even. `BF16` retains the upper 16 binary32 bits after
rounding. `F16` uses IEEE 754 binary16 with five exponent bits and ten stored
fraction bits, including gradual underflow. Signed zero and finite subnormals
are preserved. NaN, infinity, binary32 overflow, and conversions that would
produce a 16-bit infinity fail closed; gradual underflow follows the target
IEEE format, and there is no saturation or precision fallback.

Scalar APIs consume the shared `BFloat16` and `Float16` storage types and
return a `ReferenceFloatValue` wrapper for decoded values. The wrapper avoids
the unsupported Seen 0.18.1 generic `Result<Float, E>` ABI tracked by SeenLang
FEL-1550 without changing codec semantics. Bounded buffer APIs copy caller
values into an owned `UInt16` payload,
retain the canonical codec ID (`BF16` or `F16`), and require an explicit
positive element limit for both encoding and decoding. A decoder rejects a
codec mismatch rather than reinterpreting it. `Reference16Buffer.close()`
releases the payload deterministically, is idempotent, and invalidates future
access.

Diagnostics are stable, non-retryable, and use these codes:

- `qwen.codec.input` for a non-finite or otherwise invalid source/payload;
- `qwen.codec.range` when finite conversion would overflow;
- `qwen.codec.limit` when a caller's element bound is exceeded;
- `qwen.codec.mismatch` when a buffer reaches the wrong decoder;
- `qwen.codec.closed` for use after deterministic cleanup.

The implementation is native Seen and has no C/C++ ABI, CUDA dependency,
allocation fallback, retry path, asynchronous work, or cancellation wait.
Decoded arrays are owned by the caller and must be freed. No default
quantization/profile choice is made by this contract.
