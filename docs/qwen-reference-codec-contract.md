# Qwen BF16, F16, and Q8 reference codec contract

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

## Q8_SYM_G64

`encodeQ8SymG64Buffer(values, rowElements, elementLimit)` treats the input as
complete row-major rows and resets grouping at every row boundary. Each group
contains at most 64 logical FP32 values. The API requires a positive row width
and element limit, rejects incomplete rows, and checks group and scale geometry
before allocating output.

For each group, the encoder first rejects non-finite input and rounds every
source value to binary32. It computes `max_abs` over only the group's logical
values. An all-zero group stores FP16 zero scale and zero codes. Otherwise it
computes the binary32 scale `max_abs / 127`, requires that scale to encode as a
finite nonzero FP16 value, and quantizes each binary32 value against the
computed binary32 scale. Quantization uses deterministic round-to-nearest,
ties-to-even, then clamps to `[-127, 127]`. Code `-128` is never emitted.

The payload owns one signed two's-complement byte per logical value and owns one
FP16 scale per group. A final short group is conceptually zero-padded for group
semantics, but the payload stores no padding bytes and retains the logical row
width and element count. `decodeQ8SymG64Buffer` interprets each code as signed,
rejects `-128`, rejects negative or non-finite scales, rejects nonzero codes
paired with a zero scale, and reconstructs `float(code) * float(fp16_scale)`.
It validates the codec identity, row geometry, payload length, scale count, and
caller limit before allocating decoded output.

`ReferenceQ8Buffer` owns both arrays. `close()` frees them, clears geometry,
is idempotent, and invalidates future decoding. The decoded `Array<Float>` is
owned by the caller. In addition to the shared diagnostics above, Q8 uses
`qwen.codec.geometry` for non-canonical row, payload, or scale geometry.

This is a CPU correctness reference. It defines no automatic codec selection,
CUDA behavior, performance claim, retry, repair, padding storage, zero point,
or fallback.
