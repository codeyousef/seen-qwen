# Qwen cuBLASLt projection selection contract

FEL-1445 maps a caller-owned token-major row-major input `X[tokens,input]`
and weight `W[output,input]` to `Y[tokens,output]`. The packaged Seen
cuBLASLt ABI observes the same bytes as column-major views and computes
`Y^T = W * X^T`; the descriptor is therefore `(m,n,k) =
(output,tokens,input)`, transpose-A, with leading dimensions
`(input,input,output)`.

Only F16 and BF16 are admitted. Dimensions must be non-zero and fit the
packaged 31-bit cuBLASLt bound; the workspace cap must fit signed 64-bit.
Invalid geometry is rejected before CUDA is called. There is no allocation,
fallback, implicit default stream, or synchronization in descriptor creation
or selection.

The selection object owns only copied descriptor and algorithm metadata. It
borrows a live, device-matched Seen `CublasLtHandle` during selection and never
stores, closes, or decodes that handle. A selection is reusable only for an
exact device and descriptor match, is never serialized across a process,
driver, device, or toolchain boundary, and is deterministically invalidated by
an idempotent `close`. Buffer ownership and execution belong to QWN-043B.

Hardware qualification covers official Qwen widths, deterministic repeated
selection, workspace bounds, distinct descriptor identities, invalid handles,
invalid descriptors, and deterministic handle cleanup on an RTX 4090.
