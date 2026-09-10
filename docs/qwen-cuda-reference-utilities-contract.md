# Qwen CUDA reference utility contract

QWN-040B extends the experimental-hardware QWN-040A CUDA surface with the
remaining bounded reference utilities required by later semantic-engine work:
offset-weight RMSNorm, row L2 normalization, SiLU, SwiGLU, sigmoid gating,
partial rotate-half text RoPE, contiguous KV append, greedy argmax, and
deterministic top-k.

The FP32 formulas and tie behavior match the existing Seen CPU reference.
RMSNorm uses `1 + weight`, RoPE uses the pinned Qwen text position and
rotate-half contract, and equal sampling logits select the lower token ID.
Top-k is a deliberately scalar deterministic oracle, not a production sampler;
seeded top-p/min-p and penalties remain owned by their later sampling issue.

Every function consumes one immediately nested
`seen-cuda-stream-launch-token-v1` borrow and enqueues only on that exact
Seen-owned stream. The Qwen adapter does not retain or decode handles, allocate,
synchronize, destroy resources, select a default stream, or choose fallback or
policy. Inputs and outputs are fixed-width borrowed device views. Geometry,
position/capacity, allocation domain, byte extent, and unsupported overlap are
validated before launch. KV cache allocation and the authoritative used length
remain owned by the caller; the adapter performs only the explicitly positioned
copy.

Run `scripts/cuda/run_qwn_040b.sh` for the complete gate. It verifies the exact
Seen v0.20.4 release payload, checks and compiles the Seen declaration surface,
builds serially for SM89, runs CPU/CUDA differential, boundary, same-stream,
graph-capture, and deterministic-cleanup coverage, and executes CUDA memcheck,
initcheck, racecheck, and synccheck. All generated files remain under ignored
`.seen` paths and the runner proves that the toolchain and tracked-area object
inventories are unchanged.
