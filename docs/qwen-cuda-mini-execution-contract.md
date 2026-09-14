# Qwen CUDA mini-model execution contract

QWN-045B composes the frozen eight-layer hybrid mini model on one Seen-owned
CUDA stream. Layers 3 and 7 use full attention; the other six layers use Gated
DeltaNet. Prefill and incremental decode use the same resident weights, fixed
scratch arena, convolution histories, recurrent states, attention KV caches,
and greedy LM-head result path. There is no silent fallback, default-stream
launch, per-step allocation, device-wide synchronization, Python, PyTorch, or
LibTorch initialization.

The model input remains the content-locked Safetensors fixture. The complete
file is validated and copied into one device allocation before execution;
bounded tensor views retain its exact data offsets. All allocations occur
before execution. Each Qwen kernel borrows a launch token from one Seen-owned
stream for the immediately nested enqueue and never retains or destroys it.
Result-boundary event synchronization makes the sampled token and last-row
logits visible before state is committed. Reset clears all persistent device
state deterministically; cancellation rejects new work; close releases IDs,
state, scratch, weights, event, and stream in reverse ownership order and is
idempotent.

Deterministic allocation-failure injection rejects each of the four device
allocation steps in turn. Every partial construction releases all earlier
allocations plus the event and stream before returning failure; repeated close
remains safe. CUDA memcheck certifies those simulated OOM paths leak no device
or host resource without consuming the machine's remaining VRAM.

Three small model-specific FP32 adapters complete the already ledgered
primitive surface: bounded row-major linear projection, GDN convolution-output
preparation, and GDN scalar-parameter transformation. They accept only fixed
width views and a borrowed stream token, validate geometry, allocation extent,
device, and non-overlap before enqueue, and own no allocation, scheduling,
fallback, or model lifecycle policy.

The native executable is a conformance harness, not a production policy owner.
It mirrors the numeric plan fixed by the Seen `QwenExecutionPlan` and proves the
separately built kernels compose correctly. Runtime ownership and fallback
policy remain in Seen. Required local verification compares every one of the
four last-token 256-logit rows and greedy IDs against the content-addressed CPU
oracle, exercises prefill, three incremental decodes, reset, cancellation,
repeat, and idempotent teardown, and runs CUDA memcheck, initcheck, racecheck,
and synccheck on the RTX 4090.
