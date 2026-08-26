# Seen Qwen agent guide

## Scope and project state

These instructions apply to this entire repository. A more deeply nested
`AGENTS.md` overrides them for its subtree.

This is the standalone implementation repository for the text-only
Qwen3.8-27B inference proof on one RTX 4090. The repository intentionally starts
with no implementation code. The first dependency-ready Linear work packages
create the code and package topology deliberately; do not copy the historical
monorepo scaffold wholesale.

There is no configured Git remote. Do not create a remote, push, publish, or
change repository visibility without explicit owner authorization.

## Authority order

Resolve conflicts in this order:

1. A later written owner decision recorded in Linear.
2. The linked Linear issue, its latest comments, relations, and accepted evidence.
3. `docs/private/seen_qwen_spec_pack/01_DECISIONS_AND_SCOPE.md`.
4. Qwen documents 04 through 13 in the private pack.
5. `docs/private/seen_qwen_spec_pack/20_CODEX_EXECUTION_PROTOCOL.md`.
6. Immutable source/model/environment locks and their recorded upstream sources.
7. Focused tests covering current implementation behavior.

Documents 00, 02, 03, and 22 preserve source-pack history. Their Seen v0.13
baseline, prerequisite Backlog states, aggregate SeenLang project count, and
`projects/seen_ml/qwen38` placement are historical. The completed prerequisite
baseline is Seen v0.14.0 at
`db488156b9cd666f8ebd5a6928141ef79b7dc1f3`. This repository root supersedes
the historical monorepo placement for new Qwen work.

Do not invent model semantics, tensor mappings, formats, precision rules,
fallbacks, public APIs, or benchmark policy. Use an authoritative source,
frozen decision, existing conformance test, prescribed experiment, or request
an owner decision in Linear.

## Linear workflow

Linear project `Seen Qwen` is canonical. Existing FEL identifiers must be
preserved. At the start of a session:

1. Read the issue and latest comments.
2. Read its parent, direct blockers, and relevant accepted evidence.
3. Read this file and the issue's referenced private-pack documents.
4. Confirm the repository branch, exact commit, dirty state, and Seen dependency lock.
5. Confirm every direct blocker is terminal.
6. Also require the parent capability tracker to be unblocked and active; Linear
   does not automatically propagate a parent's blockers to its children.
7. State permitted files, acceptance tests, hardware requirement, and evidence outputs.

Implement one dependency-ready atomic leaf per branch and PR. Use
`codex/FEL-<number>-<short-slug>`, `[FEL-<number>]` commits, and an atomic PR to
`main`. Trackers, reviews, and decisions are branchless unless the issue
explicitly authorizes remediation or reusable certification tooling.

The entry tracker is FEL-1214 / QWN-020; its first bounded review is FEL-1393 /
QWN-020A. Do not skip phase or capability gates because a leaf appears
individually unblocked.

An issue is Done only after implementation and exact evidence are complete,
the branch is merged to `main`, required merged-main CI is green, and Linear
contains commands, artifacts, hardware/environment identity, PR, merge SHA,
and CI evidence. Reviews and decisions require their stated owner approval.

## Cross-repository ownership

SeenLang remains the owner of shared language, compiler, standard-library,
runtime, reusable accelerator ABI, and release-toolchain behavior. Consume it
through an exact released source/toolchain lock.

If a Qwen issue exposes a genuine SeenLang defect or missing shared contract:

- use the existing related FEL issue when one exists;
- otherwise obtain a bounded SeenLang issue before changing upstream code;
- implement, test, merge, and certify the change in SeenLang;
- update this repository's dependency lock only after the accepted upstream result;
- never copy or fork a shared Seen subsystem into this repository to bypass the gate.

Reusable resource symbols use `seen_cuda_*`. Model-specific kernel symbols use
`seen_qwen_*`. A foreign shim is the smallest ledgered C ABI adapter and owns no
allocation policy, scheduling, caching, fallback, model semantics, or protocol.
No C++/STL type or Seen string crosses the ABI; use fixed-width scalars, byte
views, explicit lengths, typed statuses, and opaque handles.

## Frozen product contract

- Official Qwen3.8-27B text generation is project zero.
- Target Linux x86-64, CUDA, RTX 4090, Ada/SM89.
- Include language weights, LM head, tokenizer/chat/sampling semantics, and MTP.
- Exclude the vision tower and all image/video preprocessing.
- Keep Qwen-specific model/state/linear/cache abstractions until the G5 decision.
- Python and PyTorch are allowed only for pinned oracle and conversion tooling;
  the production runtime must not initialize Python, PyTorch, or LibTorch.
- Safetensors is canonical input. SQW is deterministic, versioned, checksummed,
  reproducible derived storage, never the only exchange format.
- Use vendor primitives when they win, but certify at least one material
  Seen-controlled or Qwen-specific hotspot with end-to-end attribution.
- `serve` is deferred and non-release-blocking.

No result may silently change backend, model revision, precision, codec,
context, batch, MTP, graph execution, residency, or offload policy. Every
fallback is explicit, semantically equivalent, diagnosed, counted, and labeled
in evidence. A GPU-resident claim may not hide repeated host work or active
weight/cache/state offload.

## Source, security, and artifacts

Use only pinned official model/tokenizer/config sources. Record immutable
revisions, licenses, and SHA-256 digests. Prohibit pickle, arbitrary remote
code, executable model repositories, and `trust_remote_code` behavior in the
production path.

Treat JSON, manifests, paths, shard metadata, tensor geometry, native statuses,
and device data as hostile. Validate before allocation or execution. Use
checked 64-bit arithmetic, bounded windows/queues/assets, deterministic cleanup,
atomic promotion, and fail-closed diagnostics.

Track source, schemas, small golden vectors, deterministic manifests, tests,
and bounded evidence summaries. Keep model weights, converted SQW, cubins/PTX
unless intentionally tiny golden fixtures, baseline builds, profiler traces,
large raw samples, temporary files, and secrets under ignored project-local
artifact roots. Audit staged files before every commit.

## Build and memory safety

Never run an uncapped build, test, compiler bootstrap, native build, model run,
or benchmark. Before execution:

- read current total and available memory;
- use the current Seen hard-scope wrapper or an equivalent repository-certified wrapper;
- cap aggregate memory at the derived safe value and never above 64 GiB;
- set aggregate swap to zero and read it back;
- bound tasks, per-command virtual memory, and wall time;
- use serial compiler, optimizer, native, package, and test workers;
- record cgroup/scope identity, limits, peaks, OOM events, task-limit events, and exit status.

Containment failure is a test failure. Do not raise limits blindly or retry a
failed full build before diagnosing the first real error.

## Correctness, hardware, and performance evidence

Run evidence in this order:

1. source/static and schema policy;
2. focused unit and malformed-input tests;
3. bounded fuzzing;
4. CPU/reference and mini-model differential evidence;
5. CUDA mini-model and sanitizer evidence;
6. full-model correctness;
7. qualified performance/context experiments;
8. affected repository and release gates.

Use maturity states `unsupported`, `compile-only`, `experimental-hardware`,
`verified`, and `production-certified`. Compile-only evidence never closes a
hardware leaf. RTX 4090 leaves require exact GPU UUID/model, SM, driver, CUDA,
library/toolchain, clocks/power/thermal state, command, and artifact hashes.

T0 through T5 evidence is release-blocking. Performance claims require equal
model/source, quality, prompt, context, sampling, residency, and hardware;
qualified strongest baselines; correctness first; warm-up policy; 30 measured
samples; bandwidth, host memory, VRAM, transfers, and fallback counters; and
the applicable hard five-percent gate. Final certification uses three thermal
sessions. Record negative and null results.

The 512K and 1M experiments terminate on success, explicit bounded skip, or
kill evidence; success is not required. Native-262K, default codec/profile,
public claims, and the G5 proceed/pivot/stop decision require their stated owner review.

## Release discipline

The target fixed-point release is v0.15.0. Before its single tag push, certify
the exact final clean tree locally with CI and release parity under the hard
scope, merge it to `main`, and verify exact-SHA merged-main CI. Create one
annotated tag only after that run succeeds. Do not poll CI or create concurrent
runs. After completion, verify the exact tag peel, release identity, unique
assets, checksums, signatures, source digest, package versions, CPU baseline,
and installed compiler/package smoke before reporting the release verified.

## Session handoff

End every meaningful session with a Linear comment containing:

```text
Completed:
- ...

Current blocker:
- None | exact blocker

Next exact step:
- ...

Branch:
- ...

Last validation run:
- command — result
```

Do not store private progress diaries or handoff notes in tracked public files.
Do not add co-author trailers unless explicitly requested.

