#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/v0.20.8/extracted/seen-0.20.8-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_044b"

if [ "${1:-}" != "--inner" ]; then
    exec env -u PATH -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH \
        -u SEEN_RUNTIME_PATH -u SEEN_COMPILER_SOURCE_ROOT \
        -u SEEN_PACKAGE_CLIENT QWN_TASKS_MAX=32 \
        "$ROOT_DIR/scripts/oracle/run_bounded.sh" 7200 \
        /usr/bin/env PATH=/usr/bin:/bin QWN_044B_HARD_SCOPE=1 \
        SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" "$0" --inner
fi

[ "${QWN_044B_HARD_SCOPE:-0}" = 1 ] || exit 126
SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
mkdir -p "$ARTIFACT_ROOT"
IR_ROOT=$(mktemp -d "$ARTIFACT_ROOT/ir-v0.20.8.XXXXXX")
mkdir -p "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"

printf '%s  %s\n' 76833346fbe3e01cda0aeb2b34d585a2115086ccee3e87205059b055d22ac2b6 "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' 9cfeeb645ed31f51d2a32f3348b9e586027363a7e2085c2316a793f132049b7e "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 6bc6cc29032f834035a0c65391ddee25069b5fb3496f24a33a0c9eaf4eada7ee "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.8'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.8 version | grep -Fx 'seen-pkg 0.20.8 (SEENPKG1)'
readelf -n "$SEEN_BIN" | grep -F 'Build ID: 33dfad8d2671de0176409fa1c197a3c4c2a03329'

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

python3 -m unittest tests/test_qwen_sampler.py
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_BIN" check tests/qwn_044b_sampler_test.seen --frozen
"$SEEN_BIN" compile tests/qwn_044b_sampler_test.seen \
    "$ARTIFACT_ROOT/qwn_044b_ir_only" --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen \
    --emit-module-ir-dir "$IR_ROOT" --stop-after-ir
rng_ir=$(grep -l 'define .*qwenSamplerNext24' "$IR_ROOT"/*.ll)
[ -n "$rng_ir" ] || exit 1
sed -n '/define .*qwenSamplerNext24/,/^}/p' "$rng_ir" > "$IR_ROOT/qwen_sampler_rng.ll"
[ "$(grep -c 'lshr i64' "$IR_ROOT/qwen_sampler_rng.ll")" -ge 3 ]
[ "$(grep -c 'ashr i64' "$IR_ROOT/qwen_sampler_rng.ll")" -eq 0 ]

"$SEEN_BIN" compile tests/qwn_044b_sampler_test.seen \
    "$ARTIFACT_ROOT/qwn_044b_sampler_fast" --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_044b_sampler_fast"
"$SEEN_BIN" compile tests/qwn_044b_sampler_test.seen \
    "$ARTIFACT_ROOT/qwn_044b_sampler_release" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_044b_sampler_release"
"$SEEN_BIN" compile tests/qwn_044b_sampler_test.seen \
    "$ARTIFACT_ROOT/qwn_044b_sampler_ubsan" --sanitize undefined \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$ARTIFACT_ROOT/qwn_044b_sampler_ubsan"

# QWN-044B is the declared host sampler. Re-run the producing RTX 4090 LM-head
# gate to prove the borrowed host-visible logits source, stream ordering,
# graph behavior, device failures, and CUDA sanitizer/leak contract.
env QWN_044A_HARD_SCOPE=1 SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
    "$ROOT_DIR/scripts/cuda/run_qwn_044a.sh" --inner

sha256sum "$IR_ROOT"/*.ll "$ARTIFACT_ROOT/qwn_044b_sampler_fast" \
    "$ARTIFACT_ROOT/qwn_044b_sampler_release" \
    "$ARTIFACT_ROOT/qwn_044b_sampler_ubsan"
toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || exit 126
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || exit 126
echo "PASS: QWN-044B v0.20.8 deterministic official sampler, RTX 4090 source, sanitizer, bounds, and cleanup gates"
