#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/seen-0.20.4-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_042b"

if [ "${1:-}" != "--inner" ]; then
    exec env -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH -u SEEN_RUNTIME_PATH \
        -u SEEN_COMPILER_SOURCE_ROOT -u SEEN_PACKAGE_CLIENT \
        QWN_TASKS_MAX=32 "$ROOT_DIR/scripts/oracle/run_bounded.sh" 4200 \
        env QWN_042B_HARD_SCOPE=1 SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
        "$0" --inner
fi

[ "${QWN_042B_HARD_SCOPE:-0}" = 1 ] || exit 126
SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
BUILD_ROOT="$ROOT_DIR/.seen/artifacts/qwn_041c/build"
printf '%s  %s\n' 79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c "$SEEN_BIN" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.4'
mkdir -p "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
python3 -m unittest tests/test_cuda_kv_cache.py
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_042b_kv_cache_test.seen" --frozen
"$SEEN_BIN" compile "$ROOT_DIR/tests/qwn_042b_kv_cache_test.seen" \
    "$ARTIFACT_ROOT/qwn_042b_seen_surface_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_042b_seen_surface_test"
QWN_042A_HARD_SCOPE=1 "$ROOT_DIR/scripts/cuda/run_qwn_042a.sh" --inner
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader
"$BUILD_ROOT/qwn_042b_cuda_test"
for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 86 --target-processes application-only)
    if [ "$tool" = memcheck ]; then args+=(--leak-check full); fi
    if [ "$tool" = racecheck ]; then args+=(--racecheck-num-workers 1); fi
    /opt/cuda/bin/compute-sanitizer "${args[@]}" "$BUILD_ROOT/qwn_042b_cuda_test"
done
sha256sum "$BUILD_ROOT/libseen_qwen_cuda.so" "$BUILD_ROOT/qwn_042b_cuda_test" \
    "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0"
echo "PASS: QWN-042B v0.20.4 deterministic Seen-owned KV-cache gates"
