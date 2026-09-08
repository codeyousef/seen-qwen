#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
DEFAULT_TOOLCHAIN="$SHARED_ROOT/.seen/toolchains/seen-0.20.1-linux-x64"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$DEFAULT_TOOLCHAIN}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_040a"

if [ "${1:-}" != "--inner" ]; then
    exec env -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH -u SEEN_RUNTIME_PATH \
        -u SEEN_COMPILER_SOURCE_ROOT -u SEEN_PACKAGE_CLIENT \
        QWN_TASKS_MAX=32 "$ROOT_DIR/scripts/oracle/run_bounded.sh" 3600 \
        env QWN_040A_HARD_SCOPE=1 SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
        "$0" --inner
fi

[ "${QWN_040A_HARD_SCOPE:-0}" = 1 ] || {
    echo "qwn-040a: verified hard scope is required" >&2
    exit 126
}

SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
SEEN_CUDA_ROOT="$TOOLCHAIN_ROOT/lib/seen/runtime/cuda"
BUILD_ROOT="$ARTIFACT_ROOT/build"

printf '%s  %s\n' 7f823add11df162a9597d3ae225b6d3a6f48dfacd878f4ea0392b4f1435eb3fc "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' 7cc8f5c265280e04fb5e5bc182e24dbe069a58e6373d5603d47f72d69bbb4224 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 55c3aff948ea8a587dd5dad3bc511cc6d7fd3028b7a458240ba07542680f4603 "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.1'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.1 version | grep -Fx 'seen-pkg 0.20.1 (SEENPKG1)'
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["release_version"] == "0.20.1" and d["components"]["runtime"]["abi"] == "runtime-v4" and d["components"]["package_client"] == {"protocol": "SEENPKG1", "version": "0.20.1"}' "$COMPATIBILITY_MANIFEST"

mkdir -p "$BUILD_ROOT" "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_040a_reference_primitives_test.seen" --frozen
"$SEEN_BIN" compile "$ROOT_DIR/tests/qwn_040a_reference_primitives_test.seen" \
    "$ARTIFACT_ROOT/qwn_040a_seen_surface_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_040a_seen_surface_test"
cmake -S "$ROOT_DIR/native" -B "$BUILD_ROOT" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_ROOT" --parallel 1
"$BUILD_ROOT/qwn_040a_header_contract"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader
"$BUILD_ROOT/qwn_040a_cuda_test"

for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 86
        --target-processes application-only)
    if [ "$tool" = memcheck ]; then args+=(--leak-check full); fi
    if [ "$tool" = racecheck ]; then args+=(--racecheck-num-workers 1); fi
    /opt/cuda/bin/compute-sanitizer "${args[@]}" \
        "$BUILD_ROOT/qwn_040a_cuda_test"
done

sha256sum "$BUILD_ROOT/libseen_qwen_cuda.so" \
    "$BUILD_ROOT/qwn_040a_cuda_test" "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free,temperature.gpu,power.draw,pstate \
    --format=csv,noheader
echo "PASS: QWN-040A v0.20.1 CPU, RTX 4090, graph, sanitizer, and leak gates"
