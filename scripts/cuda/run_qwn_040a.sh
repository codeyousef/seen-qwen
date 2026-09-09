#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
DEFAULT_TOOLCHAIN="$SHARED_ROOT/.seen/toolchains/seen-0.20.2/seen-0.20.2-linux-x64"
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

printf '%s  %s\n' c3a528a7375d34d4209e8dcfd506d603b9c46623357e6aafdd22036c02868032 "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' eda0988c1966722e086b0ef86ffb0c4dbb502276dbca4a6d2553fb403ed93421 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 2b5e034c3316d01c23be99cc32e36e74f7379b24adb3c0b14b2e7f43fcf69a32 "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.2'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.2 version | grep -Fx 'seen-pkg 0.20.2 (SEENPKG1)'
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["release_version"] == "0.20.2" and d["components"]["compiler"]["version"] == "0.20.2" and d["components"]["runtime"]["abi"] == "runtime-v4" and d["components"]["package_client"] == {"protocol": "SEENPKG1", "version": "0.20.2"}' "$COMPATIBILITY_MANIFEST"

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

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

toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || {
    echo "qwn-040a: compiler installation changed during CUDA verification" >&2
    exit 126
}
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || {
    echo "qwn-040a: compiler objects changed outside the ignored .seen root" >&2
    exit 126
}
echo "PASS: QWN-040A v0.20.2 CPU, RTX 4090, graph, sanitizer, and leak gates"
