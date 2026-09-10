#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
DEFAULT_TOOLCHAIN="$SHARED_ROOT/.seen/toolchains/seen-0.20.4-linux-x64"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$DEFAULT_TOOLCHAIN}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_041c"

if [ "${1:-}" != "--inner" ]; then
    exec env -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH -u SEEN_RUNTIME_PATH \
        -u SEEN_COMPILER_SOURCE_ROOT -u SEEN_PACKAGE_CLIENT \
        QWN_TASKS_MAX=32 "$ROOT_DIR/scripts/oracle/run_bounded.sh" 3600 \
        env QWN_041C_HARD_SCOPE=1 SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
        "$0" --inner
fi

[ "${QWN_041C_HARD_SCOPE:-0}" = 1 ] || {
    echo "qwn-041c: verified hard scope is required" >&2
    exit 126
}

SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
SEEN_CUDA_ROOT="$TOOLCHAIN_ROOT/lib/seen/runtime/cuda"
BUILD_ROOT="$ARTIFACT_ROOT/build"

printf '%s  %s\n' 79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' bfed49cea60c983751c26cef81b21e3374360f3a43de677e8134c14a3c30a158 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 69441bbf20755f0bbf12a4241fffadf3ad20df6f9d1155a6b0b5ab92992e9e2c "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.4'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.4 version | grep -Fx 'seen-pkg 0.20.4 (SEENPKG1)'
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["release_version"] == "0.20.4" and d["components"]["compiler"]["version"] == "0.20.4" and d["components"]["runtime"]["abi"] == "runtime-v4" and d["components"]["package_client"] == {"protocol": "SEENPKG1", "version": "0.20.4"}' "$COMPATIBILITY_MANIFEST"

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

mkdir -p "$BUILD_ROOT" "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
python3 -m unittest tests/test_cuda_reference_primitives.py \
    tests/test_cuda_reference_utilities.py tests/test_cuda_gdn_state.py \
    tests/test_cuda_gdn_recurrent_decode.py tests/test_cuda_gdn_recurrent_prefill.py
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_041c_gdn_prefill_test.seen" --frozen
"$SEEN_BIN" compile "$ROOT_DIR/tests/qwn_041c_gdn_prefill_test.seen" \
    "$ARTIFACT_ROOT/qwn_041c_seen_surface_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_041c_seen_surface_test"
cmake -S "$ROOT_DIR/native" -B "$BUILD_ROOT" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_ROOT" --parallel 1
"$BUILD_ROOT/qwn_040a_header_contract"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader
"$BUILD_ROOT/qwn_040a_cuda_test"
"$BUILD_ROOT/qwn_040b_cuda_test"
"$BUILD_ROOT/qwn_041a_cuda_test"
"$BUILD_ROOT/qwn_041b_cuda_test"
"$BUILD_ROOT/qwn_041c_cuda_test"

for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 86 --target-processes application-only)
    if [ "$tool" = memcheck ]; then args+=(--leak-check full); fi
    if [ "$tool" = racecheck ]; then args+=(--racecheck-num-workers 1); fi
    /opt/cuda/bin/compute-sanitizer "${args[@]}" "$BUILD_ROOT/qwn_041c_cuda_test"
done

sha256sum "$BUILD_ROOT/libseen_qwen_cuda.so" "$BUILD_ROOT/qwn_041c_cuda_test" \
    "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free,temperature.gpu,power.draw,pstate \
    --format=csv,noheader

toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || {
    echo "qwn-041c: compiler installation changed during verification" >&2; exit 126;
}
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || {
    echo "qwn-041c: compiler objects changed outside the ignored .seen root" >&2; exit 126;
}
echo "PASS: QWN-041C v0.20.4 chunked GDN prefill CUDA gates"
