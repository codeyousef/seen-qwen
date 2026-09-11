#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/seen-0.20.5-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_043a"

if [ "${1:-}" != "--inner" ]; then
    exec env -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH -u SEEN_RUNTIME_PATH \
        -u SEEN_COMPILER_SOURCE_ROOT -u SEEN_PACKAGE_CLIENT \
        QWN_TASKS_MAX=32 "$ROOT_DIR/scripts/oracle/run_bounded.sh" 4200 \
        env QWN_043A_HARD_SCOPE=1 SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
        "$0" --inner
fi

[ "${QWN_043A_HARD_SCOPE:-0}" = 1 ] || exit 126
SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_CUDA_ROOT="$TOOLCHAIN_ROOT/lib/seen/runtime/cuda"
BUILD_ROOT="$ARTIFACT_ROOT/build"
printf '%s  %s\n' 03a06cc002355251b7aeea3539a3ceb466d447733a66e5b0ee3ab8c184672124 "$SEEN_BIN" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.5'

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

mkdir -p "$BUILD_ROOT" "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
python3 -m unittest tests/test_cuda_projection_descriptor.py
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_043a_projection_descriptor_test.seen" --frozen
"$SEEN_BIN" compile "$ROOT_DIR/tests/qwn_043a_projection_descriptor_test.seen" \
    "$ARTIFACT_ROOT/qwn_043a_seen_test_fast" --target-cpu=x86-64 \
    --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_043a_seen_test_fast"
"$SEEN_BIN" compile "$ROOT_DIR/tests/qwn_043a_projection_descriptor_test.seen" \
    "$ARTIFACT_ROOT/qwn_043a_seen_test" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_043a_seen_test"
cmake -S "$ROOT_DIR/native" -B "$BUILD_ROOT" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_ROOT" --parallel 1
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader
"$BUILD_ROOT/qwn_043a_cuda_test"
for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 86 --target-processes application-only)
    if [ "$tool" = memcheck ]; then args+=(--leak-check full); fi
    if [ "$tool" = racecheck ]; then args+=(--racecheck-num-workers 1); fi
    /opt/cuda/bin/compute-sanitizer "${args[@]}" "$BUILD_ROOT/qwn_043a_cuda_test"
done
sha256sum "$BUILD_ROOT/qwn_043a_cuda_test" "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0"

toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || exit 126
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || exit 126
echo "PASS: QWN-043A v0.20.5 descriptor, RTX 4090, sanitizer, and cleanup gates"
