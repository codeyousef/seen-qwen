#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/v0.20.7/seen-0.20.7-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_043b"

if [ "${1:-}" != "--inner" ]; then
    exec env -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH -u SEEN_RUNTIME_PATH \
        -u SEEN_COMPILER_SOURCE_ROOT -u SEEN_PACKAGE_CLIENT \
        QWN_TASKS_MAX=32 "$ROOT_DIR/scripts/oracle/run_bounded.sh" 5400 \
        env QWN_043B_HARD_SCOPE=1 SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
        "$0" --inner
fi

[ "${QWN_043B_HARD_SCOPE:-0}" = 1 ] || exit 126
SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
SEEN_CUDA_ROOT="$TOOLCHAIN_ROOT/lib/seen/runtime/cuda"
BUILD_ROOT="$ARTIFACT_ROOT/build"
mkdir -p "$ARTIFACT_ROOT"
IR_ROOT=$(mktemp -d "$ARTIFACT_ROOT/ir-v0.20.7.XXXXXX")
SEEN_PROJECT=$(mktemp -d "$ARTIFACT_ROOT/seen-project-v0.20.7.XXXXXX")
printf '%s  %s\n' 5fa95f150e652843795611810f02de9ca1fb6e2914ac1159fd75912bc4ae8231 "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' 91b9db87826131517830e8b0b65280c71a5a5d4ea953325c60e4fee1fbf4a754 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' b8a85368c2092fd635a5be45e505082f40a9e25107bcd33bf12e0caa45c540ee "$TOOLCHAIN_ROOT/bin/compatibility-manifest.json" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.7'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.7 version | grep -Fx 'seen-pkg 0.20.7 (SEENPKG1)'

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

mkdir -p "$BUILD_ROOT" "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
python3 -m unittest tests/test_cuda_projection_descriptor.py tests/test_cuda_ffn_execution.py
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_043a_projection_descriptor_test.seen" --frozen
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_043b_ffn_execution_test.seen" --frozen
"$SEEN_BIN" compile "$ROOT_DIR/tests/qwn_043b_ffn_execution_test.seen" \
    "$ARTIFACT_ROOT/qwn_043b_ir_only" --target-cpu=x86-64 \
    --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen \
    --emit-module-ir-dir "$IR_ROOT" --stop-after-ir
grep -R -q 'QwenCudaFfnExecutor_execute' "$IR_ROOT"
projection_ir=$(grep -l 'define .*QwenCudaFfnExecutor_execute' "$IR_ROOT"/*.ll)
[ -n "$projection_ir" ] || exit 1
execute_ir="$IR_ROOT/qwn_043b_execute.ll"
sed -n '/define .*QwenCudaFfnExecutor_execute/,/^}/p' "$projection_ir" > "$execute_ir"
[ "$(grep -c -E 'ptrtoint ptr .* to i64 ; &member' "$execute_ir")" -ge 3 ]
[ "$(grep -c 'call %CudaNativeStatus @seen_cublaslt_matmul' "$execute_ir")" -eq 3 ]
grep -q -E 'getelementptr inbounds \{ %CudaMatmulDescriptor, %CudaAlgorithm' \
    "$execute_ir"
sha256sum "$IR_ROOT"/*.ll

cmake -S "$ROOT_DIR/native" -B "$BUILD_ROOT" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_ROOT" --parallel 1
cp -R "$ROOT_DIR/src" "$SEEN_PROJECT/src"
mkdir -p "$SEEN_PROJECT/tests"
cp "$ROOT_DIR/tests/qwn_043b_ffn_execution_test.seen" \
    "$SEEN_PROJECT/tests/qwn_043b_ffn_execution_test.seen"
cp "$ROOT_DIR/tests/qwn_043b_project/Seen.toml" "$SEEN_PROJECT/Seen.toml"
mkdir -p "$SEEN_PROJECT/native/lib"
cp -a "$BUILD_ROOT"/libseen_qwen_cuda.so* "$SEEN_PROJECT/native/lib/"
cp -a "$BUILD_ROOT"/seen_cuda_build/libseen_cuda.so* "$SEEN_PROJECT/native/lib/"
"$SEEN_BIN" compile "$SEEN_PROJECT/tests/qwn_043b_ffn_execution_test.seen" \
    "$ARTIFACT_ROOT/qwn_043b_seen_test_fast" --target-cpu=x86-64 \
    --no-cache --jobs 1 --opt-jobs 1 --no-fork --offline
"$ARTIFACT_ROOT/qwn_043b_seen_test_fast"
"$SEEN_BIN" compile "$SEEN_PROJECT/tests/qwn_043b_ffn_execution_test.seen" \
    "$ARTIFACT_ROOT/qwn_043b_seen_test" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --offline
"$ARTIFACT_ROOT/qwn_043b_seen_test"
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader
"$BUILD_ROOT/qwn_043a_cuda_test"
"$BUILD_ROOT/qwn_043b_cuda_test"
for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 86 --target-processes application-only)
    if [ "$tool" = memcheck ]; then args+=(--leak-check full); fi
    if [ "$tool" = racecheck ]; then args+=(--racecheck-num-workers 1); fi
    /opt/cuda/bin/compute-sanitizer "${args[@]}" "$BUILD_ROOT/qwn_043b_cuda_test"
done
sha256sum "$BUILD_ROOT/qwn_043b_cuda_test" \
    "$BUILD_ROOT/libseen_qwen_cuda.so" \
    "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0"

toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || exit 126
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || exit 126
echo "PASS: QWN-043B v0.20.7 exact FFN, RTX 4090, sanitizer, graph, and cleanup gates"
