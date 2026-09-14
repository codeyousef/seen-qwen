#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/seen-0.20.9-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_045b"

if [ "${1:-}" != "--inner" ]; then
    exec env -u PATH -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH \
        -u SEEN_RUNTIME_PATH -u SEEN_COMPILER_SOURCE_ROOT \
        -u SEEN_PACKAGE_CLIENT QWN_TASKS_MAX=32 \
        "$ROOT_DIR/scripts/oracle/run_bounded.sh" 5400 \
        /usr/bin/env PATH=/usr/bin:/bin:/opt/cuda/bin QWN_045B_HARD_SCOPE=1 \
        SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" "$0" --inner
fi

[ "${QWN_045B_HARD_SCOPE:-0}" = 1 ] || exit 126
case "$TOOLCHAIN_ROOT" in
    "$SHARED_ROOT/.seen/toolchains/"*) ;;
    *) echo "QWN-045B toolchain is outside the ignored project root" >&2; exit 126 ;;
esac
[ -d "$TOOLCHAIN_ROOT" ] && [ ! -L "$TOOLCHAIN_ROOT" ] || exit 126

SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
SEEN_CUDA_ROOT="$TOOLCHAIN_ROOT/lib/seen/runtime/cuda"
BUILD_ROOT="$ARTIFACT_ROOT/build"
MODEL="$ROOT_DIR/tests/fixtures/qwen3_8_hybrid_mini/model.safetensors"
ORACLE="$ROOT_DIR/tests/fixtures/qwn_024e_cpu_engine_oracle.json"
mkdir -p "$ARTIFACT_ROOT" "$BUILD_ROOT" "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
export SEEN_JOBS=1 SEEN_OPT_JOBS=1 SEEN_LOW_MEMORY=1
export SEEN_MEMORY_LIMIT_BYTES=4294967296 SEEN_MAIN_VMEM_KB=4194304
export SEEN_OPT_VMEM_KB=2097152

printf '%s  %s\n' c2c7f814359d699a7c5d7659184c8bb3fb8e6ffb063c1e8865b7fa121f05b5cc "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' b672c79a8d40254447c7a02214b3e3eb9ba22b74009e79a9355d770c68a96935 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 426530e3eb99e367ec06b2805aefc7bcad10cc18e1265ff3d5d4263b6a607ba2 "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.9'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.9 version | grep -Fx 'seen-pkg 0.20.9 (SEENPKG1)'
readelf -n "$SEEN_BIN" | grep -F 'Build ID: 9ad1afa96e2a848ce4fd75651dce4ee0f41a5755'
printf '%s  %s\n' 16ecca9cb396099db0c92d835840264e7b45d12cd6221d7af5462ac8576c94a9 "$MODEL" | sha256sum -c -
printf '%s  %s\n' 350bc70fb4c7dc010643cfd0c44f93fb60cb75cacfcddda08bc02eff26906857 "$ORACLE" | sha256sum -c -

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

python3 -m unittest tests.test_cuda_reference_primitives \
    tests.test_cuda_gdn_state tests.test_cuda_gdn_recurrent_prefill \
    tests.test_cuda_gdn_recurrent_decode tests.test_cuda_gdn_gated_output \
    tests.test_cuda_attention_prefill tests.test_cuda_attention_decode \
    tests.test_cuda_lm_head_greedy tests.test_cuda_mini_execution
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_BIN" check "$ROOT_DIR/tests/qwn_040a_reference_primitives_test.seen" --frozen

cmake -S "$ROOT_DIR/native" -B "$BUILD_ROOT" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_ROOT" --target qwn_045b_cuda_test --parallel 1

gpu_identity=$(nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader)
printf 'QWN-045B hardware: %s\n' "$gpu_identity"
printf '%s\n' "$gpu_identity" | grep -F 'NVIDIA GeForce RTX 4090'
printf '%s\n' "$gpu_identity" | grep -F '8.9'
"$BUILD_ROOT/qwn_045b_cuda_test" "$MODEL" "$ORACLE"

for tool in memcheck initcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 86 --target-processes application-only)
    if [ "$tool" = memcheck ]; then args+=(--leak-check full); fi
    if [ "$tool" = racecheck ]; then args+=(--racecheck-num-workers 1); fi
    /opt/cuda/bin/compute-sanitizer "${args[@]}" \
        "$BUILD_ROOT/qwn_045b_cuda_test" "$MODEL" "$ORACLE"
done

sha256sum "$BUILD_ROOT/qwn_045b_cuda_test" \
    "$BUILD_ROOT/libseen_qwen_cuda.so" \
    "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0"
toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || exit 126
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || exit 126
echo "PASS: QWN-045B v0.20.9 end-to-end CUDA mini-model, CPU differential, reset, cancellation, sanitizer, RTX 4090, and cleanup gates"
