#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/seen-0.22.2/seen-0.22.2-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_046a"
ENGINE_ROOT=${QWN_046A_ENGINE_ROOT:-}

if [ "${1:-}" != "--inner" ]; then
    [ "$#" -eq 1 ] || { echo "usage: run_qwn_046a.sh ENGINE_ROOT" >&2; exit 64; }
    exec env -u PATH -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH \
        -u SEEN_RUNTIME_PATH -u SEEN_COMPILER_SOURCE_ROOT \
        -u SEEN_PACKAGE_CLIENT QWN_TASKS_MAX=32 \
        "$ROOT_DIR/scripts/oracle/run_bounded.sh" 14400 \
        /usr/bin/env PATH=/usr/bin:/bin:/opt/cuda/bin QWN_046A_HARD_SCOPE=1 \
        QWN_046A_ENGINE_ROOT="$1" SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
        "$0" --inner
fi

[ "${QWN_046A_HARD_SCOPE:-0}" = 1 ] || exit 126
case "$ENGINE_ROOT" in "$SHARED_ROOT/.seen/artifacts/qwn_046a/"*) ;; *) exit 126 ;; esac
[ -d "$ENGINE_ROOT" ] && [ ! -L "$ENGINE_ROOT" ] || exit 126
SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
SEEN_CUDA_ROOT="$TOOLCHAIN_ROOT/lib/seen/runtime/cuda"
WEIGHTS="$ENGINE_ROOT/weights.sqw"
BUILD_ROOT="$ARTIFACT_ROOT/build"
mkdir -p "$ARTIFACT_ROOT" "$BUILD_ROOT" "$ARTIFACT_ROOT/seen"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
export SEEN_JOBS=1 SEEN_OPT_JOBS=1 SEEN_LOW_MEMORY=1
export SEEN_MEMORY_LIMIT_BYTES=4294967296 SEEN_MAIN_VMEM_KB=4194304
export SEEN_OPT_VMEM_KB=2097152

printf '%s  %s\n' dd544d342401a9066d9d76d6849e41972a60968d8ee841c678ba23cd6d82c34e "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' 3c7af4b74da3199d652cd545814731d7695123a67f54817b2ab925055b201f59 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' cc7deff18b3319b36ed85ceb050e867b3fc36491e928f54e1908c8a260146be6 "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.22.2'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.22.2 version | grep -Fx 'seen-pkg 0.22.2 (SEENPKG1)'
readelf -n "$SEEN_BIN" | grep -F 'Build ID: c9865c92973d525338c018a2c6e84684a7abe8bb'
(cd "$ENGINE_ROOT" && sha256sum -c checksums.sha256)
python3 - "$ENGINE_ROOT/engine.json" <<'PY'
import json, sys
document = json.load(open(sys.argv[1], encoding="utf-8"))
assert document["maturity"] == "experimental-hardware"
assert document["format"]["tensor_count"] == 866
assert document["quantization"] == {
    "fallback": "prohibited", "lossy": True, "policy_id": "q4-bringup-v1",
    "policy_sha256": document["quantization"]["policy_sha256"],
    "quality_approved": False, "runtime_codec": "Q4_SYM_G64",
    "source_dtype": "BF16", "tensor_count": 866,
}
assert document["memory"]["weight_bytes"] == "14515042384"
PY

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

python3 -m unittest tests.test_qwn_046a_q4_artifact
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_BIN" check tests/qwn_046a_full_memory_plan_test.seen --frozen
"$SEEN_BIN" check tests/qwn_046a_full_cuda_frontend_test.seen --frozen
"$SEEN_BIN" check tests/qwn_046a_full_cuda_hardware_test.seen --frozen
"$SEEN_BIN" compile tests/qwn_046a_full_memory_plan_test.seen \
    "$ARTIFACT_ROOT/qwn_046a_release" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_046a_release"
"$SEEN_BIN" compile tests/qwn_046a_full_cuda_hardware_test.seen \
    "$ARTIFACT_ROOT/qwn_046a_seen_cuda_fast" \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$SEEN_BIN" compile tests/qwn_046a_full_cuda_hardware_test.seen \
    "$ARTIFACT_ROOT/qwn_046a_seen_cuda" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen

cmake -S "$ROOT_DIR/native" -B "$BUILD_ROOT" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_ROOT" --target qwn_046a_cuda_test --parallel 1

gpu_identity=$(nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader)
printf 'QWN-046A hardware: %s\n' "$gpu_identity"
printf '%s\n' "$gpu_identity" | grep -F 'NVIDIA GeForce RTX 4090'
printf '%s\n' "$gpu_identity" | grep -F '8.9'
LD_LIBRARY_PATH="$BUILD_ROOT/seen_cuda_build" \
    "$ARTIFACT_ROOT/qwn_046a_seen_cuda_fast" "$WEIGHTS"
LD_LIBRARY_PATH="$BUILD_ROOT/seen_cuda_build" \
    "$ARTIFACT_ROOT/qwn_046a_seen_cuda" "$WEIGHTS"
"$BUILD_ROOT/qwn_046a_cuda_test" "$WEIGHTS"
/opt/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
    --error-exitcode 86 --target-processes application-only \
    "$BUILD_ROOT/qwn_046a_cuda_test" "$WEIGHTS" --sanitizer-smoke

sha256sum "$BUILD_ROOT/qwn_046a_cuda_test" \
    "$BUILD_ROOT/seen_cuda_build/libseen_cuda.so.1.0.0" "$WEIGHTS"
toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || exit 126
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || exit 126
echo "PASS: QWN-046A v0.22.2 complete Q4 model plan, residency, transfer, memcheck, and cleanup gates"
