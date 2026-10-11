#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_046b"
ENGINE_ROOT=${QWN_046B_ENGINE_ROOT:-}
SEEN_BIN=${SEEN_BIN:-}

fail() {
    echo "qwn-046b: $*" >&2
    exit 126
}

if [ "${1:-}" != "--inner" ]; then
    [ "$#" -eq 1 ] || fail "usage: run_qwn_046b.sh ENGINE_ROOT"
    resolved_seen=${SEEN_BIN:-}
    [ -n "$resolved_seen" ] || fail "set SEEN_BIN to the audited Seen 0.22.9 compiler"
    [ "${resolved_seen#/}" != "$resolved_seen" ] && [ -x "$resolved_seen" ] ||
        fail "SEEN_BIN must be an explicit executable absolute path"
    exec env -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH -u SEEN_RUNTIME_PATH \
        -u SEEN_COMPILER_SOURCE_ROOT -u SEEN_PACKAGE_CLIENT \
        QWN_TASKS_MAX=24 QWN_046B_HARD_SCOPE=1 \
        QWN_046B_ENGINE_ROOT="$1" SEEN_BIN="$resolved_seen" \
        "$ROOT_DIR/scripts/oracle/run_bounded.sh" 28800 "$0" --inner
fi

[ "${QWN_046B_HARD_SCOPE:-0}" = 1 ] || fail "missing verified hard scope"
[ -n "$ENGINE_ROOT" ] || fail "missing engine root"
[ -d "$ENGINE_ROOT" ] && [ ! -L "$ENGINE_ROOT" ] || fail "unsafe engine root"
WEIGHTS="$ENGINE_ROOT/weights.sqw"
ORACLE="$ROOT_DIR/tests/fixtures/qwn_025b_full_model_oracles.json"
SEEN_PACKAGE_CLIENT="$(dirname "$SEEN_BIN")/seen-pkg"
COMPATIBILITY_MANIFEST="$(dirname "$SEEN_BIN")/compatibility-manifest.json"
SEEN_PREFIX="$(cd -P -- "$(dirname "$SEEN_BIN")/.." && pwd -P)"
SEEN_CUDA_ROOT="$SEEN_PREFIX/lib/seen/runtime/cuda"
NATIVE_BUILD="$ARTIFACT_ROOT/native-build"
mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_ROOT/tmp" \
    "$ARTIFACT_ROOT/seen" "$ROOT_DIR/.seen/native/lib"
export TMPDIR="$ARTIFACT_ROOT/tmp"
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
export SEEN_PACKAGE_CLIENT
export SEEN_JOBS=1 SEEN_OPT_JOBS=1 SEEN_LOW_MEMORY=1
export SEEN_MEMORY_LIMIT_BYTES=4294967296 SEEN_MAIN_VMEM_KB=4194304
export SEEN_OPT_VMEM_KB=2097152 CMAKE_BUILD_PARALLEL_LEVEL=1

printf '%s  %s\n' 58d7181f8b5adf1bb823ee5dcfc0868f9962b302af4dfaabf6a9e806aea16db2 "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' e6e4d78bee4f01ac5580a6115b35b648adfad75fd9f05603ec8a726e5e32171a "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 9fc9f4a7d5b59e494148935e863576946cb5f57c768b30b260bfd9fd40359514 "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.22.9'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.22.9 version --machine |
    grep -Fx 'version=0.22.9'
readelf -n "$SEEN_BIN" | grep -F 'Build ID: 2e8228365322fc6462555d115160a863d1ed1ce7'
(cd "$ENGINE_ROOT" && sha256sum -c checksums.sha256)
printf '%s  %s\n' 42ba8e0b9d5adec7c5ce1708bfc60d7b9911c126af9e6703b35efd108fedc921 "$ORACLE" | sha256sum -c -

toolchain_hash_before=$(find "$SEEN_BIN" "$SEEN_PACKAGE_CLIENT" \
    "$COMPATIBILITY_MANIFEST" "$SEEN_PREFIX/lib/seen" -type f \
    -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

python3 -m unittest tests.test_qwn_046a_full_cuda_contract \
    tests.test_cuda_reference_primitives
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_BIN" check tests/qwn_046b_full_cuda_oracle_test.seen --frozen

cmake -S "$ROOT_DIR/native" -B "$NATIVE_BUILD" -G Ninja \
    -DSEEN_CUDA_SOURCE_ROOT="$SEEN_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$NATIVE_BUILD" --target seen_qwen_cuda --parallel 1
cp "$NATIVE_BUILD/libseen_qwen_cuda.so" \
    "$ROOT_DIR/.seen/native/lib/libseen_qwen_cuda.so"

"$SEEN_BIN" compile tests/qwn_046b_full_cuda_oracle_test.seen \
    "$ARTIFACT_ROOT/qwn_046b_oracle_fast" --target-cpu=x86-64 \
    --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$SEEN_BIN" compile tests/qwn_046b_full_cuda_oracle_test.seen \
    "$ARTIFACT_ROOT/qwn_046b_oracle_release" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$SEEN_BIN" compile tests/qwn_046b_full_cuda_oracle_test.seen \
    "$ARTIFACT_ROOT/qwn_046b_oracle_ubsan" --sanitize undefined \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen

compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d')
[ -z "$compute_pids" ] || fail "GPU has an active compute process; no process was terminated"
gpu_identity=$(nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu --format=csv,noheader)
printf 'QWN-046B hardware: %s\n' "$gpu_identity"
printf '%s\n' "$gpu_identity" | grep -F 'NVIDIA GeForce RTX 4090'
printf '%s\n' "$gpu_identity" | grep -F '8.9'

"$ARTIFACT_ROOT/qwn_046b_oracle_fast" "$WEIGHTS" "$ORACLE"
"$ARTIFACT_ROOT/qwn_046b_oracle_release" "$WEIGHTS" "$ORACLE"
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$ARTIFACT_ROOT/qwn_046b_oracle_ubsan" "$WEIGHTS" "$ORACLE"
/opt/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
    --error-exitcode 86 --target-processes application-only \
    "$ARTIFACT_ROOT/qwn_046b_oracle_release" "$WEIGHTS" "$ORACLE"

sha256sum "$ARTIFACT_ROOT/qwn_046b_oracle_fast" \
    "$ARTIFACT_ROOT/qwn_046b_oracle_release" \
    "$ARTIFACT_ROOT/qwn_046b_oracle_ubsan" \
    "$NATIVE_BUILD/libseen_qwen_cuda.so" "$WEIGHTS" "$ORACLE"
toolchain_hash_after=$(find "$SEEN_BIN" "$SEEN_PACKAGE_CLIENT" \
    "$COMPATIBILITY_MANIFEST" "$SEEN_PREFIX/lib/seen" -type f \
    -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] ||
    fail "compiler installation changed during verification"
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] ||
    fail "generated runtime objects escaped the ignored .seen directory"
echo "PASS: QWN-046B 188-position two-prompt oracle, sanitizer, and cleanup gates"
