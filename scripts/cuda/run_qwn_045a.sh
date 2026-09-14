#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --path-format=absolute --git-common-dir)"
SHARED_ROOT="${GIT_COMMON_DIR%/.git}"
TOOLCHAIN_ROOT="${SEEN_TOOLCHAIN_ROOT:-$SHARED_ROOT/.seen/toolchains/seen-0.20.9-linux-x64}"
ARTIFACT_ROOT="$ROOT_DIR/.seen/artifacts/qwn_045a"

if [ "${1:-}" != "--inner" ]; then
    exec env -u PATH -u SEEN_DATA_PATH -u SEEN_STDLIB_PATH \
        -u SEEN_RUNTIME_PATH -u SEEN_COMPILER_SOURCE_ROOT \
        -u SEEN_PACKAGE_CLIENT QWN_TASKS_MAX=16 \
        "$ROOT_DIR/scripts/oracle/run_bounded.sh" 3600 \
        /usr/bin/env PATH=/usr/bin:/bin QWN_045A_HARD_SCOPE=1 \
        SEEN_TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" "$0" --inner
fi

[ "${QWN_045A_HARD_SCOPE:-0}" = 1 ] || exit 126
case "$TOOLCHAIN_ROOT" in
    "$SHARED_ROOT/.seen/toolchains/"*) ;;
    *) echo "QWN-045A toolchain is outside the ignored project root" >&2; exit 126 ;;
esac
[ -d "$TOOLCHAIN_ROOT" ] && [ ! -L "$TOOLCHAIN_ROOT" ] || exit 126

SEEN_BIN="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
mkdir -p "$ARTIFACT_ROOT/seen"
IR_ROOT=$(mktemp -d "$ARTIFACT_ROOT/ir-v0.20.9.XXXXXX")
export SEEN_ARTIFACT_ROOT="$ARTIFACT_ROOT/seen"
export SEEN_JOBS=1
export SEEN_OPT_JOBS=1
export SEEN_LOW_MEMORY=1
export SEEN_MEMORY_LIMIT_BYTES=4294967296
export SEEN_MAIN_VMEM_KB=4194304
export SEEN_OPT_VMEM_KB=2097152

printf '%s  %s\n' c2c7f814359d699a7c5d7659184c8bb3fb8e6ffb063c1e8865b7fa121f05b5cc "$SEEN_BIN" | sha256sum -c -
printf '%s  %s\n' b672c79a8d40254447c7a02214b3e3eb9ba22b74009e79a9355d770c68a96935 "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' 426530e3eb99e367ec06b2805aefc7bcad10cc18e1265ff3d5d4263b6a607ba2 "$COMPATIBILITY_MANIFEST" | sha256sum -c -
"$SEEN_BIN" --version | grep -Fx 'Seen 0.20.9'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.9 version | grep -Fx 'seen-pkg 0.20.9 (SEENPKG1)'
readelf -n "$SEEN_BIN" | grep -F 'Build ID: 9ad1afa96e2a848ce4fd75651dce4ee0f41a5755'
python3 -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); assert d["release_version"] == "0.20.9"; assert d["components"]["compiler"]["version"] == "0.20.9"; assert d["components"]["package_client"] == {"protocol": "SEENPKG1", "version": "0.20.9"}; assert d["components"]["runtime"]["abi"] == "runtime-v4"; assert d["platforms"]["linux-x86_64"] == "required"' "$COMPATIBILITY_MANIFEST"

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

python3 -m unittest tests.test_qwen_engine_ownership \
    tests.test_hybrid_mini_contract tests.test_hybrid_mini_assets \
    tests.test_hybrid_mini_oracle
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_BIN" check tests/qwn_045a_engine_ownership_test.seen --frozen
"$SEEN_BIN" compile tests/qwn_045a_engine_ownership_test.seen \
    "$ARTIFACT_ROOT/qwn_045a_ir_only" --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen \
    --emit-module-ir-dir "$IR_ROOT" --stop-after-ir
engine_ir=$(grep -l 'define .*QwenSession_requestFallback' "$IR_ROOT"/*.ll)
[ -n "$engine_ir" ] || exit 1
grep -q 'define .*QwenSession_requestFallback' "$engine_ir"
grep -q 'i32 -1' "$engine_ir"
! grep -q 'i64 -)' "$engine_ir"

"$SEEN_BIN" compile tests/qwn_045a_engine_ownership_test.seen \
    "$ARTIFACT_ROOT/qwn_045a_fast" --fast --target-cpu=x86-64 \
    --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_045a_fast"
"$SEEN_BIN" compile tests/qwn_045a_engine_ownership_test.seen \
    "$ARTIFACT_ROOT/qwn_045a_release" --release --lto=thin \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
"$ARTIFACT_ROOT/qwn_045a_release"
"$SEEN_BIN" compile tests/qwn_045a_engine_ownership_test.seen \
    "$ARTIFACT_ROOT/qwn_045a_ubsan" --sanitize=undefined \
    --target-cpu=x86-64 --no-cache --jobs 1 --opt-jobs 1 --no-fork --frozen
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$ARTIFACT_ROOT/qwn_045a_ubsan"

gpu_identity=$(nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,memory.used,memory.free,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader)
printf 'QWN-045A hardware: %s\n' "$gpu_identity"
printf '%s\n' "$gpu_identity" | grep -F 'NVIDIA GeForce RTX 4090'
printf '%s\n' "$gpu_identity" | grep -F '8.9'
if ldd "$ARTIFACT_ROOT/qwn_045a_release" | grep -Eiq 'lib(cuda|cublas|nvrtc)'; then
    echo "QWN-045A host ownership executable unexpectedly links a CUDA library" >&2
    exit 1
fi
! grep -Eq 'seen_(cuda|qwen)_' src/runtime/engine.seen

sha256sum "$IR_ROOT"/*.ll "$ARTIFACT_ROOT/qwn_045a_fast" \
    "$ARTIFACT_ROOT/qwn_045a_release" "$ARTIFACT_ROOT/qwn_045a_ubsan"
toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] || exit 126
outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] || exit 126

echo "PASS: QWN-045A v0.20.9 bounded ownership, fallback, cancellation, deadline, 1,000-session cleanup, UBSan, RTX 4090 identity, and CPU-only CUDA isolation"
