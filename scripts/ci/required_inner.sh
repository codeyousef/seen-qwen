#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=/workspace
TOOLCHAIN_ROOT="$ROOT_DIR/.seen/toolchains/seen-0.15.0-linux-x64"
SEEN_COMPILER="$TOOLCHAIN_ROOT/bin/seen"
SEEN_PACKAGE_CLIENT="$TOOLCHAIN_ROOT/bin/seen-pkg"
COMPATIBILITY_MANIFEST="$TOOLCHAIN_ROOT/bin/compatibility-manifest.json"
ASSET_ROOT="$ROOT_DIR/.seen/oracle-assets-qwen38"
OUTPUT_ROOT="$ROOT_DIR/.seen/ci/output"
ARTIFACT_ROOT="$ROOT_DIR/.seen/ci/artifacts"

fail() {
    echo "ci-inner: $*" >&2
    exit 126
}

read_cgroup() {
    local name=$1
    local value=""
    [ -r "/sys/fs/cgroup/$name" ] || fail "missing cgroup readback: $name"
    IFS= read -r value < "/sys/fs/cgroup/$name"
    printf '%s' "$value"
}

report_metrics() {
    local status=$?
    echo "ci-inner: exit_status=$status"
    for metric in memory.current memory.peak memory.events pids.current pids.peak pids.events; do
        if [ -r "/sys/fs/cgroup/$metric" ]; then
            echo "ci-inner: $metric"
            cat "/sys/fs/cgroup/$metric"
        fi
    done
    return "$status"
}
trap report_metrics EXIT

memory_max=$(read_cgroup memory.max)
swap_max=$(read_cgroup memory.swap.max)
pids_max=$(read_cgroup pids.max)
oom_group=$(read_cgroup memory.oom.group)
[ "$memory_max" = "${SEEN_EXPECTED_MEMORY_BYTES:-}" ] ||
    fail "memory.max does not match the current-memory-derived cap"
[ "$memory_max" -le 4294967296 ] || fail "memory.max exceeds the 4 GiB ceiling"
[ "$swap_max" = "0" ] || fail "memory.swap.max is not zero"
[ "$pids_max" = "24" ] || fail "pids.max is not 24"
case "$oom_group" in 0|1) ;; *) fail "memory.oom.group is not numeric" ;; esac
echo "ci-inner: verified cgroup=/sys/fs/cgroup memory.max=$memory_max memory.swap.max=$swap_max memory.oom.group=$oom_group pids.max=$pids_max"

ulimit -v "${SEEN_MAIN_VMEM_KB:?missing Seen virtual-memory cap}"
[ "$(ulimit -v)" = "$SEEN_MAIN_VMEM_KB" ] ||
    fail "per-process virtual memory does not match the derived hard cap"
[ "$(ulimit -s)" = "8192" ] || fail "stack is not exactly 8 MiB"
[ "${SEEN_JOBS:-}" = "1" ] && [ "${SEEN_OPT_JOBS:-}" = "1" ] ||
    fail "serial Seen worker settings are missing"

[ "$(clang --version | sed -n '1s/.*version \([0-9][0-9.]*\).*/\1/p')" = "21.1.8" ] ||
    fail "clang is not the pinned 21.1.8 toolchain"
for tool in opt llc llvm-as ld.lld; do
    command -v "$tool" >/dev/null 2>&1 || fail "missing LLVM tool: $tool"
    "$tool" --version | grep -Eq \
        '(LLVM|LLD) version 21\.1\.8|Debian (LLVM version|LLD) 21\.1\.8' ||
        fail "$tool is not LLVM 21.1.8"
done

printf '%s  %s\n' \
    0a9b56f81fcaeab8f6f0e22e30d908832f843e112dacd6a6a67954106e881516 \
    "$SEEN_COMPILER" | sha256sum -c -
printf '%s  %s\n' \
    cb15b697946941ea18fc56f26a1dc9c5d97400fccb84797ca0a40dd7e524a700 \
    "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' \
    3472e3b9e99234d51bdcf62aef985909cb0b6d574283ae5fcb76127c699c368d \
    "$COMPATIBILITY_MANIFEST" | sha256sum -c -
printf '%s  %s\n' \
    ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 \
    "$ASSET_ROOT/vocab.json" | sha256sum -c -
printf '%s  %s\n' \
    a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d \
    "$ASSET_ROOT/merges.txt" | sha256sum -c -
[ "$(stat -c '%s' "$ASSET_ROOT/vocab.json")" = "6722759" ] ||
    fail "vocabulary byte length changed"
[ "$(stat -c '%s' "$ASSET_ROOT/merges.txt")" = "3353259" ] ||
    fail "merge-table byte length changed"

python3 -c 'import json; p="/workspace/.seen/toolchains/seen-0.15.0-linux-x64/bin/compatibility-manifest.json"; d=json.load(open(p, encoding="utf-8")); assert d["schema"] == "seen-compatibility-manifest-v1"; assert d["release_version"] == "0.15.0"; assert d["components"]["compiler"]["version"] == "0.15.0"; assert d["components"]["package_client"] == {"protocol": "SEENPKG1", "version": "0.15.0"}; assert d["components"]["llvm"]["minimum_major"] == 19; assert d["platforms"]["linux-x86_64"] == "required"'

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

"$SEEN_COMPILER" --version | grep -Fx 'Seen 0.15.0'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.15.0 version |
    grep -Fx 'seen-pkg 0.15.0 (SEENPKG1)'
python3 -m unittest tests/test_ci_contract.py tests/test_qwen_tokenizer_oracles.py
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_COMPILER" check tests/qwn_022b_tokenizer_test.seen --frozen
"$SEEN_COMPILER" compile tests/qwn_022b_tokenizer_test.seen \
    "$OUTPUT_ROOT/qwn_022b_tokenizer_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_022b_tokenizer_test"

toolchain_hash_after=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
[ "$toolchain_hash_after" = "$toolchain_hash_before" ] ||
    fail "compiler installation changed during required CI"

outside_objects_after=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
[ "$outside_objects_after" = "$outside_objects_before" ] ||
    fail "compiler objects changed outside the ignored .seen root"
[ -d "$ARTIFACT_ROOT" ] && [ ! -L "$ARTIFACT_ROOT" ] ||
    fail "project artifact root is unsafe"

echo "PASS: exact locked Seen Qwen tokenizer gate"
