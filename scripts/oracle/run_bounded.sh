#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
INNER="$ROOT_DIR/scripts/oracle/run_bounded_inner.sh"
MEMORY_CEILING_BYTES=51539607552
MEMORY_RESERVE_BYTES=8589934592
VMEM_CEILING_BYTES=68719476736
TASKS_MAX=16

fail() {
    echo "oracle-scope: $*" >&2
    exit 126
}

[ "$#" -ge 2 ] || fail "usage: run_bounded.sh TIMEOUT_SECONDS COMMAND [ARG ...]"
timeout_seconds=$1
shift
case "$timeout_seconds" in *[!0-9]*|"") fail "timeout must be an integer" ;; esac
[ "$timeout_seconds" -ge 1 ] && [ "$timeout_seconds" -le 28800 ] ||
    fail "timeout must be in 1..28800 seconds"
[ -x "$INNER" ] && [ ! -L "$INNER" ] || fail "bounded inner runner is missing or unsafe"
command -v systemd-run >/dev/null 2>&1 || fail "systemd-run is required"

memory_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
memory_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
case "$memory_total_kib:$memory_available_kib" in
    *[!0-9:]*|:|*::*) fail "host memory readback is invalid" ;;
esac
memory_available_bytes=$((memory_available_kib * 1024))
[ "$memory_available_bytes" -gt "$MEMORY_RESERVE_BYTES" ] ||
    fail "host has no memory beyond the required 8 GiB reserve"
memory_bytes=$((memory_available_bytes - MEMORY_RESERVE_BYTES))
[ "$memory_bytes" -le "$MEMORY_CEILING_BYTES" ] || memory_bytes=$MEMORY_CEILING_BYTES
[ "$memory_bytes" -le 68719476736 ] || fail "derived memory cap exceeds 64 GiB"
[ "$memory_bytes" -ge 2147483648 ] || fail "derived memory cap is below 2 GiB"
vmem_bytes=$((memory_bytes * 4))
[ "$vmem_bytes" -le "$VMEM_CEILING_BYTES" ] || vmem_bytes=$VMEM_CEILING_BYTES
vmem_kib=$((vmem_bytes / 1024))

echo "oracle-scope: host MemTotal=${memory_total_kib}KiB MemAvailable=${memory_available_kib}KiB derived_memory.max=${memory_bytes} memory.swap.max=0 pids.max=${TASKS_MAX} vmem=${vmem_bytes} timeout=${timeout_seconds}s workers=1"

exec systemd-run --user --scope --quiet --expand-environment=no \
    -p "MemoryMax=$memory_bytes" \
    -p MemorySwapMax=0 \
    -p "TasksMax=$TASKS_MAX" \
    -p CPUQuota=100% \
    --setenv="QWN_EXPECTED_MEMORY_BYTES=$memory_bytes" \
    --setenv="QWN_EXPECTED_TASKS_MAX=$TASKS_MAX" \
    --setenv="QWN_VMEM_KIB=$vmem_kib" \
    --setenv="QWN_TIMEOUT_SECONDS=$timeout_seconds" \
    --setenv=OMP_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=MKL_NUM_THREADS=1 \
    --setenv=RAYON_NUM_THREADS=1 \
    --setenv=TOKENIZERS_PARALLELISM=false \
    -- "$INNER" "$@"
