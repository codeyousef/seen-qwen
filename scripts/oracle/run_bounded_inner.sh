#!/usr/bin/env bash

set -euo pipefail

fail() {
    echo "oracle-inner: $*" >&2
    exit 126
}

cgroup_path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
[ -n "$cgroup_path" ] || fail "unified cgroup path is unavailable"
cgroup_root="/sys/fs/cgroup$cgroup_path"
read_metric() {
    [ -r "$cgroup_root/$1" ] || fail "missing cgroup readback: $1"
    cat "$cgroup_root/$1"
}

report_metrics() {
    status=${1:-$?}
    echo "oracle-inner: exit_status=$status"
    for metric in memory.current memory.peak memory.events pids.current pids.peak pids.events; do
        if [ -r "$cgroup_root/$metric" ]; then
            echo "oracle-inner: $metric"
            cat "$cgroup_root/$metric"
        fi
    done
    return "$status"
}
trap report_metrics EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

memory_max=$(read_metric memory.max)
swap_max=$(read_metric memory.swap.max)
pids_max=$(read_metric pids.max)
cpu_max=$(read_metric cpu.max)
[ "$memory_max" = "${QWN_EXPECTED_MEMORY_BYTES:?}" ] || fail "memory.max readback mismatch"
[ "$memory_max" -le 68719476736 ] || fail "memory.max exceeds 64 GiB"
[ "$swap_max" = 0 ] || fail "memory.swap.max is not zero"
[ "$pids_max" = "${QWN_EXPECTED_TASKS_MAX:?}" ] || fail "pids.max readback mismatch"
[ "$cpu_max" = "100000 100000" ] || fail "CPU quota does not enforce one aggregate worker"
ulimit -v "${QWN_VMEM_KIB:?}"
ulimit -s 8192
[ "$(ulimit -v)" = "$QWN_VMEM_KIB" ] || fail "virtual-memory readback mismatch"
[ "$(ulimit -s)" = 8192 ] || fail "stack readback is not 8 MiB"
echo "oracle-inner: verified cgroup=$cgroup_path memory.max=$memory_max memory.swap.max=$swap_max pids.max=$pids_max cpu.max='$cpu_max' vmem_kib=$QWN_VMEM_KIB stack_kib=$(ulimit -s)"

set +e
timeout --foreground --signal=TERM --kill-after=30 "${QWN_TIMEOUT_SECONDS:?}" "$@"
status=$?
set -e
trap - EXIT
report_metrics "$status"
exit "$status"
