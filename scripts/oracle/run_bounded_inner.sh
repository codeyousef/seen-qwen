#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
PROCESS_GROUP_RUNNER="$ROOT_DIR/scripts/oracle/run_owned_process_group.sh"
EVENT_GATE="$ROOT_DIR/scripts/oracle/check_scope_events.sh"

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

declare -A baseline_pids=()
owned_extras=()
snapshot_baseline_pids() {
    local pid=""
    while IFS= read -r pid; do
        case "$pid" in *[!0-9]*|"") fail "cgroup.procs contains an invalid PID" ;; esac
        baseline_pids["$pid"]=1
    done < "$cgroup_root/cgroup.procs"
}

collect_owned_extras() {
    local pid=""
    owned_extras=()
    while IFS= read -r pid; do
        case "$pid" in *[!0-9]*|"") fail "cgroup.procs contains an invalid PID" ;; esac
        if [ -z "${baseline_pids[$pid]+x}" ]; then
            owned_extras+=("$pid")
        fi
    done < "$cgroup_root/cgroup.procs"
}

cleanup_owned_extras() {
    local signal=$1
    local pid=""
    collect_owned_extras
    for pid in "${owned_extras[@]}"; do
        # Re-read the PID's unified cgroup immediately before signaling. PID
        # reuse cannot widen cleanup beyond this invocation's dedicated scope.
        local actual_path=""
        if [ -r "/proc/$pid/cgroup" ]; then
            actual_path=$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")
        fi
        if [ "$actual_path" = "$cgroup_path" ]; then
            kill -"$signal" "$pid" 2>/dev/null || true
        fi
    done
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
[ -x "$PROCESS_GROUP_RUNNER" ] && [ ! -L "$PROCESS_GROUP_RUNNER" ] ||
    fail "owned process-group runner is missing or unsafe"
[ -x "$EVENT_GATE" ] && [ ! -L "$EVENT_GATE" ] || fail "scope event gate is missing or unsafe"
ulimit -v "${QWN_VMEM_KIB:?}"
ulimit -s 8192
[ "$(ulimit -v)" = "$QWN_VMEM_KIB" ] || fail "virtual-memory readback mismatch"
[ "$(ulimit -s)" = 8192 ] || fail "stack readback is not 8 MiB"
echo "oracle-inner: verified cgroup=$cgroup_path memory.max=$memory_max memory.swap.max=$swap_max pids.max=$pids_max cpu.max='$cpu_max' vmem_kib=$QWN_VMEM_KIB stack_kib=$(ulimit -s)"

pids_current_before=$(read_metric pids.current)
case "$pids_current_before" in
    *[!0-9]*|"") fail "pids.current baseline is invalid" ;;
esac
snapshot_baseline_pids
memory_events_before=$(read_metric memory.events)
pids_events_before=$(read_metric pids.events)

set +e
"$PROCESS_GROUP_RUNNER" "${QWN_TIMEOUT_SECONDS:?}" "$@"
status=$?
collect_owned_extras
if [ "${#owned_extras[@]}" -ne 0 ]; then
    printf 'oracle-inner: %s owned cgroup descendant(s) escaped the process group\n' \
        "${#owned_extras[@]}" >&2
    cleanup_owned_extras TERM
    for ((attempt = 0; attempt < 100; attempt += 1)); do
        collect_owned_extras
        [ "${#owned_extras[@]}" -eq 0 ] && break
        sleep 0.01
    done
    collect_owned_extras
    if [ "${#owned_extras[@]}" -ne 0 ]; then
        cleanup_owned_extras KILL
        for ((attempt = 0; attempt < 100; attempt += 1)); do
            collect_owned_extras
            [ "${#owned_extras[@]}" -eq 0 ] && break
            sleep 0.01
        done
    fi
    collect_owned_extras
    [ "${#owned_extras[@]}" -eq 0 ] || fail \
        "owned cgroup descendants survived bounded cleanup"
    [ "$status" -ne 0 ] || status=125
fi
pids_current_after=$(read_metric pids.current)
case "$pids_current_after" in
    *[!0-9]*|"") fail "pids.current final value is invalid" ;;
esac
if [ "$pids_current_after" -gt "$pids_current_before" ]; then
    printf 'oracle-inner: pids.current baseline/final=%s/%s\n' \
        "$pids_current_before" "$pids_current_after" >&2
    [ "$status" -ne 0 ] || status=125
fi
memory_events_after=$(read_metric memory.events)
pids_events_after=$(read_metric pids.events)
"$EVENT_GATE" "$status" "$memory_events_before" "$memory_events_after" \
    "$pids_events_before" "$pids_events_after"
status=$?
set -e
trap - EXIT
report_metrics "$status"
exit "$status"
