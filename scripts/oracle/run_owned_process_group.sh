#!/usr/bin/env bash

set -euo pipefail

fail() {
    printf 'oracle-process-group: %s\n' "$*" >&2
    exit 126
}

[ "$#" -ge 2 ] || fail \
    "usage: run_owned_process_group.sh TIMEOUT_SECONDS COMMAND [ARG ...]"
timeout_seconds=$1
shift
case "$timeout_seconds" in
    *[!0-9]*|"") fail "timeout must be an integer" ;;
esac
[ "$timeout_seconds" -ge 1 ] && [ "$timeout_seconds" -le 28800 ] ||
    fail "timeout must be within 1..28800 seconds"
command -v setsid >/dev/null 2>&1 || fail "setsid is required"
command -v timeout >/dev/null 2>&1 || fail "timeout is required"

# The child stops itself after setsid(2), before executing user code. This gives
# the supervisor an unambiguous owned process-group identity without a file,
# FIFO, PID search, or contact with any process outside this invocation.
setsid bash -c '
    kill -STOP "$$"
    duration=$1
    shift
    exec timeout --foreground --signal=TERM --kill-after=2 \
        "$duration" "$@"
' bash "$timeout_seconds" "$@" &
group_leader=$!

stopped=0
for ((attempt = 0; attempt < 100; attempt += 1)); do
    if [ ! -r "/proc/$group_leader/status" ]; then
        break
    fi
    state=""
    while read -r field value _; do
        if [ "$field" = "State:" ]; then
            state=$value
            break
        fi
    done < "/proc/$group_leader/status"
    if [ "$state" = T ] || [ "$state" = t ]; then
        stopped=1
        break
    fi
    sleep 0.01
done
[ "$stopped" -eq 1 ] || {
    wait "$group_leader" 2>/dev/null || true
    fail "owned process group did not reach its startup barrier"
}
read -r _ _ _ _ actual_group _ < "/proc/$group_leader/stat" || {
    kill -KILL "$group_leader" 2>/dev/null || true
    wait "$group_leader" 2>/dev/null || true
    fail "owned process-group identity disappeared"
}
[ "$actual_group" = "$group_leader" ] || {
    kill -KILL "$group_leader" 2>/dev/null || true
    wait "$group_leader" 2>/dev/null || true
    fail "owned process-group identity is ambiguous"
}
kill -CONT "$group_leader"

set +e
wait "$group_leader"
status=$?
set -e

descendants_remained=0
if kill -0 -- "-$group_leader" 2>/dev/null; then
    descendants_remained=1
    printf 'oracle-process-group: owned descendants remained after leader exit\n' >&2
    kill -TERM -- "-$group_leader" 2>/dev/null || true
    for ((attempt = 0; attempt < 100; attempt += 1)); do
        kill -0 -- "-$group_leader" 2>/dev/null || break
        sleep 0.01
    done
    if kill -0 -- "-$group_leader" 2>/dev/null; then
        kill -KILL -- "-$group_leader" 2>/dev/null || true
        for ((attempt = 0; attempt < 100; attempt += 1)); do
            kill -0 -- "-$group_leader" 2>/dev/null || break
            sleep 0.01
        done
    fi
    kill -0 -- "-$group_leader" 2>/dev/null &&
        fail "owned descendants survived bounded cleanup"
fi

if [ "$status" -ne 0 ]; then
    exit "$status"
fi
[ "$descendants_remained" -eq 0 ] || exit 125
exit 0
