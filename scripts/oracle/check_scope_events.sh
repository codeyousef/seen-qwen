#!/usr/bin/env bash

set -euo pipefail

fail() {
    printf 'oracle-events: %s\n' "$*" >&2
    exit 126
}

[ "$#" -eq 5 ] || fail \
    "usage: check_scope_events.sh CHILD_STATUS MEMORY_BEFORE MEMORY_AFTER PIDS_BEFORE PIDS_AFTER"

child_status=$1
memory_before=$2
memory_after=$3
pids_before=$4
pids_after=$5

case "$child_status" in
    *[!0-9]*|"") fail "child status must be an integer" ;;
esac
[ "$child_status" -le 255 ] || fail "child status must be within 0..255"

counter() {
    local snapshot=$1
    local key=$2
    local value=""
    if ! value=$(awk -v key="$key" '
        $1 == key { count += 1; value = $2 }
        END {
            if (count != 1) exit 1
            print value
        }
    ' <<< "$snapshot"); then
        fail "event snapshot must contain exactly one $key counter"
    fi
    case "$value" in
        *[!0-9]*|"") fail "$key counter must be a nonnegative integer" ;;
    esac
    printf '%s' "$value"
}

containment_failed=0
for key in oom oom_kill oom_group_kill; do
    before=$(counter "$memory_before" "$key")
    after=$(counter "$memory_after" "$key")
    [ "$after" -ge "$before" ] || fail "$key counter decreased"
    if [ "$after" -gt "$before" ]; then
        printf 'oracle-events: memory.events %s increased by %s\n' \
            "$key" "$((after - before))" >&2
        containment_failed=1
    fi
done

before=$(counter "$pids_before" max)
after=$(counter "$pids_after" max)
[ "$after" -ge "$before" ] || fail "pids.events max counter decreased"
if [ "$after" -gt "$before" ]; then
    printf 'oracle-events: pids.events max increased by %s\n' \
        "$((after - before))" >&2
    containment_failed=1
fi

if [ "$child_status" -ne 0 ]; then
    exit "$child_status"
fi
[ "$containment_failed" -eq 0 ] || exit 125
exit 0
