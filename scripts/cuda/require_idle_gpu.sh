#!/usr/bin/env bash

set -euo pipefail

NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}

fail() {
    printf 'qwen-gpu-preflight: %s\n' "$*" >&2
    exit 75
}

trim() {
    local value=$1
    value=${value#"${value%%[![:space:]]*}"}
    value=${value%"${value##*[![:space:]]}"}
    printf '%s' "$value"
}

if ! inventory=$(
    "$NVIDIA_SMI_BIN" --query-gpu=index,uuid --format=csv,noheader,nounits 2>&1
); then
    fail "cannot enumerate GPUs: $inventory"
fi

mapfile -t devices < <(printf '%s\n' "$inventory" | sed '/^[[:space:]]*$/d')
[ "${#devices[@]}" -gt 0 ] || fail "no GPU is visible"

selector=
if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
    selector=$(trim "$CUDA_VISIBLE_DEVICES")
    [ -n "$selector" ] || fail "CUDA_VISIBLE_DEVICES hides every GPU"
    case "$selector" in
        *,*) fail "CUDA_VISIBLE_DEVICES must select exactly one GPU" ;;
    esac
fi

selected_index=
selected_uuid=
for device in "${devices[@]}"; do
    IFS=, read -r raw_index raw_uuid extra <<< "$device"
    [ -z "${extra:-}" ] || fail "malformed GPU inventory row: $device"
    index=$(trim "$raw_index")
    uuid=$(trim "$raw_uuid")
    [[ "$index" =~ ^[0-9]+$ ]] || fail "malformed GPU index: $index"
    [[ "$uuid" =~ ^GPU-[[:alnum:]-]+$ ]] || fail "malformed GPU UUID: $uuid"
    if [ -z "$selector" ]; then
        [ "${#devices[@]}" -eq 1 ] || fail \
            "multiple GPUs are visible; select exactly one with CUDA_VISIBLE_DEVICES"
        selected_index=$index
        selected_uuid=$uuid
    elif [ "$selector" = "$index" ] || [ "$selector" = "$uuid" ]; then
        [ -z "$selected_uuid" ] || fail "GPU selector is ambiguous: $selector"
        selected_index=$index
        selected_uuid=$uuid
    fi
done

[ -n "$selected_uuid" ] || fail "selected GPU is not visible: $selector"

if ! applications=$(
    "$NVIDIA_SMI_BIN" --id="$selected_uuid" \
        --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>&1
); then
    fail "cannot prove GPU $selected_uuid is idle: $applications"
fi

mapfile -t processes < <(printf '%s\n' "$applications" | sed '/^[[:space:]]*$/d')
for process in "${processes[@]}"; do
    IFS=, read -r raw_pid process_name <<< "$process"
    pid=$(trim "$raw_pid")
    [[ "$pid" =~ ^[0-9]+$ ]] || fail "malformed compute-process row: $process"
    fail "GPU $selected_uuid is busy with compute PID $pid (${process_name:-unknown})"
done

printf 'qwen-gpu-preflight: GPU %s (%s) is idle\n' \
    "$selected_index" "$selected_uuid" >&2
printf '%s\n' "$selected_uuid"
