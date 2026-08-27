#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
PREPARE_INPUTS="$ROOT_DIR/scripts/ci/prepare_inputs.sh"
INNER_GATE="$ROOT_DIR/scripts/ci/required_inner.sh"
CI_ROOT="$ROOT_DIR/.seen/ci"
CI_IMAGE="silkeh/clang@sha256:a370fe4e8ecd284143bbfde1185bef4c1b6b72f45af4823812b9afe84cd1a14d"
MEMORY_CEILING_BYTES=6442450944
MEMORY_RESERVE_BYTES=1073741824
TASKS_MAX=24
TIMEOUT_SECS=900

fail() {
    echo "ci-required: $*" >&2
    exit 126
}

[ "$(uname -s)" = "Linux" ] || fail "Linux is required"
[ -x "$PREPARE_INPUTS" ] && [ ! -L "$PREPARE_INPUTS" ] ||
    fail "input preparation entrypoint is missing or unsafe"
[ -x "$INNER_GATE" ] && [ ! -L "$INNER_GATE" ] ||
    fail "inner required gate is missing or unsafe"
command -v docker >/dev/null 2>&1 || fail "Docker is required for the kernel hard scope"
docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"

memory_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
memory_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
case "$memory_total_kib:$memory_available_kib" in
    *[!0-9:]*|:|*::*) fail "host memory readback is invalid" ;;
esac
memory_available_bytes=$((memory_available_kib * 1024))
[ "$memory_available_bytes" -gt "$MEMORY_RESERVE_BYTES" ] ||
    fail "host has no memory beyond the required 1 GiB reserve"
memory_bytes=$((memory_available_bytes - MEMORY_RESERVE_BYTES))
[ "$memory_bytes" -le "$MEMORY_CEILING_BYTES" ] ||
    memory_bytes=$MEMORY_CEILING_BYTES
memory_vmem_kib=$((memory_bytes / 1024))
[ "$memory_vmem_kib" -ge 1048576 ] ||
    fail "derived hard scope is below the 1 GiB minimum"
echo "ci-required: host MemTotal=${memory_total_kib}KiB MemAvailable=${memory_available_kib}KiB derived_memory.max=${memory_bytes} memory.swap.max=0 pids.max=${TASKS_MAX}"

before_status=$(git -C "$ROOT_DIR" status --porcelain=v1 --untracked-files=all)
"$PREPARE_INPUTS"

mkdir -p -- "$CI_ROOT/artifacts" "$CI_ROOT/home" "$CI_ROOT/output" "$CI_ROOT/tmp"
for writable in "$CI_ROOT/artifacts" "$CI_ROOT/home" "$CI_ROOT/output" "$CI_ROOT/tmp"; do
    [ -d "$writable" ] && [ ! -L "$writable" ] ||
        fail "CI writable root is unsafe: $writable"
done

runner_uid=$(id -u)
runner_gid=$(id -g)
case "$runner_uid:$runner_gid" in
    *[!0-9:]*|:|*::*) fail "runner uid/gid are invalid" ;;
esac

docker pull "$CI_IMAGE"
docker run --rm --platform linux/amd64 \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$runner_uid:$runner_gid" \
    --memory "$memory_bytes" \
    --memory-swap "$memory_bytes" \
    --pids-limit "$TASKS_MAX" \
    --cpus 2 \
    --ulimit stack=8388608:8388608 \
    --ulimit nofile=1024:1024 \
    --mount "type=bind,src=$ROOT_DIR,dst=/workspace,readonly" \
    --mount "type=bind,src=$ROOT_DIR/.seen,dst=/workspace/.seen" \
    --mount "type=bind,src=$CI_ROOT/tmp,dst=/tmp" \
    --workdir /workspace \
    --env HOME=/workspace/.seen/ci/home \
    --env TMPDIR=/tmp \
    --env SEEN_ARTIFACT_ROOT=/workspace/.seen/ci/artifacts \
    --env SEEN_LOW_MEMORY=1 \
    --env SEEN_JOBS=1 \
    --env SEEN_OPT_JOBS=1 \
    --env SEEN_EXPECTED_MEMORY_BYTES="$memory_bytes" \
    --env SEEN_MAIN_VMEM_KB="$memory_vmem_kib" \
    --env SEEN_OPT_VMEM_KB=2097152 \
    --env SEEN_MEMORY_LIMIT_BYTES="$memory_bytes" \
    --env RAYON_NUM_THREADS=1 \
    --env OMP_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env GOMAXPROCS=1 \
    "$CI_IMAGE" \
    timeout --foreground --signal=KILL "$TIMEOUT_SECS" \
        /workspace/scripts/ci/required_inner.sh

after_status=$(git -C "$ROOT_DIR" status --porcelain=v1 --untracked-files=all)
[ "$after_status" = "$before_status" ] ||
    fail "required CI changed tracked or untracked repository state"

echo "PASS: required Seen Qwen CI gates"
