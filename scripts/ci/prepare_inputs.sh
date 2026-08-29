#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
CI_ROOT="$ROOT_DIR/.seen/ci"
DOWNLOAD_ROOT="$CI_ROOT/downloads"
TOOLCHAIN_PARENT="$ROOT_DIR/.seen/toolchains"
TOOLCHAIN_ROOT="$TOOLCHAIN_PARENT/seen-0.18.1-linux-x64"
ASSET_ROOT="$ROOT_DIR/.seen/oracle-assets-qwen38"

SEEN_ARCHIVE_URL="https://github.com/codeyousef/SeenLang/releases/download/v0.18.1/seen-0.18.1-linux-x64.tar.gz"
SEEN_ARCHIVE_SHA256="3b25d7e693bba340de502558aaeed55b0ad61ebe2284b14f3c95d7349a81cfdf"
SEEN_COMPILER_SHA256="276320b7495786838708c3d350b849b10d065abc640cca315f632050199d9a30"
SEEN_PACKAGE_CLIENT_SHA256="eb1b592ff6132bad9027dcf7f13057c886accfa2c03ceeb26047fe7e2c163845"
SEEN_COMPATIBILITY_SHA256="52bbc506c0fbf30b8eb868e3323db3ccafac50f194705ff7f9680b73c794bf9c"
QWEN_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN_VOCAB_URL="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/$QWEN_REVISION/vocab.json?download=true"
QWEN_MERGES_URL="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/$QWEN_REVISION/merges.txt?download=true"
QWEN_GENERATION_URL="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/$QWEN_REVISION/generation_config.json?download=true"
QWEN_MODEL_CARD_URL="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/$QWEN_REVISION/README.md?download=true"
QWEN_VOCAB_SHA256="ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
QWEN_MERGES_SHA256="a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
QWEN_GENERATION_SHA256="e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
QWEN_MODEL_CARD_SHA256="57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
QWEN_VOCAB_BYTES=6722759
QWEN_MERGES_BYTES=3353259
QWEN_GENERATION_BYTES=202
QWEN_MODEL_CARD_BYTES=65012

fail() {
    echo "ci-inputs: $*" >&2
    exit 1
}

verify_file() {
    local path=$1
    local expected_sha=$2
    local expected_bytes=${3:-}
    local actual_sha=""
    local actual_bytes=""

    [ -f "$path" ] && [ ! -L "$path" ] || return 1
    actual_sha=$(sha256sum -- "$path" | awk '{print $1}')
    [ "$actual_sha" = "$expected_sha" ] || return 1
    if [ -n "$expected_bytes" ]; then
        actual_bytes=$(stat -c '%s' -- "$path")
        [ "$actual_bytes" = "$expected_bytes" ] || return 1
    fi
}

download_verified() {
    local url=$1
    local destination=$2
    local expected_sha=$3
    local expected_bytes=${4:-}
    local temporary="$destination.part.$$"

    if [ -e "$destination" ]; then
        verify_file "$destination" "$expected_sha" "$expected_bytes" ||
            fail "existing input failed its exact lock: $destination"
        return 0
    fi
    [ ! -e "$temporary" ] || fail "temporary download path already exists: $temporary"
    if ! curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
        --retry 3 --retry-all-errors --connect-timeout 30 --max-time 600 \
        --output "$temporary" "$url"; then

        rm -f -- "$temporary"
        fail "could not download exact input: $destination"
    fi
    if ! verify_file "$temporary" "$expected_sha" "$expected_bytes"; then
        rm -f -- "$temporary"
        fail "downloaded input failed its exact lock: $destination"
    fi
    mv -- "$temporary" "$destination"
}

verify_toolchain() {
    verify_file "$TOOLCHAIN_ROOT/bin/seen" "$SEEN_COMPILER_SHA256" || return 1
    verify_file "$TOOLCHAIN_ROOT/bin/seen-pkg" "$SEEN_PACKAGE_CLIENT_SHA256" || return 1
    verify_file "$TOOLCHAIN_ROOT/bin/compatibility-manifest.json" \
        "$SEEN_COMPATIBILITY_SHA256" || return 1
}

mkdir -p -- "$DOWNLOAD_ROOT" "$TOOLCHAIN_PARENT" "$ASSET_ROOT"

archive="$DOWNLOAD_ROOT/seen-0.18.1-linux-x64.tar.gz"
download_verified "$SEEN_ARCHIVE_URL" "$archive" "$SEEN_ARCHIVE_SHA256"

if [ -e "$TOOLCHAIN_ROOT" ]; then
    [ -d "$TOOLCHAIN_ROOT" ] && [ ! -L "$TOOLCHAIN_ROOT" ] ||
        fail "toolchain path is not a safe directory"
    verify_toolchain || fail "existing toolchain failed its exact release lock"
else
    extract_root=$(mktemp -d "$CI_ROOT/toolchain.extract.XXXXXX")
    cleanup_extract() {
        local status=$?
        case "$extract_root" in
            "$CI_ROOT"/toolchain.extract.*)
                [ -d "$extract_root" ] && [ ! -L "$extract_root" ] &&
                    rm -rf -- "$extract_root"
                ;;
            *) return 1 ;;
        esac
        return "$status"
    }
    trap cleanup_extract EXIT
    while IFS= read -r member; do
        case "$member" in
            seen-0.18.1-linux-x64|seen-0.18.1-linux-x64/*) ;;
            *) fail "release archive contains an unsafe member: $member" ;;
        esac
        case "/$member/" in
            */../*) fail "release archive contains parent traversal: $member" ;;
        esac
    done < <(tar -tzf "$archive")
    tar -xzf "$archive" -C "$extract_root" --no-same-owner --no-same-permissions
    extracted="$extract_root/seen-0.18.1-linux-x64"
    [ -d "$extracted" ] && [ ! -L "$extracted" ] ||
        fail "release archive did not contain the expected root"
    mv -- "$extracted" "$TOOLCHAIN_ROOT"
    verify_toolchain || fail "extracted toolchain failed its exact release lock"
    trap - EXIT
    cleanup_extract
fi

download_verified "$QWEN_VOCAB_URL" "$ASSET_ROOT/vocab.json" \
    "$QWEN_VOCAB_SHA256" "$QWEN_VOCAB_BYTES"
download_verified "$QWEN_MERGES_URL" "$ASSET_ROOT/merges.txt" \
    "$QWEN_MERGES_SHA256" "$QWEN_MERGES_BYTES"
download_verified "$QWEN_GENERATION_URL" "$ASSET_ROOT/generation_config.json" \
    "$QWEN_GENERATION_SHA256" "$QWEN_GENERATION_BYTES"
download_verified "$QWEN_MODEL_CARD_URL" "$ASSET_ROOT/README.md" \
    "$QWEN_MODEL_CARD_SHA256" "$QWEN_MODEL_CARD_BYTES"

echo "PASS: exact Seen v0.18.1 toolchain and Qwen tokenizer/sampling inputs verified"
