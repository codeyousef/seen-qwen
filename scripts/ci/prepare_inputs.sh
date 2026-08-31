#!/usr/bin/env bash

set -euo pipefail
umask 022

ROOT_DIR="$(cd -P -- "${BASH_SOURCE[0]%/*}/../.." && pwd -P)"
CI_ROOT="$ROOT_DIR/.seen/ci"
DOWNLOAD_ROOT="$CI_ROOT/downloads"
TOOLCHAIN_PARENT="$ROOT_DIR/.seen/toolchains"
TOOLCHAIN_ROOT="$TOOLCHAIN_PARENT/seen-0.19.2-linux-x64"
ASSET_ROOT="$ROOT_DIR/.seen/oracle-assets-qwen38"

SEEN_ARCHIVE_URL="https://github.com/codeyousef/SeenLang/releases/download/v0.19.2/seen-0.19.2-linux-x64.tar.gz"
SEEN_ARCHIVE_SHA256="e830da3fb246fd03e64d203dd7291e0b38390b211bbc794a9793b80fa6b901aa"
SEEN_COMPILER_SHA256="e7dc3fab02292a7c04303e5d1574d7f87bfcbe72b6827364c2f3588192134d95"
SEEN_PACKAGE_CLIENT_SHA256="8de71225c7600093df230129fbd71d9ec2f8b5b5a59fe9b4ec59305e977cbc4f"
SEEN_COMPATIBILITY_SHA256="f5fe5bebb9a6d533f65f0726026b6a2e2e7b82d8ef0ed7be9c21899edd9ad313"
SEEN_SOURCE_COMMIT="336baa3ac728bf0887deaf119f405313527409d9"
SEEN_BUILD_ID="5c05b97f921349ef603908cabf0531e09c071acb"
SEEN_CPU_BASELINE="x86-64"
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
    local root=$1

    verify_file "$root/bin/seen" "$SEEN_COMPILER_SHA256" || return 1
    verify_file "$root/bin/seen-pkg" "$SEEN_PACKAGE_CLIENT_SHA256" || return 1
    verify_file "$root/bin/compatibility-manifest.json" \
        "$SEEN_COMPATIBILITY_SHA256" || return 1
}

verify_provenance() {
    local root=$1
    local verifier="$root/lib/seen/toolchain/verify-compiler-provenance.sh"
    local manifest="$root/share/seen/compiler-provenance.env"
    local actual_build_id=""

    command -v readelf >/dev/null 2>&1 || return 1
    [ -f "$verifier" ] && [ -x "$verifier" ] && [ ! -L "$verifier" ] ||
        return 1
    [ -f "$manifest" ] && [ ! -L "$manifest" ] || return 1
    "$verifier" "$manifest" "$root/bin/seen" 0.19.2 || return 1
    grep -Fqx -- "source_commit=$SEEN_SOURCE_COMMIT" "$manifest" || return 1
    grep -Fqx -- "compiler_build_id=$SEEN_BUILD_ID" "$manifest" || return 1
    grep -Fqx -- "cpu_baseline=$SEEN_CPU_BASELINE" "$manifest" || return 1
    actual_build_id=$(readelf -n "$root/bin/seen" 2>/dev/null |
        awk '/Build ID:/ {print tolower($3); exit}')
    [ "$actual_build_id" = "$SEEN_BUILD_ID" ] || return 1
}

write_tree_inventory() {
    local root=$1
    (
        cd -P -- "$root"
        while IFS= read -r -d '' entry; do
            local kind=""
            local mode=""
            local size="-"
            local identity="-"
            local links="-"

            mode=$(stat -c '%a' -- "$entry")
            if [ -L "$entry" ]; then
                kind="symlink"
                identity=$(readlink -- "$entry")
            elif [ -f "$entry" ]; then
                kind="file"
                size=$(stat -c '%s' -- "$entry")
                links=$(stat -c '%h' -- "$entry")
                [ "$links" = 1 ] || return 1
                identity=$(sha256sum -- "$entry" | awk '{print $1}')
            elif [ -d "$entry" ]; then
                kind="directory"
            else
                kind=$(stat -c '%F' -- "$entry")
            fi
            printf '%s\0%s\0%s\0%s\0%s\0%s\0' \
                "${entry#./}" "$kind" "$mode" "$size" "$links" \
                "$identity"
        done < <(LC_ALL=C find . -print0 | LC_ALL=C sort -z)
    )
}

ensure_local_directory() {
    local path=$1
    local resolved=""

    case "$path" in
        "$ROOT_DIR/.seen"|"$ROOT_DIR/.seen/"*) ;;
        *) fail "artifact directory is outside the project-local .seen root" ;;
    esac
    if [ -e "$path" ] || [ -L "$path" ]; then
        [ -d "$path" ] && [ ! -L "$path" ] ||
            fail "artifact directory is not a safe local directory: $path"
    else
        mkdir -- "$path"
    fi
    resolved=$(cd -P -- "$path" && pwd -P)
    [ "$resolved" = "$path" ] ||
        fail "artifact directory resolves outside its project-local path: $path"
}

compare_toolchain_trees() {
    local expected=$1
    local actual=$2
    local scratch=$3

    [ -d "$expected" ] && [ ! -L "$expected" ] || return 1
    [ -d "$actual" ] && [ ! -L "$actual" ] || return 1
    [ -d "$scratch" ] && [ ! -L "$scratch" ] || return 1
    write_tree_inventory "$expected" > "$scratch/expected.inventory"
    write_tree_inventory "$actual" > "$scratch/actual.inventory"
    cmp -s -- "$scratch/expected.inventory" "$scratch/actual.inventory" ||
        return 1
    diff --no-dereference --recursive --brief -- "$expected" "$actual" \
        >/dev/null || return 1
}

if [ "${1:-}" = "--compare-toolchain-trees" ]; then
    [ "$#" -eq 4 ] ||
        fail "usage: --compare-toolchain-trees EXPECTED ACTUAL SCRATCH"
    compare_toolchain_trees "$2" "$3" "$4" ||
        fail "toolchain tree comparison failed"
    echo "PASS: complete toolchain trees match"
    exit 0
fi

ensure_local_directory "$ROOT_DIR/.seen"
ensure_local_directory "$CI_ROOT"
ensure_local_directory "$DOWNLOAD_ROOT"
ensure_local_directory "$TOOLCHAIN_PARENT"
ensure_local_directory "$ASSET_ROOT"

archive="$DOWNLOAD_ROOT/seen-0.19.2-linux-x64.tar.gz"
download_verified "$SEEN_ARCHIVE_URL" "$archive" "$SEEN_ARCHIVE_SHA256"

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
        seen-0.19.2-linux-x64|seen-0.19.2-linux-x64/*) ;;
        *) fail "release archive contains an unsafe member: $member" ;;
    esac
    case "/$member/" in
        */../*) fail "release archive contains parent traversal: $member" ;;
    esac
done < <(tar -tzf "$archive")
tar -xzf "$archive" -C "$extract_root" --no-same-owner --no-same-permissions
extracted="$extract_root/seen-0.19.2-linux-x64"
[ -d "$extracted" ] && [ ! -L "$extracted" ] ||
    fail "release archive did not contain the expected root"
unexpected_type=$(find "$extracted" -mindepth 1 ! -type f ! -type d -print -quit)
[ -z "$unexpected_type" ] ||
    fail "release archive contains an unsupported entry type: $unexpected_type"
verify_toolchain "$extracted" ||
    fail "freshly extracted toolchain failed its exact release lock"
verify_provenance "$extracted" ||
    fail "freshly extracted compiler failed its provenance identity"

if [ -e "$TOOLCHAIN_ROOT" ]; then
    [ -d "$TOOLCHAIN_ROOT" ] && [ ! -L "$TOOLCHAIN_ROOT" ] ||
        fail "toolchain path is not a safe directory"
    verify_toolchain "$TOOLCHAIN_ROOT" ||
        fail "existing toolchain failed its exact release lock"
    compare_toolchain_trees "$extracted" "$TOOLCHAIN_ROOT" "$extract_root" ||
        fail "existing toolchain payload differs from the audited archive"
    verify_provenance "$TOOLCHAIN_ROOT" ||
        fail "existing compiler failed its provenance identity"
else
    mv -- "$extracted" "$TOOLCHAIN_ROOT"
fi
verify_toolchain "$TOOLCHAIN_ROOT" || fail "toolchain failed its exact release lock"
verify_provenance "$TOOLCHAIN_ROOT" || fail "compiler failed its provenance identity"
trap - EXIT
cleanup_extract

download_verified "$QWEN_VOCAB_URL" "$ASSET_ROOT/vocab.json" \
    "$QWEN_VOCAB_SHA256" "$QWEN_VOCAB_BYTES"
download_verified "$QWEN_MERGES_URL" "$ASSET_ROOT/merges.txt" \
    "$QWEN_MERGES_SHA256" "$QWEN_MERGES_BYTES"
download_verified "$QWEN_GENERATION_URL" "$ASSET_ROOT/generation_config.json" \
    "$QWEN_GENERATION_SHA256" "$QWEN_GENERATION_BYTES"
download_verified "$QWEN_MODEL_CARD_URL" "$ASSET_ROOT/README.md" \
    "$QWEN_MODEL_CARD_SHA256" "$QWEN_MODEL_CARD_BYTES"

echo "PASS: exact Seen v0.19.2 toolchain and Qwen tokenizer/sampling inputs verified"
