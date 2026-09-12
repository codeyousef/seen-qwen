#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=/workspace
TOOLCHAIN_ROOT="$ROOT_DIR/.seen/toolchains/seen-0.20.7-linux-x64"
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
[ "$memory_max" -le 7516192768 ] || fail "memory.max exceeds the 7 GiB ceiling"
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
    5fa95f150e652843795611810f02de9ca1fb6e2914ac1159fd75912bc4ae8231 \
    "$SEEN_COMPILER" | sha256sum -c -
printf '%s  %s\n' \
    91b9db87826131517830e8b0b65280c71a5a5d4ea953325c60e4fee1fbf4a754 \
    "$SEEN_PACKAGE_CLIENT" | sha256sum -c -
printf '%s  %s\n' \
    b8a85368c2092fd635a5be45e505082f40a9e25107bcd33bf12e0caa45c540ee \
    "$COMPATIBILITY_MANIFEST" | sha256sum -c -
printf '%s  %s\n' \
    ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 \
    "$ASSET_ROOT/vocab.json" | sha256sum -c -
printf '%s  %s\n' \
    a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d \
    "$ASSET_ROOT/merges.txt" | sha256sum -c -
printf '%s  %s\n' \
    e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e \
    "$ASSET_ROOT/generation_config.json" | sha256sum -c -
printf '%s  %s\n' \
    57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641 \
    "$ASSET_ROOT/README.md" | sha256sum -c -
[ "$(stat -c '%s' "$ASSET_ROOT/vocab.json")" = "6722759" ] ||
    fail "vocabulary byte length changed"
[ "$(stat -c '%s' "$ASSET_ROOT/merges.txt")" = "3353259" ] ||
    fail "merge-table byte length changed"

python3 -c 'import json; p="/workspace/.seen/toolchains/seen-0.20.7-linux-x64/bin/compatibility-manifest.json"; d=json.load(open(p, encoding="utf-8")); assert d["schema"] == "seen-compatibility-manifest-v1"; assert d["release_version"] == "0.20.7"; assert d["components"]["compiler"]["version"] == "0.20.7"; assert d["components"]["package_client"] == {"protocol": "SEENPKG1", "version": "0.20.7"}; assert d["components"]["runtime"]["abi"] == "runtime-v4"; assert d["components"]["standard_library"] == {"module_manifest_version": 1, "version": "0.5.0"}; assert d["components"]["llvm"]["minimum_major"] == 19; assert d["platforms"]["linux-x86_64"] == "required"; assert d["determinism"]["certification"]["installed_archive_required"] is True; assert d["determinism"]["certification"]["signed_evidence_required"] is True'

toolchain_hash_before=$(find "$TOOLCHAIN_ROOT" -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | awk '{print $1}')
outside_objects_before=$(find "$ROOT_DIR" -path "$ROOT_DIR/.seen" -prune -o \
    -type f \( -name '*.o' -o -name '*.sig' -o -name '*.a' \) -print0 |
    sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')

"$SEEN_COMPILER" --version | grep -Fx 'Seen 0.20.7'
"$SEEN_PACKAGE_CLIENT" --expect-version 0.20.7 version |
    grep -Fx 'seen-pkg 0.20.7 (SEENPKG1)'
python3 -m unittest tests/test_ci_contract.py tests/test_cuda_reference_primitives.py \
    tests/test_cuda_reference_utilities.py tests/test_cuda_gdn_state.py \
    tests/test_cuda_gdn_recurrent_decode.py tests/test_cuda_gdn_recurrent_prefill.py \
    tests/test_cuda_gdn_gated_output.py tests/test_cuda_attention_qk.py \
    tests/test_cuda_kv_cache.py tests/test_cuda_attention_decode.py \
    tests/test_cuda_attention_prefill.py \
    tests/test_cuda_attention_output_gate.py \
    tests/test_cuda_projection_descriptor.py \
    tests/test_cuda_ffn_execution.py \
    tests/test_qwen_tokenizer_oracles.py \
    tests/test_sampling_profiles.py tests/test_hybrid_mini_contract.py \
    tests/test_hybrid_mini_assets.py tests/test_hybrid_mini_oracle.py \
    tests/test_cpu_attention_oracle.py tests/test_cpu_gdn_oracle.py \
    tests/test_cpu_head_oracle.py tests/test_cpu_engine_oracle.py \
    tests/test_official_operator_layer_oracles.py \
    tests/test_official_full_model_oracles.py tests/test_sqw_contract.py \
    tests/test_sqw_reader.py tests/test_sqw_writer.py \
    tests/test_q8_reference_codec.py tests/test_q4_reference_codec.py \
    tests/test_conversion_plan.py tests/test_shard_stream.py \
    tests/test_conversion_evidence.py tests/test_conversion_finalizer.py \
    tests/test_calibration_lock.py tests/test_sensitivity_statistics.py \
    tests/test_quantization_policy_resolver.py tests/test_engine_artifact.py
"$SEEN_PACKAGE_CLIENT" audit --lock Seen.lock
"$SEEN_COMPILER" check tests/qwn_023b_hybrid_mini_assets_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_023a_hybrid_mini_contract_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_024a_cpu_reference_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_024b_cpu_attention_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_024c_cpu_gdn_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_024d_cpu_head_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_025a_operator_layer_oracle_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_025b_full_model_oracle_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_030a_sqw_contract_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_030b_sqw_reader_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_030c_sqw_writer_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_031a_reference_codec_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_031b_q8_reference_codec_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_031c_q4_reference_codec_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_032a_conversion_plan_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_032b_shard_stream_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_032c_conversion_evidence_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_032d_conversion_finalizer_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_033a_calibration_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_033b_sensitivity_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_033c_policy_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_034a_engine_artifact_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_040a_reference_primitives_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_040b_reference_utilities_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_041a_gdn_state_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_041b_gdn_decode_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_041c_gdn_prefill_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_041d_gdn_output_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_042a_attention_qk_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_042b_kv_cache_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_042c_attention_decode_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_042d_attention_prefill_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_042e_attention_output_gate_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_043a_projection_descriptor_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_043b_ffn_execution_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_022d_sampling_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_022b_tokenizer_test.seen --frozen
"$SEEN_COMPILER" check tests/qwn_022c_chat_template_test.seen --frozen

"$SEEN_COMPILER" compile tests/qwn_030b_sqw_reader_test.seen \
    "$OUTPUT_ROOT/qwn_030b_sqw_reader_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_030b_sqw_reader_test_fast"
"$SEEN_COMPILER" compile tests/qwn_040a_reference_primitives_test.seen \
    "$OUTPUT_ROOT/qwn_040a_reference_primitives_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_040a_reference_primitives_test_fast"
"$SEEN_COMPILER" compile tests/qwn_040a_reference_primitives_test.seen \
    "$OUTPUT_ROOT/qwn_040a_reference_primitives_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_040a_reference_primitives_test"
"$SEEN_COMPILER" compile tests/qwn_043a_projection_descriptor_test.seen \
    "$OUTPUT_ROOT/qwn_043a_projection_descriptor_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_043a_projection_descriptor_test_fast"
"$SEEN_COMPILER" compile tests/qwn_043a_projection_descriptor_test.seen \
    "$OUTPUT_ROOT/qwn_043a_projection_descriptor_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_043a_projection_descriptor_test"
QWN_043B_CPU_PROJECT=$(mktemp -d "$OUTPUT_ROOT/qwn_043b_cpu_project.XXXXXX")
cp -R "$ROOT_DIR/src" "$QWN_043B_CPU_PROJECT/src"
mkdir -p "$QWN_043B_CPU_PROJECT/tests" "$QWN_043B_CPU_PROJECT/native/lib"
cp "$ROOT_DIR/tests/qwn_043b_ffn_execution_test.seen" \
    "$QWN_043B_CPU_PROJECT/tests/qwn_043b_ffn_execution_test.seen"
cp "$ROOT_DIR/tests/qwn_043b_project/Seen.toml" \
    "$QWN_043B_CPU_PROJECT/Seen.toml"
clang -shared -fPIC -O2 -Wl,--no-undefined \
    "$ROOT_DIR/tests/qwn_043b_project/seen_cuda_link_stubs.c" \
    -o "$QWN_043B_CPU_PROJECT/native/lib/libseen_cuda.so"
clang -shared -fPIC -O2 -Wl,--no-undefined \
    "$ROOT_DIR/tests/qwn_043b_project/seen_qwen_cuda_link_stubs.c" \
    -o "$QWN_043B_CPU_PROJECT/native/lib/libseen_qwen_cuda.so"
"$SEEN_COMPILER" compile \
    "$QWN_043B_CPU_PROJECT/tests/qwn_043b_ffn_execution_test.seen" \
    "$OUTPUT_ROOT/qwn_043b_ffn_execution_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --offline
"$OUTPUT_ROOT/qwn_043b_ffn_execution_test_fast"
"$SEEN_COMPILER" compile \
    "$QWN_043B_CPU_PROJECT/tests/qwn_043b_ffn_execution_test.seen" \
    "$OUTPUT_ROOT/qwn_043b_ffn_execution_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --offline
"$OUTPUT_ROOT/qwn_043b_ffn_execution_test"
"$SEEN_COMPILER" compile tests/qwn_040b_reference_utilities_test.seen \
    "$OUTPUT_ROOT/qwn_040b_reference_utilities_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_040b_reference_utilities_test_fast"
"$SEEN_COMPILER" compile tests/qwn_040b_reference_utilities_test.seen \
    "$OUTPUT_ROOT/qwn_040b_reference_utilities_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_040b_reference_utilities_test"
"$SEEN_COMPILER" compile tests/qwn_041a_gdn_state_test.seen \
    "$OUTPUT_ROOT/qwn_041a_gdn_state_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041a_gdn_state_test_fast"
"$SEEN_COMPILER" compile tests/qwn_041a_gdn_state_test.seen \
    "$OUTPUT_ROOT/qwn_041a_gdn_state_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041a_gdn_state_test"
"$SEEN_COMPILER" compile tests/qwn_041b_gdn_decode_test.seen \
    "$OUTPUT_ROOT/qwn_041b_gdn_decode_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041b_gdn_decode_test_fast"
"$SEEN_COMPILER" compile tests/qwn_041b_gdn_decode_test.seen \
    "$OUTPUT_ROOT/qwn_041b_gdn_decode_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041b_gdn_decode_test"
"$SEEN_COMPILER" compile tests/qwn_041c_gdn_prefill_test.seen \
    "$OUTPUT_ROOT/qwn_041c_gdn_prefill_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041c_gdn_prefill_test_fast"
"$SEEN_COMPILER" compile tests/qwn_041c_gdn_prefill_test.seen \
    "$OUTPUT_ROOT/qwn_041c_gdn_prefill_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041c_gdn_prefill_test"
"$SEEN_COMPILER" compile tests/qwn_041d_gdn_output_test.seen \
    "$OUTPUT_ROOT/qwn_041d_gdn_output_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041d_gdn_output_test_fast"
"$SEEN_COMPILER" compile tests/qwn_041d_gdn_output_test.seen \
    "$OUTPUT_ROOT/qwn_041d_gdn_output_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_041d_gdn_output_test"
"$SEEN_COMPILER" compile tests/qwn_042a_attention_qk_test.seen \
    "$OUTPUT_ROOT/qwn_042a_attention_qk_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042a_attention_qk_test_fast"
"$SEEN_COMPILER" compile tests/qwn_042a_attention_qk_test.seen \
    "$OUTPUT_ROOT/qwn_042a_attention_qk_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042a_attention_qk_test"
"$SEEN_COMPILER" compile tests/qwn_042b_kv_cache_test.seen \
    "$OUTPUT_ROOT/qwn_042b_kv_cache_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042b_kv_cache_test_fast"
"$SEEN_COMPILER" compile tests/qwn_042b_kv_cache_test.seen \
    "$OUTPUT_ROOT/qwn_042b_kv_cache_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042b_kv_cache_test"
"$SEEN_COMPILER" compile tests/qwn_042c_attention_decode_test.seen \
    "$OUTPUT_ROOT/qwn_042c_attention_decode_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042c_attention_decode_test_fast"
"$SEEN_COMPILER" compile tests/qwn_042c_attention_decode_test.seen \
    "$OUTPUT_ROOT/qwn_042c_attention_decode_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042c_attention_decode_test"
"$SEEN_COMPILER" compile tests/qwn_042d_attention_prefill_test.seen \
    "$OUTPUT_ROOT/qwn_042d_attention_prefill_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042d_attention_prefill_test_fast"
"$SEEN_COMPILER" compile tests/qwn_042d_attention_prefill_test.seen \
    "$OUTPUT_ROOT/qwn_042d_attention_prefill_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042d_attention_prefill_test"
"$SEEN_COMPILER" compile tests/qwn_042e_attention_output_gate_test.seen \
    "$OUTPUT_ROOT/qwn_042e_attention_output_gate_test_fast" \
    --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042e_attention_output_gate_test_fast"
"$SEEN_COMPILER" compile tests/qwn_042e_attention_output_gate_test.seen \
    "$OUTPUT_ROOT/qwn_042e_attention_output_gate_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_042e_attention_output_gate_test"
"$SEEN_COMPILER" compile tests/qwn_024e_cpu_engine_test.seen \
    "$OUTPUT_ROOT/qwn_024e_cpu_engine_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_024e_cpu_engine_test"
"$SEEN_COMPILER" compile tests/qwn_023b_hybrid_mini_assets_test.seen \
    "$OUTPUT_ROOT/qwn_023b_hybrid_mini_assets_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_023b_hybrid_mini_assets_test"
"$SEEN_COMPILER" compile tests/qwn_023a_hybrid_mini_contract_test.seen \
    "$OUTPUT_ROOT/qwn_023a_hybrid_mini_contract_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_023a_hybrid_mini_contract_test"
"$SEEN_COMPILER" compile tests/qwn_024a_cpu_reference_test.seen \
    "$OUTPUT_ROOT/qwn_024a_cpu_reference_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_024a_cpu_reference_test"
"$SEEN_COMPILER" compile tests/qwn_024b_cpu_attention_test.seen \
    "$OUTPUT_ROOT/qwn_024b_cpu_attention_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_024b_cpu_attention_test"
"$SEEN_COMPILER" compile tests/qwn_024c_cpu_gdn_test.seen \
    "$OUTPUT_ROOT/qwn_024c_cpu_gdn_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_024c_cpu_gdn_test"
"$SEEN_COMPILER" compile tests/qwn_024d_cpu_head_test.seen \
    "$OUTPUT_ROOT/qwn_024d_cpu_head_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_024d_cpu_head_test"
"$SEEN_COMPILER" compile tests/qwn_025a_operator_layer_oracle_test.seen \
    "$OUTPUT_ROOT/qwn_025a_operator_layer_oracle_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_025a_operator_layer_oracle_test"
"$SEEN_COMPILER" compile tests/qwn_025b_full_model_oracle_test.seen \
    "$OUTPUT_ROOT/qwn_025b_full_model_oracle_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_025b_full_model_oracle_test"
"$SEEN_COMPILER" compile tests/qwn_030a_sqw_contract_test.seen \
    "$OUTPUT_ROOT/qwn_030a_sqw_contract_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_030a_sqw_contract_test"
"$SEEN_COMPILER" compile tests/qwn_030b_sqw_reader_test.seen \
    "$OUTPUT_ROOT/qwn_030b_sqw_reader_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_030b_sqw_reader_test"
"$SEEN_COMPILER" compile tests/qwn_030c_sqw_writer_test.seen \
    "$OUTPUT_ROOT/qwn_030c_sqw_writer_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_030c_sqw_writer_test"
"$SEEN_COMPILER" compile tests/qwn_031a_reference_codec_test.seen \
    "$OUTPUT_ROOT/qwn_031a_reference_codec_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_031a_reference_codec_test"
"$SEEN_COMPILER" compile tests/qwn_031b_q8_reference_codec_test.seen \
    "$OUTPUT_ROOT/qwn_031b_q8_reference_codec_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_031b_q8_reference_codec_test"
"$SEEN_COMPILER" compile tests/qwn_031c_q4_reference_codec_test.seen \
    "$OUTPUT_ROOT/qwn_031c_q4_reference_codec_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_031c_q4_reference_codec_test"
"$SEEN_COMPILER" compile tests/qwn_032a_conversion_plan_test.seen \
    "$OUTPUT_ROOT/qwn_032a_conversion_plan_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_032a_conversion_plan_test"
"$SEEN_COMPILER" compile tests/qwn_032b_shard_stream_test.seen \
    "$OUTPUT_ROOT/qwn_032b_shard_stream_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_032b_shard_stream_test"
"$SEEN_COMPILER" compile tests/qwn_032c_conversion_evidence_test.seen \
    "$OUTPUT_ROOT/qwn_032c_conversion_evidence_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_032c_conversion_evidence_test"
"$SEEN_COMPILER" compile tests/qwn_032d_conversion_finalizer_test.seen \
    "$OUTPUT_ROOT/qwn_032d_conversion_finalizer_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_032d_conversion_finalizer_test"
"$SEEN_COMPILER" compile tests/qwn_033a_calibration_test.seen \
    "$OUTPUT_ROOT/qwn_033a_calibration_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_033a_calibration_test"
"$SEEN_COMPILER" compile tests/qwn_033b_sensitivity_test.seen \
    "$OUTPUT_ROOT/qwn_033b_sensitivity_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_033b_sensitivity_test"
"$SEEN_COMPILER" compile tests/qwn_033c_policy_test.seen \
    "$OUTPUT_ROOT/qwn_033c_policy_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_033c_policy_test"
"$SEEN_COMPILER" compile tests/qwn_034a_engine_artifact_test.seen \
    "$OUTPUT_ROOT/qwn_034a_engine_artifact_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_034a_engine_artifact_test"
"$SEEN_COMPILER" compile tests/qwn_022c_chat_template_test.seen \
    "$OUTPUT_ROOT/qwn_022c_chat_template_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_022c_chat_template_test"
"$SEEN_COMPILER" compile tests/qwn_022d_sampling_test.seen \
    "$OUTPUT_ROOT/qwn_022d_sampling_test" \
    --release --lto=thin --target-cpu=x86-64 --no-cache \
    --jobs 1 --opt-jobs 1 --no-fork --frozen
"$OUTPUT_ROOT/qwn_022d_sampling_test"
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

echo "PASS: exact locked Seen Qwen mini-model assets, tokenizer, chat-template, and sampling gates"
