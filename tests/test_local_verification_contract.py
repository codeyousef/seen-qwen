#!/usr/bin/env python3
"""Static fail-closed contracts for Seen Qwen local verification."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
RUNNER = ROOT / "scripts/verification/run_required.sh"
PREPARE = ROOT / "scripts/verification/prepare_inputs.sh"
INNER = ROOT / "scripts/verification/required_inner.sh"
LOCK = ROOT / "Seen.lock"


class LocalVerificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.prepare = PREPARE.read_text(encoding="utf-8")
        cls.inner = INNER.read_text(encoding="utf-8")
        cls.lock = LOCK.read_text(encoding="utf-8")

    def test_hosted_ci_is_absent_and_local_gate_is_bounded(self) -> None:
        self.assertFalse(WORKFLOW.exists())
        self.assertIn("TIMEOUT_SECS=1500", self.runner)

    def test_container_scope_is_immutable_and_bounded(self) -> None:
        self.assertIn(
            "silkeh/clang@sha256:a370fe4e8ecd284143bbfde1185bef4c1b6b72f45af4823812b9afe84cd1a14d",
            self.runner,
        )
        for required in (
            '--network none',
            '--gpus all',
            '--read-only',
            '--cap-drop ALL',
            '--security-opt no-new-privileges',
            '--memory "$memory_bytes"',
            '--memory-swap "$memory_bytes"',
            '--pids-limit "$TASKS_MAX"',
            '--ulimit stack=8388608:8388608',
            'dst=/workspace,readonly',
            'dst=/workspace/.seen',
            'src=/opt/cuda,dst=/opt/cuda,readonly',
            'dst=/tmp',
            'PYTHONPATH=/workspace/.seen/verification/python',
            'CUDAToolkit_ROOT=/opt/cuda',
            'NVCC_CCBIN=/usr/bin/clang++',
            'CUDAHOSTCXX=/usr/bin/clang++',
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.runner + self.prepare + self.inner)
        self.assertNotIn("pkexec", self.runner + self.prepare + self.inner)
        self.assertNotIn("/usr/local/bin/seen", self.runner + self.inner)

    def test_inputs_match_dependency_and_oracle_locks(self) -> None:
        identities = (
            "6b0b354c1433cd5850c7350424b5630d4204bd26",
            "00f61783fa2871e9c9d75675bf81ad2febb5dac7",
            "5e4cfce183f1d1679d647b9800a86abffd3a33db",
            "8cd66ce69e6f506959a172b5ecb169e198b74189e6dbc261704f0efec6554cd7",
            "dd544d342401a9066d9d76d6849e41972a60968d8ee841c678ba23cd6d82c34e",
            "3c7af4b74da3199d652cd545814731d7695123a67f54817b2ab925055b201f59",
            "cc7deff18b3319b36ed85ceb050e867b3fc36491e928f54e1908c8a260146be6",
            "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
            "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
            "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
            "11e06aa0af8c0f05104d56450d6093ee639e15f24ecf62d417329d06e522e017",
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "a370fe4e8ecd284143bbfde1185bef4c1b6b72f45af4823812b9afe84cd1a14d",
        )
        for identity in identities:
            self.assertIn(identity, self.prepare + self.inner + self.lock)

    def test_first_engine_artifact_gate_is_required(self) -> None:
        for required in (
            "tests/test_engine_artifact.py",
            "tests/qwn_034a_engine_artifact_test.seen --frozen",
            '"$OUTPUT_ROOT/qwn_034a_engine_artifact_test"',
        ):
            self.assertIn(required, self.inner)

    def test_complete_model_memory_gate_is_required(self) -> None:
        for required in (
            "tests/test_qwn_046a_q4_artifact.py",
            "tests/test_qwn_046a_full_cuda_contract.py",
            "tests/qwn_046a_full_memory_plan_test.seen --frozen",
            "tests/qwn_046a_full_cuda_frontend_test.seen --frozen",
            "qwn_046a_full_memory_plan_test_fast",
            '"$OUTPUT_ROOT/qwn_046a_full_memory_plan_test"',
        ):
            self.assertIn(required, self.inner)

        manifest = (ROOT / "Seen.toml").read_text(encoding="utf-8")
        self.assertIn("[native.dependencies]", manifest)
        self.assertIn("seen_cuda = { bundled = true }", manifest)

    def test_cuda_reference_contract_is_required(self) -> None:
        self.assertIn("tests/test_cuda_reference_primitives.py", self.inner)
        self.assertIn("tests/test_cuda_reference_utilities.py", self.inner)
        self.assertIn(
            "tests/qwn_040a_reference_primitives_test.seen --frozen", self.inner
        )
        self.assertIn("qwn_030b_sqw_reader_test_fast", self.inner)
        self.assertIn("qwn_040a_reference_primitives_test_fast", self.inner)
        self.assertIn("qwn_040a_reference_primitives_test", self.inner)
        self.assertIn(
            "tests/qwn_040b_reference_utilities_test.seen --frozen", self.inner
        )
        self.assertIn("qwn_040b_reference_utilities_test_fast", self.inner)
        self.assertIn("qwn_040b_reference_utilities_test", self.inner)
        self.assertIn("tests/test_cuda_gdn_recurrent_decode.py", self.inner)
        self.assertIn("tests/qwn_041b_gdn_decode_test.seen --frozen", self.inner)
        self.assertIn("qwn_041b_gdn_decode_test_fast", self.inner)
        self.assertIn("qwn_041b_gdn_decode_test", self.inner)
        self.assertIn("tests/test_cuda_gdn_recurrent_prefill.py", self.inner)
        self.assertIn("tests/qwn_041c_gdn_prefill_test.seen --frozen", self.inner)
        self.assertIn("qwn_041c_gdn_prefill_test_fast", self.inner)
        self.assertIn("qwn_041c_gdn_prefill_test", self.inner)
        self.assertIn("tests/test_cuda_gdn_gated_output.py", self.inner)
        self.assertIn("tests/qwn_041d_gdn_output_test.seen --frozen", self.inner)
        self.assertIn("qwn_041d_gdn_output_test_fast", self.inner)
        self.assertIn("qwn_041d_gdn_output_test", self.inner)
        self.assertIn("tests/test_cuda_attention_qk.py", self.inner)
        self.assertIn("tests/qwn_042a_attention_qk_test.seen --frozen", self.inner)
        self.assertIn("qwn_042a_attention_qk_test_fast", self.inner)
        self.assertIn("qwn_042a_attention_qk_test", self.inner)
        self.assertIn("tests/test_cuda_kv_cache.py", self.inner)
        self.assertIn("tests/qwn_042b_kv_cache_test.seen --frozen", self.inner)
        self.assertIn("qwn_042b_kv_cache_test_fast", self.inner)
        self.assertIn("qwn_042b_kv_cache_test", self.inner)
        self.assertIn("tests/test_cuda_attention_decode.py", self.inner)
        self.assertIn("tests/qwn_042c_attention_decode_test.seen --frozen", self.inner)
        self.assertIn("qwn_042c_attention_decode_test_fast", self.inner)
        self.assertIn("qwn_042c_attention_decode_test", self.inner)
        self.assertIn("tests/test_cuda_attention_prefill.py", self.inner)
        self.assertIn("tests/qwn_042d_attention_prefill_test.seen --frozen", self.inner)
        self.assertIn("qwn_042d_attention_prefill_test_fast", self.inner)
        self.assertIn("qwn_042d_attention_prefill_test", self.inner)
        self.assertIn("tests/test_cuda_attention_output_gate.py", self.inner)
        self.assertIn("tests/qwn_042e_attention_output_gate_test.seen --frozen", self.inner)
        self.assertIn("qwn_042e_attention_output_gate_test_fast", self.inner)
        self.assertIn("qwn_042e_attention_output_gate_test", self.inner)
        self.assertIn("tests/test_cuda_projection_descriptor.py", self.inner)
        self.assertIn(
            "tests/qwn_043a_projection_descriptor_test.seen --frozen", self.inner
        )
        self.assertIn("qwn_043a_projection_descriptor_test_fast", self.inner)
        self.assertIn("qwn_043a_projection_descriptor_test", self.inner)
        self.assertIn("tests/test_cuda_ffn_execution.py", self.inner)
        self.assertIn(
            "tests/qwn_043b_ffn_execution_test.seen --frozen", self.inner
        )
        self.assertIn("qwn_043b_ffn_execution_test_fast", self.inner)
        self.assertIn("qwn_043b_ffn_execution_test", self.inner)
        self.assertIn("qwn_043b_cpu_project", self.inner)
        self.assertIn("seen_cuda_link_stubs.c", self.inner)
        self.assertIn("seen_qwen_cuda_link_stubs.c", self.inner)
        self.assertIn("tests/test_cuda_lm_head_greedy.py", self.inner)
        self.assertIn("tests/qwn_044a_lm_head_test.seen --frozen", self.inner)
        self.assertIn("qwn_044a_lm_head_test_fast", self.inner)
        self.assertIn("qwn_044a_lm_head_test", self.inner)
        self.assertIn("qwn_044a_cpu_project", self.inner)
        self.assertIn("tests/test_qwen_sampler.py", self.inner)
        self.assertIn("tests/qwn_044b_sampler_test.seen --frozen", self.inner)
        self.assertIn("qwn_044b_sampler_test_fast", self.inner)
        self.assertIn("qwn_044b_sampler_test", self.inner)
        self.assertIn("clang -shared -fPIC -O2 -Wl,--no-undefined", self.inner)
        self.assertNotIn("nvcc", self.inner)

    def test_qwen_engine_ownership_contract_is_required(self) -> None:
        for required in (
            "tests/test_qwen_engine_ownership.py",
            "tests/qwn_045a_engine_ownership_test.seen --frozen",
            "qwn_045a_engine_ownership_test_fast",
            "qwn_045a_engine_ownership_test",
        ):
            self.assertIn(required, self.inner)

    def test_seen_release_provenance_is_exact_and_current(self) -> None:
        compiler_sha256 = (
            "dd544d342401a9066d9d76d6849e41972a60968d8ee841c678ba23cd6d82c34e"
        )
        archive_sha256 = (
            "8cd66ce69e6f506959a172b5ecb169e198b74189e6dbc261704f0efec6554cd7"
        )
        source_commit = "6b0b354c1433cd5850c7350424b5630d4204bd26"
        build_id = "c9865c92973d525338c018a2c6e84684a7abe8bb"

        for lock_entry in (
            '# release_tag = "v0.22.2"',
            f'# certified_commit = "{source_commit}"',
            f'# linux_x64_archive_sha256 = "{archive_sha256}"',
            f'# packaged_compiler_sha256 = "{compiler_sha256}"',
            f'# compiler_build_id = "{build_id}"',
            "compiler=0.22.2",
            "target=linux-x86_64",
            "cpu=x86-64",
        ):
            self.assertIn(lock_entry, self.lock)

        for prepare_entry in (
            "releases/download/v0.22.2/seen-0.22.2-linux-x64.tar.gz",
            archive_sha256,
            compiler_sha256,
            source_commit,
            build_id,
            "SEEN_CPU_BASELINE=\"x86-64\"",
            "verify-compiler-provenance.sh",
            "compiler-provenance.env",
            "command -v readelf",
            "compare_toolchain_trees",
            "write_tree_inventory",
            "ensure_local_directory",
            "LC_ALL=C find . -print0",
            "sha256sum -- \"$entry\"",
            "stat -c '%h' -- \"$entry\"",
        ):
            self.assertIn(prepare_entry, self.prepare)

        for inner_entry in (
            "seen-0.22.2-linux-x64",
            compiler_sha256,
            "Seen 0.22.2",
            "--target-cpu=x86-64",
        ):
            self.assertIn(inner_entry, self.inner)

        release_surfaces = self.prepare + self.inner + self.lock
        self.assertNotIn("5d868b0b", release_surfaces)
        self.assertNotIn("0.18.1", release_surfaces)
        self.assertNotIn("v0.18.1", release_surfaces)
        self.assertNotIn("seen-0.18.1-linux-x64", release_surfaces)

    def test_complete_toolchain_comparison_rejects_tree_mutations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwn-toolchain-tree-") as root_text:
            root = Path(root_text)
            expected = root / "expected"
            actual = root / "actual"
            strict = expected / "lib/seen/std/json/strict.seen"
            strict.parent.mkdir(parents=True)
            strict.write_text("exact published payload\n", encoding="utf-8")
            (expected / "bin").mkdir()
            compiler = expected / "bin/seen"
            compiler.write_bytes(b"compiler")
            compiler.chmod(0o755)
            shutil.copytree(expected, actual, copy_function=shutil.copy2)
            scratch = root / "scratch"
            scratch.mkdir()
            command = [
                "bash",
                str(PREPARE),
                "--compare-toolchain-trees",
                str(expected),
                str(actual),
                str(scratch),
            ]
            matching = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            self.assertEqual(matching.returncode, 0, matching.stderr)

            def assert_rejected(label: str) -> None:
                rejected = subprocess.run(
                    command, cwd=ROOT, text=True, capture_output=True, check=False
                )
                self.assertNotEqual(rejected.returncode, 0, label)
                self.assertIn("toolchain tree comparison failed", rejected.stderr)

            actual_strict = actual / "lib/seen/std/json/strict.seen"
            strict_mode = strict.stat().st_mode & 0o777
            actual_strict.write_text(
                "tampered stdlib payload\n", encoding="utf-8"
            )
            assert_rejected("changed regular-file content was accepted")
            actual_strict.write_text("exact published payload\n", encoding="utf-8")

            actual_strict.unlink()
            assert_rejected("missing payload entry was accepted")
            actual_strict.write_text("exact published payload\n", encoding="utf-8")
            actual_strict.chmod(strict_mode)

            extra = actual / "lib/seen/std/json/extra.seen"
            extra.write_text("extra\n", encoding="utf-8")
            assert_rejected("extra payload entry was accepted")
            extra.unlink()

            external = root / "external-strict.seen"
            external.write_text("exact published payload\n", encoding="utf-8")
            actual_strict.unlink()
            actual_strict.symlink_to(external)
            assert_rejected("symlinked payload entry was accepted")
            actual_strict.unlink()
            actual_strict.write_text("exact published payload\n", encoding="utf-8")
            actual_strict.chmod(strict_mode)

            actual_strict.chmod(0o600)
            assert_rejected("changed regular-file mode was accepted")
            actual_strict.chmod(strict_mode)

            actual_lib = actual / "lib"
            expected_lib_mode = (expected / "lib").stat().st_mode & 0o777
            actual_lib.chmod(0o700)
            assert_rejected("changed directory mode was accepted")
            actual_lib.chmod(expected_lib_mode)

            expected_root_mode = expected.stat().st_mode & 0o777
            actual.chmod(0o700)
            assert_rejected("changed toolchain-root mode was accepted")
            actual.chmod(expected_root_mode)

            actual_strict.unlink()
            os.link(external, actual_strict)
            assert_rejected("hardlinked payload entry was accepted")

    def test_inner_gate_reads_limits_and_runs_exact_oracle(self) -> None:
        for required in (
            "memory.max",
            "memory.swap.max",
            "memory.oom.group",
            "pids.max",
            "memory.peak",
            "memory.events",
            "pids.peak",
            "pids.events",
            "SEEN_JOBS",
            "SEEN_OPT_JOBS",
            "MemAvailable",
            "MEMORY_CEILING_BYTES=7516192768",
            "SEEN_EXPECTED_MEMORY_BYTES",
            "--release --lto=thin --target-cpu=x86-64 --no-cache",
            "--jobs 1 --opt-jobs 1 --no-fork --frozen",
            "qwn_022b_tokenizer_test",
            "qwn_022c_chat_template_test",
            "qwn_022d_sampling_test",
            "qwn_023a_hybrid_mini_contract_test",
            "qwn_023b_hybrid_mini_assets_test",
            "qwn_024a_cpu_reference_test",
            "qwn_024b_cpu_attention_test",
            "qwn_024c_cpu_gdn_test",
            "qwn_024d_cpu_head_test",
            "qwn_024e_cpu_engine_test",
            "qwn_025a_operator_layer_oracle_test",
            "qwn_025b_full_model_oracle_test",
            "qwn_030a_sqw_contract_test",
            "qwn_030b_sqw_reader_test",
            "qwn_030c_sqw_writer_test",
            "qwn_031a_reference_codec_test",
            "qwn_031b_q8_reference_codec_test",
            "qwn_031c_q4_reference_codec_test",
            "qwn_032a_conversion_plan_test",
            "qwn_032b_shard_stream_test",
            "qwn_032c_conversion_evidence_test",
            "qwn_033a_calibration_test",
            "qwn_033b_sensitivity_test",
            "qwn_033c_policy_test",
            "qwn_041b_gdn_decode_test",
            "qwn_041c_gdn_prefill_test",
            "qwn_041d_gdn_output_test",
            "qwn_042a_attention_qk_test",
            "test_sampling_profiles.py",
            "test_hybrid_mini_contract.py",
            "test_hybrid_mini_assets.py",
            "test_hybrid_mini_oracle.py",
            "test_cpu_attention_oracle.py",
            "test_cpu_gdn_oracle.py",
            "test_cpu_head_oracle.py",
            "test_cpu_engine_oracle.py",
            "test_official_operator_layer_oracles.py",
            "test_official_full_model_oracles.py",
            "test_sqw_contract.py",
            "test_sqw_reader.py",
            "test_sqw_writer.py",
            "test_q8_reference_codec.py",
            "test_q4_reference_codec.py",
            "test_conversion_plan.py",
            "test_shard_stream.py",
            "test_conversion_evidence.py",
            "test_calibration_lock.py",
            "test_sensitivity_statistics.py",
            "test_quantization_policy_resolver.py",
            "test_qwen_tokenizer_oracles.py",
            "seen-pkg",
            "outside_objects_before",
            "outside_objects_after",
        ):
            self.assertIn(required, self.runner + self.inner)
        self.assertRegex(self.inner, re.compile(r'\[ "\$\(ulimit -s\)" = "8192" \]'))
        self.assertIn('[ "$memory_max" -le 7516192768 ]', self.inner)


if __name__ == "__main__":
    unittest.main()
