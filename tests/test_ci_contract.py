#!/usr/bin/env python3
"""Static fail-closed contracts for the standalone Seen Qwen CI workflow."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
RUNNER = ROOT / "scripts/ci/run_required.sh"
PREPARE = ROOT / "scripts/ci/prepare_inputs.sh"
INNER = ROOT / "scripts/ci/required_inner.sh"
LOCK = ROOT / "Seen.lock"


class CiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.prepare = PREPARE.read_text(encoding="utf-8")
        cls.inner = INNER.read_text(encoding="utf-8")
        cls.lock = LOCK.read_text(encoding="utf-8")

    def test_workflow_identity_and_triggers_are_pinned(self) -> None:
        self.assertIn("push:\n    branches: [main]", self.workflow)
        self.assertIn("pull_request:", self.workflow)
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn("runs-on: ubuntu-24.04", self.workflow)
        self.assertIn("timeout-minutes: 30", self.workflow)
        self.assertIn(
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            self.workflow,
        )
        self.assertNotRegex(self.workflow, r"uses: [^\n]+@(v|main|master)")

    def test_container_scope_is_immutable_and_bounded(self) -> None:
        self.assertIn(
            "silkeh/clang@sha256:a370fe4e8ecd284143bbfde1185bef4c1b6b72f45af4823812b9afe84cd1a14d",
            self.runner,
        )
        for required in (
            '--network none',
            '--read-only',
            '--cap-drop ALL',
            '--security-opt no-new-privileges',
            '--memory "$memory_bytes"',
            '--memory-swap "$memory_bytes"',
            '--pids-limit "$TASKS_MAX"',
            '--ulimit stack=8388608:8388608',
            'dst=/workspace,readonly',
            'dst=/workspace/.seen',
            'dst=/tmp',
        ):
            self.assertIn(required, self.runner)
        self.assertNotIn("sudo", self.workflow + self.runner + self.prepare + self.inner)
        self.assertNotIn("pkexec", self.workflow + self.runner + self.prepare + self.inner)
        self.assertNotIn("/usr/local/bin/seen", self.workflow + self.runner + self.inner)

    def test_inputs_match_dependency_and_oracle_locks(self) -> None:
        identities = (
            "2e7c39a742455ba211a7b5e253e78f0a5f68309e",
            "4f413423857dcdd5f162e7df2b92e2a50614df2e",
            "25ff44a7257de770afde96e8123eed0b6296caa8",
            "17e5dce78b6129bc0dfebe92a8c3e0d69fb2e1b5b8bd9c7655bd9c39195a9b42",
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c",
            "bfed49cea60c983751c26cef81b21e3374360f3a43de677e8134c14a3c30a158",
            "69441bbf20755f0bbf12a4241fffadf3ad20df6f9d1155a6b0b5ab92992e9e2c",
            "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
            "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
            "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
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

    def test_cuda_reference_contract_is_required(self) -> None:
        self.assertIn("tests/test_cuda_reference_primitives.py", self.inner)
        self.assertIn(
            "tests/qwn_040a_reference_primitives_test.seen --frozen", self.inner
        )
        self.assertIn("qwn_030b_sqw_reader_test_fast", self.inner)
        self.assertIn("qwn_040a_reference_primitives_test_fast", self.inner)
        self.assertIn("qwn_040a_reference_primitives_test", self.inner)

    def test_seen_release_provenance_is_exact_and_current(self) -> None:
        compiler_sha256 = (
            "79293f057890f0edf133910d5f2055613006829f815eb6504197c233a0a6c57c"
        )
        archive_sha256 = (
            "17e5dce78b6129bc0dfebe92a8c3e0d69fb2e1b5b8bd9c7655bd9c39195a9b42"
        )
        source_commit = "2e7c39a742455ba211a7b5e253e78f0a5f68309e"
        build_id = "e9fb8f24a59299db5fe63fed053ebff6d6c2a262"

        for lock_entry in (
            '# release_tag = "v0.20.4"',
            f'# certified_commit = "{source_commit}"',
            f'# linux_x64_archive_sha256 = "{archive_sha256}"',
            f'# packaged_compiler_sha256 = "{compiler_sha256}"',
            f'# compiler_build_id = "{build_id}"',
            "compiler=0.20.4",
            "target=linux-x86_64",
            "cpu=x86-64",
        ):
            self.assertIn(lock_entry, self.lock)

        for prepare_entry in (
            "releases/download/v0.20.4/seen-0.20.4-linux-x64.tar.gz",
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
            "seen-0.20.4-linux-x64",
            compiler_sha256,
            "Seen 0.20.4",
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
