import os
from pathlib import Path
import stat
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts/cuda/require_idle_gpu.sh"
RUNNERS = sorted((ROOT / "scripts/cuda").glob("run_qwn_*.sh"))


class GpuIdlePreflightTests(unittest.TestCase):
    def run_preflight(self, fake_body: str, visible: str | None = None):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "nvidia-smi"
            fake.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + fake_body)
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            environment = os.environ.copy()
            environment["NVIDIA_SMI_BIN"] = str(fake)
            if visible is None:
                environment.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                environment["CUDA_VISIBLE_DEVICES"] = visible
            return subprocess.run(
                [str(PREFLIGHT)],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_idle_single_gpu_passes(self):
        result = self.run_preflight(textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf '0, GPU-aaaa\\n' ;;
                --id=*) exit 0 ;;
                *) exit 91 ;;
            esac
        """))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "GPU-aaaa\n")
        self.assertIn("GPU 0 (GPU-aaaa) is idle", result.stderr)

    def test_busy_gpu_fails_without_process_control(self):
        result = self.run_preflight(textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf '0, GPU-aaaa\\n' ;;
                --id=*) printf '4242, model-server\\n' ;;
                *) exit 91 ;;
            esac
        """))
        self.assertEqual(result.returncode, 75)
        self.assertIn("busy with compute PID 4242", result.stderr)
        source = PREFLIGHT.read_text()
        self.assertNotIn("kill ", source)
        self.assertNotIn("pkill", source)

    def test_multiple_visible_devices_require_one_selector(self):
        body = textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf '0, GPU-aaaa\\n1, GPU-bbbb\\n' ;;
                --id=*) exit 0 ;;
                *) exit 91 ;;
            esac
        """)
        rejected = self.run_preflight(body)
        self.assertEqual(rejected.returncode, 75)
        self.assertIn("multiple GPUs are visible", rejected.stderr)
        selected = self.run_preflight(body, "1")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(selected.stdout, "GPU-bbbb\n")
        self.assertIn("GPU 1 (GPU-bbbb) is idle", selected.stderr)

    def test_empty_or_ambiguous_visibility_fails(self):
        body = textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf '0, GPU-aaaa\\n1, GPU-bbbb\\n' ;;
                --id=*) exit 0 ;;
                *) exit 91 ;;
            esac
        """)
        for visible in ("", "0,1", "9"):
            with self.subTest(visible=visible):
                self.assertEqual(self.run_preflight(body, visible).returncode, 75)

    def test_inventory_query_failure_fails_closed(self):
        result = self.run_preflight("printf 'driver unavailable\\n' >&2\nexit 17\n")
        self.assertEqual(result.returncode, 75)
        self.assertIn("cannot enumerate GPUs", result.stderr)

    def test_compute_query_failure_fails_closed(self):
        result = self.run_preflight(textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf '0, GPU-aaaa\\n' ;;
                --id=*) printf 'permission denied\\n' >&2; exit 19 ;;
                *) exit 91 ;;
            esac
        """))
        self.assertEqual(result.returncode, 75)
        self.assertIn("cannot prove GPU GPU-aaaa is idle", result.stderr)

    def test_malformed_output_fails_closed(self):
        malformed_inventory = self.run_preflight(textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf 'zero, not-a-uuid\\n' ;;
                *) exit 91 ;;
            esac
        """))
        self.assertEqual(malformed_inventory.returncode, 75)
        malformed_process = self.run_preflight(textwrap.dedent("""\
            case "$1" in
                --query-gpu=*) printf '0, GPU-aaaa\\n' ;;
                --id=*) printf 'not-a-pid, process\\n' ;;
                *) exit 91 ;;
            esac
        """))
        self.assertEqual(malformed_process.returncode, 75)

    def test_every_hardware_runner_uses_preflight_before_identity_query(self):
        guarded = []
        for runner in RUNNERS:
            source = runner.read_text()
            if "nvidia-smi" not in source:
                continue
            guarded.append(runner.name)
            guard = 'QWEN_GPU_UUID=$("$ROOT_DIR/scripts/cuda/require_idle_gpu.sh")'
            self.assertIn(guard, source, runner.name)
            self.assertLess(source.index(guard), source.index("nvidia-smi"), runner.name)
            self.assertIn('export CUDA_VISIBLE_DEVICES="$QWEN_GPU_UUID"', source)
            self.assertIn('nvidia-smi --id="$QWEN_GPU_UUID"', source)
        self.assertEqual(len(guarded), 18)


if __name__ == "__main__":
    unittest.main()
