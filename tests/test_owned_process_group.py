from pathlib import Path
import os
import subprocess
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/oracle/run_owned_process_group.sh"
OUTER = ROOT / "scripts/oracle/run_bounded.sh"
INNER = ROOT / "scripts/oracle/run_bounded_inner.sh"
BOUNDED = ROOT / "scripts/oracle/run_bounded.sh"


class OwnedProcessGroupTests(unittest.TestCase):
    def run_owned(self, timeout_seconds, *command):
        return subprocess.run(
            [str(RUNNER), str(timeout_seconds), *command],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=8,
        )

    def test_clean_command_preserves_output_and_success(self):
        result = self.run_owned(5, "bash", "-lc", "printf clean-output")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "clean-output")

    def test_command_failure_is_preserved(self):
        result = self.run_owned(5, "bash", "-lc", "exit 37")
        self.assertEqual(result.returncode, 37, result.stderr)

    def test_cooperative_background_descendant_is_cleaned_and_rejected(self):
        started = time.monotonic()
        result = self.run_owned(5, "bash", "-lc", "sleep 30 & exit 0")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertLess(time.monotonic() - started, 4)
        self.assertIn("owned descendants remained", result.stderr)

    def test_term_ignoring_descendant_is_forced_down_and_rejected(self):
        command = "(trap '' TERM; while :; do sleep 1; done) & exit 0"
        started = time.monotonic()
        result = self.run_owned(5, "bash", "-lc", command)
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn("owned descendants remained", result.stderr)

    def test_timeout_is_bounded_and_nonzero(self):
        started = time.monotonic()
        result = self.run_owned(
            1, "bash", "-lc", "trap '' TERM; while :; do sleep 1; done"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 6)

    def test_unrelated_process_group_is_untouched(self):
        unrelated = subprocess.Popen(["sleep", "1"])
        try:
            result = self.run_owned(5, "bash", "-lc", "sleep 30 & exit 0")
            self.assertEqual(result.returncode, 125, result.stderr)
            self.assertIsNone(unrelated.poll())
            unrelated.wait(timeout=3)
        finally:
            if unrelated.poll() is None:
                unrelated.wait(timeout=3)

    def test_cgroup_cleanup_catches_delayed_session_escape(self):
        environment = os.environ.copy()
        environment["QWN_TASKS_MAX"] = "12"
        command = "(trap '' TERM; sleep 0.2; exec setsid sleep 30) & exit 0"
        result = subprocess.run(
            [str(BOUNDED), "10", "bash", "-lc", command],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=8,
        )
        self.assertEqual(result.returncode, 125, result.stdout)
        self.assertIn("owned cgroup descendant(s) escaped", result.stdout)
        self.assertIn("oracle-inner: pids.current\n2", result.stdout)
        self.assertIn("oracle-inner: pids.events\nmax 0", result.stdout)

    def test_bounded_wrapper_requires_owned_group_and_pid_baseline(self):
        outer = OUTER.read_text(encoding="utf-8")
        inner = INNER.read_text(encoding="utf-8")
        expected = 'PROCESS_GROUP_RUNNER="$ROOT_DIR/scripts/oracle/run_owned_process_group.sh"'
        self.assertIn(expected, outer)
        self.assertIn(expected, inner)
        self.assertIn('pids_current_before=$(read_metric pids.current)', inner)
        self.assertIn('pids_current_after=$(read_metric pids.current)', inner)
        self.assertIn('snapshot_baseline_pids', inner)
        self.assertIn('cleanup_owned_extras TERM', inner)
        self.assertIn('cleanup_owned_extras KILL', inner)
        self.assertIn('actual_path" = "$cgroup_path', inner)
        self.assertIn('pids.current baseline/final=', inner)
        self.assertNotIn("timeout --foreground", inner)


if __name__ == "__main__":
    unittest.main()
