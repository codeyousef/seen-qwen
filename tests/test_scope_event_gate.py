from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/oracle/check_scope_events.sh"
OUTER = ROOT / "scripts/oracle/run_bounded.sh"
INNER = ROOT / "scripts/oracle/run_bounded_inner.sh"
REQUIRED_OUTER = ROOT / "scripts/verification/run_required.sh"
REQUIRED_INNER = ROOT / "scripts/verification/required_inner.sh"


def memory_events(*, oom=0, oom_kill=0, oom_group_kill=0, maximum=0):
    return (
        f"low 0\nhigh 0\nmax {maximum}\noom {oom}\n"
        f"oom_kill {oom_kill}\noom_group_kill {oom_group_kill}\n"
    )


def pids_events(maximum=0):
    return f"max {maximum}\n"


class ScopeEventGateTests(unittest.TestCase):
    def run_gate(self, status, before_memory, after_memory,
                 before_pids=None, after_pids=None):
        return subprocess.run(
            [
                str(GATE), str(status), before_memory, after_memory,
                before_pids if before_pids is not None else pids_events(),
                after_pids if after_pids is not None else pids_events(),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_clean_scope_preserves_success(self):
        result = self.run_gate(
            0, memory_events(maximum=2), memory_events(maximum=9000)
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_each_memory_containment_event_fails_successful_child(self):
        for name in ("oom", "oom_kill", "oom_group_kill"):
            with self.subTest(name=name):
                before = {"oom": 3, "oom_kill": 4, "oom_group_kill": 5}
                after = before.copy()
                after[name] += 1
                result = self.run_gate(
                    0, memory_events(**before), memory_events(**after)
                )
                self.assertEqual(result.returncode, 125)
                self.assertIn(f"memory.events {name} increased by 1", result.stderr)

    def test_task_limit_event_fails_successful_child(self):
        result = self.run_gate(
            0, memory_events(), memory_events(), pids_events(8), pids_events(9)
        )
        self.assertEqual(result.returncode, 125)
        self.assertIn("pids.events max increased by 1", result.stderr)

    def test_nonzero_child_status_is_preserved_with_diagnostic(self):
        result = self.run_gate(
            42, memory_events(oom=1), memory_events(oom=2)
        )
        self.assertEqual(result.returncode, 42)
        self.assertIn("memory.events oom increased by 1", result.stderr)

    def test_nonzero_child_status_is_preserved_without_event(self):
        result = self.run_gate(73, memory_events(), memory_events())
        self.assertEqual(result.returncode, 73)

    def test_missing_duplicate_or_malformed_counters_fail_closed(self):
        bad_snapshots = (
            "oom 0\noom_kill 0\n",
            "oom 0\noom 1\noom_kill 0\noom_group_kill 0\n",
            "oom nope\noom_kill 0\noom_group_kill 0\n",
        )
        for snapshot in bad_snapshots:
            with self.subTest(snapshot=snapshot):
                result = self.run_gate(0, snapshot, memory_events())
                self.assertEqual(result.returncode, 126)

    def test_counter_reset_fails_closed(self):
        result = self.run_gate(
            0, memory_events(oom=2), memory_events(oom=1)
        )
        self.assertEqual(result.returncode, 126)
        self.assertIn("oom counter decreased", result.stderr)

    def test_wrapper_snapshots_and_gates_authoritative_events(self):
        outer = OUTER.read_text(encoding="utf-8")
        inner = INNER.read_text(encoding="utf-8")
        self.assertIn('EVENT_GATE="$ROOT_DIR/scripts/oracle/check_scope_events.sh"', outer)
        self.assertIn('memory_events_before=$(read_metric memory.events)', inner)
        self.assertIn('pids_events_before=$(read_metric pids.events)', inner)
        self.assertIn('memory_events_after=$(read_metric memory.events)', inner)
        self.assertIn('pids_events_after=$(read_metric pids.events)', inner)
        self.assertIn('"$EVENT_GATE" "$status"', inner)

    def test_required_gate_uses_the_same_event_contract(self):
        outer = REQUIRED_OUTER.read_text(encoding="utf-8")
        inner = REQUIRED_INNER.read_text(encoding="utf-8")
        expected = 'EVENT_GATE="$ROOT_DIR/scripts/oracle/check_scope_events.sh"'
        self.assertIn(expected, outer)
        self.assertIn(expected, inner)
        self.assertIn('memory_events_before=$(read_cgroup memory.events)', inner)
        self.assertIn('pids_events_before=$(read_cgroup pids.events)', inner)
        self.assertIn('memory_events_after=$(read_cgroup memory.events)', inner)
        self.assertIn('pids_events_after=$(read_cgroup pids.events)', inner)
        self.assertIn('"$EVENT_GATE" "$status"', inner)


if __name__ == "__main__":
    unittest.main()
