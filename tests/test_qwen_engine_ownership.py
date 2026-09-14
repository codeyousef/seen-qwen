"""Static independent contract checks for QWN-045A ownership."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QwenEngineOwnershipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "src/runtime/engine.seen").read_text()
        cls.contract = (ROOT / "docs/qwen-engine-session-plan-contract.md").read_text().lower()

    def test_plan_is_numeric_bounded_and_complete(self) -> None:
        for value in (
            "QWEN_PLAN_EMBEDDING", "QWEN_PLAN_GDN_PREFILL",
            "QWEN_PLAN_GDN_DECODE", "QWEN_PLAN_ATTENTION_PREFILL",
            "QWEN_PLAN_ATTENTION_DECODE", "QWEN_PLAN_MLP",
            "QWEN_PLAN_FINAL_NORM", "QWEN_PLAN_LM_HEAD",
            "QWEN_PLAN_SAMPLE", "QWEN_PLAN_MTP",
            "QWEN_PLAN_MAX_STEPS", "validatePhasePlan",
        ):
            self.assertIn(value, self.source)
        self.assertNotIn("dispatch by string", self.source.lower().replace("never dispatch by string", ""))

    def test_owners_are_move_only_and_cleanup_is_explicit(self) -> None:
        for declaration in (
            "@move\npub class QwenExecutionPlan",
            "@move\npub class QwenSession",
            "@move\npub class QwenEngine",
            "engine cannot close while sessions remain live",
            "fun releaseSession(borrow session: QwenSession)",
            "session.ownerToken != this.ownerToken",
            "this.sampler.close()",
            "this.plan.close()",
        ):
            self.assertIn(declaration, self.source)

    def test_fallback_and_late_completion_fail_closed(self) -> None:
        for value in (
            "QWEN_FALLBACK_PROHIBITED",
            "QWEN_FALLBACK_EXPLICIT_CPU_REFERENCE",
            "semanticEquivalent", "explicitlyApproved",
            "fallbackCount", "lastFallbackReason",
            "qwen.session.fallback-prohibited",
            "qwen.session.cancelled", "qwen.session.deadline",
        ):
            self.assertIn(value, self.source)
        for phrase in (
            "no string/reflection dispatch", "retains no borrowed engine",
            "error never changes", "fallback = prohibited",
            "late, cancelled", "does not add a second cuda resource stack",
        ):
            self.assertIn(phrase, self.contract)


if __name__ == "__main__":
    unittest.main()
