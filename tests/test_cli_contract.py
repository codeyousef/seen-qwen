"""Static independent contract checks for FEL-1457 / QWN-047A."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class QwenCliContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "src/cli/command.seen").read_text(encoding="utf-8")
        cls.main = (ROOT / "src/cli/main.seen").read_text(encoding="utf-8")
        cls.contract = (ROOT / "docs/qwen-cli-contract.md").read_text(
            encoding="utf-8"
        )
        cls.manifest = (ROOT / "Seen.toml").read_text(encoding="utf-8")

    def test_manifest_and_command_surface_are_explicit(self) -> None:
        self.assertIn('"src/cli/command.seen"', self.manifest)
        for command in ("inspect", "convert", "verify", "run", "bench"):
            self.assertIn(f'command == "{command}"', self.source)
            self.assertIn(f"`{command}`", self.contract)
        self.assertIn('arguments[1] == "serve"', self.source)
        self.assertIn("explicitly unsupported", self.contract)

    def test_parser_is_bounded_and_fail_closed(self) -> None:
        for required in (
            "QWEN_CLI_MAX_ARGUMENTS: Int = 64",
            "QWEN_CLI_MAX_ARGUMENT_BYTES: Int = 4096",
            "QWEN_CLI_MAX_TOTAL_BYTES: Int = 16384",
            "QWEN_CLI_MAX_JOBS: Int = 64",
            "QWEN_CLI_MAX_NEW_TOKENS: UInt64 = 262144",
            '"qwen.cli.unknown-command"',
            '"qwen.cli.unknown-option"',
            '"qwen.cli.duplicate-option"',
            '"qwen.cli.integer-overflow"',
            '"qwen.cli.prompt-source"',
        ):
            self.assertIn(required, self.source)
        self.assertNotIn("fallback =", self.source.lower())

    def test_ownership_and_diagnostics_are_explicit(self) -> None:
        for required in (
            "@move\npub class QwenCliInvocation",
            "fun qwenCliSetString(borrow invocation: QwenCliInvocation",
            "fun qwenCliValidateInvocation(borrow invocation: QwenCliInvocation)",
            "seen_string_clone_owned",
            "seen_string_release_owned",
            "invocation.close(); invocation.free()",
            "jsonAppendEscapedString",
            "jsonFinishOwned",
            '"seen-qwen-diagnostic-v1"',
        ):
            self.assertIn(required, self.source)

    def test_exit_codes_and_terminal_dispatch_are_stable(self) -> None:
        for name, value in (
            ("QWEN_EXIT_SUCCESS", 0),
            ("QWEN_EXIT_CLI_CONFIG", 2),
            ("QWEN_EXIT_ARTIFACT_VALIDATION", 3),
            ("QWEN_EXIT_UNSUPPORTED", 4),
            ("QWEN_EXIT_HOST_ALLOCATION", 5),
            ("QWEN_EXIT_DEVICE_ALLOCATION", 6),
            ("QWEN_EXIT_CUDA_RUNTIME", 7),
            ("QWEN_EXIT_CORRECTNESS", 8),
            ("QWEN_EXIT_BENCHMARK", 9),
            ("QWEN_EXIT_CANCELLED", 10),
            ("QWEN_EXIT_INTERNAL", 11),
        ):
            self.assertIn(f"pub let {name}: Int = {value}", self.source)
        self.assertIn("qwen.cli.command-not-installed", self.main)
        self.assertIn("return QWEN_EXIT_UNSUPPORTED", self.main)
        self.assertNotIn("CudaDevice", self.main)


if __name__ == "__main__":
    unittest.main()
