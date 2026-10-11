from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "scripts/oracle/run_qwn_047c_convert.py"
BUILDER = ROOT / "scripts/oracle/build_qwn_034a_bf16_artifact.py"


def load_bridge():
    sys.path.insert(0, str(BRIDGE.parent))
    spec = importlib.util.spec_from_file_location("qwn_047c_bridge", BRIDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConvertVerifyContractTests(unittest.TestCase):
    def test_native_dispatch_is_exact_request_and_fail_closed(self) -> None:
        source = (ROOT / "src/cli/convert_verify.seen").read_text()
        main = (ROOT / "src/cli/main.seen").read_text()
        manifest = (ROOT / "Seen.toml").read_text()
        self.assertIn('"src/cli/convert_verify.seen"', manifest)
        self.assertIn("runProgramRequest(", source)
        self.assertNotIn("runCommand(", source)
        self.assertIn('"qwen.convert.resume"', source)
        self.assertIn('"qwen.verify.suite-not-installed"', source)
        self.assertIn("QWEN_CLI_CONVERT", main)
        self.assertIn("QWEN_CLI_VERIFY", main)

    def test_builder_attests_the_locked_seen_release(self) -> None:
        source = BUILDER.read_text()
        self.assertIn('SEEN_VERSION = "0.22.9"', source)
        self.assertIn(
            'COMPILER_SHA256 = "58d7181f8b5adf1bb823ee5dcfc0868f9962b302af4dfaabf6a9e806aea16db2"',
            source,
        )
        self.assertIn(
            'SEEN_COMMIT = "d39a22295e469a47cfa0b76df49a659389ee78a2"',
            source,
        )

    def test_exact_request_derives_only_sibling_locked_inputs(self) -> None:
        bridge = load_bridge()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {
                "schema": "seen-qwen-convert-request-v1",
                "model_lock": str(root / "model-lock.json"),
                "output_root": str(root / "out"),
                "quant_profile": "bf16-bringup-v1",
                "quantization_config": str(root / "quantization.toml"),
                "calibration_lock": "",
                "jobs": 1,
                "max_host_memory": 0,
                "resume": False,
                "verify_only": True,
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request))
            loaded = bridge.load_request(request_path)
            with mock.patch.object(bridge.builder, "build", return_value=root / "out") as build:
                result = bridge.execute(loaded)
            self.assertEqual(result, root / "out")
            arguments = build.call_args.args[0]
            self.assertEqual(arguments.source_root, root)
            self.assertEqual(arguments.asset_root, root)
            self.assertEqual(arguments.index, root / "model.safetensors.index.json")
            self.assertTrue(arguments.verify_only)

    def test_request_rejects_parallel_resume_and_hidden_profiles(self) -> None:
        bridge = load_bridge()
        base = {
            "schema": "seen-qwen-convert-request-v1",
            "model_lock": "model-lock.json",
            "output_root": ".seen/out",
            "quant_profile": "bf16-bringup-v1",
            "quantization_config": "configs/quantization.toml",
            "calibration_lock": "",
            "jobs": 1,
            "max_host_memory": 0,
            "resume": False,
            "verify_only": True,
        }
        for change in (
            {"jobs": 2},
            {"resume": True},
            {"quant_profile": "q4-hidden-policy"},
            {"max_host_memory": 64 * 1024**3 + 1},
        ):
            request = dict(base)
            request.update(change)
            with self.assertRaises(ValueError):
                bridge.execute(request)


if __name__ == "__main__":
    unittest.main()
