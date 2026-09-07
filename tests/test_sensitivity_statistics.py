import json
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_033b_sensitivity_oracle.json"
SCHEMA = ROOT / "schemas/qwen-sensitivity-oracle.schema.json"
SOURCE = ROOT / "src/quant/sensitivity.seen"


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def load_strict(path: pathlib.Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )


def activation(values):
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "maximum_absolute": max(abs(value) for value in values),
        "mean": sum(values) / len(values),
        "mean_absolute": sum(abs(value) for value in values) / len(values),
        "root_mean_square": math.sqrt(
            sum(value * value for value in values) / len(values)
        ),
    }


def comparison(reference, candidate):
    errors = [actual - expected for expected, actual in zip(reference, candidate)]
    squared_error = sum(value * value for value in errors)
    reference_squared = sum(value * value for value in reference)
    candidate_squared = sum(value * value for value in candidate)
    cosine_defined = reference_squared > 0.0 and candidate_squared > 0.0
    relative_defined = reference_squared > 0.0
    return {
        "count": len(reference),
        "maximum_absolute_error": max(abs(value) for value in errors),
        "mean_absolute_error": sum(abs(value) for value in errors) / len(errors),
        "root_mean_square_error": math.sqrt(squared_error / len(errors)),
        "relative_l2_error": (
            math.sqrt(squared_error / reference_squared) if relative_defined else 0.0
        ),
        "relative_l2_defined": relative_defined,
        "cosine_similarity": (
            sum(a * b for a, b in zip(reference, candidate))
            / math.sqrt(reference_squared * candidate_squared)
            if cosine_defined
            else 0.0
        ),
        "cosine_defined": cosine_defined,
        "exact": max(abs(value) for value in errors) == 0.0,
    }


class SensitivityStatisticsContractTest(unittest.TestCase):
    def setUp(self):
        self.oracle = load_strict(ORACLE)
        self.schema = load_strict(SCHEMA)

    def assert_metrics(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            if isinstance(value, bool) or isinstance(value, int):
                self.assertEqual(actual[key], value)
            else:
                self.assertTrue(math.isfinite(value))
                self.assertAlmostEqual(actual[key], value, places=14)

    def test_closed_identity_and_ordering(self):
        self.assertEqual(
            set(self.oracle),
            {
                "schema", "version", "ordering", "model_id", "model_revision",
                "calibration_lock_sha256", "element_limit",
                "activation_records", "comparison_records",
            },
        )
        self.assertEqual(self.oracle["schema"], "seen-qwen-sensitivity-oracle-v1")
        self.assertEqual(self.oracle["version"], 1)
        self.assertEqual(self.oracle["ordering"], "lexicographic-record-id")
        self.assertEqual(self.oracle["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(len(self.oracle["calibration_lock_sha256"]), 64)
        for records in (
            self.oracle["activation_records"], self.oracle["comparison_records"]
        ):
            ids = [record["id"] for record in records]
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(len(ids), len(set(ids)))

    def test_activation_oracles(self):
        for record in self.oracle["activation_records"]:
            self.assertLessEqual(len(record["values"]), self.oracle["element_limit"])
            self.assert_metrics(activation(record["values"]), record["expected"])

    def test_comparison_oracles(self):
        for record in self.oracle["comparison_records"]:
            self.assertIn(record["kind"], {"tensor-quantization", "layer-output"})
            self.assertEqual(len(record["reference"]), len(record["candidate"]))
            self.assertLessEqual(len(record["reference"]), self.oracle["element_limit"])
            self.assert_metrics(
                comparison(record["reference"], record["candidate"]),
                record["expected"],
            )

    def test_schema_is_closed_and_bounded(self):
        self.assertFalse(self.schema["additionalProperties"])
        self.assertEqual(self.schema["properties"]["schema"]["const"],
                         "seen-qwen-sensitivity-oracle-v1")
        for name in ("activationRecord", "activationExpected",
                     "comparisonRecord", "comparisonExpected"):
            self.assertFalse(self.schema["$defs"][name]["additionalProperties"])
        self.assertEqual(
            self.schema["properties"]["element_limit"]["maximum"], 1048576
        )

    def test_native_contract_is_fail_closed(self):
        source = SOURCE.read_text(encoding="utf-8")
        for diagnostic in (
            "qwen.statistics.input", "qwen.statistics.empty",
            "qwen.statistics.limit", "qwen.statistics.geometry",
            "qwen.statistics.nonfinite", "qwen.statistics.range",
        ):
            self.assertIn(diagnostic, source)
        self.assertIn("relativeL2Defined", source)
        self.assertIn("cosineDefined", source)
        self.assertIn("retryable: false", source)
        self.assertNotIn("parallel", source.replace("no parallel", ""))

    def test_hostile_duplicate_and_nonfinite_json_rejected(self):
        text = ORACLE.read_text(encoding="utf-8")
        duplicate = text.replace('"version": 1,', '"version": 1, "version": 2,', 1)
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            json.loads(duplicate, object_pairs_hook=lambda items: self._pairs(items),
                       parse_constant=reject_constant)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            json.loads('{"value": NaN}', parse_constant=reject_constant)

    @staticmethod
    def _pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result


if __name__ == "__main__":
    unittest.main()
