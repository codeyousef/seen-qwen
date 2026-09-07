#!/usr/bin/env python3
"""Independent oracle and static checks for QWN-033C policy resolution."""

import copy
import json
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/fixtures/qwn_033c_policy_oracle.json"
SCHEMA = ROOT / "schemas/qwen-policy-resolver-oracle.schema.json"
SOURCE = ROOT / "src/quant/policy.seen"
TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


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


def matches(rule, tensor):
    if rule["tensor_name"] and rule["tensor_name"] != tensor["name"]:
        return False
    if rule["layer_start"] >= 0 and not (
        rule["layer_start"] <= tensor["layer_index"] <= rule["layer_end"]
    ):
        return False
    for rule_key, tensor_key in (
        ("layer_type", "layer_type"),
        ("semantic_role", "semantic_role"),
        ("outlier_bucket", "outlier_bucket"),
    ):
        if rule[rule_key] and rule[rule_key] != tensor[tensor_key]:
            return False
    return True


def resolve(tensors, rules):
    resolutions = []
    for tensor in tensors:
        for index, rule in enumerate(rules):
            if matches(rule, tensor):
                resolutions.append(
                    {
                        "tensor_name": tensor["name"],
                        "rule_index": index,
                        "rule_id": rule["id"],
                        "codec": rule["codec"],
                    }
                )
                break
        else:
            raise ValueError(f"unmatched tensor: {tensor['name']}")
    return resolutions


class QuantizationPolicyResolverTest(unittest.TestCase):
    def setUp(self):
        self.oracle = load_strict(ORACLE)
        self.schema = load_strict(SCHEMA)

    def test_closed_identity_and_bounds(self):
        self.assertEqual(
            set(self.oracle),
            {"schema", "version", "ordering", "profile", "hard_bounds",
             "tensors", "rules", "expected"},
        )
        self.assertEqual(self.oracle["schema"],
                         "seen-qwen-policy-resolver-oracle-v1")
        self.assertEqual(self.oracle["version"], 1)
        self.assertEqual(self.oracle["ordering"], "declared-rule-order")
        self.assertEqual(
            self.oracle["hard_bounds"],
            {"max_tensors": 2048, "max_rules": 4096,
             "max_layer_index": 63},
        )
        self.assertLessEqual(len(self.oracle["tensors"]), 2048)
        self.assertLessEqual(len(self.oracle["rules"]), 4096)

    def test_oracle_resolves_exactly_in_declared_order(self):
        self.assertEqual(
            resolve(self.oracle["tensors"], self.oracle["rules"]),
            self.oracle["expected"],
        )
        self.assertEqual(
            resolve(self.oracle["tensors"], self.oracle["rules"]),
            resolve(self.oracle["tensors"], self.oracle["rules"]),
        )

    def test_every_authorized_selector_dimension_is_exercised(self):
        rules = self.oracle["rules"]
        self.assertTrue(any(rule["tensor_name"] for rule in rules))
        self.assertTrue(any(rule["layer_start"] >= 0 for rule in rules))
        self.assertTrue(any(rule["layer_type"] for rule in rules))
        self.assertTrue(any(rule["semantic_role"] for rule in rules))
        self.assertTrue(any(rule["outlier_bucket"] for rule in rules))
        self.assertTrue(any(
            rule["layer_start"] == -1 and not rule["tensor_name"] and
            not rule["layer_type"] and not rule["semantic_role"] and
            not rule["outlier_bucket"] for rule in rules
        ))

    def test_rule_order_is_explicit_not_hidden_specificity(self):
        rules = copy.deepcopy(self.oracle["rules"])
        catch_all = rules.pop()
        rules.insert(0, catch_all)
        resolved = resolve(self.oracle["tensors"], rules)
        self.assertTrue(all(item["rule_id"] == "declared-catch-all"
                            for item in resolved))

    def test_unmatched_and_duplicate_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unmatched tensor"):
            resolve(self.oracle["tensors"][:1], self.oracle["rules"][1:2])
        tensor_names = [item["name"] for item in self.oracle["tensors"]]
        rule_ids = [item["id"] for item in self.oracle["rules"]]
        predicates = [
            (item["tensor_name"], item["layer_start"], item["layer_end"],
             item["layer_type"], item["semantic_role"],
             item["outlier_bucket"])
            for item in self.oracle["rules"]
        ]
        self.assertEqual(len(tensor_names), len(set(tensor_names)))
        self.assertEqual(len(rule_ids), len(set(rule_ids)))
        self.assertEqual(len(predicates), len(set(predicates)))

    def test_fixture_tokens_and_layer_ranges_are_canonical(self):
        self.assertRegex(self.oracle["profile"], TOKEN)
        for tensor in self.oracle["tensors"]:
            self.assertRegex(tensor["name"], TOKEN)
            self.assertGreaterEqual(tensor["layer_index"], -1)
            self.assertLessEqual(tensor["layer_index"], 63)
        for rule in self.oracle["rules"]:
            self.assertRegex(rule["id"], TOKEN)
            self.assertRegex(rule["codec"], TOKEN)
            wildcard = rule["layer_start"] == rule["layer_end"] == -1
            bounded = 0 <= rule["layer_start"] <= rule["layer_end"] <= 63
            self.assertTrue(wildcard or bounded)
            for key in ("tensor_name", "layer_type", "semantic_role",
                        "outlier_bucket"):
                self.assertTrue(not rule[key] or TOKEN.fullmatch(rule[key]))

    def test_schema_is_closed_and_native_contract_is_fail_closed(self):
        self.assertFalse(self.schema["additionalProperties"])
        self.assertEqual(self.schema["properties"]["schema"]["const"],
                         "seen-qwen-policy-resolver-oracle-v1")
        for name in ("bounds", "tensor", "rule", "resolution"):
            self.assertFalse(self.schema["$defs"][name]["additionalProperties"])
        source = SOURCE.read_text(encoding="utf-8")
        for diagnostic in (
            "qwen.policy.schema", "qwen.policy.bounds", "qwen.policy.limit",
            "qwen.policy.input", "qwen.policy.duplicate",
            "qwen.policy.ambiguous", "qwen.policy.unmatched",
            "qwen.policy.closed", "qwen.policy.index",
        ):
            self.assertIn(diagnostic, source)
        self.assertIn("retryable: false", source)
        self.assertIn("while ruleIndex < rules.length() and matchedRule < 0",
                      source)

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        text = ORACLE.read_text(encoding="utf-8")
        duplicate = text.replace('"version": 1,',
                                 '"version": 1, "version": 2,', 1)
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            json.loads(duplicate, object_pairs_hook=self._pairs,
                       parse_constant=reject_constant)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            json.loads('{"layer_index": NaN}', parse_constant=reject_constant)

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
