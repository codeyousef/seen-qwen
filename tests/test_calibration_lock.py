#!/usr/bin/env python3

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/calibration.lock.json"
SCHEMA = ROOT / "schemas/qwen-calibration-lock.schema.json"
SUPPLEMENT = ROOT / "tests/fixtures/qwn_033a_calibration_supplement.json"
LOCK_SHA256 = "b72f162aa04c62c6796c5c0822bbb823a569b0324477a19077dc42a7d0c5b98e"
SUPPLEMENT_SHA256 = "d40134919a871d57a039a259880ee283e75b8344a7c18b0858e41e7da4c75a30"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ROOT_FIELDS = {
    "schema", "version", "classification", "privacy", "ordering", "model",
    "sources", "samples", "bounds", "coverage",
}
SOURCE_FIELDS = {
    "id", "kind", "path", "revision", "license", "bytes", "sha256",
    "runtime_executable", "private_data",
}
SAMPLE_FIELDS = {
    "id", "source_id", "payload_kind", "payload_sha256", "token_count", "role",
}
REQUIRED_ROLES = {
    "natural-language", "seen-code", "arabic-natural-language",
    "multilingual-natural-language", "long-document", "long-context-retrieval",
    "thinking-enabled", "thinking-disabled",
    "high-variance-activation-probe-candidate",
    "low-variance-activation-probe-candidate",
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_closed_lock(document: dict[str, object]) -> None:
    if set(document) != ROOT_FIELDS:
        raise ValueError("root fields")
    if document["schema"] != "seen-qwen-calibration-lock-v1":
        raise ValueError("schema")
    if document["privacy"] != "public-and-project-authored-no-private-user-data":
        raise ValueError("privacy")
    model = document["model"]
    if model["model_revision"] != MODEL_REVISION or model["trust_remote_code"] is not False:
        raise ValueError("model")
    sources = document["sources"]
    if len(sources) != 4 or [x["id"] for x in sources] != sorted(x["id"] for x in sources):
        raise ValueError("source order")
    for source in sources:
        if set(source) != SOURCE_FIELDS:
            raise ValueError("source fields")
        if source["runtime_executable"] is not False or source["private_data"] is not False:
            raise ValueError("unsafe source")
        if source["revision"] != f"sha256:{source['sha256']}":
            raise ValueError("source revision")
    samples = document["samples"]
    if len(samples) != 14 or [x["id"] for x in samples] != sorted(x["id"] for x in samples):
        raise ValueError("sample order")
    source_ids = {x["id"] for x in sources}
    for sample in samples:
        if set(sample) != SAMPLE_FIELDS or sample["source_id"] not in source_ids:
            raise ValueError("sample fields")
        if len(sample["payload_sha256"]) != 64 or sample["token_count"] <= 0:
            raise ValueError("sample bounds")
    bounds = document["bounds"]
    if bounds != {
        "source_count": 4,
        "sample_count": 14,
        "aggregate_tokens": 426873,
        "max_sample_tokens": 262144,
        "max_generated_bytes": "8388608",
        "workers": 1,
    }:
        raise ValueError("bounds")
    if sum(x["token_count"] for x in samples) != bounds["aggregate_tokens"]:
        raise ValueError("aggregate")
    roles = {x["role"] for x in samples}
    if not REQUIRED_ROLES <= roles or not all(document["coverage"].values()):
        raise ValueError("coverage")


class CalibrationLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"), object_pairs_hook=strict_object)
        cls.supplement = json.loads(
            SUPPLEMENT.read_text(encoding="utf-8"), object_pairs_hook=strict_object
        )

    def test_lock_is_content_addressed_closed_and_safe(self) -> None:
        self.assertEqual(digest(LOCK), LOCK_SHA256)
        validate_closed_lock(self.lock)
        self.assertEqual(self.lock["classification"], "calibration-input-lock")
        self.assertEqual(self.lock["ordering"], "lexicographic-sample-id")
        self.assertEqual(self.lock["model"]["license"], "Apache-2.0")
        self.assertEqual(self.lock["model"]["tokenizer_revision"], MODEL_REVISION)

    def test_every_source_is_repo_relative_exact_and_non_executable(self) -> None:
        for source in self.lock["sources"]:
            relative = Path(source["path"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            path = ROOT / relative
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_size, int(source["bytes"]))
            self.assertEqual(digest(path), source["sha256"])
            self.assertFalse(source["runtime_executable"])
            self.assertFalse(source["private_data"])

    def test_supplement_recipes_are_bounded_deterministic_and_token_safe(self) -> None:
        self.assertEqual(digest(SUPPLEMENT), SUPPLEMENT_SHA256)
        self.assertEqual(self.supplement["schema"], "seen-qwen-calibration-supplement-v1")
        self.assertEqual(self.supplement["privacy"], "project-authored-no-private-user-data")
        self.assertEqual(self.supplement["tokenizer_revision"], MODEL_REVISION)
        recipes = self.supplement["recipes"]
        self.assertEqual([x["id"] for x in recipes], sorted(x["id"] for x in recipes))
        locked = {
            x["id"]: x for x in self.lock["samples"]
            if x["source_id"] == "qwn-033a-supplement"
        }
        self.assertEqual(set(locked), {x["id"] for x in recipes})
        for recipe in recipes:
            self.assertEqual(canonical_digest(recipe), locked[recipe["id"]]["payload_sha256"])
            self.assertEqual(recipe["token_count"], locked[recipe["id"]]["token_count"])
            token_values = []
            if recipe["kind"] == "explicit_token_ids":
                token_values = recipe["token_ids"]
                self.assertEqual(len(token_values), recipe["token_count"])
            elif recipe["kind"] == "repeat_token":
                token_values = [recipe["token_id"]]
                self.assertEqual(recipe["repetitions"], recipe["token_count"])
            elif recipe["kind"] == "cycle_token_ids":
                token_values = recipe["cycle_token_ids"]
                self.assertTrue(token_values)
                self.assertEqual(recipe["target_tokens"], recipe["token_count"])
            elif recipe["kind"] == "fill_and_insert":
                token_values = [recipe["filler_token_id"], *recipe["needle_token_ids"]]
                self.assertEqual(recipe["target_tokens"], recipe["token_count"])
                offsets = recipe["needle_offsets"]
                self.assertEqual(offsets, sorted(set(offsets)))
                width = len(recipe["needle_token_ids"])
                self.assertTrue(all(0 <= x <= recipe["target_tokens"] - width for x in offsets))
                self.assertTrue(all(b - a >= width for a, b in zip(offsets, offsets[1:])))
            else:
                self.fail(f"unknown recipe kind: {recipe['kind']}")
            self.assertTrue(all(isinstance(x, int) and 0 <= x < 248320 for x in token_values))

    def test_strict_duplicate_and_hostile_mutations_fail_closed(self) -> None:
        duplicate = LOCK.read_text(encoding="utf-8").replace(
            '"schema": "seen-qwen-calibration-lock-v1",',
            '"schema": "seen-qwen-calibration-lock-v1", "schema": "evil",',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            json.loads(duplicate, object_pairs_hook=strict_object)
        for mutation in ("extra", "private", "executable", "reordered", "oversized"):
            value = deepcopy(self.lock)
            if mutation == "extra":
                value["unexpected"] = True
            elif mutation == "private":
                value["sources"][0]["private_data"] = True
            elif mutation == "executable":
                value["sources"][1]["runtime_executable"] = True
            elif mutation == "reordered":
                value["samples"][0], value["samples"][1] = value["samples"][1], value["samples"][0]
            else:
                value["samples"][-1]["token_count"] = 262145
            with self.assertRaises(ValueError, msg=mutation):
                validate_closed_lock(value)

    def test_schema_is_closed_and_matches_the_runtime_cardinality(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"), object_pairs_hook=strict_object)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), ROOT_FIELDS)
        self.assertEqual(schema["properties"]["sources"]["minItems"], 4)
        self.assertEqual(schema["properties"]["samples"]["maxItems"], 14)
        self.assertFalse(schema["properties"]["model"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["source"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["sample"]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
