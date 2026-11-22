"""Every committed data file validates against the published JSON Schemas.

A schema the code no longer follows misleads anyone who validates with it (the manifest and truth
schemas once still required schema_version "2.0" after the data moved on). The validator below covers
exactly the keywords our schemas use, so the test needs no third-party package.
"""
import csv
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
DATA = ROOT / "data"
KEYWORDS = {"$schema", "$id", "$ref", "title", "description", "format", "type", "const", "enum", "pattern",
            "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "items",
            "properties", "required", "additionalProperties", "oneOf"}
TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def load(name):
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def errors(value, schema, path="$"):
    unknown = set(schema) - KEYWORDS
    if unknown:
        return [f"{path}: validator does not know {sorted(unknown)}"]
    if "$ref" in schema:
        return errors(value, load(schema["$ref"]), path)
    found = []
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        ok = any((kind == "integer" and type(value) is int) or
                 (kind == "number" and type(value) in (int, float)) or
                 (kind in TYPES and type(value) is TYPES[kind]) for kind in kinds)
        if not ok:
            return [f"{path}: expected {kinds}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        found.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{path}: {value!r} not in enum")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            found.append(f"{path}: {value!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            found.append(f"{path}: length {len(value)}")
    if type(value) in (int, float):
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            found.append(f"{path}: {value} out of range")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            found.append(f"{path}: {len(value)} items")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            found.append(f"{path}: items not unique")
        for index, item in enumerate(value):
            found += errors(item, schema.get("items", {}), f"{path}[{index}]")
    if isinstance(value, dict):
        for name in schema.get("required", ()):
            if name not in value:
                found.append(f"{path}: missing {name}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                found += errors(item, properties[name], f"{path}.{name}")
            elif extra is False:
                found.append(f"{path}: unexpected {name}")
            elif isinstance(extra, dict):
                found += errors(item, extra, f"{path}.{name}")
    if "oneOf" in schema:
        matching = [option for option in schema["oneOf"] if not errors(value, option, path)]
        if len(matching) != 1:
            found.append(f"{path}: matches {len(matching)} oneOf options")
    return found


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            yield row


class CommittedDataValidatesTests(unittest.TestCase):
    def setUp(self):
        if not (DATA / "episodes").is_dir():
            self.skipTest("data/ not exported")

    def assert_valid(self, value, schema_name, where):
        problems = errors(value, load(schema_name))
        self.assertEqual(problems[:5], [], f"{where}")

    def test_episode_files(self):
        folders = sorted(path.parent for path in (DATA / "episodes").rglob("manifest.json"))
        self.assertEqual(len(folders), 20)
        for folder in folders:
            self.assert_valid(json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
                              "manifest.schema.json", folder.name)
            for name, schema in (("invoices.jsonl", "invoice.schema.json"), ("payments.jsonl", "payment.schema.json")):
                for line in (folder / name).read_text(encoding="utf-8").splitlines():
                    self.assert_valid(json.loads(line), schema, f"{folder.name}/{name}")
            for row in csv_rows(folder / "fx.csv"):
                self.assert_valid(row, "fx_row.schema.json", f"{folder.name}/fx.csv")

    def test_truth_files(self):
        paths = sorted((DATA / "answers").rglob("truth.json"))
        self.assertEqual(len(paths), 20)  # an empty answers folder must not pass silently
        for path in paths:
            self.assert_valid(json.loads(path.read_text(encoding="utf-8")), "truth.schema.json", str(path))

    def test_trajectory_records(self):
        lines = (DATA / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines)
        for line in lines[:400]:
            self.assert_valid(json.loads(line), "trajectory_record.schema.json", "trajectories.jsonl")


if __name__ == "__main__":
    unittest.main()
