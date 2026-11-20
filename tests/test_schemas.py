"""Published JSON Schemas stay in sync with the code and accept the committed data."""
import json
import re
import unittest
from pathlib import Path

from audit_env.contracts import ERROR_TYPES, TABLE_FIELDS, Config

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
FILES = {"ledger": "ledger_row", "invoices": "invoice", "payments": "payment", "fx": "fx_row"}


def load(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


class SchemaTests(unittest.TestCase):
    def test_table_schemas_match_the_columns(self):
        for table, name in FILES.items():
            schema = load(name)
            self.assertEqual(tuple(schema["properties"]), TABLE_FIELDS[table])
            self.assertEqual(tuple(schema["required"]), TABLE_FIELDS[table])
            self.assertFalse(schema["additionalProperties"])

    def test_config_and_truth_schemas_match_the_code(self):
        config = load("config")
        self.assertEqual(set(config["properties"]), set(Config.__dataclass_fields__))
        truth = load("truth")
        self.assertEqual(truth["properties"]["issues"]["items"]["properties"]["type"]["enum"], list(ERROR_TYPES))

    def test_committed_rows_follow_the_patterns(self):
        data = ROOT / "data" / "episodes"
        if not data.is_dir():
            self.skipTest("data/ not exported")
        instant = re.compile(load("ledger_row")["properties"]["timestamp"]["pattern"])
        rate = re.compile(load("fx_row")["properties"]["rate"]["pattern"])
        folder = next((data / "train").iterdir())
        for line in (folder / "invoices.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            self.assertEqual(tuple(item), TABLE_FIELDS["invoices"])
            self.assertRegex(item["issued_at"], instant)
        for line in (folder / "fx.csv").read_text(encoding="utf-8").splitlines()[1:]:
            self.assertRegex(line.split(",")[-1], rate)


if __name__ == "__main__":
    unittest.main()
