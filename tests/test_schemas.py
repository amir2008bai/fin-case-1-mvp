"""Published JSON Schemas stay in sync with the code and accept the committed data."""
import json
import re
import unittest
from pathlib import Path

from audit_env.contracts import ALL_TYPES, TABLE_FIELDS, Config

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
        self.assertEqual(truth["properties"]["issues"]["items"]["properties"]["type"]["enum"], list(ALL_TYPES))
        self.assertEqual(set(config["properties"]["errors"]["properties"]), set(ALL_TYPES))

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
        header, *lines = (folder / "fx.csv").read_text(encoding="utf-8").splitlines()
        columns = header.split(",")
        self.assertEqual(tuple(columns), TABLE_FIELDS["fx"])
        self.assertEqual(columns[:4], ["date", "source_currency", "target_currency", "rate"])  # the brief's order
        for line in lines:
            self.assertRegex(line.split(",")[columns.index("rate")], rate)

    def test_fx_csv_with_the_old_column_order_still_loads(self):
        """Before v2.1 fx.csv started with fx_id; an import reads columns by name."""
        import tempfile

        from audit_env.contracts import Config
        from audit_env.episode_io import load_episode, write_episode
        from audit_env.generator import generate

        episode = generate(17, Config(keyspace="open"))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "ep"
            write_episode(episode, folder)
            path = next(folder.rglob("fx.csv"))
            header, *lines = path.read_text(encoding="utf-8").splitlines()
            old = ["fx_id", "date", "source_currency", "target_currency", "rate"]
            index = [header.split(",").index(name) for name in old]
            rows = [",".join(line.split(",")[i] for i in index) for line in lines]
            path.write_text("\n".join([",".join(old), *rows]) + "\n", encoding="utf-8")
            loaded = load_episode(folder)
        self.assertEqual(loaded.tables["fx"], episode.tables["fx"])


if __name__ == "__main__":
    unittest.main()
