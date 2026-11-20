"""Loading user-edited folders, format errors, auditing without truth, and repository hygiene."""
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from audit_env.checks import audit
from audit_env.contracts import Config
from audit_env.episode_io import FormatError, load_episode, write_episode
from audit_env.generator import generate

ROOT = Path(__file__).resolve().parents[1]
IBAN = re.compile(r"KZ\d{2}[0-9A-Z]{16}")
BIN = re.compile(r"(?<![0-9A-Za-z_])\d{12}(?![0-9A-Za-z])")
MARKERS = re.compile(r"1CClientBankExchange|СЕКЦИЯДОКУМЕНТ|СекцияДокумент", re.IGNORECASE)
FORBIDDEN_SUFFIXES = {".xlsx", ".xls", ".rar", ".zip", ".mp4", ".webm", ".m4a", ".ogg", ".pdf", ".1cd", ".dt", ".cf"}


class EpisodeIoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "episode"
        self.episode = generate(61, Config(keyspace="open", errors={"duplicate_invoice": 1, "payment_allocation": 0,
                                                                     "fx_rate": 0, "period_cutoff": 0}))
        write_episode(self.episode, self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, name, function):
        path = self.folder / name
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(function(lines)) + "\n", encoding="utf-8", newline="\n")

    def test_removing_a_duplicate_document_clears_the_finding(self):
        findings = audit(self.episode.manifest, self.episode.policy, self.episode.tables)
        self.assertEqual([f["type"] for f in findings], ["duplicate_invoice"])
        duplicate = findings[0]["affected_ids"][1]
        self.edit("invoices.jsonl", lambda lines: [line for line in lines if duplicate not in line])
        self.edit("ledger.csv", lambda lines: [line for line in lines if duplicate not in line])
        loaded = load_episode(self.folder)
        self.assertEqual(audit(loaded.manifest, loaded.policy, loaded.tables), [])
        self.assertEqual(loaded.manifest["counts"]["invoices"], len(self.episode.tables["invoices"]) - 1)

    def test_format_problems_are_reported(self):
        self.edit("ledger.csv", lambda lines: lines[:1] + [lines[1].replace(",", ";", 1)] + lines[2:])
        self.edit("payments.jsonl", lambda lines: lines[:1] + ["{not json"] + lines[1:])
        with self.assertRaises(FormatError) as context:
            load_episode(self.folder)
        text = " ".join(context.exception.problems)
        self.assertIn("ledger.csv:2", text)
        self.assertIn("payments.jsonl:2", text)

    def test_unknown_references_and_bad_values(self):
        def corrupt(lines):
            item = json.loads(lines[0])
            item["invoice_refs"] = ["invoice_" + "9" * 24]
            item["timestamp"] = "2026-03-01 10:00"
            return [json.dumps(item, ensure_ascii=False)] + lines[1:]
        self.edit("payments.jsonl", corrupt)
        with self.assertRaises(FormatError) as context:
            load_episode(self.folder)
        self.assertTrue(any("offset" in problem for problem in context.exception.problems))

    def test_missing_files(self):
        (self.folder / "fx.csv").unlink()
        with self.assertRaises(FormatError):
            load_episode(self.folder)


class HygieneTests(unittest.TestCase):
    def test_no_real_banking_data_in_tracked_files(self):
        try:
            listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                                    text=True, encoding="utf-8", check=True).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            listed = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
                      if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts]
        for name in listed:
            path = ROOT / name
            if not path.is_file():
                continue
            with self.subTest(file=name):
                self.assertNotIn(path.suffix.lower(), FORBIDDEN_SUFFIXES)
                text = path.read_text(encoding="utf-8", errors="ignore")
                self.assertIsNone(IBAN.search(text), "IBAN-like account number")
                self.assertIsNone(BIN.search(text), "12-digit BIN/IIN-like number")
                if name not in {"tests/test_episode_io_hygiene.py", ".gitignore"}:
                    self.assertIsNone(MARKERS.search(text), "1C bank exchange marker")


if __name__ == "__main__":
    unittest.main()
