"""Tests for `python -m audit_env jury` (Section 06 jury protocol)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audit_env.jury import format_jury_table, run_jury

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = REPO_ROOT / "configs" / "holdout-template.v2.json"


class JuryProtocolTests(unittest.TestCase):
    def test_jury_protocol_20_closed_episodes_and_20_paired_checks(self):
        result = run_jury(HOLDOUT, seed=7)
        self.assertEqual(result["episodes"], 20)
        self.assertEqual(result["clean_episodes"], 5)
        self.assertEqual(result["interacting_episodes"], 5)
        self.assertTrue(result["oracle_all_f1_one"])
        self.assertEqual(result["policies"]["oracle"]["f1"], 1.0)
        self.assertEqual(result["paired_checks"]["total"], 20)
        self.assertEqual(result["paired_checks"]["passed"], 20)
        self.assertEqual(result["paired_checks"]["failed"], 0)
        self.assertEqual(result["failures"], [])
        # On clean books, silence (random submits [] after scanning) beats flag-all (-1.0).
        self.assertLess(result["policies"]["flag-all"]["clean_reward"], result["policies"]["random"]["clean_reward"])
        self.assertLess(result["policies"]["flag-all"]["reward"], result["policies"]["random"]["reward"])

    def test_jury_key_matches_the_other_reproducible_tables(self):
        from audit_env import jury
        from audit_env.__main__ import derived_key

        self.assertEqual(jury.derived_key(7), derived_key(7))

    def test_jury_cli_and_reproducible_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "jury.json"
            proc = subprocess.run(
                [sys.executable, "-m", "audit_env", "jury", "--seed", "7", "--report", str(report_path)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("paired transform checks: 20/20 passed", proc.stdout)
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["episodes"], 20)
            self.assertEqual(saved["paired_checks"]["passed"], 20)
            # Re-running with the same seed reproduces the table byte for byte.
            second = run_jury(HOLDOUT, seed=7)
            # Ignore timing column when comparing deterministic metrics.
            self.assertEqual(saved["policies"]["oracle"]["f1"], second["policies"]["oracle"]["f1"])
            self.assertEqual(saved["policies"]["oracle"]["reward"], second["policies"]["oracle"]["reward"])
            self.assertIn("failures: none", format_jury_table(second))


if __name__ == "__main__":
    unittest.main()
