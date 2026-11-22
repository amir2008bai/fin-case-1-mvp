"""Tests for per-type diagnostics, side-by-side materiality reward, and API session eviction."""
import unittest

from audit_env.api import MAX_SESSIONS, App
from audit_env.contracts import Config
from audit_env.diagnostics import evaluate_diagnostics
from audit_env.generator import generate

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1,
          "vat_on_top": 1, "uncollected_service": 1}


class DiagnosticsAndHygieneTests(unittest.TestCase):
    def test_diagnostics_reports_official_and_experimental_separately(self):
        episodes = [
            generate(10, Config(keyspace="open", errors=ERRORS)),
            generate(11, Config(keyspace="open", errors={k: 0 for k in ERRORS})),
        ]
        result = evaluate_diagnostics(episodes, policies=("oracle", "expert", "inspector", "reconcile"))
        by_policy = {row["policy"]: row for row in result["summary"]}
        self.assertEqual(by_policy["expert"]["common_f1"], 1.0)
        self.assertEqual(by_policy["expert"]["materiality_f1"], 1.0)
        self.assertEqual(by_policy["inspector"]["common_f1"], 1.0)
        self.assertEqual(by_policy["inspector"]["tool_compliance"], 1.0)
        self.assertLess(by_policy["reconcile"]["common_reward"], by_policy["expert"]["common_reward"])
        self.assertLess(by_policy["reconcile"]["materiality_reward"], by_policy["expert"]["materiality_reward"])
        types_covered = {row["type"] for row in result["per_type"] if row["policy"] == "expert"}
        self.assertEqual(types_covered, set(ERRORS))

    def test_api_evicts_oldest_sessions_beyond_limit(self):
        app = App()
        first_id = app.create({"seed": 1, "config": {"keyspace": "open", "errors": {k: 0 for k in ERRORS}}})["session_id"]
        for idx in range(MAX_SESSIONS):
            app.create({"seed": 2 + idx, "config": {"keyspace": "open", "errors": {k: 0 for k in ERRORS}}})
        self.assertLessEqual(len(app.sessions), MAX_SESSIONS)
        self.assertNotIn(first_id, app.sessions)


if __name__ == "__main__":
    unittest.main()
