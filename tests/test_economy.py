"""Step economy: a complete auditor wastes no step, reading less of the ledger loses reward."""
import unittest
from dataclasses import replace

from audit_env.agents import create_policy
from audit_env.attacks import play
from audit_env.contracts import Config
from audit_env.economy import SkimPolicy, largest_books, minimal_steps, pages, skim_curve, waste
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.runner import run_policy

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
KEY = b"economy-test-key"


class EconomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs = [
            Config(keyspace="secret", errors={key: 0 for key in ERRORS}),
            Config(keyspace="secret", errors=ERRORS),
            Config(keyspace="secret", foreign_percent=0, errors={**ERRORS, "fx_rate": 0}),
            Config(keyspace="secret", errors={"vat_on_top": 1, "uncollected_service": 1}),
            Config(keyspace="secret", difficulty="hard", errors=ERRORS),
        ]
        cls.episodes = [generate(seed, config, key=KEY) for seed, config in enumerate(configs, start=500)]

    def test_complete_auditors_waste_and_skip_nothing(self):
        result, violations = waste(self.episodes)
        self.assertEqual(violations, [])
        for name in ("expert", "micro"):
            self.assertEqual((result[name]["wasted_steps"], result[name]["skipped_steps"]), (0, 0))

    def test_rates_are_not_read_without_foreign_currency(self):
        # Imported books (episode_io) usually carry a full rate table even when every document is in KZT.
        generated = self.episodes[2]
        self.assertTrue(all(row["currency"] == "KZT" for name in ("invoices", "payments")
                            for row in generated.tables[name]))
        rates = self.episodes[1].tables["fx"]
        domestic = replace(generated, tables={**generated.tables, "fx": rates},
                           manifest={**generated.manifest, "counts": {**generated.manifest["counts"], "fx": len(rates)}})
        env = AuditEnv(training=False)
        observation, _ = env.load_episode(domestic)
        outcome, trace, _ = run_policy(env, observation, create_policy("expert"))
        tables = {step["action"].get("table") for step in trace[1:] if isinstance(step["action"], dict)}
        self.assertNotIn("fx", tables)
        self.assertEqual((outcome.f1, len(trace) - 1), (1.0, minimal_steps(domestic)))

    def test_rates_are_read_when_a_document_is_foreign(self):
        episode = self.episodes[1]
        env = AuditEnv(training=False)
        observation, _ = env.load_episode(episode)
        _, trace, _ = run_policy(env, observation, create_policy("expert"))
        fx_pages = sum(1 for step in trace[1:] if isinstance(step["action"], dict)
                       and step["action"].get("table") == "fx")
        self.assertEqual(fx_pages, pages(len(episode.tables["fx"])))

    def test_skim_reads_exactly_the_kept_ledger_pages(self):
        episode = self.episodes[1]
        total = pages(len(episode.tables["ledger"]))
        full = play(episode, lambda: SkimPolicy(lambda count: count))
        half = play(episode, lambda: SkimPolicy(lambda count: count // 2))
        self.assertEqual(full["steps"], minimal_steps(episode))
        self.assertEqual(full["steps"] - half["steps"], total - total // 2)

    def test_reading_half_or_none_of_the_ledger_loses_reward(self):
        rows, violations = skim_curve(self.episodes)
        robust = [line for line in violations if line.startswith(("skim 50%", "no ledger"))]
        self.assertEqual(robust, [])
        by_name = {row["name"]: row for row in rows}
        self.assertEqual((by_name["complete read"]["fp"], by_name["complete read"]["fn"]), (0, 0))
        self.assertEqual(by_name["no ledger"]["tp"], 0)

    def test_largest_books_fit_the_step_limit(self):
        result = largest_books(seeds=range(1))
        self.assertTrue(result["ok"])
        self.assertGreater(result["counts"]["ledger"], 1800)
        self.assertLessEqual(result["complete_read_steps"], 60)


if __name__ == "__main__":
    unittest.main()
