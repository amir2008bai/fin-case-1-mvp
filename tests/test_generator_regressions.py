"""Seeds on which the generator used to record a false truth or crash.

Looking for a free slot, a scenario target's service end walked out of the period its scenario
assumes; the truth then declared an error the records do not contain, and the oracle, which
replays the truth, still scored 1. The independent auditor is the judge here.
"""
import unittest
from pathlib import Path

from audit_env.checks import audit_episode
from audit_env.contracts import DISTRACTORS, ERROR_TYPES
from audit_env.dataset import suite
from audit_env.generator import MIN_DOCUMENTS, generate
from audit_env.provenance import restore_clean
from audit_env.selfcheck import compute

HOLDOUT = Path(__file__).resolve().parent.parent / "configs" / "holdout-template.v2.json"


class TargetStaysInItsPeriodTests(unittest.TestCase):
    def assert_truth_holds(self, episode):
        result = compute(episode, audit_episode(episode))
        self.assertEqual((result.tp, result.fp, result.fn), (len(episode.issues), 0, 0))
        self.assertEqual(audit_episode(restore_clean(episode)), [])
        ledger = {row["row_id"] for row in episode.tables["ledger"]}
        for issue in episode.issues:
            for proof in issue.evidence_sets:
                self.assertTrue(ledger & set(proof), f"{issue.type}: evidence without a ledger row")

    def test_prior_period_act_does_not_walk_into_the_period(self):
        self.assert_truth_holds(generate(18356, {}))

    def test_fx_target_of_a_shipped_holdout_config_stays_in_the_period(self):
        config = next(config for _, case_id, _, _, config in suite(str(HOLDOUT)) if case_id == "holdout-13")
        self.assert_truth_holds(generate(3000150, config.as_dict() | {"keyspace": "open"}))

    def test_duplicate_target_stays_in_the_period_when_customers_are_few(self):
        config = {"acts": 80, "customers": 5, "difficulty": "hard", "filler": 0, "foreign_percent": 20,
                  "goods_percent": 50, "month": 4, "timezone": "+03:00", "vat": False,
                  "errors": {"duplicate_invoice": 6, "fx_rate": 6}}
        self.assert_truth_holds(generate(499338349, config))


class SmallestEpisodeTests(unittest.TestCase):
    def test_smallest_config_without_look_alikes_has_enough_documents(self):
        config = {"acts": 15, "customers": 5, "filler": 0, "foreign_percent": 0, "goods_percent": 0,
                  "difficulty": "easy", "month": 2, "timezone": "+06:00",
                  "errors": {kind: 0 for kind in ERROR_TYPES},
                  "distractors": {kind: False for kind in DISTRACTORS}}
        for seed in (512, 7, 99):
            episode = generate(seed, config)
            self.assertGreaterEqual(episode.manifest["document_count"], MIN_DOCUMENTS)
            self.assertEqual(audit_episode(episode), [])


if __name__ == "__main__":
    unittest.main()


class FxAmountBasisTests(unittest.TestCase):
    def test_every_fx_rate_issue_spells_out_its_conversion(self):
        """The fx_rate half of fx_boundary_payment is not a before/after difference: the receipt is not in
        the clean books at all. Provenance gives posted and correct conversions with both rate IDs."""
        from audit_env.contracts import Config

        mix = {"fx_boundary_payment": 1, "fx_partial_allocation": 1}
        for seed in range(25):
            episode = generate(seed, Config(keyspace="open", interactions=mix))
            fx = {row["fx_id"] for row in episode.tables["fx"]}
            amounts = {item["issue_id"]: item for change in episode.provenance["changes"]
                       for item in change.get("amounts", ())}
            for issue in episode.issues:
                if issue.type != "fx_rate":
                    continue
                basis = amounts[issue.issue_id]
                self.assertEqual(abs(basis["posted_kzt_minor"] - basis["correct_kzt_minor"]),
                                 issue.expected_amount_minor)
                self.assertIn(basis["rate_correct"], fx)
                if "rate_used_value" in basis:  # a mistyped rate is not a row of fx.csv
                    self.assertNotIn("rate_used", basis)
                else:
                    self.assertIn(basis["rate_used"], fx)
                    self.assertNotEqual(basis["rate_used"], basis["rate_correct"])
