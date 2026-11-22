"""Error targets must look like ordinary documents of their kind.

Before the fix a policy could skip half the audit without reading: errors were only in service acts
(never in waybills), a cut-off invoice was never paid, a receipt with an error came 1-5 days after the
invoice, and an uncollected renewal was always the only business of a brand-new customer.
"""
import random
import unittest
from collections import Counter

from audit_env.contracts import Config
from audit_env.generator import generate
from audit_env.leakage import STRUCTURE_THRESHOLD, audit_leakage, structural_pairs

ALL = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 2,
       "vat_on_top": 1, "uncollected_service": 1}


def episodes(count, seed=0, **overrides):
    rng = random.Random(seed)
    return [generate(rng.randrange(2**31), Config(keyspace="open", errors=ALL, **overrides)) for _ in range(count)]


class StructuralTellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = episodes(60)

    def test_error_targets_include_waybills(self):
        labels = Counter()
        for episode in self.sample:
            for waybill, label in structural_pairs(episode)["goods_waybill"]:
                if label:
                    labels[waybill] += 1
        self.assertGreater(labels[1.0], 0, "no error ever sits in a waybill")
        self.assertGreater(labels[0.0], 0)

    def test_cut_off_invoices_are_sometimes_paid(self):
        cited = [value for episode in self.sample
                 for value, label in structural_pairs(episode)["cited_by_receipt"] if label]
        self.assertTrue(cited)
        self.assertTrue(0 < sum(cited) < len(cited), "cut-off targets are always or never paid")

    def test_uncollected_customer_is_not_always_new(self):
        counts = {value for episode in self.sample
                  for value, label in structural_pairs(episode)["customer_invoices"] if label}
        self.assertGreater(len(counts), 1, "the uncollected customer always has the same number of invoices")

    def test_structural_features_stay_near_chance(self):
        result = audit_leakage(self.sample)
        rows = {row["feature"]: row for row in result["features"]}
        for name in ("goods_waybill", "cited_by_receipt", "receipt_lag_days"):
            self.assertEqual(rows[name]["threshold"], STRUCTURE_THRESHOLD)
            # a small sample: a loose bound that the old generator (drift 0.21-0.39) still failed
            self.assertLess(rows[name]["drift"], 0.17, rows[name])


if __name__ == "__main__":
    unittest.main()
