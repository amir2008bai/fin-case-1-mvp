"""Data and truth: balanced books, declared discrepancies, independent sufficiency of evidence."""
import unittest
from collections import Counter, defaultdict
from datetime import datetime

from audit_env.checks import audit_episode
from audit_env.contracts import ERROR_TYPES, Config
from audit_env.generator import generate
from audit_env.provenance import restore_clean
from audit_env.scoring import score
from audit_env.selfcheck import data_invariants

ZERO = {key: 0 for key in ERROR_TYPES}


def seen_all(episode):
    return {value for rows in episode.tables.values() for row in rows for key, value in row.items()
            if key.endswith("_id") and isinstance(value, str)}


def balanced(rows):
    return sum(row["debit_minor"] - row["credit_minor"] for row in rows) == 0


class WorldTests(unittest.TestCase):
    def test_random_configs_prove_truth_independently(self):
        failures, kinds = data_invariants(120, seed_base=2026)
        self.assertEqual(failures, [])
        self.assertEqual(kinds, sorted(ERROR_TYPES))

    def test_books_balance_before_and_after_errors(self):
        for seed in range(12):
            episode = generate(seed, Config(keyspace="open", errors={key: 2 for key in ERROR_TYPES}))
            self.assertTrue(balanced(episode.tables["ledger"]))
            self.assertTrue(balanced(restore_clean(episode).tables["ledger"]))

    def test_each_type_is_independently_configurable(self):
        for kind in ERROR_TYPES:
            for count in (0, 1, 3):
                episode = generate(7, Config(keyspace="open", errors=ZERO | {kind: count}))
                self.assertEqual([issue.type for issue in episode.issues], [kind] * count)

    def test_interactions_put_two_issues_on_one_receipt(self):
        for name, pair in (("fx_boundary_payment", {"period_cutoff", "fx_rate"}),
                           ("fx_partial_allocation", {"payment_allocation", "fx_rate"})):
            episode = generate(11, Config(keyspace="open", errors=ZERO, interactions={name: 2}))
            self.assertEqual(len(episode.issues), 4)
            by_target = defaultdict(set)
            for issue in episode.issues:
                by_target[issue.affected_ids].add(issue.type)
            self.assertEqual(list(by_target.values()), [pair, pair])
            amounts = defaultdict(set)
            for issue in episode.issues:
                amounts[issue.affected_ids].add(issue.expected_amount_minor)
            self.assertTrue(all(len(values) == 2 for values in amounts.values()))

    def test_clean_episode_has_no_issues_and_no_findings(self):
        for seed in range(8):
            episode = generate(seed, Config(keyspace="open", errors=ZERO))
            self.assertEqual(episode.issues, ())
            self.assertEqual(audit_episode(episode), [])

    def test_each_evidence_set_is_independently_sufficient(self):
        episode = generate(5, Config(keyspace="open", errors={key: 2 for key in ERROR_TYPES},
                                     interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1}))
        findings = {tuple(sorted(f["affected_ids"])) + (f["type"],): f for f in audit_episode(episode)}
        for issue in episode.issues:
            finding = findings[tuple(sorted(issue.affected_ids)) + (issue.type,)]
            self.assertEqual(finding["claimed_amount_minor"], issue.expected_amount_minor)
            for proof in issue.evidence_sets:
                result = score({"episode_id": episode.manifest["episode_id"],
                                "findings": [issue.as_finding() | {"evidence_ids": list(proof)}]},
                               episode.manifest["episode_id"], (issue,), set(proof), 0)
                self.assertEqual(result.tp, 1)
                self.assertTrue(set(proof) <= seen_all(episode))

    def test_provenance_reverts_to_the_clean_hash(self):
        for seed in (3, 14, 159):
            episode = generate(seed, Config(keyspace="open", errors={key: 3 for key in ERROR_TYPES}))
            clean = restore_clean(episode)
            self.assertEqual(len(episode.provenance["changes"]), len(episode.issues))
            for change in episode.provenance["changes"]:
                self.assertTrue(change["causes"] and change["scenario"])
                self.assertTrue(change["before"] or change["after"])
            self.assertLess(len(clean.tables["invoices"]), len(episode.tables["invoices"]))

    def test_sizes_formats_and_null_links(self):
        for seed, acts in ((1, 15), (2, 80)):
            episode = generate(seed, Config(keyspace="open", acts=acts, customers=30,
                                            errors={key: 3 for key in ERROR_TYPES}))
            manifest = episode.manifest
            self.assertTrue(200 <= manifest["counts"]["ledger"] <= 2000)
            self.assertTrue(20 <= manifest["document_count"] <= 200)
            for row in episode.tables["ledger"]:
                self.assertIsNotNone(datetime.fromisoformat(row["timestamp"]).tzinfo)
                self.assertIsInstance(row["debit_minor"], int)
            for row in episode.tables["fx"]:
                self.assertRegex(row["rate"], r"^\d+\.\d{2}$")
            refs = Counter(row["invoice_refs"] is None for row in episode.tables["payments"])
            self.assertTrue(refs[True] and refs[False])
            self.assertTrue(any(row["invoice_id"] is None for row in episode.tables["ledger"]))

    def test_distractors_are_present(self):
        episode = generate(9, Config(keyspace="open", acts=70, customers=30, foreign_percent=30))
        summary = episode.provenance["world"]["summary"]
        for key in ("payments.split", "payments.unreferenced", "acts.late", "acts.next_period",
                    "payments.boundary_inside", "acts.prior_open"):
            self.assertIn(key, summary)
        accounts = Counter(row["account"] for row in episode.tables["ledger"])
        for code in ("1030", "1210", "3510", "6010", "3130", "5610"):
            self.assertIn(code, accounts)
        self.assertTrue(accounts["6250"] + accounts["7430"])


if __name__ == "__main__":
    unittest.main()
