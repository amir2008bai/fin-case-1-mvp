"""Data and truth: balanced books, declared discrepancies, independent sufficiency of evidence."""
import random
import unittest
from collections import Counter, defaultdict
from datetime import datetime

from audit_env.checks import audit_episode
from audit_env.contracts import ALL_TYPES, ERROR_TYPES, EXTRA_TYPES, MAX_ISSUES, Config
from audit_env.generator import generate
from audit_env.provenance import restore_clean
from audit_env.scoring import score
from audit_env.selfcheck import data_invariants, random_config

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
        self.assertEqual(kinds, sorted(ALL_TYPES))

    def test_prior_cutoff_invoice_stays_in_the_prior_period_after_a_collision(self):
        config = Config(keyspace="open", month=2, acts=40, customers=5, foreign_percent=0,
                        errors={"period_cutoff": 6, "duplicate_invoice": 2, "payment_allocation": 2})
        episode = generate(877, config)
        result = score({"episode_id": episode.manifest["episode_id"], "findings": audit_episode(episode)},
                       episode.manifest["episode_id"], episode.issues, seen_all(episode), 0)
        self.assertEqual((result.tp, result.fp, result.fn), (10, 0, 0))

    def test_random_configs_respect_the_issue_cap(self):
        # Regression: verify draws fresh random configs; about one in 2 300 used to exceed the cap and crash it.
        rng = random.Random(7)
        counts = [random_config(rng).issue_count for _ in range(6000)]
        self.assertLessEqual(max(counts), MAX_ISSUES)
        self.assertEqual(max(counts), MAX_ISSUES, "the cap itself stays reachable")

    def test_real_practice_types_are_independently_configurable(self):
        for kind in EXTRA_TYPES:
            for count in (1, 3):
                episode = generate(13, Config(keyspace="open", errors=ZERO | {kind: count}))
                self.assertEqual([issue.type for issue in episode.issues], [kind] * count)
                self.assertEqual(audit_episode(restore_clean(episode)), [])
                result = score({"episode_id": episode.manifest["episode_id"], "findings": audit_episode(episode)},
                               episode.manifest["episode_id"], episode.issues, seen_all(episode), 0)
                self.assertEqual(result.f1, 1.0)

    def test_difficulty_scales_look_alikes_and_keeps_truth(self):
        totals = {}
        for level in ("easy", "medium", "hard"):
            summary = Counter()
            for seed in range(12):
                episode = generate(seed, Config(keyspace="open", difficulty=level, acts=60, customers=28,
                                                errors={key: 1 for key in ERROR_TYPES} | {key: 1 for key in EXTRA_TYPES}))
                summary.update(episode.provenance["world"]["summary"])
                self.assertEqual(audit_episode(restore_clean(episode)), [])
                result = score({"episode_id": episode.manifest["episode_id"], "findings": audit_episode(episode)},
                               episode.manifest["episode_id"], episode.issues, seen_all(episode), 0)
                self.assertEqual(result.f1, 1.0)
            totals[level] = summary
        look_alikes = ("acts.late", "acts.prior_open", "acts.next_period", "payments.boundary_inside")
        for name in look_alikes:
            self.assertLess(totals["easy"][name], totals["medium"][name], name)
            self.assertLess(totals["medium"][name], totals["hard"][name], name)
        for name in ("payments.double_payment", "payments.underpaid_vat", "acts.paid_renewal", "payments.partial_refs"):
            self.assertEqual(totals["medium"][name], 0, name)
            self.assertGreater(totals["hard"][name], 0, name)

    def test_long_hard_books_stay_within_the_case_limits(self):
        for seed in range(4):
            episode = generate(seed, Config(keyspace="open", difficulty="hard", acts=80, customers=35, filler=800,
                                            errors={key: 2 for key in ERROR_TYPES}))
            self.assertTrue(1500 <= episode.manifest["counts"]["ledger"] <= 2000)
            self.assertLessEqual(episode.manifest["document_count"], 200)

    def test_vat_on_top_leaves_exactly_the_added_vat_unpaid(self):
        episode = generate(17, Config(keyspace="open", errors=ZERO | {"vat_on_top": 1}))
        issue = episode.issues[0]
        invoice = next(row for row in episode.tables["invoices"] if row["invoice_id"] == issue.affected_ids[0])
        base = sum(line["quantity"] * line["unit_price_minor"] for line in invoice["line_items"])
        self.assertEqual(invoice["total_minor"] - base, issue.expected_amount_minor)
        receivable = sum(row["debit_minor"] - row["credit_minor"] for row in episode.tables["ledger"]
                         if row["invoice_id"] == invoice["invoice_id"] and row["account"] == "1210")
        self.assertEqual(receivable, issue.expected_amount_minor)

    def test_goods_are_sold_with_waybills_cost_and_vat(self):
        episode = generate(21, Config(keyspace="open", acts=60, foreign_percent=0, errors=ZERO))
        kinds = Counter(doc["document_kind"] for doc in episode.documents.values() if doc["doc_type"] == "invoice")
        self.assertLessEqual(abs(kinds["waybill"] - kinds["act"]), 3)
        accounts = Counter(row["account"] for row in episode.tables["ledger"])
        for code in ("1330", "7010", "1420"):
            self.assertIn(code, accounts)
        waybills = {doc["invoice_id"] for doc in episode.documents.values() if doc.get("document_kind") == "waybill"}
        cogs = {row["invoice_id"] for row in episode.tables["ledger"] if row["account"] == "7010"}
        revenue = {row["invoice_id"] for row in episode.tables["ledger"] if row["account"] == "6010"}
        self.assertEqual(cogs, waybills & revenue)
        for doc in episode.documents.values():
            if doc.get("document_kind") == "waybill":
                self.assertIn("З-2", doc["title"])
                if doc["vat_minor"]:
                    self.assertEqual(doc["vat_invoice"]["vat_minor"], doc["vat_minor"])
                for line in doc["line_items"]:
                    self.assertEqual(line["total_minor"], line["quantity"] * line["unit_price_minor"])
                    self.assertEqual(line["service_period_start"], line["service_period_end"])
        no_goods = generate(21, Config(keyspace="open", acts=60, goods_percent=0, errors=ZERO))
        self.assertFalse(any(row["account"] in ("1330", "7010") for row in no_goods.tables["ledger"]))

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
