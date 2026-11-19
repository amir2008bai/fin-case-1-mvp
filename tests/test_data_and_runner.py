import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from audit_env.contracts import ALLOCATION, DUPLICATE, Config
from audit_env.dataset import export
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.runner import run_subprocess

SUITE = Path(__file__).resolve().parents[1] / "configs/open-suite.v1.json"


class DataTests(unittest.TestCase):
    def test_independent_accounting_and_evidence(self):
        coverage = set()
        for seed in range(8):
            episode = generate(seed, Config(duplicate_count=3, allocation_count=6))
            totals = Counter()
            ledger = {row["row_id"]: row for row in episode.tables["ledger"]}
            invoice_docs = {row["doc_id"]: row for row in episode.tables["invoices"]}
            payment_docs = {row["doc_id"]: row for row in episode.tables["payments"]}
            for row in ledger.values():
                totals[row["currency"]] += row["debit_minor"] - row["credit_minor"]
            self.assertFalse(any(totals.values()))
            for issue, change in zip(episode.issues, episode.provenance["changes"]):
                self.assertEqual(issue.issue_id, change["issue_id"])
                for doc_id, *row_ids in issue.evidence_sets:
                    rows = [ledger[key] for key in row_ids]
                    if issue.type == DUPLICATE:
                        invoice = invoice_docs[doc_id]
                        self.assertEqual(issue.affected_ids, (invoice["invoice_id"],))
                        self.assertEqual(len(rows), 2)
                        self.assertEqual({r["invoice_id"] for r in rows}, {invoice["invoice_id"]})
                        self.assertEqual({r["debit_minor"] + r["credit_minor"] for r in rows},
                                         {invoice["total_minor"]})
                        self.assertEqual(invoice["total_minor"], issue.expected_amount_minor)
                        self.assertEqual(change["before"], [])
                    elif issue.type == ALLOCATION:
                        payment = payment_docs[doc_id]
                        self.assertEqual(issue.affected_ids, (payment["payment_id"],))
                        self.assertEqual({r["payment_id"] for r in rows}, {payment["payment_id"]})
                        allocated = sum(r["debit_minor"] for r in rows if r["account"] == "accounts_payable")
                        amount = payment["amount_minor"]
                        coverage.add("partial" if allocated else "missing")
                        self.assertEqual(amount - allocated, issue.expected_amount_minor)
                        self.assertEqual(sum(r["credit_minor"] for r in rows), amount)
                        self.assertEqual(change["before"][0]["debit_minor"], amount)
                    else:
                        self.fail(f"Unknown fixture type: {issue.type}")
        self.assertEqual(coverage, {"partial", "missing"})

    def test_provenance_can_restore_clean_books(self):
        for seed in (14, 65, 692):
            with self.subTest(seed=seed):
                clean = generate(seed, Config(duplicate_count=0, allocation_count=0))
                changed = generate(seed, Config(duplicate_count=6, allocation_count=6))
                restored = {row["row_id"]: row for row in changed.tables["ledger"]}
                for change in reversed(changed.provenance["changes"]):
                    for after in change["after"]:
                        self.assertEqual(restored.pop(after["row_id"]), after)
                    for before in change["before"]:
                        restored[before["row_id"]] = before
                self.assertEqual(restored, {row["row_id"]: row for row in clean.tables["ledger"]})
                balance = Counter()
                for row in clean.tables["ledger"]:
                    if row["account"] == "accounts_payable":
                        balance[row["invoice_id"]] += row["credit_minor"] - row["debit_minor"]
                self.assertEqual(set(balance), {invoice["invoice_id"] for invoice in clean.tables["invoices"]})
                self.assertFalse(any(balance.values()))

    def test_export_layout_and_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            export(root, SUITE)
            for split in ("train", "dev"):
                public = root / "public" / split
                private = root / "private" / split
                self.assertEqual(len(list(public.iterdir())), 10)
                self.assertEqual(sorted(p.name for p in public.iterdir()),
                                 sorted(p.name for p in private.iterdir()))
                first = next(public.iterdir())
                self.assertEqual({p.name for p in first.iterdir()}, {
                    "manifest.json", "policy.json", "ledger.csv", "fx.csv",
                    "invoices.jsonl", "payments.jsonl",
                })
                with (first / "ledger.csv").open() as stream:
                    self.assertTrue(any(row["payment_id"] == "null" for row in csv.DictReader(stream)))
            trace = [json.loads(line) for line in (root / "trajectories.jsonl").read_text().splitlines()]
            self.assertEqual(sum(item["terminated"] for item in trace), 20)
            self.assertTrue(all(set(item) == {
                "episode_id", "step", "action", "observation", "reward", "terminated", "truncated",
            } for item in trace))
            self.assertTrue(all("truth" not in item["observation"] for item in trace))


class RunnerTests(unittest.TestCase):
    def test_reference_agent_across_real_pipe(self):
        env = AuditEnv(training=False)
        observation, _ = env.reset(11, Config())
        result, trace, _ = run_subprocess(
            env, observation, [sys.executable, "-m", "audit_env.agents", "reconcile"],
        )
        self.assertEqual((result.tp, result.fp, result.fn), (2, 0, 0))
        self.assertIsNone(trace[-1]["reward"])

    def test_runner_enforces_deadline_and_protocol(self):
        cases = (
            ("import time; time.sleep(5)", "timeout", 0.4),
            ("print('not JSON', flush=True)", "agent_protocol", 5),
            ("print('x' * 300000, flush=True)", "agent_protocol", 5),
            ('print("{\\"n\\":1e1000}", flush=True)', "agent_protocol", 5),
        )
        for script, reason, timeout in cases:
            with self.subTest(reason=reason, script=script):
                env = AuditEnv(training=False, timeout_seconds=timeout)
                observation, _ = env.reset(11, Config())
                result, trace, _ = run_subprocess(env, observation, [sys.executable, "-c", script])
                self.assertEqual(result.error, reason)
                self.assertTrue(trace[-1]["truncated"])
                self.assertIsNone(trace[-1]["reward"])
