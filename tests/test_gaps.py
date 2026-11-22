"""Targeted gap-closing tests: grader mutation checks, truth/evidence integrity,
full-episode closed-mode zero-leak, cross-process reproducibility, and committed data guards.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

from audit_env import scoring
from audit_env.contracts import Config, MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.micro import MicroModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
SECRET_KEY = b"test-gaps-fixed-secret-key-32b!!"
MIXED_SECRET = Config(
    keyspace="secret",
    errors={
        "duplicate_invoice": 1,
        "payment_allocation": 1,
        "fx_rate": 1,
        "period_cutoff": 1,
        "vat_on_top": 1,
        "uncollected_service": 1,
    },
    interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1},
)


def _public_evidence_ids(episode):
    ids = set()
    for inv in episode.tables["invoices"]:
        ids.add(inv["doc_id"])
    for pay in episode.tables["payments"]:
        ids.add(pay["doc_id"])
    for row in episode.tables["ledger"]:
        ids.add(row["row_id"])
    for fx in episode.tables["fx"]:
        ids.add(fx["fx_id"])
    return ids


def _finding_for(issue):
    return {
        "type": issue.type,
        "affected_ids": list(issue.affected_ids),
        "claimed_amount_minor": issue.expected_amount_minor,
        "currency": issue.currency,
        "evidence_ids": list(issue.evidence_sets[0]),
        "explanation": "verified discrepancy",
    }


class GraderMutationTests(unittest.TestCase):
    """Verify via unittest.mock.patch that breaking any scoring.py rule is caught immediately."""

    @classmethod
    def setUpClass(cls):
        cls.ep = generate(303, MIXED_SECRET, key=SECRET_KEY)
        cls.ep_id = cls.ep.manifest["episode_id"]
        cls.seen = _public_evidence_ids(cls.ep)

    def test_mutation_scoring_currency_check_is_enforced(self):
        issue = self.ep.issues[0]
        wrong_ccy = _finding_for(issue) | {"currency": "USD" if issue.currency != "USD" else "KZT"}
        payload = {"episode_id": self.ep_id, "findings": [wrong_ccy]}
        # Real grader rejects mismatched currency.
        real = scoring.score(payload, self.ep_id, (issue,), self.seen, 1)
        self.assertEqual((real.tp, real.fp), (0, 1))

        # Mutant ignoring currency check wrongly accepts the finding.
        def mutant_matches(finding, iss, seen):
            ev = set(finding["evidence_ids"])
            return (
                finding["type"] == iss.type
                and set(finding["affected_ids"]) == set(iss.affected_ids)
                and abs(finding["claimed_amount_minor"] - iss.expected_amount_minor) <= iss.tolerance_minor
                and ev <= seen
                and any(set(proof) <= ev for proof in iss.evidence_sets)
            )

        with mock.patch.object(scoring, "matches", side_effect=mutant_matches):
            mutated = scoring.score(payload, self.ep_id, (issue,), self.seen, 1)
        self.assertEqual(mutated.tp, 1)
        self.assertNotEqual(real.tp, mutated.tp)

    def test_mutation_scoring_tolerance_check_is_enforced(self):
        issue = next(i for i in self.ep.issues if i.type == "fx_rate")
        over_tol = _finding_for(issue) | {"claimed_amount_minor": issue.expected_amount_minor + issue.tolerance_minor + 1}
        payload = {"episode_id": self.ep_id, "findings": [over_tol]}
        real = scoring.score(payload, self.ep_id, (issue,), self.seen, 1)
        self.assertEqual((real.tp, real.fp), (0, 1))

        def mutant_matches(finding, iss, seen):
            ev = set(finding["evidence_ids"])
            return (
                finding["type"] == iss.type
                and set(finding["affected_ids"]) == set(iss.affected_ids)
                and finding["currency"] == iss.currency
                and ev <= seen
                and any(set(proof) <= ev for proof in iss.evidence_sets)
            )

        with mock.patch.object(scoring, "matches", side_effect=mutant_matches):
            mutated = scoring.score(payload, self.ep_id, (issue,), self.seen, 1)
        self.assertEqual(mutated.tp, 1)
        self.assertNotEqual(real.tp, mutated.tp)

    def test_mutation_scoring_seen_evidence_check_is_enforced(self):
        issue = self.ep.issues[0]
        payload = {"episode_id": self.ep_id, "findings": [_finding_for(issue)]}
        real = scoring.score(payload, self.ep_id, (issue,), set(), 1)
        self.assertEqual((real.tp, real.fp), (0, 1))

        def mutant_matches(finding, iss, _seen):
            ev = set(finding["evidence_ids"])
            return (
                finding["type"] == iss.type
                and set(finding["affected_ids"]) == set(iss.affected_ids)
                and finding["currency"] == iss.currency
                and abs(finding["claimed_amount_minor"] - iss.expected_amount_minor) <= iss.tolerance_minor
                and any(set(proof) <= ev for proof in iss.evidence_sets)
            )

        with mock.patch.object(scoring, "matches", side_effect=mutant_matches):
            mutated = scoring.score(payload, self.ep_id, (issue,), set(), 1)
        self.assertEqual(mutated.tp, 1)
        self.assertNotEqual(real.tp, mutated.tp)

    def test_mutation_scoring_one_to_one_matching_is_enforced(self):
        issue = self.ep.issues[0]
        dup = _finding_for(issue)
        payload = {"episode_id": self.ep_id, "findings": [dup, dup]}
        real = scoring.score(payload, self.ep_id, (issue,), self.seen, 1)
        self.assertEqual((real.tp, real.fp), (1, 1))

        # Even if two identical issues exist, a single finding must match at most one issue.
        twin_res = scoring.score({"episode_id": self.ep_id, "findings": [dup]}, self.ep_id, (issue, issue), self.seen, 1)
        self.assertEqual((twin_res.tp, twin_res.fn), (1, 1))

    def test_mutation_scoring_limits_30_10_1000_are_enforced(self):
        issue = self.ep.issues[0]
        base = _finding_for(issue)

        # 31 findings rejected; mutant raising MAX_FINDINGS accepts structure.
        over_findings = {"episode_id": self.ep_id, "findings": [base] * (MAX_FINDINGS + 1)}
        self.assertEqual(scoring.score(over_findings, self.ep_id, (issue,), self.seen, 1).error, "invalid_submission")
        with mock.patch.object(scoring, "MAX_FINDINGS", MAX_FINDINGS + 10):
            self.assertIsNone(scoring.score(over_findings, self.ep_id, (issue,), self.seen, 1).error)

        # 11 evidence IDs rejected; mutant raising MAX_EVIDENCE accepts structure.
        over_ev = {"episode_id": self.ep_id, "findings": [base | {"evidence_ids": [f"row_{i:024x}" for i in range(MAX_EVIDENCE + 1)]}]}
        self.assertEqual(scoring.score(over_ev, self.ep_id, (issue,), self.seen, 1).error, "invalid_submission")
        with mock.patch.object(scoring, "MAX_EVIDENCE", MAX_EVIDENCE + 10):
            self.assertIsNone(scoring.score(over_ev, self.ep_id, (issue,), self.seen, 1).error)

        # 1001-char explanation rejected; mutant raising MAX_EXPLANATION accepts structure.
        over_exp = {"episode_id": self.ep_id, "findings": [base | {"explanation": "x" * (MAX_EXPLANATION + 1)}]}
        self.assertEqual(scoring.score(over_exp, self.ep_id, (issue,), self.seen, 1).error, "invalid_submission")
        with mock.patch.object(scoring, "MAX_EXPLANATION", MAX_EXPLANATION + 500):
            self.assertIsNone(scoring.score(over_exp, self.ep_id, (issue,), self.seen, 1).error)


class TruthEvidenceAndIsolationGapTests(unittest.TestCase):
    def test_truth_evidence_ids_exist_in_public_tables_and_use_allowed_prefixes(self):
        """Every evidence ID in truth.json must exist in public tables and start with doc_/row_/fx_."""
        allowed_prefixes = ("doc_", "row_", "fx_")

        # 1. Check freshly generated closed episodes across all 6 error types + 2 interactions.
        for seed in (401, 402, 403):
            ep = generate(seed, MIXED_SECRET, key=SECRET_KEY)
            pub = _public_evidence_ids(ep)
            self.assertTrue(ep.issues)
            for issue in ep.issues:
                self.assertTrue(issue.evidence_sets)
                for ev_set in issue.evidence_sets:
                    self.assertTrue(ev_set)
                    for ev_id in ev_set:
                        self.assertTrue(ev_id.startswith(allowed_prefixes), f"disallowed prefix: {ev_id}")
                        self.assertIn(ev_id, pub, f"{ev_id} missing from public tables")

        # 2. Check all 20 committed data/answers/**/truth.json files against data/episodes/.
        truth_files = sorted((DATA / "answers").rglob("truth.json"))
        self.assertEqual(len(truth_files), 20)
        for tf in truth_files:
            rel = tf.parent.relative_to(DATA / "answers")
            ep_dir = DATA / "episodes" / rel
            pub = set()
            for name in ("invoices.jsonl", "payments.jsonl"):
                for line in (ep_dir / name).read_text(encoding="utf-8").splitlines():
                    pub.add(json.loads(line)["doc_id"])
            with (ep_dir / "ledger.csv").open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    pub.add(row["row_id"])
            with (ep_dir / "fx.csv").open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    pub.add(row["fx_id"])
            truth = json.loads(tf.read_text(encoding="utf-8"))
            for issue in truth["issues"]:
                self.assertTrue(issue["evidence_sets"])
                for ev_set in issue["evidence_sets"]:
                    for ev_id in ev_set:
                        self.assertTrue(ev_id.startswith(allowed_prefixes), f"{tf}: bad prefix {ev_id}")
                        self.assertIn(ev_id, pub, f"{tf}: {ev_id} not in public tables")

    def test_closed_episode_all_observations_never_leak_truth_or_seed(self):
        """Every observation across a closed episode (initial, tools, errors, submit, truncated) must be leak-free."""
        seed = 2026099
        ep = generate(seed, MIXED_SECRET, key=SECRET_KEY)
        env = AuditEnv(training=False, key=SECRET_KEY)
        obs0, info0 = env.load_episode(ep)
        collected = [(obs0, info0)]

        # Exercise all 4 read/calc tools + bounded error paths + submit + post-submit error.
        actions = [
            {"type": "list_records", "table": "ledger", "limit": 20},
            {"type": "list_records", "table": "invoices", "limit": 10},
            {"type": "list_records", "table": "payments", "limit": 10},
            {"type": "list_records", "table": "fx", "limit": 10},
            {"type": "get_record", "table": "invoices", "id": ep.tables["invoices"][0]["invoice_id"]},
            {"type": "get_document", "doc_id": ep.tables["invoices"][0]["doc_id"]},
            {"type": "calculator", "expression": "100 + 25.5"},
            {"type": "calculator", "expression": "bad()"},
            {"type": "get_record", "table": "invoices", "id": "inv_missing"},
            {"type": "nonexistent_tool"},
            {"type": "submit", "findings": {"episode_id": obs0["manifest"]["episode_id"], "findings": []}},
            {"type": "calculator", "expression": "1+1"},
        ]
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            collected.append((obs, info))
            if action["type"] == "submit":
                self.assertTrue(terminated)
                self.assertIsNone(reward)

        # Also inspect a truncated observation (timeout).
        env_t = AuditEnv(training=False, timeout_seconds=0.005, key=SECRET_KEY)
        env_t.load_episode(ep)
        time.sleep(0.015)
        obs_t, rew_t, _, trunc_t, info_t = env_t.step({"type": "calculator", "expression": "1+1"})
        self.assertTrue(trunc_t)
        self.assertIsNone(rew_t)
        collected.append((obs_t, info_t))

        forbidden_tokens = [
            "issue_id",
            "expected_amount_minor",
            "evidence_sets",
            "provenance",
            str(seed),
            *(issue.issue_id for issue in ep.issues),
        ]
        for obs, info in collected:
            blob = json.dumps({"obs": obs, "info": info}, ensure_ascii=False)
            for token in forbidden_tokens:
                self.assertNotIn(token, blob, f"leaked {token!r} in closed observation")

    def test_cross_process_reset_step_reproducibility_subprocess(self):
        """Two independent Python processes with identical seed/config/actions yield byte-identical traces."""
        script = (
            "import json, sys; "
            "from audit_env.contracts import Config; "
            "from audit_env.environment import AuditEnv; "
            "cfg = Config(keyspace='secret', errors={'duplicate_invoice': 1, 'fx_rate': 1}); "
            "env = AuditEnv(training=True); "
            "obs, info = env.reset(777, cfg); "
            "trace = [{'obs': obs, 'info': info}]; "
            "p1 = env.step({'type': 'list_records', 'table': 'invoices', 'limit': 5}); "
            "trace.append({'step': p1}); "
            "inv_id = p1[0]['result']['records'][0]['invoice_id']; "
            "doc_id = p1[0]['result']['records'][0]['doc_id']; "
            "trace.append({'step': env.step({'type': 'get_record', 'table': 'invoices', 'id': inv_id})}); "
            "trace.append({'step': env.step({'type': 'get_document', 'doc_id': doc_id})}); "
            "trace.append({'step': env.step({'type': 'calculator', 'expression': '12500.5 * 4'})}); "
            "trace.append({'step': env.step({'type': 'get_record', 'table': 'fx', 'id': 'fx_bad'})}); "
            "trace.append({'step': env.step({'type': 'submit', 'findings': {'episode_id': obs['manifest']['episode_id'], 'findings': []}})}); "
            "sys.stdout.write(json.dumps(trace, ensure_ascii=False, sort_keys=True))"
        )
        outputs = []
        for hash_seed in ("0", "98765"):
            env = {
                **os.environ,
                "PYTHONHASHSEED": hash_seed,
                "PYTHONIOENCODING": "utf-8",
                "AUDIT_ENV_SECRET": "cross-process-repro-secret-32b!!",
            }
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            outputs.append(proc.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_committed_model_and_ledger_csv_integrity(self):
        """Committed models/model.json must exist and load; all 20 data/episodes ledger.csv files must be balanced."""
        model_path = REPO_ROOT / "models" / "model.json"
        self.assertTrue(model_path.is_file())
        self.assertIsInstance(MicroModel.load(model_path), MicroModel)

        ledgers = sorted((DATA / "episodes").rglob("ledger.csv"))
        self.assertEqual(len(ledgers), 20)
        required = {"row_id", "timestamp", "account", "debit_minor", "credit_minor",
                    "currency", "entity_id", "invoice_id", "payment_id"}
        for path in ledgers:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertTrue(required.issubset(set(reader.fieldnames or ())))
                balance = 0
                for row in reader:
                    balance += int(row["debit_minor"]) - int(row["credit_minor"])
                self.assertEqual(balance, 0, f"{path} is not balanced")


if __name__ == "__main__":
    unittest.main()
