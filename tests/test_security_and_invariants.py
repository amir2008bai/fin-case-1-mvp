"""Regression tests for AppSec, Red-Team, and Accounting/World audit fixes."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audit_env import keys
from audit_env.api import local_origin
from audit_env.checks import audit_episode
from audit_env.contracts import Config, Issue
from audit_env.environment import AuditEnv
from audit_env.episode_io import FormatError, load_episode, write_episode
from audit_env.generator import generate, transform
from audit_env.provenance import restore_clean
from audit_env.scoring import materiality_reward, matches, score

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})


class SecurityAndInvariantsTests(unittest.TestCase):
    def test_agent_protocol_error_never_beats_honest_on_clean_books(self):
        """A protocol crash at step 0 on clean books must score -1.0, never +1.0."""
        env = AuditEnv(training=True)
        env.reset(18, Config(keyspace="open", errors={k: 0 for k in ERRORS}))
        _, reward, terminated, truncated, _ = env.expire("agent_protocol")
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(env.outcome.error, "agent_protocol")

    def test_materiality_reward_blocks_zero_amount_fp_flooding(self):
        """Submitting fake claimed_amount_minor=0 FPs must strictly reduce materiality_reward."""
        ep = generate(17, MIXED)
        ep_id = ep.manifest["episode_id"]
        seen = {item for issue in ep.issues for proof in issue.evidence_sets for item in proof}
        honest = [issue.as_finding() for issue in ep.issues]
        base = materiality_reward({"episode_id": ep_id, "findings": honest}, ep_id, ep.issues, seen, 10)
        fake_fp = honest[0] | {"type": "unknown_fake", "claimed_amount_minor": 0}
        flooded = materiality_reward({"episode_id": ep_id, "findings": honest + [fake_fp] * 10}, ep_id, ep.issues, seen, 10)
        self.assertLess(flooded.materiality_precision, 1.0)
        self.assertLess(flooded.materiality_reward, base.materiality_reward)

    def test_empty_relevant_ids_does_not_allow_evidence_padding(self):
        """Issue with empty relevant_ids must still reject unrelated seen IDs."""
        issue = Issue("issue_1", "fx_rate", ("invoice_1",), 100, "KZT", (("doc_1", "row_1"),), 0, ())
        finding = {"type": "fx_rate", "affected_ids": ["invoice_1"], "claimed_amount_minor": 100,
                   "currency": "KZT", "evidence_ids": ["doc_1", "row_1", "row_unrelated"], "explanation": "x"}
        self.assertFalse(matches(finding, issue, {"doc_1", "row_1", "row_unrelated"}))

    def test_transform_and_restore_clean_commute(self):
        """MR-6: restore_clean(transform(ep, seed)) must succeed and match clean_ledger_sha256."""
        ep = generate(17, Config(keyspace="open", errors=ERRORS | {"vat_on_top": 1}))
        mirror = transform(ep, 7777)
        clean = restore_clean(mirror)
        self.assertEqual(audit_episode(clean), [])
        # Also verify vat_on_top document fields were restored in clean.documents
        for inv in clean.tables["invoices"]:
            doc = clean.documents.get(inv["doc_id"])
            if doc is not None:
                self.assertEqual(doc["total_minor"], inv["total_minor"])

    def test_honest_fx_citation_accepted_on_interacting_errors(self):
        """MR-8: Citing the payment's fx_id on fx_boundary_payment / fx_partial_allocation must still match."""
        ep = generate(17, MIXED)
        seen = {item for issue in ep.issues for item in issue.relevant_ids}
        fx_by_pay = {}
        for issue in ep.issues:
            if issue.type == "fx_rate" and issue.affected_ids[0].startswith("payment_"):
                fx_ids = [x for x in issue.evidence_sets[0] if x.startswith("fx_")]
                if fx_ids:
                    fx_by_pay[issue.affected_ids[0]] = fx_ids[0]
        checked = 0
        for issue in ep.issues:
            if issue.type in ("period_cutoff", "payment_allocation") and issue.affected_ids[0] in fx_by_pay:
                finding = issue.as_finding()
                fx_id = fx_by_pay[issue.affected_ids[0]]
                if fx_id not in finding["evidence_ids"]:
                    finding["evidence_ids"] = finding["evidence_ids"] + [fx_id]
                self.assertTrue(matches(finding, issue, seen), f"{issue.type} rejected valid fx_id")
                checked += 1
        self.assertGreater(checked, 0)

    def test_duplicate_invoice_no_minute_second_clone_or_out_of_hours(self):
        """Duplicate invoices must stay in [10:00..18:59] and not clone exact (minute, second)."""
        clones = 0
        out_of_hours = 0
        for seed in range(1, 21):
            ep = generate(seed, Config(keyspace="open", errors={"duplicate_invoice": 2, "payment_allocation": 0,
                                                                "fx_rate": 0, "period_cutoff": 0}))
            by_id = {r["invoice_id"]: r for r in ep.tables["invoices"]}
            for issue in ep.issues:
                a, b = (by_id[i] for i in issue.affected_ids)
                if a["issued_at"][14:19] == b["issued_at"][14:19]:
                    clones += 1
                for item in (a, b):
                    hour = int(item["issued_at"][11:13])
                    if not (10 <= hour <= 18):
                        out_of_hours += 1
        self.assertEqual(out_of_hours, 0)
        self.assertLess(clones, 5)

    def test_high_filler_keeps_non_negative_ending_cash(self):
        """Ending 1030 Bank balance must remain >= 0 even at filler=800."""
        ep = generate(55, Config(keyspace="open", filler=800, errors={k: 0 for k in ERRORS}))
        bank = sum(row["debit_minor"] - row["credit_minor"] for row in ep.tables["ledger"] if row["account"] == "1030")
        self.assertGreaterEqual(bank, 0)

    def test_episode_io_rejects_infinity_fx_and_non_dict_manifest(self):
        ep = generate(17, MIXED)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "ep"
            write_episode(ep, folder)
            # Infinity FX rate
            fx_lines = (folder / "fx.csv").read_text(encoding="utf-8").splitlines()
            if len(fx_lines) > 1:
                parts = fx_lines[1].split(",")
                parts[fx_lines[0].split(",").index("rate")] = "Infinity"
                fx_lines[1] = ",".join(parts)
                (folder / "fx.csv").write_text("\n".join(fx_lines) + "\n", encoding="utf-8")
                with self.assertRaises(FormatError):
                    load_episode(folder)
            # Non-dict manifest
            write_episode(ep, folder)
            (folder / "manifest.json").write_text("[]\n", encoding="utf-8")
            with self.assertRaises(FormatError):
                load_episode(folder)

    def test_non_ascii_episode_id_loads_cleanly_in_env(self):
        ep = generate(17, MIXED)
        ep.manifest["episode_id"] = "эпизод_17"
        env = AuditEnv(training=True)
        obs, _ = env.load_episode(ep)
        self.assertEqual(obs["manifest"]["episode_id"], "эпизод_17")

    def test_local_origin_rejects_userinfo_query_fragment(self):
        for bad in ("http://evil@localhost", "http://localhost?x=1", "http://localhost#frag"):
            self.assertFalse(local_origin(bad), bad)

    def test_short_env_var_secret_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret.key"
            with mock.patch.dict(os.environ, {keys.ENV_VAR: "short", keys.KEY_FILE_VAR: str(path)}, clear=False):
                with mock.patch.object(keys, "_process_key", None):
                    val = keys.secret_key()
                    self.assertGreaterEqual(len(val), keys.MIN_KEY_BYTES)
                    self.assertNotEqual(val, b"short")


if __name__ == "__main__":
    unittest.main()
