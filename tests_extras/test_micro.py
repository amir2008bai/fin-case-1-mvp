import json
import tempfile
import unittest
from pathlib import Path

from audit_env.agents import ScanPolicy, submission
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate, transform
from audit_env.micro import FEATURES, MicroModel
from audit_env.monitor import EpisodeMonitor
from audit_env.rag import EvidenceIndex
from audit_env.runner import run_policy
from audit_env.training import collect_public, train

SUITE = Path(__file__).resolve().parents[1] / "configs/open-suite.v1.json"


class MicroTests(unittest.TestCase):
    def test_training_split_reload_and_closed_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = train(SUITE, root)
            first = (root / "model.json").read_bytes()
            self.assertTrue(train(SUITE, root) == report)
            self.assertEqual(first, (root / "model.json").read_bytes())
            model = MicroModel.load(root / "model.json")
            self.assertEqual(len(model.weights), len(FEATURES))
            self.assertGreater(report["dev_candidates"]["trained"]["f1"],
                               report["dev_candidates"]["untrained"]["f1"])
            self.assertEqual(report["dev_episodes_metrics"]["trained"]["mean_f1"], 1.0)
            rows = (root / "train_examples.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), report["train_examples"])
            self.assertTrue(all(json.loads(line).keys() == {"features", "label"} for line in rows))
            self.assertEqual(json.loads(first).keys(), {"version", "features", "weights", "bias", "threshold"})

            cases = [
                generate(112, Config(duplicate_count=3, allocation_count=4)),
                generate(19, Config(duplicate_count=0, allocation_count=0)),
            ]
            for episode in [*cases, transform(cases[0], 710)]:
                env = AuditEnv(training=False)
                start, _ = env.load_episode(episode)
                monitor = EpisodeMonitor()
                outcome, trace, _ = run_policy(env, start, ScanPolicy(model.findings), monitor)
                self.assertEqual(outcome.tp, len(episode.issues))
                self.assertEqual(outcome.fp, 0)
                self.assertEqual(monitor.report()["events"], len(trace))
                self.assertEqual(monitor.report()["flags"], {})
                self.assertIsNone(trace[-1]["reward"])

    def test_rag_retrieves_public_source_by_link_and_text(self):
        episode = generate(112, Config())
        public = collect_public(episode)
        index = EvidenceIndex(public)
        invoice = public["invoices"][0]
        rows = index.linked("ledger", "invoice_id", invoice["invoice_id"])
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row["invoice_id"] == invoice["invoice_id"] for row in rows))
        result = index.search(invoice["invoice_id"], table="invoices", limit=2)
        self.assertEqual([hit["evidence_id"] for hit in result], [invoice["doc_id"]])
        self.assertEqual(index.search("this_does_not_exist"), [])

    def test_monitor_flags_unseen_citations_and_repeats(self):
        episode = generate(112, Config())
        env = AuditEnv(training=False)
        start, _ = env.load_episode(episode)
        finding = episode.issues[0].as_finding()
        monitor = EpisodeMonitor()
        result, trace, _ = run_policy(
            env, start, lambda _: submission(episode.manifest["episode_id"], [finding] * 2), monitor,
        )
        self.assertEqual(result.tp, 0)
        self.assertIsNone(trace[-1]["reward"])
        self.assertEqual(monitor.report()["flags"], {
            "unobserved_citations": 2 * len(finding["evidence_ids"]), "repeat_subject": 1,
        })
        self.assertEqual(monitor.report()["submitted_findings"], 2)
        self.assertNotIn("reward", monitor.report())
