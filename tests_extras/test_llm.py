import json
import tempfile
import unittest
from pathlib import Path

from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.hf_train import verify_dataset
from audit_env.llm import decode_action, rollout
from audit_env.llm_data import export_llm, generated_cases, teacher_trace
from audit_env.monitor import EpisodeMonitor
from audit_env.rag_env import RagAuditEnv


class LlmTests(unittest.TestCase):
    def setUp(self):
        self.episode = generate(351, Config(duplicate_count=3, allocation_count=4))

    def test_rag_obeys_scope_limits_and_source_tracking(self):
        env = RagAuditEnv(training=False)
        env.load_episode(self.episode)
        issue = self.episode.issues[0]
        subject = issue.affected_ids[0]
        obs, *_ = env.step({"type": "group_ledger", "by": "invoice_id", "account": "expense", "min_count": 2})
        self.assertIn(subject, {row["id"] for row in obs["result"]["groups"]})
        self.assertEqual(env.seen, set())
        response, *_ = env.step({
            "type": "search_records", "table": "ledger", "query": "expense",
            "link": {"field": "invoice_id", "id": subject}, "limit": 3,
        })
        assert response["result"]["records"]
        self.assertTrue(all(row["invoice_id"] == subject and row["account"] == "expense"
                            for row in response["result"]["records"]))
        self.assertEqual(env.seen, {row["evidence_id"] for row in response["result"]["sources"]})
        invalid, *_ = env.step({"type": "search_records", "query": "expense", "limit": 50})
        self.assertEqual(invalid["error"], "invalid_limit")
        original = AuditEnv()
        original.load_episode(self.episode)
        unavailable, *_ = original.step({"type": "search_records", "query": "expense"})
        self.assertEqual(unavailable["error"], "invalid_action")
        env.load_episode(generate(99, Config()))
        stale, *_ = env.step({
            "type": "search_records", "table": "ledger",
            "link": {"field": "invoice_id", "id": subject},
        })
        self.assertEqual(stale["result"]["records"], [])

    def test_teacher_trace_replays_through_json_action_gateway(self):
        score, trace = teacher_trace(self.episode)
        self.assertEqual(score.tp, 7)
        actions = iter(row["action"] for row in trace[1:])
        env = RagAuditEnv(training=False)
        obs, _ = env.load_episode(self.episode)
        monitor = EpisodeMonitor()

        def policy(messages):
            self.assertFalse(any("expected_amount_minor" in message["content"] for message in messages))
            return json.dumps(next(actions))

        result = rollout(env, obs, policy, observer=monitor)
        self.assertEqual(score.reward, result.reward)
        self.assertEqual(monitor.report()["flags"], {})
        self.assertEqual(monitor.report()["actions"]["search_records"], 14)
        self.assertIsNone(decode_action('{"type":"a","type":"b"}'))
        self.assertIsNone(decode_action('{"type":"a"} trailing'))
        self.assertIsNone(decode_action("["))

    def test_export_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            first = export_llm(out / "a", generated_cases(81, 8, 4))
            second = export_llm(out / "b", generated_cases(81, 8, 4))
            self.assertEqual(first, second)

            def read(split, kind):
                return [json.loads(line) for line in (out / "a" / f"{split}.{kind}.jsonl").read_text().splitlines()]

            train, dev = read("train", "rollouts"), read("dev", "rollouts")
            self.assertFalse({row["episode_id"] for row in train} & {row["episode_id"] for row in dev})
            self.assertTrue(all(row["reward"] > 0 for row in train + dev))
            self.assertTrue(all("seed" not in row for row in train + dev))
            samples = read("train", "sft")
            self.assertGreater(len(samples), 20)
            self.assertTrue(all(row["completion"][0]["role"] == "assistant" for row in samples))
            self.assertTrue(all("ground_truth" not in message["content"] for row in samples
                                for message in row["prompt"]))
            self.assertTrue(read("train", "preferences"))


    def test_training_rejects_modified_data_and_dev_roster_is_stable(self):
        def dev_seeds(train_count):
            return [seed for split, seed, _ in generated_cases(81, train_count, 4) if split == "dev"]

        self.assertEqual(dev_seeds(2), dev_seeds(20))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "data"
            export_llm(out, generated_cases(81, 2, 2))
            verify_dataset(out)
            with (out / "train.sft.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_dataset(out)
