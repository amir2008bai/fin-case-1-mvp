"""Contract and isolation: reset/step shape, limits, bounded errors, termination, no leaks."""
import json
import unittest

from audit_env.agents import create_policy, oracle, submission
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import collect_ids, generate
from audit_env.runner import run_policy

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})
CLEAN = Config(keyspace="open", errors={key: 0 for key in ERRORS})


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.env = AuditEnv()
        self.observation, self.info = self.env.reset(31, MIXED)
        self.episode_id = self.observation["manifest"]["episode_id"]

    def test_reset_observation(self):
        self.assertEqual(self.info, {})
        self.assertLessEqual({"task", "manifest", "policy", "remaining_steps"}, set(self.observation))
        self.assertEqual(self.observation["remaining_steps"], 100)
        manifest = self.observation["manifest"]
        for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                    "counts", "document_count", "minor_units", "rounding"):
            self.assertIn(key, manifest)
        json.dumps(self.observation)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"unknown": 1})
        with self.assertRaises(ValueError):
            Config(errors={"duplicate_invoice": 7})
        with self.assertRaises(ValueError):
            Config(errors={k: 3 for k in ERRORS}, interactions={"fx_boundary_payment": 3})
        with self.assertRaises(ValueError):
            Config(errors={"made_up": 1})
        with self.assertRaises(ValueError):
            generate(-1, MIXED)

    def test_step_shape_and_detached_results(self):
        action = {"type": "list_records", "table": "ledger", "limit": 3}
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))
        pristine = observation["result"]["records"][0]["account"]
        observation["result"]["records"][0]["account"] = "damaged"
        again = self.env.step(action)[0]
        self.assertEqual(again["result"]["records"][0]["account"], pristine)
        self.assertEqual(again["remaining_steps"], 98)

    def test_pagination_is_stable_and_complete(self):
        for table, field in (("invoices", "invoice_id"), ("payments", "payment_id"), ("fx", "fx_id")):
            ids, cursor = [], None
            while True:
                result = self.env.step({"type": "list_records", "table": table, "limit": 50, "cursor": cursor})[0]["result"]
                self.assertLessEqual(len(result["records"]), 50)
                ids.extend(row[field] for row in result["records"])
                cursor = result["next_cursor"]
                if cursor is None:
                    break
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), self.observation["manifest"]["counts"][table])

    def test_get_record_and_document(self):
        invoice = self.env.step({"type": "list_records", "table": "invoices", "limit": 1})[0]["result"]["records"][0]
        record = self.env.step({"type": "get_record", "table": "invoices", "id": invoice["invoice_id"]})[0]["result"]
        self.assertEqual(record, invoice)
        document = self.env.step({"type": "get_document", "doc_id": invoice["doc_id"]})[0]["result"]
        self.assertEqual(document["invoice_id"], invoice["invoice_id"])
        self.assertIn("line_items", document)
        self.assertEqual(document["doc_type"], "invoice")

    def test_bounded_errors_cost_a_step(self):
        actions = [
            {"type": "list_records", "table": "ledger", "limit": 51},
            {"type": "list_records", "table": "ledger", "limit": 0},
            {"type": "list_records", "table": "ledger", "limit": True},
            {"type": "list_records", "table": []},
            {"type": "list_records", "table": "secret"},
            {"type": "list_records", "table": "ledger", "cursor": "invalid"},
            {"type": "list_records", "table": "ledger", "extra": 1},
            {"type": "get_record", "table": "ledger", "id": "unknown"},
            {"type": "get_document", "doc_id": {}},
            {"type": "calculator", "expression": "2**8"},
            {"type": "calculator", "expression": "1/0"},
            {"type": "calculator", "expression": "__import__('os')"},
            {"type": "calculator", "expression": "1e9"},
            {"type": "shell", "command": "ls"},
            "not an object",
            None,
        ]
        for index, action in enumerate(actions, start=1):
            with self.subTest(action=action):
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.assertEqual(set(observation), {"error", "remaining_steps"})
                self.assertEqual(observation["remaining_steps"], 100 - index)
                self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))

    def test_calculator(self):
        cases = {"0.1 + 0.2": "0.3", "(2 + 3) * 4": "20", "10 / 4": "2.5", "-5 + 2": "-3",
                 "4800000 * 512.34": "2459232000", "123456 - 0.01": "123455.99"}
        for expression, expected in cases.items():
            self.assertEqual(self.env.step({"type": "calculator", "expression": expression})[0]["result"], expected)

    def test_submit_terminates_and_later_steps_are_bounded(self):
        observation, reward, terminated, truncated, _ = self.env.step(submission(self.episode_id, []))
        self.assertEqual((terminated, truncated, observation["status"]), (True, False, "submitted"))
        self.assertEqual(reward, -1.0)
        after = self.env.step(submission(self.episode_id, []))
        self.assertEqual(after[0]["error"], "episode_finished")
        self.assertEqual(after[2:4], (True, False))
        self.assertEqual(self.env.outcome.reward, -1.0)

    def test_step_before_reset_is_bounded(self):
        observation, reward, terminated, truncated, _ = AuditEnv().step({"type": "list_records", "table": "fx"})
        self.assertEqual(observation["error"], "reset_required")
        self.assertEqual((reward, terminated, truncated), (0.0, False, True))

    def test_submit_on_the_last_step(self):
        for _ in range(99):
            self.env.step(None)
        _, _, terminated, truncated, _ = self.env.step(submission(self.episode_id, []))
        self.assertEqual((terminated, truncated), (True, False))
        self.assertEqual(self.env.outcome.inspection_steps, 99)

    def test_step_limit_scores_an_empty_answer(self):
        env = AuditEnv()
        env.reset(32, CLEAN)
        for _ in range(100):
            observation, reward, terminated, truncated, _ = env.step(None)
        self.assertEqual((terminated, truncated, observation["findings"]), (False, True, []))
        self.assertAlmostEqual(reward, 1 - 0.002 * 100)
        self.assertEqual(env.outcome.termination, "step_limit")
        for _ in range(100):
            observation, reward, terminated, truncated, _ = self.env.step(None)
        self.assertEqual(reward, -1.0)

    def test_timeout(self):
        now = [0.0]
        env = AuditEnv(training=False, timeout_seconds=1, clock=lambda: now[0])
        env.reset(31, MIXED)
        env.step({"type": "list_records", "table": "fx"})
        now[0] = 2.0
        observation, reward, terminated, truncated, _ = env.step(submission(self.episode_id, []))
        self.assertEqual((reward, terminated, truncated, observation["findings"]), (None, False, True, []))
        self.assertEqual(env.outcome.termination, "timeout")

    def test_closed_mode_hides_reward_until_the_end(self):
        episode = generate(31, MIXED)
        env = AuditEnv(training=False)
        start, _ = env.load_episode(episode)
        outcome, trace, _ = run_policy(env, start, oracle(episode))
        self.assertEqual(outcome.f1, 1.0)
        self.assertIsNone(trace[-1]["reward"])
        self.assertTrue(all(step["reward"] == 0 for step in trace[:-1]))
        self.assertEqual(set(trace[-1]["observation"]), {"status", "remaining_steps"})

    def test_no_private_data_in_observations(self):
        episode = generate(33, MIXED)
        secret = {issue.issue_id for issue in episode.issues} | {episode.provenance["key_fingerprint"]}
        for name in ("expert", "reconcile", "random", "flag-all"):
            env = AuditEnv(training=False)
            start, _ = env.load_episode(episode)
            _, trace, _ = run_policy(env, start, create_policy(name))
            for step in trace:
                found = set()
                collect_ids(step["observation"], found)
                self.assertFalse(found & secret)
                text = json.dumps(step["observation"], ensure_ascii=False)
                for word in ("evidence_sets", "expected_amount", "provenance", "issue_id", "relevant_ids", "\"seed\""):
                    self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
