import unittest
from dataclasses import replace

from audit_env.agents import create_policy, oracle, submission
from audit_env.contracts import ALLOCATION, DUPLICATE, Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate, transform
from audit_env.runner import run_policy
from audit_env.scoring import score


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = AuditEnv()
        self.seed = 289
        self.config = Config()
        self.start, _ = self.env.reset(self.seed, self.config)
        self.episode_id = self.start["manifest"]["episode_id"]

    def test_reset_and_steps_are_reproducible_and_detached(self):
        action = {"type": "list_records", "table": "ledger", "limit": 2}
        response = self.env.step(action)
        pristine = response[0]["result"]["records"][0]["account"]
        response[0]["result"]["records"][0]["account"] = "damaged"
        again = self.env.step(action)
        self.assertEqual(again[0]["result"]["records"][0]["account"], pristine)
        self.start["manifest"]["episode_id"] = "damaged"
        restored, info = self.env.reset(self.seed, self.config)
        self.assertEqual(restored["manifest"]["episode_id"], self.episode_id)
        self.assertEqual(info, {})
        self.assertEqual(self.env.step(action)[0]["result"], again[0]["result"])
        self.assertEqual(self.env.steps, 1)

    def test_pagination_and_get_document(self):
        ids = []
        cursor = None
        while True:
            observation, reward, terminated, truncated, _ = self.env.step(
                {"type": "list_records", "table": "invoices", "limit": 17, "cursor": cursor}
            )
            result = observation["result"]
            self.assertEqual((reward, terminated, truncated), (0, False, False))
            ids.extend(row["invoice_id"] for row in result["records"])
            cursor = result["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(len(set(ids)), self.config.invoice_count)
        result = self.env.step({"type": "get_record", "table": "invoices", "id": ids[0]})
        doc = result[0]["result"]
        self.assertEqual(self.env.step({"type": "get_document", "doc_id": doc["doc_id"]})[0]["result"], doc)

    def test_errors_are_bounded_and_cost_a_step(self):
        for index, action in enumerate([
            {"type": "list_records", "table": "ledger", "limit": 51},
            {"type": "list_records", "table": []},
            {"type": "list_records", "table": "ledger", "cursor": "invalid"},
            {"type": "get_record", "table": "ledger", "id": "unknown"},
            {"type": "get_document", "doc_id": {}},
            {"type": "calculator", "expression": "2**8"},
            {"type": "calculator", "expression": "1/0"},
            {"type": "calculator", "expression": "__import__('os')"},
            {"type": "arbitrary"},
            None,
        ], start=1):
            with self.subTest(action=action):
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.assertEqual(set(observation), {"error", "remaining_steps"})
                self.assertEqual(observation["remaining_steps"], 100 - index)
                self.assertEqual((reward, terminated, truncated, info), (0, False, False, {}))
        self.assertEqual(self.env.step({"type": "calculator", "expression": "0.1 + 0.2"})[0]["result"], "0.3")

    def test_submit_on_last_step_and_repeat(self):
        for _ in range(99):
            self.env.step(None)
        result = self.env.step(submission(self.episode_id, []))
        self.assertEqual(result[1:4], (-1, True, False))
        self.assertEqual(self.env.outcome.inspection_steps, 99)
        with self.assertRaises(RuntimeError):
            self.env.step(submission(self.episode_id, []))

    def test_step_limit_and_timeout(self):
        for _ in range(100):
            last = self.env.step(None)
        self.assertEqual(last[1:4], (-1, False, True))
        self.assertEqual(last[0]["findings"], [])
        self.assertEqual(self.env.outcome.error, "step_limit")

        now = [10]
        env = AuditEnv(training=False, timeout_seconds=1, clock=lambda: now[0])
        env.reset(self.seed, self.config)
        now[0] = 11
        last = env.step(submission(self.episode_id, []))
        self.assertEqual(last[1:4], (None, False, True))
        self.assertEqual(env.outcome.error, "timeout")

    def test_closed_feedback_and_replay(self):
        episode = generate(self.seed, self.config)
        traces = []
        for _ in range(2):
            env = AuditEnv(training=False)
            start, _ = env.load_episode(episode)
            result, trace, _ = run_policy(env, start, create_policy("reconcile"))
            traces.append(trace)
            self.assertEqual(result.f1, 1.0)
            self.assertEqual(trace[-1]["reward"], None)
            self.assertEqual(trace[-1]["observation"]["status"], "submitted")
            self.assertTrue(all(step["reward"] == 0 for step in trace[:-1]))
            for step in trace:
                self.assertFalse({"truth", "seed", "issues", "provenance"} & step["observation"].keys())
        self.assertEqual(*traces)

    def test_oracle_after_order_and_id_change(self):
        episode = generate(self.seed, self.config)
        remapped = transform(episode, 171)
        results = []
        for current in (episode, remapped):
            start, _ = self.env.load_episode(current)
            result, _, _ = run_policy(self.env, start, oracle(current.issues))
            results.append((result.f1, result.reward))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], 1.0)
        self.assertNotEqual(episode.issues[0].affected_ids, remapped.issues[0].affected_ids)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.episode = generate(11, Config())
        self.episode_id = self.episode.manifest["episode_id"]
        self.issues = self.episode.issues
        self.findings = [issue.as_finding() for issue in self.issues]
        self.seen = {identifier for issue in self.issues for proof in issue.evidence_sets for identifier in proof}

    def compute(self, findings, *, issues=None, seen=None, steps=0):
        return score(
            {"episode_id": self.episode_id, "findings": findings}, self.episode_id,
            self.issues if issues is None else issues, self.seen if seen is None else seen, steps,
        )

    def test_reward_order(self):
        correct = self.compute(self.findings)
        self.assertEqual((correct.tp, correct.fp, correct.fn, correct.reward), (2, 0, 0, 1))
        self.assertLess(self.compute([]).reward, correct.reward)
        self.assertGreater(self.compute([], issues=()).reward, self.compute(self.findings, issues=()).reward)
        self.assertAlmostEqual(correct.reward - self.compute(self.findings, steps=3).reward, 0.006)
        self.assertEqual(self.compute([], issues=()).f1, 1.0)

    def test_repeat_and_guess_cannot_improve_reward(self):
        duplicate = self.compute(self.findings + self.findings)
        self.assertEqual((duplicate.tp, duplicate.fp, duplicate.fn), (2, 2, 0))
        self.assertLess(duplicate.reward, self.compute(self.findings).reward)
        self.assertEqual(self.compute(self.findings, seen=set()).tp, 0)
        self.assertEqual(self.compute([self.findings[0] | {"currency": "USD"}]).tp, 0)
        self.assertEqual(self.compute([self.findings[0] | {"evidence_ids": ["guessed"]}]).tp, 0)
        self.assertEqual(self.compute([self.findings[0] | {"affected_ids": ["guessed"]}]).tp, 0)
        self.assertEqual(self.compute(self.findings * 16).error, "invalid_submission")
        self.assertEqual(self.compute([self.findings[0] | {"claimed_amount_minor": True}]).error, "invalid_submission")

    def test_ambiguous_matching_uses_augmenting_path(self):
        first = self.issues[0]
        second = replace(first, issue_id="z", evidence_sets=(first.evidence_sets[0],))
        broad = first.as_finding()
        narrow = broad | {"evidence_ids": list(first.evidence_sets[1])}
        result = self.compute([broad, narrow], issues=(first, second))
        self.assertEqual(result.tp, 2)

    def test_mark_all_policy_scores_poorly(self):
        for config in (Config(), Config(duplicate_count=0, allocation_count=0)):
            env = AuditEnv()
            observation, _ = env.reset(11, config)
            result, _, _ = run_policy(env, observation, create_policy("mark-all"))
            self.assertEqual(result.reward, -1)
            self.assertEqual(result.fp, 30)

    def test_supported_types_independently_configurable(self):
        for config, expected in (
            (Config(duplicate_count=2, allocation_count=0), [DUPLICATE] * 2),
            (Config(duplicate_count=0, allocation_count=3), [ALLOCATION] * 3),
            (Config(duplicate_count=0, allocation_count=0), []),
        ):
            self.assertEqual([issue.type for issue in generate(11, config).issues], expected)
