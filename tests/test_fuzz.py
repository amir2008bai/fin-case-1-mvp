"""Deterministic fuzzing & boundary tests for AuditEnv, calculator, scoring, and strict JSON parsing."""
import json
import random
import unittest

from audit_env.contracts import MAX_AMOUNT_MINOR, Config
from audit_env.environment import AuditEnv, calculate
from audit_env.generator import generate
from audit_env.runner import strict_loads
from audit_env.scoring import materiality_reward, score

MIXED = Config(keyspace="open", errors={"duplicate_invoice": 1, "payment_allocation": 1,
                                        "fx_rate": 1, "period_cutoff": 1})


class FuzzAndBoundaryTests(unittest.TestCase):
    def test_out_of_range_claimed_amount_is_rejected(self):
        episode = generate(17, MIXED)
        ep_id = episode.manifest["episode_id"]
        finding = episode.issues[0].as_finding()
        for bad_amount in (10**5000, -(10**5000), MAX_AMOUNT_MINOR + 1, -(MAX_AMOUNT_MINOR + 1), True, False, 1.0, "10"):
            payload = {"episode_id": ep_id, "findings": [finding | {"claimed_amount_minor": bad_amount}]}
            result = score(payload, ep_id, episode.issues, set(finding["evidence_ids"]), 1)
            self.assertEqual(result.reward, -1.0, f"failed to reject {type(bad_amount).__name__}")
            self.assertEqual(result.error, "invalid_submission")

    def test_calculator_fuzz_never_executes_code_or_hangs(self):
        rng = random.Random(2026)
        alphabet = "0123456789.+-*/() \t\nabc;[]{}'\"\\"
        adversarial = [
            "", " ", ".", "+.", "-.", "1..2", "1/0", "0/0", "2**10000", "1e9999",
            "__import__('os').system('id')", "lambda: 1", "1;2", "9" * 256,
            "(" * 128 + "1" + ")" * 128, "(" * 300 + "1" + ")" * 300,
            "1 / 0.00000000000000000000000000000000000000000000000000001",
            "9" * 250 + "*" + "9" * 5,
        ]
        for _ in range(300):
            length = rng.randint(1, 64)
            adversarial.append("".join(rng.choice(alphabet) for _ in range(length)))
        for expr in adversarial:
            try:
                out = calculate(expr)
                self.assertIsInstance(out, str)
                self.assertLessEqual(len(out), 600)
            except (ValueError, SyntaxError, ArithmeticError, RecursionError):
                pass

    def test_env_step_fuzz_always_bounded_and_json_serializable(self):
        rng = random.Random(99)
        env = AuditEnv(training=True)
        obs, _ = env.reset(21, MIXED)
        ep_id = obs["manifest"]["episode_id"]
        malformed = [
            None, 0, True, "list_records", [], {}, {"type": None}, {"type": "eval", "code": "1"},
            {"type": "list_records", "table": "truth"}, {"type": "list_records", "table": "ledger", "limit": 0},
            {"type": "list_records", "table": "ledger", "limit": True},
            {"type": "list_records", "table": "ledger", "limit": 51},
            {"type": "list_records", "table": "ledger", "cursor": "x" * 100},
            {"type": "get_record", "table": "ledger", "id": 123},
            {"type": "get_document", "doc_id": "\x00"},
            {"type": "calculator", "expression": 123},
            {"type": "calculator", "expression": "1/0"},
        ]
        for index in range(60):
            action = rng.choice(malformed)
            step_obs, reward, terminated, truncated, info = env.step(action)
            json.dumps((step_obs, reward, terminated, truncated, info))
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIn("error", step_obs)
            self.assertEqual(step_obs["remaining_steps"], 100 - (index + 1))
        # Final valid empty submit still terminates cleanly
        _, reward, terminated, truncated, _ = env.step({"type": "submit", "findings": {"episode_id": ep_id, "findings": []}})
        self.assertTrue(terminated and not truncated)
        self.assertEqual(reward, -1.0)

    def test_strict_loads_rejects_duplicates_and_non_finite(self):
        for raw in ('{"a": 1, "a": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e500}'):
            with self.assertRaises(ValueError):
                strict_loads(raw)

    def test_cross_episode_cursor_cannot_be_reused(self):
        env = AuditEnv(training=True)
        env.reset(17, MIXED)
        cursor = env.step({"type": "list_records", "table": "ledger", "limit": 10})[0]["result"]["next_cursor"]
        self.assertIsNotNone(cursor)
        env.reset(18, MIXED)
        obs, reward, terminated, truncated, _ = env.step(
            {"type": "list_records", "table": "ledger", "limit": 10, "cursor": cursor}
        )
        self.assertEqual(obs["error"], "invalid_cursor")
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))

    def test_materiality_reward_is_separate_and_risk_aware(self):
        episode = generate(17, MIXED)
        ep_id = episode.manifest["episode_id"]
        seen = {item for issue in episode.issues for proof in issue.evidence_sets for item in proof}
        truth = [issue.as_finding() for issue in episode.issues]
        full = materiality_reward({"episode_id": ep_id, "findings": truth}, ep_id, episode.issues, seen, 10)
        self.assertEqual(full.common_reward, score({"episode_id": ep_id, "findings": truth}, ep_id, episode.issues, seen, 10).reward)
        self.assertAlmostEqual(full.materiality_f1, 1.0, places=9)
        self.assertEqual(full.tool_compliance, 1.0)
        # Missing the largest issue hurts materiality_reward more than missing the smallest issue
        by_size = sorted(episode.issues, key=lambda issue: issue.expected_amount_minor)
        if by_size[0].expected_amount_minor < by_size[-1].expected_amount_minor:
            drop_small = [issue.as_finding() for issue in by_size[1:]]
            drop_big = [issue.as_finding() for issue in by_size[:-1]]
            res_small = materiality_reward({"episode_id": ep_id, "findings": drop_small}, ep_id, episode.issues, seen, 10)
            res_big = materiality_reward({"episode_id": ep_id, "findings": drop_big}, ep_id, episode.issues, seen, 10)
            self.assertEqual(res_small.common_reward, res_big.common_reward)
            self.assertGreater(res_small.materiality_reward, res_big.materiality_reward)


if __name__ == "__main__":
    unittest.main()
