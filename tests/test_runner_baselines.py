"""Agents over real pipes, deadlines, protocol errors, baselines and anti-tell checks."""
import sys
import unittest
from collections import defaultdict

from audit_env.agents import create_policy, oracle
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.runner import run_policy, run_subprocess
from audit_env.scoring import score

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})


class RunnerTests(unittest.TestCase):
    def test_agent_across_a_real_pipe(self):
        env = AuditEnv(training=False)
        observation, _ = env.reset(51, MIXED)
        outcome, trace, _ = run_subprocess(env, observation, [sys.executable, "-m", "audit_env.agents", "expert"])
        self.assertEqual(outcome.f1, 1.0)
        self.assertIsNone(trace[-1]["reward"])

    def test_deadline_and_protocol_errors(self):
        cases = (
            ("import time; time.sleep(5)", "timeout", 0.5),
            ("print('not JSON', flush=True)", "agent_protocol", 5),
            ("print('x' * 300000, flush=True)", "agent_protocol", 5),
            ('print("{\\"n\\":1e1000}", flush=True)', "agent_protocol", 5),
            ("import sys; sys.exit(0)", "agent_protocol", 5),
        )
        for script, reason, timeout in cases:
            with self.subTest(reason=reason, script=script):
                env = AuditEnv(training=False, timeout_seconds=timeout)
                observation, _ = env.reset(51, MIXED)
                outcome, trace, _ = run_subprocess(env, observation, [sys.executable, "-c", script])
                self.assertEqual(outcome.termination, reason)
                self.assertTrue(trace[-1]["truncated"])
                self.assertIsNone(trace[-1]["reward"])


class BaselineTests(unittest.TestCase):
    def test_policies_order_on_erroneous_and_clean_books(self):
        results = defaultdict(list)
        for seed in range(6):
            for config in (MIXED, Config(keyspace="open", errors={key: 0 for key in ERRORS})):
                episode = generate(seed, config)
                for name in ("oracle", "expert", "reconcile", "random", "flag-all"):
                    env = AuditEnv()
                    start, _ = env.load_episode(episode)
                    policy = oracle(episode) if name == "oracle" else create_policy(name)
                    outcome, trace, _ = run_policy(env, start, policy)
                    self.assertLessEqual(len(trace) - 1, 100)
                    results[(name, bool(episode.issues))].append(outcome)
        mean = lambda name, dirty: sum(o.reward for o in results[(name, dirty)]) / len(results[(name, dirty)])  # noqa: E731
        self.assertTrue(all(o.f1 == 1.0 for o in results[("oracle", True)] + results[("oracle", False)]))
        self.assertGreater(mean("oracle", True), mean("reconcile", True))
        self.assertGreater(mean("random", False), mean("reconcile", False))
        self.assertLessEqual(mean("flag-all", True), -0.8)
        self.assertEqual(mean("flag-all", False), -1.0)


class TellTests(unittest.TestCase):
    """Structural shortcuts must not solve the task: naive detectors stay below F1 0.5."""

    def test_naive_detectors_fail(self):
        totals = defaultdict(lambda: [0, 0, 0])
        for seed in range(40):
            episode = generate(seed, Config(keyspace="open", acts=45, customers=20, foreign_percent=30,
                                            errors={key: seed % 2 for key in ERRORS},
                                            interactions={"fx_boundary_payment": int(seed % 3 == 0),
                                                          "fx_partial_allocation": int(seed % 3 == 1)}))
            truth = defaultdict(set)
            for issue in episode.issues:
                truth[issue.type].add(issue.affected_ids)
            ledger = episode.tables["ledger"]
            guesses = {
                "payment_allocation": {(row["payment_id"],) for row in ledger
                                       if row["account"] == "3510" and row["credit_minor"] and row["payment_id"]},
                "fx_rate": {(row["payment_id"],) for row in ledger if row["account"] in ("6250", "7430")},
                "period_cutoff": {(row["payment_id"],) for row in episode.tables["payments"]
                                  if not episode.manifest["period_start"][:10] <= row["timestamp"][:10]
                                  < episode.manifest["period_end"][:10]},
            }
            by_entity = defaultdict(list)
            for row in episode.tables["invoices"]:
                by_entity[(row["entity_id"], row["total_minor"])].append(row["invoice_id"])
            guesses["duplicate_invoice"] = {tuple(sorted(ids[:2])) for ids in by_entity.values() if len(ids) > 1}
            for kind, guessed in guesses.items():
                real = truth[kind]
                totals[kind][0] += len(guessed & real)
                totals[kind][1] += len(guessed - real)
                totals[kind][2] += len(real - guessed)
        for kind, (tp, fp, fn) in totals.items():
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
            with self.subTest(kind=kind):
                self.assertLess(f1, 0.5, f"naive {kind} detector reaches F1 {f1:.2f}")


if __name__ == "__main__":
    unittest.main()
