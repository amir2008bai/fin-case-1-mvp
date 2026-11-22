"""Metamorphic invariance tests: ID/order permutations, filler scaling, timezones, causal ablation, read-only repro."""
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from audit_env.agents import create_policy, oracle
from audit_env.checks import audit_episode
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate, transform
from audit_env.gym_adapter import GymAuditEnv
from audit_env.micro import MicroModel
from audit_env.provenance import restore_clean
from audit_env.runner import run_policy

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})


def _play(episode, policy):
    env = AuditEnv(training=True)
    obs, _ = env.load_episode(episode)
    return run_policy(env, obs, policy)


class MetamorphicTests(unittest.TestCase):
    def test_transform_invariance_holds_for_oracle_expert_and_micro(self):
        """MR-1: Renaming all opaque IDs and shuffling all tables preserves F1, TP, FP, FN and reward."""
        model = MicroModel.load("models/model.json")
        episode = generate(101, MIXED)
        mirror = transform(episode, 7777)
        for name, factory_a, factory_b in (
            ("oracle", lambda: oracle(episode), lambda: oracle(mirror)),
            ("expert", lambda: create_policy("expert"), lambda: create_policy("expert")),
            ("micro", model.policy, model.policy),
        ):
            out_a, _, _ = _play(episode, factory_a())
            out_b, _, _ = _play(mirror, factory_b())
            self.assertEqual((out_a.tp, out_a.fp, out_a.fn, out_a.f1, out_a.reward),
                             (out_b.tp, out_b.fp, out_b.fn, out_b.f1, out_b.reward),
                             f"{name} drifted under ID/order transform")

    def test_filler_noise_monotonicity(self):
        """MR-2: Scaling filler rows changes only step count (0.002 per step), never TP/FP/FN/F1."""
        small = generate(202, Config(keyspace="open", filler=10, errors=ERRORS))
        large = generate(202, Config(keyspace="open", filler=350, errors=ERRORS))
        self.assertGreater(len(large.tables["ledger"]), len(small.tables["ledger"]) + 100)
        out_small, _, _ = _play(small, create_policy("expert"))
        out_large, _, _ = _play(large, create_policy("expert"))
        self.assertEqual((out_small.tp, out_small.fp, out_small.fn, out_small.f1),
                         (out_large.tp, out_large.fp, out_large.fn, out_large.f1))
        self.assertGreater(out_large.inspection_steps, out_small.inspection_steps)
        expected_delta = 0.002 * (out_large.inspection_steps - out_small.inspection_steps)
        self.assertAlmostEqual(out_small.reward - out_large.reward, expected_delta, places=9)

    def test_timezone_shift_invariance(self):
        """MR-3: Shifting company timezone (+03/+05/+06) keeps clean books clean and dirty books solvable."""
        for tz in ("+03:00", "+05:00", "+06:00"):
            ep = generate(303, replace(MIXED, timezone=tz))
            self.assertEqual(audit_episode(restore_clean(ep)), [], f"clean tz {tz} had findings")
            out, _, _ = _play(ep, create_policy("expert"))
            self.assertEqual(out.f1, 1.0, f"dirty tz {tz} failed")

    def test_causal_error_ablation(self):
        """MR-4: Removing one error type from config reduces truth count cleanly without affecting clean restore."""
        full = generate(404, Config(keyspace="open", errors=ERRORS))
        ablated = generate(404, Config(keyspace="open", errors=ERRORS | {"duplicate_invoice": 0}))
        self.assertEqual(len(full.issues), 4)
        self.assertEqual(len(ablated.issues), 3)
        self.assertNotIn("duplicate_invoice", {issue.type for issue in ablated.issues})
        self.assertEqual(audit_episode(restore_clean(ablated)), [])
        out, _, _ = _play(ablated, create_policy("expert"))
        self.assertEqual(out.f1, 1.0)

    def test_default_training_reset_is_reproducible_without_writable_key_file(self):
        code = (
            "import json; from audit_env.environment import AuditEnv; "
            "print(json.dumps(AuditEnv().reset(42), sort_keys=True))"
        )
        env = os.environ.copy()
        env.pop("AUDIT_ENV_SECRET", None)
        with tempfile.TemporaryDirectory() as root:
            blocker = Path(root) / "regular-file"
            blocker.touch()
            env["AUDIT_ENV_KEY_FILE"] = str(blocker / "secret.key")
            first = subprocess.check_output([sys.executable, "-c", code], env=env)
            second = subprocess.check_output([sys.executable, "-c", code], env=env)
        self.assertEqual(first, second)

    def test_closed_reset_preserves_secret_keyspace(self):
        key = b"test-only-closed-key"
        env = AuditEnv(training=False, key=key)
        expected = generate(42, Config(keyspace="secret"), key=key).manifest
        for config in (None, {}):
            obs, _ = env.reset(42, config)
            self.assertEqual(obs["manifest"], expected)

    def test_default_reset_reproducible_across_processes_in_readonly_home(self):
        """MR-5: Bare AuditEnv().reset(seed) must be deterministic across processes even with unwritable HOME."""
        code = (
            "from audit_env.environment import AuditEnv; "
            "obs, _ = AuditEnv().reset(42); "
            "print(obs['manifest']['episode_id'])"
        )
        env = {k: v for k, v in os.environ.items() if k not in ("AUDIT_ENV_SECRET", "AUDIT_ENV_KEY_FILE")}
        env["HOME"] = "/nonexistent_readonly_home_dir"
        first = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
        second = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
        self.assertEqual(first, second)

    def test_gym_adapter_and_inspector_use_all_five_tools(self):
        """Verify optional GymAuditEnv adapter and InspectorPolicy exercise all 5 contract tools."""
        env = GymAuditEnv(training=True)
        obs, info = env.reset(seed=17, options={"config": MIXED})
        self.assertEqual(info, {})
        policy = create_policy("inspector")
        used = set()
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(obs)
            used.add(action["type"])
            obs, reward, terminated, truncated, _ = env.step(action)
        self.assertTrue(terminated and not truncated)
        self.assertEqual(env.outcome.f1, 1.0)
        self.assertGreater(reward, 0.85)
        self.assertEqual(used, {"list_records", "get_record", "get_document", "calculator", "submit"})


if __name__ == "__main__":
    unittest.main()
