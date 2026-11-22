"""Tests for RLVR standards: Gymnasium #1111 reset isolation, offline rescore/pass^k, and API import_root."""
import tempfile
import unittest
from pathlib import Path

from audit_env.api import App
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.episode_io import write_episode
from audit_env.generator import generate
from audit_env.rescore import pass_at_k, rescore_trajectories

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})


class RLVRStandardsTests(unittest.TestCase):
    def test_reset_determinism_after_intervening_noisy_episode(self):
        """Gymnasium #1111: reset(42) -> step must match bit-for-bit after an intervening episode on seed 99."""
        env = AuditEnv(training=True)
        obs1, _ = env.reset(42, MIXED)
        step1 = env.step({"type": "list_records", "table": "ledger", "limit": 25})
        # Run an intervening noisy episode with errors, calculator, and submit on the same AuditEnv instance
        obs_mid, _ = env.reset(99, Config(keyspace="open", difficulty="hard", errors=ERRORS))
        env.step({"type": "calculator", "expression": "123.45 * 6.78"})
        env.step({"type": "unknown_bad_action"})
        env.step({"type": "submit", "findings": {"episode_id": obs_mid["manifest"]["episode_id"], "findings": []}})
        # Replay seed 42 on the same instance
        obs2, _ = env.reset(42, MIXED)
        step2 = env.step({"type": "list_records", "table": "ledger", "limit": 25})
        self.assertEqual(obs1, obs2)
        self.assertEqual(step1, step2)

    def test_unbiased_pass_k_math(self):
        self.assertEqual(pass_at_k(5, 5, 1), 1.0)
        self.assertEqual(pass_at_k(5, 5, 3), 1.0)
        self.assertEqual(pass_at_k(5, 0, 1), 0.0)
        self.assertAlmostEqual(pass_at_k(5, 3, 2), 3 / 10, places=9)
        self.assertIsNone(pass_at_k(2, 2, 3))

    def test_offline_rescore_on_committed_trajectories(self):
        result = rescore_trajectories("data/trajectories.jsonl", "data/episodes", "data/answers")
        self.assertEqual(result["episodes"], 20)
        self.assertEqual(result["missing_answers"], [])
        self.assertEqual(result["mean_common_f1"], 1.0)
        self.assertEqual(result["fail_to_pass_rate"], 1.0)
        self.assertEqual(result["pass_to_pass_rate"], 1.0)
        self.assertEqual(result["pass_k"]["pass^1"], 1.0)
        self.assertEqual(result["pass_k"]["pass^3"], 1.0)

    def test_api_import_root_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            inside = root / "ep1"
            outside = Path(tmp) / "outside"
            ep = generate(7, Config(keyspace="open", errors={"duplicate_invoice": 1, "payment_allocation": 0,
                                                             "fx_rate": 0, "period_cutoff": 0}))
            write_episode(ep, inside)
            write_episode(ep, outside)
            app = App(import_root=root)
            imported = app.import_folder({"folder": str(inside)})
            self.assertIn("session_id", imported)
            with self.assertRaises(ValueError):
                app.import_folder({"folder": str(outside)})
            with self.assertRaises(ValueError):
                app.audit({"folder": str(outside)})


if __name__ == "__main__":
    unittest.main()
