"""The learned micro model: public features only, save/load, beats the simple reconciliation on held-out data."""
import json
import tempfile
import unittest
from pathlib import Path

from audit_env.agents import create_policy
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.micro import MicroModel, candidates
from audit_env.runner import run_policy
from audit_env.training import train

ROOT = Path(__file__).resolve().parents[1]
ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}


class MicroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.model = train(ROOT / "configs/suite.v2.json", cls.temp.name, seeds_per_config=3)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_candidates_use_only_public_tables(self):
        episode = generate(3, Config(keyspace="open", errors=ERRORS))
        found = candidates(episode.tables, episode.manifest, episode.policy)
        public = {value for rows in episode.tables.values() for row in rows for value in row.values()
                  if isinstance(value, str)}
        for candidate in found:
            self.assertTrue(set(candidate.evidence) <= public)
            self.assertTrue(set(candidate.affected) <= public)

    def test_save_load_round_trip(self):
        path = Path(self.temp.name) / "copy.json"
        self.model.save(path)
        loaded = MicroModel.load(path)
        self.assertEqual((loaded.weights, loaded.threshold), (self.model.weights, self.model.threshold))
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            MicroModel.load(path)

    def test_beats_reconciliation_on_held_out_seeds(self):
        rewards = {"micro": [], "reconcile": []}
        for seed in range(900, 906):
            episode = generate(seed, Config(keyspace="secret", errors=ERRORS,
                                            interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 0}),
                               key=b"held-out-test-key")
            for name, policy in (("micro", self.model.policy()), ("reconcile", create_policy("reconcile"))):
                env = AuditEnv(training=False)
                start, _ = env.load_episode(episode)
                outcome, _, _ = run_policy(env, start, policy)
                rewards[name].append(outcome.reward)
        self.assertGreater(sum(rewards["micro"]) / 6, sum(rewards["reconcile"]) / 6 + 0.5)

    def test_committed_model_loads(self):
        path = ROOT / "models" / "model.json"
        self.assertTrue(path.is_file(), "models/model.json is part of the submission")
        self.assertIsInstance(MicroModel.load(path), MicroModel)


if __name__ == "__main__":
    unittest.main()
