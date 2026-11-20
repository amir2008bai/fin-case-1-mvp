"""Reproducibility and generalization: exact replay, keyed seeds, paired ID/order checks, file round trips."""
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from audit_env.agents import create_policy, oracle
from audit_env.contracts import Config
from audit_env.dataset import checksums
from audit_env.environment import AuditEnv
from audit_env.episode_io import load_episode, write_episode
from audit_env.generator import generate, transform
from audit_env.runner import run_policy

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})
ROOT = Path(__file__).resolve().parents[1]


def run(episode, policy):
    env = AuditEnv()
    start, _ = env.load_episode(episode)
    return run_policy(env, start, policy)


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_config_and_actions_replay_exactly(self):
        first, second = generate(77, MIXED), generate(77, MIXED)
        self.assertEqual(first.tables, second.tables)
        self.assertEqual(first.issues, second.issues)
        self.assertEqual(first.documents, second.documents)
        _, trace_a, _ = run(first, create_policy("expert"))
        _, trace_b, _ = run(second, create_policy("expert"))
        self.assertEqual(trace_a, trace_b)

    def test_files_are_byte_identical_across_hash_seeds(self):
        script = ("import sys; from audit_env.generator import generate; from audit_env.contracts import Config; "
                  "from audit_env.episode_io import write_episode; from audit_env.dataset import checksums; "
                  "e = generate(5, Config(keyspace='open', errors={'fx_rate': 2, 'period_cutoff': 1})); "
                  "write_episode(e, sys.argv[1] + '/p', sys.argv[1] + '/a'); sys.stdout.write(checksums(sys.argv[1]))")
        outputs = []
        for hash_seed in ("0", "12345"):
            with tempfile.TemporaryDirectory() as temp:
                env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
                result = subprocess.run([sys.executable, "-c", script, temp], cwd=ROOT, env=env,
                                        capture_output=True, text=True, encoding="utf-8", check=True)
                outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_key_controls_everything(self):
        secret = replace(MIXED, keyspace="secret")
        a = generate(5, secret, key=b"key-a")
        b = generate(5, secret, key=b"key-b")
        self.assertNotEqual(a.manifest["episode_id"], b.manifest["episode_id"])
        self.assertNotEqual(a.tables["ledger"], b.tables["ledger"])
        self.assertEqual(generate(5, secret, key=b"key-a").tables, a.tables)

    def test_presentation_seed_pairs_keep_the_answer(self):
        base = generate(21, MIXED)
        pair = generate(21, replace(MIXED, presentation_seed=9))
        self.assertNotEqual({row["row_id"] for row in base.tables["ledger"]},
                            {row["row_id"] for row in pair.tables["ledger"]})
        self.assertEqual(sorted((i.type, i.expected_amount_minor) for i in base.issues),
                         sorted((i.type, i.expected_amount_minor) for i in pair.issues))
        first, _, _ = run(base, oracle(base))
        second, _, _ = run(pair, oracle(pair))
        self.assertEqual((first.f1, first.reward), (second.f1, second.reward))

    def test_transform_renames_and_reorders(self):
        for seed in (1, 2, 3):
            episode = generate(seed, MIXED)
            mirror = transform(episode, 100 + seed)
            self.assertNotEqual(episode.tables["ledger"][0]["row_id"], mirror.tables["ledger"][0]["row_id"])
            self.assertNotEqual(episode.manifest["episode_id"], mirror.manifest["episode_id"])
            first, _, _ = run(episode, oracle(episode))
            second, _, _ = run(mirror, oracle(mirror))
            self.assertEqual((first.f1, first.reward), (second.f1, second.reward))
            third, _, _ = run(mirror, create_policy("expert"))
            self.assertEqual(third.f1, 1.0)

    def test_file_round_trip_preserves_the_episode(self):
        episode = generate(8, MIXED)
        with tempfile.TemporaryDirectory() as temp:
            write_episode(episode, Path(temp) / "public", Path(temp) / "answers")
            text = (Path(temp) / "public" / "ledger.csv").read_bytes()
            self.assertNotIn(b"\r\n", text)
            loaded = load_episode(Path(temp) / "public", Path(temp) / "answers")
            again = Path(temp) / "again"
            write_episode(loaded, again / "public", again / "answers")
            for name in ("ledger.csv", "fx.csv", "invoices.jsonl", "payments.jsonl", "documents.jsonl"):
                self.assertEqual((Path(temp) / "public" / name).read_bytes(), (again / "public" / name).read_bytes())
        self.assertEqual(loaded.tables, episode.tables)
        self.assertEqual(loaded.issues, episode.issues)
        first, _, _ = run(episode, oracle(episode))
        second, _, _ = run(loaded, oracle(loaded))
        self.assertEqual((first.f1, first.reward), (second.f1, second.reward))

    def test_checksums_are_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ("a", "b"):
                write_episode(generate(4, MIXED), Path(temp) / name / "p", Path(temp) / name / "s")
            self.assertEqual(checksums(Path(temp) / "a"), checksums(Path(temp) / "b"))


if __name__ == "__main__":
    unittest.main()
