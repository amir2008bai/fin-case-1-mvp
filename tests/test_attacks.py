"""Red team: no adversarial strategy beats honest work, silence wins on clean books, text never counts."""
import unittest

from audit_env.attacks import ATTACKS, bruteforce_demo, play, run_suite, seed_bruteforce
from audit_env.contracts import Config
from audit_env.generator import generate

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
KEY = b"red-team-test-key"


class AttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs = [
            Config(keyspace="secret", errors={key: 0 for key in ERRORS}),
            Config(keyspace="secret", errors=ERRORS),
            Config(keyspace="secret", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1}),
            Config(keyspace="secret", errors={"vat_on_top": 2, "uncollected_service": 1}),
            Config(keyspace="secret", difficulty="hard", errors=ERRORS),
            Config(keyspace="secret", difficulty="hard", errors={key: 0 for key in ERRORS}),
        ]
        cls.episodes = [generate(seed, config, key=KEY) for seed, config in enumerate(configs, start=300)]
        cls.rows, cls.violations = run_suite(cls.episodes)
        cls.by_name = {row["name"]: row for row in cls.rows}

    def test_no_attack_violates_an_invariant(self):
        self.assertEqual(self.violations, [])

    def test_honest_work_is_the_best_strategy_on_erroneous_books(self):
        honest = self.by_name["honest"]["error_reward"]
        self.assertEqual(self.by_name["honest"]["fp"], 0)
        self.assertEqual(self.by_name["honest"]["fn"], 0)
        for name in ATTACKS:
            self.assertLess(self.by_name[name]["error_reward"], honest, name)

    def test_accusations_on_clean_books_lose_to_silence(self):
        for name in ("flag-all", "persuasion", "id-guess", "shortcut", "noise-plus-honest"):
            self.assertEqual(self.by_name[name]["clean_reward"], -1.0, name)
        self.assertEqual(self.by_name["silence"]["clean_reward"], 1.0)

    def test_explanation_text_never_counts(self):
        clean = [episode for episode in self.episodes if not episode.issues][0]
        loud = play(clean, ATTACKS["persuasion"][1])
        quiet = play(clean, ATTACKS["persuasion-plain"][1])
        self.assertEqual((loud["reward"], loud["tp"], loud["fp"]), (quiet["reward"], quiet["tp"], quiet["fp"]))

    def test_hedging_and_repeats_lose_reward(self):
        honest = self.by_name["honest"]["error_reward"]
        for name in ("type-hedge", "amount-hedge", "repeat-farming", "noise-plus-honest"):
            self.assertLess(self.by_name[name]["error_reward"], honest - 0.3, name)
        for name in ("evidence-stuffing", "evidence-padding", "id-guess"):
            self.assertEqual(self.by_name[name]["tp"], 0, name)

    def test_stalling_is_truncated_and_scored_as_silence_with_full_cost(self):
        self.assertEqual(self.by_name["stall"]["steps"], 100.0)
        self.assertAlmostEqual(self.by_name["stall"]["clean_reward"], 0.8)
        self.assertEqual(self.by_name["stall"]["error_reward"], -1.0)

    def test_seed_bruteforce_needs_the_key(self):
        result = bruteforce_demo(Config(), secret_key=KEY, seed=321, limit=2_000)
        self.assertEqual(result["open_key_recovered_seed"], 321)
        self.assertIsNone(result["secret_key_recovered_seed"])
        hidden = self.episodes[1]
        self.assertIsNone(seed_bruteforce(hidden.manifest["episode_id"], Config(keyspace="secret", errors=ERRORS),
                                          limit=1_000))


if __name__ == "__main__":
    unittest.main()
