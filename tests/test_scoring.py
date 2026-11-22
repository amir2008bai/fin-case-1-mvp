"""Reward robustness: matching rules, clean cases, FP, duplicates, cost and attacks."""
import unittest
from dataclasses import replace

from audit_env.agents import create_policy
from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.runner import run_policy
from audit_env.scoring import score

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})
CLEAN = Config(keyspace="open", errors={key: 0 for key in ERRORS})


def seen_all(episode):
    return {value for rows in episode.tables.values() for row in rows for key, value in row.items()
            if key.endswith("_id") and isinstance(value, str)}


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = generate(41, MIXED)
        cls.clean = generate(42, CLEAN)
        cls.truth = [issue.as_finding() for issue in cls.episode.issues]

    def compute(self, findings, *, episode=None, issues=None, seen=None, steps=0):
        episode = episode or self.episode
        return score({"episode_id": episode.manifest["episode_id"], "findings": findings},
                     episode.manifest["episode_id"], episode.issues if issues is None else issues,
                     seen_all(episode) if seen is None else seen, steps)

    def test_full_answer_beats_silence_on_erroneous_books(self):
        full, silent = self.compute(self.truth), self.compute([])
        self.assertEqual((full.tp, full.fp, full.fn, full.reward), (len(self.truth), 0, 0, 1.0))
        self.assertEqual(silent.reward, -1.0)
        for index in range(len(self.truth)):
            partial = self.compute(self.truth[:index] + self.truth[index + 1:])
            self.assertGreater(full.reward, partial.reward)
            self.assertGreater(partial.reward, silent.reward)

    def test_silence_beats_false_accusations_on_clean_books(self):
        silent = self.compute([], episode=self.clean, issues=())
        self.assertEqual((silent.f1, silent.reward), (1.0, 1.0))
        accusation = self.compute([self.truth[0]], episode=self.clean, issues=())
        self.assertEqual((accusation.f1, accusation.reward), (0.0, -1.0))

    def test_empty_truth_rules(self):
        self.assertEqual(self.compute([], issues=()).f1, 1.0)
        self.assertEqual(self.compute([self.truth[0]], issues=()).f1, 0.0)
        result = self.compute([])
        self.assertEqual((result.precision, result.recall), (0.0, 0.0))

    def test_repeats_give_no_gain(self):
        once, twice = self.compute(self.truth), self.compute(self.truth + self.truth)
        self.assertEqual(twice.tp, once.tp)
        self.assertEqual(twice.fp, len(self.truth))
        self.assertLess(twice.reward, once.reward)

    def test_flag_everything_scores_badly(self):
        for episode in (self.episode, self.clean):
            env = AuditEnv()
            start, _ = env.load_episode(episode)
            outcome, _, _ = run_policy(env, start, create_policy("flag-all"))
            self.assertLessEqual(outcome.reward, -0.8)
        expert_env = AuditEnv()
        start, _ = expert_env.load_episode(self.episode)
        expert, _, _ = run_policy(expert_env, start, create_policy("expert"))
        self.assertGreater(expert.reward, 0.9)

    def test_step_cost_only_slightly_separates_equal_answers(self):
        cheap, dear = self.compute(self.truth, steps=10), self.compute(self.truth, steps=60)
        self.assertAlmostEqual(cheap.reward - dear.reward, 0.1)
        self.assertGreater(dear.reward, self.compute(self.truth[1:], steps=10).reward)

    def test_evidence_must_be_seen_relevant_and_complete(self):
        first = self.truth[0]
        issue = self.episode.issues[0]
        self.assertEqual(self.compute([first], seen=set()).tp, 0)
        self.assertEqual(self.compute([first | {"evidence_ids": ["row_" + "0" * 24]}]).tp, 0)
        stranger = next(row["row_id"] for row in self.episode.tables["ledger"] if row["row_id"] not in issue.relevant_ids)
        self.assertEqual(self.compute([first | {"evidence_ids": first["evidence_ids"][:9] + [stranger]}]).tp, 0)
        self.assertEqual(self.compute([first | {"evidence_ids": first["evidence_ids"][:-1]}]).tp, 0)
        other = self.episode.issues[1]
        foreign_proof = list(other.evidence_sets[0])[:10]
        self.assertEqual(self.compute([first | {"evidence_ids": foreign_proof}]).tp, 0)
        for proof in issue.evidence_sets:
            self.assertEqual(self.compute([first | {"evidence_ids": list(proof)}]).tp, 1)

    def test_guessed_ids_type_currency_and_tolerance(self):
        for index, finding in enumerate(self.truth):
            issue = self.episode.issues[index]
            with self.subTest(issue=issue.type):
                self.assertEqual(self.compute([finding | {"affected_ids": ["invoice_" + "1" * 24]}]).tp, 0)
                self.assertEqual(self.compute([finding | {"currency": "EUR" if issue.currency != "EUR" else "KZT"}]).tp, 0)
                wrong_type = "fx_rate" if issue.type != "fx_rate" else "period_cutoff"
                self.assertEqual(self.compute([finding | {"type": wrong_type}]).tp, 0)
                edge = finding["claimed_amount_minor"] + issue.tolerance_minor
                self.assertEqual(self.compute([finding | {"claimed_amount_minor": edge}]).tp, 1)
                self.assertEqual(self.compute([finding | {"claimed_amount_minor": edge + 1}]).tp, 0)

    def test_one_finding_cannot_cover_two_interacting_issues(self):
        pairs = {}
        for issue in self.episode.issues:
            pairs.setdefault(issue.affected_ids, []).append(issue)
        shared = [items for items in pairs.values() if len(items) == 2]
        self.assertTrue(shared)
        first, second = shared[0]
        merged = first.as_finding() | {"evidence_ids": list(dict.fromkeys(first.evidence_sets[0] + second.evidence_sets[0]))[:10]}
        result = score({"episode_id": self.episode.manifest["episode_id"], "findings": [merged]},
                       self.episode.manifest["episode_id"], (first, second), seen_all(self.episode), 0)
        self.assertEqual(result.tp, 1)  # exactly one: the finding matches one issue, never both
        self.assertEqual(result.fn, 1)

    def test_malformed_submissions(self):
        episode_id = self.episode.manifest["episode_id"]
        first = self.truth[0]
        payloads = [
            None, [], "x", {"findings": []}, {"episode_id": "other", "findings": []},
            {"episode_id": episode_id, "findings": [first] * 31},
            {"episode_id": episode_id, "findings": [first | {"claimed_amount_minor": True}]},
            {"episode_id": episode_id, "findings": [first | {"claimed_amount_minor": 1.0}]},
            {"episode_id": episode_id, "findings": [first | {"evidence_ids": [f"row_{i:024x}" for i in range(11)]}]},
            {"episode_id": episode_id, "findings": [first | {"evidence_ids": []}]},
            {"episode_id": episode_id, "findings": [first | {"evidence_ids": ["a", "a"]}]},
            {"episode_id": episode_id, "findings": [first | {"explanation": "x" * 1001}]},
            {"episode_id": episode_id, "findings": [first | {"currency": "kzt"}]},
            {"episode_id": episode_id, "findings": [first | {"extra": 1}]},
            {"episode_id": episode_id, "findings": [{k: v for k, v in first.items() if k != "explanation"}]},
            {"episode_id": episode_id, "findings": [first], "note": 1},
        ]
        for payload in payloads:
            with self.subTest(payload=str(payload)[:60]):
                result = score(payload, episode_id, self.episode.issues, seen_all(self.episode), 0)
                self.assertEqual((result.reward, result.error), (-1.0, "invalid_submission"))

    def test_unknown_type_is_a_false_positive(self):
        result = self.compute(self.truth + [self.truth[0] | {"type": "made_up"}])
        self.assertEqual((result.tp, result.fp, result.error), (len(self.truth), 1, None))

    def test_matching_is_one_to_one_and_order_free(self):
        first = self.episode.issues[0]
        twin = replace(first, issue_id="issue_" + "f" * 24)
        result = self.compute([first.as_finding()], issues=(first, twin))
        self.assertEqual((result.tp, result.fn), (1, 1))
        forward = self.compute(self.truth)
        backward = self.compute(list(reversed(self.truth)))
        self.assertEqual((forward.tp, forward.reward), (backward.tp, backward.reward))


if __name__ == "__main__":
    unittest.main()
