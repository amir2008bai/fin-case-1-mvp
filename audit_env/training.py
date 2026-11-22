"""Train the micro model on generated train episodes; compare on held-out dev and hold-out configs.

1. Generate train episodes (open key): the train split configs with many seeds.
2. Supervised fit: per-type logistic weights on candidate features, labels from truth.
3. Reward tuning: choose the submit threshold that maximizes the mean terminal reward
   of the full policy on train episodes (the environment's reward is the signal).
4. Held-out comparison: dev split and hold-out template configs with fresh seeds and a
   secret key, against reconcile, expert and oracle. Nothing from dev is used for fitting.
"""
import random
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .agents import create_policy, oracle
from .contracts import EXTRA_TYPES, MAX_ISSUES, Config
from .dataset import suite
from .environment import AuditEnv
from .generator import generate
from .micro import FEATURES, MicroModel, TYPES, fit_logistic, labelled
from .runner import run_policy

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def train_configs(suite_path):
    return [(case_id, config) for split, case_id, _, _, config in suite(suite_path) if split == "train"]


def run(episode, policy, closed=False):
    env = AuditEnv(training=not closed)
    observation, _ = env.load_episode(episode)
    outcome, trace, seconds = run_policy(env, observation, policy)
    return outcome, outcome.inspection_steps, seconds


def augmented(config, index):
    """Training mixes difficulty levels, and every other episode carries the real-practice error types."""
    config = replace(config, difficulty=("medium", "hard", "easy")[index % 3])
    if index % 2 == 0:
        return config
    extra = {kind: 1 for kind in EXTRA_TYPES}
    errors = dict(config.errors) | extra
    if sum(errors.values()) + 2 * sum(config.interactions.values()) > MAX_ISSUES:
        return config
    return replace(config, errors=errors)


def train(suite_path, output, *, seeds_per_config=12, seed=20261120):
    rng = random.Random(seed)
    configs = train_configs(suite_path)
    episodes = [generate(rng.randrange(2**31), augmented(config, index))
                for _, config in configs for index in range(seeds_per_config)]
    rows = defaultdict(list)
    for episode in episodes:
        for candidate, label in labelled(episode):
            rows[candidate.type].append((candidate.features, label))
    weights = {kind: fit_logistic(rows[kind], len(FEATURES[kind])) if rows[kind] else [0.0] * len(FEATURES[kind])
               for kind in TYPES}
    tuning = {}
    val_rng = random.Random(seed + 10_000)
    sample = [generate(val_rng.randrange(2**31), augmented(config, i))
              for _, config in configs for i in range(2)]
    for threshold in THRESHOLDS:
        model = MicroModel(weights, threshold)
        tuning[threshold] = statistics.mean(run(episode, model.policy())[0].reward for episode in sample)
    best = max(THRESHOLDS, key=lambda value: (round(tuning[value], 9), -abs(value - 0.5)))
    model = MicroModel(weights, best, {
        "train_episodes": len(episodes), "candidates": {kind: len(rows[kind]) for kind in TYPES},
        "positives": {kind: sum(label for _, label in rows[kind]) for kind in TYPES},
        "threshold_rewards": {str(key): round(value, 6) for key, value in tuning.items()},
        "suite": Path(suite_path).name, "seed": seed,
    })
    model.save(Path(output) / "model.json")
    return model


def compare(model, suite_path, holdout_path, extended_path=None, curriculum_path=None, *, holdout_seed=99,
            key=b"micro-holdout-key"):
    cases = [(split, case_id, category, seed, config) for split, case_id, category, seed, config in suite(suite_path)
             if split == "dev"]
    rng = random.Random(holdout_seed)
    for path in (holdout_path, extended_path, curriculum_path):
        if path is None:
            continue
        for split, case_id, category, _, config in suite(path):
            cases.append((split, case_id, category, rng.randrange(2**31), config))
    results = defaultdict(list)
    for split, case_id, category, seed, config in cases:
        if split != "dev":
            episode = generate(seed, replace(config, keyspace="secret"), key=key)
        else:
            episode = generate(seed, config)
        policies = {"micro": model.policy(), "expert": create_policy("expert"),
                    "reconcile": create_policy("reconcile"), "oracle": oracle(episode)}
        for name, policy in policies.items():
            outcome, steps, seconds = run(episode, policy, closed=True)
            results[(split, name, category.split(":")[0])].append(
                {"precision": outcome.precision, "recall": outcome.recall, "f1": outcome.f1,
                 "reward": outcome.reward, "steps": steps, "seconds": seconds, "fp": outcome.fp, "fn": outcome.fn})
    summary = []
    for (split, name, category), rows in sorted(results.items()):
        summary.append({"split": split, "policy": name, "category": category, "episodes": len(rows)} | {
            key: round(statistics.mean(row[key] for row in rows), 4)
            for key in ("precision", "recall", "f1", "reward", "steps", "fp", "fn")})
    return summary


def collect_public(episode):
    """Compatibility helper for the RAG tools: the tables an agent can read."""
    return episode.tables


DEFAULT_EPOCHS = 300
