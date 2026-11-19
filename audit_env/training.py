"""CPU training on open train episodes and evaluation on a disjoint dev split."""
from collections import Counter
from pathlib import Path

from .agents import ScanPolicy, reconcile
from .contracts import TABLE_IDS
from .dataset import suite, write_json, write_lines
from .environment import AuditEnv
from .generator import generate
from .micro import FEATURES, MODEL_VERSION, MicroModel, candidates
from .runner import run_policy

LEARNING_RATE = 0.45
L2 = 0.001
DEFAULT_EPOCHS = 600


def collect_public(episode):
    observed = {}

    def capture(tables):
        observed.update(tables)
        return []

    env = AuditEnv(training=False)
    observation, _ = env.load_episode(episode)
    outcome, _, _ = run_policy(env, observation, ScanPolicy(capture, tables=TABLE_IDS))
    if outcome.error:
        raise ValueError(f"could not collect observations: {outcome.error}")
    return observed


def extract(episode):
    observed = candidates(collect_public(episode))
    # Join with the answer key only after public-only feature extraction.
    positives = {(issue.type, issue.affected_ids[0]) for issue in episode.issues}
    return [
        (candidate.features, int((candidate.kind, candidate.affected_id) in positives))
        for candidate in observed
    ]


def fit(examples, epochs=DEFAULT_EPOCHS):
    if not 1 <= epochs <= 10_000:
        raise ValueError("epochs must be in [1, 10000]")
    frequencies = Counter(examples)
    classes = Counter(label for _, label in examples)
    if not classes[0] or not classes[1]:
        raise ValueError("training requires both positive and negative examples")
    points = sorted(frequencies.items())
    weights = (0.0,) * len(FEATURES)
    bias = 0.0
    for _ in range(epochs):
        model = MicroModel(weights, bias)
        gradient = [0.0] * len(weights)
        delta_bias = 0.0
        for (features, label), count in points:
            # Balance rare positive examples against the clean majority.
            error = (model.probability(features) - label) * count / (2 * classes[label])
            delta_bias += error
            for index, value in enumerate(features):
                gradient[index] += error * value
        weights = tuple(
            weight - LEARNING_RATE * (delta + L2 * weight)
            for weight, delta in zip(weights, gradient)
        )
        bias -= LEARNING_RATE * delta_bias
    return MicroModel(weights, bias)


def candidate_metrics(model, examples):
    tp = fp = fn = 0
    for features, label in examples:
        predicted = model.probability(features) >= model.threshold
        tp += int(predicted and label)
        fp += int(predicted and not label)
        fn += int(not predicted and label)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def episode_metrics(episodes, findings):
    results = []
    for episode in episodes:
        env = AuditEnv(training=False)
        observation, _ = env.load_episode(episode)
        outcome, trace, _ = run_policy(env, observation, ScanPolicy(findings, tables=TABLE_IDS))
        if outcome.error:
            raise ValueError(f"dev evaluation failed: {outcome.error}")
        results.append((outcome.f1, outcome.reward, len(trace) - 1))
    return {name: round(sum(item[i] for item in results) / len(results), 6)
            for i, name in enumerate(("mean_f1", "mean_reward", "mean_steps"))}


def train(suite_path, output, epochs=DEFAULT_EPOCHS):
    splits = {"train": [], "dev": []}
    for split, seed, config in suite(suite_path):
        splits[split].append(generate(seed, config))
    if not all(splits.values()):
        raise ValueError("train and dev must both be present")
    train_examples = [example for episode in splits["train"] for example in extract(episode)]
    model = fit(train_examples, epochs)
    dev_examples = [example for episode in splits["dev"] for example in extract(episode)]
    zero = MicroModel()

    metrics = {
        "version": MODEL_VERSION, "method": "class_weighted_logistic_regression",
        "epochs": epochs, "learning_rate": LEARNING_RATE, "l2": L2, "threshold": model.threshold,
        "train_episodes": len(splits["train"]), "dev_episodes": len(splits["dev"]),
        "train_examples": len(train_examples), "dev_examples": len(dev_examples),
        "train_positives": sum(label for _, label in train_examples),
        "dev_positives": sum(label for _, label in dev_examples),
        "dev_candidates": {
            "untrained": candidate_metrics(zero, dev_examples),
            "trained": candidate_metrics(model, dev_examples),
        },
        "dev_episodes_metrics": {
            "untrained": episode_metrics(splits["dev"], zero.findings),
            "trained": episode_metrics(splits["dev"], model.findings),
            "rule_based": episode_metrics(splits["dev"], reconcile),
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model.json", model.to_dict())
    write_json(output / "report.json", metrics)
    write_lines(output / "train_examples.jsonl",
                ({"features": features, "label": label} for features, label in train_examples))
    return metrics
