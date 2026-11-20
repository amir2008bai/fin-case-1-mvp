"""Versioned suites, deterministic export of episodes, answers, trajectories and benchmarks."""
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .agents import create_policy, oracle
from .contracts import Config
from .environment import AuditEnv
from .episode_io import write_episode, write_json, write_lines
from .generator import generate
from .runner import run_policy, strict_loads

POLICIES = ("oracle", "expert", "reconcile", "random", "flag-all")
SUITE_KEYS = {"schema_version", "name", "keyspace", "splits"}
CASE_KEYS = {"case_id", "category", "seed", "config"}


def suite(path):
    """Yield (split, case_id, category, seed, Config) from a versioned suite file."""
    document = strict_loads(Path(path).read_text(encoding="utf-8"))
    if document.keys() != SUITE_KEYS or document["schema_version"] != "2.0":
        raise ValueError("unsupported suite schema")
    seeds = set()
    for split, cases in document["splits"].items():
        for case in cases:
            keys = set(case)
            if not (keys == CASE_KEYS or keys == CASE_KEYS - {"seed"}):
                raise ValueError(f"invalid case {case.get('case_id')}")
            seed = case.get("seed")
            if seed is not None:
                if type(seed) is not int or seed in seeds:
                    raise ValueError("invalid or repeated suite seed")
                seeds.add(seed)
            config = dict(case["config"], keyspace=document["keyspace"])
            yield split, case["case_id"], case["category"], seed, Config.from_dict(config)


def evaluate(episode, name, *, closed=False, observer=None):
    env = AuditEnv(training=not closed)
    observation, _ = env.load_episode(episode)
    policy = oracle(episode) if name == "oracle" else create_policy(name)
    outcome, trace, seconds = run_policy(env, observation, policy, observer)
    return outcome, trace, seconds


def summarize(rows):
    keys = ("precision", "recall", "f1", "reward", "steps")
    return {key: round(statistics.mean(row[key] for row in rows), 6) for key in keys} | {"episodes": len(rows)}


def export(root, suite_path, *, policies=POLICIES):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    index, traces, results = [], defaultdict(list), defaultdict(list)
    for split, case_id, category, seed, config in suite(suite_path):
        if seed is None:
            raise ValueError("an exported suite needs explicit seeds")
        episode = generate(seed, config)
        episode_id = episode.manifest["episode_id"]
        write_episode(episode, root / "episodes" / split / episode_id, root / "answers" / split / episode_id)
        index.append({"split": split, "case_id": case_id, "category": category, "seed": seed,
                      "episode_id": episode_id, "config": config.as_dict(), "issues": len(episode.issues)})
        for name in policies:
            outcome, trace, _ = evaluate(episode, name)
            traces[name].extend(trace)
            results[(name, split, category.split(":")[0])].append(
                {"precision": outcome.precision, "recall": outcome.recall, "f1": outcome.f1,
                 "reward": outcome.reward, "steps": len(trace) - 1})
    write_json(root / "index.json", index)
    write_lines(root / "trajectories.jsonl", traces["expert"])
    folder = root / "trajectories"
    folder.mkdir()
    for name in policies:
        if name != "expert":  # the expert run is trajectories.jsonl itself
            write_lines(folder / f"{name}.jsonl", traces[name])
    benchmark = [{"policy": name, "split": split, "category": category} | summarize(rows)
                 for (name, split, category), rows in sorted(results.items())]
    write_json(root / "benchmark.json", benchmark)
    write_checksums(root)
    return index


def checksums(root):
    root = Path(root)
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def write_checksums(root):
    Path(root, "SHA256SUMS").write_text(checksums(root), encoding="utf-8", newline="\n")


def benchmark_cases(cases, policies=POLICIES, *, paired=True, key=None):
    """Run policies on generated cases in closed mode; returns per-category rows and failures."""
    from .generator import transform

    rows, failures = defaultdict(list), []
    for split, case_id, category, seed, config in cases:
        episode = generate(seed, config, key=key)
        for name in policies:
            outcome, trace, seconds = evaluate(episode, name, closed=True)
            rows[(name, category.split(":")[0])].append({
                "precision": outcome.precision, "recall": outcome.recall, "f1": outcome.f1,
                "reward": outcome.reward, "steps": len(trace) - 1, "seconds": seconds,
                "fp": outcome.fp, "fn": outcome.fn,
            })
            if name == "oracle" and outcome.f1 != 1.0:
                failures.append(f"{case_id}: oracle F1 {outcome.f1}")
        if paired:
            mirror = transform(episode, (seed or 0) + 1_000_003)
            first, _, _ = evaluate(episode, "oracle", closed=True)
            second, _, _ = evaluate(mirror, "oracle", closed=True)
            if (first.f1, first.reward) != (second.f1, second.reward):
                failures.append(f"{case_id}: paired check changed the oracle result")
    return rows, failures


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
