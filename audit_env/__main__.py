import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .agents import ScanPolicy, create_policy, oracle
from .contracts import Config, TABLE_IDS
from .dataset import export, suite, write_json, write_lines
from .environment import AuditEnv
from .generator import generate, transform
from .micro import MicroModel
from .monitor import EpisodeMonitor
from .rag import EvidenceIndex
from .runner import run_container, run_policy
from .training import DEFAULT_EPOCHS, collect_public, train

DEFAULT_SUITE = "configs/open-suite.v1.json"
POLICIES = ("reconcile", "random", "mark-all", "oracle")


def evaluate(episode, name, *, closed=False, agent_image=None, model=None, observer=None):
    env = AuditEnv(training=not closed)
    observation, _ = env.load_episode(episode)
    if agent_image:
        outcome, trace, seconds = run_container(env, observation, agent_image, name, observer)
    else:
        if name == "oracle":
            policy = oracle(episode.issues)
        elif name == "micro":
            policy = ScanPolicy(model.findings, tables=TABLE_IDS)
        else:
            policy = create_policy(name)
        outcome, trace, seconds = run_policy(env, observation, policy, observer)
    return asdict(outcome) | {
        "episode_id": episode.manifest["episode_id"], "policy": name,
        "steps": len(trace) - 1, "seconds": round(seconds, 6),
    }, trace


def benchmark(suite_path):
    count = 0
    aggregates = {name: [] for name in POLICIES}
    for split, seed, config in suite(suite_path):
        episode = generate(seed, config)
        mirror = transform(episode, 1_000_000 + seed)
        for name in POLICIES:
            result, _ = evaluate(episode, name, closed=True)
            aggregates[name].append(result)
        perfect = aggregates["oracle"][-1]
        replay, _ = evaluate(mirror, "oracle", closed=True)
        if perfect["f1"] != 1 or (perfect["f1"], perfect["reward"]) != (replay["f1"], replay["reward"]):
            raise SystemExit(f"Oracle or transform failed for {split} case {count}")
        count += 1
    if count == 0:
        raise SystemExit("Empty suite")
    for name, rows in aggregates.items():
        averages = {
            key: round(sum(row[key] for row in rows) / count, 6)
            for key in ("precision", "recall", "f1", "reward", "steps", "seconds")
        }
        print(json.dumps({"policy": name, "episodes": count, "mean": averages}))


def main():
    parser = argparse.ArgumentParser(description="Synthetic financial audit environment")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a policy and print evaluator diagnostics")
    demo.add_argument("--seed", type=int, default=202601)
    demo.add_argument("--policy", choices=(*POLICIES, "micro"), default="reconcile")
    demo.add_argument("--closed", action="store_true")
    demo.add_argument("--agent-image", help="run the reference agent in a separate local Docker image")
    demo.add_argument("--trajectory", type=Path)
    demo.add_argument("--monitor", type=Path)
    demo.add_argument("--model", type=Path)
    bench = commands.add_parser("benchmark", help="compare policies and check transformed pairs")
    bench.add_argument("--suite", default=DEFAULT_SUITE)
    data = commands.add_parser("dataset", help="export public episodes, private answers and trajectories")
    data.add_argument("output", type=Path)
    data.add_argument("--suite", default=DEFAULT_SUITE)
    training = commands.add_parser("train", help="fit the micro model and compare on open dev")
    training.add_argument("--suite", default=DEFAULT_SUITE)
    training.add_argument("--output", type=Path, default=Path("build/micro"))
    training.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    retrieve = commands.add_parser("retrieve", help="query the episode's public evidence index")
    retrieve.add_argument("query")
    retrieve.add_argument("--table", choices=TABLE_IDS)
    retrieve.add_argument("--limit", type=int, choices=range(1, 21), default=5)
    retrieve.add_argument("--seed", type=int, default=202601)
    args = parser.parse_args()
    if args.command == "dataset":
        export(args.output, args.suite)
    elif args.command == "benchmark":
        benchmark(args.suite)
    elif args.command == "train":
        print(json.dumps(train(args.suite, args.output, args.epochs), ensure_ascii=False))
    elif args.command == "retrieve":
        tables = collect_public(generate(args.seed, Config()))
        evidence = EvidenceIndex(tables).search(args.query, table=args.table, limit=args.limit)
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    else:
        if args.agent_image and (not args.closed or args.policy == "oracle"):
            parser.error("--agent-image requires --closed and a public policy")
        if args.model and (args.policy != "micro" or args.agent_image):
            parser.error("--model is for the local micro policy")
        model = None
        if args.policy == "micro" and not args.agent_image:
            path = args.model or Path("build/micro/model.json")
            if not path.is_file():
                parser.error(f"micro model not found: {path}; run make train")
            model = MicroModel.load(path)
        observer = EpisodeMonitor() if args.monitor else None
        metric, trace = evaluate(generate(args.seed, Config()), args.policy, closed=args.closed,
                                 agent_image=args.agent_image, model=model, observer=observer)
        if observer is not None:
            write_json(args.monitor, observer.report())
        if args.trajectory:
            write_lines(args.trajectory, trace)
        print(json.dumps(metric, ensure_ascii=False))
        if metric["error"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
