"""Build conversational SFT, preference pairs and replayable reward trajectories."""
import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from random import Random

from .agents import reconcile, submission
from .contracts import Config
from .dataset import suite, write_json
from .domain import EXPENSE, SUSPENSE
from .generator import generate
from .llm import Conversation, json_text, message
from .rag_env import RagAuditEnv
from .runner import run_policy

PROBES = (
    {"type": "group_ledger", "by": "invoice_id", "account": EXPENSE, "min_count": 2, "limit": 12},
    {"type": "group_ledger", "by": "payment_id", "account": SUSPENSE, "min_count": 1, "limit": 12},
)


def lookup(table, field, subject):
    return {"type": "search_records", "table": table, "link": {"field": field, "id": subject},
            "limit": 12}


class RetrievalTeacher:
    """Demonstrator using public tools; never receives issues, seed or labels."""
    def actions(self):
        for probe, doc_table in zip(PROBES, ("invoices", "payments")):
            observation = yield probe
            for group in observation["result"]["groups"]:
                for table in ("ledger", doc_table):
                    observation = yield lookup(table, probe["by"], group["id"])
                    for record in observation["result"]["records"]:
                        key = record.get("doc_id") or record["row_id"]
                        self.tables[table][key] = record
        yield submission(self.episode_id, reconcile(
            {table: list(rows.values()) for table, rows in self.tables.items()},
        ))

    def __init__(self):
        self.tables = {"ledger": {}, "invoices": {}, "payments": {}}
        self.workflow = self.actions()

    def __call__(self, observation):
        if "manifest" in observation:
            self.episode_id = observation["manifest"]["episode_id"]
            return next(self.workflow)
        return self.workflow.send(observation)


def generated_cases(seed_base, train_count, dev_count):
    if train_count < 1 or dev_count < 1:
        raise ValueError("both splits must contain at least one episode")
    shape = ((0, 0), (1, 0), (0, 1), (2, 3), (4, 1), (1, 5), (5, 4), (0, 0))
    for split, count in (("train", train_count), ("dev", dev_count)):
        rng = Random(f"{seed_base}:{split}")
        for index in range(count):
            duplicates, allocations = shape[index % len(shape)]
            yield split, rng.randrange(1 << 120), Config(
                invoice_count=rng.randint(50, 95), duplicate_count=duplicates,
                allocation_count=allocations,
            )


def teacher_trace(episode):
    env = RagAuditEnv(training=True)
    start, _ = env.load_episode(episode)
    score, trace, _ = run_policy(env, start, RetrievalTeacher())
    if score.error or score.f1 != 1:
        raise ValueError(f"teacher failed on {episode.manifest['episode_id']}")
    return score, trace


def samples(trace):
    conversation = Conversation(trace[0]["observation"])
    for row in trace[1:]:
        action = row["action"]
        yield {"prompt": conversation.messages.copy(),
               "completion": [message("assistant", json_text(action))]}
        if not (row["terminated"] or row["truncated"]):
            conversation.observe(action, row["observation"])


def comparison(episode, trace, final_sample, teacher_reward):
    full = trace[-1]["action"]["findings"]
    if not full["findings"]:
        return None
    rejected = submission(full["episode_id"], full["findings"][1:])
    env = RagAuditEnv(training=True)
    env.load_episode(episode)
    for row in trace[1:-1]:
        env.step(row["action"])
    env.step(rejected)
    if env.outcome.reward >= teacher_reward:
        raise ValueError("preference must improve the actual evaluator reward")
    return {
        "prompt": final_sample["prompt"], "chosen": final_sample["completion"],
        "rejected": [message("assistant", json_text(rejected))],
    }


def export_llm(destination, cases):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"version": 1, "splits": {}}
    # Stream records by episode. The datasets never contain truth, provenance or seed.
    used_ids = set()
    with ExitStack() as resources:
        streams = {}
        for split in ("train", "dev"):
            for kind in ("sft", "preferences", "rollouts"):
                streams[(split, kind)] = resources.enter_context(
                    (destination / f"{split}.{kind}.jsonl").open("w", encoding="utf-8"),
                )
            metadata["splits"][split] = dict(episodes=0, sft=0, preferences=0, rollouts=0)
        for split, seed, config in cases:
            episode = generate(seed, config)
            episode_id = episode.manifest["episode_id"]
            if episode_id in used_ids:
                raise ValueError("episode present in more than one position or split")
            used_ids.add(episode_id)
            score, trace = teacher_trace(episode)
            rows = list(samples(trace))
            preference = comparison(episode, trace, rows[-1], score.reward)
            metadata["splits"][split]["episodes"] += 1
            items = {"sft": rows, "preferences": [preference] if preference else [],
                     "rollouts": [{"episode_id": episode_id, "trajectory": trace, "reward": score.reward}]}
            for kind, entries in items.items():
                for entry in entries:
                    streams[(split, kind)].write(json_text(entry) + "\n")
                metadata["splits"][split][kind] += len(entries)
    if not all(split["episodes"] for split in metadata["splits"].values()):
        raise ValueError("both splits must contain at least one episode")
    metadata["sha256"] = {}
    for path in sorted(destination.glob("*.jsonl")):
        with path.open("rb") as stream:
            metadata["sha256"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(destination / "manifest.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", type=Path, help="use an explicit open suite instead of generated cases")
    parser.add_argument("--seed-base", type=int, default=7_001_031)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--dev-episodes", type=int, default=24)
    args = parser.parse_args()
    cases = suite(args.suite) if args.suite else generated_cases(
        args.seed_base, args.train_episodes, args.dev_episodes,
    )
    print(json.dumps(export_llm(args.output, cases), ensure_ascii=False))


if __name__ == "__main__":
    main()
