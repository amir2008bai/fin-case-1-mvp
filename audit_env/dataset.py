"""Export public episodes, private answers, and trajectories to distinct trees."""
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .agents import create_policy
from .contracts import Config
from .environment import AuditEnv
from .generator import generate
from .runner import run_policy, strict_loads

LEDGER_FIELDS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
)
FX_FIELDS = ("date", "source_currency", "target_currency", "rate")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_lines(path, items):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "null" if value is None else value for key, value in row.items()})


def suite(path):
    document = strict_loads(Path(path).read_text(encoding="utf-8"))
    if document.keys() != {"schema_version", "train", "dev"} or document["schema_version"] != "1.0":
        raise ValueError("unsupported suite schema")
    used = set()
    for split in ("train", "dev"):
        for row in document[split]:
            if row.keys() != {"seed", "config"} or type(row["seed"]) is not int or row["seed"] in used:
                raise ValueError("invalid or repeated suite seed")
            used.add(row["seed"])
            yield split, row["seed"], Config(**row["config"])


def export(root, suite_path):
    cases = list(suite(suite_path))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "private").mkdir(mode=0o700)
    trajectory = []
    for split, seed, config in cases:
        episode = generate(seed, config)
        episode_id = episode.manifest["episode_id"]
        public = root / "public" / split / episode_id
        private = root / "private" / split / episode_id
        public.mkdir(parents=True, exist_ok=False)
        private.mkdir(parents=True, exist_ok=False)
        write_json(public / "manifest.json", episode.manifest)
        write_json(public / "policy.json", episode.policy)
        write_csv(public / "ledger.csv", LEDGER_FIELDS, episode.tables["ledger"])
        write_csv(public / "fx.csv", FX_FIELDS, episode.tables["fx"])
        for table in ("invoices", "payments"):
            write_lines(public / f"{table}.jsonl", episode.tables[table])
        write_json(private / "truth.json", {"issues": [asdict(issue) for issue in episode.issues]})
        write_json(private / "provenance.json", episode.provenance)
        env = AuditEnv()
        obs, _ = env.load_episode(episode)
        _, trace, _ = run_policy(env, obs, create_policy("reconcile"))
        trajectory.extend(trace)
    write_lines(root / "trajectories.jsonl", trajectory)
