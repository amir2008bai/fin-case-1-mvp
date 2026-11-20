"""Command line: one-command verification, benchmarks, export, demo and auditing of folders."""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path

DEFAULT_SUITE = "configs/suite.v2.json"
HOLDOUT = "configs/holdout-template.v2.json"
DATA = Path("data")
POLICIES = ("oracle", "expert", "reconcile", "random", "flag-all")


def out(text=""):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def table(rows):
    lines = [f"{'policy':10s} {'category':12s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'return':>7s} {'steps':>6s} {'ms':>7s}"]
    for (policy, category), items in sorted(rows.items()):
        mean = lambda key: statistics.mean(item[key] for item in items)  # noqa: E731
        lines.append(f"{policy:10s} {category:12s} {len(items):3d} {mean('precision'):6.3f} {mean('recall'):6.3f} "
                     f"{mean('f1'):6.3f} {mean('reward'):+7.3f} {mean('steps'):6.1f} {1000 * mean('seconds'):7.1f}")
    return "\n".join(lines)


def holdout_cases(path, seed_source):
    from .dataset import suite

    for split, case_id, category, seed, config in suite(path):
        yield split, case_id, category, seed if seed is not None else seed_source.randrange(2**31), config


def command_verify(args):
    import random

    from .dataset import benchmark_cases, checksums, export
    from .selfcheck import contract_checks, data_invariants, fresh_key

    started = time.time()
    report = {"python": sys.version.split()[0], "sections": {}}
    failed = []

    out("== 1. Unit tests")
    suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=1 if args.quick else 2, stream=sys.stdout).run(suite)
    report["sections"]["unit_tests"] = {"run": result.testsRun, "failures": len(result.failures),
                                        "errors": len(result.errors)}
    if not result.wasSuccessful():
        failed.append("unit tests")

    out("\n== 2. Contract, isolation and reward checks")
    checks = contract_checks()
    for item in checks:
        out(f"  [{'PASS' if item['ok'] else 'FAIL'}] {item['category']:15s} {item['name']} — {item['detail']}")
    passed = sum(item["ok"] for item in checks)
    out(f"  {passed}/{len(checks)} passed")
    report["sections"]["contract_checks"] = checks
    if passed != len(checks):
        failed.append("contract checks")

    out("\n== 3. Independent truth verification on random configs (secret throwaway key)")
    failures, kinds = data_invariants(args.cases, key=fresh_key(), seed_base=int.from_bytes(os.urandom(4), "big"))
    out(f"  {args.cases} episodes, error types covered: {', '.join(kinds)}; failures: {len(failures)}")
    for line in failures[:10]:
        out(f"  FAIL {line}")
    report["sections"]["truth_verification"] = {"episodes": args.cases, "failures": failures, "types": kinds}
    if failures:
        failed.append("truth verification")

    out("\n== 4. Committed open episodes are byte-identical to a fresh export")
    if (DATA / "SHA256SUMS").is_file():
        with tempfile.TemporaryDirectory() as temp:
            export(Path(temp) / "data", DEFAULT_SUITE)
            fresh = checksums(Path(temp) / "data")
        committed = (DATA / "SHA256SUMS").read_text(encoding="utf-8")
        same = fresh == committed and checksums(DATA) == committed
        out(f"  {'identical' if same else 'DIFFERENT'} ({len(committed.splitlines())} files)")
        report["sections"]["data"] = {"identical": same}
        if not same:
            failed.append("committed data")
    else:
        out("  data/ not exported yet — run: python -m audit_env dataset data")
        report["sections"]["data"] = {"identical": None}

    out("\n== 5. Hold-out benchmark: 4 baselines + expert, closed mode, fresh secret key, paired checks")
    rows, problems = benchmark_cases(holdout_cases(HOLDOUT, random.Random(os.urandom(8))), POLICIES, key=fresh_key())
    out(table(rows))
    for line in problems:
        out(f"  FAIL {line}")
    report["sections"]["holdout"] = {
        f"{policy}/{category}": {key: round(statistics.mean(item[key] for item in items), 6)
                                 for key in ("precision", "recall", "f1", "reward", "steps", "seconds")}
        for (policy, category), items in sorted(rows.items())
    }
    if problems:
        failed.append("hold-out oracle/paired checks")

    report["seconds"] = round(time.time() - started, 1)
    report["ok"] = not failed
    target = Path(args.report)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        target = Path(tempfile.gettempdir()) / "verify-report.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out(f"\n== {'OK' if not failed else 'FAILED: ' + ', '.join(failed)} in {report['seconds']} s "
        f"(report: {target})")
    return 0 if not failed else 1


def command_benchmark(args):
    import random

    from .dataset import benchmark_cases, suite
    from .selfcheck import fresh_key

    if args.suite == HOLDOUT or args.holdout:
        cases = holdout_cases(HOLDOUT, random.Random(args.seed))
        key = fresh_key()
    else:
        cases, key = suite(args.suite), None
    rows, problems = benchmark_cases(cases, POLICIES, key=key)
    out(table(rows))
    for line in problems:
        out(f"FAIL {line}")
    return 1 if problems else 0


def command_dataset(args):
    from .dataset import export

    index = export(args.output, args.suite)
    out(f"exported {len(index)} episodes to {args.output}")
    return 0


def command_selfcheck(args):
    from .selfcheck import data_invariants

    failures, kinds = data_invariants(args.n, seed_base=args.seed)
    out(f"{args.n} episodes, types {kinds}, failures {len(failures)}")
    for line in failures:
        out(line)
    return 1 if failures else 0


def command_demo(args):
    from .contracts import Config
    from .dataset import evaluate
    from .episode_io import write_lines
    from .generator import generate

    config = Config.from_dict(json.loads(args.config)) if args.config else Config(keyspace="open")
    episode = generate(args.seed, config)
    outcome, trace, seconds = evaluate(episode, args.policy, closed=args.closed)
    if args.trajectory:
        write_lines(args.trajectory, trace)
    out(json.dumps({"episode_id": episode.manifest["episode_id"], "policy": args.policy,
                    "issues": len(episode.issues), "tp": outcome.tp, "fp": outcome.fp, "fn": outcome.fn,
                    "precision": outcome.precision, "recall": outcome.recall, "f1": outcome.f1,
                    "reward": outcome.reward, "steps": len(trace) - 1, "seconds": round(seconds, 4),
                    "termination": outcome.termination}, ensure_ascii=False))
    return 0


def command_audit(args):
    from .checks import audit
    from .episode_io import FormatError, load_episode

    try:
        episode = load_episode(args.folder)
    except FormatError as exc:
        out(json.dumps({"ok": False, "problems": exc.problems}, ensure_ascii=False, indent=2))
        return 2
    findings = audit(episode.manifest, episode.policy, episode.tables)
    out(json.dumps({"ok": True, "episode_id": episode.manifest.get("episode_id"), "findings": findings},
                   ensure_ascii=False, indent=2))
    return 0


def command_audit_truth(args):
    from .checks import audit
    from .episode_io import load_episode
    from .scoring import score

    episode = load_episode(args.public, args.answers)
    findings = audit(episode.manifest, episode.policy, episode.tables)
    seen = {value for rows in episode.tables.values() for row in rows for key, value in row.items()
            if key.endswith("_id") and isinstance(value, str)}
    result = score({"episode_id": episode.manifest["episode_id"], "findings": findings},
                   episode.manifest["episode_id"], episode.issues, seen, 0)
    out(json.dumps({"issues": len(episode.issues), "independent_findings": len(findings), "tp": result.tp,
                    "fp": result.fp, "fn": result.fn, "agrees": result.f1 == 1.0}, ensure_ascii=False))
    return 0 if result.f1 == 1.0 else 1


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m audit_env", description="Synthetic financial audit RL environment")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="one-command verification of the whole environment")
    verify.add_argument("--quick", action="store_true")
    verify.add_argument("--cases", type=int, default=60)
    verify.add_argument("--report", default="build/verify-report.json")
    bench = commands.add_parser("benchmark", help="compare baselines per category (closed mode)")
    bench.add_argument("--suite", default=DEFAULT_SUITE)
    bench.add_argument("--holdout", action="store_true")
    bench.add_argument("--seed", type=int, default=7)
    data = commands.add_parser("dataset", help="export open episodes, answers, trajectories, checksums")
    data.add_argument("output", type=Path)
    data.add_argument("--suite", default=DEFAULT_SUITE)
    selfcheck = commands.add_parser("selfcheck", help="prove truth correctness on N random configs")
    selfcheck.add_argument("--n", type=int, default=200)
    selfcheck.add_argument("--seed", type=int, default=1)
    demo = commands.add_parser("demo", help="run one policy on one generated episode")
    demo.add_argument("--seed", type=int, default=202601)
    demo.add_argument("--config", help="JSON config")
    demo.add_argument("--policy", choices=POLICIES, default="expert")
    demo.add_argument("--closed", action="store_true")
    demo.add_argument("--trajectory", type=Path)
    audit = commands.add_parser("audit", help="audit a folder with the six episode files (no truth needed)")
    audit.add_argument("folder", type=Path)
    truth = commands.add_parser("audit-truth", help="independently re-derive a folder's truth")
    truth.add_argument("public", type=Path)
    truth.add_argument("answers", type=Path)
    args = parser.parse_args()
    handlers = {"verify": command_verify, "benchmark": command_benchmark, "dataset": command_dataset,
                "selfcheck": command_selfcheck, "demo": command_demo, "audit": command_audit,
                "audit-truth": command_audit_truth}
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
