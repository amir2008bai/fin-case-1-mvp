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
EXTENDED = "configs/holdout-extended.v2.json"
CURRICULUM = "configs/curriculum.v2.json"
LEVELS = ("easy", "medium", "hard")
DATA = Path("data")
POLICIES = ("oracle", "expert", "reconcile", "random", "flag-all")


def out(text=""):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def table(rows, order=None):
    lines = [f"{'policy':10s} {'category':12s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'return':>7s} {'steps':>6s} {'ms':>7s}"]
    rank = {name: index for index, name in enumerate(order or ())}
    for (policy, category), items in sorted(rows.items(), key=lambda item: (item[0][0], rank.get(item[0][1], 99),
                                                                              item[0][1])):
        mean = lambda key: statistics.mean(item[key] for item in items)  # noqa: E731
        lines.append(f"{policy:10s} {category:12s} {len(items):3d} {mean('precision'):6.3f} {mean('recall'):6.3f} "
                     f"{mean('f1'):6.3f} {mean('reward'):+7.3f} {mean('steps'):6.1f} {1000 * mean('seconds'):7.1f}")
    return "\n".join(lines)


def closed_episodes(seed_source, key, repeats=1):
    """Closed episodes of the jury template, the extended template and the hard level."""
    from dataclasses import replace

    from .dataset import suite
    from .generator import generate

    episodes = []
    for path in (HOLDOUT, EXTENDED, CURRICULUM):
        for split, *_, config in suite(path):
            if path == CURRICULUM and split != "hard":
                continue
            for _ in range(repeats):
                episodes.append(generate(seed_source.randrange(2**31), replace(config, keyspace="secret"), key=key))
    return episodes


def derived_key(seed):
    """A secret key fixed by --seed, so that a printed table can be reproduced byte for byte."""
    import hashlib

    return hashlib.sha256(f"red-team:{seed}".encode()).digest()


def economy_lines(result):
    lines = [f"  {result['episodes']} episodes: jury template + extended template + hard level"]
    for name, item in result["waste"].items():
        lines.append(f"  {name}: complete read {item['complete_read_steps']} steps on average; "
                     f"wasted {item['wasted_steps']}, skipped {item['skipped_steps']} steps")
    lines.append("  reading less of the ledger, honest rules on what was read:")
    lines.append(f"  {'read':16s} {'clean':>8s} {'errors':>8s} {'all':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'steps':>6s}")
    for row in result["skim_curve"]:
        clean = "—" if row["clean_reward"] is None else f"{row['clean_reward']:+.3f}"
        errors = "—" if row["error_reward"] is None else f"{row['error_reward']:+.3f}"
        lines.append(f"  {row['name']:16s} {clean:>8s} {errors:>8s} {row['reward']:+8.3f} {row['tp']:5d} "
                     f"{row['fp']:5d} {row['fn']:5d} {row['steps']:6.1f}")
    largest = result["largest_books"]
    counts = ", ".join(f"{name} {value}" for name, value in largest["counts"].items())
    lines.append(f"  largest allowed books ({counts}): complete read {largest['complete_read_steps']} "
                 f"steps of {largest['step_limit']}")
    lines.append("  violations: " + (str(len(result["violations"])) if result["violations"] else "none"))
    lines.extend(f"  FAIL {line}" for line in result["violations"][:20])
    return "\n".join(lines)


def command_economy(args):
    import random

    from .economy import report as economy_report
    from .selfcheck import fresh_key

    key = fresh_key() if args.fresh_key else derived_key(args.seed)
    result = economy_report(closed_episodes(random.Random(args.seed), key, repeats=args.repeats))
    out(economy_lines(result))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if result["violations"] else 0


def red_team(episodes, key):
    """Every attack on the given closed episodes, plus the seed brute force against `key`."""
    from .attacks import bruteforce_demo, run_suite
    from .contracts import Config

    rows, violations = run_suite(episodes)
    bruteforce = bruteforce_demo(Config(errors={"duplicate_invoice": 1}), secret_key=key)
    if not bruteforce["ok"]:
        violations.append("seed brute force: the open-key control failed or the secret key leaked")
    return {"episodes": len(episodes), "clean_episodes": sum(not episode.issues for episode in episodes),
            "errors": sum(len(episode.issues) for episode in episodes),
            "attacks": rows, "violations": violations, "seed_bruteforce": bruteforce}


def red_team_lines(result):
    lines = [f"  {result['episodes']} episodes ({result['clean_episodes']} clean, {result['errors']} errors): "
             "jury template + extended template + hard level"]
    lines.append(f"  {'agent':18s} {'clean':>8s} {'errors':>8s} {'TP':>5s} {'FP':>6s} {'steps':>6s}  what it tries")
    for row in result["attacks"]:
        clean = "—" if row["clean_reward"] is None else f"{row['clean_reward']:+.3f}"
        errors = "—" if row["error_reward"] is None else f"{row['error_reward']:+.3f}"
        lines.append(f"  {row['name']:18s} {clean:>8s} {errors:>8s} {row['tp']:5d} {row['fp']:6d} {row['steps']:6.1f}"
                     f"  {row['description']}")
    bruteforce = result["seed_bruteforce"]
    lines.append(f"  seed brute force: open key -> seed {bruteforce['open_key_recovered_seed']}, "
                 f"secret key -> {bruteforce['secret_key_recovered_seed']} after {bruteforce['tried_seeds']} seeds")
    lines.append("  violations: " + (str(len(result["violations"])) if result["violations"] else "none"))
    lines.extend(f"  FAIL {line}" for line in result["violations"][:20])
    return "\n".join(lines)


def command_attacks(args):
    """Red team. By default the secret key is derived from --seed, so the printed table is reproducible."""
    import random

    from .selfcheck import fresh_key

    key = fresh_key() if args.fresh_key else derived_key(args.seed)
    result = red_team(closed_episodes(random.Random(args.seed), key, repeats=args.repeats), key)
    out(red_team_lines(result))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if result["violations"] else 0


def holdout_cases(path, seed_source, by_split=False):
    """Cases of a template; seeds are drawn here. by_split groups results by split (the difficulty level)."""
    from .dataset import suite

    for split, case_id, category, seed, config in suite(path):
        yield (split, case_id, split if by_split else category,
               seed if seed is not None else seed_source.randrange(2**31), config)


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

    out("\n== 6. Extended hold-out: real-practice errors (VAT on top, uncollected renewals) and goods")
    rows, problems = benchmark_cases(holdout_cases(EXTENDED, random.Random(os.urandom(8))), POLICIES, key=fresh_key())
    out(table(rows))
    for line in problems:
        out(f"  FAIL {line}")
    report["sections"]["extended"] = {
        f"{policy}/{category}": {key: round(statistics.mean(item[key] for item in items), 6)
                                 for key in ("precision", "recall", "f1", "reward", "steps", "seconds")}
        for (policy, category), items in sorted(rows.items())
    }
    if problems:
        failed.append("extended hold-out oracle/paired checks")

    out("\n== 7. Difficulty levels: easy → medium → hard (more look-alikes, longer books up to ~2000 rows)")
    rows, problems = benchmark_cases(holdout_cases(CURRICULUM, random.Random(os.urandom(8)), by_split=True),
                                     POLICIES, key=fresh_key())
    out(table(rows, order=LEVELS))
    for line in problems:
        out(f"  FAIL {line}")
    report["sections"]["curriculum"] = {
        f"{policy}/{level}": {key: round(statistics.mean(item[key] for item in items), 6)
                              for key in ("precision", "recall", "f1", "reward", "steps", "seconds")}
        for (policy, level), items in sorted(rows.items())
    }
    if problems:
        failed.append("curriculum oracle/paired checks")

    out("\n== 8. Red team: adversarial answers against the grader (closed mode, fresh secret key)")
    key = fresh_key()
    episodes = closed_episodes(random.Random(os.urandom(8)), key, repeats=1)
    result = red_team(episodes, key)
    out(red_team_lines(result))
    report["sections"]["red_team"] = result
    if result["violations"]:
        failed.append("red team")

    out("\n== 9. Step economy: no wasted step, reading less never pays, the largest books fit the limit")
    from .economy import report as economy_report

    result = economy_report(episodes)
    out(economy_lines(result))
    report["sections"]["economy"] = result
    if result["violations"]:
        failed.append("step economy")

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

    if args.curriculum:
        rows, problems = benchmark_cases(holdout_cases(CURRICULUM, random.Random(args.seed), by_split=True),
                                         POLICIES, key=fresh_key())
        out(table(rows, order=LEVELS))
        for line in problems:
            out(f"FAIL {line}")
        return 1 if problems else 0
    if args.extended:
        cases = holdout_cases(EXTENDED, random.Random(args.seed))
        key = fresh_key()
    elif args.suite == HOLDOUT or args.holdout:
        cases = holdout_cases(HOLDOUT, random.Random(args.seed))
        key = fresh_key()
    else:
        cases, key = suite(args.suite), None
    rows, problems = benchmark_cases(cases, POLICIES, key=key)
    out(table(rows))
    for line in problems:
        out(f"FAIL {line}")
    return 1 if problems else 0


def command_failures(args):
    import random
    from dataclasses import replace

    from .dataset import failure_breakdown, suite

    rng = random.Random(args.seed)
    cases = [(split, case_id, category, rng.randrange(2**31), replace(config, keyspace="secret"))
             for split, case_id, category, _, config in suite(
                 CURRICULUM if args.curriculum else EXTENDED if args.extended else HOLDOUT)
             for _ in range(args.repeats)]
    counts = failure_breakdown(cases, ("reconcile", "micro", "expert", "random", "flag-all"),
                               key=b"failure-analysis-key", model_path=args.model)
    out(f"{len(cases)} hold-out episodes (seed {args.seed}, {args.repeats} per config)")
    for name, counter in counts.items():
        out(f"\n== {name}")
        for label, value in sorted(counter.items()):
            out(f"  {label:45s} {value}")
    return 0


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
    if args.agent_image:
        from .environment import AuditEnv
        from .runner import run_container

        if args.policy == "oracle" or not args.closed:
            raise SystemExit("--agent-image needs --closed and a public policy")
        env = AuditEnv(training=False)
        observation, _ = env.load_episode(episode)
        try:
            outcome, trace, seconds = run_container(env, observation, args.agent_image, args.policy)
        except FileNotFoundError:
            raise SystemExit("docker is not installed or not on PATH") from None
    else:
        outcome, trace, seconds = evaluate(episode, args.policy, closed=args.closed)
    if args.trajectory:
        write_lines(args.trajectory, trace)
    out(json.dumps({"episode_id": episode.manifest["episode_id"], "policy": args.policy,
                    "issues": len(episode.issues), "tp": outcome.tp, "fp": outcome.fp, "fn": outcome.fn,
                    "precision": outcome.precision, "recall": outcome.recall, "f1": outcome.f1,
                    "reward": outcome.reward, "steps": len(trace) - 1, "seconds": round(seconds, 4),
                    "termination": outcome.termination}, ensure_ascii=False))
    return 0


def command_train(args):
    from .episode_io import write_json
    from .training import compare, train

    started = time.time()
    model = train(args.suite, args.output, seeds_per_config=args.seeds)
    out(f"trained on {model.meta['train_episodes']} episodes in {time.time() - started:.1f} s; "
        f"threshold {model.threshold} (chosen by mean terminal reward on train)")
    summary = compare(model, args.suite, HOLDOUT, EXTENDED, CURRICULUM)
    write_json(Path(args.output) / "report.json", {"model": model.meta | {"threshold": model.threshold},
                                                   "held_out": summary})
    out(f"{'split':8s} {'policy':10s} {'category':12s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'return':>7s}")
    for row in summary:
        out(f"{row['split']:8s} {row['policy']:10s} {row['category']:12s} {row['episodes']:3d} {row['precision']:6.3f} "
            f"{row['recall']:6.3f} {row['f1']:6.3f} {row['reward']:+7.3f}")
    out(f"model: {Path(args.output) / 'model.json'}, report: {Path(args.output) / 'report.json'}")
    return 0


def command_serve(args):
    from .api import serve

    server = serve(args.host, args.port, args.model)
    out(f"audit_env API on http://{args.host}:{server.server_address[1]}/api/health (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def command_audit(args):
    from .checks import audit
    from .episode_io import FormatError, load_episode

    try:
        episode = load_episode(args.folder)
    except FormatError as exc:
        out(json.dumps({"ok": False, "problems": exc.problems}, ensure_ascii=False, indent=2))
        return 2
    if args.model:
        from .micro import MicroModel

        findings = MicroModel.load(args.model).findings(episode.tables, episode.manifest, episode.policy)
    else:
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
    bench.add_argument("--extended", action="store_true", help="real-practice error types and goods")
    bench.add_argument("--curriculum", action="store_true", help="easy / medium / hard difficulty levels")
    bench.add_argument("--seed", type=int, default=7)
    failures = commands.add_parser("failures", help="measured failure breakdown per policy on hold-out configs")
    failures.add_argument("--repeats", type=int, default=5)
    failures.add_argument("--seed", type=int, default=5)
    failures.add_argument("--model", default="models/model.json")
    failures.add_argument("--extended", action="store_true", help="use the extended hold-out (real-practice types)")
    failures.add_argument("--curriculum", action="store_true", help="use the easy/medium/hard curriculum")
    attacks = commands.add_parser("attacks", help="red team: adversarial agents against the grader")
    attacks.add_argument("--repeats", type=int, default=3, help="seeds per configuration")
    attacks.add_argument("--seed", type=int, default=7, help="episode seeds and the derived secret key")
    attacks.add_argument("--fresh-key", action="store_true", help="random secret key instead of one derived from --seed")
    attacks.add_argument("--report", help="write the full result as JSON")
    economy = commands.add_parser("economy", help="step economy: wasted steps, the cost of reading less, the largest books")
    economy.add_argument("--repeats", type=int, default=3, help="seeds per configuration")
    economy.add_argument("--seed", type=int, default=7, help="episode seeds and the derived secret key")
    economy.add_argument("--fresh-key", action="store_true", help="random secret key instead of one derived from --seed")
    economy.add_argument("--report", help="write the full result as JSON")
    data = commands.add_parser("dataset", help="export open episodes, answers, trajectories, checksums")
    data.add_argument("output", type=Path)
    data.add_argument("--suite", default=DEFAULT_SUITE)
    selfcheck = commands.add_parser("selfcheck", help="prove truth correctness on N random configs")
    selfcheck.add_argument("--n", type=int, default=200)
    selfcheck.add_argument("--seed", type=int, default=1)
    demo = commands.add_parser("demo", help="run one policy on one generated episode")
    demo.add_argument("--seed", type=int, default=202601)
    demo.add_argument("--config", help="JSON config")
    demo.add_argument("--policy", choices=(*POLICIES, "micro"), default="expert")
    demo.add_argument("--closed", action="store_true")
    demo.add_argument("--agent-image", help="run the agent in this local Docker image (no network, read-only)")
    demo.add_argument("--trajectory", type=Path)
    training = commands.add_parser("train", help="fit the micro model on train, compare on held-out data")
    training.add_argument("--suite", default=DEFAULT_SUITE)
    training.add_argument("--output", default="models")
    training.add_argument("--seeds", type=int, default=12)
    audit = commands.add_parser("audit", help="audit a folder with the six episode files (no truth needed)")
    audit.add_argument("folder", type=Path)
    audit.add_argument("--model", type=Path, help="use the trained micro model instead of the rule-based auditor")
    serve = commands.add_parser("serve", help="local HTTP/JSON API for a UI (localhost only)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model", default="models/model.json")
    truth = commands.add_parser("audit-truth", help="independently re-derive a folder's truth")
    truth.add_argument("public", type=Path)
    truth.add_argument("answers", type=Path)
    args = parser.parse_args()
    handlers = {"verify": command_verify, "benchmark": command_benchmark, "dataset": command_dataset,
                "selfcheck": command_selfcheck, "demo": command_demo, "audit": command_audit,
                "audit-truth": command_audit_truth, "train": command_train, "serve": command_serve,
                "failures": command_failures, "attacks": command_attacks, "economy": command_economy}
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
