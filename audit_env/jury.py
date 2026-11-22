"""One-command jury evaluation protocol (Section 06 of the hackathon brief).

Runs the 4 organizer policies (`oracle`, `reconcile`, `random`, `flag-all`) on
the 20 closed hold-out episodes from `configs/holdout-template.v2.json`
(5 clean, 5 interacting, 4 single, 6 multi) plus 20 paired `generator.transform`
checks verifying that renaming IDs and shuffling row order preserve oracle F1
and reward. Reuses `dataset.benchmark_cases` and `dataset.suite` directly.
"""
from dataclasses import replace
import hashlib
import random
from statistics import mean

from .dataset import benchmark_cases, suite

JURY_POLICIES = ("oracle", "reconcile", "random", "flag-all")


def derived_key(seed):
    """Secret key fixed by --seed, matching `audit_env.__main__.derived_key`."""
    return hashlib.sha256(f"red-team:{seed}".encode()).digest()


def _holdout_cases(path, seed):
    rng = random.Random(seed)
    for split, case_id, category, case_seed, config in suite(path):
        ep_seed = case_seed if case_seed is not None else rng.randrange(2**31)
        yield (split, case_id, category, ep_seed, replace(config, keyspace="secret"))


def run_jury(holdout_path, *, seed=7, key=None):
    """Execute the Section 06 jury protocol on 20 closed episodes + 20 paired checks."""
    secret_key = derived_key(seed) if key is None else key
    cases = list(_holdout_cases(holdout_path, seed))
    rows, failures = benchmark_cases(cases, JURY_POLICIES, paired=True, key=secret_key)

    policies_summary = {}
    categories_summary = {}
    for policy in JURY_POLICIES:
        all_items = []
        by_cat = {}
        for (pol, cat), items in sorted(rows.items()):
            if pol != policy:
                continue
            all_items.extend(items)
            by_cat[cat] = {
                "episodes": len(items),
                "precision": round(mean(x["precision"] for x in items), 4),
                "recall": round(mean(x["recall"] for x in items), 4),
                "f1": round(mean(x["f1"] for x in items), 4),
                "reward": round(mean(x["reward"] for x in items), 4),
                "steps": round(mean(x["steps"] for x in items), 2),
                "ms": round(mean(x["seconds"] * 1000 for x in items), 2),
            }
        categories_summary[policy] = by_cat
        policies_summary[policy] = {
            "episodes": len(all_items),
            "precision": round(mean(x["precision"] for x in all_items), 4) if all_items else 0.0,
            "recall": round(mean(x["recall"] for x in all_items), 4) if all_items else 0.0,
            "f1": round(mean(x["f1"] for x in all_items), 4) if all_items else 0.0,
            "reward": round(mean(x["reward"] for x in all_items), 4) if all_items else 0.0,
            "steps": round(mean(x["steps"] for x in all_items), 2) if all_items else 0.0,
            "ms": round(mean(x["seconds"] * 1000 for x in all_items), 2) if all_items else 0.0,
            "clean_reward": by_cat.get("clean", {}).get("reward"),
        }

    paired_failures = [msg for msg in failures if "paired check" in msg]
    oracle_failures = [msg for msg in failures if "oracle F1" in msg]
    clean_count = sum(1 for _, _, cat, _, _ in cases if cat.split(":")[0] == "clean")
    interacting_count = sum(1 for _, _, cat, _, _ in cases if cat.split(":")[0] == "interacting")

    return {
        "seed": seed,
        "episodes": len(cases),
        "clean_episodes": clean_count,
        "interacting_episodes": interacting_count,
        "policies": policies_summary,
        "categories": categories_summary,
        "paired_checks": {
            "total": len(cases),
            "passed": len(cases) - len(paired_failures),
            "failed": len(paired_failures),
        },
        "oracle_all_f1_one": len(oracle_failures) == 0,
        "failures": failures,
    }


def format_jury_table(result):
    """Format the Section 06 jury evaluation summary as a reproducible text table."""
    lines = [
        f"Jury protocol (Section 06): {result['episodes']} closed episodes "
        f"({result['clean_episodes']} clean, {result['interacting_episodes']} interacting, seed {result['seed']})",
        f"{'policy':10s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'return':>8s} {'steps':>6s} {'ms':>7s}",
    ]
    for policy in JURY_POLICIES:
        s = result["policies"][policy]
        lines.append(
            f"{policy:10s} {s['episodes']:3d} {s['precision']:6.3f} {s['recall']:6.3f} "
            f"{s['f1']:6.3f} {s['reward']:+8.3f} {s['steps']:6.1f} {s['ms']:7.1f}"
        )
    paired = result["paired_checks"]
    lines.append(
        f"paired transform checks: {paired['passed']}/{paired['total']} passed; "
        f"oracle F1=1 on all {result['episodes']}: {'YES' if result['oracle_all_f1_one'] else 'NO'}"
    )
    lines.append("failures: " + (str(len(result["failures"])) if result["failures"] else "none"))
    lines.extend(f"  FAIL {msg}" for msg in result["failures"])
    return "\n".join(lines)
