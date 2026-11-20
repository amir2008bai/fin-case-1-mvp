"""Step economy: the cheapest complete read, and what reading less costs.

Ledger rows are ordered by opaque row IDs, so the postings of any document can sit on any page.
An auditor that wants to see every posting has to read every page. `minimal_steps` counts the pages
of a complete read at the maximum page size, plus the submit. Every page left unread saves 0.002
reward and puts whole findings at risk; `skim_curve` measures that trade on closed episodes.
"""
import math
from statistics import mean

from .agents import ScanPolicy, create_policy, expert
from .attacks import play
from .contracts import MAX_PAGE, MAX_STEPS, Config

DOCUMENT_TABLES = ("invoices", "payments")
READ_TABLES = ("invoices", "payments", "fx", "ledger")
SPARE_STEPS = 40  # the largest allowed books must leave at least this many steps unused


def pages(rows):
    return math.ceil(rows / MAX_PAGE)


def foreign(tables, base):
    return any(row["currency"] != base for name in DOCUMENT_TABLES for row in tables[name])


def minimal_steps(episode):
    """Steps of a complete read: every page of every table a finding can depend on, plus the submit.

    The exchange-rate table counts only when an invoice or a payment is in a foreign currency.
    """
    tables = episode.tables
    needed = [name for name in READ_TABLES if name != "fx" or foreign(tables, episode.manifest["base_currency"])]
    return sum(pages(len(tables[name])) for name in needed) + 1


class SkimPolicy(ScanPolicy):
    """The honest rules (expert) applied to a partial read: only the first `keep(total)` ledger pages."""

    def __init__(self, keep):
        super().__init__(expert)
        self.keep = keep

    def wants_more(self, table):
        if table != "ledger":
            return True
        return self.pages["ledger"] < self.keep(pages(self.manifest["counts"]["ledger"]))


SKIMS = {
    "complete read": ("все страницы книги", lambda total: total),
    "skip 1 page": ("книга без последней страницы", lambda total: total - 1),
    "skim 90%": ("90 % страниц книги", lambda total: math.floor(0.9 * total)),
    "skim 75%": ("75 % страниц книги", lambda total: math.floor(0.75 * total)),
    "skim 50%": ("половина страниц книги", lambda total: math.floor(0.5 * total)),
    "no ledger": ("только документы, книгу не читать", lambda total: 0),
}


def _summary(name, description, outcomes, clean):
    dirty = [item for item, is_clean in zip(outcomes, clean) if not is_clean]
    spotless = [item for item, is_clean in zip(outcomes, clean) if is_clean]
    return {
        "name": name, "description": description,
        "clean_reward": round(mean(item["reward"] for item in spotless), 4) if spotless else None,
        "error_reward": round(mean(item["reward"] for item in dirty), 4) if dirty else None,
        "reward": round(mean(item["reward"] for item in outcomes), 4),
        "tp": sum(item["tp"] for item in outcomes), "fp": sum(item["fp"] for item in outcomes),
        "fn": sum(item["fn"] for item in outcomes),
        "steps": round(mean(item["steps"] for item in outcomes), 1),
    }


def skim_curve(episodes):
    """Honest rules on less and less of the ledger. Reading less must lose reward on average."""
    clean = [not episode.issues for episode in episodes]
    rows = [_summary(name, description, [play(episode, lambda keep=keep: SkimPolicy(keep)) for episode in episodes],
                     clean)
            for name, (description, keep) in SKIMS.items()]
    complete, violations = rows[0], []
    for row in rows[1:]:
        if row["error_reward"] is not None and not row["error_reward"] < complete["error_reward"]:
            violations.append(f"{row['name']}: reading less does not lose reward on erroneous books")
        if not row["reward"] < complete["reward"]:
            violations.append(f"{row['name']}: reading less does not lose reward over all episodes")
    return rows, violations


def waste(episodes, policies=("expert", "micro")):
    """Steps above the complete read, per policy. A complete auditor wastes none and skips nothing."""
    result, violations = {}, []
    for name in policies:
        extra = [play(episode, lambda name=name: create_policy(name))["steps"] - minimal_steps(episode)
                 for episode in episodes]
        result[name] = {"episodes": len(extra), "complete_read_steps": round(mean(map(minimal_steps, episodes)), 1),
                        "wasted_steps": sum(value for value in extra if value > 0),
                        "skipped_steps": -sum(value for value in extra if value < 0)}
        if any(extra):
            violations.append(f"{name}: steps differ from the complete read on {sum(map(bool, extra))} episodes")
    return result, violations


def largest_books(seeds=range(3)):
    """The biggest books the config allows: a complete read still fits the step limit with steps to spare."""
    from .generator import generate

    config = Config(keyspace="secret", difficulty="hard", acts=80, customers=60, foreign_percent=50,
                    goods_percent=80, filler=800, errors={"duplicate_invoice": 3, "payment_allocation": 3,
                                                          "fx_rate": 3, "period_cutoff": 3})
    worst = max((generate(seed, config, key=b"largest-books") for seed in seeds), key=minimal_steps)
    steps = minimal_steps(worst)
    return {"counts": worst.manifest["counts"], "complete_read_steps": steps, "step_limit": MAX_STEPS,
            "ok": steps <= MAX_STEPS - SPARE_STEPS}


def report(episodes):
    """Everything the economy section of verify prints; violations make it fail."""
    curve, violations = skim_curve(episodes)
    spent, problems = waste(episodes)
    largest = largest_books()
    if not largest["ok"]:
        problems.append(f"the largest books need {largest['complete_read_steps']} steps of {MAX_STEPS}")
    return {"episodes": len(episodes), "waste": spent, "skim_curve": curve, "largest_books": largest,
            "violations": problems + violations}
