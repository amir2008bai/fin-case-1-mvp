"""Trusted generation: build balanced history, then apply recorded changes."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from random import Random

from . import __version__
from .contracts import SCHEMA_VERSION, Config, Episode
from .domain import CURRENCY
from .history import assert_balanced, build_history
from .scenarios import duplicate_invoice, payment_allocation


def generate(seed: int, config: Config) -> Episode:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = Random(seed)

    def identifier(kind):
        return f"{kind}_{rng.getrandbits(96):024x}"

    episode_id = identifier("episode")
    zone = timezone(timedelta(hours=rng.choice([-3, 0, 3, 5, 6])))
    start = datetime(2026, rng.randint(1, 12), 1, tzinfo=zone)
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    history = build_history(rng, identifier, start, config.invoice_count)
    issues, changes = [], []
    eligible_invoices = history.invoices[5:]
    rng.shuffle(eligible_invoices)
    duplicate_targets = eligible_invoices[:config.duplicate_count]
    duplicate_ids = {invoice["invoice_id"] for invoice in duplicate_targets}
    eligible_payments = [
        payment for payment in history.eligible_payments if payment["invoice_refs"][0] not in duplicate_ids
    ]
    rng.shuffle(eligible_payments)
    for invoice in duplicate_targets:
        issue, change = duplicate_invoice.inject(history, invoice, identifier)
        issues.append(issue)
        changes.append(change)
    for payment in eligible_payments[:config.allocation_count]:
        original = payment["amount_minor"]
        missing = original if rng.randrange(2) else original // 3
        issue, change = payment_allocation.inject(history, payment, missing, identifier)
        issues.append(issue)
        changes.append(change)

    assert_balanced(history.ledger)
    tables = {
        "ledger": history.ledger,
        "invoices": history.invoices,
        "payments": history.payments,
        "fx": [],
    }
    for rows in tables.values():
        rng.shuffle(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": CURRENCY,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "timezone": start.strftime("%z")[:3] + ":" + start.strftime("%z")[3:],
        "counts": {table: len(rows) for table, rows in tables.items()},
        "document_count": len(history.invoices) + len(history.payments),
        "minor_units": {CURRENCY: 2},
        "rounding": "HALF_EVEN",
    }
    policy = {
        "fx_date": "transaction_local_date",
        "period_rule": "start_inclusive_end_exclusive",
        "rounding": "HALF_EVEN",
        "invoice_rule": "Один счёт порождает одну пару расход / задолженность.",
        "payment_rule": (
            "В этой компании однозначная ссылка в платеже означает полное отнесение "
            "суммы платежа к указанному счёту. Частично оплачивать сам счёт можно."
        ),
    }
    return Episode(manifest, policy, tables, tuple(issues), {
        "seed": seed, "config": asdict(config), "generator_version": __version__, "changes": changes,
    })


def transform(episode: Episode, seed: int) -> Episode:
    """Remap opaque IDs and public order without changing accounting semantics."""
    rng = Random(seed)
    opaque = {}
    prefixes = {"episode", "row", "doc", "entity", "invoice", "payment", "issue", "fx"}

    def remap(value):
        if isinstance(value, str):
            kind, _, suffix = value.partition("_")
            if kind in prefixes and len(suffix) == 24:
                if value not in opaque:
                    opaque[value] = f"{kind}_{rng.getrandbits(96):024x}"
                return opaque[value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(remap(item) for item in value)
        return value

    tables = remap(episode.tables)
    for rows in tables.values():
        rng.shuffle(rows)
    issues = tuple(
        replace(issue, issue_id=remap(issue.issue_id), affected_ids=remap(issue.affected_ids),
                evidence_sets=remap(issue.evidence_sets))
        for issue in episode.issues
    )
    provenance = remap(episode.provenance) | {"transform_seed": seed}
    return Episode(remap(episode.manifest), deepcopy(episode.policy), tables, issues, provenance)
