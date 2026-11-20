"""Error scenarios: each turns a reserved target into wrong bookkeeping decisions.

A scenario only edits Overrides (or adds a duplicate act). The truth for each
issue is resolved afterwards from the books with errors, so the declared amount
and evidence always describe rows the agent can actually read.
"""
from dataclasses import dataclass, field
from datetime import timedelta

from ..contracts import ALLOCATION, CUTOFF, DUPLICATE, FX_BOUNDARY, FX_PARTIAL, FX_RATE, TOLERANCE_MINOR
from ..books import Overrides
from ..domain import BASE_CURRENCY
from ..fx import convert
from . import duplicate_invoice, fx_rate, interactions, payment_allocation, period_cutoff

FX_EVIDENCE_WINDOW = timedelta(days=7)


@dataclass
class Spec:
    type: str
    scenario: str
    acts: tuple = ()
    payments: tuple = ()
    cause: str = ""
    group: int = 0
    amount: int = 0
    currency: str = BASE_CURRENCY
    affected: tuple = ()
    evidence_sets: tuple = ()
    relevant: tuple = field(default_factory=tuple)


def inject(rng, world, config):
    overrides = Overrides()
    specs = []
    candidates = world.candidates
    group = 0

    def add(new_specs):
        nonlocal group
        group += 1
        for spec in new_specs:
            spec.group = group
            specs.append(spec)

    for act_key in candidates["dup"]:
        add(duplicate_invoice.apply(rng, world, overrides, act_key, Spec))
    for act_key, payment_key in candidates["alloc"]:
        add(payment_allocation.apply(rng, world, overrides, act_key, payment_key, Spec))
    for variant, act_key, payment_key in candidates["fx"]:
        add(fx_rate.apply(rng, world, overrides, variant, act_key, payment_key, Spec))
    for variant, act_key, payment_key in candidates["cut"]:
        add(period_cutoff.apply(rng, world, overrides, variant, act_key, payment_key, Spec))
    for act_key, payment_key in candidates[FX_BOUNDARY]:
        add(interactions.boundary(rng, world, overrides, act_key, payment_key, Spec))
    for act_key, payment_key in candidates[FX_PARTIAL]:
        add(interactions.partial(rng, world, overrides, act_key, payment_key, Spec))
    return overrides, specs


# ---- truth resolution against the books with errors -------------------------
def _revenue_rows(rows, act_key):
    return [row for row in rows if row.act == act_key and row.account == "6010"]


def _receivable_rows(rows, act_key):
    return [row for row in rows if row.act == act_key and row.account == "1210" and row.debit
            and row.key.startswith(f"REC:{act_key}:")]


def _fx_refs(world, currency, days):
    low, high = min(days) - FX_EVIDENCE_WINDOW, max(days) + FX_EVIDENCE_WINDOW
    return tuple(("fx", (day, currency)) for (day, code) in sorted(world.rates)
                 if code == currency and low <= day <= high)


def resolve(spec, world, overrides, rows):
    acts = {**world.acts, **overrides.extra_acts}
    by_key = {row.key: row for row in rows}
    related = tuple(("row", row.key) for row in rows
                    if row.act in spec.acts or row.payment in spec.payments)
    docs = tuple(("doc", key) for key in (*spec.acts, *spec.payments))
    spec.relevant = docs + related
    if spec.type == DUPLICATE:
        first, second = (acts[key] for key in spec.acts)
        spec.amount, spec.currency = first.total_minor, first.currency
        spec.affected = (("invoice", first.key), ("invoice", second.key))
        revenue = [("row", row.key) for key in spec.acts for row in _revenue_rows(rows, key)]
        receivable = [("row", row.key) for key in spec.acts for row in _receivable_rows(rows, key)]
        sets = [docs + tuple(revenue)]
        if len(docs) + len(receivable) <= 10:
            sets.append(docs + tuple(receivable))
        spec.evidence_sets = tuple(sets)
    elif spec.type == ALLOCATION:
        payment_key = spec.payments[0]
        parked = by_key[f"PAY:{payment_key}:park"]
        spec.amount, spec.currency = parked.credit, BASE_CURRENCY
        spec.affected = (("payment", payment_key),)
        spec.evidence_sets = ((("doc", payment_key), ("doc", spec.acts[0]),
                               ("row", f"PAY:{payment_key}:bank"), ("row", parked.key)),)
    elif spec.type == FX_RATE and spec.payments:
        payment = world.payments[spec.payments[0]]
        correct_day = world.local_date(payment.instant)
        posted = by_key[f"PAY:{payment.key}:bank"].debit
        correct = convert(payment.amount_minor, payment.currency, world.rate(correct_day, payment.currency))
        spec.amount, spec.currency = abs(posted - correct), BASE_CURRENCY
        spec.affected = (("payment", payment.key),)
        spec.evidence_sets = ((("doc", payment.key), ("fx", (correct_day, payment.currency)),
                               ("row", f"PAY:{payment.key}:bank")),)
        used = overrides.payment_rate_date.get(payment.key, correct_day)
        spec.relevant += _fx_refs(world, payment.currency, (correct_day, used))
    elif spec.type == FX_RATE:
        act = acts[spec.acts[0]]
        revenue = _revenue_rows(rows, act.key)
        posted = sum(row.credit for row in revenue)
        correct = convert(act.total_minor, act.currency, world.rate(act.service_end, act.currency))
        spec.amount, spec.currency = abs(posted - correct), BASE_CURRENCY
        spec.affected = (("invoice", act.key),)
        fx_ref = ("fx", (act.service_end, act.currency))
        spec.evidence_sets = (
            (("doc", act.key), fx_ref, *(("row", row.key) for row in revenue)),
            (("doc", act.key), fx_ref, *(("row", row.key) for row in _receivable_rows(rows, act.key))),
        )
        used = overrides.act_rate_date.get(act.key, act.service_end)
        spec.relevant += _fx_refs(world, act.currency, (act.service_end, used, world.local_date(act.issued_at)))
    elif spec.type == CUTOFF and spec.payments:
        payment_key = spec.payments[0]
        spec.amount = by_key[f"PAY:{payment_key}:bank"].debit
        spec.currency = BASE_CURRENCY
        spec.affected = (("payment", payment_key),)
        spec.evidence_sets = ((("doc", payment_key), ("row", f"PAY:{payment_key}:bank")),)
    elif spec.type == CUTOFF:
        act_key = spec.acts[0]
        revenue = _revenue_rows(rows, act_key)
        spec.amount, spec.currency = sum(row.credit for row in revenue), BASE_CURRENCY
        spec.affected = (("invoice", act_key),)
        spec.evidence_sets = ((("doc", act_key), *(("row", row.key) for row in revenue)),)
    else:
        raise ValueError(f"unknown spec {spec.type}")
    spec.tolerance = TOLERANCE_MINOR[spec.type]
    return spec
