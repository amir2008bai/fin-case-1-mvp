"""Duplicate invoice: a second manager issues a second invoice for the same supply.

Both invoices are posted, so revenue and the receivable are overstated by the
duplicate total. Legitimate look-alikes: the same monthly fee in other service
periods, equal prices across customers, multi-line invoices.
"""
from dataclasses import replace
from datetime import datetime, time, timedelta

from ..contracts import DUPLICATE
from ..world import MANAGERS


def apply(rng, world, overrides, act_key, Spec):
    act = world.acts[act_key]
    orig_day = act.issued_at.astimezone(world.tz).date()
    days = [act.service_end + timedelta(days=offset) for offset in range(0, 6)]
    other_days = [d for d in days if d != orig_day] or days
    day = rng.choice(other_days)
    issued = datetime.combine(day, time(rng.randrange(10, 19), rng.randrange(60), rng.randrange(60)), world.tz)
    manager = rng.choice([name for name in MANAGERS if name != act.manager])
    duplicate = replace(act, key=f"{act_key}D", issued_at=issued, manager=manager, origin="duplicate")
    overrides.extra_acts[duplicate.key] = duplicate
    return [Spec(DUPLICATE, "duplicate_invoice", acts=(act_key, duplicate.key),
                 cause="Второй менеджер выставил повторный счёт на ту же услугу и период; оба проведены в выручку.")]
