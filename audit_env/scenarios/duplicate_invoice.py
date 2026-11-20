"""Duplicate invoice: a second manager issues a second invoice for the same supply.

Both invoices are posted, so revenue and the receivable are overstated by the
duplicate total. Legitimate look-alikes: the same monthly fee in other service
periods, equal prices across customers, multi-line invoices.
"""
from dataclasses import replace
from datetime import timedelta

from ..contracts import DUPLICATE
from ..world import MANAGERS


def apply(rng, world, overrides, act_key, Spec):
    act = world.acts[act_key]
    shift = timedelta(days=rng.randrange(1, 6), hours=rng.randrange(-4, 5))
    issued = act.issued_at + shift if rng.randrange(2) else act.issued_at - shift
    manager = rng.choice([name for name in MANAGERS if name != act.manager])
    duplicate = replace(act, key=f"{act_key}D", issued_at=issued, manager=manager, origin="duplicate")
    overrides.extra_acts[duplicate.key] = duplicate
    return [Spec(DUPLICATE, "duplicate_invoice", acts=(act_key, duplicate.key),
                 cause="Второй менеджер выставил повторный счёт на ту же услугу и период; оба проведены в выручку.")]
