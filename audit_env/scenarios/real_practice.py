"""Two error types from real practice of a Kazakhstan service and retail-equipment company.

vat_on_top: the price was agreed and paid including VAT, but the invoice adds 16 %
VAT on top (total = price × 1.16). The receivable is overstated by the added VAT,
which is the declared amount; revenue and VAT payable are misstated too (net and VAT
are split from the inflated total). The customer's payment leaves exactly that VAT open.

uncollected_service: a monthly service was renewed (a new invoice issued) after the
previous invoice of the same customer and service became overdue with no payment at
all — "extended on request, the money was never collected".
"""
from dataclasses import replace
from datetime import datetime, time, timedelta

from ..contracts import UNCOLLECTED, VAT_ON_TOP
from ..world import MANAGERS


def vat_on_top(rng, world, overrides, act_key, payment_key, Spec):
    act = world.acts[act_key]
    lines = tuple(replace(line, vat_rate="0.16", vat_on_top=True) for line in act.lines)
    overrides.replace_acts[act_key] = replace(act, lines=lines)
    return [Spec(VAT_ON_TOP, "vat_on_top", acts=(act_key,), payments=(payment_key,),
                 cause="НДС 16 % начислен сверху согласованной цены вместо включения в цену; покупатель оплатил "
                       "согласованную сумму, разница висит в дебиторке, выручка и НДС к уплате искажены.")]


def uncollected(rng, world, overrides, act_key, Spec):
    act = world.acts[act_key]
    day = act.due_at.astimezone(world.tz).date() + timedelta(days=rng.randrange(1, 5))
    issued = datetime.combine(day, time(rng.randrange(10, 18), rng.randrange(60), rng.randrange(60)), world.tz)
    if issued >= world.period_end:
        raise RuntimeError("renewal must be issued inside the period")
    start = act.service_end + timedelta(days=1)
    lines = tuple(replace(line, period_start=start, period_end=start + timedelta(days=29)) for line in act.lines)
    renewal = replace(act, key=f"{act_key}R", issued_at=issued, lines=lines, origin="renewal",
                      manager=rng.choice(MANAGERS))
    overrides.extra_acts[renewal.key] = renewal
    return [Spec(UNCOLLECTED, "uncollected_service", acts=(act_key, renewal.key),
                 cause="Подписку продлили новым счётом, хотя предыдущий счёт того же клиента просрочен и не оплачен.")]

