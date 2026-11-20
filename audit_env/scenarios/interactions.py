"""Interacting errors: two issues on one foreign receipt, each with its own amount.

IX-A fx_boundary_payment: a partial USD/RUB receipt arrives just after the period
end, its wall-clock date in the payer's offset is still inside the period; the
bookkeeper books it in the period AND converts it at that naive date's rate.
-> period_cutoff (receipt) + fx_rate (receipt).

IX-B fx_partial_allocation: a foreign receipt for an open invoice is partly parked
in advances AND converted at the invoice date's rate instead of the operation date.
-> payment_allocation + fx_rate (receipt).
"""
from ..contracts import ALLOCATION, CUTOFF, FX_BOUNDARY, FX_PARTIAL, FX_RATE
from .period_cutoff import naive_booking


def boundary(rng, world, overrides, act_key, payment_key, Spec):
    payment = world.payments[payment_key]
    booked = naive_booking(world, payment)
    if not world.in_period(booked):
        raise RuntimeError("naive booking must fall inside the period")
    overrides.payment_booked_at[payment_key] = booked
    overrides.payment_rate_date[payment_key] = booked.date()
    return [
        Spec(CUTOFF, FX_BOUNDARY, acts=(act_key,), payments=(payment_key,),
             cause="Частичная валютная оплата после конца периода проведена в периоде по дате плательщика."),
        Spec(FX_RATE, FX_BOUNDARY, acts=(act_key,), payments=(payment_key,),
             cause="Та же оплата пересчитана по курсу «наивной» даты вместо местной даты операции."),
    ]


def partial(rng, world, overrides, act_key, payment_key, Spec):
    payment = world.payments[payment_key]
    act = world.acts[act_key]
    overrides.parked[payment_key] = payment.amount_minor * rng.randrange(40, 61) // 100
    overrides.parked_ref[payment_key] = bool(rng.randrange(2))
    overrides.payment_rate_date[payment_key] = act.service_end
    return [
        Spec(ALLOCATION, FX_PARTIAL, acts=(act_key,), payments=(payment_key,),
             cause="Часть валютного поступления по открытому счёту отнесена в авансы полученные."),
        Spec(FX_RATE, FX_PARTIAL, acts=(act_key,), payments=(payment_key,),
             cause="Поступление пересчитано по курсу даты счёта вместо даты операции."),
    ]
