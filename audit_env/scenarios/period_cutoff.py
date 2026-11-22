"""Wrong period cut-off.

Revenue belongs to the period of the service end date; a receipt belongs to the
period of its operation instant in the company time zone. Errors: a prior-period
service booked on its (in-period) issue date, a next-period service booked on its
issue date, or a receipt after the period end booked by its foreign wall-clock date.
Legitimate look-alikes: late invoices with opening balances, invoices issued after
the period end, receipts whose shown wall date is outside the period but whose
instant is inside (and vice versa, not booked).
"""
from datetime import datetime, time

from ..contracts import CUTOFF
from ..world import parse_offset


def naive_booking(world, payment):
    wall = payment.instant.astimezone(parse_offset(payment.shown_offset))
    return wall.replace(tzinfo=world.tz)


def apply(rng, world, overrides, variant, act_key, payment_key, Spec):
    if variant in ("act_prior", "act_next"):
        act = world.acts[act_key]
        # Booked on the issue date, at 00:00 like every other recognition: the issue time of day on the
        # revenue rows gave these errors away (AUROC 1.0 on "posting time is not midnight").
        issue_day = act.issued_at.astimezone(world.tz).date()
        overrides.act_booked_at[act_key] = datetime.combine(issue_day, time(0), world.tz)
        cause = ("Услуга прошлого периода признана в выручке по дате выписки счёта." if variant == "act_prior"
                 else "Услуга следующего периода признана в выручке по дате выписки счёта.")
        return [Spec(CUTOFF, f"period_cutoff.{variant}", acts=(act_key,), cause=cause)]
    payment = world.payments[payment_key]
    booked = naive_booking(world, payment)
    if not world.in_period(booked):
        raise RuntimeError("naive booking must fall inside the period")
    overrides.payment_booked_at[payment_key] = booked
    return [Spec(CUTOFF, "period_cutoff.receipt_after_end", acts=(act_key,), payments=(payment_key,),
                 cause="Поступление после окончания периода проведено в периоде по дате в часовом поясе плательщика.")]
