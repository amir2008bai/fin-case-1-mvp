"""Wrong FX date or rate.

An invoice must be converted at the rate of its service end date and a receipt
at the rate of its local operation date. The bookkeeper uses the issue date, the
previous day, the invoice date for a receipt, or mistypes the rate. Legitimate
look-alikes: ordinary exchange differences (6250/7430) on every foreign receipt.
"""
from datetime import timedelta
from decimal import Decimal

from ..contracts import FX_RATE
from ..fx import convert
from ..world import MIN_FX_GAP_MINOR


def gap(world, amount, currency, used_rate, correct_day):
    return abs(convert(amount, currency, used_rate) - convert(amount, currency, world.rate(correct_day, currency)))


def typo_rate(rng, world, amount, currency, correct_day):
    rate = world.rate(correct_day, currency)
    digits = f"{rate:.2f}".replace(".", "")
    options = []
    for index in range(len(digits) - 1):
        if digits[index] != digits[index + 1]:
            swapped = digits[:index] + digits[index + 1] + digits[index] + digits[index + 2:]
            value = Decimal(swapped[:-2] + "." + swapped[-2:])
            if value > 0 and gap(world, amount, currency, value, correct_day) >= MIN_FX_GAP_MINOR:
                options.append(value)
    if options:
        return rng.choice(options)
    return rate + Decimal(rng.choice((-3, -2, 2, 3)))


def date_options(world, amount, currency, correct_day, days):
    result = []
    for day in days:
        if day != correct_day and (day, currency) in world.rates and day not in result:
            if gap(world, amount, currency, world.rate(day, currency), correct_day) >= MIN_FX_GAP_MINOR:
                result.append(day)
    return result


def apply(rng, world, overrides, variant, act_key, payment_key, Spec):
    if variant == "act":
        act = world.acts[act_key]
        correct = act.service_end
        days = date_options(world, act.total_minor, act.currency, correct,
                            (world.local_date(act.issued_at), correct - timedelta(days=1)))
        if days and rng.randrange(3):
            overrides.act_rate_date[act_key] = rng.choice(days)
            scenario, cause = "fx_rate.invoice_wrong_date", "Счёт пересчитан по курсу не той даты (нужна дата окончания услуги)."
        else:
            overrides.act_rate[act_key] = typo_rate(rng, world, act.total_minor, act.currency, correct)
            scenario, cause = "fx_rate.invoice_wrong_rate", "При пересчёте счёта введён неверный курс."
        return [Spec(FX_RATE, scenario, acts=(act_key,), cause=cause)]
    payment = world.payments[payment_key]
    act = world.acts[act_key]
    correct = world.local_date(payment.instant)
    days = date_options(world, payment.amount_minor, payment.currency, correct,
                        (correct - timedelta(days=1), act.service_end, correct + timedelta(days=1)))
    if days and rng.randrange(4):
        overrides.payment_rate_date[payment_key] = rng.choice(days)
        scenario, cause = "fx_rate.receipt_wrong_date", "Поступление пересчитано по курсу не той даты (нужна дата операции)."
    else:
        overrides.payment_rate[payment_key] = typo_rate(rng, world, payment.amount_minor, payment.currency, correct)
        scenario, cause = "fx_rate.receipt_wrong_rate", "При пересчёте поступления введён неверный курс."
    return [Spec(FX_RATE, scenario, acts=(act_key,), payments=(payment_key,), cause=cause)]
