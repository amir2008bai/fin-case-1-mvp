"""Missing or partial payment allocation.

A receipt cites an open invoice whose revenue is already recognised, but the
bookkeeper credits all or part of it to advances received (3510) instead of the
receivable (1210). Legitimate look-alikes: prepayments before the service ends,
overpayments, unreferenced receipts, opening advances and their offsets.
"""
from ..contracts import ALLOCATION


def apply(rng, world, overrides, act_key, payment_key, Spec):
    payment = world.payments[payment_key]
    full = rng.randrange(2) == 0
    parked = payment.amount_minor if full else payment.amount_minor * rng.randrange(30, 71) // 100
    overrides.parked[payment_key] = parked
    overrides.parked_ref[payment_key] = bool(rng.randrange(2))
    return [Spec(ALLOCATION, f"payment_allocation.{'missing' if full else 'partial'}",
                 acts=(act_key,), payments=(payment_key,),
                 cause="Поступление со ссылкой на открытый счёт отнесено в авансы полученные вместо погашения дебиторки.")]
