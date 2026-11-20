"""Exchange rates and the single published conversion rule."""
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal

from .domain import BASE_CURRENCY, MINOR_UNITS

# KZT per one unit of the source currency, two decimals as the national bank quotes.
RATE_RANGE = {"USD": (Decimal("470.00"), Decimal("540.00")), "RUB": (Decimal("5.40"), Decimal("6.60"))}
DAILY_STEPS = {"USD": (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20), "RUB": (1, 2, 3)}
CENT = Decimal("0.01")


def convert(amount_minor, currency, rate, base=BASE_CURRENCY):
    """Base-currency minor units: amount × rate, rounded once, half up."""
    if currency == base:
        return amount_minor
    scale = Decimal(10) ** (MINOR_UNITS[base] - MINOR_UNITS[currency])
    value = Decimal(amount_minor) * Decimal(rate) * scale
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def build_rates(rng, currencies, first, last):
    """A daily rate for every calendar day; consecutive days always differ."""
    rates = {}
    for currency in currencies:
        low, high = RATE_RANGE[currency]
        span = int((high - low) / CENT)
        rate = low + CENT * rng.randrange(span // 4, 3 * span // 4)
        day = first
        while day <= last:
            rates[(day, currency)] = rate
            step = CENT * rng.choice(DAILY_STEPS[currency])
            if rate + step > high or (rate - step >= low and rng.randrange(2)):
                step = -step
            rate += step
            day += timedelta(days=1)
    return rates
