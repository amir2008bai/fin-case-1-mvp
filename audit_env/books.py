"""Posting engine: hidden events plus optional bookkeeping overrides -> ledger rows.

Clean books are post(world, Overrides()). An audit error is a wrong bookkeeping
decision expressed as an override (wrong rate date, wrong booking date, part of a
receipt parked in advances) or an extra document (a duplicate act). Every journal
entry balances, so both versions of the books balance by construction.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from .domain import ADVANCES, BANK, BASE_CURRENCY, FX_GAIN, FX_LOSS, OPENING, RECEIVABLE, REVENUE, VAT
from .fx import convert


@dataclass
class Overrides:
    extra_acts: dict = field(default_factory=dict)         # act key -> Act added to the books
    act_rate_date: dict = field(default_factory=dict)      # act key -> date whose rate was used
    act_rate: dict = field(default_factory=dict)           # act key -> Decimal rate actually used
    payment_rate_date: dict = field(default_factory=dict)  # payment key -> date whose rate was used
    payment_rate: dict = field(default_factory=dict)
    act_booked_at: dict = field(default_factory=dict)      # act key -> instant revenue was booked
    payment_booked_at: dict = field(default_factory=dict)  # payment key -> instant the receipt was booked
    parked: dict = field(default_factory=dict)             # payment key -> amount credited to 3510 instead of 1210
    parked_ref: dict = field(default_factory=dict)         # payment key -> parked row keeps the invoice_id


@dataclass(frozen=True)
class Row:
    key: str
    at: datetime
    account: str
    debit: int
    credit: int
    entity: str | None
    act: str | None
    payment: str | None


def assert_balanced(rows):
    totals = defaultdict(int)
    for row in rows:
        if row.debit < 0 or row.credit < 0 or (row.debit and row.credit):
            raise ValueError(f"malformed row {row.key}")
        totals["KZT"] += row.debit - row.credit
    if any(totals.values()):
        raise ValueError("unbalanced ledger")


class Books:
    def __init__(self, world, overrides):
        self.world = world
        self.ov = overrides
        self.acts = {**world.acts, **overrides.extra_acts}
        self.rows = []
        self.outstanding = {}  # act key -> open amount in the act currency
        self.carrying = {}     # act key -> open receivable in KZT
        self.prepaid = defaultdict(list)  # act key -> [(payment key, KZT held in 3510)]
        self.unmatched = {}    # unreferenced payment key -> KZT still held in 3510
        self.opening_left = dict(world.opening_advances)

    # ---- primitives ----------------------------------------------------
    def entry(self, tag, at, entity, legs):
        if sum(debit - credit for _, _, debit, credit, _, _ in legs):
            raise ValueError(f"unbalanced entry {tag}")
        for label, account, debit, credit, act, payment in legs:
            if debit or credit:
                self.rows.append(Row(f"{tag}:{label}", at, account, debit, credit, entity, act, payment))

    def booked_at(self, act):
        return self.ov.act_booked_at.get(act.key) or self.world.recognized_at(act)

    def act_rate(self, act):
        if act.currency == BASE_CURRENCY:
            return None
        if act.key in self.ov.act_rate:
            return self.ov.act_rate[act.key]
        return self.world.rate(self.ov.act_rate_date.get(act.key, act.service_end), act.currency)

    def payment_rate(self, payment):
        if payment.currency == BASE_CURRENCY:
            return None
        if payment.key in self.ov.payment_rate:
            return self.ov.payment_rate[payment.key]
        day = self.ov.payment_rate_date.get(payment.key, self.world.local_date(payment.instant))
        return self.world.rate(day, payment.currency)

    def line_amounts(self, act):
        rate = self.act_rate(act)
        if rate is None:
            return [(line.line_no, line.net_minor, line.vat_minor) for line in act.lines]
        return [(line.line_no, convert(line.total_minor, act.currency, rate), 0) for line in act.lines]

    def carry_part(self, act_key, part):
        if part == self.outstanding[act_key]:
            return self.carrying[act_key]
        if self.acts[act_key].currency == BASE_CURRENCY:
            return part
        share = Decimal(self.carrying[act_key]) * part / self.outstanding[act_key]
        return int(share.quantize(Decimal(1), ROUND_HALF_UP))

    def settle(self, act_key, part, kzt):
        self.outstanding[act_key] -= part
        self.carrying[act_key] -= kzt

    # ---- posting -------------------------------------------------------
    def post(self):
        world = self.world
        self.opening()
        events = []
        for act in self.acts.values():
            at = self.booked_at(act)
            if world.period_start <= at < world.period_end:
                events.append((at, 0, act.key, "recognize"))
        for payment in world.payments.values():
            at = self.ov.payment_booked_at.get(payment.key, payment.instant)
            if world.period_start <= at < world.period_end:
                events.append((at, 1, payment.key, "receipt"))
                if payment.match_act and world.period_start <= payment.match_at < world.period_end:
                    events.append((payment.match_at, 2, payment.key, "match"))
        events.sort(key=lambda item: (item[0], item[1], item[2]))
        for at, _, key, kind in events:
            if kind == "recognize":
                self.recognize(self.acts[key], at)
            elif kind == "receipt":
                self.receipt(world.payments[key], at)
            else:
                self.match(world.payments[key], at)
        for item in world.fillers:
            self.entry(f"FILL:{item.key}", item.at, item.entity, [
                ("dr", item.debit, item.amount_minor, 0, None, None),
                ("cr", item.credit, 0, item.amount_minor, None, None),
            ])
        assert_balanced(self.rows)
        return self.rows

    def opening(self):
        world = self.world
        at = world.period_start
        self.entry("OPEN:bank", at, None, [
            ("dr", BANK, world.opening_bank_minor, 0, None, None),
            ("cr", OPENING, 0, world.opening_bank_minor, None, None),
        ])
        for key in sorted(self.acts):
            act = self.acts[key]
            if self.booked_at(act) < world.period_start:
                kzt = sum(net + vat for _, net, vat in self.line_amounts(act))
                self.entry(f"OPEN:{key}", at, act.customer, [
                    ("dr", RECEIVABLE, kzt, 0, key, None),
                    ("cr", OPENING, 0, kzt, key, None),
                ])
                self.outstanding[key] = act.total_minor
                self.carrying[key] = kzt
        for customer in sorted(world.opening_advances):
            amount = world.opening_advances[customer]
            self.entry(f"OPENADV:{customer}", at, customer, [
                ("dr", OPENING, amount, 0, None, None),
                ("cr", ADVANCES, 0, amount, None, None),
            ])

    def recognize(self, act, at):
        carrying = 0
        for line_no, net, vat in self.line_amounts(act):
            self.entry(f"REC:{act.key}:{line_no}", at, act.customer, [
                ("dr", RECEIVABLE, net, 0, act.key, None),
                ("cr", REVENUE, 0, net, act.key, None),
            ])
            if vat:
                self.entry(f"REC:{act.key}:{line_no}:vat", at, act.customer, [
                    ("dr", RECEIVABLE, vat, 0, act.key, None),
                    ("cr", VAT, 0, vat, act.key, None),
                ])
            carrying += net + vat
        self.outstanding[act.key] = act.total_minor
        self.carrying[act.key] = carrying
        for payment_key, held in self.prepaid.pop(act.key, []):
            use = min(held, self.carrying[act.key])
            if use:
                self.entry(f"OFF:{act.key}:{payment_key}", at, act.customer, [
                    ("dr", ADVANCES, use, 0, act.key, payment_key),
                    ("cr", RECEIVABLE, 0, use, act.key, payment_key),
                ])
                self.settle(act.key, use, use)
        left = self.opening_left.get(act.customer, 0)
        if left and act.currency == BASE_CURRENCY and self.carrying[act.key]:
            use = min(left, self.carrying[act.key])
            self.entry(f"OFFOPEN:{act.key}", at, act.customer, [
                ("dr", ADVANCES, use, 0, act.key, None),
                ("cr", RECEIVABLE, 0, use, act.key, None),
            ])
            self.settle(act.key, use, use)
            self.opening_left[act.customer] = left - use

    def receipt(self, payment, at):
        key = payment.key
        if payment.amount_minor < 0:  # refund of an advance to the customer
            amount = -payment.amount_minor
            self.entry(f"PAY:{key}", at, payment.customer, [
                ("adv", ADVANCES, amount, 0, None, key),
                ("bank", BANK, 0, amount, None, key),
            ])
            return
        rate = self.payment_rate(payment)

        def kzt(value):
            return convert(value, payment.currency, rate) if rate is not None else value

        bank = kzt(payment.amount_minor)
        legs = [("bank", BANK, bank, 0, None, key)]
        if payment.refs is None:
            legs.append(("adv", ADVANCES, 0, bank, None, key))
            self.unmatched[key] = bank
            self.entry(f"PAY:{key}", at, payment.customer, legs)
            return
        remaining = payment.amount_minor
        to_park = self.ov.parked.get(key, 0)
        for ref in payment.refs:
            if remaining <= 0 or ref not in self.acts:
                continue
            act = self.acts[ref]
            if ref in self.outstanding:
                alloc = min(self.outstanding[ref], remaining)
                if alloc <= 0:
                    continue
                wrong = min(to_park, alloc)
                to_park -= wrong
                right = alloc - wrong
                if right:
                    credit = self.carry_part(ref, right)
                    legs.append((f"ar:{ref}", RECEIVABLE, 0, credit, ref, key))
                    self.settle(ref, right, credit)
                if wrong:
                    legs.append(("park", ADVANCES, 0, kzt(wrong), ref if self.ov.parked_ref.get(key) else None, key))
                remaining -= alloc
            elif act.currency == BASE_CURRENCY and payment.currency == BASE_CURRENCY:
                held = sum(amount for _, amount in self.prepaid[ref])
                use = min(act.total_minor - held, remaining)
                if use > 0:
                    legs.append((f"pre:{ref}", ADVANCES, 0, use, ref, key))
                    self.prepaid[ref].append((key, use))
                    remaining -= use
        if remaining > 0:
            legs.append(("excess", ADVANCES, 0, kzt(remaining), None, key))
        difference = bank - sum(credit for _, _, _, credit, _, _ in legs)
        if difference > 0:
            legs.append(("fx", FX_GAIN, 0, difference, None, key))
        elif difference < 0:
            legs.append(("fx", FX_LOSS, -difference, 0, None, key))
        self.entry(f"PAY:{key}", at, payment.customer, legs)

    def match(self, payment, at):
        act_key, left = payment.match_act, self.unmatched.get(payment.key, 0)
        if act_key in self.outstanding and self.carrying[act_key] > 0 and left > 0:
            use = min(left, self.carrying[act_key])
            self.entry(f"MATCH:{payment.key}", at, payment.customer, [
                ("adv", ADVANCES, use, 0, act_key, payment.key),
                ("ar", RECEIVABLE, 0, use, act_key, payment.key),
            ])
            self.settle(act_key, use, use)
            self.unmatched[payment.key] = left - use


def post(world, overrides=None):
    return Books(world, overrides or Overrides()).post()
