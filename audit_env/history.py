"""Balanced source history; no audit errors are created in this module."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from .domain import BANK, CURRENCY, EXPENSE, PAYABLE


@dataclass
class History:
    ledger: list[dict]
    invoices: list[dict]
    payments: list[dict]
    invoice_rows: dict[str, tuple[dict, dict]]
    payment_rows: dict[str, tuple[dict, dict]]
    eligible_payments: list[dict]


def posting(identifier, doc, timestamp, debit, credit, amount):
    pair = tuple(
        {
            "row_id": identifier("row"),
            "timestamp": timestamp,
            "account": account,
            "debit_minor": amount if is_debit else 0,
            "credit_minor": 0 if is_debit else amount,
            "currency": CURRENCY,
            "entity_id": doc["entity_id"],
            "invoice_id": doc.get("invoice_id"),
            "payment_id": doc.get("payment_id"),
        }
        for account, is_debit in ((debit, True), (credit, False))
    )
    return pair


def assert_balanced(rows):
    totals = defaultdict(int)
    for row in rows:
        totals[row["currency"]] += row["debit_minor"] - row["credit_minor"]
    if any(totals.values()):
        raise ValueError("Unbalanced ledger")


def build_history(rng, identifier, start, invoice_count):
    entities = [identifier("entity") for _ in range(12)]
    ledger, invoices, payments = [], [], []
    invoice_rows, payment_rows, eligible_payments = {}, {}, []
    shared_amount = rng.randrange(50000, 500000, 25)
    for index in range(invoice_count):
        issued = start + timedelta(days=rng.randrange(0, 13), hours=rng.randrange(24))
        # Two real invoices from the same supplier can have the same amount.
        amount = shared_amount if index < 2 else rng.randrange(50000, 900000, 25)
        invoice = {
            "doc_id": identifier("doc"),
            "invoice_id": identifier("invoice"),
            "entity_id": entities[0] if index < 2 else rng.choice(entities),
            "issued_at": issued.isoformat(),
            "due_at": (issued + timedelta(days=14)).isoformat(),
            "currency": CURRENCY,
            "total_minor": amount,
            "line_items": [{"description": "Поставка", "amount_minor": amount}],
        }
        invoices.append(invoice)
        pair = posting(identifier, invoice, invoice["issued_at"], EXPENSE, PAYABLE, amount)
        invoice_rows[invoice["invoice_id"]] = pair
        ledger.extend(pair)
        # A legitimate split settles one invoice in multiple independent payments.
        amounts = (amount // 3, amount - amount // 3) if index < 5 else (amount,)
        for offset, part in enumerate(amounts):
            paid_at = issued + timedelta(days=2 + offset * 3, hours=3)
            payment = {
                "doc_id": identifier("doc"),
                "payment_id": identifier("payment"),
                "timestamp": paid_at.isoformat(),
                "currency": CURRENCY,
                "amount_minor": part,
                "invoice_refs": [invoice["invoice_id"]],
            }
            payments.append(payment)
            linked = payment | {
                "invoice_id": invoice["invoice_id"],
                "entity_id": invoice["entity_id"],
            }
            pair = posting(identifier, linked, payment["timestamp"], PAYABLE, BANK, part)
            payment_rows[payment["payment_id"]] = pair
            ledger.extend(pair)
            if index >= 5:
                eligible_payments.append(payment)

    assert_balanced(ledger)
    return History(ledger, invoices, payments, invoice_rows, payment_rows, eligible_payments)
