"""Independent rule-based auditor over public files only.

It is written from policy.json, not from the generator or scenario code:
- on clean books it must find nothing;
- on books with errors it must find exactly the hidden truth;
- on user-supplied documents (no truth) it is a deterministic auditor.
"""
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal

TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff", "vat_on_top", "uncollected_service")
TOLERANCE = {"duplicate_invoice": 0, "payment_allocation": 0, "fx_rate": 2, "period_cutoff": 0,
             "vat_on_top": 0, "uncollected_service": 0}
BANK, RECEIVABLE, ADVANCES, REVENUE = "1030", "1210", "3510", "6010"


def kzt(amount, currency, rate, units, base="KZT"):
    if currency == base:
        return amount
    value = Decimal(amount) * Decimal(rate) * (Decimal(10) ** (units[base] - units[currency]))
    return int(value.quantize(Decimal(1), ROUND_HALF_UP))


def finding(kind, affected, amount, currency, evidence, explanation):
    return {"type": kind, "affected_ids": sorted(affected), "claimed_amount_minor": amount,
            "currency": currency, "evidence_ids": list(dict.fromkeys(evidence))[:10],
            "explanation": explanation[:1000]}


class Auditor:
    def __init__(self, manifest, policy, tables):
        self.base = manifest["base_currency"]
        self.units = manifest["minor_units"]
        self.start = datetime.fromisoformat(policy["period"]["start"])
        self.end = datetime.fromisoformat(policy["period"]["end"])
        self.zone = self.start.tzinfo
        self.rates = {(date.fromisoformat(row["date"]), row["source_currency"]): row
                      for row in tables["fx"]}
        self.invoices = {row["invoice_id"]: row for row in tables["invoices"]}
        self.payments = {row["payment_id"]: row for row in tables["payments"]}
        self.ledger = tables["ledger"]
        self.by_invoice, self.by_payment = defaultdict(list), defaultdict(list)
        for row in self.ledger:
            if row["invoice_id"] is not None:
                self.by_invoice[row["invoice_id"]].append(row)
            if row["payment_id"] is not None:
                self.by_payment[row["payment_id"]].append(row)

    # ---- helpers -------------------------------------------------------
    def service_end(self, invoice):
        return max(date.fromisoformat(line["service_period_end"]) for line in invoice["line_items"])

    def recognized_at(self, invoice):
        return datetime.combine(self.service_end(invoice), time(0), self.zone)

    def in_period(self, moment):
        return self.start <= moment < self.end

    def instant(self, payment):
        return datetime.fromisoformat(payment["timestamp"])

    def local_date(self, moment):
        return moment.astimezone(self.zone).date()

    def rate(self, day, currency):
        return self.rates[(day, currency)]

    def required_rates(self):
        """(date, currency) -> doc_id of the first document whose check looks that rate up."""
        needed = {}
        for invoice_id in sorted(self.invoices):
            invoice = self.invoices[invoice_id]
            if (invoice["currency"] != self.base and self.revenue_rows(invoice_id)
                    and self.in_period(self.recognized_at(invoice))):
                needed.setdefault((self.service_end(invoice), invoice["currency"]), invoice["doc_id"])
        for payment_id in sorted(self.payments):
            payment = self.payments[payment_id]
            if payment["currency"] == self.base or payment["amount_minor"] <= 0:
                continue
            moment = self.instant(payment)
            if self.bank_rows(payment_id) or self.in_period(moment):
                needed.setdefault((self.local_date(moment), payment["currency"]), payment["doc_id"])
        return needed

    def revenue_rows(self, invoice_id):
        return [row for row in self.by_invoice[invoice_id] if row["account"] == REVENUE and row["credit_minor"]]

    def bank_rows(self, payment_id):
        return [row for row in self.by_payment[payment_id] if row["account"] == BANK]

    # ---- checks --------------------------------------------------------
    def audit(self):
        found = []
        found += self.invoice_checks()
        found += self.duplicate_checks()
        found += self.payment_checks()
        found += self.allocation_checks()
        found += self.vat_checks()
        found += self.uncollected_checks()
        return found

    def vat_checks(self):
        """Prices include VAT: a line total above quantity × unit price means VAT was added on top."""
        found = []
        for invoice_id in sorted(self.invoices):
            invoice = self.invoices[invoice_id]
            extra = sum(line["total_minor"] - line["quantity"] * line["unit_price_minor"]
                        for line in invoice["line_items"])
            if extra <= 0:
                continue
            vat_rows = [row["row_id"] for row in self.by_invoice[invoice_id] if row["account"] == "3130" and row["credit_minor"]]
            found.append(finding("vat_on_top", [invoice_id], extra, invoice["currency"],
                                 [invoice["doc_id"], *vat_rows],
                                 "НДС начислен сверху цены, а по условиям цена включает НДС."))
        return found

    def uncollected_checks(self):
        """An overdue invoice with no payment at all, and the same service renewed for the customer after the due date."""
        found = []
        for invoice_id in sorted(self.invoices):
            invoice = self.invoices[invoice_id]
            rows = self.by_invoice[invoice_id]
            debit = sum(row["debit_minor"] for row in rows if row["account"] == RECEIVABLE)
            credit = sum(row["credit_minor"] for row in rows if row["account"] == RECEIVABLE)
            due = datetime.fromisoformat(invoice["due_at"])
            if invoice["currency"] != self.base or debit != invoice["total_minor"] or credit or debit == 0 or due >= self.end:
                continue
            follows = {(line["service_code"], (date.fromisoformat(line["service_period_end"]) + timedelta(days=1)).isoformat())
                       for line in invoice["line_items"]}
            renewals = sorted((other for other in self.invoices.values()
                               if other["invoice_id"] != invoice_id and other["entity_id"] == invoice["entity_id"]
                               and datetime.fromisoformat(other["issued_at"]) > due
                               and follows & {(line["service_code"], line["service_period_start"])
                                              for line in other["line_items"]}),
                              key=lambda item: (item["issued_at"], item["invoice_id"]))
            if renewals:
                evidence = [invoice["doc_id"], renewals[0]["doc_id"],
                            *(row["row_id"] for row in rows if row["account"] == RECEIVABLE and row["debit_minor"])]
                found.append(finding("uncollected_service", [invoice_id], invoice["total_minor"], invoice["currency"],
                                     evidence, "Счёт просрочен и не оплачен, а услугу клиенту уже продлили новым счётом."))
        return found

    def invoice_checks(self):
        found = []
        for invoice_id in sorted(self.invoices):
            invoice = self.invoices[invoice_id]
            revenue = self.revenue_rows(invoice_id)
            if not revenue:
                continue
            posted = sum(row["credit_minor"] for row in revenue)
            evidence = [invoice["doc_id"], *(row["row_id"] for row in revenue)]
            if not self.in_period(self.recognized_at(invoice)):
                found.append(finding("period_cutoff", [invoice_id], posted, self.base, evidence,
                                     "Выручка признана в периоде, хотя услуга окончена вне периода."))
            elif invoice["currency"] != self.base:
                day = self.service_end(invoice)
                fx = self.rate(day, invoice["currency"])
                expected = kzt(invoice["total_minor"], invoice["currency"], fx["rate"], self.units)
                if abs(posted - expected) > TOLERANCE["fx_rate"]:
                    found.append(finding("fx_rate", [invoice_id], abs(posted - expected), self.base,
                                         [invoice["doc_id"], fx["fx_id"], *(row["row_id"] for row in revenue)],
                                         "Счёт пересчитан не по курсу даты окончания услуги."))
        return found

    def duplicate_checks(self):
        groups = defaultdict(list)
        for invoice in self.invoices.values():
            if not self.revenue_rows(invoice["invoice_id"]) or not self.in_period(self.recognized_at(invoice)):
                continue
            signature = (invoice["entity_id"], invoice["currency"], tuple(sorted(
                (line["service_code"], line["service_period_start"], line["service_period_end"], line["total_minor"])
                for line in invoice["line_items"])))
            groups[signature].append(invoice)
        found = []
        for signature in sorted(groups, key=repr):
            members = sorted(groups[signature], key=lambda item: (item["issued_at"], item["invoice_id"]))
            for first, second in zip(members[::2], members[1::2]):
                evidence = [first["doc_id"], second["doc_id"],
                            *(row["row_id"] for item in (first, second) for row in self.revenue_rows(item["invoice_id"]))]
                found.append(finding("duplicate_invoice", [first["invoice_id"], second["invoice_id"]],
                                     second["total_minor"], second["currency"], evidence,
                                     "Два счёта одному покупателю на одну и ту же услугу и период проведены в выручку."))
        return found

    def payment_checks(self):
        found = []
        for payment_id in sorted(self.payments):
            payment = self.payments[payment_id]
            bank = self.bank_rows(payment_id)
            if not bank:
                continue
            moment = self.instant(payment)
            debit = sum(row["debit_minor"] for row in bank)
            if not self.in_period(moment):
                found.append(finding("period_cutoff", [payment_id], debit, self.base,
                                     [payment["doc_id"], *(row["row_id"] for row in bank)],
                                     "Поступление вне периода (по местному времени) проведено в периоде."))
            if payment["currency"] != self.base and payment["amount_minor"] > 0:
                fx = self.rate(self.local_date(moment), payment["currency"])
                expected = kzt(payment["amount_minor"], payment["currency"], fx["rate"], self.units)
                if abs(debit - expected) > TOLERANCE["fx_rate"]:
                    found.append(finding("fx_rate", [payment_id], abs(debit - expected), self.base,
                                         [payment["doc_id"], fx["fx_id"], *(row["row_id"] for row in bank)],
                                         "Поступление пересчитано не по курсу местной даты операции."))
        return found

    def allocation_checks(self):
        """Replay the published allocation rule and compare expected advances with booked 3510 credits."""
        outstanding, held, opening_left = {}, defaultdict(list), defaultdict(int)
        for row in self.ledger:
            if row["account"] == ADVANCES and row["credit_minor"] and row["payment_id"] is None \
                    and datetime.fromisoformat(row["timestamp"]) == self.start and row["entity_id"]:
                opening_left[row["entity_id"]] += row["credit_minor"]
        for invoice_id, invoice in self.invoices.items():
            if self.recognized_at(invoice) < self.start:
                outstanding[invoice_id] = invoice["total_minor"]
        events = []
        for invoice_id, invoice in self.invoices.items():
            moment = self.recognized_at(invoice)
            if self.in_period(moment):
                events.append((moment, 0, invoice_id, "recognize"))
        for payment_id, payment in self.payments.items():
            moment = self.instant(payment)
            if self.in_period(moment) and payment["amount_minor"] > 0:
                events.append((moment, 1, payment_id, "receipt"))
        for row in self.ledger:  # bookkeeper matches of unreferenced receipts are decisions, not derivable
            payment = self.payments.get(row["payment_id"])
            if (payment is not None and payment["invoice_refs"] is None and row["account"] == ADVANCES
                    and row["debit_minor"] and row["invoice_id"] is not None):
                events.append((datetime.fromisoformat(row["timestamp"]), 2, row["row_id"], "match"))
        events.sort(key=lambda item: (item[0], item[1], item[2]))
        rows = {row["row_id"]: row for row in self.ledger}
        found = []
        for _, _, key, kind in events:
            if kind == "recognize":
                invoice = self.invoices[key]
                outstanding[key] = invoice["total_minor"]
                for amount in held.pop(key, []):
                    outstanding[key] -= min(amount, outstanding[key])
                entity = invoice["entity_id"]
                if opening_left[entity] and invoice["currency"] == self.base and outstanding[key]:
                    use = min(opening_left[entity], outstanding[key])
                    outstanding[key] -= use
                    opening_left[entity] -= use
            elif kind == "match":
                row = rows[key]
                if row["invoice_id"] in outstanding:
                    outstanding[row["invoice_id"]] -= min(row["debit_minor"], outstanding[row["invoice_id"]])
            else:
                result = self.receipt(self.payments[key], outstanding, held)
                if result:
                    found.append(result)
        return found

    def receipt(self, payment, outstanding, held):
        payment_id = payment["payment_id"]
        currency = payment["currency"]
        rate = None
        if currency != self.base:
            rate = self.rate(self.local_date(self.instant(payment)), currency)["rate"]
        expected = 0
        remaining = payment["amount_minor"]
        if payment["invoice_refs"] is None:
            expected = kzt(remaining, currency, rate, self.units) if rate is not None else remaining
            remaining = 0
        else:
            for ref in payment["invoice_refs"]:
                invoice = self.invoices.get(ref)
                if remaining <= 0 or invoice is None:
                    continue
                if ref in outstanding:
                    take = min(outstanding[ref], remaining)
                    outstanding[ref] -= take
                    remaining -= take
                elif invoice["currency"] == self.base and currency == self.base:
                    use = min(invoice["total_minor"] - sum(held[ref]), remaining)
                    if use > 0:
                        held[ref].append(use)
                        expected += use
                        remaining -= use
        if remaining > 0:
            expected += kzt(remaining, currency, rate, self.units) if rate is not None else remaining
        if not self.bank_rows(payment_id):
            return None
        credits = [row for row in self.by_payment[payment_id] if row["account"] == ADVANCES and row["credit_minor"]]
        actual = sum(row["credit_minor"] for row in credits)
        if actual - expected <= TOLERANCE["payment_allocation"]:
            return None
        difference = actual - expected
        exact = [row for row in credits if row["credit_minor"] == difference]
        cited = [ref for ref in payment["invoice_refs"] or () if ref in self.invoices]
        evidence = [payment["doc_id"], *(self.invoices[ref]["doc_id"] for ref in cited[:1]),
                    *(row["row_id"] for row in self.bank_rows(payment_id)),
                    *(row["row_id"] for row in (exact[:1] or credits))]
        return finding("payment_allocation", [payment_id], difference, self.base, evidence,
                       "Поступление по открытому счёту отнесено в авансы вместо погашения дебиторки.")


def audit(manifest, policy, tables):
    return Auditor(manifest, policy, tables).audit()


def audit_episode(episode):
    return audit(episode.manifest, episode.policy, episode.tables)
