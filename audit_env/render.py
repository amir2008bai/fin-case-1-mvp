"""Presentation: hidden events and rows -> public tables, documents, truth, provenance.

IDs come from an HMAC of the evaluator key, so they reveal neither the seed nor
which records were changed. Rows are ordered by their opaque IDs, document
numbers are chronological, and all of it happens after errors are injected.
"""
import hashlib
import json

from . import __version__, keys
from .contracts import SCHEMA_VERSION, TABLE_FIELDS, Episode, Issue
from .domain import BASE_CURRENCY, MINOR_UNITS
from .policy import policy_document

DIRECTIONS = {True: "incoming", False: "outgoing"}


def iso(moment, zone):
    return moment.astimezone(zone).isoformat(timespec="seconds")


class Presenter:
    def __init__(self, world, overrides, *, key, seed, config):
        self.world = world
        self.acts = {**world.acts, **overrides.extra_acts}
        self.salt = (seed, config.semantic_json(), config.presentation_seed)
        self.key = key
        self.rng = keys.stream(key, "presentation", *self.salt)
        self.episode_id = keys.opaque(key, "episode", *self.salt)
        currencies = {BASE_CURRENCY} | {customer.currency for customer in world.customers.values()}
        self.currencies = sorted(currencies)
        self.numbers = {}
        base = self.rng.randrange(100, 5000)
        for index, act in enumerate(sorted(self.acts.values(), key=lambda item: (item.issued_at, item.key))):
            self.numbers[act.key] = f"АВР-{base + index:05d}"
        base = self.rng.randrange(100, 5000)
        for index, payment in enumerate(sorted(world.payments.values(), key=lambda item: (item.instant, item.key))):
            self.numbers[payment.key] = f"ПП-{base + index:05d}"

    def oid(self, kind, internal):
        return keys.opaque(self.key, kind, *self.salt, internal)

    def doc(self, key):
        return self.oid("doc", key)

    def invoice(self, key):
        return None if key is None else self.oid("invoice", key)

    def payment(self, key):
        return None if key is None else self.oid("payment", key)

    def entity(self, key):
        return None if key is None else self.oid("entity", key)

    def fx(self, day, currency):
        return self.oid("fx", f"{day.isoformat()}:{currency}")

    def row_id(self, key):
        return self.oid("row", key)

    def public(self, ref):
        kind, value = ref
        if kind == "doc":
            return self.doc(value)
        if kind == "row":
            return self.row_id(value)
        if kind == "fx":
            return self.fx(*value)
        if kind == "invoice":
            return self.invoice(value)
        if kind == "payment":
            return self.payment(value)
        raise ValueError(kind)

    # ---- records -------------------------------------------------------
    def ledger_row(self, row):
        return {
            "row_id": self.row_id(row.key),
            "timestamp": iso(row.at, self.world.tz),
            "account": row.account,
            "debit_minor": row.debit,
            "credit_minor": row.credit,
            "currency": BASE_CURRENCY,
            "entity_id": self.entity(row.entity),
            "invoice_id": self.invoice(row.act),
            "payment_id": self.payment(row.payment),
        }

    def line_item(self, line):
        return {
            "line_no": line.line_no,
            "description": line.description,
            "service_code": line.service_code,
            "quantity": line.quantity,
            "unit_price_minor": line.unit_price_minor,
            "net_minor": line.net_minor,
            "vat_rate": line.vat_rate,
            "vat_minor": line.vat_minor,
            "total_minor": line.total_minor,
            "service_period_start": line.period_start.isoformat(),
            "service_period_end": line.period_end.isoformat(),
        }

    def invoice_row(self, act):
        zone = self.world.tz
        return {
            "doc_id": self.doc(act.key),
            "invoice_id": self.invoice(act.key),
            "entity_id": self.entity(act.customer),
            "issued_at": iso(act.issued_at, zone),
            "due_at": iso(act.due_at, zone),
            "currency": act.currency,
            "total_minor": act.total_minor,
            "line_items": [self.line_item(line) for line in act.lines],
        }

    def payment_row(self, payment):
        from .world import parse_offset
        return {
            "doc_id": self.doc(payment.key),
            "payment_id": self.payment(payment.key),
            "timestamp": iso(payment.instant, parse_offset(payment.shown_offset)),
            "currency": payment.currency,
            "amount_minor": payment.amount_minor,
            "invoice_refs": None if payment.refs is None else [self.invoice(ref) for ref in payment.refs],
        }

    def counterparty(self, key):
        customer = self.world.customers[key]
        return {"entity_id": self.entity(key), "name": customer.name, "country": customer.country}

    def purpose(self, payment):
        acts = [self.acts[ref] for ref in payment.refs or ()]
        if payment.amount_minor < 0:
            return "Возврат излишне перечисленных средств по заявлению покупателя"
        if not acts:
            return self.rng.choice(("Оплата за услуги", "Оплата по договору за сопровождение ПО",
                                    "За программное обеспечение, без НДС", "Оплата за услуги согласно договору"))
        cited = ", ".join(f"{self.numbers[act.key]} от {act.issued_at.astimezone(self.world.tz).date().isoformat()}"
                          for act in acts)
        if len(acts) > 1:
            return f"Оплата по счетам {cited}"
        description = acts[0].lines[0].description.lower()
        if payment.kind in ("prepayment", "prepayment_next"):
            return f"Предоплата по счёту {cited} за {description}"
        if payment.kind in ("split", "partial"):
            return f"Частичная оплата по счёту {cited}"
        return self.rng.choice((f"Оплата по счёту {cited}", f"Оплата за {description} по счёту {cited}",
                                f"Согласно счёту на оплату {cited}. {acts[0].lines[0].description}"))

    def invoice_document(self, act):
        return self.invoice_row(act) | {
            "doc_type": "invoice",
            "number": self.numbers[act.key],
            "customer": self.counterparty(act.customer),
            "contract": act.contract,
            "manager": act.manager,
            "vat_minor": sum(line.vat_minor for line in act.lines),
            "status": "confirmed",
        }

    def payment_document(self, payment):
        row = self.payment_row(payment)
        return row | {
            "doc_type": "payment",
            "number": self.numbers[payment.key],
            "direction": DIRECTIONS[payment.amount_minor > 0],
            "counterparty": self.counterparty(payment.customer),
            "operation_local_date": self.world.local_date(payment.instant).isoformat(),
            "purpose": self.purpose(payment),
        }


def canonical_hash(rows):
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in sorted(rows, key=lambda r: r["row_id"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def present(world, clean_rows, rows, overrides, specs, *, key, seed, config):
    view = Presenter(world, overrides, key=key, seed=seed, config=config)
    acts = view.acts
    ledger = sorted((view.ledger_row(row) for row in rows), key=lambda item: item["row_id"])
    invoices = sorted((view.invoice_row(act) for act in acts.values()), key=lambda item: item["invoice_id"])
    payments = sorted((view.payment_row(payment) for payment in world.payments.values()),
                      key=lambda item: item["payment_id"])
    used = [code for code in view.currencies if code != BASE_CURRENCY]
    fx = sorted(({"fx_id": view.fx(day, code), "date": day.isoformat(), "source_currency": code,
                  "target_currency": BASE_CURRENCY, "rate": f"{rate:.2f}"}
                 for (day, code), rate in world.rates.items() if code in used), key=lambda item: item["fx_id"])
    tables = {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx}
    for name, items in tables.items():
        for item in items:
            if tuple(item) != TABLE_FIELDS[name]:
                raise AssertionError(f"{name} fields drifted")
    documents = {view.doc(act.key): view.invoice_document(act) for act in acts.values()}
    documents |= {view.doc(payment.key): view.payment_document(payment) for payment in world.payments.values()}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": view.episode_id,
        "company": {"name": world.company, "country": "KZ"},
        "base_currency": BASE_CURRENCY,
        "period_start": world.period_start.isoformat(),
        "period_end": world.period_end.isoformat(),
        "timezone": world.tz_text,
        "counts": {name: len(items) for name, items in tables.items()},
        "document_count": len(invoices) + len(payments),
        "minor_units": {code: MINOR_UNITS[code] for code in view.currencies},
        "rounding": {"mode": "ROUND_HALF_UP", "fx": "per converted amount", "vat": "per line, included in price"},
        "id_fields": {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id",
                      "documents": "doc_id"},
    }
    policy = policy_document(world, view.currencies)

    issues = []
    for index, spec in enumerate(specs):
        issues.append(Issue(
            issue_id=view.oid("issue", str(index)),
            type=spec.type,
            affected_ids=tuple(sorted(view.public(ref) for ref in spec.affected)),
            expected_amount_minor=spec.amount,
            currency=spec.currency,
            evidence_sets=tuple(tuple(view.public(ref) for ref in proof) for proof in spec.evidence_sets),
            tolerance_minor=spec.tolerance,
            relevant_ids=tuple(sorted({view.public(ref) for ref in spec.relevant}
                                      | {item for proof in spec.evidence_sets for item in map(view.public, proof)})),
        ))
    provenance = build_provenance(view, world, clean_rows, rows, overrides, specs, issues, seed, config, key)
    return Episode(manifest, policy, tables, tuple(issues), provenance, documents)


def build_provenance(view, world, clean_rows, rows, overrides, specs, issues, seed, config, key):
    clean = {row.key: row for row in clean_rows}
    dirty = {row.key: row for row in rows}
    changed = sorted(row_key for row_key in clean.keys() | dirty.keys() if clean.get(row_key) != dirty.get(row_key))

    def links(row):
        found = {row.act, row.payment}
        if row.payment in world.payments and world.payments[row.payment].refs:
            found |= set(world.payments[row.payment].refs)
        return found - {None}

    groups = {}
    for spec, issue in zip(specs, issues):
        group = groups.setdefault(spec.group, {"issue_ids": [], "scenario": spec.scenario, "causes": [],
                                               "targets": set(), "before": [], "after": [],
                                               "documents_added": []})
        group["issue_ids"].append(issue.issue_id)
        group["causes"].append(spec.cause)
        group["targets"] |= set(spec.acts) | set(spec.payments)
    for row_key in changed:
        row = dirty.get(row_key) or clean.get(row_key)
        owners = [number for number, group in groups.items() if links(row) & group["targets"]]
        if len(owners) != 1:
            raise AssertionError(f"change {row_key} is not attributable to exactly one scenario")
        group = groups[owners[0]]
        if row_key in clean:
            group["before"].append(view.ledger_row(clean[row_key]))
        if row_key in dirty:
            group["after"].append(view.ledger_row(dirty[row_key]))
    for act in overrides.extra_acts.values():
        owners = [group for group in groups.values() if act.key in group["targets"]]
        owners[0]["documents_added"].append({"doc_id": view.doc(act.key), "invoice_id": view.invoice(act.key)})
    changes = []
    for number in sorted(groups):
        group = groups[number]
        changes.append({
            "issue_ids": group["issue_ids"],
            "scenario": group["scenario"],
            "causes": group["causes"],
            "documents_added": group["documents_added"],
            "before": sorted(group["before"], key=lambda item: item["row_id"]),
            "after": sorted(group["after"], key=lambda item: item["row_id"]),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": __version__,
        "episode_id": view.episode_id,
        "seed": seed,
        "config": config.as_dict(),
        "keyspace": config.keyspace,
        "key_fingerprint": keys.fingerprint(key),
        "world": {"summary": dict(sorted(world.summary.items())), "customers": len(world.customers),
                  "invoices": len(view.acts), "payments": len(world.payments)},
        "clean_ledger_sha256": canonical_hash([view.ledger_row(row) for row in clean_rows]),
        "changes": changes,
    }
