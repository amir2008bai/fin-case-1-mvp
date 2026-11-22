"""A small learned auditor: broad candidates -> public features -> per-type logistic scores.

The agent scans the tables through the public tools, proposes many candidate findings
(high recall, low precision) for all four types, scores each with a per-type logistic
model and submits those above the threshold. Amount and evidence come from the same
public records. Weights are fit on train episodes (labels from open-mode truth); the
threshold is then tuned on the environment's terminal reward. No hidden data is used
at inference time.
"""
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from .agents import ScanPolicy, report
from .checks import Auditor, kzt

MODEL_VERSION = 4
FX_TOLERANCE = 2  # the published matcher tolerance for fx_rate, in minor units
TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff", "vat_on_top", "uncollected_service")
FEATURES = {
    "vat_on_top": ("bias", "extra_share", "has_extra", "vat_line_share", "paid_base_share"),
    "uncollected_service": ("bias", "overdue", "renewed_after_due", "same_code_later", "overdue_months"),
    "duplicate_invoice": ("bias", "identical_supply", "issue_gap_days", "both_recognized", "line_count"),
    "payment_allocation": ("bias", "advance_share", "recognized_before", "expected_excess_share", "refs_count",
                           "foreign", "unexplained_share", "fully_explained"),
    "fx_rate": ("bias", "log_gap_correct", "log_gap_issue_or_wall", "is_payment", "booked_outside", "over_tolerance"),
    "period_cutoff": ("bias", "service_in_period", "issue_in_period", "instant_in_period", "wall_in_period",
                      "is_payment"),
}


@dataclass(frozen=True)
class Candidate:
    type: str
    affected: tuple
    amount: int
    currency: str
    evidence: tuple
    features: tuple

    def finding(self):
        return report(self.type, list(self.affected), self.amount, self.currency, list(self.evidence),
                      "Найдено обученной моделью по публичным записям.")


def log_gap(value):
    return math.log1p(abs(value)) / 10


def candidates(tables, manifest, policy):
    view = Auditor(manifest, policy, tables)
    base = view.base
    found = []
    invoices = list(view.invoices.values())
    # duplicate pairs: same customer, currency and total (a deliberately broad superset)
    groups = defaultdict(list)
    for invoice in invoices:
        groups[(invoice["entity_id"], invoice["currency"], invoice["total_minor"])].append(invoice)
    for members in groups.values():
        members = sorted(members, key=lambda item: (item["issued_at"], item["invoice_id"]))
        for index, first in enumerate(members):
            for second in members[index + 1:]:
                lines = lambda item: sorted((l["service_code"], l["total_minor"]) for l in item["line_items"])  # noqa: E731
                periods = lambda item: sorted((l["service_period_start"], l["service_period_end"])  # noqa: E731
                                              for l in item["line_items"])
                gap = abs((datetime.fromisoformat(second["issued_at"]) - datetime.fromisoformat(first["issued_at"])).days)
                recognized = all(view.revenue_rows(item["invoice_id"]) for item in (first, second))
                evidence = (first["doc_id"], second["doc_id"],
                            *(row["row_id"] for item in (first, second) for row in view.revenue_rows(item["invoice_id"])))
                same_lines = lines(first) == lines(second)
                same_period = periods(first) == periods(second)
                found.append(Candidate("duplicate_invoice", (first["invoice_id"], second["invoice_id"]),
                                       second["total_minor"], second["currency"], evidence[:10],
                                       (1.0, float(same_lines and same_period), min(gap, 60) / 30,
                                        float(recognized), len(first["line_items"]) / 2)))
    for invoice in invoices:
        revenue = view.revenue_rows(invoice["invoice_id"])
        if not revenue or not invoice.get("line_items"):
            continue
        posted = sum(row["credit_minor"] for row in revenue)
        service_end = view.service_end(invoice)
        issued = datetime.fromisoformat(invoice["issued_at"])
        service_in = view.in_period(view.recognized_at(invoice))
        issue_in = view.in_period(issued)
        evidence = (invoice["doc_id"], *(row["row_id"] for row in revenue))
        found.append(Candidate("period_cutoff", (invoice["invoice_id"],), posted, base, evidence[:10],
                               (1.0, float(service_in), float(issue_in), float(service_in), float(issue_in), 0.0)))
        if invoice["currency"] != base and invoice["currency"] in view.units and (service_end, invoice["currency"]) in view.rates:
            fx = view.rate(service_end, invoice["currency"])
            correct = kzt(invoice["total_minor"], invoice["currency"], fx["rate"], view.units)
            issue_day = view.local_date(issued)
            other = view.rates.get((issue_day, invoice["currency"]), fx)
            at_issue = kzt(invoice["total_minor"], invoice["currency"], other["rate"], view.units)
            found.append(Candidate("fx_rate", (invoice["invoice_id"],), abs(posted - correct), base,
                                   (invoice["doc_id"], fx["fx_id"], *(row["row_id"] for row in revenue))[:10],
                                   (1.0, log_gap(posted - correct), log_gap(posted - at_issue), 0.0, float(not service_in),
                                    float(abs(posted - correct) > FX_TOLERANCE))))
    received = defaultdict(int)
    for row in view.ledger:
        if row["account"] == "1210" and row["credit_minor"] and row["invoice_id"]:
            received[row["invoice_id"]] += row["credit_minor"]
    for invoice in invoices:
        lines = invoice["line_items"]
        taxed = [line for line in lines if line["vat_rate"]]
        if not taxed or invoice["currency"] != base:
            continue
        extra = sum(line["total_minor"] - line["quantity"] * line["unit_price_minor"] for line in lines)
        vat_rows = [row["row_id"] for row in view.by_invoice[invoice["invoice_id"]]
                    if row["account"] == "3130" and row["credit_minor"]]
        total = abs(invoice["total_minor"]) or 1
        base_amount = abs(sum(line["quantity"] * line["unit_price_minor"] for line in lines)) or 1
        found.append(Candidate("vat_on_top", (invoice["invoice_id"],), max(extra, 0), invoice["currency"],
                               (invoice["doc_id"], *vat_rows)[:10],
                               (1.0, max(extra, 0) / total, float(extra > 0), len(taxed) / len(lines),
                                min(received[invoice["invoice_id"]] / base_amount, 2.0))))
    for invoice in invoices:
        rows = view.by_invoice[invoice["invoice_id"]]
        debits = [row for row in rows if row["account"] == "1210" and row["debit_minor"]]
        if invoice["currency"] != base or not debits or received[invoice["invoice_id"]]:
            continue
        due = datetime.fromisoformat(invoice["due_at"])
        codes = {line["service_code"] for line in invoice["line_items"]}
        follows = {(line["service_code"], (date.fromisoformat(line["service_period_end"]) + timedelta(days=1)).isoformat())
                   for line in invoice["line_items"]}
        later = [other for other in invoices if other["invoice_id"] != invoice["invoice_id"]
                 and other["entity_id"] == invoice["entity_id"]
                 and codes & {line["service_code"] for line in other["line_items"]}
                 and other["issued_at"] > invoice["issued_at"]]
        renewals = [other for other in later if datetime.fromisoformat(other["issued_at"]) > due
                    and follows & {(line["service_code"], line["service_period_start"]) for line in other["line_items"]}]
        overdue_days = max(0, (view.end - due).days)
        evidence = (invoice["doc_id"], *(item["doc_id"] for item in renewals[:1]), *(row["row_id"] for row in debits))
        found.append(Candidate("uncollected_service", (invoice["invoice_id"],), invoice["total_minor"], base,
                               evidence[:10],
                               (1.0, float(due < view.end), float(bool(renewals)), float(bool(later)),
                                min(overdue_days, 90) / 30)))
    for payment in view.payments.values():
        bank = view.bank_rows(payment["payment_id"])
        if not bank or payment["amount_minor"] <= 0:
            continue
        moment = view.instant(payment)
        debit = sum(row["debit_minor"] for row in bank)
        wall = moment.replace(tzinfo=view.zone)
        instant_in = view.in_period(moment)
        evidence = (payment["doc_id"], *(row["row_id"] for row in bank))
        found.append(Candidate("period_cutoff", (payment["payment_id"],), debit, base, evidence[:10],
                               (1.0, float(instant_in), float(view.in_period(wall)), float(instant_in),
                                float(view.in_period(wall)), 1.0)))
        if payment["currency"] != base:
            day = view.local_date(moment)
            if payment["currency"] not in view.units or (day, payment["currency"]) not in view.rates:
                continue
            fx = view.rate(day, payment["currency"])
            correct = kzt(payment["amount_minor"], payment["currency"], fx["rate"], view.units)
            wall_fx = view.rates.get((wall.date(), payment["currency"]), fx)
            at_wall = kzt(payment["amount_minor"], payment["currency"], wall_fx["rate"], view.units)
            found.append(Candidate("fx_rate", (payment["payment_id"],), abs(debit - correct), base,
                                   (payment["doc_id"], fx["fx_id"], *(row["row_id"] for row in bank))[:10],
                                   (1.0, log_gap(debit - correct), log_gap(debit - at_wall), 1.0, float(not instant_in),
                                    float(abs(debit - correct) > FX_TOLERANCE))))
        advances = [row for row in view.by_payment[payment["payment_id"]]
                    if row["account"] == "3510" and row["credit_minor"]]
        refs = [ref for ref in payment["invoice_refs"] or () if ref in view.invoices]
        if refs and advances and instant_in:
            first = view.invoices[refs[0]]
            parked = sum(row["credit_minor"] for row in advances)
            amount_kzt = debit or 1
            recognized = view.recognized_at(first) <= moment
            others = sum(row["credit_minor"] for row in view.by_invoice[first["invoice_id"]]
                         if row["account"] == "1210" and row["credit_minor"] and row["payment_id"] != payment["payment_id"])
            open_part = max(0, sum(view.invoices[ref]["total_minor"] for ref in refs) - others)
            excess = max(0, payment["amount_minor"] - open_part) if payment["currency"] == base else 0
            chosen = min(advances, key=lambda row: abs(row["credit_minor"] - (parked - excess)))
            unexplained = max(0, parked - excess)
            found.append(Candidate("payment_allocation", (payment["payment_id"],), parked - excess if excess < parked else parked,
                                   base, (payment["doc_id"], first["doc_id"], *(row["row_id"] for row in bank),
                                          chosen["row_id"])[:10],
                                   (1.0, parked / amount_kzt, float(recognized), excess / max(1, payment["amount_minor"]),
                                    len(refs) / 2, float(payment["currency"] != base), unexplained / amount_kzt,
                                    float(unexplained == 0))))
    return found


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


class MicroModel:
    def __init__(self, weights, threshold=0.5, meta=None):
        self.weights = weights
        self.threshold = threshold
        self.meta = meta or {}

    def probability(self, candidate):
        weights = self.weights.get(candidate.type)
        return 0.0 if weights is None else sigmoid(sum(w * x for w, x in zip(weights, candidate.features)))

    def findings(self, tables, manifest, policy):
        scored = sorted(((self.probability(c), c) for c in candidates(tables, manifest, policy)),
                        key=lambda item: -item[0])
        chosen, used = [], set()
        for probability, candidate in scored:
            key = (candidate.type, candidate.affected)
            if probability >= self.threshold and key not in used:
                used.add(key)
                chosen.append(candidate.finding())
        return chosen[:30]

    def policy(self):
        return ScanPolicy(self.findings)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"version": MODEL_VERSION, "features": FEATURES, "weights": self.weights,
                                          "threshold": self.threshold, "meta": self.meta},
                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != MODEL_VERSION or data.get("features") != {k: list(v) for k, v in FEATURES.items()}:
            raise ValueError("model file does not match this code version")
        return cls(data["weights"], data["threshold"], data.get("meta"))


def fit_logistic(rows, size, epochs=300, rate=0.5, l2=1e-3):
    """Class-balanced batch gradient descent; deterministic, pure Python."""
    if not rows:
        return [0.0] * size
    weights = [0.0] * size
    positives = sum(label for _, label in rows) or 1
    negatives = (len(rows) - positives) or 1
    scale = {1: len(rows) / (2 * positives), 0: len(rows) / (2 * negatives)}
    for _ in range(epochs):
        gradient = [0.0] * size
        for features, label in rows:
            error = (sigmoid(sum(w * x for w, x in zip(weights, features))) - label) * scale[label]
            for index, value in enumerate(features):
                gradient[index] += error * value
        weights = [w - rate * (g / len(rows) + l2 * w) for w, g in zip(weights, gradient)]
    return [round(w, 6) for w in weights]


def labelled(episode):
    truth = {(issue.type, tuple(sorted(issue.affected_ids))) for issue in episode.issues}
    return [(candidate, int((candidate.type, tuple(sorted(candidate.affected))) in truth))
            for candidate in candidates(episode.tables, episode.manifest, episode.policy)]

