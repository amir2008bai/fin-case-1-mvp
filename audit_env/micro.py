"""Feature extraction, a tiny logistic model, and grounded findings."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .contracts import ALLOCATION, DUPLICATE, MAX_EVIDENCE, MAX_FINDINGS
from .domain import BANK, EXPENSE, PAYABLE, SUSPENSE
from .rag import EvidenceIndex

MODEL_VERSION = 1
THRESHOLD = 0.60
FEATURES = (
    "is_payment", "expense_entry_count", "expense_equal_amount_count",
    "payable_shortfall_share", "suspense_share", "bank_coverage_share",
    "currency_mismatch", "invoice_refs_count",
)


@dataclass(frozen=True)
class Candidate:
    kind: str
    affected_id: str
    currency: str
    features: tuple[float, ...]
    amount: int
    evidence: tuple[str, ...]
    explanation: str

    def finding(self):
        if self.amount <= 0 or not 1 < len(self.evidence) <= MAX_EVIDENCE:
            return None
        return {
            "type": self.kind, "affected_ids": [self.affected_id],
            "claimed_amount_minor": self.amount, "currency": self.currency,
            "evidence_ids": list(self.evidence), "explanation": self.explanation,
        }


def bounded_share(amount, total):
    return min(2.0, max(0.0, amount / total)) if total > 0 else 0.0


def invoice_candidate(invoice, index):
    invoice_id = invoice["invoice_id"]
    currency, total = invoice["currency"], invoice["total_minor"]
    expenses = [row for row in index.linked("ledger", "invoice_id", invoice_id)
                if row["account"] == EXPENSE]
    comparable = [row for row in expenses if row["currency"] == currency]
    repeated = [row for row in comparable if row["debit_minor"] == total]
    features = (
        0.0, float(min(len(expenses), 4)), float(min(len(repeated), 4)),
        0.0, 0.0, 0.0, float(len(comparable) != len(expenses)), 0.0,
    )
    evidence = (invoice["doc_id"], *(row["row_id"] for row in repeated))
    amount = total * max(0, len(repeated) - 1)
    return Candidate(DUPLICATE, invoice_id, currency, features, amount, evidence,
                     f"Документ на {total} связан с {len(repeated)} одинаковыми проводками расхода.")


def payment_candidate(payment, index):
    payment_id = payment["payment_id"]
    currency, total = payment["currency"], payment["amount_minor"]
    linked = index.linked("ledger", "payment_id", payment_id)
    same_currency = [row for row in linked if row["currency"] == currency]
    bank = [row for row in same_currency if row["account"] == BANK]
    suspense = [row for row in same_currency if row["account"] == SUSPENSE]
    refs = payment["invoice_refs"]
    allocated = [row for row in same_currency if row["account"] == PAYABLE and row["invoice_id"] in refs]
    allocated_sum = sum(row["debit_minor"] for row in allocated)
    missing = total - allocated_sum
    suspense_sum = sum(row["debit_minor"] for row in suspense)
    bank_sum = sum(row["credit_minor"] for row in bank)
    features = (
        1.0, 0.0, 0.0, bounded_share(missing, total), bounded_share(suspense_sum, total),
        bounded_share(bank_sum, total), float(len(linked) != len(same_currency)),
        float(min(len(refs), 4)),
    )
    confirmed = len(refs) == 1 and len(same_currency) == len(linked) and bank_sum == total and (
        missing > 0 and suspense_sum == missing
    )
    amount = missing if confirmed else 0
    evidence = (payment["doc_id"], *(row["row_id"] for row in [*bank, *suspense, *allocated]))
    return Candidate(ALLOCATION, payment_id, currency, features, amount, evidence,
                     f"Платёж {total}, отнесено {allocated_sum}, на транзитном счёте {suspense_sum}.")


def candidates(tables):
    index = EvidenceIndex(tables)
    return [
        *(invoice_candidate(invoice, index) for invoice in tables["invoices"]),
        *(payment_candidate(payment, index) for payment in tables["payments"]),
    ]


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    odds = math.exp(value)
    return odds / (1.0 + odds)


@dataclass(frozen=True)
class MicroModel:
    weights: tuple[float, ...] = (0.0,) * len(FEATURES)
    bias: float = 0.0
    threshold: float = THRESHOLD

    def probability(self, features):
        if len(features) != len(self.weights):
            raise ValueError("unexpected feature vector size")
        return sigmoid(self.bias + math.fsum(w * x for w, x in zip(self.weights, features)))

    def to_dict(self):
        return {"version": MODEL_VERSION, "features": list(FEATURES),
                "weights": list(self.weights), "bias": self.bias, "threshold": self.threshold}

    @classmethod
    def load(cls, path):
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("model file is too large")
        data = json.loads(raw)
        if (not isinstance(data, dict)
            or data.keys() != {"version", "features", "weights", "bias", "threshold"}
            or type(data["version"]) is not int or data["version"] != MODEL_VERSION
            or data["features"] != list(FEATURES)
            or not isinstance(data["weights"], list) or len(data["weights"]) != len(FEATURES)
            or not all(type(value) in (int, float) and -1000 <= value <= 1000
                       for value in [*data["weights"], data["bias"], data["threshold"]])
            or not 0 < data["threshold"] < 1):
            raise ValueError("invalid model artifact")
        return cls(tuple(data["weights"]), data["bias"], data["threshold"])

    def findings(self, tables):
        ranked = []
        for candidate in candidates(tables):
            probability = self.probability(candidate.features)
            if probability >= self.threshold:
                finding = candidate.finding()
                if finding is not None:
                    ranked.append((probability, candidate.affected_id, finding))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [finding for _, _, finding in ranked[:MAX_FINDINGS]]
