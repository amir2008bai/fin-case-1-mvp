from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.0"
MAX_STEPS = 100
MAX_PAGE = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
MAX_FRAME_BYTES = 256 * 1024
DUPLICATE = "duplicate_invoice"
ALLOCATION = "payment_allocation"
SUPPORTED_TYPES = frozenset({DUPLICATE, ALLOCATION})
TABLE_IDS = {
    "ledger": "row_id",
    "invoices": "invoice_id",
    "payments": "payment_id",
    "fx": "fx_id",
}


@dataclass(frozen=True)
class Config:
    invoice_count: int = 56
    duplicate_count: int = 1
    allocation_count: int = 1

    def __post_init__(self):
        for key, low, high in (
            ("invoice_count", 50, 95),
            ("duplicate_count", 0, 6),
            ("allocation_count", 0, 6),
        ):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key}: expected integer in [{low}, {high}]")


@dataclass(frozen=True)
class Issue:
    issue_id: str
    type: str
    affected_ids: tuple[str, ...]
    expected_amount_minor: int
    currency: str
    evidence_sets: tuple[tuple[str, ...], ...]
    tolerance_minor: int = 0

    def as_finding(self):
        return {
            "type": self.type,
            "affected_ids": list(self.affected_ids),
            "claimed_amount_minor": self.expected_amount_minor,
            "currency": self.currency,
            "evidence_ids": list(self.evidence_sets[0]),
            "explanation": "Расхождение подтверждено перечисленными записями.",
        }


@dataclass
class Episode:
    manifest: dict[str, Any]
    policy: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    issues: tuple[Issue, ...]
    provenance: dict[str, Any]
