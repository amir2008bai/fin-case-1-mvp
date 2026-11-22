"""Shared limits, finding types, configuration and episode containers."""
import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "2.0"
MAX_STEPS = 100
MAX_PAGE = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
MAX_FRAME_BYTES = 256 * 1024
MAX_ISSUES = 12
MAX_AMOUNT_MINOR = 10**18

DUPLICATE = "duplicate_invoice"
ALLOCATION = "payment_allocation"
FX_RATE = "fx_rate"
CUTOFF = "period_cutoff"
ERROR_TYPES = (DUPLICATE, ALLOCATION, FX_RATE, CUTOFF)
# Extra types from real practice of a Kazakhstan service company; off by default.
VAT_ON_TOP = "vat_on_top"
UNCOLLECTED = "uncollected_service"
EXTRA_TYPES = (VAT_ON_TOP, UNCOLLECTED)
ALL_TYPES = ERROR_TYPES + EXTRA_TYPES
SUPPORTED_TYPES = frozenset(ALL_TYPES)
# Published matcher tolerances in minor units of the finding currency.
TOLERANCE_MINOR = {DUPLICATE: 0, ALLOCATION: 0, FX_RATE: 2, CUTOFF: 0, VAT_ON_TOP: 0, UNCOLLECTED: 0}

FX_BOUNDARY = "fx_boundary_payment"
FX_PARTIAL = "fx_partial_allocation"
INTERACTIONS = (FX_BOUNDARY, FX_PARTIAL)
DISTRACTORS = (
    "split_payments", "multi_act_payments", "overpayments", "prepayments",
    "unreferenced_payments", "opening_balances", "late_acts", "boundary_timestamps",
    "repeated_amounts", "refunds",
    # look-alikes that only the hard level adds
    "double_payments", "paid_renewals", "underpaid_vat", "partial_refs",
)
DIFFICULTIES = ("easy", "medium", "hard")
# How densely distractors appear: easy halves, hard doubles and adds the hard-only look-alikes.
INTENSITY = {"easy": 0.5, "medium": 1.0, "hard": 2.0}
TIMEZONES = ("+03:00", "+05:00", "+06:00")
KEYSPACES = ("open", "secret")

TABLE_IDS = {
    "ledger": "row_id",
    "invoices": "invoice_id",
    "payments": "payment_id",
    "fx": "fx_id",
}
TABLE_FIELDS = {
    "ledger": ("row_id", "timestamp", "account", "debit_minor", "credit_minor",
               "currency", "entity_id", "invoice_id", "payment_id"),
    "invoices": ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at",
                 "currency", "total_minor", "line_items"),
    "payments": ("doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"),
    "fx": ("fx_id", "date", "source_currency", "target_currency", "rate"),
}


def _int_in(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected integer in [{low}, {high}]")


@dataclass(frozen=True)
class Config:
    """Episode configuration. Unknown keys and out-of-range values are rejected."""

    keyspace: str = "secret"  # fail safe: an open-key episode can be regenerated from the public code
    presentation_seed: int = 0
    difficulty: str = "medium"
    month: int | None = None
    timezone: str = "+05:00"
    acts: int = 40
    customers: int = 18
    foreign_percent: int = 20
    goods_percent: int = 50
    vat: bool = True
    filler: int = 30
    errors: dict = field(default_factory=lambda: {kind: 1 for kind in ERROR_TYPES})
    interactions: dict = field(default_factory=lambda: {kind: 0 for kind in INTERACTIONS})
    distractors: dict = field(default_factory=lambda: {kind: True for kind in DISTRACTORS})

    def __post_init__(self):
        if self.keyspace not in KEYSPACES:
            raise ValueError(f"keyspace: expected one of {KEYSPACES}")
        _int_in("presentation_seed", self.presentation_seed, 0, 2**63 - 1)
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty: expected one of {DIFFICULTIES}")
        if self.month is not None:
            _int_in("month", self.month, 1, 12)
        if self.timezone not in TIMEZONES:
            raise ValueError(f"timezone: expected one of {TIMEZONES}")
        _int_in("acts", self.acts, 15, 80)
        _int_in("customers", self.customers, 5, 60)
        _int_in("foreign_percent", self.foreign_percent, 0, 50)
        _int_in("goods_percent", self.goods_percent, 0, 80)
        if type(self.vat) is not bool:
            raise ValueError("vat: expected boolean")
        _int_in("filler", self.filler, 0, 800)
        object.__setattr__(self, "errors", self._merge("errors", self.errors, ALL_TYPES, 0))
        object.__setattr__(self, "interactions",
                           self._merge("interactions", self.interactions, INTERACTIONS, 0))
        object.__setattr__(self, "distractors",
                           self._merge("distractors", self.distractors, DISTRACTORS, True))
        for kind, count in self.errors.items():
            _int_in(f"errors.{kind}", count, 0, 6)
        for kind, count in self.interactions.items():
            _int_in(f"interactions.{kind}", count, 0, 3)
        for kind, enabled in self.distractors.items():
            if type(enabled) is not bool:
                raise ValueError(f"distractors.{kind}: expected boolean")
        if self.issue_count > MAX_ISSUES:
            raise ValueError(f"at most {MAX_ISSUES} issues per episode")

    @staticmethod
    def _merge(name, value, keys, default):
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected object")
        unknown = sorted(set(value) - set(keys))
        if unknown:
            raise ValueError(f"{name}: unknown keys {unknown}")
        return {key: value.get(key, default) for key in keys}

    @property
    def issue_count(self):
        return sum(self.errors.values()) + 2 * sum(self.interactions.values())

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, Config):
            return value
        if not isinstance(value, dict):
            raise ValueError("config: expected object")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"config: unknown keys {unknown}")
        return cls(**value)

    def as_dict(self):
        return asdict(self)

    def semantic_json(self):
        """Canonical JSON of everything that shapes the hidden world (not IDs/order)."""
        data = self.as_dict()
        data.pop("presentation_seed")
        return json.dumps(data, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Issue:
    issue_id: str
    type: str
    affected_ids: tuple[str, ...]
    expected_amount_minor: int
    currency: str
    evidence_sets: tuple[tuple[str, ...], ...]
    tolerance_minor: int = 0
    relevant_ids: tuple[str, ...] = ()

    def as_finding(self, explanation="Расхождение подтверждено перечисленными записями."):
        return {
            "type": self.type,
            "affected_ids": list(self.affected_ids),
            "claimed_amount_minor": self.expected_amount_minor,
            "currency": self.currency,
            "evidence_ids": list(self.evidence_sets[0]),
            "explanation": explanation,
        }


@dataclass
class Episode:
    manifest: dict[str, Any]
    policy: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    issues: tuple[Issue, ...]
    provenance: dict[str, Any]
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
