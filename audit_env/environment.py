"""Trusted JSON-compatible reset/step API (Gymnasium-style). External agents use the runner."""
import ast
import hashlib
import re
import time
from copy import deepcopy
from decimal import Decimal, DecimalException, localcontext
from operator import add, mul, neg, pos, sub, truediv

from .contracts import MAX_PAGE, MAX_STEPS, TABLE_IDS, Config, Episode
from .generator import generate
from .scoring import reject, score, score_empty

NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
BINARY = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: truediv}
UNARY = {ast.UAdd: pos, ast.USub: neg}
TASK = ("Проверьте записи компании за период и отправьте выводы об ошибках с доказательствами. "
        "Возможные типы и правила выводов — в policy.findings: duplicate_invoice, payment_allocation, fx_rate, "
        "period_cutoff, vat_on_top, uncollected_service. Ошибок может не быть. Правила учёта — в policy.")
TOOLS = {
    "list_records": {"table": sorted(TABLE_IDS), "cursor": "string|null", "limit": f"1..{MAX_PAGE}"},
    "get_record": {"table": sorted(TABLE_IDS), "id": "row_id | invoice_id | payment_id | fx_id"},
    "get_document": {"doc_id": "doc_id счёта или платежа"},
    "calculator": {"expression": "арифметика: + - * / ( ) и десятичные числа"},
    "submit": {"findings": "{episode_id, findings:[{type, affected_ids, claimed_amount_minor, currency, "
                           "evidence_ids, explanation}]}"},
}




def _clone_row(table, row):
    if table in ("ledger", "fx"):
        return dict(row)
    out = dict(row)
    if table == "payments" and isinstance(out.get("invoice_refs"), list):
        out["invoice_refs"] = list(out["invoice_refs"])
    elif table == "invoices" and isinstance(out.get("line_items"), list):
        out["line_items"] = [dict(line) for line in out["line_items"]]
    return out


def calculate(expression):
    if not isinstance(expression, str) or not re.fullmatch(r"[ 0-9.+*/()-]{1,256}", expression):
        raise ValueError("invalid_expression")
    expression = expression.strip()
    parsed = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            literal = ast.get_source_segment(expression, node)
            if literal and NUMBER.fullmatch(literal):
                return Decimal(literal)
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            return BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            return UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError("invalid_expression")

    with localcontext(prec=50, Emax=256, Emin=-256):
        value = evaluate(parsed.body)
        return "0" if not value else format(value.normalize(), "f")


class AuditEnv:
    def __init__(self, *, training=True, timeout_seconds=60, clock=time.monotonic, key=None):
        if not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be in (0, 60]")
        self.training = training
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self.key = key
        self.episode = None
        self.outcome = None
        self.final = (False, False)
        self.steps = 0

    def reset(self, seed: int, config: Config | dict | None = None) -> tuple[dict, dict]:
        return self.load_episode(generate(seed, Config() if config is None else Config.from_dict(config),
                                          key=self.key))

    def load_episode(self, episode: Episode) -> tuple[dict, dict]:
        self.episode = episode
        self.index = {
            table: {row[TABLE_IDS[table]]: row for row in rows}
            for table, rows in episode.tables.items()
        }
        self.documents = {}
        for table in ("invoices", "payments"):
            for row in episode.tables[table]:
                self.documents[row["doc_id"]] = (table, episode.documents.get(row["doc_id"], row))
        self.cursor_scope = {
            table: hashlib.sha256(
                (episode.manifest["episode_id"] + ":" + table).encode("utf-8")
            ).hexdigest()[:16] for table in episode.tables
        }
        self.seen = set()
        self.steps = 0
        self.outcome = None
        self.final = (False, False)
        self.deadline = self.clock() + self.timeout_seconds
        return deepcopy({
            "task": TASK,
            "manifest": episode.manifest,
            "policy": episode.policy,
            "tools": TOOLS,
            "remaining_steps": MAX_STEPS,
        }), {}

    def step(self, action) -> tuple[dict, float | None, bool, bool, dict]:
        if self.episode is None:
            return {"error": "reset_required", "remaining_steps": 0}, 0.0, False, True, {}
        if self.outcome is not None:
            return ({"error": "episode_finished", "remaining_steps": MAX_STEPS - self.steps},
                    0.0, *self.final, {})
        if self.clock() >= self.deadline:
            return self.expire("timeout")
        self.steps += 1
        if isinstance(action, dict) and action.get("type") == "submit":
            payload = action.get("findings") if action.keys() == {"type", "findings"} else None
            self.outcome = score(payload, self.episode.manifest["episode_id"],
                                 self.episode.issues, self.seen, self.steps - 1)
            return self.finish(terminated=True, status="submitted")
        if self.steps == MAX_STEPS:
            return self.expire("step_limit")
        try:
            observation = {"result": self.inspect(action)}
        except (ValueError, SyntaxError, DecimalException, RecursionError) as exc:
            error = exc.args[0] if type(exc) is ValueError and exc.args else "invalid_expression"
            observation = {"error": error}
        if self.clock() >= self.deadline:
            return self.expire("timeout")
        observation["remaining_steps"] = MAX_STEPS - self.steps
        return observation, 0.0, False, False, {}

    def expire(self, reason="timeout"):
        if self.episode is None or self.outcome is not None:
            return {"error": "episode_finished", "remaining_steps": 0}, 0.0, *self.final, {}
        if reason == "agent_protocol":
            self.outcome = reject(self.episode.issues, self.steps, error=reason, termination=reason)
        else:
            self.outcome = score_empty(self.episode.issues, self.steps, reason)
        return self.finish(terminated=False, status="truncated")

    def finish(self, *, terminated, status):
        self.final = (terminated, not terminated)
        reward = self.outcome.reward if self.training else None
        observation = {"status": status, "remaining_steps": MAX_STEPS - self.steps}
        if not terminated:
            observation["findings"] = []
            if self.outcome is not None and self.outcome.error == "agent_protocol":
                observation["error"] = "agent_protocol"
        return observation, reward, terminated, not terminated, {}

    def expose(self, table, row):
        self.seen.add(row[TABLE_IDS[table]])
        if "doc_id" in row:
            self.seen.add(row["doc_id"])
        return row

    def inspect(self, action):
        if not isinstance(action, dict):
            raise ValueError("invalid_action")
        kind = action.get("type")
        if kind == "calculator" and action.keys() == {"type", "expression"}:
            return calculate(action["expression"])
        if kind == "get_document" and action.keys() == {"type", "doc_id"}:
            doc_id = action["doc_id"]
            if not isinstance(doc_id, str) or doc_id not in self.documents:
                raise ValueError("unknown_id")
            table, document = self.documents[doc_id]
            self.expose(table, document)
            return deepcopy(document)
        if kind == "get_record" and action.keys() == {"type", "table", "id"}:
            table = self.table(action["table"])
            record_id = action["id"]
            if not isinstance(record_id, str) or record_id not in self.index[table]:
                raise ValueError("unknown_id")
            return _clone_row(table, self.expose(table, self.index[table][record_id]))
        if kind == "list_records" and action.keys() <= {"type", "table", "cursor", "limit"}:
            table = self.table(action.get("table"))
            limit = MAX_PAGE if action.get("limit") is None else action["limit"]
            if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
                raise ValueError("invalid_limit")
            rows = self.episode.tables[table]
            start = self.offset(table, action.get("cursor"), len(rows))
            end = start + limit
            return {
                "records": [_clone_row(table, self.expose(table, row)) for row in rows[start:end]],
                "next_cursor": f"{self.cursor_scope[table]}:{end}" if end < len(rows) else None,
            }
        raise ValueError("invalid_action")

    def table(self, value):
        if not isinstance(value, str) or value not in self.index:
            raise ValueError("unknown_table")
        return value

    def offset(self, table, cursor, size):
        if cursor is None:
            return 0
        if isinstance(cursor, str) and len(cursor) <= 32:
            scope, _, raw = cursor.partition(":")
            if scope == self.cursor_scope[table] and raw.isascii() and raw.isdecimal():
                value = int(raw)
                if 0 < value < size and raw == str(value):
                    return value
        raise ValueError("invalid_cursor")
