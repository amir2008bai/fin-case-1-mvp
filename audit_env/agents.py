"""Reference scripted policies. Only the trusted CLI constructs the oracle."""
import argparse
import json
import sys
from collections import defaultdict
from random import Random

from .contracts import ALLOCATION, DUPLICATE, MAX_FINDINGS, TABLE_IDS
from .domain import BANK, EXPENSE, PAYABLE, SUSPENSE


def submission(episode_id, findings):
    return {"type": "submit", "findings": {"episode_id": episode_id, "findings": findings}}


def report(kind, affected_id, amount, currency, evidence, explanation):
    return {
        "type": kind, "affected_ids": [affected_id], "claimed_amount_minor": amount,
        "currency": currency, "evidence_ids": evidence, "explanation": explanation,
    }


def reconcile(tables):
    """Simple joins over observations; the private evaluator is absent here."""
    expenses, payments = defaultdict(list), defaultdict(list)
    for row in tables["ledger"]:
        if row["account"] == EXPENSE:
            expenses[row["invoice_id"]].append(row)
        if row["payment_id"] is not None:
            payments[row["payment_id"]].append(row)

    findings = []
    for invoice in tables["invoices"]:
        rows = expenses[invoice["invoice_id"]]
        if len(rows) == 2 and all(
            row["debit_minor"] == invoice["total_minor"]
            and row["currency"] == invoice["currency"] for row in rows
        ):
            findings.append(report(
                DUPLICATE, invoice["invoice_id"], invoice["total_minor"], invoice["currency"],
                [invoice["doc_id"], *(row["row_id"] for row in rows)],
                "Один счёт дважды признан в расходах.",
            ))
    for payment in tables["payments"]:
        rows = payments[payment["payment_id"]]
        applicable = [row for row in rows if row["account"] in {BANK, SUSPENSE, PAYABLE}]
        if any(row["currency"] != payment["currency"] for row in applicable):
            continue
        bank = [row for row in applicable if row["account"] == BANK]
        suspense = [row for row in applicable if row["account"] == SUSPENSE]
        allocated = [
            row for row in applicable
            if row["account"] == PAYABLE and row["invoice_id"] in payment["invoice_refs"]
        ]
        missing = payment["amount_minor"] - sum(row["debit_minor"] for row in allocated)
        if (
            len(payment["invoice_refs"]) == 1 and len(bank) == len(suspense) == 1
            and bank[0]["credit_minor"] == payment["amount_minor"]
            and missing > 0 and suspense[0]["debit_minor"] == missing
        ):
            findings.append(report(
                ALLOCATION, payment["payment_id"], missing, payment["currency"],
                [payment["doc_id"], bank[0]["row_id"], suspense[0]["row_id"],
                 *(row["row_id"] for row in allocated)],
                "Сумма на транзитном счёте не распределена по указанному счёту.",
            ))
    return findings[:MAX_FINDINGS]


def mark_all(tables):
    findings = []
    for invoice, row in zip(tables["invoices"], tables["ledger"]):
        findings.append(report(
            DUPLICATE, invoice["invoice_id"], invoice["total_minor"], invoice["currency"],
            [invoice["doc_id"], row["row_id"]], "Массовая пометка.",
        ))
        if len(findings) == MAX_FINDINGS:
            return findings
    return findings


class ScanPolicy:
    def __init__(self, findings_function, tables=("ledger", "invoices", "payments")):
        self.findings_function = findings_function
        self.tables = {table: [] for table in tables}

    def __call__(self, observation):
        if "manifest" in observation:
            self.episode_id = observation["manifest"]["episode_id"]
            counts = observation["manifest"]["counts"]
            self.pending = (table for table in self.tables if counts[table])
            self.table = next(self.pending, None)
            cursor = None
        else:
            page = observation["result"]
            self.tables[self.table].extend(page["records"])
            cursor = page["next_cursor"]
            if cursor is None:
                self.table = next(self.pending, None)
        if self.table is None or observation["remaining_steps"] <= 1:
            return submission(self.episode_id, self.findings_function(self.tables))
        return {"type": "list_records", "table": self.table, "limit": 50, "cursor": cursor}


class RandomPolicy:
    def __init__(self):
        self.random = Random(481)
        self.cursors = {}

    def __call__(self, observation):
        if "manifest" in observation:
            self.episode_id = observation["manifest"]["episode_id"]
        else:
            self.cursors[self.table] = observation["result"]["next_cursor"]
        if observation["remaining_steps"] <= 88:
            return submission(self.episode_id, [])
        self.table = self.random.choice(("ledger", "invoices", "payments", "fx"))
        return {"type": "list_records", "table": self.table,
                "limit": self.random.randint(1, 50), "cursor": self.cursors.get(self.table)}


def create_policy(name):
    return {
        "reconcile": lambda: ScanPolicy(reconcile),
        "mark-all": lambda: ScanPolicy(mark_all),
        "random": RandomPolicy,
    }[name]()


def oracle(issues):
    return ScanPolicy(lambda _: [issue.as_finding() for issue in issues])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=["reconcile", "mark-all", "random", "micro"])
    parser.add_argument("--model", default="/app/model.json")
    args = parser.parse_args()
    if args.policy == "micro":
        from .micro import MicroModel

        policy = ScanPolicy(MicroModel.load(args.model).findings, tables=TABLE_IDS)
    else:
        policy = create_policy(args.policy)
    for line in sys.stdin:
        frame = json.loads(line)
        if frame["terminated"] or frame["truncated"]:
            return
        print(json.dumps(policy(frame["observation"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
