"""Baseline policies. Only the trusted evaluator constructs the oracle.

random     - fixed-seed explorer, submits nothing (organizer baseline 1)
reconcile  - deliberately simple reconciliation with naive date handling (baseline 2)
flag-all   - attack: flags every invoice and payment (baseline 3)
oracle     - evaluator-only: fetches the truth evidence, then submits it (baseline 4)
expert     - rule-based auditor (checks.py) over scanned tables; a strong public agent
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from random import Random

from .checks import audit as expert_audit, kzt
from .contracts import ALLOCATION, CUTOFF, DUPLICATE, ERROR_TYPES, FX_RATE, MAX_FINDINGS, TABLE_IDS

SCAN_ORDER = ("invoices", "payments", "fx", "ledger")


def submission(episode_id, findings):
    return {"type": "submit", "findings": {"episode_id": episode_id, "findings": findings[:MAX_FINDINGS]}}


def report(kind, affected, amount, currency, evidence, explanation):
    return {
        "type": kind, "affected_ids": sorted(affected), "claimed_amount_minor": amount,
        "currency": currency, "evidence_ids": list(dict.fromkeys(evidence))[:10], "explanation": explanation,
    }


class ScanPolicy:
    """Read every table page by page (within the step budget), then submit findings."""

    def __init__(self, findings_function, tables=SCAN_ORDER):
        self.findings_function = findings_function
        self.order = tables

    def __call__(self, observation):
        if "manifest" in observation:
            self.episode_id = observation["manifest"]["episode_id"]
            self.manifest, self.policy = observation["manifest"], observation["policy"]
            self.tables = {table: [] for table in TABLE_IDS}
            counts = self.manifest["counts"]
            self.pending = iter([table for table in self.order if counts.get(table)])
            self.table = next(self.pending, None)
            cursor = None
        else:
            page = observation.get("result") or {}
            self.tables[self.table].extend(page.get("records", []))
            cursor = page.get("next_cursor")
            if cursor is None:
                self.table = next(self.pending, None)
        if self.table is None or observation["remaining_steps"] <= 1:
            return submission(self.episode_id, self.findings_function(self.tables, self.manifest, self.policy))
        return {"type": "list_records", "table": self.table, "limit": 50, "cursor": cursor}


def naive_rates(tables):
    return {(row["date"], row["source_currency"]): row for row in tables["fx"]}


def reconcile(tables, manifest, policy):
    """A simple reconciliation: joins records and reads dates as plain strings.

    Deliberately naive: it compares issue dates instead of service dates and wall-clock
    dates instead of local operation dates, and treats any advance on a cited invoice as
    an allocation error. It solves easy cases and fails on the documented traps.
    """
    start, end = manifest["period_start"][:10], manifest["period_end"][:10]
    rates = naive_rates(tables)
    by_invoice, by_payment = defaultdict(list), defaultdict(list)
    for row in tables["ledger"]:
        if row["invoice_id"]:
            by_invoice[row["invoice_id"]].append(row)
        if row["payment_id"]:
            by_payment[row["payment_id"]].append(row)
    findings = []
    invoices = sorted(tables["invoices"], key=lambda row: (row["entity_id"], row["currency"], row["total_minor"], row["issued_at"]))
    for first, second in zip(invoices, invoices[1:]):
        if (first["entity_id"], first["currency"], first["total_minor"]) == (
                second["entity_id"], second["currency"], second["total_minor"]):
            gap = abs((datetime.fromisoformat(second["issued_at"]) - datetime.fromisoformat(first["issued_at"])).days)
            if gap <= 7:
                revenue = [row["row_id"] for item in (first, second) for row in by_invoice[item["invoice_id"]]
                           if row["account"] == "6010"]
                findings.append(report(DUPLICATE, [first["invoice_id"], second["invoice_id"]], second["total_minor"],
                                       second["currency"], [first["doc_id"], second["doc_id"], *revenue],
                                       "Одинаковые счёт и сумма одному покупателю в течение недели."))
    for invoice in tables["invoices"]:
        revenue = [row for row in by_invoice[invoice["invoice_id"]] if row["account"] == "6010"]
        if not revenue:
            continue
        posted = sum(row["credit_minor"] for row in revenue)
        issued = invoice["issued_at"][:10]
        if not start <= issued < end:
            findings.append(report(CUTOFF, [invoice["invoice_id"]], posted, "KZT",
                                   [invoice["doc_id"], *(row["row_id"] for row in revenue)],
                                   "Дата счёта вне периода, а выручка в периоде."))
        elif invoice["currency"] != "KZT" and (issued, invoice["currency"]) in rates:
            fx = rates[(issued, invoice["currency"])]
            expected = kzt(invoice["total_minor"], invoice["currency"], fx["rate"], manifest["minor_units"])
            if abs(posted - expected) > 2:
                findings.append(report(FX_RATE, [invoice["invoice_id"]], abs(posted - expected), "KZT",
                                       [invoice["doc_id"], fx["fx_id"], *(row["row_id"] for row in revenue)],
                                       "Пересчёт счёта не совпал с курсом на дату счёта."))
    for payment in tables["payments"]:
        rows = by_payment[payment["payment_id"]]
        bank = [row for row in rows if row["account"] == "1030" and row["debit_minor"]]
        if not bank:
            continue
        wall = payment["timestamp"][:10]
        debit = sum(row["debit_minor"] for row in bank)
        if not start <= wall < end:
            findings.append(report(CUTOFF, [payment["payment_id"]], debit, "KZT",
                                   [payment["doc_id"], *(row["row_id"] for row in bank)],
                                   "Дата платежа вне периода, а деньги проведены в периоде."))
        if payment["currency"] != "KZT" and (wall, payment["currency"]) in rates:
            fx = rates[(wall, payment["currency"])]
            expected = kzt(payment["amount_minor"], payment["currency"], fx["rate"], manifest["minor_units"])
            if abs(debit - expected) > 2:
                findings.append(report(FX_RATE, [payment["payment_id"]], abs(debit - expected), "KZT",
                                       [payment["doc_id"], fx["fx_id"], *(row["row_id"] for row in bank)],
                                       "Пересчёт платежа не совпал с курсом на дату платежа."))
        advances = [row for row in rows if row["account"] == "3510" and row["credit_minor"]]
        if payment["invoice_refs"] and advances:
            cited = [item["doc_id"] for item in tables["invoices"] if item["invoice_id"] == payment["invoice_refs"][0]]
            findings.append(report(ALLOCATION, [payment["payment_id"]], sum(row["credit_minor"] for row in advances),
                                   "KZT", [payment["doc_id"], *cited, *(row["row_id"] for row in bank),
                                           *(row["row_id"] for row in advances)],
                                   "Платёж по счёту частично ушёл в авансы."))
    return findings[:MAX_FINDINGS]


def flag_all(tables, manifest, policy):
    """Attack baseline: accuse every document of something, citing all its rows."""
    by_link = defaultdict(list)
    for row in tables["ledger"]:
        for key in ("invoice_id", "payment_id"):
            if row[key]:
                by_link[row[key]].append(row["row_id"])
    findings = []
    documents = [("invoice_id", row, row["total_minor"]) for row in tables["invoices"]]
    documents += [("payment_id", row, abs(row["amount_minor"])) for row in tables["payments"]]
    for index, (field, row, amount) in enumerate(documents[:MAX_FINDINGS]):
        kind = ERROR_TYPES[index % len(ERROR_TYPES)]
        findings.append(report(kind, [row[field]], amount, row["currency"],
                               [row["doc_id"], *by_link[row[field]][:9]], "Массовая пометка."))
    return findings


def expert(tables, manifest, policy):
    return expert_audit(manifest, policy, tables)[:MAX_FINDINGS]


class RandomPolicy:
    def __init__(self, seed=481):
        self.random = Random(seed)
        self.cursors = {}

    def __call__(self, observation):
        if "manifest" in observation:
            self.episode_id = observation["manifest"]["episode_id"]
        elif isinstance(observation.get("result"), dict):
            self.cursors[self.table] = observation["result"].get("next_cursor")
        if observation["remaining_steps"] <= 88:
            return submission(self.episode_id, [])
        self.table = self.random.choice(tuple(TABLE_IDS))
        return {"type": "list_records", "table": self.table,
                "limit": self.random.randint(1, 50), "cursor": self.cursors.get(self.table)}


class OraclePolicy:
    """Evaluator-only: look at the first acceptable evidence set of every issue, then submit the truth."""

    def __init__(self, episode):
        self.issues = episode.issues
        self.where = {}
        for table, rows in episode.tables.items():
            for row in rows:
                self.where[row[TABLE_IDS[table]]] = ("record", table, row[TABLE_IDS[table]])
                if "doc_id" in row:
                    self.where[row["doc_id"]] = ("document", table, row["doc_id"])

    def __call__(self, observation):
        if "manifest" in observation:
            self.episode_id = observation["manifest"]["episode_id"]
            wanted = []
            for issue in self.issues:
                for item in issue.evidence_sets[0]:
                    if item not in wanted:
                        wanted.append(item)
            self.queue = wanted
        if self.queue and observation["remaining_steps"] > 1:
            kind, table, item = self.where[self.queue.pop(0)]
            if kind == "document":
                return {"type": "get_document", "doc_id": item}
            return {"type": "get_record", "table": table, "id": item}
        return submission(self.episode_id, [issue.as_finding() for issue in self.issues])


def create_policy(name, model_path="models/model.json"):
    if name == "micro":
        from .micro import MicroModel

        return MicroModel.load(model_path).policy()
    return {
        "reconcile": lambda: ScanPolicy(reconcile),
        "flag-all": lambda: ScanPolicy(flag_all),
        "mark-all": lambda: ScanPolicy(flag_all),
        "expert": lambda: ScanPolicy(expert),
        "random": RandomPolicy,
    }[name]()


def oracle(episode):
    return OraclePolicy(episode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=["reconcile", "flag-all", "mark-all", "random", "expert", "micro"])
    parser.add_argument("--model", default="models/model.json")
    args = parser.parse_args()
    if args.policy == "micro":
        from .micro import MicroModel

        policy = MicroModel.load(args.model).policy()
    else:
        policy = create_policy(args.policy)
    for line in sys.stdin:
        frame = json.loads(line)
        if frame["terminated"] or frame["truncated"]:
            return
        print(json.dumps(policy(frame["observation"]), ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
