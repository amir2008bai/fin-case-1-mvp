"""Read and write episodes in the brief's file formats, with strict validation.

Public folder: manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl,
fx.csv and optionally documents.jsonl (rich document payloads for get_document).
Answers folder (evaluator only): truth.json, provenance.json.

Loading lets anyone add, remove or edit documents and then audit them; validation
reports every problem with file and line so a UI can show it.
"""
import csv
import io
import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .checks import Auditor
from .contracts import SCHEMA_VERSION, TABLE_FIELDS, Episode, Issue

PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
LINE_KEYS = {"line_no", "description", "service_code", "quantity", "unit_price_minor", "net_minor",
             "vat_rate", "vat_minor", "total_minor", "service_period_start", "service_period_end"}
NULL = "null"
MAX_PROBLEMS = 200


class FormatError(ValueError):
    def __init__(self, problems):
        self.problems = problems[:MAX_PROBLEMS]
        super().__init__(f"{len(problems)} format problem(s): " + "; ".join(problems[:5]))


# ---- writing -----------------------------------------------------------------
def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def write_lines(path, items):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(dumps(item) + "\n")


def csv_text(fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([NULL if row[name] is None else row[name] for name in fields])
    return buffer.getvalue()


def write_csv(path, fields, rows):
    Path(path).write_text(csv_text(fields, rows), encoding="utf-8", newline="\n")


def truth_document(episode):
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode.manifest["episode_id"],
        "issues": [{
            "issue_id": issue.issue_id, "type": issue.type, "affected_ids": list(issue.affected_ids),
            "expected_amount_minor": issue.expected_amount_minor, "currency": issue.currency,
            "tolerance_minor": issue.tolerance_minor,
            "evidence_sets": [list(proof) for proof in issue.evidence_sets],
            "relevant_ids": list(issue.relevant_ids),
        } for issue in episode.issues],
    }


def write_episode(episode, public_dir, answers_dir=None, *, documents=True):
    public = Path(public_dir)
    public.mkdir(parents=True, exist_ok=True)
    write_json(public / "manifest.json", episode.manifest)
    write_json(public / "policy.json", episode.policy)
    write_csv(public / "ledger.csv", TABLE_FIELDS["ledger"], episode.tables["ledger"])
    write_csv(public / "fx.csv", TABLE_FIELDS["fx"], episode.tables["fx"])
    write_lines(public / "invoices.jsonl", episode.tables["invoices"])
    write_lines(public / "payments.jsonl", episode.tables["payments"])
    if documents and episode.documents:
        write_lines(public / "documents.jsonl", [episode.documents[key] for key in sorted(episode.documents)])
    if answers_dir is not None:
        answers = Path(answers_dir)
        answers.mkdir(parents=True, exist_ok=True)
        write_json(answers / "truth.json", truth_document(episode))
        write_json(answers / "provenance.json", episode.provenance)


# ---- reading and validation -------------------------------------------------
def nullable(value):
    return None if value == NULL else value


def is_instant(text):
    try:
        return datetime.fromisoformat(text).tzinfo is not None
    except (TypeError, ValueError):
        return False


def is_date(text):
    try:
        date.fromisoformat(text)
        return isinstance(text, str) and len(text) == 10
    except (TypeError, ValueError):
        return False


def is_currency(text):
    return isinstance(text, str) and len(text) == 3 and text.isascii() and text.isalpha() and text.isupper()


def is_int(value):
    return type(value) is int


def read_csv(path, name, problems):
    text = Path(path).read_text(encoding="utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not rows or tuple(rows[0]) != TABLE_FIELDS[name]:
        problems.append(f"{path.name}: header must be {','.join(TABLE_FIELDS[name])}")
        return []
    records = []
    for number, values in enumerate(rows[1:], start=2):
        if len(values) != len(TABLE_FIELDS[name]):
            problems.append(f"{path.name}:{number}: expected {len(TABLE_FIELDS[name])} columns")
            continue
        records.append((number, dict(zip(TABLE_FIELDS[name], values))))
    return records


def read_jsonl(path, problems):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append((number, json.loads(line)))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}:{number}: invalid JSON ({exc.msg})")
    return records


def parse_ledger(records, problems):
    rows = []
    for number, raw in records:
        where = f"ledger.csv:{number}"
        try:
            debit, credit = int(raw["debit_minor"]), int(raw["credit_minor"])
        except ValueError:
            problems.append(f"{where}: debit_minor/credit_minor must be integers in minor units")
            continue
        if debit < 0 or credit < 0:
            problems.append(f"{where}: amounts must be non-negative")
        if not raw["row_id"]:
            problems.append(f"{where}: empty row_id")
        if not is_instant(raw["timestamp"]):
            problems.append(f"{where}: timestamp must be ISO-8601 with an offset")
        if not is_currency(raw["currency"]):
            problems.append(f"{where}: currency must be a 3-letter code")
        rows.append({"row_id": raw["row_id"], "timestamp": raw["timestamp"], "account": raw["account"],
                     "debit_minor": debit, "credit_minor": credit, "currency": raw["currency"],
                     "entity_id": nullable(raw["entity_id"]), "invoice_id": nullable(raw["invoice_id"]),
                     "payment_id": nullable(raw["payment_id"])})
    return rows


def parse_fx(records, problems):
    rows = []
    for number, raw in records:
        where = f"fx.csv:{number}"
        try:
            if Decimal(raw["rate"]) <= 0:
                problems.append(f"{where}: rate must be positive")
        except InvalidOperation:
            problems.append(f"{where}: rate must be a decimal string")
        if not is_date(raw["date"]):
            problems.append(f"{where}: date must be YYYY-MM-DD")
        if not (is_currency(raw["source_currency"]) and is_currency(raw["target_currency"])):
            problems.append(f"{where}: currencies must be 3-letter codes")
        rows.append(dict(raw))
    return rows


def check_invoice(number, item, problems):
    where = f"invoices.jsonl:{number}"
    if not isinstance(item, dict) or tuple(item) != TABLE_FIELDS["invoices"]:
        problems.append(f"{where}: keys must be {', '.join(TABLE_FIELDS['invoices'])} in this order")
        return False
    if not (is_instant(item["issued_at"]) and is_instant(item["due_at"])):
        problems.append(f"{where}: issued_at/due_at must be ISO-8601 with an offset")
    if not is_currency(item["currency"]) or not is_int(item["total_minor"]):
        problems.append(f"{where}: currency code and integer total_minor are required")
    lines = item["line_items"]
    if not isinstance(lines, list) or not lines:
        problems.append(f"{where}: line_items must be a non-empty list")
        return False
    for line in lines:
        if not isinstance(line, dict) or set(line) != LINE_KEYS:
            problems.append(f"{where}: each line item needs {', '.join(sorted(LINE_KEYS))}")
            return False
        if not (is_date(line["service_period_start"]) and is_date(line["service_period_end"])):
            problems.append(f"{where}: service period dates must be YYYY-MM-DD")
        if not all(is_int(line[key]) for key in ("quantity", "unit_price_minor", "net_minor", "vat_minor", "total_minor")):
            problems.append(f"{where}: line amounts must be integers in minor units")
        elif line["net_minor"] + line["vat_minor"] != line["total_minor"]:
            problems.append(f"{where}: line {line['line_no']}: net + vat must equal total")
    if is_int(item["total_minor"]) and all(is_int(line.get("total_minor")) for line in lines):
        if sum(line["total_minor"] for line in lines) != item["total_minor"]:
            problems.append(f"{where}: total_minor must equal the sum of line totals")
    return True


def check_payment(number, item, problems):
    where = f"payments.jsonl:{number}"
    if not isinstance(item, dict) or tuple(item) != TABLE_FIELDS["payments"]:
        problems.append(f"{where}: keys must be {', '.join(TABLE_FIELDS['payments'])} in this order")
        return False
    if not is_instant(item["timestamp"]):
        problems.append(f"{where}: timestamp must be ISO-8601 with an offset")
    if not is_currency(item["currency"]) or not is_int(item["amount_minor"]):
        problems.append(f"{where}: currency code and integer amount_minor are required")
    refs = item["invoice_refs"]
    if refs is not None and not (isinstance(refs, list) and refs and all(isinstance(ref, str) for ref in refs)):
        problems.append(f"{where}: invoice_refs must be null or a non-empty list of invoice_id")
    return True


def validate(manifest, policy, tables, problems):
    for name in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                 "minor_units"):
        if name not in manifest:
            problems.append(f"manifest.json: missing {name}")
    for name in ("period",):
        if name not in policy:
            problems.append(f"policy.json: missing {name}")
    for table, field in (("ledger", "row_id"), ("invoices", "invoice_id"), ("payments", "payment_id"), ("fx", "fx_id")):
        ids = [row[field] for row in tables[table]]
        if len(ids) != len(set(ids)):
            problems.append(f"{table}: {field} values must be unique")
    docs = [row["doc_id"] for table in ("invoices", "payments") for row in tables[table]]
    if len(docs) != len(set(docs)):
        problems.append("doc_id values must be unique across invoices and payments")
    invoices = {row["invoice_id"] for row in tables["invoices"]}
    payments = {row["payment_id"] for row in tables["payments"]}
    for row in tables["payments"]:
        for ref in row["invoice_refs"] or ():
            if ref not in invoices:
                problems.append(f"payments: {row['payment_id']} cites unknown invoice {ref}")
    for row in tables["ledger"]:
        if row["invoice_id"] is not None and row["invoice_id"] not in invoices:
            problems.append(f"ledger: {row['row_id']} links unknown invoice {row['invoice_id']}")
        if row["payment_id"] is not None and row["payment_id"] not in payments:
            problems.append(f"ledger: {row['row_id']} links unknown payment {row['payment_id']}")
    units = manifest.get("minor_units", {})
    for table in ("ledger", "invoices", "payments"):
        for row in tables[table]:
            if row["currency"] not in units:
                problems.append(f"{table}: currency {row['currency']} has no minor_units in manifest")
                break
    if not problems:
        check_rates(manifest, policy, tables, problems)


def check_rates(manifest, policy, tables, problems):
    """Every foreign-currency document the checks convert needs its rate in fx.csv."""
    try:
        needed = Auditor(manifest, policy, tables).required_rates()
    except (KeyError, TypeError, ValueError, AttributeError):
        problems.append("policy.json: period.start and period.end must be ISO-8601 with an offset")
        return
    available = {(row["date"], row["source_currency"]) for row in tables["fx"]}
    for (day, currency), doc_id in sorted(needed.items()):
        if (day.isoformat(), currency) not in available:
            problems.append(f"fx.csv: no {currency} rate for {day.isoformat()} (needed by {doc_id})")


def load_episode(public_dir, answers_dir=None):
    public = Path(public_dir)
    problems = []
    missing = [name for name in PUBLIC_FILES if not (public / name).is_file()]
    if missing:
        raise FormatError([f"missing file {name}" for name in missing])
    try:
        manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8-sig"))
        policy = json.loads((public / "policy.json").read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise FormatError([f"manifest/policy: invalid JSON ({exc.msg})"]) from None
    tables = {
        "ledger": parse_ledger(read_csv(public / "ledger.csv", "ledger", problems), problems),
        "fx": parse_fx(read_csv(public / "fx.csv", "fx", problems), problems),
        "invoices": [item for number, item in read_jsonl(public / "invoices.jsonl", problems)
                     if check_invoice(number, item, problems)],
        "payments": [item for number, item in read_jsonl(public / "payments.jsonl", problems)
                     if check_payment(number, item, problems)],
    }
    tables = {name: tables[name] for name in ("ledger", "invoices", "payments", "fx")}
    if not problems:
        validate(manifest, policy, tables, problems)
    if problems:
        raise FormatError(problems)
    documents = {}
    if (public / "documents.jsonl").is_file():
        for _, item in read_jsonl(public / "documents.jsonl", problems):
            if isinstance(item, dict) and "doc_id" in item:
                documents[item["doc_id"]] = item
    manifest = dict(manifest)
    manifest["counts"] = {name: len(rows) for name, rows in tables.items()}
    manifest["document_count"] = len(tables["invoices"]) + len(tables["payments"])
    issues, provenance = (), {}
    if answers_dir is not None:
        answers = Path(answers_dir)
        truth = json.loads((answers / "truth.json").read_text(encoding="utf-8"))
        issues = tuple(Issue(
            issue_id=item["issue_id"], type=item["type"], affected_ids=tuple(item["affected_ids"]),
            expected_amount_minor=item["expected_amount_minor"], currency=item["currency"],
            evidence_sets=tuple(tuple(proof) for proof in item["evidence_sets"]),
            tolerance_minor=item.get("tolerance_minor", 0),
            relevant_ids=tuple(item.get("relevant_ids", ())),
        ) for item in truth["issues"])
        if (answers / "provenance.json").is_file():
            provenance = json.loads((answers / "provenance.json").read_text(encoding="utf-8"))
    return Episode(manifest, policy, tables, issues, provenance, documents)
