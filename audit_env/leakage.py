"""Statistical tell-leakage audit: verify superficial features never predict errors above chance.

Computes exact tie-adjusted ROC-AUC (Mann-Whitney U) over closed episodes for
- presentation metadata: table row positions, HMAC ID hex values, timestamp seconds;
- structural shortcuts a policy could use without auditing: is the invoice a waybill, is it cited by a
  receipt, how many days after the invoice the receipt came, how many invoices the customer has.
Each structural feature is compared inside the population where the error type can occur (for example,
only invoices whose service lies outside the period for the cut-off tell). Every feature must stay near
0.50 on the sample; this screens for the listed shortcuts only, it does not prove that no other exists.
"""
from collections import Counter, defaultdict
from datetime import datetime

# |AUROC - 0.50| <= 0.06 on 360 episodes: about four standard errors for the rarest label (payments),
# while the `:00` seconds leak this audit once caught had a drift of 0.10.
LEAKAGE_THRESHOLD = 0.06
LEAKAGE_REPEATS = 10  # seeds per configuration: 36 configurations -> 360 episodes
LEAKAGE_SEED = 7  # fixed sample, so `verify` prints the same table on every run
# Structural populations have 100-300 positives on 360 episodes (standard error about 0.03); before the
# generator fix these tells drifted by 0.21-0.39.
STRUCTURE_THRESHOLD = 0.08
MANDATORY_INVOICE_TYPES = ("duplicate_invoice", "payment_allocation", "period_cutoff", "vat_on_top")
RECEIPT_TYPES = ("payment_allocation", "fx_rate", "vat_on_top")
STRUCTURE_DESCRIPTIONS = {
    "goods_waybill": "накладная или акт среди рублёвых (KZT) счетов за услуги/товары периода",
    "cited_by_receipt": "есть ли оплата со ссылкой на счёт, у которого услуга вне периода",
    "receipt_lag_days": "дней от счёта до оплаты (оплата в периоде, одна ссылка)",
    "customer_invoices": "число счетов у клиента (KZT)",
    "revenue_posting_time": "время суток строки выручки 6010 не полночь (ошибка периода против остальных)",
    "invoice_amount_digit": "первая цифра суммы счёта (закон Бенфорда)",
    "payment_amount_digit": "первая цифра суммы платежа (закон Бенфорда)",
}


def roc_auc(pairs):
    """Exact Mann-Whitney U ROC-AUC with tie handling; returns None if one class is missing."""
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    total = len(ordered)
    while index < total:
        end = index + 1
        while end < total and ordered[end][0] == ordered[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        pos_in_tie = sum(1 for _, label in ordered[index:end] if label)
        rank_sum += avg_rank * pos_in_tie
        index = end
    u_stat = rank_sum - positives * (positives + 1) / 2.0
    return u_stat / (positives * negatives)


def _hex_uniform(opaque_id):
    return int(opaque_id.split("_", 1)[1][:8], 16) / 0xFFFFFFFF


def _seconds(iso_text):
    return datetime.fromisoformat(iso_text).second / 59.0


def structural_pairs(episode):
    """(value, label) pairs of the structural features from the public files and the truth labels."""
    manifest = episode.manifest
    start, end = datetime.fromisoformat(manifest["period_start"]), datetime.fromisoformat(manifest["period_end"])
    first, stop = start.date().isoformat(), end.date().isoformat()
    invoices, payments = episode.tables["invoices"], episode.tables["payments"]
    by_invoice = {row["invoice_id"]: row for row in invoices}
    by_payment = {row["payment_id"]: row for row in payments}
    cited = Counter(ref for row in payments for ref in row["invoice_refs"] or ())
    per_customer = Counter(row["entity_id"] for row in invoices)
    hit = defaultdict(set)  # type -> affected ids and the invoices cited by affected receipts
    for issue in episode.issues:
        for item in issue.affected_ids:
            hit[issue.type].add(item)
            hit[issue.type].update(by_payment.get(item, {}).get("invoice_refs") or ())
    mandatory = set().union(*(hit[kind] for kind in MANDATORY_INVOICE_TYPES))
    receipts = set().union(*(hit[kind] for kind in RECEIPT_TYPES))
    out = defaultdict(list)

    def service_end(row):
        return max(line["service_period_end"] for line in row["line_items"])

    for row in invoices:
        inside = first <= service_end(row) < stop
        key = row["invoice_id"]
        if row["currency"] == manifest["base_currency"]:
            if inside:
                waybill = any(line.get("service_code", "").startswith("G-") for line in row["line_items"])
                out["goods_waybill"].append((float(waybill), int(key in mandatory)))
            out["customer_invoices"].append((per_customer[row["entity_id"]], int(key in hit["uncollected_service"])))
        if not inside and first <= row["issued_at"][:10] < stop:
            out["cited_by_receipt"].append((float(cited[key] > 0), int(key in hit["period_cutoff"])))
    cutoff_rows = {item for issue in episode.issues if issue.type == "period_cutoff"
                   for proof in issue.evidence_sets for item in proof}
    affected = {item for issue in episode.issues for item in issue.affected_ids}
    for row in episode.tables["ledger"]:
        if row["account"] == "6010":
            out["revenue_posting_time"].append((float(row["timestamp"][11:19] != "00:00:00"),
                                                int(row["row_id"] in cutoff_rows)))
    for row in invoices:
        if row["total_minor"]:
            out["invoice_amount_digit"].append((int(str(abs(row["total_minor"]))[0]), int(row["invoice_id"] in affected)))
    for row in payments:
        if row["amount_minor"]:
            out["payment_amount_digit"].append((int(str(abs(row["amount_minor"]))[0]), int(row["payment_id"] in affected)))
    for row in payments:
        refs = row["invoice_refs"] or []
        if len(refs) != 1 or refs[0] not in by_invoice or row["amount_minor"] <= 0:
            continue
        at = datetime.fromisoformat(row["timestamp"])
        if not start <= at < end:
            continue
        invoice = by_invoice[refs[0]]
        base = max(datetime.fromisoformat(invoice["issued_at"]),
                   datetime.fromisoformat(f"{service_end(invoice)}T00:00:00{manifest['timezone']}"))
        out["receipt_lag_days"].append(((at - base).total_seconds() / 86400, int(row["payment_id"] in receipts)))
    return out


def audit_leakage(episodes, threshold=LEAKAGE_THRESHOLD):
    """Evaluate presentation-leakage AUROC across episodes and return rows + violations."""
    buckets = {
        "invoice_row_pos": [],
        "payment_row_pos": [],
        "ledger_row_pos": [],
        "invoice_id_hex": [],
        "payment_id_hex": [],
        "ledger_id_hex": [],
        "invoice_issued_sec": [],
        "payment_ts_sec": [],
    }
    descriptions = {
        "invoice_row_pos": "нормализованная позиция строки в invoices",
        "payment_row_pos": "нормализованная позиция строки в payments",
        "ledger_row_pos": "нормализованная позиция строки в ledger",
        "invoice_id_hex": "старшие биты HMAC-идентификатора invoice_id",
        "payment_id_hex": "старшие биты HMAC-идентификатора payment_id",
        "ledger_id_hex": "старшие биты HMAC-идентификатора row_id",
        "invoice_issued_sec": "секунды в метке времени issued_at",
        "payment_ts_sec": "секунды в метке времени timestamp платежа",
    }
    for episode in episodes:
        affected = {item for issue in episode.issues for item in issue.affected_ids}
        evidenced = {item for issue in episode.issues for proof in issue.evidence_sets for item in proof}
        invoices = episode.tables["invoices"]
        inv_denom = max(len(invoices) - 1, 1)
        for idx, row in enumerate(invoices):
            label = 1 if row["invoice_id"] in affected else 0
            buckets["invoice_row_pos"].append((idx / inv_denom, label))
            buckets["invoice_id_hex"].append((_hex_uniform(row["invoice_id"]), label))
            buckets["invoice_issued_sec"].append((_seconds(row["issued_at"]), label))
        payments = episode.tables["payments"]
        pay_denom = max(len(payments) - 1, 1)
        for idx, row in enumerate(payments):
            label = 1 if row["payment_id"] in affected else 0
            buckets["payment_row_pos"].append((idx / pay_denom, label))
            buckets["payment_id_hex"].append((_hex_uniform(row["payment_id"]), label))
            buckets["payment_ts_sec"].append((_seconds(row["timestamp"]), label))
        ledger = episode.tables["ledger"]
        led_denom = max(len(ledger) - 1, 1)
        for idx, row in enumerate(ledger):
            label = 1 if row["row_id"] in evidenced else 0
            buckets["ledger_row_pos"].append((idx / led_denom, label))
            buckets["ledger_id_hex"].append((_hex_uniform(row["row_id"]), label))
        for name, pairs in structural_pairs(episode).items():
            buckets.setdefault(name, []).extend(pairs)
    limits = {name: threshold for name in descriptions}
    descriptions.update(STRUCTURE_DESCRIPTIONS)
    limits.update({name: max(threshold, STRUCTURE_THRESHOLD) for name in STRUCTURE_DESCRIPTIONS})

    rows = []
    violations = []
    for name in descriptions:
        pairs = buckets.get(name, [])
        limit = limits[name]
        auc = roc_auc(pairs)
        positives = sum(1 for _, label in pairs if label)
        drift = None if auc is None else abs(auc - 0.5)
        ok = drift is not None and drift <= limit
        if auc is None:
            violations.append(f"{name}: AUROC needs positive and negative examples")
        elif not ok:
            violations.append(f"{name}: AUROC={auc} exceeds 0.50±{limit:.2f}")
        rows.append({
            "feature": name,
            "threshold": limit,
            "description": descriptions[name],
            "samples": len(pairs),
            "positives": positives,
            "auroc": None if auc is None else round(auc, 4),
            "drift": None if drift is None else round(drift, 4),
            "ok": ok,
        })
    return {"episodes": len(episodes), "threshold": threshold, "features": rows, "violations": violations}
