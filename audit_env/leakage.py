"""Statistical tell-leakage audit: verify superficial metadata never predicts errors above chance.

Computes exact tie-adjusted ROC-AUC (Mann-Whitney U) over closed episodes for
presentation and metadata features (table row positions, HMAC ID hex values,
timestamp seconds). Every presentation feature must stay near 0.50, proving
an agent cannot shortcut the audit via IDs, row order, or timestamp artifacts.
"""
from datetime import datetime

# |AUROC - 0.50| <= 0.06 on 360 episodes: about four standard errors for the rarest label (payments),
# while the `:00` seconds leak this audit once caught had a drift of 0.10.
LEAKAGE_THRESHOLD = 0.06
LEAKAGE_REPEATS = 10  # seeds per configuration: 36 configurations -> 360 episodes


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

    rows = []
    violations = []
    for name, pairs in buckets.items():
        auc = roc_auc(pairs)
        positives = sum(1 for _, label in pairs if label)
        drift = None if auc is None else abs(auc - 0.5)
        ok = drift is not None and drift <= threshold
        if not ok:
            violations.append(f"{name}: AUROC={auc} exceeds 0.50±{threshold:.2f}")
        rows.append({
            "feature": name,
            "description": descriptions[name],
            "samples": len(pairs),
            "positives": positives,
            "auroc": None if auc is None else round(auc, 4),
            "drift": None if drift is None else round(drift, 4),
            "ok": ok,
        })
    return {"episodes": len(episodes), "threshold": threshold, "features": rows, "violations": violations}
