"""Use the recorded causal changes: revert them to recover the clean books."""
from copy import deepcopy

from .contracts import Episode
from .render import canonical_hash


def restore_clean(episode):
    """Revert every recorded change; the result must hash to clean_ledger_sha256."""
    ledger = {row["row_id"]: row for row in episode.tables["ledger"]}
    removed_invoices = set()
    for change in reversed(episode.provenance["changes"]):
        for row in change["after"]:
            if ledger.pop(row["row_id"], None) != row:
                raise ValueError(f"changed row {row['row_id']} does not match the books")
        for row in change["before"]:
            ledger[row["row_id"]] = row
        removed_invoices |= {item["invoice_id"] for item in change["documents_added"]}
    tables = deepcopy(episode.tables)
    tables["ledger"] = sorted(ledger.values(), key=lambda row: row["row_id"])
    tables["invoices"] = [row for row in tables["invoices"] if row["invoice_id"] not in removed_invoices]
    if canonical_hash(tables["ledger"]) != episode.provenance["clean_ledger_sha256"]:
        raise ValueError("reverted ledger does not match the recorded clean ledger hash")
    manifest = deepcopy(episode.manifest)
    manifest["counts"] = {name: len(rows) for name, rows in tables.items()}
    manifest["document_count"] = len(tables["invoices"]) + len(tables["payments"])
    documents = {key: value for key, value in episode.documents.items()
                 if value.get("invoice_id") not in removed_invoices}
    return Episode(manifest, deepcopy(episode.policy), tables, (), deepcopy(episode.provenance), documents)
