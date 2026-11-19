from copy import deepcopy

from ..contracts import ALLOCATION, Issue
from ..domain import CURRENCY, SUSPENSE


def inject(history, payment, missing, identifier):
    """Replace part or all of the payable debit with a suspense debit."""
    debit, bank = history.payment_rows[payment["payment_id"]]
    before = deepcopy(debit)
    debit["debit_minor"] -= missing
    suspense = debit | {
        "row_id": identifier("row"), "account": SUSPENSE, "invoice_id": None,
        "debit_minor": missing,
    }
    evidence = [payment["doc_id"], bank["row_id"], suspense["row_id"]]
    if debit["debit_minor"]:
        history.ledger.append(suspense)
        evidence.append(debit["row_id"])
        after = [deepcopy(debit), deepcopy(suspense)]
    else:
        history.ledger.remove(debit)
        history.ledger.append(suspense)
        after = [deepcopy(suspense)]
    issue = Issue(
        identifier("issue"), ALLOCATION, (payment["payment_id"],),
        missing, CURRENCY, (tuple(evidence),),
    )
    return issue, {"issue_id": issue.issue_id, "before": [before], "after": after}
