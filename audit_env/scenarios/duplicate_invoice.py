from copy import deepcopy

from ..contracts import DUPLICATE, Issue
from ..domain import CURRENCY, EXPENSE, PAYABLE
from ..history import posting


def inject(history, invoice, identifier):
    """Recognize a second posting under the same source invoice."""
    original = history.invoice_rows[invoice["invoice_id"]]
    copy = posting(
        identifier, invoice, invoice["issued_at"], EXPENSE, PAYABLE, invoice["total_minor"]
    )
    history.ledger.extend(copy)
    issue = Issue(
        identifier("issue"), DUPLICATE, (invoice["invoice_id"],),
        invoice["total_minor"], CURRENCY,
        tuple((invoice["doc_id"], left["row_id"], right["row_id"])
              for left, right in zip(original, copy)),
    )
    return issue, {"issue_id": issue.issue_id, "before": [], "after": deepcopy(copy)}
