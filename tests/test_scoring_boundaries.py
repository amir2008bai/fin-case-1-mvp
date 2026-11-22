"""Exact limits of a finding: the value at the limit is accepted, one past it is rejected.

Written for the mutants that survived tools/grader_mutation.py: each assertion pins one boundary.
"""
import unittest

from audit_env.contracts import MAX_AMOUNT_MINOR, MAX_EVIDENCE, MAX_EXPLANATION
from audit_env.contracts import Issue
from audit_env.scoring import matches, valid_finding

ID = "row_" + "0" * 24


def finding(**changes):
    base = {"type": "fx_rate", "affected_ids": ["payment_" + "1" * 24], "claimed_amount_minor": 100,
            "currency": "KZT", "evidence_ids": [ID], "explanation": ""}
    return base | changes


class FindingLimitTests(unittest.TestCase):
    def test_explanation_length(self):
        self.assertTrue(valid_finding(finding(explanation="x" * MAX_EXPLANATION)))
        self.assertFalse(valid_finding(finding(explanation="x" * (MAX_EXPLANATION + 1))))

    def test_amount_bound(self):
        self.assertTrue(valid_finding(finding(claimed_amount_minor=MAX_AMOUNT_MINOR)))
        self.assertTrue(valid_finding(finding(claimed_amount_minor=-MAX_AMOUNT_MINOR)))
        self.assertFalse(valid_finding(finding(claimed_amount_minor=MAX_AMOUNT_MINOR + 1)))
        self.assertFalse(valid_finding(finding(claimed_amount_minor=10 * MAX_AMOUNT_MINOR)))

    def test_currency_is_exactly_three_letters(self):
        self.assertTrue(valid_finding(finding(currency="USD")))
        for bad in ("US", "USDT", "usd", "U$D", ""):
            self.assertFalse(valid_finding(finding(currency=bad)), bad)

    def test_id_length(self):
        self.assertTrue(valid_finding(finding(evidence_ids=["r" * 128])))
        self.assertFalse(valid_finding(finding(evidence_ids=["r" * 129])))
        self.assertFalse(valid_finding(finding(evidence_ids=[""])))

    def test_type_length(self):
        self.assertTrue(valid_finding(finding(type="t" * 64)))
        self.assertFalse(valid_finding(finding(type="t" * 65)))
        self.assertFalse(valid_finding(finding(type="")))

    def test_evidence_count(self):
        ids = [f"row_{i:024x}" for i in range(MAX_EVIDENCE + 1)]
        self.assertTrue(valid_finding(finding(evidence_ids=ids[:MAX_EVIDENCE])))
        self.assertFalse(valid_finding(finding(evidence_ids=ids)))


class EvidenceSubsetTests(unittest.TestCase):
    """Evidence equal to everything the agent saw, or to everything related, still counts (not a strict subset)."""

    def test_evidence_may_equal_seen_and_related(self):
        doc, fx_id, row = "doc_" + "2" * 24, "fx_" + "3" * 24, ID
        issue = Issue(issue_id="issue_" + "4" * 24, type="fx_rate", affected_ids=("payment_" + "1" * 24,),
                      expected_amount_minor=100, currency="KZT", evidence_sets=((doc, fx_id, row),),
                      tolerance_minor=2, relevant_ids=(doc, fx_id, row))
        claim = finding(evidence_ids=[doc, fx_id, row])
        self.assertTrue(matches(claim, issue, seen={doc, fx_id, row}))


if __name__ == "__main__":
    unittest.main()
