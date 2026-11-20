"""Validate findings and count maximum-cardinality one-to-one matches.

Published matcher rules:
- structure is strict: an invalid submission ends the episode with reward -1;
- an unknown finding type is valid structure but never matches, so it is a FP;
- a finding matches an issue when the type, the set of affected_ids and the
  currency are equal, |claimed - expected| <= tolerance, every cited evidence ID
  was shown to the agent and relates to the issue, and the citations contain at
  least one complete acceptable evidence set;
- ties are resolved by a maximum one-to-one matching (augmenting paths), so the
  reward depends only on the match count, never on finding order;
- unmatched and repeated findings are FP, unmatched issues FN.
"""
from dataclasses import dataclass, field

from .contracts import MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS

FINDING_KEYS = {
    "type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation",
}
STEP_COST = 0.002


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    reward: float
    inspection_steps: int
    error: str | None = None
    termination: str = "submitted"
    matches: tuple = field(default=())


def valid_ids(value, limit):
    return (
        isinstance(value, list) and 1 <= len(value) <= limit
        and all(isinstance(item, str) and 0 < len(item) <= 128 for item in value)
        and len(set(value)) == len(value)
    )


def valid_finding(finding):
    return (
        isinstance(finding, dict) and finding.keys() == FINDING_KEYS
        and isinstance(finding["type"], str) and 0 < len(finding["type"]) <= 64
        and valid_ids(finding["affected_ids"], 32)
        and type(finding["claimed_amount_minor"]) is int
        and isinstance(finding["currency"], str) and len(finding["currency"]) == 3
        and finding["currency"].isascii() and finding["currency"].isalpha()
        and finding["currency"].isupper()
        and valid_ids(finding["evidence_ids"], MAX_EVIDENCE)
        and isinstance(finding["explanation"], str)
        and len(finding["explanation"]) <= MAX_EXPLANATION
    )


def matches(finding, issue, seen):
    evidence = set(finding["evidence_ids"])
    related = set(issue.relevant_ids) if issue.relevant_ids else evidence
    return (
        finding["type"] == issue.type
        and set(finding["affected_ids"]) == set(issue.affected_ids)
        and finding["currency"] == issue.currency
        and abs(finding["claimed_amount_minor"] - issue.expected_amount_minor) <= issue.tolerance_minor
        and evidence <= seen
        and evidence <= related
        and any(set(proof) <= evidence for proof in issue.evidence_sets)
    )


def maximum_matches(edges):
    owner = {}

    def augment(candidate, visited):
        for issue_index in edges[candidate]:
            if issue_index in visited:
                continue
            visited.add(issue_index)
            if issue_index not in owner or augment(owner[issue_index], visited):
                owner[issue_index] = candidate
                return True
        return False

    for candidate in range(len(edges)):
        augment(candidate, set())
    return owner


def metrics(tp, fp, fn, steps, *, termination="submitted", pairs=()):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0
    reward = min(1.0, max(-1.0, 2 * f1 - 1 - STEP_COST * steps))
    return Score(tp, fp, fn, precision, recall, f1, reward, steps, None, termination, tuple(pairs))


def reject(issues, steps, error="invalid_submission", termination="submitted"):
    return Score(0, 0, len(issues), 0.0, 0.0, 0.0, -1.0, steps, error, termination)


def score(payload, episode_id, issues, seen, steps, termination="submitted"):
    if (
        not isinstance(payload, dict) or payload.keys() != {"episode_id", "findings"}
        or payload["episode_id"] != episode_id
        or not isinstance(payload["findings"], list)
        or len(payload["findings"]) > MAX_FINDINGS
        or not all(map(valid_finding, payload["findings"]))
    ):
        return reject(issues, steps, termination=termination)

    findings = payload["findings"]
    ordered = sorted(issues, key=lambda issue: issue.issue_id)
    edges = [
        [position for position, issue in enumerate(ordered) if matches(finding, issue, seen)]
        for finding in findings
    ]
    owner = maximum_matches(edges)
    tp = len(owner)
    pairs = sorted((finding_index, ordered[issue_index].issue_id) for issue_index, finding_index in owner.items())
    return metrics(tp, len(findings) - tp, len(issues) - tp, steps, termination=termination, pairs=pairs)


def score_empty(issues, steps, termination):
    """Step limit, timeout or a broken agent: the final answer is empty and scored by the formula."""
    return metrics(0, 0, len(issues), steps, termination=termination)
