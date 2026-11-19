"""Validate findings and count maximum-cardinality one-to-one matches."""
from dataclasses import dataclass

from .contracts import MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS, SUPPORTED_TYPES

FINDING_KEYS = {
    "type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation",
}


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


def valid_ids(value, limit):
    return (
        isinstance(value, list) and 1 <= len(value) <= limit
        and all(isinstance(item, str) and 0 < len(item) <= 128 for item in value)
        and len(set(value)) == len(value)
    )


def valid_finding(finding):
    return (
        isinstance(finding, dict) and finding.keys() == FINDING_KEYS
        and isinstance(finding["type"], str) and finding["type"] in SUPPORTED_TYPES
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
    return (
        finding["type"] == issue.type
        and set(finding["affected_ids"]) == set(issue.affected_ids)
        and finding["currency"] == issue.currency
        and abs(finding["claimed_amount_minor"] - issue.expected_amount_minor) <= issue.tolerance_minor
        and evidence <= seen
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
    return len(owner)


def reject(issues, steps, error):
    return Score(0, 0, len(issues), 0.0, 0.0, 0.0, -1.0, steps, error)


def score(payload, episode_id, issues, seen, steps):
    if (
        not isinstance(payload, dict) or payload.keys() != {"episode_id", "findings"}
        or payload["episode_id"] != episode_id
        or not isinstance(payload["findings"], list)
        or len(payload["findings"]) > MAX_FINDINGS
        or not all(map(valid_finding, payload["findings"]))
    ):
        return reject(issues, steps, "invalid_submission")

    findings = payload["findings"]
    ordered = sorted(issues, key=lambda issue: issue.issue_id)
    edges = [
        [position for position, issue in enumerate(ordered) if matches(finding, issue, seen)]
        for finding in findings
    ]
    tp = maximum_matches(edges)
    fp, fn = len(findings) - tp, len(issues) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom else 1.0
    reward = min(1.0, max(-1.0, 2 * f1 - 1 - 0.002 * steps))
    return Score(tp, fp, fn, precision, recall, f1, reward, steps)
