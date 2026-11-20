"""Read-only observer of each action and result; the answer key is absent."""
from collections import Counter

from .contracts import TABLE_IDS

ACTIONS = {"reset", "list_records", "get_record", "get_document", "calculator", "submit",
           "search_records", "group_ledger"}


class EpisodeMonitor:
    def __init__(self):
        self.actions = Counter()
        self.errors = Counter()
        self.tables = Counter()
        self.seen = set()
        self.flags = Counter()
        self.event_count = 0
        self.time_ms = 0.0
        self.max_step_ms = 0.0
        self.terminal = None
        self.findings = 0

    def __call__(self, entry, elapsed):
        self.event_count += 1
        elapsed_ms = elapsed * 1000
        self.time_ms += elapsed_ms
        self.max_step_ms = max(self.max_step_ms, elapsed_ms)
        action, observation = entry["action"], entry["observation"]
        self.episode_id = entry["episode_id"]
        kind = action.get("type") if isinstance(action, dict) else None
        self.actions[kind if isinstance(kind, str) and kind in ACTIONS else "invalid_or_no_action"] += 1
        if "error" in observation:
            self.errors[observation["error"]] += 1
        if isinstance(action, dict) and isinstance(action.get("table"), str):
            table = action["table"]
            self.tables[table if table in TABLE_IDS else "unknown"] += 1
        result = observation.get("result")
        if isinstance(result, dict):
            for row in result.get("records", [result]):
                for name in ("row_id", "doc_id", "fx_id"):
                    if isinstance(row.get(name), str):
                        self.seen.add(row[name])
                if "doc_id" in row:
                    key = "invoice_id" if "invoice_id" in row else "payment_id"
                    if isinstance(row.get(key), str):
                        self.seen.add(row[key])
        if kind == "submit":
            self.check_submission(action.get("findings"))
        if entry["terminated"] or entry["truncated"]:
            self.terminal = observation.get("status")
        self.remaining_steps = observation["remaining_steps"]

    def check_submission(self, payload):
        findings = payload.get("findings") if isinstance(payload, dict) else None
        if not isinstance(findings, list):
            self.flags["invalid_submission_structure"] += 1
            return
        self.findings += len(findings)
        keys = set()
        for finding in findings:
            if not isinstance(finding, dict):
                self.flags["invalid_finding_structure"] += 1
                continue
            evidence = finding.get("evidence_ids")
            subjects = finding.get("affected_ids")
            kind = finding.get("type")
            if not isinstance(evidence, list) or not all(isinstance(value, str) for value in evidence):
                self.flags["invalid_evidence_structure"] += 1
            else:
                self.flags["unobserved_citations"] += sum(value not in self.seen for value in evidence)
            if (isinstance(kind, str) and isinstance(subjects, list)
                    and all(isinstance(value, str) for value in subjects)):
                key = (kind, tuple(sorted(subjects)))
                if key in keys:
                    self.flags["repeat_subject"] += 1
                keys.add(key)

    def report(self):
        return {
            "episode_id": self.episode_id, "events": self.event_count,
            "remaining_steps": self.remaining_steps, "terminal": self.terminal,
            "actions": dict(self.actions), "observation_errors": dict(self.errors),
            "table_requests": dict(self.tables), "observed_evidence": len(self.seen),
            "submitted_findings": self.findings,
            "flags": {key: count for key, count in self.flags.items() if count},
            "total_ms": round(self.time_ms, 3), "max_step_ms": round(self.max_step_ms, 3),
        }
