"""Opt-in public retrieval tools. The standard competition environment is unchanged."""
from collections import Counter

from .environment import AuditEnv
from .rag import EvidenceIndex, LINKS


class RagAuditEnv(AuditEnv):
    def load_episode(self, episode):
        self.evidence_index = EvidenceIndex(episode.tables)
        return super().load_episode(episode)

    def inspect(self, action):
        if isinstance(action, dict):
            if action.get("type") == "search_records":
                return self.search_records(action)
            if action.get("type") == "group_ledger":
                return self.group_ledger(action)
        return super().inspect(action)

    def search_records(self, action):
        if not action.keys() <= {"type", "query", "table", "link", "limit"}:
            raise ValueError("invalid_action")
        table = self.table(action["table"]) if "table" in action else None
        query, link, limit = action.get("query", ""), action.get("link"), action.get("limit", 6)
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("invalid_query")
        if type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("invalid_limit")
        if link is not None:
            if (not isinstance(link, dict) or link.keys() != {"field", "id"}
                    or link["field"] not in LINKS or not isinstance(link["id"], str)
                    or not 1 <= len(link["id"]) <= 128):
                raise ValueError("invalid_link")
            link = (link["field"], link["id"])
        if not query.strip() and link is None:
            raise ValueError("invalid_query")
        matches = self.evidence_index.search(query, table=table, limit=limit, link=link)
        return {
            "records": [self.expose(hit["table"], hit["record"]) for hit in matches],
            "sources": [{"table": hit["table"], "evidence_id": hit["evidence_id"]} for hit in matches],
        }

    def group_ledger(self, action):
        if not {"type", "by"} <= action.keys() <= {"type", "by", "account", "min_count", "limit"}:
            raise ValueError("invalid_action")
        field = action["by"]
        account = action.get("account")
        minimum, limit = action.get("min_count", 1), action.get("limit", 12)
        if field not in LINKS or (account is not None and (not isinstance(account, str) or len(account) > 64)):
            raise ValueError("invalid_group")
        if (type(minimum) is not int or not 1 <= minimum <= 2000
                or type(limit) is not int or not 1 <= limit <= 20):
            raise ValueError("invalid_limit")
        counts = Counter(
            row[field] for row in self.episode.tables["ledger"]
            if row[field] is not None and (account is None or row["account"] == account)
        )
        groups = sorted(((-count, key) for key, count in counts.items() if count >= minimum))
        # Only group keys and counts are returned. Search for source records before citing evidence.
        return {
            "groups": [{"id": key, "count": -count} for count, key in groups[:limit]],
            "total_groups": len(groups),
        }
