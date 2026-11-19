"""Deterministic BM25 and exact link retrieval over episode records."""
import math
import re
from collections import Counter, defaultdict

from .contracts import TABLE_IDS

TOKEN = re.compile(r"[\w]+", re.UNICODE)
LINKS = ("invoice_id", "payment_id", "entity_id")


def terms(value):
    return TOKEN.findall(str(value).casefold())


class EvidenceIndex:
    def __init__(self, tables):
        self.rows = {}
        self.by_link = defaultdict(set)
        self.by_word = defaultdict(dict)
        self.length = {}
        for table, rows in tables.items():
            for row in rows:
                identifier = row["doc_id"] if "doc_id" in row else row[TABLE_IDS[table]]
                key = (table, identifier)
                self.rows[key] = row
                for field in LINKS:
                    if row.get(field) is not None:
                        self.by_link[(table, field, row[field])].add(key)
                if table == "payments":
                    for invoice_id in row["invoice_refs"]:
                        self.by_link[(table, "invoice_id", invoice_id)].add(key)
                counts = Counter(terms(row))
                self.length[key] = sum(counts.values())
                for word, count in counts.items():
                    self.by_word[word][key] = count
        self.average_length = sum(self.length.values()) / max(1, len(self.rows))

    def linked(self, table, field, identifier):
        keys = self.by_link.get((table, field, identifier), ())
        return [self.rows[key] for key in sorted(keys)]

    def search(self, query, *, table=None, limit=5, link=None):
        # A lexical query and a structured link, when both given, are conjunctive.
        scopes = {table} if table else TABLE_IDS
        linked = None if link is None else {
            key for name in scopes for key in self.by_link.get((name, *link), ())
        }
        query_terms = sorted(set(terms(query)))
        scores = defaultdict(float)
        if not query_terms:
            scores.update(dict.fromkeys(linked or (), 0.0))
        for word in query_terms:
            matches = self.by_word.get(word, {})
            weight = math.log1p((len(self.rows) - len(matches) + .5) / (len(matches) + .5))
            for key, count in matches.items():
                if key[0] not in scopes or (linked is not None and key not in linked):
                    continue
                length_penalty = 1.2 * (.25 + .75 * self.length[key] / self.average_length)
                scores[key] += weight * (count * 2.2) / (count + length_penalty)
        keys = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
        return [{"table": table, "evidence_id": identifier, "record": self.rows[(table, identifier)]}
                for table, identifier in keys]
