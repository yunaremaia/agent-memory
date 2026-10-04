"""Lexical recall over the memory store.

Retrieval is pure stdlib: tokenise, score, sort. There are no embeddings here,
which is a real limitation rather than a hidden one -- see the README. What this
does provide is deterministic, inspectable ranking that an agent can reason
about, and a reinforcement signal that makes frequently-recalled memories rank
higher over time.

Score components, all additive:

``term``
    how often query terms appear in the content and tags, normalised by content
    length so a long memory does not win by being long
``importance``
    the author's explicit 0-5 weight
``reinforcement``
    a log curve over retrieval count, so the fifth retrieval matters less than
    the first

A memory with no query-term overlap is excluded regardless of importance or
reinforcement: importance is a tiebreaker, never a reason to return something
that does not answer the question.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .store import MAX_IMPORTANCE, Memory, MemoryStore

__all__ = ["Recall", "Recaller", "tokenize"]

_TOKEN = re.compile(r"[^\W_]+")
_MAX_TERM_SCORE = 6.0
_MAX_IMPORTANCE_SCORE = 3.0
_MAX_REINFORCEMENT_SCORE = 2.0


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; punctuation is a separator, not a word."""
    return _TOKEN.findall((text or "").lower())


@dataclass(frozen=True)
class Recall:
    """One retrieved memory plus why it matched."""

    memory: Memory
    score: float
    matched_terms: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        payload = self.memory.as_dict()
        payload["score"] = round(self.score, 4)
        payload["matched_terms"] = list(self.matched_terms)
        return payload


class Recaller:
    """Ranks stored memories against a query."""

    def __init__(self, store: MemoryStore, reinforce: bool = True) -> None:
        self.store = store
        self.reinforce = reinforce

    def _score(self, memory: Memory, query_terms: Counter[str]) -> tuple[float, tuple[str, ...]] | None:
        content_terms = Counter(tokenize(memory.content))
        tag_terms = Counter(tokenize(" ".join(memory.tags)))
        matched = tuple(term for term in query_terms if term in content_terms or term in tag_terms)
        if not matched:
            return None

        # Length normalisation keeps a paragraph from outranking a one-liner
        # that matches the same number of times.
        total_tokens = sum(content_terms.values()) or 1
        hits = sum(
            query_terms[term] * (content_terms[term] + 0.5 * tag_terms[term])
            for term in matched
        )
        term_score = _MAX_TERM_SCORE * min(1.0, hits / total_tokens)

        importance_score = _MAX_IMPORTANCE_SCORE * (memory.importance / MAX_IMPORTANCE)
        reinforcement_score = _MAX_REINFORCEMENT_SCORE * math.log1p(memory.access_count)

        return term_score + importance_score + reinforcement_score, matched

    def recall(
        self,
        query: str,
        limit: int = 5,
        kind: str | None = None,
        tag: str | None = None,
        reinforce: bool | None = None,
    ) -> list[Recall]:
        """Return the best matches for ``query``, most relevant first."""
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        candidates = self.store.all(kind=kind, tag=tag)
        scored: list[Recall] = []
        for memory in candidates:
            result = self._score(memory, query_terms)
            if result is None:
                continue
            score, matched = result
            scored.append(Recall(memory=memory, score=score, matched_terms=matched))

        # created_at as tiebreaker keeps ordering deterministic between runs,
        # so ranking does not depend on uuid ordering.
        scored.sort(key=lambda hit: (-hit.score, hit.memory.created_at))
        top = scored[:limit]

        should_reinforce = self.reinforce if reinforce is None else reinforce
        if should_reinforce:
            reinforced: list[Recall] = []
            for hit in top:
                self.store.touch(hit.memory.id)
                # Re-read so the caller sees the post-touch access count.
                fresh = self.store.get(hit.memory.id)
                reinforced.append(
                    Recall(
                        memory=fresh if fresh is not None else hit.memory,
                        score=hit.score,
                        matched_terms=hit.matched_terms,
                    )
                )
            return reinforced
        return top
