"""Consolidation: how a memory store improves instead of only growing.

Three operations, all opt-in and all reported:

**promote**
    An episodic memory that keeps getting recalled is a fact, not an event.
    Once its retrieval count crosses ``promote_after`` it becomes ``semantic``.
**merge**
    Near-duplicate episodes ("deploy runs migrations before deploy" and
    "deploy: runs migrations before") collapse into one record, keeping the
    highest importance and the union of tags. Merging never deletes a semantic
    memory -- a distilled fact outranks a raw episode.
**forget**
    Memories below an importance floor that were never recalled are dead
    weight. A memory that *has* been recalled is exempt: being retrieved once is
    evidence it matters, whatever its author-assigned weight.

Similarity is Jaccard overlap over content tokens, which is crude but
inspectable. Embedding-based similarity is a future change, not a hidden
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .recall import tokenize
from .store import Memory, MemoryStore

__all__ = ["ConsolidationResult", "Consolidator", "cluster_key", "similarity_score"]

# Words too common to identify a memory's subject.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "before", "but", "by",
        "for", "from", "has", "have", "in", "is", "it", "its", "of", "on",
        "or", "run", "runs", "so", "that", "the", "then", "this", "to", "was",
        "we", "were", "with",
    }
)


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1}


def cluster_key(text: str) -> str:
    """Order-insensitive, case-insensitive signature of a memory's subject."""
    return " ".join(sorted(_content_tokens(text)))


def similarity_score(left: str, right: str) -> float:
    """Jaccard overlap of significant tokens, in ``0.0..1.0``."""
    a, b = _content_tokens(left), _content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ConsolidationResult:
    """What a consolidation pass actually changed."""

    promoted: int = 0
    merged: int = 0
    forgotten: list[str] = field(default_factory=list)
    kept_ids: list[str] = field(default_factory=list)
    merged_ids: list[str] = field(default_factory=list)
    merge_pairs: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "merged": self.merged,
            "forgotten": list(self.forgotten),
        }


class Consolidator:
    """Runs promote/merge/forget passes over a :class:`MemoryStore`."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def consolidate(
        self,
        promote_after: int = 3,
        merge_similar: bool = True,
        similarity: float = 0.6,
        forget_below_importance: int | None = None,
    ) -> ConsolidationResult:
        """Run every enabled pass and report what changed."""
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("similarity must be between 0.0 and 1.0")
        result = ConsolidationResult()

        if promote_after > 0:
            for memory in self.store.all(kind="episodic"):
                if memory.access_count >= promote_after:
                    self.store.update(memory.id, kind="semantic")
                    result.promoted += 1

        if merge_similar:
            self._merge(similarity, result)

        if forget_below_importance is not None:
            for memory in self.store.all():
                # Never recalled is the exemption: being retrieved once is
                # evidence a memory matters, whatever its weight.
                never_recalled = memory.access_count == 0
                too_light = memory.importance < forget_below_importance
                if too_light and never_recalled and self.store.forget(memory.id):
                    result.forgotten.append(memory.id)

        return result

    def _merge(self, threshold: float, result: ConsolidationResult) -> None:
        survivors: list[Memory] = list(self.store.all())

        for index, keeper in enumerate(survivors):
            if keeper.id in result.merged_ids:
                continue
            for candidate in survivors[index + 1 :]:
                if candidate.id in result.merged_ids:
                    continue
                if similarity_score(keeper.content, candidate.content) < threshold:
                    continue

                # A distilled semantic fact wins over a raw episode; when both
                # are the same kind the more heavily weighted one wins.
                if _prefers(keeper, candidate):
                    loser, winner = keeper, candidate
                else:
                    loser, winner = candidate, keeper

                self.store.forget(loser.id)
                self.store.update(
                    winner.id,
                    importance=max(winner.importance, loser.importance),
                )
                self.store.add_tags(winner.id, loser.tags)

                result.merged += 1
                result.merged_ids.append(loser.id)
                result.kept_ids.append(winner.id)
                result.merge_pairs.append((winner.id, loser.id))
                break


def _prefers(left: Memory, right: Memory) -> bool:
    """True when ``left`` should be discarded in favour of ``right``."""
    if left.kind == right.kind:
        return left.importance < right.importance
    return left.kind != "semantic" and right.kind == "semantic"
