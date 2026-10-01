"""Persistent memory storage for agent-memory.

SQLite, stdlib only, one row per memory plus a side table for tags. The store is
deliberately boring: an agent should be able to hand it a path, write memories
across sessions, and read them back after the process is gone.

Memory kinds
------------
``episodic``
    Something that happened -- "the deploy script is ./deploy.sh". The default.
``semantic``
    A durable fact distilled from repeated episodes, promoted by
    :mod:`agent_memory.consolidate`.
``procedure``
    How to do something, learned from doing it.

Validation is explicit rather than permissive: an unknown kind or an
out-of-range importance raises immediately, because a memory system that quietly
stores malformed records fails later and far from the cause.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

__all__ = ["KINDS", "MAX_IMPORTANCE", "Memory", "MemoryStore"]

KINDS = ("episodic", "semantic", "procedure")
MAX_IMPORTANCE = 5
DEFAULT_IMPORTANCE = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    importance   INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    last_access  TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    supersedes   TEXT
);
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tag       TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON memory_tags(tag);
"""


def _now() -> str:
    # Microsecond precision is required, not cosmetic: with second precision two
    # memories written in the same session tie on created_at and "newest first"
    # silently becomes an arbitrary uuid ordering.
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Memory:
    """One stored memory."""

    id: str
    content: str
    kind: str = "episodic"
    importance: int = DEFAULT_IMPORTANCE
    tags: tuple[str, ...] = ()
    created_at: str = ""
    last_access: str | None = None
    access_count: int = 0
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown memory kind {self.kind!r}; expected one of {KINDS}")
        if not isinstance(self.importance, int) or not 0 <= self.importance <= MAX_IMPORTANCE:
            raise ValueError(f"importance must be an int in 0..{MAX_IMPORTANCE}")

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view; timestamps are already ISO strings."""
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "importance": self.importance,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "supersedes": self.supersedes,
        }


class MemoryStore:
    """SQLite-backed storage for :class:`Memory` records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    # Closing is idempotent so `with` and explicit close() can coexist.

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writing ---------------------------------------------------------

    def remember(
        self,
        content: str,
        tags: list[str] | tuple[str, ...] = (),
        kind: str = "episodic",
        importance: int = DEFAULT_IMPORTANCE,
        supersedes: str | None = None,
    ) -> Memory:
        """Store a new memory and return it."""
        text = (content or "").strip()
        if not text:
            raise ValueError("memory content cannot be empty")
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind!r}; expected one of {KINDS}")
        if not isinstance(importance, int) or not 0 <= importance <= MAX_IMPORTANCE:
            raise ValueError(f"importance must be an int in 0..{MAX_IMPORTANCE}")

        clean_tags = tuple(sorted({str(tag).strip() for tag in tags if str(tag).strip()}))
        memory = Memory(
            id=uuid.uuid4().hex,
            content=text,
            kind=kind,
            importance=importance,
            tags=clean_tags,
            created_at=_now(),
            supersedes=supersedes,
        )
        try:
            self._connection.execute(
                "INSERT INTO memories (id, content, kind, importance, created_at, supersedes)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.content,
                    memory.kind,
                    memory.importance,
                    memory.created_at,
                    supersedes,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # content is UNIQUE: a duplicate should read as "you already know
            # this", not as a driver error leaking through the API.
            self._connection.rollback()
            raise ValueError(f"memory already stored: {text!r}") from exc
        self._connection.executemany(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
            [(memory.id, tag) for tag in clean_tags],
        )
        self._connection.commit()
        return memory

    def forget(self, memory_id: str) -> bool:
        """Delete a memory. Returns False when the id is unknown."""
        cursor = self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._connection.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
        self._connection.commit()
        return cursor.rowcount > 0

    def add_tags(self, memory_id: str, tags: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Attach tags to an existing memory; returns the full tag set.

        Used by consolidation to merge the tag sets of two memories without
        reaching into the private connection.
        """
        if self.get(memory_id) is None:
            raise ValueError(f"unknown memory id: {memory_id}")
        clean = sorted({str(tag).strip() for tag in tags if str(tag).strip()})
        self._connection.executemany(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
            [(memory_id, tag) for tag in clean],
        )
        self._connection.commit()
        return self._tags_for(memory_id)

    def touch(self, memory_id: str) -> bool:
        """Record a retrieval: bump the access counter and timestamp."""
        cursor = self._connection.execute(
            "UPDATE memories SET access_count = access_count + 1, last_access = ? WHERE id = ?",
            (_now(), memory_id),
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def update(
        self,
        memory_id: str,
        kind: str | None = None,
        importance: int | None = None,
        supersedes: str | None = None,
    ) -> Memory | None:
        """Change a memory's kind/importance/supersedes in place."""
        existing = self.get(memory_id)
        if existing is None:
            return None
        new_kind = kind if kind is not None else existing.kind
        new_importance = importance if importance is not None else existing.importance
        supersedes_id = supersedes if supersedes is not None else existing.supersedes
        # Validate before writing so a bad value cannot land in the table.
        Memory(
            id=existing.id,
            content=existing.content,
            kind=new_kind,
            importance=new_importance,
            created_at=existing.created_at,
        )
        self._connection.execute(
            "UPDATE memories SET kind = ?, importance = ?, supersedes = ? WHERE id = ?",
            (new_kind, new_importance, supersedes_id, memory_id),
        )
        self._connection.commit()
        return self.get(memory_id)

    # -- reading ---------------------------------------------------------

    def _tags_for(self, memory_id: str) -> tuple[str, ...]:
        rows = self._connection.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY tag", (memory_id,)
        ).fetchall()
        return tuple(row["tag"] for row in rows)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            kind=row["kind"],
            importance=row["importance"],
            tags=self._tags_for(row["id"]),
            created_at=row["created_at"],
            last_access=row["last_access"],
            access_count=row["access_count"],
            supersedes=row["supersedes"],
        )

    def get(self, memory_id: str) -> Memory | None:
        row = self._connection.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return None if row is None else self._row_to_memory(row)

    def all(self, kind: str | None = None, tag: str | None = None) -> list[Memory]:
        """Every memory, newest first, optionally filtered."""
        query = "SELECT * FROM memories"
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if tag is not None:
            clauses.append("id IN (SELECT memory_id FROM memory_tags WHERE tag = ?)")
            params.append(tag)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC"
        return [self._row_to_memory(row) for row in self._connection.execute(query, params)]

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])

    def tags(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT DISTINCT tag FROM memory_tags ORDER BY tag"
        ).fetchall()
        return [row["tag"] for row in rows]

    def stats(self) -> dict[str, object]:
        rows = self._connection.execute(
            "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind ORDER BY kind"
        ).fetchall()
        return {
            "total": self.count(),
            "by_kind": {row["kind"]: int(row["n"]) for row in rows},
            "tags": self.tags(),
        }
