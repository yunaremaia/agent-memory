"""Command line interface for agent-memory.

Commands
--------
``remember TEXT``   store a memory
``recall QUERY``    retrieve the most relevant memories
``consolidate``     promote, merge and forget
``list``            dump stored memories
``forget ID``       delete one memory
``stats``           counts by kind and the tags in use
``export``          write memories as JSON
``import FILE``     read memories back from JSON

Exit codes: ``0`` success, ``1`` unused by design, ``2`` invalid input or
unreadable database. Every command takes ``--db PATH`` (default
``$AGENT_MEMORY_DB`` or ``./agent-memory.db``) so the CLI works from scripts,
cron and CI without configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .consolidate import Consolidator
from .recall import Recaller
from .store import KINDS, MemoryStore

EXIT_OK = 0
EXIT_ERROR = 2

DEFAULT_DB = "./agent-memory.db"


class UsageError(Exception):
    """Any condition that must exit with code 2."""


def _default_db() -> str:
    return __import__("os").environ.get("AGENT_MEMORY_DB", DEFAULT_DB)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-memory",
        description="Persistent, learnable memory for AI agents.",
    )
    parser.add_argument("--version", action="version", version=f"agent-memory {__version__}")
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=_default_db(),
        help=f"memory database path (default: $AGENT_MEMORY_DB or {DEFAULT_DB})",
    )
    # Repeated on every subcommand so both `agent-memory --db X list` and
    # `agent-memory list --db X` work. SUPPRESS keeps the subcommand copy from
    # overwriting a value already parsed from before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_format(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--format", choices=("text", "json"), default="text")

    remember = subparsers.add_parser("remember", parents=[common], help="store a memory")
    remember.add_argument("content")
    remember.add_argument("--tags", default="", help="comma-separated tags")
    remember.add_argument("--kind", choices=KINDS, default="episodic")
    remember.add_argument("--importance", type=int, default=3)
    add_format(remember)

    recall = subparsers.add_parser(
        "recall", parents=[common], help="retrieve the most relevant memories"
    )
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=5)
    recall.add_argument("--kind", choices=KINDS, default=None)
    recall.add_argument("--tag", default=None)
    recall.add_argument(
        "--no-reinforce",
        action="store_true",
        help="do not count these retrievals toward consolidation",
    )
    add_format(recall)

    consolidate = subparsers.add_parser(
        "consolidate",
        parents=[common],
        help="promote recurring memories, merge duplicates, forget noise",
    )
    consolidate.add_argument("--promote-after", type=int, default=3)
    consolidate.add_argument("--similarity", type=float, default=0.6)
    consolidate.add_argument("--no-merge", action="store_true")
    consolidate.add_argument(
        "--forget-below-importance",
        type=int,
        default=None,
        help="forget never-recalled memories below this importance (default: keep all)",
    )
    add_format(consolidate)

    listing = subparsers.add_parser("list", parents=[common], help="dump stored memories")
    listing.add_argument("--kind", choices=KINDS, default=None)
    listing.add_argument("--tag", default=None)
    add_format(listing)

    forget = subparsers.add_parser("forget", parents=[common], help="delete one memory by id")
    forget.add_argument("memory_id")

    stats = subparsers.add_parser("stats", parents=[common], help="counts by kind and tags in use")
    add_format(stats)

    export = subparsers.add_parser("export", parents=[common], help="write memories as JSON")
    export.add_argument("--output", default="-", help="output path, or - for stdout")
    add_format(export)

    importer = subparsers.add_parser("import", parents=[common], help="read memories from JSON")
    importer.add_argument("source", help="JSON file to read")
    importer.add_argument(
        "--skip-existing",
        action="store_true",
        help="ignore records whose content is already stored",
    )
    add_format(importer)

    return parser


def _split_tags(raw: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in raw.split(",") if tag.strip())


def _open_store(args: argparse.Namespace) -> MemoryStore:
    try:
        return MemoryStore(args.db)
    except (OSError, ValueError) as exc:
        raise UsageError(f"cannot open memory database {args.db}: {exc}") from exc


def _cmd_remember(args: argparse.Namespace, store: MemoryStore) -> int:
    try:
        memory = store.remember(
            args.content,
            tags=_split_tags(args.tags),
            kind=args.kind,
            importance=args.importance,
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc

    if args.format == "json":
        print(json.dumps(memory.as_dict(), indent=2))
    else:
        print(f"remembered {memory.id}: {memory.content}")
    return EXIT_OK


def _cmd_recall(args: argparse.Namespace, store: MemoryStore) -> int:
    recaller = Recaller(store, reinforce=not args.no_reinforce)
    hits = recaller.recall(args.query, limit=args.limit, kind=args.kind, tag=args.tag)

    if args.format == "json":
        print(json.dumps([hit.as_dict() for hit in hits], indent=2))
        return EXIT_OK

    for hit in hits:
        print(f"[{hit.score:.2f}] ({hit.memory.kind}) {hit.memory.content}")
    return EXIT_OK


def _cmd_consolidate(args: argparse.Namespace, store: MemoryStore) -> int:
    consolidator = Consolidator(store)
    try:
        result = consolidator.consolidate(
            promote_after=args.promote_after,
            merge_similar=not args.no_merge,
            similarity=args.similarity,
            forget_below_importance=args.forget_below_importance,
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc

    if args.format == "json":
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(
            f"promoted {result.promoted}, merged {result.merged}, "
            f"forgot {len(result.forgotten)}"
        )
    return EXIT_OK


def _cmd_list(args: argparse.Namespace, store: MemoryStore) -> int:
    memories = store.all(kind=args.kind, tag=args.tag)
    if args.format == "json":
        print(json.dumps([memory.as_dict() for memory in memories], indent=2))
        return EXIT_OK
    for memory in memories:
        tags = f" [{','.join(memory.tags)}]" if memory.tags else ""
        print(f"{memory.id}  ({memory.kind}, importance {memory.importance}){tags}  {memory.content}")
    return EXIT_OK


def _cmd_forget(args: argparse.Namespace, store: MemoryStore) -> int:
    if not store.forget(args.memory_id):
        raise UsageError(f"unknown memory id: {args.memory_id}")
    print(f"forgot {args.memory_id}")
    return EXIT_OK


def _cmd_stats(args: argparse.Namespace, store: MemoryStore) -> int:
    payload = store.stats()
    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(f"{payload['total']} memories")
    for kind, count in dict(payload["by_kind"]).items():
        print(f"  {kind}: {count}")
    tags = list(payload["tags"])
    print(f"tags: {', '.join(tags) if tags else '(none)'}")
    return EXIT_OK


def _cmd_export(args: argparse.Namespace, store: MemoryStore) -> int:
    payload = [memory.as_dict() for memory in store.all()]
    text = json.dumps(payload, indent=2) + "\n"
    if args.output == "-":
        print(text, end="")
        return EXIT_OK
    try:
        Path(args.output).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise UsageError(f"cannot write {args.output}: {exc}") from exc
    print(f"exported {len(payload)} memories to {args.output}")
    return EXIT_OK


def _cmd_import(args: argparse.Namespace, store: MemoryStore) -> int:
    source = Path(args.source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read memories from {source}: {exc}") from exc

    if not isinstance(payload, list):
        raise UsageError(f"{source} must contain a JSON list of memories")

    imported, skipped = 0, 0
    for record in payload:
        if not isinstance(record, dict) or "content" not in record:
            skipped += 1
            continue
        try:
            store.remember(
                str(record["content"]),
                tags=list(record.get("tags", [])),
                kind=str(record.get("kind", "episodic")),
                importance=int(record.get("importance", 3)),
            )
        except ValueError as exc:
            if not args.skip_existing:
                raise UsageError(f"cannot import record: {record!r}") from exc
            # Skipped records must not also count as imported.
            skipped += 1
            continue
        imported += 1

    if args.format == "json":
        print(json.dumps({"imported": imported, "skipped": skipped}, indent=2))
    else:
        print(f"imported {imported} memories, skipped {skipped}")
    return EXIT_OK


_HANDLERS = {
    "remember": _cmd_remember,
    "recall": _cmd_recall,
    "consolidate": _cmd_consolidate,
    "list": _cmd_list,
    "forget": _cmd_forget,
    "stats": _cmd_stats,
    "export": _cmd_export,
    "import": _cmd_import,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = _HANDLERS[args.command]
    store = None
    try:
        store = _open_store(args)
        return handler(args, store)
    except UsageError as exc:
        print(f"agent-memory: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
