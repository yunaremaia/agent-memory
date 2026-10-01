# agent-memory

Persistent, learnable memory system for AI agents — stores, retrieves, and evolves agent context across sessions.

> **Status: 0.1.0.** The store, the retrieval ranking and the consolidation passes below are implemented and tested. Embedding-based semantic recall is **not** implemented — recall is lexical (see [Limitations](#limitations)).

## Problem

An agent that starts every session from zero has to rediscover everything: the deploy command, where the staging database lives, that this user wants tabs. If that context only lives in a transcript, it dies with the session and cannot be corrected when wrong.

Chat-history-as-memory does not fix this. History grows without bound, gets truncated mid-sentence, and treats "the deploy script is ./deploy.sh" and "yes" as equally worth keeping. Memories also do not improve: nothing distinguishes a fact that proved useful eleven times from a note written once and never read again.

## Solution

`agent-memory` is a small SQLite-backed memory store built around three operations:

| Operation | Module | What it does |
|-----------|--------|--------------|
| **store** | `agent_memory.store` | Persists typed, tagged, importance-weighted memories across sessions |
| **retrieve** | `agent_memory.recall` | Ranks memories against a query; every retrieval reinforces the hit |
| **evolve** | `agent_memory.consolidate` | Promotes recurring memories to facts, merges duplicates, forgets dead weight |

The `evolve` half is the differentiator: a store that only appends gets worse with use. Here, what gets recalled repeatedly gets promoted, near-duplicates collapse, and what never gets retrieved can be pruned.

### Install

agent-memory is not on PyPI yet. Install it from the repository:

```bash
pip install git+https://github.com/yunaremaia/agent-memory.git
```

Or, for development:

```bash
git clone https://github.com/yunaremaia/agent-memory.git
cd agent-memory
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
# Store something worth remembering
agent-memory remember "the deploy script is ./deploy.sh" --tags ops,ci --importance 5

# Retrieve it later, in any session, from any working directory
agent-memory recall "how do I deploy?"

# Inspect and maintain
agent-memory list
agent-memory stats
agent-memory forget <memory-id>
agent-memory export --output memories.json
agent-memory import memories.json --skip-existing

# Let the store improve
agent-memory consolidate
```

By default the database is `./agent-memory.db`. Point it anywhere with `--db PATH` (works before or after the subcommand) or the `AGENT_MEMORY_DB` environment variable:

```bash
agent-memory --db ~/.agent/memory.db recall "release checklist"
AGENT_MEMORY_DB=./project.db agent-memory remember "the CI gate is ./scripts/ci.sh"
```

### Memory kinds

| Kind | Meaning |
|------|---------|
| `episodic` | Something that happened — "the deploy script is ./deploy.sh". The default. |
| `semantic` | A durable fact, promoted automatically from a repeatedly-recalled episode. |
| `procedure` | How to do something. |

Importance is an explicit `0`–`5` weight. `--importance 5` marks a fact you want returned first; `0` marks trivia that consolidation may prune.

### How recall ranks

Every candidate is scored, and only memories with actual query-term overlap are returned:

| Component | Max contribution | What it measures |
|-----------|------------------|------------------|
| `term` | 6.0 | Query-term frequency in content and tags, normalised by content length so a long memory cannot win by being long |
| `importance` | 3.0 | The weight you assigned at write time |
| `reinforcement` | 2.0 | `log1p(access_count)` — a log curve, so the fifth retrieval matters less than the first |

Term scores saturate at the maximum, so one extremely repetitive memory cannot dominate the ranking outright.

Retrieval is **reinforcing by default**: a returned memory's access counter increments, which raises its future score and makes it a candidate for promotion. Pass `--no-reinforce` to explore without changing the store (a read-only recall then leaves every counter untouched).

### How consolidation evolves the store

```bash
agent-memory consolidate --promote-after 3 --forget-below-importance 1
```

**promote** — an `episodic` memory recalled at least `--promote-after` times becomes `semantic`. Something an agent keeps needing was never an event, it was a fact.

**merge** — memories whose significant tokens overlap by at least `--similarity` (default `0.6`) collapse into one, keeping the higher importance and the union of tags. A `semantic` memory always wins over an `episodic` one: a distilled fact outranks a raw episode.

**forget** — with `--forget-below-importance N`, memories below `N` that have **never** been recalled are deleted. Recalled-once memories are exempt whatever their weight: retrieval is evidence.

Each pass reports exactly what it changed:

```
$ agent-memory consolidate --promote-after 2 --format json
{
  "promoted": 1,
  "merged": 0,
  "forgotten": []
}
```

## Library use

```python
from agent_memory import Consolidator, MemoryStore, Recaller

store = MemoryStore("agent-memory.db")
store.remember("the deploy script is ./deploy.sh", tags=["ops"], importance=5)

hits = Recaller(store).recall("how do I deploy?")
print(hits[0].memory.content, hits[0].score, hits[0].matched_terms)

result = Consolidator(store).consolidate(promote_after=3)
print(result.promoted, result.merged, result.forgotten)
```

## Limitations

These are real gaps, not oversights:

- **Recall is lexical, not semantic.** "how do I ship?" will not find "the deploy script is ./deploy.sh" — there is no shared token. Embedding-based recall is the obvious next step and is not implemented; adding it means a `Recaller` subclass with the same interface, not a rewrite.
- **Similarity is Jaccard token overlap.** It catches reworded duplicates, not paraphrases. The threshold is tunable because it is imprecise.
- **One writer.** SQLite serialises access, so concurrent agents writing the same database contend. There is no vector index and no background scheduler: consolidation is an explicit call.

## Roadmap

- [x] SQLite store with kinds, tags, importance, access counting
- [x] Lexical recall with inspectable scoring and reinforcement
- [x] Consolidation: promote, merge, forget
- [x] CLI (`remember`, `recall`, `consolidate`, `list`, `forget`, `stats`, `export`, `import`)
- [x] JSON output for every command
- [ ] Embedding-based recall behind the same `Recaller` interface
- [ ] Decay/forgetting over time (age-weighted scoring)
- [ ] Namespaces and scoping per project or per agent role
- [ ] Concurrent-writer safety beyond SQLite's own locking

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

CI runs the suite plus a cross-process CLI smoke test on Python 3.11, 3.12 and 3.13.

## Stack

- **Language:** Python 3.11+
- **Storage:** SQLite (stdlib `sqlite3`)
- **Retrieval:** pure stdlib tokenisation and scoring — no embeddings, no vector database
- **Dependencies:** none at runtime
- **Tests:** `pytest`

## License

MIT

## Project Links

- Repository: https://github.com/yunaremaia/agent-memory
- Issues: https://github.com/yunaremaia/agent-memory/issues

## Related

- [`context-bridge`](https://github.com/yunaremaia/context-bridge) — universal session memory: capture, index, recall across any AI coding agent. agent-memory differs by focusing on the *evolution* step: promoting what keeps getting recalled and pruning what never does.
