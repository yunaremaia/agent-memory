"""End-to-end tests for the agent-memory CLI, run as real subprocesses.

The point of an in-process test would be the return value; the point of a CLI is
the process contract, so these invoke ``python -m agent_memory`` and assert on
stdout and exit codes.
"""

import json
import subprocess
import sys

import pytest


@pytest.fixture
def home(tmp_path):
    """An isolated memory database path passed via --db."""
    db = tmp_path / "memory.db"
    return str(db)


def run(*args, db, expect=None):
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory", *args, "--db", db],
        capture_output=True,
        text=True,
        check=False,
    )
    if expect is not None:
        assert result.returncode == expect, (
            f"expected exit {expect}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def test_help_exits_zero(home):
    result = run("--help", db=home, expect=0)
    assert "recall" in result.stdout


def test_version_exits_zero(home):
    result = run("--version", db=home, expect=0)
    assert "agent-memory" in result.stdout


def test_remember_then_list(home):
    run("remember", "the deploy script is ./deploy.sh", "--tags", "ops,ci", db=home, expect=0)
    result = run("list", db=home, expect=0)
    assert "the deploy script is ./deploy.sh" in result.stdout


def test_memories_persist_across_processes(home):
    run("remember", "redis lives on port 6379", db=home, expect=0)
    result = run("list", "--format", "json", db=home, expect=0)
    assert [m["content"] for m in json.loads(result.stdout)] == ["redis lives on port 6379"]


def test_duplicate_remember_exits_two(home):
    run("remember", "same thing", db=home, expect=0)
    result = run("remember", "same thing", db=home, expect=2)
    assert "already stored" in result.stderr.lower()


def test_empty_remember_exits_two(home):
    run("remember", "   ", db=home, expect=2)


def test_recall_ranks_matches_first(home):
    run("remember", "the deploy script is ./deploy.sh", db=home)
    run("remember", "coffee machine is on floor 3", db=home)
    result = run("recall", "deploy script", db=home, expect=0)
    assert "deploy script" in result.stdout
    # A memory with no query-term overlap is excluded, not ranked last.
    assert "coffee machine" not in result.stdout


def test_recall_json_includes_score_and_terms(home):
    run("remember", "the deploy script is ./deploy.sh", "--tags", "ops", db=home)
    result = run("recall", "deploy", "--format", "json", db=home, expect=0)
    payload = json.loads(result.stdout)
    assert payload[0]["score"] > 0
    assert "deploy" in payload[0]["matched_terms"]
    assert payload[0]["access_count"] == 1


def test_recall_with_no_match_exits_zero_with_empty_output(home):
    run("remember", "deploy script", db=home)
    result = run("recall", "nothing related", db=home, expect=0)
    assert result.stdout.strip() == ""


def test_recall_respects_limit(home):
    for index in range(4):
        run("remember", f"deploy note {index}", db=home)
    result = run("recall", "deploy", "--limit", "2", db=home, expect=0)
    assert len(result.stdout.strip().splitlines()) == 2


def test_recall_can_be_scoped_by_kind(home):
    run("remember", "deploy needs a ticket", "--kind", "procedure", db=home)
    run("remember", "deploy also needs a ticket", "--kind", "episodic", db=home)
    result = run("recall", "deploy ticket", "--kind", "procedure", db=home, expect=0)
    assert "also" not in result.stdout


def test_consolidate_promotes_recalled_memories(home):
    run("remember", "promoted fact", db=home)
    for _ in range(3):
        run("recall", "promoted", db=home)
    run("consolidate", "--promote-after", "2", db=home, expect=0)
    listed = json.loads(run("list", "--format", "json", db=home, expect=0).stdout)
    assert listed[0]["kind"] == "semantic"


def test_consolidate_json_reports_what_changed(home):
    run("remember", "promoted fact", db=home)
    for _ in range(3):
        run("recall", "promoted", db=home)
    result = run("consolidate", "--promote-after", "2", "--format", "json", db=home, expect=0)
    assert json.loads(result.stdout)["promoted"] == 1


def test_consolidate_merges_near_duplicates(home):
    run("remember", "deploy runs migrations before deploy", db=home)
    run("remember", "deploy: runs migrations before", db=home)
    run("consolidate", "--format", "json", db=home, expect=0)
    listed = json.loads(run("list", "--format", "json", db=home, expect=0).stdout)
    assert len(listed) == 1


def test_consolidate_on_empty_store_succeeds(home):
    run("consolidate", db=home, expect=0)


def test_stats_json_reports_counts(home):
    run("remember", "one", "--tags", "ops", db=home)
    run("remember", "two", "--kind", "semantic", db=home)
    payload = json.loads(run("stats", "--format", "json", db=home, expect=0).stdout)
    assert payload["total"] == 2
    assert payload["by_kind"] == {"episodic": 1, "semantic": 1}


def test_forget_removes_by_id(home):
    stored = json.loads(run("remember", "temporary", "--format", "json", db=home, expect=0).stdout)
    run("forget", stored["id"], db=home, expect=0)
    assert json.loads(run("list", "--format", "json", db=home, expect=0).stdout) == []


def test_forget_unknown_id_exits_two(home):
    run("forget", "no-such-id", db=home, expect=2)


def test_export_json_is_valid_json(home):
    run("remember", "one", "--tags", "ops", db=home)
    run("remember", "two", db=home)
    exported = run("export", "--format", "json", db=home, expect=0).stdout
    assert sorted(m["content"] for m in json.loads(exported)) == ["one", "two"]


def test_import_from_json_file_restores_memories(home, tmp_path):
    run("remember", "one", "--tags", "ops", db=home)
    run("remember", "two", db=home)
    exported = run("export", "--format", "json", db=home, expect=0).stdout

    source = tmp_path / "export.json"
    source.write_text(exported, encoding="utf-8")

    other = str(tmp_path / "other.db")
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory", "import", str(source), "--db", other],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    restored = json.loads(
        subprocess.run(
            [sys.executable, "-m", "agent_memory", "list", "--format", "json", "--db", other],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    )
    assert sorted(m["content"] for m in restored) == ["one", "two"]
    assert [m["tags"] for m in restored if m["content"] == "one"] == [["ops"]]


def test_import_of_a_malformed_file_exits_two(home, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory", "import", str(bad), "--db", home],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2


def test_invalid_importance_exits_two(home):
    run("remember", "x", "--importance", "9", db=home, expect=2)


def test_invalid_kind_exits_two(home):
    run("remember", "x", "--kind", "nonsense", db=home, expect=2)


def test_skip_existing_imports_nothing_and_counts_one_skip(tmp_path, home):
    stored = json.loads(
        subprocess.run(
            [sys.executable, "-m", "agent_memory", "remember", "only one", "--format", "json", "--db", home],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    )
    assert stored["content"] == "only one"

    export = tmp_path / "one.json"
    export.write_text(
        json.dumps([{"content": "only one", "kind": "episodic", "importance": 3, "tags": []}]),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory", "import", str(export), "--skip-existing",
         "--format", "json", "--db", home],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"imported": 0, "skipped": 1}
