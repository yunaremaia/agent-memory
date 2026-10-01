"""Tests for lexical recall and ranking."""

from agent_memory.recall import Recall, Recaller, tokenize
from agent_memory.store import MemoryStore


def test_tokenize_lowercases_and_splits_on_non_word_characters():
    assert tokenize("The Deploy-Script is ./deploy.sh") == [
        "the",
        "deploy",
        "script",
        "is",
        "deploy",
        "sh",
    ]


def test_tokenize_drops_empty_tokens():
    assert tokenize("   ---   ") == []


def test_recall_returns_the_matching_memory_first(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("the deploy script is ./deploy.sh", tags=["ops"])
    store.remember("coffee machine is on floor 3")
    results = Recaller(store).recall("deploy script")
    assert results[0].memory.content == "the deploy script is ./deploy.sh"


def test_recall_returns_nothing_for_an_unrelated_query(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("the deploy script is ./deploy.sh")
    assert Recaller(store).recall("quantum chromodynamics") == []


def test_recall_ranks_higher_importance_above_plain_lexical_match(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations first", importance=1)
    store.remember("deploy runs migrations first, then smoke tests", importance=5)
    results = Recaller(store).recall("deploy migrations")
    assert results[0].memory.importance == 5


def test_recall_prefers_reinforced_memories_on_a_tie(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    often = store.remember("deploy script location")
    store.remember("deploy runbook owner")
    for _ in range(3):
        store.touch(often.id)
    results = Recaller(store).recall("deploy")
    assert results[0].memory.id == often.id


def test_recall_can_be_filtered_by_kind(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy always needs a ticket", kind="procedure")
    store.remember("deploy also needs a ticket", kind="episodic")
    results = Recaller(store).recall("deploy ticket", kind="procedure")
    assert [r.memory.kind for r in results] == ["procedure"]


def test_recall_can_be_filtered_by_tag(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy needs a ticket", tags=["ops"])
    store.remember("deploy needs a ticket too", tags=["ui"])
    results = Recaller(store).recall("deploy ticket", tag="ops")
    assert [r.memory.tags for r in results] == [("ops",)]


def test_recall_respects_the_limit(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    for index in range(5):
        store.remember(f"deploy note number {index}")
    assert len(Recaller(store).recall("deploy", limit=2)) == 2


def test_recall_reinforces_what_it_returns(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy script location")
    Recaller(store).recall("deploy")
    assert store.all()[0].access_count == 1


def test_recall_does_not_reinforce_when_asked_not_to(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy script location")
    Recaller(store).recall("deploy", reinforce=False)
    assert store.all()[0].access_count == 0


def test_recall_can_disable_reinforcement_globally(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy script location")
    Recaller(store, reinforce=False).recall("deploy")
    assert store.all()[0].access_count == 0


def test_empty_query_returns_nothing(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("anything")
    assert Recaller(store).recall("   ") == []


def test_tag_names_contribute_to_matching(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("use the blue one", tags=["urgent"])
    results = Recaller(store).recall("urgent")
    assert results[0].memory.content == "use the blue one"


def test_recall_result_exposes_score_and_reason(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy script location", tags=["ops"])
    result = Recaller(store).recall("deploy script")[0]
    assert isinstance(result, Recall)
    assert result.score > 0
    assert "deploy" in result.matched_terms


def test_scores_are_ordered_descending(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy")
    store.remember("deploy deploy deploy")
    store.remember("deploy runbook")
    scores = [r.score for r in Recaller(store).recall("deploy")]
    assert scores == sorted(scores, reverse=True)


def test_multiword_query_matches_any_term(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("the staging database host")
    assert Recaller(store).recall("database hostname")[0].memory.content == (
        "the staging database host"
    )
