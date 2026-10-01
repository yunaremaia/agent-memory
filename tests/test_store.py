"""Tests for the persistent memory store."""

import pytest

from agent_memory.store import Memory, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def test_remember_returns_a_memory_with_an_id(store):
    memory = store.remember("the deploy script is ./deploy.sh", tags=["ops"])
    assert memory.id
    assert memory.content == "the deploy script is ./deploy.sh"
    assert memory.tags == ("ops",)
    assert memory.kind == "episodic"


def test_remembered_memory_can_be_read_back(store):
    stored = store.remember("redis lives on port 6379")
    fetched = store.get(stored.id)
    assert fetched.content == "redis lives on port 6379"
    assert fetched.created_at == stored.created_at


def test_get_returns_none_for_an_unknown_id(store):
    assert store.get("does-not-exist") is None


def test_memories_survive_a_new_store_instance(tmp_path):
    path = tmp_path / "memory.db"
    first = MemoryStore(path)
    first.remember("user prefers tabs over spaces")

    second = MemoryStore(path)
    assert [m.content for m in second.all()] == ["user prefers tabs over spaces"]


def test_all_returns_memories_newest_first(store):
    store.remember("first")
    store.remember("second")
    assert [m.content for m in store.all()] == ["second", "first"]


def test_count_reflects_stored_memories(store):
    store.remember("one")
    store.remember("two")
    assert store.count() == 2


def test_duplicate_content_is_rejected(store):
    store.remember("same thing")
    with pytest.raises(ValueError):
        store.remember("same thing")


def test_empty_content_is_rejected(store):
    with pytest.raises(ValueError):
        store.remember("   ")


def test_tags_are_normalised_to_a_sorted_tuple(store):
    memory = store.remember("content", tags=["zeta", "alpha", "alpha"])
    assert memory.tags == ("alpha", "zeta")


def test_kind_defaults_to_episodic_and_can_be_set(store):
    assert store.remember("a").kind == "episodic"
    assert store.remember("b", kind="semantic").kind == "semantic"


def test_unknown_kind_is_rejected(store):
    with pytest.raises(ValueError):
        store.remember("a", kind="nonsense")


def test_importance_defaults_to_middle_and_is_bounded(store):
    assert store.remember("a").importance == 3
    assert store.remember("b", importance=0).importance == 0
    assert store.remember("c", importance=5).importance == 5
    with pytest.raises(ValueError):
        store.remember("d", importance=9)


def test_forget_removes_a_memory(store):
    memory = store.remember("temporary note")
    assert store.forget(memory.id) is True
    assert store.get(memory.id) is None
    assert store.count() == 0


def test_forget_returns_false_for_an_unknown_id(store):
    assert store.forget("nope") is False


def test_all_can_be_filtered_by_kind(store):
    store.remember("an event", kind="episodic")
    store.remember("a fact", kind="semantic")
    assert [m.kind for m in store.all(kind="semantic")] == ["semantic"]


def test_all_can_be_filtered_by_tag(store):
    store.remember("one", tags=["ops"])
    store.remember("two", tags=["ui"])
    assert [m.content for m in store.all(tag="ops")] == ["one"]


def test_memory_serialises_to_a_dict_with_string_keys(store):
    memory = store.remember("serialisable")
    payload = memory.as_dict()
    assert payload["content"] == "serialisable"
    assert isinstance(payload["created_at"], str)
    assert payload["access_count"] == 0


def test_access_count_increments_on_touch(store):
    memory = store.remember("reinforce me")
    store.touch(memory.id)
    store.touch(memory.id)
    assert store.get(memory.id).access_count == 2


def test_duplicate_tags_do_not_create_duplicate_rows(store):
    store.remember("one", tags=["a", "a"])
    assert store.tags() == ["a"]


def test_tags_lists_every_tag_in_use(store):
    store.remember("one", tags=["ops"])
    store.remember("two", tags=["ops", "ui"])
    assert store.tags() == ["ops", "ui"]


def test_database_is_created_with_its_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "memory.db"
    MemoryStore(nested).remember("deep")
    assert nested.exists()


def test_memory_dataclass_rejects_an_unknown_kind_on_construction():
    with pytest.raises(ValueError):
        Memory(id="1", content="x", kind="bogus")


def test_store_reports_stats(store):
    store.remember("one", tags=["ops"], importance=5)
    store.remember("two", tags=["ops"], kind="semantic")
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["by_kind"] == {"episodic": 1, "semantic": 1}
    assert stats["tags"] == ["ops"]


def test_add_tags_attaches_tags_to_an_existing_memory(store):
    memory = store.remember("one", tags=["ops"])
    assert store.add_tags(memory.id, ["ci", "ops"]) == ("ci", "ops")


def test_add_tags_on_an_unknown_id_is_rejected(store):
    with pytest.raises(ValueError):
        store.add_tags("nope", ["ci"])


def test_add_tags_to_a_memory_without_tags(store):
    memory = store.remember("plain")
    assert store.add_tags(memory.id, ["ops"]) == ("ops",)
    assert store.get(memory.id).tags == ("ops",)
