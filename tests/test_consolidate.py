"""Tests for consolidation -- the "evolves" half of the memory system."""

from agent_memory.consolidate import Consolidator, cluster_key
from agent_memory.store import MemoryStore


def test_cluster_key_normalises_wording_and_order():
    assert cluster_key("We run migrations before the deploy") == cluster_key(
        "deploy: migrations run before"
    )


def test_cluster_key_ignores_short_words_and_case():
    assert cluster_key("The Deploy Runs Migrations") == cluster_key("migrations deploy")


def test_cluster_key_of_empty_text_is_empty():
    assert cluster_key("   ") == ""


def test_recalled_memories_are_promoted_to_semantic(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("the deploy script is ./deploy.sh")
    for _ in range(3):
        store.touch(memory.id)
    result = Consolidator(store).consolidate(promote_after=2)
    assert result.promoted == 1
    assert store.get(memory.id).kind == "semantic"


def test_rarely_recalled_memory_is_left_alone(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("rare note")
    result = Consolidator(store).consolidate(promote_after=3)
    assert result.promoted == 0
    assert store.get(memory.id).kind == "episodic"


def test_already_semantic_memory_is_not_promoted_again(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("known fact", kind="semantic")
    for _ in range(5):
        store.touch(memory.id)
    assert Consolidator(store).consolidate(promote_after=2).promoted == 0


def test_near_duplicate_episodes_are_merged_into_one(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations before deploy")
    store.remember("deploy: runs migrations before")
    result = Consolidator(store).consolidate(merge_similar=True, similarity=0.6)
    assert result.merged == 1
    assert store.count() == 1


def test_distinct_memories_are_not_merged(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations")
    store.remember("coffee machine is on floor 3")
    result = Consolidator(store).consolidate(merge_similar=True, similarity=0.6)
    assert result.merged == 0
    assert store.count() == 2


def test_merge_can_be_disabled(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations before deploy")
    store.remember("deploy: runs migrations before")
    result = Consolidator(store).consolidate(merge_similar=False)
    assert result.merged == 0
    assert store.count() == 2


def test_low_importance_never_retrieved_memories_are_forgotten(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    stale = store.remember("old trivia", importance=0)
    keeper = store.remember("important fact", importance=5)
    result = Consolidator(store).consolidate(forget_below_importance=1)
    assert result.forgotten == [stale.id]
    assert store.get(stale.id) is None
    assert store.get(keeper.id) is not None


def test_a_reinforced_low_importance_memory_is_kept(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("old but useful", importance=0)
    for _ in range(2):
        store.touch(memory.id)
    result = Consolidator(store).consolidate(forget_below_importance=1)
    assert result.forgotten == []
    assert store.get(memory.id) is not None


def test_forgetting_is_disabled_by_default(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("trivia", importance=0)
    Consolidator(store).consolidate()
    assert store.get(memory.id) is not None


def test_merged_memory_inherits_the_highest_importance(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations before deploy", importance=1)
    store.remember("deploy: runs migrations before", importance=5)
    Consolidator(store).consolidate(merge_similar=True, similarity=0.6)
    assert store.all()[0].importance == 5


def test_merged_memory_keeps_the_union_of_tags(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations before deploy", tags=["ops"])
    store.remember("deploy: runs migrations before", tags=["ci"])
    Consolidator(store).consolidate(merge_similar=True, similarity=0.6)
    assert store.all()[0].tags == ("ci", "ops")


def test_merging_never_removes_a_semantic_memory(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations before deploy", kind="semantic")
    store.remember("deploy: runs migrations before", kind="episodic")
    Consolidator(store).consolidate(merge_similar=True, similarity=0.6)
    assert store.count() == 1
    assert store.all()[0].kind == "semantic"


def test_consolidate_result_reports_what_it_did(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("promoted fact", importance=4)
    for _ in range(3):
        store.touch(memory.id)
    store.remember("trivia", importance=0)
    result = Consolidator(store).consolidate(promote_after=2, forget_below_importance=1)
    assert result.promoted == 1
    assert len(result.forgotten) == 1
    assert result.as_dict()["promoted"] == 1


def test_consolidate_on_an_empty_store_is_a_no_op(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    result = Consolidator(store).consolidate()
    assert result.as_dict() == {"promoted": 0, "merged": 0, "forgotten": []}


def test_consolidation_reports_memories_it_merged_with(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("deploy runs migrations before deploy")
    store.remember("deploy: runs migrations before")
    result = Consolidator(store).consolidate(merge_similar=True, similarity=0.6)
    assert result.merge_pairs == [(result.kept_ids[0], result.merged_ids[0])]


def test_consolidation_can_be_run_twice_without_double_promoting(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    memory = store.remember("promoted fact")
    for _ in range(3):
        store.touch(memory.id)
    Consolidator(store).consolidate(promote_after=2)
    second = Consolidator(store).consolidate(promote_after=2)
    assert second.promoted == 0
