"""Tokenisation and recall coverage for non-ASCII memory text.

Regression coverage for the lexical tokeniser: accented Latin must survive
tokenisation intact (not be chopped into meaningless fragments), while ASCII
behaviour -- including underscore as a separator -- must not change.
"""

from agent_memory import MemoryStore, Recaller
from agent_memory.recall import tokenize


def test_tokenize_keeps_accented_latin_words_whole():
    assert tokenize("café em Mossoró") == ["café", "em", "mossoró"]
    assert tokenize("ação e reação") == ["ação", "e", "reação"]
    assert tokenize("mañana es temprano") == ["mañana", "es", "temprano"]


def test_tokenize_does_not_fragment_an_accented_word_into_stems():
    """The pre-fix [a-z0-9]+ class turned 'café' into ['caf'] and 'ação' into
    ['a', 'o'], which made the recall score depend on stray ASCII letters."""
    assert "café" in tokenize("café")
    assert tokenize("ação") == ["ação"]


def test_tokenize_keeps_cjk_as_word_characters():
    assert tokenize("日本の首都は東京です") == ["日本の首都は東京です"]


def test_tokenize_treats_underscore_as_a_separator_not_a_word_character():
    assert tokenize("snake_case") == ["snake", "case"]
    assert tokenize("the snake_case helper") == ["the", "snake", "case", "helper"]


def test_tokenize_ascii_contract_is_unchanged():
    assert tokenize("The Deploy-Script is ./deploy.sh") == [
        "the",
        "deploy",
        "script",
        "is",
        "deploy",
        "sh",
    ]
    assert tokenize("   ---   ") == []
    assert tokenize("v1.2.3-rc1") == ["v1", "2", "3", "rc1"]


def test_recall_finds_an_accented_memory_and_reports_the_whole_term(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    try:
        store.remember("café em Mossoró")
        store.remember("ação e reação")

        assert Recaller(store, reinforce=False).recall("café")[0].as_dict()["content"] == (
            "café em Mossoró"
        )
        assert Recaller(store, reinforce=False).recall("Mossoró")[0].as_dict()["content"] == (
            "café em Mossoró"
        )

        for query, expected in (("café", "café em Mossoró"), ("ação", "ação e reação")):
            hits = Recaller(store, reinforce=False).recall(query)
            assert hits, f"query {query!r} found nothing"
            assert hits[0].as_dict()["content"] == expected
            # The matched term must be the accented word itself, not a stem.
            assert hits[0].matched_terms == (query,)
    finally:
        store.close()


def test_recall_still_matches_ascii_words_around_an_underscore(tmp_path):
    """Guards the ASCII path: '_' must keep separating tokens, so a query for
    one half of snake_case still finds the memory."""
    store = MemoryStore(tmp_path / "m.db")
    try:
        store.remember("the snake_case helper is here")
        assert Recaller(store, reinforce=False).recall("snake")
        assert Recaller(store, reinforce=False).recall("case")
        assert Recaller(store, reinforce=False).recall("snake_case")
    finally:
        store.close()


def test_recall_remains_case_insensitive_for_ascii(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    try:
        store.remember("the Deploy Script runs")
        assert Recaller(store, reinforce=False).recall("DEPLOY")
        assert Recaller(store, reinforce=False).recall("deploy")
    finally:
        store.close()


def test_recall_does_not_return_an_unrelated_memory_for_an_accented_query(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    try:
        store.remember("the deploy script runs at midnight")
        assert Recaller(store, reinforce=False).recall("ação") == []
    finally:
        store.close()


