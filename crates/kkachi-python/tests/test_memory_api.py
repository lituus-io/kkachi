"""Tests for the Memory Python API — upsert, search_diverse, learn, error propagation."""

from kkachi import Memory


class TestUpsert:
    def test_upsert_returns_id(self):
        mem = Memory()
        doc_id = mem.upsert("hello world")
        assert isinstance(doc_id, str)
        assert doc_id.startswith("upsert:")

    def test_upsert_idempotent(self):
        mem = Memory()
        id1 = mem.upsert("same content")
        id2 = mem.upsert("same content")
        assert id1 == id2
        assert len(mem) == 1

    def test_upsert_different_content(self):
        mem = Memory()
        id1 = mem.upsert("content A")
        id2 = mem.upsert("content B")
        assert id1 != id2
        assert len(mem) == 2

    def test_upsert_tagged(self):
        mem = Memory()
        doc_id = mem.upsert_tagged("mytag", "tagged content")
        assert isinstance(doc_id, str)
        assert "mytag" in mem.tags()


class TestSearchDiverse:
    def test_search_diverse_returns_results(self):
        mem = Memory()
        mem.add("Rust systems programming")
        mem.add("Rust memory safety")
        mem.add("Python data science")
        results = mem.search_diverse("programming", 2, 0.5)
        assert len(results) == 2

    def test_search_diverse_default_lambda(self):
        mem = Memory()
        mem.add("doc one")
        mem.add("doc two")
        results = mem.search_diverse("doc", 2)
        assert len(results) == 2


class TestSearchAbove:
    def test_search_above_filters(self):
        mem = Memory()
        mem.add("exact match query text")
        mem.add("completely unrelated xyz")
        results = mem.search_above("exact match query text", 5, 0.0)
        assert len(results) >= 1


class TestLearn:
    def test_learn_below_threshold(self):
        mem = Memory()
        mem = mem.learn_above(0.8)
        mem.learn("question", "bad answer", 0.5)
        assert len(mem) == 0

    def test_learn_above_threshold(self):
        mem = Memory()
        mem = mem.learn_above(0.8)
        mem.learn("question", "good answer", 0.9)
        assert len(mem) == 1


class TestDiversity:
    def test_diversity_fluent(self):
        mem = Memory()
        mem = mem.diversity(0.5)
        mem.add("doc A")
        mem.add("doc B")
        results = mem.search("doc", 2)
        assert len(results) == 2


class TestErrorPropagation:
    def test_add_returns_string(self):
        mem = Memory()
        doc_id = mem.add("test")
        assert isinstance(doc_id, str)

    def test_search_returns_list(self):
        mem = Memory()
        mem.add("test content")
        results = mem.search("test", 1)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_remove_returns_bool(self):
        mem = Memory()
        doc_id = mem.add("to remove")
        assert mem.remove(doc_id) is True
        assert mem.remove("nonexistent") is False

    def test_update_returns_bool(self):
        mem = Memory()
        doc_id = mem.add("old")
        assert mem.update(doc_id, "new") is True
        assert mem.get(doc_id) == "new"

    def test_tags_returns_list(self):
        mem = Memory()
        mem.add_tagged("rust", "content")
        tags = mem.tags()
        assert isinstance(tags, list)
        assert "rust" in tags

    def test_list_returns_recalls(self):
        mem = Memory()
        mem.add("doc1")
        mem.add("doc2")
        all_docs = mem.list()
        assert len(all_docs) == 2

    def test_is_empty(self):
        mem = Memory()
        assert mem.is_empty() is True
        mem.add("x")
        assert mem.is_empty() is False
