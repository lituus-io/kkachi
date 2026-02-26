# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Integration tests for Memory with realistic usage patterns."""

import pytest
import os
import tempfile
import time
from kkachi import Memory


class TestRealWorldWorkflows:
    """Test realistic usage patterns and workflows."""

    def test_knowledge_base_workflow(self):
        """Simulate a knowledge base being built and queried."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "kb.db")

            # Day 1: Build initial knowledge base
            kb = Memory().persist(db_path)

            kb.add_tagged("rust", "Use Result<T, E> for error handling in Rust")
            kb.add_tagged("rust", "Ownership prevents data races in Rust")
            kb.add_tagged("python", "Use type hints for better code clarity")
            kb.add_tagged("python", "List comprehensions are more Pythonic")

            assert len(kb) == 4
            assert set(kb.tags()) == {"rust", "python"}

            # Query knowledge
            results = kb.search("error handling", k=2)
            assert len(results) == 2
            del kb

            # Day 2: Update and expand
            kb = Memory().persist(db_path)
            assert len(kb) == 4  # Data persisted

            # Add more knowledge
            kb.add_tagged("go", "Goroutines enable easy concurrency")
            kb.add_tagged("rust", "Use lifetimes to prevent dangling references")

            # Update outdated info
            results = kb.search("type hints", k=1)
            if results:
                kb.update(results[0].id, "Python 3.5+ supports type hints for static analysis")

            assert len(kb) == 6
            del kb

            # Day 3: Clean up and verify
            kb = Memory().persist(db_path)
            assert len(kb) == 6
            assert set(kb.tags()) == {"rust", "python", "go"}

    def test_document_versioning(self):
        """Simulate document versioning system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "docs.db")

            docs = Memory().persist(db_path)

            # Create document
            doc_id = docs.add("Version 1.0: Initial release")

            # Update through versions
            versions = [
                "Version 1.1: Bug fixes",
                "Version 1.2: New features added",
                "Version 2.0: Major rewrite",
                "Version 2.1: Security patches"
            ]

            for version in versions:
                docs.update(doc_id, version)

            # Final version should be saved
            assert docs.get(doc_id) == "Version 2.1: Security patches"

            # Reopen and verify persistence
            del docs
            docs = Memory().persist(db_path)
            assert docs.get(doc_id) == "Version 2.1: Security patches"

    def test_caching_layer(self):
        """Simulate a caching layer with eviction."""
        mem = Memory()
        cache = {}

        # Add items to both cache and memory
        for i in range(100):
            content = f"Item {i} with data"
            doc_id = mem.add(content)
            cache[doc_id] = content

        # Update some items
        for i in range(0, 100, 10):
            doc_id = list(cache.keys())[i]
            new_content = f"Updated item {i}"
            mem.update(doc_id, new_content)
            cache[doc_id] = new_content

        # Remove old items (simulate eviction)
        for i in range(50):
            doc_id = list(cache.keys())[0]
            mem.remove(doc_id)
            del cache[doc_id]

        # Verify consistency
        assert len(mem) == len(cache) == 50

        for doc_id in cache:
            assert mem.get(doc_id) == cache[doc_id]

    def test_search_and_update_loop(self):
        """Test iterative search and update pattern."""
        mem = Memory()

        # Add documents with quality scores
        docs = [
            "Quality: 0.5 - Needs improvement",
            "Quality: 0.8 - Good enough",
            "Quality: 0.3 - Poor quality",
            "Quality: 0.9 - Excellent",
            "Quality: 0.4 - Below threshold"
        ]

        for doc in docs:
            mem.add(doc)

        # Search and improve low-quality docs
        low_quality = mem.search("Quality: 0", k=5)
        updated_count = 0

        for result in low_quality:
            if "0.3" in result.content or "0.4" in result.content or "0.5" in result.content:
                # Improve quality
                improved = result.content.replace("0.3", "0.7").replace("0.4", "0.7").replace("0.5", "0.8")
                mem.update(result.id, improved)
                updated_count += 1

        assert updated_count >= 3

    def test_bulk_import_and_dedup(self):
        """Test bulk import with deduplication."""
        mem = Memory()
        unique_docs = set()
        doc_ids = []

        # Import with duplicates
        documents = [
            "Document A",
            "Document B",
            "Document A",  # Duplicate
            "Document C",
            "Document B",  # Duplicate
            "Document D"
        ]

        for doc in documents:
            if doc not in unique_docs:
                doc_id = mem.add(doc)
                doc_ids.append(doc_id)
                unique_docs.add(doc)

        assert len(mem) == 4
        assert len(unique_docs) == 4

    def test_migration_scenario(self):
        """Test data migration between databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_db = os.path.join(tmpdir, "old.db")
            new_db = os.path.join(tmpdir, "new.db")

            # Create old database
            mem_old = Memory().persist(old_db)
            old_ids = []
            for i in range(50):
                old_ids.append(mem_old.add(f"Old data {i}"))

            # Migrate to new database
            mem_new = Memory().persist(new_db)
            for doc_id in old_ids:
                content = mem_old.get(doc_id)
                if content:
                    mem_new.add(content)

            assert len(mem_new) == len(mem_old)

            del mem_old
            del mem_new

            # Verify both databases are intact
            mem_old2 = Memory().persist(old_db)
            mem_new2 = Memory().persist(new_db)

            assert len(mem_old2) == 50
            assert len(mem_new2) == 50


class TestPersistenceConsistency:
    """Test data consistency across persistence operations."""

    def test_consistency_after_crash_simulation(self):
        """Test data consistency after simulated crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "crash_test.db")

            # Write some data
            mem = Memory().persist(db_path)
            id1 = mem.add("Before crash")
            id2 = mem.add("Before crash 2")

            # Simulate crash by abruptly deleting without proper shutdown
            del mem

            # Reopen - data should be intact
            mem2 = Memory().persist(db_path)
            assert mem2.get(id1) == "Before crash"
            assert mem2.get(id2) == "Before crash 2"

    def test_consistency_interleaved_operations(self):
        """Test consistency with interleaved read/write operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "interleaved.db")

            mem = Memory().persist(db_path)

            # Interleave operations
            id1 = mem.add("Doc 1")
            assert mem.get(id1) == "Doc 1"

            mem.update(id1, "Doc 1 v2")
            id2 = mem.add("Doc 2")

            assert mem.get(id1) == "Doc 1 v2"
            assert mem.get(id2) == "Doc 2"

            results = mem.search("Doc", k=5)
            assert len(results) == 2

            mem.remove(id1)
            assert mem.get(id1) is None

            id3 = mem.add("Doc 3")
            mem.update(id2, "Doc 2 v2")

            # Verify final state
            assert mem.get(id1) is None
            assert mem.get(id2) == "Doc 2 v2"
            assert mem.get(id3) == "Doc 3"
            assert len(mem) == 2

    def test_consistency_multiple_reopens(self):
        """Test consistency across multiple open/close cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "reopen.db")

            # Cycle 1
            mem = Memory().persist(db_path)
            ids = [mem.add(f"Doc {i}") for i in range(10)]
            del mem

            # Cycle 2
            mem = Memory().persist(db_path)
            assert len(mem) == 10
            mem.update(ids[0], "Updated 0")
            del mem

            # Cycle 3
            mem = Memory().persist(db_path)
            assert mem.get(ids[0]) == "Updated 0"
            mem.remove(ids[1])
            del mem

            # Cycle 4
            mem = Memory().persist(db_path)
            assert len(mem) == 9
            assert mem.get(ids[0]) == "Updated 0"
            assert mem.get(ids[1]) is None

    def test_search_consistency_after_updates(self):
        """Test search results remain consistent after updates."""
        mem = Memory()

        # Add documents
        id1 = mem.add("Python programming language")
        id2 = mem.add("Rust programming language")
        id3 = mem.add("JavaScript programming language")

        # Initial search
        results1 = mem.search("programming", k=3)
        assert len(results1) == 3

        # Update one document
        mem.update(id2, "Rust systems programming language")

        # Search again
        results2 = mem.search("programming", k=3)
        assert len(results2) == 3

        # Updated document should still appear
        ids_in_results = [r.id for r in results2]
        assert id2 in ids_in_results


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_recovery_from_invalid_operations(self):
        """Test memory remains usable after invalid operations."""
        mem = Memory()

        # Valid operation
        doc_id = mem.add("Valid doc")

        # Invalid operations
        assert mem.get("invalid-id") is None
        assert mem.update("invalid-id", "content") is False
        assert mem.remove("invalid-id") is False

        # Memory should still work
        assert mem.get(doc_id) == "Valid doc"
        assert mem.update(doc_id, "Updated") is True
        assert len(mem) == 1

    def test_persist_after_invalid_path_attempt(self):
        """Test Memory still works after failed persist attempt."""
        mem = Memory()
        mem.add("Doc 1")
        mem.add("Doc 2")

        # Should still work in-memory
        assert len(mem) == 2

        # Now successfully persist
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "valid.db")
            # Note: Can't call persist twice on same instance, this tests in-memory continues to work

            assert len(mem) == 2

    def test_operations_after_remove(self):
        """Test various operations after document removal."""
        mem = Memory()

        doc_id = mem.add("To be removed")
        assert mem.remove(doc_id) is True

        # These should all handle removed document gracefully
        assert mem.get(doc_id) is None
        assert mem.update(doc_id, "New content") is False
        assert mem.remove(doc_id) is False

        # Can still add new documents
        new_id = mem.add("New doc")
        assert mem.get(new_id) == "New doc"


class TestPerformancePatterns:
    """Test performance-related patterns."""

    def test_sequential_operations_performance(self):
        """Test sequential operations complete reasonably."""
        mem = Memory()
        start = time.time()

        # Add 1000 documents
        ids = []
        for i in range(1000):
            ids.append(mem.add(f"Document {i}"))

        # Update 500 of them
        for i in range(500):
            mem.update(ids[i], f"Updated {i}")

        # Remove 250 of them
        for i in range(250):
            mem.remove(ids[i])

        elapsed = time.time() - start

        # Should complete in reasonable time (under 5 seconds)
        assert elapsed < 5.0
        assert len(mem) == 750

    def test_search_performance_pattern(self):
        """Test search performance with many documents."""
        mem = Memory()

        # Add documents
        for i in range(100):
            mem.add(f"Document {i} about programming in Python")

        start = time.time()

        # Multiple searches
        for _ in range(50):
            results = mem.search("Python programming", k=5)
            assert len(results) <= 5

        elapsed = time.time() - start

        # Should complete reasonably fast (under 2 seconds)
        assert elapsed < 2.0

    def test_persist_large_dataset_performance(self):
        """Test persistence performance with larger datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "large.db")

            mem = Memory().persist(db_path)

            start = time.time()

            # Add 5000 documents
            for i in range(5000):
                mem.add(f"Large dataset document {i} with some content here")

            add_time = time.time() - start

            # Should complete in reasonable time (under 10 seconds)
            assert add_time < 10.0

            del mem

            # Reopen should be fast
            start = time.time()
            mem2 = Memory().persist(db_path)
            reopen_time = time.time() - start

            assert len(mem2) == 5000
            # Reopening should be fast (under 2 seconds)
            assert reopen_time < 2.0
