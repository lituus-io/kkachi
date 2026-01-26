# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Edge case and robustness tests for Memory persist() and update()."""

import pytest
import os
import tempfile
import string
import random
from kkachi import Memory


class TestPersistEdgeCases:
    """Edge cases for persist() method."""

    def test_persist_empty_database(self):
        """Test persist with empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.db")

            # Create empty database
            mem = Memory().persist(db_path)
            assert len(mem) == 0
            assert mem.is_empty()

            # Reopen and verify still empty
            del mem
            mem2 = Memory().persist(db_path)
            assert len(mem2) == 0

    def test_persist_same_path_twice(self):
        """Test calling persist with same path twice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create and add data
            mem1 = Memory().persist(db_path)
            mem1.add("Document 1")
            del mem1

            # Open again
            mem2 = Memory().persist(db_path)
            mem2.add("Document 2")

            # Should have both documents
            assert len(mem2) == 2

    def test_persist_unicode_path(self):
        """Test persist with Unicode characters in path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "测试_数据库.db")

            mem = Memory().persist(db_path)
            doc_id = mem.add("Unicode content")

            assert mem.get(doc_id) == "Unicode content"

    def test_persist_relative_path(self):
        """Test persist with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mem = Memory().persist("./relative_db.db")
                mem.add("Relative path test")
                assert len(mem) == 1
            finally:
                os.chdir(old_cwd)

    def test_persist_nested_directory(self):
        """Test persist creates nested directories if parent exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested")
            os.makedirs(nested_dir)
            db_path = os.path.join(nested_dir, "test.db")

            mem = Memory().persist(db_path)
            mem.add("Nested test")
            assert len(mem) == 1

    def test_persist_large_database(self):
        """Test persist with many documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "large.db")

            mem = Memory().persist(db_path)

            # Add 1000 documents
            ids = []
            for i in range(1000):
                doc_id = mem.add(f"Document {i} with some content")
                ids.append(doc_id)

            assert len(mem) == 1000

            # Reopen and verify
            del mem
            mem2 = Memory().persist(db_path)
            assert len(mem2) == 1000

            # Sample check a few documents
            assert mem2.get(ids[0]) == "Document 0 with some content"
            assert mem2.get(ids[500]) == "Document 500 with some content"
            assert mem2.get(ids[999]) == "Document 999 with some content"


class TestUpdateEdgeCases:
    """Edge cases for update() method."""

    def test_update_empty_content(self):
        """Test update with empty string."""
        mem = Memory()
        doc_id = mem.add("Original")

        assert mem.update(doc_id, "") is True
        assert mem.get(doc_id) == ""

    def test_update_same_content(self):
        """Test update with identical content."""
        mem = Memory()
        doc_id = mem.add("Content")

        assert mem.update(doc_id, "Content") is True
        assert mem.get(doc_id) == "Content"

    def test_update_very_long_content(self):
        """Test update with very long content."""
        mem = Memory()
        doc_id = mem.add("Short")

        # 1MB of content
        long_content = "x" * (1024 * 1024)
        assert mem.update(doc_id, long_content) is True
        assert len(mem.get(doc_id)) == 1024 * 1024

    def test_update_unicode_content(self):
        """Test update with Unicode content."""
        mem = Memory()
        doc_id = mem.add("ASCII only")

        unicode_content = "Hello 世界 🌍 مرحبا мир"
        assert mem.update(doc_id, unicode_content) is True
        assert mem.get(doc_id) == unicode_content

    def test_update_multiline_content(self):
        """Test update with multiline content."""
        mem = Memory()
        doc_id = mem.add("Single line")

        multiline = "Line 1\nLine 2\nLine 3\n\nLine 5"
        assert mem.update(doc_id, multiline) is True
        assert mem.get(doc_id) == multiline

    def test_update_special_characters(self):
        """Test update with special characters."""
        mem = Memory()
        doc_id = mem.add("Normal")

        special = "Quotes: \"'`\nTabs:\t\t\nBackslash: \\ Null byte: \x00"
        assert mem.update(doc_id, special) is True
        assert mem.get(doc_id) == special

    def test_update_persisted_document(self):
        """Test update persists correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            mem = Memory().persist(db_path)
            doc_id = mem.add("Before")
            mem.update(doc_id, "After")

            # Reopen and verify update persisted
            del mem
            mem2 = Memory().persist(db_path)
            assert mem2.get(doc_id) == "After"

    def test_update_multiple_times(self):
        """Test multiple updates to same document."""
        mem = Memory()
        doc_id = mem.add("Version 1")

        assert mem.update(doc_id, "Version 2") is True
        assert mem.update(doc_id, "Version 3") is True
        assert mem.update(doc_id, "Version 4") is True

        assert mem.get(doc_id) == "Version 4"

    def test_update_after_remove_fails(self):
        """Test update fails after document is removed."""
        mem = Memory()
        doc_id = mem.add("Content")

        assert mem.remove(doc_id) is True
        assert mem.update(doc_id, "New") is False

    def test_update_with_custom_id(self):
        """Test update works with custom IDs."""
        mem = Memory()
        mem.add_with_id("custom-123", "Original")

        assert mem.update("custom-123", "Updated") is True
        assert mem.get("custom-123") == "Updated"

    def test_update_tagged_document(self):
        """Test update preserves tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            mem = Memory().persist(db_path)
            doc_id = mem.add_tagged("category", "Content")

            # Update content
            mem.update(doc_id, "New content")

            # Tag should still be present
            assert "category" in mem.tags()


class TestCRUDIntegration:
    """Integration tests for complete CRUD workflows."""

    def test_crud_concurrent_operations(self):
        """Test CRUD operations in various orders."""
        mem = Memory()

        # Create multiple
        id1 = mem.add("Doc 1")
        id2 = mem.add("Doc 2")
        id3 = mem.add("Doc 3")

        # Update some
        mem.update(id1, "Doc 1 Updated")
        mem.update(id3, "Doc 3 Updated")

        # Remove one
        mem.remove(id2)

        # Add more
        id4 = mem.add("Doc 4")

        # Verify state
        assert mem.get(id1) == "Doc 1 Updated"
        assert mem.get(id2) is None
        assert mem.get(id3) == "Doc 3 Updated"
        assert mem.get(id4) == "Doc 4"
        assert len(mem) == 3

    def test_crud_with_search(self):
        """Test CRUD operations don't break search."""
        mem = Memory()

        mem.add("Rust programming")
        id2 = mem.add("Python programming")
        mem.add("Go programming")

        # Update one
        mem.update(id2, "Python scripting language")

        # Search should still work
        results = mem.search("programming", k=5)
        assert len(results) >= 2

        # Updated doc should still be searchable
        results = mem.search("Python", k=5)
        assert any("scripting" in r.content for r in results)

    def test_crud_persist_full_cycle(self):
        """Test complete CRUD cycle with persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Cycle 1: Create and update
            mem = Memory().persist(db_path)
            id1 = mem.add("Version 1")
            id2 = mem.add("To be deleted")
            mem.update(id1, "Version 2")
            mem.remove(id2)
            id3 = mem.add("Version 1 of doc 3")
            del mem

            # Cycle 2: Reopen, verify, and continue
            mem2 = Memory().persist(db_path)
            assert mem2.get(id1) == "Version 2"
            assert mem2.get(id2) is None
            assert mem2.get(id3) == "Version 1 of doc 3"

            mem2.update(id3, "Version 2 of doc 3")
            id4 = mem2.add("New doc")
            del mem2

            # Cycle 3: Final verification
            mem3 = Memory().persist(db_path)
            assert len(mem3) == 3
            assert mem3.get(id1) == "Version 2"
            assert mem3.get(id3) == "Version 2 of doc 3"
            assert mem3.get(id4) == "New doc"

    def test_memory_operations_stress(self):
        """Stress test with many rapid operations."""
        mem = Memory()
        ids = []

        # Rapid creates
        for i in range(100):
            ids.append(mem.add(f"Document {i}"))

        # Rapid updates
        for i, doc_id in enumerate(ids):
            mem.update(doc_id, f"Updated {i}")

        # Rapid deletes (every other)
        for i in range(0, 100, 2):
            mem.remove(ids[i])

        # Verify remaining
        assert len(mem) == 50
        for i in range(1, 100, 2):
            assert mem.get(ids[i]) == f"Updated {i}"


class TestMemoryFuzz:
    """Fuzz testing for Memory operations."""

    def random_string(self, length=None):
        """Generate random string."""
        if length is None:
            length = random.randint(0, 1000)
        chars = string.printable + "世界🌍مرحبا"
        return ''.join(random.choice(chars) for _ in range(length))

    def test_fuzz_add_and_get(self):
        """Fuzz test add and get with random content."""
        mem = Memory()
        pairs = []

        for _ in range(100):
            content = self.random_string()
            doc_id = mem.add(content)
            pairs.append((doc_id, content))

        # Verify all
        for doc_id, content in pairs:
            assert mem.get(doc_id) == content

    def test_fuzz_update(self):
        """Fuzz test update with random content."""
        mem = Memory()

        doc_id = mem.add("Initial")

        for _ in range(50):
            new_content = self.random_string()
            assert mem.update(doc_id, new_content) is True
            assert mem.get(doc_id) == new_content

    def test_fuzz_update_invalid_ids(self):
        """Fuzz test update with random invalid IDs."""
        mem = Memory()
        mem.add("Valid doc")

        for _ in range(50):
            invalid_id = self.random_string(20)
            result = mem.update(invalid_id, "Content")
            # Should either return False or not crash
            assert result is False

    def test_fuzz_persist_random_paths(self):
        """Fuzz test persist with various path patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_paths = [
                os.path.join(tmpdir, f"db{i}.db")
                for i in range(10)
            ]

            for path in valid_paths:
                try:
                    mem = Memory().persist(path)
                    mem.add("Test")
                    assert len(mem) == 1
                except Exception as e:
                    # If it fails, should be a RuntimeError
                    assert isinstance(e, RuntimeError)

    def test_fuzz_mixed_operations(self):
        """Fuzz test with random mix of CRUD operations."""
        mem = Memory()
        doc_ids = []

        for _ in range(200):
            op = random.choice(['add', 'update', 'remove', 'get'])

            if op == 'add' or not doc_ids:
                content = self.random_string(100)
                doc_id = mem.add(content)
                doc_ids.append(doc_id)

            elif op == 'update' and doc_ids:
                doc_id = random.choice(doc_ids)
                content = self.random_string(100)
                mem.update(doc_id, content)

            elif op == 'remove' and doc_ids:
                doc_id = random.choice(doc_ids)
                mem.remove(doc_id)
                doc_ids.remove(doc_id)

            elif op == 'get' and doc_ids:
                doc_id = random.choice(doc_ids)
                mem.get(doc_id)

        # Should still be in valid state
        assert len(mem) == len(doc_ids)


class TestMemoryBoundaries:
    """Boundary condition tests."""

    def test_empty_string_content(self):
        """Test operations with empty strings."""
        mem = Memory()

        # Add empty
        doc_id = mem.add("")
        assert mem.get(doc_id) == ""

        # Update to empty
        doc_id2 = mem.add("Non-empty")
        mem.update(doc_id2, "")
        assert mem.get(doc_id2) == ""

    def test_whitespace_only_content(self):
        """Test with whitespace-only content."""
        mem = Memory()

        doc_id = mem.add("   \n\t\r\n   ")
        assert mem.get(doc_id) == "   \n\t\r\n   "

    def test_single_character_content(self):
        """Test with single character content."""
        mem = Memory()

        for char in "aA1!@\n\t 世🌍":
            doc_id = mem.add(char)
            assert mem.get(doc_id) == char

    def test_null_byte_in_content(self):
        """Test with null bytes in content."""
        mem = Memory()

        content = "Before\x00After"
        doc_id = mem.add(content)
        assert mem.get(doc_id) == content

    def test_maximum_length_id(self):
        """Test with very long custom IDs."""
        mem = Memory()

        long_id = "id" * 1000
        mem.add_with_id(long_id, "Content")
        assert mem.get(long_id) == "Content"
