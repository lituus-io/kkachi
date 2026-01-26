# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Tests for Memory persistent storage."""

import pytest
import os
import tempfile
from kkachi import Memory


def test_memory_persist_basic():
    """Test basic persistent storage functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # Create and populate
        mem = Memory().persist(db_path)
        doc1_id = mem.add("Document 1")
        doc2_id = mem.add("Document 2")

        assert len(mem) == 2

        # Close and reopen
        del mem

        mem2 = Memory().persist(db_path)
        assert len(mem2) == 2

        # Data should persist
        assert mem2.get(doc1_id) == "Document 1"
        assert mem2.get(doc2_id) == "Document 2"


def test_memory_persist_search():
    """Test search works with persistent storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        mem = Memory().persist(db_path)
        mem.add("Rust programming language")
        mem.add("Python scripting")
        mem.add("JavaScript for web")

        results = mem.search("programming", k=2)
        assert len(results) == 2
        assert results[0].score > 0


def test_memory_persist_error_invalid_path():
    """Test error handling for invalid database path."""
    with pytest.raises(RuntimeError, match="Failed to enable persistent storage"):
        Memory().persist("/invalid/path/that/does/not/exist/db.db")


def test_memory_update():
    """Test update operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        mem = Memory().persist(db_path)
        doc_id = mem.add("Original content")

        # Update should succeed
        assert mem.update(doc_id, "Updated content") is True
        assert mem.get(doc_id) == "Updated content"

        # Update non-existent doc should fail
        assert mem.update("nonexistent", "content") is False


def test_memory_crud_complete():
    """Test complete CRUD operations work together."""
    mem = Memory()

    # Create
    doc_id = mem.add("Initial")
    assert len(mem) == 1

    # Read
    assert mem.get(doc_id) == "Initial"

    # Update
    assert mem.update(doc_id, "Modified") is True
    assert mem.get(doc_id) == "Modified"

    # Delete
    assert mem.remove(doc_id) is True
    assert mem.get(doc_id) is None
    assert len(mem) == 0
