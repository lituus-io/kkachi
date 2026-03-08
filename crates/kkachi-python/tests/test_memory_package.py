# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Tests for Memory packaging into pip wheels."""

import os
import tempfile
import zipfile

import pytest
from kkachi import Memory


@pytest.fixture
def persistent_memory():
    """Create a persistent Memory with some documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        mem = Memory().persist(db_path)
        mem.add("Q: What is Rust? A: A systems programming language.")
        mem.add("Q: What is Python? A: A dynamic language.")
        yield mem, tmpdir


def test_package_basic(persistent_memory):
    """Wheel file is created with correct name."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    result = mem.package("test_kb", output_dir=dist_dir)
    assert result.wheel_name == "test_kb-0.1.0-py3-none-any.whl"
    assert os.path.exists(result.wheel_path)
    assert result.size_bytes > 0
    assert result.db_size_bytes > 0


def test_package_wheel_is_valid_zip(persistent_memory):
    """Wheel is a valid ZIP file."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    result = mem.package("zip_test", output_dir=dist_dir)

    with zipfile.ZipFile(result.wheel_path, "r") as zf:
        names = zf.namelist()
        assert "zip_test/__init__.py" in names
        assert "zip_test/data/knowledge.db.zst" in names or "zip_test/data/knowledge.db" in names
        assert "zip_test-0.1.0.dist-info/METADATA" in names
        assert "zip_test-0.1.0.dist-info/WHEEL" in names
        assert "zip_test-0.1.0.dist-info/RECORD" in names


def test_package_init_py_has_api(persistent_memory):
    """Generated __init__.py has search/memory/db_path functions."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    result = mem.package("api_test", output_dir=dist_dir)

    with zipfile.ZipFile(result.wheel_path, "r") as zf:
        init_content = zf.read("api_test/__init__.py").decode()
        assert "def db_path()" in init_content
        assert "def memory()" in init_content
        assert "def search(" in init_content


def test_package_db_integrity(persistent_memory):
    """DB in the wheel matches source byte-for-byte."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    db_path = os.path.join(tmpdir, "test.db")

    result = mem.package("integrity_test", output_dir=dist_dir, compress=False)

    # Read the source DB
    with open(db_path, "rb") as f:
        source_bytes = f.read()

    # Read the DB from the wheel (uncompressed mode)
    with zipfile.ZipFile(result.wheel_path, "r") as zf:
        wheel_bytes = zf.read("integrity_test/data/knowledge.db")

    assert source_bytes == wheel_bytes


def test_package_record_checksums(persistent_memory):
    """RECORD file contains sha256 checksums."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    result = mem.package("record_test", output_dir=dist_dir)

    with zipfile.ZipFile(result.wheel_path, "r") as zf:
        record = zf.read("record_test-0.1.0.dist-info/RECORD").decode()
        lines = [l for l in record.strip().split("\n") if l]

        # All lines except RECORD itself should have sha256
        for line in lines:
            if "RECORD" in line.split(",")[0]:
                continue
            parts = line.split(",")
            assert len(parts) == 3
            assert parts[1].startswith("sha256=")
            assert int(parts[2]) > 0


def test_package_empty_memory():
    """Empty persistent memory packages successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "empty.db")
        dist_dir = os.path.join(tmpdir, "dist")
        mem = Memory().persist(db_path)
        result = mem.package("empty_kb", output_dir=dist_dir)
        assert os.path.exists(result.wheel_path)


def test_package_in_memory_raises():
    """Non-persistent Memory raises RuntimeError."""
    mem = Memory()
    with pytest.raises(RuntimeError, match="persist"):
        mem.package("test")


def test_package_with_custom_metadata(persistent_memory):
    """Custom description and author appear in METADATA."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    result = mem.package(
        "meta_test",
        version="2.0.0",
        output_dir=dist_dir,
        description="My custom KB",
        author="Test Author",
    )

    with zipfile.ZipFile(result.wheel_path, "r") as zf:
        metadata = zf.read("meta_test-2.0.0.dist-info/METADATA").decode()
        assert "My custom KB" in metadata
        assert "Test Author" in metadata
        assert "Version: 2.0.0" in metadata


def test_package_repr(persistent_memory):
    """PackageResult has useful repr."""
    mem, tmpdir = persistent_memory
    dist_dir = os.path.join(tmpdir, "dist")
    result = mem.package("repr_test", output_dir=dist_dir)
    r = repr(result)
    assert "PackageResult" in r
    assert "repr_test" in r
