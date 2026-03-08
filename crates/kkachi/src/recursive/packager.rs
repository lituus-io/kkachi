// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! DuckDB RAG Auto-Packaging — build pip-installable wheels from a Memory store.
//!
//! Takes a persistent DuckDB knowledge base and packages it into a PEP 427 wheel
//! that can be `pip install`'d.
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::packager::PackagerBuilder;
//! use std::path::Path;
//!
//! let result = PackagerBuilder::new(Path::new("./kb.db"))
//!     .name("my_kb")
//!     .version("0.1.0")
//!     .description("My knowledge base")
//!     .author("Team")
//!     .output_dir("./dist")
//!     .build()?;
//!
//! println!("pip install {}", result.wheel_path.display());
//! ```

use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Sanitize a package name per PEP 503/427: strip path components,
/// replace hyphens/spaces/dots with underscores, remove non-alphanumeric chars.
fn sanitize_package_name(name: &str) -> Result<String> {
    // Strip any path traversal components
    let base = Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(name);

    // Replace hyphens, spaces, dots with underscores; keep only alphanumeric + underscore
    let sanitized: String = base
        .chars()
        .map(|c| match c {
            '-' | ' ' | '.' => '_',
            c if c.is_alphanumeric() || c == '_' => c,
            _ => '_',
        })
        .collect();

    // Strip leading/trailing underscores and collapse runs
    let collapsed: String = sanitized
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_");

    if collapsed.is_empty() {
        return Err(Error::storage(
            "Package name is empty or contains only invalid characters. \
             Provide a valid Python package name (e.g., 'my_knowledge_base').",
        ));
    }

    Ok(collapsed)
}

/// Metadata for the wheel package.
pub struct PackageMeta<'a> {
    /// Package name.
    pub name: Cow<'a, str>,
    /// Package version.
    pub version: Cow<'a, str>,
    /// Package description.
    pub description: Cow<'a, str>,
    /// Package author.
    pub author: Cow<'a, str>,
}

/// Builder for creating wheel packages from a DuckDB knowledge base.
pub struct PackagerBuilder<'a> {
    db_path: Cow<'a, Path>,
    meta: PackageMeta<'a>,
    output_dir: Cow<'a, Path>,
    compress: bool,
}

/// Result of a successful wheel build.
#[derive(Debug, Clone)]
pub struct PackageResult {
    /// Full path to the generated .whl file.
    pub wheel_path: PathBuf,
    /// Wheel filename (e.g., "my_kb-0.1.0-py3-none-any.whl").
    pub wheel_name: String,
    /// Total size of the wheel file in bytes.
    pub size_bytes: u64,
    /// Size of the embedded .db file in bytes.
    pub db_size_bytes: u64,
    /// Number of files in the wheel.
    pub file_count: usize,
    /// Whether the DB was zstd-compressed.
    pub compressed: bool,
    /// Compression ratio (compressed/original). 1.0 if not compressed.
    pub compression_ratio: f64,
}

impl<'a> PackagerBuilder<'a> {
    /// Create a new packager from a DuckDB file path.
    pub fn new(db_path: &'a Path) -> Self {
        Self {
            db_path: Cow::Borrowed(db_path),
            meta: PackageMeta {
                name: Cow::Borrowed("knowledge_base"),
                version: Cow::Borrowed("0.1.0"),
                description: Cow::Borrowed("Kkachi knowledge base"),
                author: Cow::Borrowed(""),
            },
            output_dir: Cow::Borrowed(Path::new(".")),
            compress: true,
        }
    }

    /// Set the package name (`&str` borrows, `String` moves).
    pub fn name(mut self, name: impl Into<Cow<'a, str>>) -> Self {
        self.meta.name = name.into();
        self
    }

    /// Set the package name from an owned string.
    pub fn name_owned(mut self, name: String) -> Self {
        self.meta.name = Cow::Owned(name);
        self
    }

    /// Set the package version (`&str` borrows, `String` moves).
    pub fn version(mut self, version: impl Into<Cow<'a, str>>) -> Self {
        self.meta.version = version.into();
        self
    }

    /// Set the package version from an owned string.
    pub fn version_owned(mut self, version: String) -> Self {
        self.meta.version = Cow::Owned(version);
        self
    }

    /// Set the package description (`&str` borrows, `String` moves).
    pub fn description(mut self, description: impl Into<Cow<'a, str>>) -> Self {
        self.meta.description = description.into();
        self
    }

    /// Set the package description from an owned string.
    pub fn description_owned(mut self, description: String) -> Self {
        self.meta.description = Cow::Owned(description);
        self
    }

    /// Set the package author (`&str` borrows, `String` moves).
    pub fn author(mut self, author: impl Into<Cow<'a, str>>) -> Self {
        self.meta.author = author.into();
        self
    }

    /// Set the package author from an owned string.
    pub fn author_owned(mut self, author: String) -> Self {
        self.meta.author = Cow::Owned(author);
        self
    }

    /// Set the output directory for the wheel file.
    pub fn output_dir(mut self, dir: &'a str) -> Self {
        self.output_dir = Cow::Borrowed(Path::new(dir));
        self
    }

    /// Set the output directory from an owned path.
    pub fn output_dir_owned(mut self, dir: PathBuf) -> Self {
        self.output_dir = Cow::Owned(dir);
        self
    }

    /// Enable or disable zstd compression of the DB file (default: true).
    pub fn compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    /// Build the wheel package.
    pub fn build(&self) -> Result<PackageResult> {
        // Validate the db file exists
        if !self.db_path.exists() {
            return Err(Error::storage(format!(
                "Cannot package: database file not found at '{}'. \
                 Call .persist(path) on Memory first, then .package(). \
                 The database must exist on disk.",
                self.db_path.display()
            )));
        }

        let db_bytes = std::fs::read(&*self.db_path).map_err(|e| {
            Error::storage(format!(
                "Cannot read database at '{}': {}. \
                 Check file permissions and ensure the file is not locked by another process.",
                self.db_path.display(),
                e
            ))
        })?;

        let db_size_bytes = db_bytes.len() as u64;

        // Sanitize and normalize the package name (PEP 503/427)
        let normalized_name = sanitize_package_name(&self.meta.name)?;

        // Generate wheel filename (PEP 427)
        let wheel_name = format!("{}-{}-py3-none-any.whl", normalized_name, self.meta.version);

        // Ensure output directory exists
        std::fs::create_dir_all(&*self.output_dir).map_err(|e| {
            Error::storage(format!(
                "Cannot create output directory '{}': {}. \
                 Ensure the parent directory exists and is writable.",
                self.output_dir.display(),
                e
            ))
        })?;

        let wheel_path = self.output_dir.join(&wheel_name);

        // Build the wheel ZIP
        let file = std::fs::File::create(&wheel_path).map_err(|e| {
            Error::storage(format!(
                "Cannot create wheel file '{}': {}. \
                 Check disk space and write permissions on the output directory.",
                wheel_path.display(),
                e
            ))
        })?;

        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        // Track files and their hashes for RECORD
        let mut records: Vec<(String, String, usize)> = Vec::new();

        // Optionally compress the DB with zstd
        let (db_data, db_filename, compressed, compression_ratio) = if self.compress {
            let compressed_bytes = zstd::encode_all(&db_bytes[..], 3).map_err(|e| {
                Error::storage(format!(
                    "Failed to zstd-compress database: {}. \
                     Try .compress(false) to package without compression.",
                    e
                ))
            })?;
            let ratio = compressed_bytes.len() as f64 / db_bytes.len().max(1) as f64;
            (compressed_bytes, "knowledge.db.zst", true, ratio)
        } else {
            let ratio = 1.0;
            (db_bytes, "knowledge.db", false, ratio)
        };

        // 1. __init__.py
        let init_py = if compressed {
            generate_init_py_compressed(&normalized_name)
        } else {
            generate_init_py(&normalized_name)
        };
        let init_path = format!("{}/{}", normalized_name, "__init__.py");
        write_zip_entry(
            &mut zip,
            &init_path,
            init_py.as_bytes(),
            options,
            &mut records,
        )?;

        // 2. data/knowledge.db or data/knowledge.db.zst
        let db_path_in_zip = format!("{}/data/{}", normalized_name, db_filename);
        write_zip_entry(&mut zip, &db_path_in_zip, &db_data, options, &mut records)?;

        // dist-info directory
        let dist_info = format!("{}-{}.dist-info", normalized_name, self.meta.version);

        // 3. METADATA
        let metadata = generate_metadata(&self.meta, &normalized_name, compressed);
        let metadata_path = format!("{}/METADATA", dist_info);
        write_zip_entry(
            &mut zip,
            &metadata_path,
            metadata.as_bytes(),
            options,
            &mut records,
        )?;

        // 4. WHEEL
        let wheel_meta = generate_wheel_metadata();
        let wheel_meta_path = format!("{}/WHEEL", dist_info);
        write_zip_entry(
            &mut zip,
            &wheel_meta_path,
            wheel_meta.as_bytes(),
            options,
            &mut records,
        )?;

        // 5. RECORD (must be last, contains hashes of everything else)
        let record_path = format!("{}/RECORD", dist_info);
        let mut record_content = String::new();
        for (path, hash, size) in &records {
            record_content.push_str(&format!("{},sha256={},{}\n", path, hash, size));
        }
        // RECORD itself has no hash
        record_content.push_str(&format!("{},,\n", record_path));

        zip.start_file(&record_path, options).map_err(|e| {
            Error::storage(format!(
                "Wheel packaging failed during RECORD generation: {}. \
                 This may indicate a corrupted ZIP — try deleting the output file and retrying.",
                e
            ))
        })?;
        zip.write_all(record_content.as_bytes()).map_err(|e| {
            Error::storage(format!(
                "Wheel packaging failed during RECORD generation: {}. \
                 This may indicate a corrupted ZIP — try deleting the output file and retrying.",
                e
            ))
        })?;

        let file_count = records.len() + 1; // +1 for RECORD itself

        zip.finish().map_err(|e| {
            Error::storage(format!(
                "Wheel packaging failed during finalization: {}. \
                 The output file may be incomplete — delete it and retry.",
                e
            ))
        })?;

        let size_bytes = std::fs::metadata(&wheel_path).map(|m| m.len()).unwrap_or(0);

        Ok(PackageResult {
            wheel_path,
            wheel_name,
            size_bytes,
            db_size_bytes,
            file_count,
            compressed,
            compression_ratio,
        })
    }
}

/// Write an entry to the ZIP and record its hash.
fn write_zip_entry(
    zip: &mut zip::ZipWriter<std::fs::File>,
    path: &str,
    data: &[u8],
    options: zip::write::SimpleFileOptions,
    records: &mut Vec<(String, String, usize)>,
) -> Result<()> {
    zip.start_file(path, options)
        .map_err(|e| Error::storage(format!("Failed to start zip entry '{}': {}", path, e)))?;
    zip.write_all(data)
        .map_err(|e| Error::storage(format!("Failed to write zip entry '{}': {}", path, e)))?;

    // SHA256 hash (base64url-encoded, no padding, per PEP 376)
    let mut hasher = Sha256::new();
    hasher.update(data);
    let hash = base64url_nopad(&hasher.finalize());

    records.push((path.to_string(), hash, data.len()));
    Ok(())
}

/// Base64url encoding without padding (PEP 376 RECORD format).
fn base64url_nopad(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut result = String::new();
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() {
            data[i + 1] as u32
        } else {
            0
        };
        let b2 = if i + 2 < data.len() {
            data[i + 2] as u32
        } else {
            0
        };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if i + 1 < data.len() {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        }
        if i + 2 < data.len() {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        }

        i += 3;
    }
    result
}

/// Generate __init__.py with lazy zstd decompression.
fn generate_init_py_compressed(package_name: &str) -> String {
    format!(
        r#""""{package_name} — auto-packaged kkachi knowledge base (zstd-compressed)."""

import os as _os
from pathlib import Path as _Path

_DATA_DIR = _Path(__file__).parent / "data"
_COMPRESSED = _DATA_DIR / "knowledge.db.zst"
_DECOMPRESSED = _DATA_DIR / "knowledge.db"

def _ensure_decompressed():
    """Decompress the DB on first access (atomic via tmp + rename)."""
    if _DECOMPRESSED.exists():
        return
    import zstandard
    tmp = _DECOMPRESSED.with_suffix(".db.tmp")
    with open(_COMPRESSED, "rb") as src:
        dctx = zstandard.ZstdDecompressor()
        with open(tmp, "wb") as dst:
            dctx.copy_stream(src, dst)
    tmp.rename(_DECOMPRESSED)

def db_path() -> str:
    """Return the absolute path to the embedded knowledge.db."""
    _ensure_decompressed()
    return str(_DECOMPRESSED)

def memory():
    """Create a kkachi Memory backed by this package's DB.

    Returns:
        kkachi.Memory: Persistent memory store.

    Raises:
        ImportError: If kkachi is not installed.
    """
    from kkachi import Memory
    return Memory().persist(db_path())

def search(query: str, k: int = 3):
    """Search the knowledge base.

    Args:
        query: Search query.
        k: Number of results (default 3).

    Returns:
        list[kkachi.Recall]: Search results.
    """
    mem = memory()
    return mem.search(query, k)
"#,
        package_name = package_name
    )
}

/// Generate the __init__.py with helper functions.
fn generate_init_py(package_name: &str) -> String {
    format!(
        r#""""{package_name} — auto-packaged kkachi knowledge base."""

import os as _os
from pathlib import Path as _Path

_DATA_DIR = _Path(__file__).parent / "data"

def db_path() -> str:
    """Return the absolute path to the embedded knowledge.db."""
    return str(_DATA_DIR / "knowledge.db")

def memory():
    """Create a kkachi Memory backed by this package's DB.

    Returns:
        kkachi.Memory: Persistent memory store.

    Raises:
        ImportError: If kkachi is not installed.
    """
    from kkachi import Memory
    return Memory().persist(db_path())

def search(query: str, k: int = 3):
    """Search the knowledge base.

    Args:
        query: Search query.
        k: Number of results (default 3).

    Returns:
        list[kkachi.Recall]: Search results.
    """
    mem = memory()
    return mem.search(query, k)
"#,
        package_name = package_name
    )
}

/// Generate PEP 566 METADATA.
fn generate_metadata(meta: &PackageMeta, normalized_name: &str, compressed: bool) -> String {
    let mut out = String::new();
    out.push_str("Metadata-Version: 2.1\n");
    out.push_str(&format!("Name: {}\n", normalized_name));
    out.push_str(&format!("Version: {}\n", meta.version));
    out.push_str(&format!("Summary: {}\n", meta.description));
    if !meta.author.is_empty() {
        out.push_str(&format!("Author: {}\n", meta.author));
    }
    out.push_str("Requires-Python: >=3.8\n");
    out.push_str("License: Proprietary\n");
    if compressed {
        out.push_str("Requires-Dist: zstandard>=0.20\n");
    }
    out
}

/// Generate PEP 427 WHEEL metadata.
fn generate_wheel_metadata() -> String {
    "Wheel-Version: 1.0\nGenerator: kkachi\nRoot-Is-Purelib: true\nTag: py3-none-any\n".to_string()
}

/// Normalize a package name per PEP 503 (hyphens → underscores).
pub fn normalize_wheel_name(name: &str, version: &str) -> String {
    let normalized = name.replace('-', "_");
    format!("{}-{}-py3-none-any.whl", normalized, version)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wheel_name_pep427() {
        let name = normalize_wheel_name("my_kb", "0.1.0");
        assert_eq!(name, "my_kb-0.1.0-py3-none-any.whl");
    }

    #[test]
    fn test_wheel_name_normalizes_hyphens() {
        let name = normalize_wheel_name("my-knowledge-base", "1.2.3");
        assert_eq!(name, "my_knowledge_base-1.2.3-py3-none-any.whl");
    }

    #[test]
    fn test_error_missing_db() {
        let result = PackagerBuilder::new(Path::new("/nonexistent/path/db.duckdb")).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_wheel_contents_uncompressed() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        std::fs::write(&db_path, b"fake-db-content").unwrap();

        let out_dir = dir.path().join("dist");
        let result = PackagerBuilder::new(&db_path)
            .name("test_pkg")
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .compress(false)
            .build()
            .unwrap();

        assert_eq!(result.file_count, 5);
        assert!(!result.compressed);
        assert!((result.compression_ratio - 1.0).abs() < f64::EPSILON);

        let file = std::fs::File::open(&result.wheel_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        let expected = [
            "test_pkg/__init__.py",
            "test_pkg/data/knowledge.db",
            "test_pkg-0.1.0.dist-info/METADATA",
            "test_pkg-0.1.0.dist-info/WHEEL",
            "test_pkg-0.1.0.dist-info/RECORD",
        ];

        for name in &expected {
            assert!(archive.by_name(name).is_ok(), "Missing entry: {}", name);
        }
    }

    #[test]
    fn test_wheel_contents_compressed() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        // Repetitive content compresses well
        std::fs::write(&db_path, "hello world ".repeat(1000)).unwrap();

        let out_dir = dir.path().join("dist");
        let result = PackagerBuilder::new(&db_path)
            .name("comp_pkg")
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .compress(true)
            .build()
            .unwrap();

        assert!(result.compressed);
        assert!(result.compression_ratio < 1.0);

        let file = std::fs::File::open(&result.wheel_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        // Should have .zst extension
        assert!(archive.by_name("comp_pkg/data/knowledge.db.zst").is_ok());
        // Should NOT have uncompressed version
        assert!(archive.by_name("comp_pkg/data/knowledge.db").is_err());

        // Metadata should require zstandard
        let mut metadata = String::new();
        {
            use std::io::Read;
            let mut entry = archive
                .by_name("comp_pkg-0.1.0.dist-info/METADATA")
                .unwrap();
            entry.read_to_string(&mut metadata).unwrap();
        }
        assert!(metadata.contains("Requires-Dist: zstandard>=0.20"));

        // __init__.py should have decompression logic
        let mut init = String::new();
        {
            use std::io::Read;
            let mut entry = archive.by_name("comp_pkg/__init__.py").unwrap();
            entry.read_to_string(&mut init).unwrap();
        }
        assert!(init.contains("zstandard"));
        assert!(init.contains("_ensure_decompressed"));
    }

    #[test]
    fn test_wheel_record_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        std::fs::write(&db_path, b"test-content").unwrap();

        let out_dir = dir.path().join("dist");
        let result = PackagerBuilder::new(&db_path)
            .name("hash_test")
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .compress(false)
            .build()
            .unwrap();

        let file = std::fs::File::open(&result.wheel_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        // Read RECORD
        let mut record_content = String::new();
        {
            use std::io::Read;
            let mut record = archive.by_name("hash_test-0.1.0.dist-info/RECORD").unwrap();
            record.read_to_string(&mut record_content).unwrap();
        }

        // Verify RECORD has hash entries
        assert!(record_content.contains("sha256="));

        // Verify the __init__.py hash is valid
        for line in record_content.lines() {
            if line.contains("__init__.py") {
                let parts: Vec<&str> = line.split(',').collect();
                assert_eq!(parts.len(), 3);
                assert!(parts[1].starts_with("sha256="));
                let size: usize = parts[2].parse().unwrap();
                assert!(size > 0);
            }
        }
    }

    #[test]
    fn test_wheel_init_py_content() {
        let init = generate_init_py("my_kb");
        assert!(init.contains("def db_path()"));
        assert!(init.contains("def memory()"));
        assert!(init.contains("def search("));
        assert!(init.contains("knowledge.db"));
    }

    #[test]
    fn test_mmap_db_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let db_content = b"This is the database content for integrity check!";
        let db_path = dir.path().join("integrity.db");
        std::fs::write(&db_path, db_content).unwrap();

        let out_dir = dir.path().join("dist");
        let result = PackagerBuilder::new(&db_path)
            .name("integrity_test")
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .compress(false)
            .build()
            .unwrap();

        // Read the db from the wheel and compare
        let file = std::fs::File::open(&result.wheel_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut db_in_wheel = archive.by_name("integrity_test/data/knowledge.db").unwrap();

        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut db_in_wheel, &mut buf).unwrap();
        assert_eq!(buf, db_content);
    }

    #[test]
    fn test_package_result_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let db_content = vec![0u8; 1024];
        let db_path = dir.path().join("sized.db");
        std::fs::write(&db_path, &db_content).unwrap();

        let out_dir = dir.path().join("dist");
        let result = PackagerBuilder::new(&db_path)
            .name("sized_pkg")
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .compress(false)
            .build()
            .unwrap();

        assert_eq!(result.db_size_bytes, 1024);
        assert!(result.size_bytes > 0);
    }

    #[test]
    fn test_base64url_nopad() {
        // Known SHA256 hash of empty string
        let mut hasher = Sha256::new();
        hasher.update(b"");
        let hash = base64url_nopad(&hasher.finalize());
        // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        // base64url(no pad) of that is a known value
        assert!(!hash.is_empty());
        assert!(!hash.contains('='));
        assert!(!hash.contains('+'));
        assert!(!hash.contains('/'));
    }
}
