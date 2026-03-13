// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python type wrappers for Kkachi types.

use pyo3::prelude::*;

use crate::error::IntoPyResult;
use kkachi::recursive::{memory, Memory, Recall, RefineResult};

/// Result from a refinement operation.
#[pyclass(name = "RefineResult")]
#[derive(Clone)]
pub struct PyRefineResult {
    /// The final output.
    #[pyo3(get)]
    pub output: String,
    /// Final quality score (0.0 - 1.0).
    #[pyo3(get)]
    pub score: f64,
    /// Number of iterations taken.
    #[pyo3(get)]
    pub iterations: u32,
    /// Whether result came from cache.
    #[pyo3(get)]
    pub from_cache: bool,
    /// Context ID for tracking.
    #[pyo3(get)]
    pub context_id: String,
}

#[pymethods]
impl PyRefineResult {
    fn __repr__(&self) -> String {
        format!(
            "RefineResult(score={:.3}, iterations={}, from_cache={})",
            self.score, self.iterations, self.from_cache
        )
    }

    fn __str__(&self) -> String {
        self.output.clone()
    }

    /// Check if the refinement was successful (score >= 0.8).
    fn is_successful(&self) -> bool {
        self.score >= 0.8
    }
}

impl From<RefineResult> for PyRefineResult {
    fn from(r: RefineResult) -> Self {
        Self {
            output: r.output,
            score: r.score,
            iterations: r.iterations,
            from_cache: r.from_cache,
            context_id: r.context_id.to_string(),
        }
    }
}

/// Result from a memory search.
#[pyclass(name = "Recall")]
#[derive(Clone)]
pub struct PyRecall {
    /// Document identifier.
    #[pyo3(get)]
    pub id: String,
    /// Document content.
    #[pyo3(get)]
    pub content: String,
    /// Similarity score (0.0 - 1.0).
    #[pyo3(get)]
    pub score: f64,
    /// Optional tag.
    #[pyo3(get)]
    pub tag: Option<String>,
}

#[pymethods]
impl PyRecall {
    #[new]
    fn new(id: String, content: String, score: f64) -> Self {
        Self {
            id,
            content,
            score,
            tag: None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Recall(id='{}', score={:.3})", self.id, self.score)
    }
}

impl From<Recall> for PyRecall {
    fn from(r: Recall) -> Self {
        Self {
            id: r.id,
            content: r.content,
            score: r.score,
            tag: r.tag,
        }
    }
}

/// In-memory store for testing and small-scale use.
#[pyclass(name = "Memory")]
pub struct PyMemory {
    inner: Memory,
}

#[pymethods]
impl PyMemory {
    /// Create a new empty memory store.
    #[new]
    fn new() -> Self {
        Self { inner: memory() }
    }

    /// Add a document to the store.
    fn add(&mut self, content: String) -> PyResult<String> {
        self.inner.add(&content).into_py()
    }

    /// Add a document with a specific ID.
    fn add_with_id(&mut self, id: String, content: String) -> PyResult<()> {
        self.inner.add_with_id(id, &content).into_py()
    }

    /// Add a document with a tag.
    fn add_tagged(&mut self, tag: String, content: String) -> PyResult<String> {
        self.inner.add_tagged(&tag, &content).into_py()
    }

    /// Search by text query.
    fn search(&self, query: &str, k: usize) -> PyResult<Vec<PyRecall>> {
        Ok(self
            .inner
            .search(query, k)
            .into_py()?
            .into_iter()
            .map(|r| r.into())
            .collect())
    }

    /// Get a document by ID.
    fn get(&self, id: &str) -> Option<String> {
        self.inner.get(id)
    }

    /// Remove a document by ID.
    fn remove(&mut self, id: &str) -> PyResult<bool> {
        self.inner.remove(id).into_py()
    }

    /// Update an existing document's content.
    fn update(&mut self, id: String, content: String) -> PyResult<bool> {
        self.inner.update(&id, &content).into_py()
    }

    /// Insert or update a document by content hash.
    fn upsert(&mut self, content: String) -> PyResult<String> {
        self.inner.upsert(&content).into_py()
    }

    /// Insert or update a tagged document by content hash.
    fn upsert_tagged(&mut self, tag: String, content: String) -> PyResult<String> {
        self.inner.upsert_tagged(&tag, &content).into_py()
    }

    /// Search with MMR for diverse results.
    #[pyo3(signature = (query, k, lambda_=0.5))]
    fn search_diverse(&self, query: &str, k: usize, lambda_: f64) -> PyResult<Vec<PyRecall>> {
        Ok(self
            .inner
            .search_diverse(query, k, lambda_)
            .into_py()?
            .into_iter()
            .map(|r| r.into())
            .collect())
    }

    /// Search filtering by minimum similarity score.
    fn search_above(&self, query: &str, k: usize, min_score: f64) -> PyResult<Vec<PyRecall>> {
        Ok(self
            .inner
            .search_above(query, k, min_score)
            .into_py()?
            .into_iter()
            .map(|r| r.into())
            .collect())
    }

    /// Learn from a successful refinement (write-back).
    fn learn(&mut self, question: &str, output: &str, score: f64) -> PyResult<()> {
        self.inner.learn(question, output, score).into_py()
    }

    /// Enable learning above a score threshold (fluent).
    fn learn_above(mut self_: PyRefMut<'_, Self>, threshold: f64) -> Py<Self> {
        self_.inner = std::mem::replace(&mut self_.inner, memory()).learn_above(threshold);
        self_.into()
    }

    /// Enable MMR diversity (fluent).
    fn diversity(mut self_: PyRefMut<'_, Self>, lambda_: f64) -> Py<Self> {
        self_.inner = std::mem::replace(&mut self_.inner, memory()).diversity(lambda_);
        self_.into()
    }

    /// Get the number of documents.
    fn __len__(&self) -> PyResult<usize> {
        self.inner.len().into_py()
    }

    /// Check if empty.
    fn is_empty(&self) -> PyResult<bool> {
        self.inner.is_empty().into_py()
    }

    /// Get all tags in the store.
    fn tags(&self) -> PyResult<Vec<String>> {
        self.inner.tags().into_py()
    }

    /// List all entries in the store.
    fn list(&self) -> PyResult<Vec<PyRecall>> {
        Ok(self
            .inner
            .all()
            .into_py()?
            .into_iter()
            .map(|r| r.into())
            .collect())
    }

    /// Enable persistent storage using DuckDB.
    #[cfg(feature = "storage")]
    fn persist(mut self_: PyRefMut<'_, Self>, path: String) -> PyResult<Py<Self>> {
        let new_inner = std::mem::replace(&mut self_.inner, memory())
            .persist(&path)
            .into_py()?;
        self_.inner = new_inner;
        Ok(self_.into())
    }

    /// Package the persistent memory into a pip-installable wheel.
    #[cfg(feature = "storage")]
    #[pyo3(signature = (name, version="0.1.0", output_dir=".", description="Kkachi knowledge base", author="", compress=true))]
    fn package(
        &self,
        name: &str,
        version: &str,
        output_dir: &str,
        description: &str,
        author: &str,
        compress: bool,
    ) -> PyResult<PyPackageResult> {
        let builder = self.inner.package(name).into_py()?;
        let result = builder
            .version_owned(version.to_string())
            .output_dir_owned(std::path::PathBuf::from(output_dir))
            .description_owned(description.to_string())
            .author_owned(author.to_string())
            .compress(compress)
            .build()
            .into_py()?;

        Ok(PyPackageResult {
            wheel_path: result.wheel_path.display().to_string(),
            wheel_name: result.wheel_name,
            size_bytes: result.size_bytes,
            db_size_bytes: result.db_size_bytes,
            file_count: result.file_count,
            compressed: result.compressed,
            compression_ratio: result.compression_ratio,
        })
    }

    /// Get the database path if using persistent storage.
    #[cfg(feature = "storage")]
    fn db_path(&self) -> Option<String> {
        self.inner.db_path().map(|s| s.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Memory(len={})", self.inner.len().into_py()?))
    }
}

impl PyMemory {
    pub fn inner_ref(&self) -> &Memory {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Memory {
        &mut self.inner
    }
}

/// Result from packaging a Memory into a pip wheel.
#[pyclass(name = "PackageResult")]
#[derive(Clone)]
pub struct PyPackageResult {
    /// Full path to the generated .whl file.
    #[pyo3(get)]
    pub wheel_path: String,
    /// Wheel filename.
    #[pyo3(get)]
    pub wheel_name: String,
    /// Total size of the wheel file in bytes.
    #[pyo3(get)]
    pub size_bytes: u64,
    /// Size of the embedded .db file in bytes.
    #[pyo3(get)]
    pub db_size_bytes: u64,
    /// Number of files in the wheel.
    #[pyo3(get)]
    pub file_count: usize,
    /// Whether the DB was zstd-compressed.
    #[pyo3(get)]
    pub compressed: bool,
    /// Compression ratio (compressed/original). 1.0 if not compressed.
    #[pyo3(get)]
    pub compression_ratio: f64,
}

#[pymethods]
impl PyPackageResult {
    fn __repr__(&self) -> String {
        format!(
            "PackageResult(wheel='{}', size={}B, db={}B, files={})",
            self.wheel_name, self.size_bytes, self.db_size_bytes, self.file_count
        )
    }

    fn __str__(&self) -> String {
        self.wheel_path.clone()
    }
}
