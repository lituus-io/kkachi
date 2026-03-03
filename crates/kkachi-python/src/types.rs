// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python type wrappers for Kkachi types.

use pyo3::prelude::*;

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
    fn add(&mut self, content: String) -> String {
        self.inner.add(&content)
    }

    /// Add a document with a specific ID.
    fn add_with_id(&mut self, id: String, content: String) {
        self.inner.add_with_id(id, &content);
    }

    /// Add a document with a tag.
    fn add_tagged(&mut self, tag: String, content: String) -> String {
        self.inner.add_tagged(&tag, &content)
    }

    /// Search by text query.
    fn search(&self, query: &str, k: usize) -> Vec<PyRecall> {
        self.inner
            .search(query, k)
            .into_iter()
            .map(|r| r.into())
            .collect()
    }

    /// Get a document by ID.
    fn get(&self, id: &str) -> Option<String> {
        self.inner.get(id)
    }

    /// Remove a document by ID.
    fn remove(&mut self, id: &str) -> bool {
        self.inner.remove(id)
    }

    /// Update an existing document's content.
    ///
    /// Args:
    ///     id (str): Document ID to update
    ///     content (str): New content for the document
    ///
    /// Returns:
    ///     bool: True if document was found and updated, False otherwise
    ///
    /// Example:
    ///     ```python
    ///     from kkachi import Memory
    ///
    ///     mem = Memory()
    ///     doc_id = mem.add("Old content")
    ///     success = mem.update(doc_id, "New content")
    ///     assert mem.get(doc_id) == "New content"
    ///     ```
    fn update(&mut self, id: String, content: String) -> bool {
        self.inner.update(&id, &content)
    }

    /// Get the number of documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get all tags in the store.
    fn tags(&self) -> Vec<String> {
        self.inner.tags()
    }

    /// List all entries in the store.
    fn list(&self) -> Vec<PyRecall> {
        self.inner
            .all()
            .into_iter()
            .map(|r| PyRecall {
                id: r.id.to_string(),
                content: r.content.to_string(),
                score: r.score,
                tag: r.tag.map(|t| t.to_string()),
            })
            .collect()
    }

    /// Enable persistent storage using DuckDB.
    ///
    /// Args:
    ///     path (str): Path to the DuckDB database file (e.g., "./memory.db")
    ///
    /// Returns:
    ///     Memory: Self with persistent storage enabled
    ///
    /// Raises:
    ///     RuntimeError: If storage initialization fails
    ///
    /// Example:
    ///     ```python
    ///     from kkachi import Memory
    ///
    ///     mem = Memory().persist("./my_knowledge.db")
    ///     mem.add("Important document")
    ///     # Data persists across program restarts
    ///     ```
    ///
    /// Note:
    ///     Requires the 'storage' feature to be enabled. The database file
    ///     will be created if it doesn't exist.
    #[cfg(feature = "storage")]
    fn persist(mut self_: PyRefMut<'_, Self>, path: String) -> PyResult<Py<Self>> {
        use pyo3::exceptions::PyRuntimeError;

        let new_inner = std::mem::replace(&mut self_.inner, memory())
            .persist(&path)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to enable persistent storage: {}", e))
            })?;

        self_.inner = new_inner;

        Ok(self_.into())
    }

    /// Package the persistent memory into a pip-installable wheel.
    ///
    /// Args:
    ///     name (str): Package name (e.g., "my_kb")
    ///     version (str): Package version (default "0.1.0")
    ///     output_dir (str): Output directory (default ".")
    ///     description (str): Package description
    ///     author (str): Package author
    ///
    /// Returns:
    ///     PackageResult: Details about the generated wheel
    ///
    /// Raises:
    ///     RuntimeError: If memory is not persistent or packaging fails
    #[cfg(feature = "storage")]
    #[pyo3(signature = (name, version="0.1.0", output_dir=".", description="Kkachi knowledge base", author=""))]
    fn package(
        &self,
        name: &str,
        version: &str,
        output_dir: &str,
        description: &str,
        author: &str,
    ) -> PyResult<PyPackageResult> {
        use pyo3::exceptions::PyRuntimeError;

        let builder = self
            .inner
            .package(name)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        let result = builder
            .version_owned(version.to_string())
            .output_dir_owned(std::path::PathBuf::from(output_dir))
            .description_owned(description.to_string())
            .author_owned(author.to_string())
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        Ok(PyPackageResult {
            wheel_path: result.wheel_path.display().to_string(),
            wheel_name: result.wheel_name,
            size_bytes: result.size_bytes,
            db_size_bytes: result.db_size_bytes,
            file_count: result.file_count,
        })
    }

    fn __repr__(&self) -> String {
        format!("Memory(len={})", self.inner.len())
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
