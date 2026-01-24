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
