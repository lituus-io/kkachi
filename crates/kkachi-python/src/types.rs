// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python type wrappers for Kkachi types.

use pyo3::prelude::*;
use std::collections::HashMap;

use kkachi::recursive::{
    HashEmbedder, InMemoryVectorStore, RefineResult, RefinementResult, VectorSearchResult,
    VectorStore,
};

/// Result from a refinement operation.
#[pyclass(name = "RefinementResult")]
#[derive(Clone)]
pub struct PyRefinementResult {
    /// The final answer.
    #[pyo3(get)]
    pub answer: String,
    /// Summary of the answer.
    #[pyo3(get)]
    pub summary: String,
    /// Final quality score (0.0 - 1.0).
    #[pyo3(get)]
    pub score: f64,
    /// Number of iterations taken.
    #[pyo3(get)]
    pub iterations: usize,
    /// Whether result came from cache.
    #[pyo3(get)]
    pub from_cache: bool,
}

#[pymethods]
impl PyRefinementResult {
    fn __repr__(&self) -> String {
        format!(
            "RefinementResult(score={:.3}, iterations={}, from_cache={})",
            self.score, self.iterations, self.from_cache
        )
    }

    fn __str__(&self) -> String {
        self.answer.clone()
    }

    /// Check if the refinement was successful (score >= 0.8).
    fn is_successful(&self) -> bool {
        self.score >= 0.8
    }
}

impl From<RefinementResult> for PyRefinementResult {
    fn from(r: RefinementResult) -> Self {
        Self {
            answer: r.answer,
            summary: r.summary,
            score: r.score,
            iterations: r.iterations,
            from_cache: r.from_cache,
        }
    }
}

/// Result from the declarative API.
#[pyclass(name = "RefineResult")]
#[derive(Clone)]
pub struct PyRefineResult {
    /// The final answer.
    #[pyo3(get)]
    pub answer: String,
    /// Summary of the answer.
    #[pyo3(get)]
    pub summary: String,
    /// Final quality score (0.0 - 1.0).
    #[pyo3(get)]
    pub score: f64,
    /// Number of iterations taken.
    #[pyo3(get)]
    pub iterations: usize,
    /// Whether result came from cache.
    #[pyo3(get)]
    pub from_cache: bool,
    /// Domain used.
    #[pyo3(get)]
    pub domain: Option<String>,
}

#[pymethods]
impl PyRefineResult {
    fn __repr__(&self) -> String {
        format!(
            "RefineResult(score={:.3}, iterations={}, domain={:?})",
            self.score, self.iterations, self.domain
        )
    }

    fn __str__(&self) -> String {
        self.answer.clone()
    }
}

impl From<RefineResult> for PyRefineResult {
    fn from(r: RefineResult) -> Self {
        Self {
            answer: r.answer,
            summary: r.summary,
            score: r.score,
            iterations: r.iterations,
            from_cache: r.from_cache,
            domain: r.domain,
        }
    }
}

/// Result from a vector store search.
#[pyclass(name = "VectorSearchResult")]
#[derive(Clone)]
pub struct PyVectorSearchResult {
    /// Document identifier.
    #[pyo3(get)]
    pub id: String,
    /// Document content.
    #[pyo3(get)]
    pub content: String,
    /// Similarity score (0.0 - 1.0).
    #[pyo3(get)]
    pub score: f32,
    /// Optional metadata.
    #[pyo3(get)]
    pub metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl PyVectorSearchResult {
    #[new]
    fn new(id: String, content: String, score: f32) -> Self {
        Self {
            id,
            content,
            score,
            metadata: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorSearchResult(id='{}', score={:.3})",
            self.id, self.score
        )
    }
}

impl From<VectorSearchResult> for PyVectorSearchResult {
    fn from(r: VectorSearchResult) -> Self {
        Self {
            id: r.id,
            content: r.content,
            score: r.score,
            metadata: r.metadata,
        }
    }
}

/// In-memory vector store for testing and small-scale use.
#[pyclass(name = "InMemoryVectorStore")]
pub struct PyInMemoryVectorStore {
    inner: InMemoryVectorStore<HashEmbedder>,
}

#[pymethods]
impl PyInMemoryVectorStore {
    /// Create a new empty store with the given embedding dimension.
    #[new]
    #[pyo3(signature = (dimension=64))]
    fn new(dimension: usize) -> Self {
        Self {
            inner: InMemoryVectorStore::new(HashEmbedder::new(dimension)),
        }
    }

    /// Add a document to the store.
    fn add(&mut self, id: String, content: String) {
        self.inner.add(id, content);
    }

    /// Add multiple documents.
    fn add_batch(&mut self, docs: Vec<(String, String)>) {
        self.inner.add_batch(docs);
    }

    /// Clear all documents.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Search by text query.
    fn search(&self, query: &str, k: usize) -> Vec<PyVectorSearchResult> {
        self.inner
            .search_text(query, k)
            .into_iter()
            .map(|r| r.into())
            .collect()
    }

    /// Get the number of documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get embedding dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!(
            "InMemoryVectorStore(len={}, dimension={})",
            self.inner.len(),
            self.inner.dimension()
        )
    }
}
