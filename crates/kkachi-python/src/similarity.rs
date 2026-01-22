// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for similarity and few-shot configuration.

use pyo3::prelude::*;

use kkachi::recursive::SimilarityWeights;

/// Weights for multi-signal similarity scoring.
#[pyclass(name = "SimilarityWeights")]
#[derive(Clone, Copy)]
pub struct PySimilarityWeights {
    /// Weight for embedding similarity (default: 0.40).
    #[pyo3(get, set)]
    pub embedding: f32,
    /// Weight for keyword similarity (default: 0.25).
    #[pyo3(get, set)]
    pub keyword: f32,
    /// Weight for metadata/tag overlap (default: 0.20).
    #[pyo3(get, set)]
    pub metadata: f32,
    /// Weight for hierarchy/category similarity (default: 0.15).
    #[pyo3(get, set)]
    pub hierarchy: f32,
}

#[pymethods]
impl PySimilarityWeights {
    /// Create new similarity weights.
    ///
    /// Weights are automatically normalized to sum to 1.0.
    #[new]
    #[pyo3(signature = (embedding=0.40, keyword=0.25, metadata=0.20, hierarchy=0.15))]
    fn new(embedding: f32, keyword: f32, metadata: f32, hierarchy: f32) -> Self {
        Self {
            embedding,
            keyword,
            metadata,
            hierarchy,
        }
    }

    /// Create default weights.
    #[staticmethod]
    fn default_weights() -> Self {
        Self::new(0.40, 0.25, 0.20, 0.15)
    }

    /// Create weights emphasizing semantic similarity.
    #[staticmethod]
    fn semantic_focus() -> Self {
        Self::new(0.60, 0.20, 0.10, 0.10)
    }

    /// Create weights emphasizing keyword matching.
    #[staticmethod]
    fn keyword_focus() -> Self {
        Self::new(0.25, 0.50, 0.15, 0.10)
    }

    /// Get normalized weights that sum to 1.0.
    fn normalized(&self) -> PySimilarityWeights {
        let total = self.embedding + self.keyword + self.metadata + self.hierarchy;
        if total == 0.0 {
            return Self::default_weights();
        }
        Self {
            embedding: self.embedding / total,
            keyword: self.keyword / total,
            metadata: self.metadata / total,
            hierarchy: self.hierarchy / total,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SimilarityWeights(embedding={:.2}, keyword={:.2}, metadata={:.2}, hierarchy={:.2})",
            self.embedding, self.keyword, self.metadata, self.hierarchy
        )
    }
}

impl From<&PySimilarityWeights> for SimilarityWeights {
    fn from(w: &PySimilarityWeights) -> Self {
        SimilarityWeights {
            embedding: w.embedding,
            keyword: w.keyword,
            metadata: w.metadata,
            hierarchy: w.hierarchy,
        }
    }
}

impl From<SimilarityWeights> for PySimilarityWeights {
    fn from(w: SimilarityWeights) -> Self {
        Self {
            embedding: w.embedding,
            keyword: w.keyword,
            metadata: w.metadata,
            hierarchy: w.hierarchy,
        }
    }
}

/// Configuration for few-shot learning.
#[pyclass(name = "FewShotConfig")]
#[derive(Clone)]
pub struct PyFewShotConfig {
    /// Number of examples to retrieve per iteration.
    #[pyo3(get, set)]
    pub k: usize,
    /// Minimum similarity to include example.
    #[pyo3(get, set)]
    pub min_similarity: f32,
    /// Include examples in prompt.
    #[pyo3(get, set)]
    pub include_in_prompt: bool,
    /// Use examples as demonstrations.
    #[pyo3(get, set)]
    pub as_demonstrations: bool,
    /// Refresh examples each iteration.
    #[pyo3(get, set)]
    pub refresh_per_iteration: bool,
}

#[pymethods]
impl PyFewShotConfig {
    /// Create a new few-shot configuration.
    #[new]
    #[pyo3(signature = (k=3, min_similarity=0.7, include_in_prompt=true, as_demonstrations=true, refresh_per_iteration=false))]
    fn new(
        k: usize,
        min_similarity: f32,
        include_in_prompt: bool,
        as_demonstrations: bool,
        refresh_per_iteration: bool,
    ) -> Self {
        Self {
            k,
            min_similarity,
            include_in_prompt,
            as_demonstrations,
            refresh_per_iteration,
        }
    }

    /// Create default configuration.
    #[staticmethod]
    fn default_config() -> Self {
        Self::new(3, 0.7, true, true, false)
    }

    /// Create with specific k value.
    #[staticmethod]
    fn with_k(k: usize) -> Self {
        Self::new(k, 0.7, true, true, false)
    }

    fn __repr__(&self) -> String {
        format!(
            "FewShotConfig(k={}, min_similarity={:.2}, as_demos={})",
            self.k, self.min_similarity, self.as_demonstrations
        )
    }
}

impl From<&PyFewShotConfig> for kkachi::recursive::FewShotConfig {
    fn from(c: &PyFewShotConfig) -> Self {
        kkachi::recursive::FewShotConfig {
            k: c.k,
            min_similarity: c.min_similarity,
            include_in_prompt: c.include_in_prompt,
            as_demonstrations: c.as_demonstrations,
            refresh_per_iteration: c.refresh_per_iteration,
        }
    }
}

impl From<kkachi::recursive::FewShotConfig> for PyFewShotConfig {
    fn from(c: kkachi::recursive::FewShotConfig) -> Self {
        Self {
            k: c.k,
            min_similarity: c.min_similarity,
            include_in_prompt: c.include_in_prompt,
            as_demonstrations: c.as_demonstrations,
            refresh_per_iteration: c.refresh_per_iteration,
        }
    }
}
