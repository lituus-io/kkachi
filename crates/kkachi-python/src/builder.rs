// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the declarative Pipeline API.

use pyo3::prelude::*;

use kkachi::recursive::{CliPipeline, Kkachi, SimilarityWeights};

use crate::similarity::PySimilarityWeights;
use crate::types::PyRefinementResult;
use crate::validator::PyCliPipeline;

/// Main entry point for Kkachi.
#[pyclass(name = "Kkachi")]
pub struct PyKkachi;

#[pymethods]
impl PyKkachi {
    /// Start building a recursive refinement pipeline.
    ///
    /// Args:
    ///     signature: The signature string (e.g., "question -> code")
    ///
    /// Returns:
    ///     RefineBuilder: A builder for configuring the refinement pipeline.
    #[staticmethod]
    fn refine(signature: String) -> PyRefineBuilder {
        PyRefineBuilder::new(signature)
    }
}

/// Builder for declarative recursive refinement.
#[pyclass(name = "RefineBuilder")]
#[derive(Clone)]
pub struct PyRefineBuilder {
    signature: String,
    domain: Option<String>,
    storage_path: Option<String>,
    max_iterations: u32,
    score_threshold: f64,
    plateau_threshold: Option<f64>,
    plateau_window: Option<usize>,
    cli_pipeline: Option<CliPipeline>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    use_semantic_cache: bool,
    similarity_threshold: f32,
    auto_condense: bool,
    cluster_threshold: f32,
    min_cluster_size: usize,
    similarity_weights: Option<SimilarityWeights>,
    few_shot_k: Option<usize>,
    few_shot_as_demos: bool,
    few_shot_refresh: bool,
    use_cot: bool,
    best_of_n: Option<u8>,
}

#[pymethods]
impl PyRefineBuilder {
    #[new]
    fn new(signature: String) -> Self {
        Self {
            signature,
            domain: None,
            storage_path: None,
            max_iterations: 10,
            score_threshold: 0.9,
            plateau_threshold: None,
            plateau_window: None,
            cli_pipeline: None,
            min_length: Some(10),
            max_length: None,
            use_semantic_cache: true,
            similarity_threshold: 0.95,
            auto_condense: true,
            cluster_threshold: 0.80,
            min_cluster_size: 3,
            similarity_weights: None,
            few_shot_k: None,
            few_shot_as_demos: true,
            few_shot_refresh: false,
            use_cot: false,
            best_of_n: None,
        }
    }

    // ===== Domain & Storage =====

    /// Set the domain namespace for storage/retrieval.
    fn domain(&self, domain: String) -> Self {
        let mut new = self.clone();
        new.domain = Some(domain);
        new
    }

    /// Set the storage path.
    fn storage(&self, path: String) -> Self {
        let mut new = self.clone();
        new.storage_path = Some(path);
        new
    }

    // ===== Convergence Criteria =====

    /// Set maximum iterations.
    fn max_iterations(&self, n: u32) -> Self {
        let mut new = self.clone();
        new.max_iterations = n;
        new
    }

    /// Set score threshold for convergence.
    fn until_score(&self, threshold: f64) -> Self {
        let mut new = self.clone();
        new.score_threshold = threshold;
        new
    }

    /// Set plateau detection for convergence.
    fn until_plateau(&self, min_improvement: f64, window: usize) -> Self {
        let mut new = self.clone();
        new.plateau_threshold = Some(min_improvement);
        new.plateau_window = Some(window);
        new
    }

    // ===== Validation =====

    /// Use a custom CLI pipeline for validation.
    ///
    /// Example:
    /// ```python
    /// validator = CliPipeline() \
    ///     .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \
    ///     .stage("compile", Cli("rustc").args(["--emit=metadata"]).required()) \
    ///     .file_ext("rs")
    ///
    /// result = Kkachi.refine("question -> code") \
    ///     .validate(validator) \
    ///     .run("Write a URL parser", generate)
    /// ```
    fn validate(&self, pipeline: PyCliPipeline) -> Self {
        let mut new = self.clone();
        new.cli_pipeline = Some(pipeline.into_inner());
        new
    }

    /// Use a heuristic critic with length bounds.
    #[pyo3(signature = (min_length=None, max_length=None))]
    fn critic_heuristic(&self, min_length: Option<usize>, max_length: Option<usize>) -> Self {
        let mut new = self.clone();
        new.cli_pipeline = None;
        new.min_length = min_length;
        new.max_length = max_length;
        new
    }

    // ===== Similarity & Retrieval =====

    /// Enable or disable semantic cache.
    fn semantic_cache(&self, enabled: bool) -> Self {
        let mut new = self.clone();
        new.use_semantic_cache = enabled;
        new
    }

    /// Set similarity threshold for cache hit.
    fn similarity_threshold(&self, threshold: f32) -> Self {
        let mut new = self.clone();
        new.similarity_threshold = threshold;
        new
    }

    /// Enable or disable auto-condensation.
    fn auto_condense(&self, enabled: bool) -> Self {
        let mut new = self.clone();
        new.auto_condense = enabled;
        new
    }

    /// Set cluster threshold.
    fn cluster_threshold(&self, threshold: f32) -> Self {
        let mut new = self.clone();
        new.cluster_threshold = threshold;
        new
    }

    /// Set minimum cluster size.
    fn min_cluster_size(&self, size: usize) -> Self {
        let mut new = self.clone();
        new.min_cluster_size = size;
        new
    }

    /// Set similarity weights.
    fn similarity_weights(&self, weights: PySimilarityWeights) -> Self {
        let mut new = self.clone();
        new.similarity_weights = Some((&weights).into());
        new
    }

    // ===== Few-Shot =====

    /// Set number of few-shot examples.
    fn few_shot_k(&self, k: usize) -> Self {
        let mut new = self.clone();
        new.few_shot_k = Some(k);
        new
    }

    /// Use few-shot examples as demonstrations.
    fn few_shot_as_demos(&self, enabled: bool) -> Self {
        let mut new = self.clone();
        new.few_shot_as_demos = enabled;
        new
    }

    /// Refresh examples each iteration.
    fn few_shot_refresh(&self, enabled: bool) -> Self {
        let mut new = self.clone();
        new.few_shot_refresh = enabled;
        new
    }

    // ===== DSPy Integration =====

    /// Enable chain of thought reasoning.
    fn with_chain_of_thought(&self) -> Self {
        let mut new = self.clone();
        new.use_cot = true;
        new
    }

    /// Enable best-of-N sampling.
    fn with_best_of_n(&self, n: u8) -> Self {
        let mut new = self.clone();
        new.best_of_n = Some(n);
        new
    }

    // ===== Execute =====

    /// Run the refinement pipeline.
    ///
    /// Args:
    ///     question: The question/input to refine.
    ///     generate: A callable that takes (iteration, feedback) and returns output.
    ///
    /// Returns:
    ///     RefinementResult: The result containing the final answer and metadata.
    fn run(&self, question: String, generate: PyObject) -> PyResult<PyRefinementResult> {
        // Build the Rust RefineBuilder
        let mut builder = Kkachi::refine(&self.signature);

        if let Some(ref domain) = self.domain {
            builder = builder.domain(domain);
        }
        if let Some(ref path) = self.storage_path {
            builder = builder.storage(path);
        }

        builder = builder.max_iterations(self.max_iterations);
        builder = builder.until_score(self.score_threshold);

        if let (Some(threshold), Some(window)) = (self.plateau_threshold, self.plateau_window) {
            builder = builder.until_plateau(threshold, window);
        }

        // Set validator
        if let Some(ref pipeline) = self.cli_pipeline {
            builder = builder.validate(pipeline.clone());
        } else {
            builder = builder.critic_heuristic(self.min_length, self.max_length);
        }

        // Similarity settings
        builder = builder.semantic_cache(self.use_semantic_cache);
        builder = builder.similarity_threshold(self.similarity_threshold);
        builder = builder.auto_condense(self.auto_condense);
        builder = builder.cluster_threshold(self.cluster_threshold);
        builder = builder.min_cluster_size(self.min_cluster_size);

        if let Some(ref weights) = self.similarity_weights {
            builder = builder.similarity_weights(*weights);
        }

        // Few-shot
        if let Some(k) = self.few_shot_k {
            builder = builder.few_shot_k(k);
        }
        builder = builder.few_shot_as_demos(self.few_shot_as_demos);
        builder = builder.few_shot_refresh(self.few_shot_refresh);

        // DSPy integration
        if self.use_cot {
            builder = builder.with_chain_of_thought();
        }
        if let Some(n) = self.best_of_n {
            builder = builder.with_best_of_n(n);
        }

        // Create the generate function that calls Python
        let generate_fn = |iteration: u32, feedback: Option<&str>| -> kkachi::Result<String> {
            Python::with_gil(|py| {
                let feedback_arg = feedback.map(|s| s.to_string());
                let result = generate
                    .call1(py, (iteration, feedback_arg))
                    .map_err(|e| kkachi::Error::module(format!("Python generate error: {}", e)))?;
                let output: String = result.extract(py).map_err(|e| {
                    kkachi::Error::module(format!("Python return type error: {}", e))
                })?;
                Ok(output)
            })
        };

        // Run the refinement
        let result = builder.run(&question, generate_fn);
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "RefineBuilder(signature='{}', domain={:?}, max_iterations={})",
            self.signature, self.domain, self.max_iterations
        )
    }
}
