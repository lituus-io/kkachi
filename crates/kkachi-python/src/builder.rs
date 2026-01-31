// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the declarative Refine API.

use pyo3::prelude::*;

use kkachi::recursive::{checks, refine, Cli, IterativeMockLlm};

use crate::types::PyRefineResult;
use crate::validator::PyCliValidator;

/// Main entry point for Kkachi.
#[pyclass(name = "Kkachi")]
pub struct PyKkachi;

#[pymethods]
impl PyKkachi {
    /// Start building a recursive refinement pipeline.
    ///
    /// Args:
    ///     prompt: The prompt/question to refine.
    ///
    /// Returns:
    ///     RefineBuilder: A builder for configuring the refinement pipeline.
    #[staticmethod]
    fn refine(prompt: String) -> PyRefineBuilder {
        PyRefineBuilder::new(prompt)
    }
}

/// Builder for declarative recursive refinement.
#[pyclass(name = "RefineBuilder")]
#[derive(Clone)]
pub struct PyRefineBuilder {
    prompt: String,
    max_iterations: u32,
    score_threshold: f64,
    cli_validator: Option<Cli>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    require_patterns: Vec<String>,
    forbid_patterns: Vec<String>,
    adaptive: bool,
    token_budget: Option<u32>,
}

#[pymethods]
impl PyRefineBuilder {
    #[new]
    fn new(prompt: String) -> Self {
        Self {
            prompt,
            max_iterations: 10,
            score_threshold: 0.9,
            cli_validator: None,
            min_length: None,
            max_length: None,
            require_patterns: Vec::new(),
            forbid_patterns: Vec::new(),
            adaptive: false,
            token_budget: None,
        }
    }

    /// Set maximum iterations.
    fn max_iter(&self, n: u32) -> Self {
        let mut new = self.clone();
        new.max_iterations = n;
        new
    }

    /// Set score threshold for convergence.
    fn target(&self, threshold: f64) -> Self {
        let mut new = self.clone();
        new.score_threshold = threshold;
        new
    }

    /// Use a CLI validator.
    fn validate(&self, validator: PyCliValidator) -> Self {
        let mut new = self.clone();
        new.cli_validator = Some(validator.into_inner());
        new
    }

    /// Require a pattern in the output.
    fn require(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.require_patterns.push(pattern);
        new
    }

    /// Forbid a pattern in the output.
    fn forbid(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.forbid_patterns.push(pattern);
        new
    }

    /// Set minimum output length.
    fn min_len(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.min_length = Some(n);
        new
    }

    /// Set maximum output length.
    fn max_len(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.max_length = Some(n);
        new
    }

    /// Enable adaptive iteration mode.
    fn adaptive(&self) -> Self {
        let mut new = self.clone();
        new.adaptive = true;
        new
    }

    /// Set token budget limit.
    fn with_budget(&self, max_tokens: u32) -> Self {
        let mut new = self.clone();
        new.token_budget = Some(max_tokens);
        new
    }

    /// Run the refinement pipeline.
    ///
    /// Args:
    ///     generate: A callable that takes (iteration, prompt, feedback) and returns output.
    ///
    /// Returns:
    ///     RefineResult: The result containing the final output and metadata.
    fn run(&self, generate: PyObject) -> PyResult<PyRefineResult> {
        // Create a mock LLM that calls the Python function
        let responses: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let prompt = self.prompt.clone();

        // Pre-generate responses by calling the Python function
        Python::with_gil(|py| -> PyResult<()> {
            for iter in 0..self.max_iterations {
                let feedback: Option<String> = None;
                let result = generate
                    .call1(py, (iter, prompt.clone(), feedback))
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Python generate error: {}",
                            e
                        ))
                    })?;
                let output: String = result.extract(py).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Python return type error: {}",
                        e
                    ))
                })?;
                responses.lock().unwrap().push(output);
            }
            Ok(())
        })?;

        let responses_clone = responses.clone();
        let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
            let resps = responses_clone.lock().unwrap();
            let idx = (iter as usize).min(resps.len().saturating_sub(1));
            resps.get(idx).cloned().unwrap_or_default()
        });

        // Build the checks validator if patterns are specified
        let mut check_builder = checks();
        for pattern in &self.require_patterns {
            check_builder = check_builder.require(pattern);
        }
        for pattern in &self.forbid_patterns {
            check_builder = check_builder.forbid(pattern);
        }
        if let Some(min) = self.min_length {
            check_builder = check_builder.min_len(min);
        }
        if let Some(max) = self.max_length {
            check_builder = check_builder.max_len(max);
        }

        // Run refinement
        let result = if let Some(ref cli) = self.cli_validator {
            // Use CLI validator
            let mut builder = refine(&llm, &self.prompt)
                .validate(cli.clone())
                .max_iter(self.max_iterations)
                .target(self.score_threshold);

            if self.adaptive {
                builder = builder.adaptive();
            }
            if let Some(budget) = self.token_budget {
                builder = builder.with_budget(budget);
            }

            builder.go()
        } else if !self.require_patterns.is_empty()
            || !self.forbid_patterns.is_empty()
            || self.min_length.is_some()
            || self.max_length.is_some()
        {
            // Use checks validator
            let mut builder = refine(&llm, &self.prompt)
                .validate(check_builder)
                .max_iter(self.max_iterations)
                .target(self.score_threshold);

            if self.adaptive {
                builder = builder.adaptive();
            }
            if let Some(budget) = self.token_budget {
                builder = builder.with_budget(budget);
            }

            builder.go()
        } else {
            // No validation
            let mut builder = refine(&llm, &self.prompt)
                .max_iter(self.max_iterations)
                .target(self.score_threshold);

            if self.adaptive {
                builder = builder.adaptive();
            }
            if let Some(budget) = self.token_budget {
                builder = builder.with_budget(budget);
            }

            builder.go()
        };

        match result {
            Ok(r) => Ok(r.into()),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Refinement error: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RefineBuilder(prompt='{}', max_iter={}, target={})",
            self.prompt, self.max_iterations, self.score_threshold
        )
    }
}
