// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for prompt optimization.

use pyo3::prelude::*;

use kkachi::recursive::optimize::{Dataset, Optimizer, Strategy};

use crate::compose::extract_validator_node;
use crate::dspy::PyCallableLlm;

// ---------------------------------------------------------------------------
// PyDataset
// ---------------------------------------------------------------------------

/// A collection of training examples used for optimization.
///
/// Example:
///     ds = Dataset().example("What is 2+2?", "4").example("What is 3*5?", "15")
#[pyclass(name = "Dataset")]
#[derive(Clone)]
pub struct PyDataset {
    examples: Vec<(String, String, Option<String>)>,
}

#[pymethods]
impl PyDataset {
    #[new]
    fn new() -> Self {
        PyDataset {
            examples: Vec::new(),
        }
    }

    /// Add an unlabeled example.
    fn example(&self, input: String, expected: String) -> Self {
        let mut next = self.clone();
        next.examples.push((input, expected, None));
        next
    }

    /// Add a labeled example.
    fn labeled_example(&self, input: String, expected: String, label: String) -> Self {
        let mut next = self.clone();
        next.examples.push((input, expected, Some(label)));
        next
    }

    fn len(&self) -> usize {
        self.examples.len()
    }

    fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    fn __len__(&self) -> usize {
        self.examples.len()
    }

    fn __repr__(&self) -> String {
        format!("Dataset(examples={})", self.examples.len())
    }
}

impl PyDataset {
    /// Convert to the Rust `Dataset` type.
    fn to_rust_dataset(&self) -> Dataset {
        let mut ds = Dataset::new();
        for (input, expected, label) in &self.examples {
            ds = match label {
                Some(l) => ds.labeled_example(input.as_str(), expected.as_str(), l.as_str()),
                None => ds.example(input.as_str(), expected.as_str()),
            };
        }
        ds
    }
}

// ---------------------------------------------------------------------------
// PyOptimizeResult
// ---------------------------------------------------------------------------

/// The result of an optimization run.
#[pyclass(name = "OptimizeResult")]
pub struct PyOptimizeResult {
    #[pyo3(get)]
    pub prompt: String,
    #[pyo3(get)]
    pub examples: Vec<(String, String)>,
    #[pyo3(get)]
    pub instruction: String,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub evaluations: u32,
    #[pyo3(get)]
    pub candidate_scores: Vec<f64>,
}

#[pymethods]
impl PyOptimizeResult {
    fn __repr__(&self) -> String {
        format!(
            "OptimizeResult(score={:.4}, evaluations={}, candidates={})",
            self.score,
            self.evaluations,
            self.candidate_scores.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyOptimizerBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and running a prompt optimization pass.
///
/// Example:
///     result = optimizer(llm, "Answer math questions") \
///         .dataset(ds) \
///         .metric(lambda pred, exp: 1.0 if exp in pred else 0.0) \
///         .strategy("bootstrap", max_examples=3) \
///         .go()
#[pyclass(name = "OptimizerBuilder")]
pub struct PyOptimizerBuilder {
    llm: PyObject,
    prompt: String,
    dataset: Option<PyDataset>,
    metric: Option<PyObject>,
    validator: Option<crate::compose::ValidatorNode>,
    strategy: Strategy,
}

impl Clone for PyOptimizerBuilder {
    fn clone(&self) -> Self {
        Self {
            llm: Python::with_gil(|py| self.llm.clone_ref(py)),
            prompt: self.prompt.clone(),
            dataset: self.dataset.clone(),
            metric: self
                .metric
                .as_ref()
                .map(|m| Python::with_gil(|py| m.clone_ref(py))),
            validator: self.validator.clone(),
            strategy: self.strategy.clone(),
        }
    }
}

#[pymethods]
impl PyOptimizerBuilder {
    /// Set the dataset for optimization.
    fn dataset(&self, dataset: PyDataset) -> Self {
        let mut new = self.clone();
        new.dataset = Some(dataset);
        new
    }

    /// Set the metric function `(prediction: str, expected: str) -> float`.
    fn metric(&self, metric: PyObject) -> Self {
        let mut new = self.clone();
        new.metric = Some(metric);
        new
    }

    /// Set the validator for optimization.
    fn validate(&self, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let node = extract_validator_node(validator)?;
        let mut new = self.clone();
        new.validator = Some(node);
        Ok(new)
    }

    /// Set the optimization strategy.
    #[pyo3(signature = (name, max_examples=3, num_candidates=5))]
    fn strategy(&self, name: &str, max_examples: usize, num_candidates: usize) -> PyResult<Self> {
        let strat = match name {
            "bootstrap" => Strategy::BootstrapFewShot { max_examples },
            "instruction" => Strategy::InstructionSearch { num_candidates },
            "combined" => Strategy::Combined {
                max_examples,
                num_candidates,
            },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown strategy '{}'. Expected 'bootstrap', 'instruction', or 'combined'.",
                    other
                )));
            }
        };
        let mut new = self.clone();
        new.strategy = strat;
        Ok(new)
    }

    /// Run the optimization and return the result.
    fn go(&self) -> PyResult<PyOptimizeResult> {
        let llm = Python::with_gil(|py| PyCallableLlm::new_from_ref(py, &self.llm));

        // Build owned Rust dataset so it outlives the Optimizer reference
        let rust_dataset = self.dataset.as_ref().map(|d| d.to_rust_dataset());

        let mut opt = Optimizer::new(&llm, &self.prompt);

        if let Some(ref ds) = rust_dataset {
            opt = opt.dataset(ds);
        }

        // Metric — wrap Python callable
        if let Some(ref metric_obj) = self.metric {
            let metric_py = Python::with_gil(|py| metric_obj.clone_ref(py));
            opt = opt.metric(move |prediction: &str, expected: &str| -> f64 {
                Python::with_gil(|py| {
                    metric_py
                        .call1(py, (prediction, expected))
                        .and_then(|v| v.extract::<f64>(py))
                        .unwrap_or(0.0)
                })
            });
        }

        // Validator — materialize from ValidatorNode
        if let Some(ref node) = self.validator {
            let validator = Python::with_gil(|py| node.materialize(py));
            opt = opt.validate(validator);
        }

        opt = opt.strategy(self.strategy.clone());

        let result = opt.go();

        Ok(PyOptimizeResult {
            prompt: result.prompt,
            examples: result
                .examples
                .iter()
                .map(|e| (e.input.clone(), e.expected.clone()))
                .collect(),
            instruction: result.instruction,
            score: result.score,
            evaluations: result.evaluations,
            candidate_scores: result.candidate_scores,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimizerBuilder(prompt={:?}, strategy={:?}, dataset={})",
            self.prompt,
            self.strategy,
            self.dataset
                .as_ref()
                .map_or("None".to_string(), |d| format!("{} examples", d.len()))
        )
    }
}

// ---------------------------------------------------------------------------
// Entry point function
// ---------------------------------------------------------------------------

/// Create an optimizer builder.
///
/// Example:
///     result = optimizer(llm, "Answer math questions").dataset(ds).go()
#[pyfunction]
#[pyo3(name = "optimizer")]
pub fn py_optimizer(llm: PyObject, prompt: String) -> PyOptimizerBuilder {
    PyOptimizerBuilder {
        llm,
        prompt,
        dataset: None,
        metric: None,
        validator: None,
        strategy: Strategy::BootstrapFewShot { max_examples: 3 },
    }
}
