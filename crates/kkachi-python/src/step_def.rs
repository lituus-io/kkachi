// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for Step combinators.
//!
//! Provides Python access to the universal `Step` composition system:
//! - `step_fn(name, callable)` — wrap a Python callable as a Step
//! - `.then(other)` — sequential composition
//! - `.race(other)` — parallel, pick best score
//! - `.par(other)` — parallel, concatenate outputs
//! - `.fallback(other)` — try primary, fallback on failure
//! - `.retry(n, target)` — retry until score >= target
//! - `.map(func)` — transform output text
//! - `run_all_steps(input, steps)` — batch concurrent execution

use pyo3::prelude::*;

use kkachi::recursive::step::{DynStep, StepOutput};

use std::future::Future;
use std::pin::Pin;

// ============================================================================
// StepNode — Tree representation of composed steps
// ============================================================================

/// Internal representation of composed steps.
///
/// Stored as a tree and materialized into `Box<dyn DynStep>` at execution time.
enum StepNode {
    /// A Python callable `(input: str) -> str`.
    Fn { name: String, func: PyObject },
    /// A Python callable `(input: str) -> (str, float)` returning (text, score).
    FnScored { name: String, func: PyObject },
    /// Sequential: run first, then feed output to second.
    Then(Box<StepNode>, Box<StepNode>),
    /// Parallel: run both, pick best score.
    Race(Box<StepNode>, Box<StepNode>),
    /// Parallel: run both, concatenate outputs.
    Par(Box<StepNode>, Box<StepNode>),
    /// Fallback: if primary score == 0, try backup.
    Fallback(Box<StepNode>, Box<StepNode>),
    /// Retry up to N times until score >= target.
    Retry(Box<StepNode>, u32, f64),
    /// Transform the output text.
    Map(Box<StepNode>, PyObject),
}

impl Clone for StepNode {
    fn clone(&self) -> Self {
        Python::with_gil(|py| match self {
            Self::Fn { name, func } => Self::Fn {
                name: name.clone(),
                func: func.clone_ref(py),
            },
            Self::FnScored { name, func } => Self::FnScored {
                name: name.clone(),
                func: func.clone_ref(py),
            },
            Self::Then(a, b) => Self::Then(a.clone(), b.clone()),
            Self::Race(a, b) => Self::Race(a.clone(), b.clone()),
            Self::Par(a, b) => Self::Par(a.clone(), b.clone()),
            Self::Fallback(a, b) => Self::Fallback(a.clone(), b.clone()),
            Self::Retry(inner, n, target) => Self::Retry(inner.clone(), *n, *target),
            Self::Map(inner, func) => Self::Map(inner.clone(), func.clone_ref(py)),
        })
    }
}

impl StepNode {
    /// Materialize this node tree into a concrete `Box<dyn DynStep>`.
    fn materialize(&self) -> Box<dyn DynStep> {
        match self {
            Self::Fn { name, func } => {
                let func = Python::with_gil(|py| func.clone_ref(py));
                let name_owned = name.clone();
                Box::new(PyFnStep {
                    func,
                    name: name_owned,
                    scored: false,
                })
            }
            Self::FnScored { name, func } => {
                let func = Python::with_gil(|py| func.clone_ref(py));
                let name_owned = name.clone();
                Box::new(PyFnStep {
                    func,
                    name: name_owned,
                    scored: true,
                })
            }
            Self::Then(a, b) => {
                let step_a = a.materialize();
                let step_b = b.materialize();
                Box::new(DynChain(step_a, step_b))
            }
            Self::Race(a, b) => {
                let step_a = a.materialize();
                let step_b = b.materialize();
                Box::new(DynRace(step_a, step_b))
            }
            Self::Par(a, b) => {
                let step_a = a.materialize();
                let step_b = b.materialize();
                Box::new(DynPar(step_a, step_b))
            }
            Self::Fallback(a, b) => {
                let step_a = a.materialize();
                let step_b = b.materialize();
                Box::new(DynFallback(step_a, step_b))
            }
            Self::Retry(inner, n, target) => {
                let step = inner.materialize();
                Box::new(DynRetry {
                    inner: step,
                    max_attempts: *n,
                    target: *target,
                })
            }
            Self::Map(inner, func) => {
                let step = inner.materialize();
                let func = Python::with_gil(|py| func.clone_ref(py));
                Box::new(DynMap { inner: step, func })
            }
        }
    }
}

// ============================================================================
// DynStep implementations for composition (boxed dynamic dispatch)
// ============================================================================

/// A Python-callable step wrapping a Python function.
struct PyFnStep {
    func: PyObject,
    name: String,
    scored: bool,
}

unsafe impl Send for PyFnStep {}
unsafe impl Sync for PyFnStep {}

impl DynStep for PyFnStep {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        let input_owned = input.to_string();
        let scored = self.scored;
        Box::pin(async move {
            let (text, score) = Python::with_gil(|py| -> PyResult<(String, f64)> {
                let result = self.func.call1(py, (input_owned,))?;
                if scored {
                    let tuple = result.extract::<(String, f64)>(py)?;
                    Ok(tuple)
                } else {
                    let text = result.extract::<String>(py)?;
                    Ok((text, 1.0))
                }
            })
            .map_err(|e| kkachi::error::Error::module(&format!("Python step error: {}", e)))?;
            Ok(StepOutput::new(text, score, 0))
        })
    }

    fn dyn_name(&self) -> &'static str {
        // Leak the name since DynStep requires &'static str.
        // This is acceptable because steps are typically long-lived.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

/// Sequential composition over boxed DynStep.
struct DynChain(Box<dyn DynStep>, Box<dyn DynStep>);

impl DynStep for DynChain {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        Box::pin(async move {
            let first = self.0.run_dyn(input).await?;
            let mut second = self.1.run_dyn(&first.text).await?;
            second.tokens += first.tokens;
            Ok(second)
        })
    }

    fn dyn_name(&self) -> &'static str {
        "chain"
    }
}

/// Parallel composition, pick best score.
struct DynRace(Box<dyn DynStep>, Box<dyn DynStep>);

impl DynStep for DynRace {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        Box::pin(async move {
            // Run both (sequentially in sync context, but API is async-ready)
            let a = self.0.run_dyn(input).await;
            let b = self.1.run_dyn(input).await;
            match (a, b) {
                (Ok(out_a), Ok(out_b)) => {
                    if out_a.score >= out_b.score {
                        Ok(out_a)
                    } else {
                        Ok(out_b)
                    }
                }
                (Ok(out), Err(_)) | (Err(_), Ok(out)) => Ok(out),
                (Err(e), Err(_)) => Err(e),
            }
        })
    }

    fn dyn_name(&self) -> &'static str {
        "race"
    }
}

/// Parallel composition, concatenate outputs.
struct DynPar(Box<dyn DynStep>, Box<dyn DynStep>);

impl DynStep for DynPar {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        Box::pin(async move {
            let a = self.0.run_dyn(input).await;
            let b = self.1.run_dyn(input).await;
            match (a, b) {
                (Ok(out_a), Ok(out_b)) => {
                    let mut text = out_a.text;
                    text.push_str("\n\n");
                    text.push_str(&out_b.text);
                    let score = (out_a.score + out_b.score) / 2.0;
                    let tokens = out_a.tokens + out_b.tokens;
                    Ok(StepOutput::new(text, score, tokens))
                }
                (Ok(out), Err(_)) | (Err(_), Ok(out)) => Ok(out),
                (Err(e), Err(_)) => Err(e),
            }
        })
    }

    fn dyn_name(&self) -> &'static str {
        "par"
    }
}

/// Fallback: if primary score == 0, try backup.
struct DynFallback(Box<dyn DynStep>, Box<dyn DynStep>);

impl DynStep for DynFallback {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        Box::pin(async move {
            match self.0.run_dyn(input).await {
                Ok(output) if output.score > 0.0 => Ok(output),
                _ => self.1.run_dyn(input).await,
            }
        })
    }

    fn dyn_name(&self) -> &'static str {
        "fallback"
    }
}

/// Retry up to N times until score >= target.
struct DynRetry {
    inner: Box<dyn DynStep>,
    max_attempts: u32,
    target: f64,
}

impl DynStep for DynRetry {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        Box::pin(async move {
            let mut best: Option<StepOutput> = None;
            for _ in 0..self.max_attempts {
                match self.inner.run_dyn(input).await {
                    Ok(output) if output.score >= self.target => return Ok(output),
                    Ok(output) => {
                        best = Some(match best {
                            Some(prev) if prev.score >= output.score => prev,
                            _ => output,
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(best.unwrap_or_else(|| StepOutput::new(String::new(), 0.0, 0)))
        })
    }

    fn dyn_name(&self) -> &'static str {
        "retry"
    }
}

/// Transform the output text using a Python callable.
struct DynMap {
    inner: Box<dyn DynStep>,
    func: PyObject,
}

unsafe impl Send for DynMap {}
unsafe impl Sync for DynMap {}

impl DynStep for DynMap {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = kkachi::error::Result<StepOutput>> + Send + 'a>> {
        Box::pin(async move {
            let mut output = self.inner.run_dyn(input).await?;
            let original = output.text.clone();
            output.text = Python::with_gil(|py| {
                self.func
                    .call1(py, (original.clone(),))
                    .and_then(|r| r.extract::<String>(py))
                    .unwrap_or(original)
            });
            Ok(output)
        })
    }

    fn dyn_name(&self) -> &'static str {
        "map"
    }
}

// ============================================================================
// PyStepResult — Result from step execution
// ============================================================================

/// Result from executing a step or step composition.
#[pyclass(name = "StepResult")]
#[derive(Clone)]
pub struct PyStepResult {
    /// The output text.
    #[pyo3(get)]
    pub text: String,
    /// Quality score (0.0–1.0).
    #[pyo3(get)]
    pub score: f64,
    /// Total tokens consumed.
    #[pyo3(get)]
    pub tokens: u32,
}

#[pymethods]
impl PyStepResult {
    fn __repr__(&self) -> String {
        format!(
            "StepResult(score={:.3}, tokens={}, text={:?})",
            self.score,
            self.tokens,
            if self.text.len() > 50 {
                format!("{}...", &self.text[..50])
            } else {
                self.text.clone()
            }
        )
    }

    fn __str__(&self) -> &str {
        &self.text
    }
}

// ============================================================================
// PyStepDef — Python-facing step combinator
// ============================================================================

/// A composable step definition for building processing pipelines.
///
/// Steps are the universal building block. They take text in and produce
/// text + score out. Steps compose via combinator methods.
///
/// # Python Example
///
/// ```python
/// # Create steps from Python callables
/// upper = step_fn("upper", lambda s: s.upper())
/// check = step_scored("check", lambda s: (s, 1.0 if "HELLO" in s else 0.0))
///
/// # Compose steps
/// composed = upper.then(check)
/// result = composed.run("hello world")
/// assert result.score == 1.0
///
/// # Parallel with fallback
/// safe = primary.fallback(backup)
/// result = safe.run("input")
/// ```
#[pyclass(name = "StepDef")]
pub struct PyStepDef {
    node: StepNode,
}

impl Clone for PyStepDef {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

#[pymethods]
impl PyStepDef {
    /// Sequential composition: run self, feed output to next.
    fn then(&self, next: &PyStepDef) -> Self {
        Self {
            node: StepNode::Then(Box::new(self.node.clone()), Box::new(next.node.clone())),
        }
    }

    /// Parallel composition: run both, pick the highest score.
    fn race(&self, other: &PyStepDef) -> Self {
        Self {
            node: StepNode::Race(Box::new(self.node.clone()), Box::new(other.node.clone())),
        }
    }

    /// Parallel composition: run both, concatenate outputs.
    fn par(&self, other: &PyStepDef) -> Self {
        Self {
            node: StepNode::Par(Box::new(self.node.clone()), Box::new(other.node.clone())),
        }
    }

    /// Fallback: if self produces score == 0, try other.
    fn fallback(&self, other: &PyStepDef) -> Self {
        Self {
            node: StepNode::Fallback(Box::new(self.node.clone()), Box::new(other.node.clone())),
        }
    }

    /// Retry up to N times until score >= target.
    #[pyo3(signature = (n=3, target=0.9))]
    fn retry(&self, n: u32, target: f64) -> Self {
        Self {
            node: StepNode::Retry(Box::new(self.node.clone()), n, target),
        }
    }

    /// Transform the output text with a Python callable.
    fn map(&self, func: PyObject) -> Self {
        Self {
            node: StepNode::Map(Box::new(self.node.clone()), func),
        }
    }

    /// Execute this step with the given input text.
    fn run(&self, input: &str) -> PyResult<PyStepResult> {
        let step = self.node.materialize();
        let output = kkachi::recursive::block_on(step.run_dyn(input))
            .map_err(|e| crate::error::KkachiError(e))?;
        Ok(PyStepResult {
            text: output.text,
            score: output.score,
            tokens: output.tokens,
        })
    }

    fn __repr__(&self) -> String {
        fn describe(node: &StepNode) -> String {
            match node {
                StepNode::Fn { name, .. } | StepNode::FnScored { name, .. } => {
                    format!("step(\"{}\")", name)
                }
                StepNode::Then(a, b) => format!("{}.then({})", describe(a), describe(b)),
                StepNode::Race(a, b) => format!("{}.race({})", describe(a), describe(b)),
                StepNode::Par(a, b) => format!("{}.par({})", describe(a), describe(b)),
                StepNode::Fallback(a, b) => {
                    format!("{}.fallback({})", describe(a), describe(b))
                }
                StepNode::Retry(inner, n, target) => {
                    format!("{}.retry({}, {:.1})", describe(inner), n, target)
                }
                StepNode::Map(inner, _) => format!("{}.map(...)", describe(inner)),
            }
        }
        format!("StepDef({})", describe(&self.node))
    }
}

// ============================================================================
// Entry point functions
// ============================================================================

/// Create a step from a Python callable.
///
/// The callable should take a single string and return a string.
/// The score will be 1.0 for all outputs.
///
/// # Python Example
///
/// ```python
/// upper = step_fn("upper", lambda s: s.upper())
/// result = upper.run("hello")
/// assert result.text == "HELLO"
/// assert result.score == 1.0
/// ```
#[pyfunction]
#[pyo3(name = "step_fn")]
pub fn py_step_fn(name: String, func: PyObject) -> PyStepDef {
    PyStepDef {
        node: StepNode::Fn { name, func },
    }
}

/// Create a scored step from a Python callable.
///
/// The callable should take a single string and return a tuple `(str, float)`.
///
/// # Python Example
///
/// ```python
/// check = step_scored("check", lambda s: (s, 1.0 if len(s) > 10 else 0.0))
/// result = check.run("hello world!")
/// assert result.score == 1.0
/// ```
#[pyfunction]
#[pyo3(name = "step_scored")]
pub fn py_step_scored(name: String, func: PyObject) -> PyStepDef {
    PyStepDef {
        node: StepNode::FnScored { name, func },
    }
}

/// Run multiple steps concurrently on the same input, returning all results.
///
/// # Python Example
///
/// ```python
/// results = run_all_steps("hello", [step_a, step_b, step_c])
/// for r in results:
///     print(f"{r.text}: {r.score}")
/// ```
#[pyfunction]
#[pyo3(name = "run_all_steps")]
pub fn py_run_all_steps(
    input: String,
    steps: Vec<PyRef<'_, PyStepDef>>,
) -> PyResult<Vec<PyStepResult>> {
    let materialized: Vec<Box<dyn DynStep>> = steps.iter().map(|s| s.node.materialize()).collect();
    let dyn_refs: Vec<&dyn DynStep> = materialized.iter().map(|s| s.as_ref()).collect();

    let results = kkachi::recursive::block_on(kkachi::recursive::step::run_all(&input, &dyn_refs));

    results
        .into_iter()
        .map(|r| {
            let output = r.map_err(|e| crate::error::KkachiError(e))?;
            Ok(PyStepResult {
                text: output.text,
                score: output.score,
                tokens: output.tokens,
            })
        })
        .collect()
}
