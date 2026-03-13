// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for Pipeline and ConcurrentRunner.
//!
//! Provides Python access to:
//! - `pipeline()` - Pipeline composition builder
//! - `concurrent()` - Concurrent pipeline execution

use pyo3::prelude::*;

use kkachi::recursive::{checks, pipeline};

use crate::compose::{extract_validator_node, ValidatorNode};
use crate::defaults::PyDefaults;
use crate::dspy::PyCallableLlm;

// ============================================================================
// Pipeline Step Definitions
// ============================================================================

/// Internal representation of a pipeline step for Python.
enum PyPipelineStepDef {
    Refine {
        validator: Option<ValidatorNode>,
        max_iter: u32,
        target: f64,
    },
    Extract {
        lang: String,
    },
    BestOf {
        n: usize,
    },
    Ensemble {
        n: usize,
    },
    Reason,
    Map {
        func: PyObject,
    },
}

impl Clone for PyPipelineStepDef {
    fn clone(&self) -> Self {
        match self {
            Self::Refine {
                validator,
                max_iter,
                target,
            } => Self::Refine {
                validator: validator.clone(),
                max_iter: *max_iter,
                target: *target,
            },
            Self::Extract { lang } => Self::Extract { lang: lang.clone() },
            Self::BestOf { n } => Self::BestOf { n: *n },
            Self::Ensemble { n } => Self::Ensemble { n: *n },
            Self::Reason => Self::Reason,
            Self::Map { func } => Self::Map {
                func: Python::with_gil(|py| func.clone_ref(py)),
            },
        }
    }
}

// ============================================================================
// Pipeline Builder
// ============================================================================

/// Pipeline composition builder.
///
/// Chains multiple operations together, feeding each step's output
/// into the next.
///
/// # Python Example
///
/// ```python
/// result = pipeline(llm, "Write an add function") \
///     .refine(checks.require("fn ")) \
///     .extract("rust") \
///     .best_of(3) \
///     .go()
/// ```
#[pyclass(name = "PipelineBuilder")]
pub struct PyPipelineBuilder {
    llm: PyObject,
    prompt: String,
    steps: Vec<PyPipelineStepDef>,
    defaults: Option<kkachi::recursive::Defaults>,
    skill_text: Option<String>,
}

impl Clone for PyPipelineBuilder {
    fn clone(&self) -> Self {
        Self {
            llm: Python::with_gil(|py| self.llm.clone_ref(py)),
            prompt: self.prompt.clone(),
            steps: self.steps.clone(),
            defaults: self.defaults.clone(),
            skill_text: self.skill_text.clone(),
        }
    }
}

#[pymethods]
impl PyPipelineBuilder {
    /// Add a refinement step with a validator.
    #[pyo3(signature = (validator, max_iter=5, target=1.0))]
    fn refine(&self, validator: &Bound<'_, PyAny>, max_iter: u32, target: f64) -> PyResult<Self> {
        let node = extract_validator_node(validator)?;
        let mut new = self.clone();
        new.steps.push(PyPipelineStepDef::Refine {
            validator: Some(node),
            max_iter,
            target,
        });
        Ok(new)
    }

    /// Add a code extraction step.
    fn extract(&self, lang: &str) -> Self {
        let mut new = self.clone();
        new.steps.push(PyPipelineStepDef::Extract {
            lang: lang.to_string(),
        });
        new
    }

    /// Add a best-of-N step.
    #[pyo3(signature = (n=3))]
    fn best_of(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.steps.push(PyPipelineStepDef::BestOf { n });
        new
    }

    /// Add an ensemble step.
    #[pyo3(signature = (n=3))]
    fn ensemble(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.steps.push(PyPipelineStepDef::Ensemble { n });
        new
    }

    /// Add a chain-of-thought reasoning step.
    fn reason(&self) -> Self {
        let mut new = self.clone();
        new.steps.push(PyPipelineStepDef::Reason);
        new
    }

    /// Add a map/transform step with a Python callable.
    fn map(&self, func: PyObject) -> Self {
        let mut new = self.clone();
        new.steps.push(PyPipelineStepDef::Map { func });
        new
    }

    /// Set runtime defaults.
    fn defaults(&self, defaults: &PyDefaults) -> Self {
        let mut new = self.clone();
        new.defaults = Some(defaults.inner.clone());
        new
    }

    /// Attach a skill.
    fn skill(&self, skill: &crate::skill::PySkill) -> Self {
        let mut new = self.clone();
        let rendered = skill.to_skill().render();
        new.skill_text = if rendered.is_empty() {
            None
        } else {
            Some(rendered)
        };
        new
    }

    /// Execute the pipeline and return the result.
    fn go(&self) -> PyResult<PyPipelineResult> {
        let llm = Python::with_gil(|py| PyCallableLlm::new_from_ref(py, &self.llm));
        let prompt = self.prompt.clone();
        let steps = self.steps.clone();

        let mut p = pipeline(&llm, &prompt);

        // Add each step
        for step_def in steps {
            p = apply_step(p, step_def);
        }

        let result = p.go();

        Ok(PyPipelineResult {
            output: result.output,
            total_tokens: result.total_tokens,
            elapsed_ms: result.elapsed.as_millis() as u64,
            steps_count: result.steps.len(),
        })
    }
}

// ============================================================================
// Pipeline Result
// ============================================================================

/// Result of a pipeline execution.
#[pyclass(name = "PipelineResult")]
#[derive(Clone)]
pub struct PyPipelineResult {
    /// The final output.
    #[pyo3(get)]
    pub output: String,
    /// Total tokens used.
    #[pyo3(get)]
    pub total_tokens: u32,
    /// Elapsed time in milliseconds.
    #[pyo3(get)]
    pub elapsed_ms: u64,
    /// Number of steps executed.
    #[pyo3(get)]
    pub steps_count: usize,
}

#[pymethods]
impl PyPipelineResult {
    fn __repr__(&self) -> String {
        format!(
            "PipelineResult(steps={}, tokens={}, elapsed={}ms)",
            self.steps_count, self.total_tokens, self.elapsed_ms
        )
    }
}

// ============================================================================
// Concurrent Runner
// ============================================================================

/// A single task for the concurrent runner.
struct PyConcurrentTaskDef {
    label: String,
    steps: Vec<PyPipelineStepDef>,
    prompt: String,
}

impl Clone for PyConcurrentTaskDef {
    fn clone(&self) -> Self {
        Self {
            label: self.label.clone(),
            steps: self.steps.clone(),
            prompt: self.prompt.clone(),
        }
    }
}

/// Runs multiple pipelines concurrently on a shared LLM.
///
/// # Python Example
///
/// ```python
/// results = concurrent(llm) \
///     .task("rust", "Write in Rust", lambda p: p.refine(checks).extract("rust")) \
///     .task("python", "Write in Python", lambda p: p.refine(checks).extract("python")) \
///     .go()
///
/// for r in results:
///     print(f"{r.label}: {r.output[:50]}...")
/// ```
#[pyclass(name = "ConcurrentRunnerBuilder")]
pub struct PyConcurrentRunnerBuilder {
    llm: PyObject,
    tasks: Vec<PyConcurrentTaskDef>,
    max_concurrency: usize,
}

impl Clone for PyConcurrentRunnerBuilder {
    fn clone(&self) -> Self {
        Self {
            llm: Python::with_gil(|py| self.llm.clone_ref(py)),
            tasks: self.tasks.clone(),
            max_concurrency: self.max_concurrency,
        }
    }
}

#[pymethods]
impl PyConcurrentRunnerBuilder {
    /// Add a task with a label, prompt, and pipeline builder callback.
    ///
    /// The callback receives a PipelineBuilder and should return a configured one.
    fn task(&self, label: &str, prompt: &str, builder_fn: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Create a temporary PipelineBuilder, call the Python function to configure it
        let temp_builder = PyPipelineBuilder {
            llm: self.llm.clone_ref(builder_fn.py()),
            prompt: prompt.to_string(),
            steps: Vec::new(),
            defaults: None,
            skill_text: None,
        };

        let configured: PyPipelineBuilder = Python::with_gil(|py| {
            let temp = Py::new(py, temp_builder)?;
            let result = builder_fn.call1((temp,))?;
            result.extract::<PyPipelineBuilder>()
        })?;

        let mut new = self.clone();
        new.tasks.push(PyConcurrentTaskDef {
            label: label.to_string(),
            steps: configured.steps,
            prompt: prompt.to_string(),
        });
        Ok(new)
    }

    /// Add a simple task (no pipeline configuration).
    fn simple_task(&self, label: &str, prompt: &str) -> Self {
        let mut new = self.clone();
        new.tasks.push(PyConcurrentTaskDef {
            label: label.to_string(),
            steps: Vec::new(),
            prompt: prompt.to_string(),
        });
        new
    }

    /// Set maximum concurrency.
    fn max_concurrency(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.max_concurrency = n;
        new
    }

    /// Execute all tasks concurrently.
    ///
    /// Note: The ConcurrentRunner requires that the pipeline builder closures
    /// don't outlive the LLM reference. We achieve this by building and running
    /// everything within a single scope where the LLM is alive.
    fn go(&self) -> PyResult<Vec<PyConcurrentTaskResult>> {
        let llm = Python::with_gil(|py| PyCallableLlm::new_from_ref(py, &self.llm));
        let tasks = self.tasks.clone();

        // Build pipelines directly and run them, since ConcurrentRunner
        // requires closures that borrow the LLM.
        // Each task gets materialized into a Pipeline and executed.
        let results = run_concurrent_tasks(&llm, tasks, self.max_concurrency);

        Ok(results)
    }
}

/// Run concurrent tasks using ConcurrentRunner.
///
/// This is a separate function to ensure proper lifetime scoping: the `llm`
/// reference is valid for the entire duration of the concurrent run.
fn run_concurrent_tasks(
    llm: &PyCallableLlm,
    tasks: Vec<PyConcurrentTaskDef>,
    max_concurrency: usize,
) -> Vec<PyConcurrentTaskResult> {
    use kkachi::recursive::concurrent::ConcurrentRunner;

    let mut runner = ConcurrentRunner::new(llm);
    if max_concurrency > 0 {
        runner = runner.max_concurrency(max_concurrency);
    }

    for task_def in tasks {
        let prompt = task_def.prompt;
        let steps = task_def.steps;
        let label = task_def.label;

        runner = runner.task(&label, move |llm| {
            let mut p = kkachi::recursive::Pipeline::new_owned(llm, prompt);
            for step in steps {
                p = apply_step(p, step);
            }
            p
        });
    }

    let results = runner.go();

    results
        .into_iter()
        .map(|r| {
            let success = r.result.is_ok();
            let output = match r.result {
                Ok(pr) => pr.output,
                Err(_) => String::new(),
            };
            PyConcurrentTaskResult {
                label: r.label,
                output,
                success,
                elapsed_ms: r.elapsed.as_millis() as u64,
            }
        })
        .collect()
}

/// Apply a single step definition to a pipeline.
fn apply_step<'a, L: kkachi::recursive::Llm>(
    p: kkachi::recursive::Pipeline<'a, L>,
    step_def: PyPipelineStepDef,
) -> kkachi::recursive::Pipeline<'a, L> {
    match step_def {
        PyPipelineStepDef::Refine {
            validator,
            max_iter,
            target,
        } => {
            if let Some(node) = validator {
                let v = Python::with_gil(|py| node.materialize(py));
                p.refine_with(v, max_iter, target)
            } else {
                p.refine_with(checks().min_len(1), max_iter, target)
            }
        }
        PyPipelineStepDef::Extract { lang } => p.extract(&lang),
        PyPipelineStepDef::BestOf { n } => p.best_of(n),
        PyPipelineStepDef::Ensemble { n } => p.ensemble(n),
        PyPipelineStepDef::Reason => p.reason(),
        PyPipelineStepDef::Map { func } => p.map(move |s| {
            let s_clone = s.to_string();
            Python::with_gil(|py| {
                func.call1(py, (s_clone,))
                    .and_then(|r| r.extract::<String>(py))
                    .unwrap_or_else(|_| s.to_string())
            })
        }),
    }
}

/// Result of a single concurrent task.
#[pyclass(name = "ConcurrentTaskResult")]
#[derive(Clone)]
pub struct PyConcurrentTaskResult {
    /// Task label.
    #[pyo3(get)]
    pub label: String,
    /// Output text.
    #[pyo3(get)]
    pub output: String,
    /// Whether the task succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Elapsed time in milliseconds.
    #[pyo3(get)]
    pub elapsed_ms: u64,
}

#[pymethods]
impl PyConcurrentTaskResult {
    fn __repr__(&self) -> String {
        format!(
            "ConcurrentTaskResult(label='{}', success={}, elapsed={}ms)",
            self.label, self.success, self.elapsed_ms
        )
    }
}

// ============================================================================
// Entry point functions
// ============================================================================

/// Create a pipeline builder.
///
/// # Python Example
///
/// ```python
/// result = pipeline(llm, "Write code").refine(checks).extract("rust").go()
/// ```
#[pyfunction]
#[pyo3(name = "pipeline")]
pub fn py_pipeline(llm: PyObject, prompt: String) -> PyPipelineBuilder {
    PyPipelineBuilder {
        llm,
        prompt,
        steps: Vec::new(),
        defaults: None,
        skill_text: None,
    }
}

/// Create a concurrent runner builder.
///
/// # Python Example
///
/// ```python
/// results = concurrent(llm).task("a", "prompt", lambda p: p.reason()).go()
/// ```
#[pyfunction]
#[pyo3(name = "concurrent")]
pub fn py_concurrent(llm: PyObject) -> PyConcurrentRunnerBuilder {
    PyConcurrentRunnerBuilder {
        llm,
        tasks: Vec::new(),
        max_concurrency: 0,
    }
}
