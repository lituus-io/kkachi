// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for CLI validators.
//!
//! Uses `PyRefMut`-based builders to avoid cloning the inner `Cli` struct
//! on every builder method call.

use pyo3::prelude::*;

use kkachi::recursive::{cli, Cli, CliCapture, Validate};

/// CLI capture result exposed to Python.
#[pyclass(name = "CliCapture")]
#[derive(Clone)]
pub struct PyCliCapture {
    /// The stage name.
    #[pyo3(get)]
    pub stage: String,
    /// Standard output from the command.
    #[pyo3(get)]
    pub stdout: String,
    /// Standard error from the command.
    #[pyo3(get)]
    pub stderr: String,
    /// Whether the command succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Exit code if available.
    #[pyo3(get)]
    pub exit_code: Option<i32>,
    /// Duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl PyCliCapture {
    fn __repr__(&self) -> String {
        format!(
            "CliCapture(stage='{}', success={}, exit_code={:?})",
            self.stage, self.success, self.exit_code
        )
    }
}

impl From<&CliCapture> for PyCliCapture {
    fn from(c: &CliCapture) -> Self {
        Self {
            stage: c.stage.clone(),
            stdout: c.stdout.clone(),
            stderr: c.stderr.clone(),
            success: c.success,
            exit_code: c.exit_code,
            duration_ms: c.duration_ms,
        }
    }
}

/// A CLI command validator with fluent API.
///
/// Uses `PyRefMut` to avoid cloning on each builder call.
///
/// Example:
/// ```python
/// validator = CliValidator("rustfmt") \
///     .args(["--check"]) \
///     .weight(0.1) \
///     .required() \
///     .then("rustc") \
///     .args(["--emit=metadata"]) \
///     .ext("rs")
/// ```
#[pyclass(name = "CliValidator")]
#[derive(Clone)]
pub struct PyCliValidator {
    inner: Cli,
}

#[pymethods]
impl PyCliValidator {
    /// Create a new CLI validator with the given command.
    #[new]
    fn new(command: String) -> Self {
        Self {
            inner: cli(&command),
        }
    }

    /// Add a single argument.
    fn arg(mut self_: PyRefMut<'_, Self>, arg: String) -> Py<Self> {
        self_.inner.push_arg(&arg);
        self_.into()
    }

    /// Add multiple arguments.
    fn args(mut self_: PyRefMut<'_, Self>, args: Vec<String>) -> Py<Self> {
        let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        self_.inner.push_args(&args_refs);
        self_.into()
    }

    /// Set the weight for this validator stage (0.0 to 1.0).
    fn weight(mut self_: PyRefMut<'_, Self>, weight: f64) -> Py<Self> {
        self_.inner.set_weight(weight);
        self_.into()
    }

    /// Mark this stage as required (failure stops the pipeline).
    fn required(mut self_: PyRefMut<'_, Self>) -> Py<Self> {
        self_.inner.set_required();
        self_.into()
    }

    /// Chain another command after this one.
    fn then(mut self_: PyRefMut<'_, Self>, command: String) -> Py<Self> {
        self_.inner.push_stage(&command);
        self_.into()
    }

    /// Set the file extension for temp files.
    fn ext(mut self_: PyRefMut<'_, Self>, ext: String) -> Py<Self> {
        self_.inner.set_ext(&ext);
        self_.into()
    }

    /// Set an environment variable for command execution.
    fn env(mut self_: PyRefMut<'_, Self>, key: String, value: String) -> Py<Self> {
        self_.inner.push_env(&key, &value);
        self_.into()
    }

    /// Inherit an environment variable from the current process.
    fn env_from(mut self_: PyRefMut<'_, Self>, key: String) -> Py<Self> {
        self_.inner.push_env_from(&key);
        self_.into()
    }

    /// Set the working directory for command execution.
    fn workdir(mut self_: PyRefMut<'_, Self>, path: String) -> Py<Self> {
        self_.inner.set_workdir(&path);
        self_.into()
    }

    /// Set the timeout in seconds.
    fn timeout(mut self_: PyRefMut<'_, Self>, secs: u64) -> Py<Self> {
        self_.inner.set_timeout(secs);
        self_.into()
    }

    /// Enable output capture.
    fn capture(mut self_: PyRefMut<'_, Self>) -> Py<Self> {
        self_.inner.set_capture();
        self_.into()
    }

    /// Compose with AND semantics (both must pass).
    #[pyo3(name = "and_")]
    fn and_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<crate::compose::PyValidator> {
        use crate::compose::{extract_validator_node, PyValidator, ValidatorNode};
        let self_node = ValidatorNode::Cli(self.inner.clone());
        let other_node = extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::And(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Compose with OR semantics (at least one must pass).
    #[pyo3(name = "or_")]
    fn or_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<crate::compose::PyValidator> {
        use crate::compose::{extract_validator_node, PyValidator, ValidatorNode};
        let self_node = ValidatorNode::Cli(self.inner.clone());
        let other_node = extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::Or(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Validate text and return a ScoreResult.
    fn validate(&self, text: String) -> crate::compose::PyScoreResult {
        crate::compose::PyScoreResult::from(self.inner.validate(&text))
    }

    /// Validate text and return both the ScoreResult and captured outputs.
    fn validate_with_captures(
        &self,
        text: String,
    ) -> (crate::compose::PyScoreResult, Vec<PyCliCapture>) {
        let score = self.inner.validate(&text);
        let captures = self
            .inner
            .get_captures()
            .iter()
            .map(PyCliCapture::from)
            .collect();
        (crate::compose::PyScoreResult::from(score), captures)
    }

    /// Get all captured outputs from previous validations.
    ///
    /// Returns captures collected during `reason().go()` or manual
    /// `.validate()` calls. Captures are shared across composed validators
    /// so they persist after `.and_()` / `.or_()` composition.
    fn get_captures(&self) -> Vec<PyCliCapture> {
        self.inner
            .get_captures()
            .iter()
            .map(PyCliCapture::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        "CliValidator(...)".to_string()
    }
}

impl PyCliValidator {
    pub fn into_inner(self) -> Cli {
        self.inner
    }

    pub fn inner_ref(&self) -> &Cli {
        &self.inner
    }
}
