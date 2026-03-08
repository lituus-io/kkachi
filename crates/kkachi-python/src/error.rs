// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Centralized Rust→Python error conversion.
//!
//! Provides a newtype wrapper around `kkachi::error::Error` that implements
//! `From<KkachiError> for PyErr`, enabling `?` via the `Into<PyErr>` conversion.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

use kkachi::error::Error;

/// Newtype wrapper for orphan-rule-safe `From<KkachiError> for PyErr`.
pub struct KkachiError(pub Error);

/// Extension trait to convert `Result<T, kkachi::Error>` to `PyResult<T>`.
pub trait IntoPyResult<T> {
    fn into_py(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for std::result::Result<T, Error> {
    fn into_py(self) -> pyo3::PyResult<T> {
        self.map_err(|e| KkachiError(e).into())
    }
}

impl From<Error> for KkachiError {
    fn from(e: Error) -> Self {
        KkachiError(e)
    }
}

impl From<KkachiError> for PyErr {
    fn from(e: KkachiError) -> PyErr {
        let err = e.0;
        match &err {
            // Memory errors already have operation/reason/suggestion
            Error::Memory { .. } => PyRuntimeError::new_err(format!("[kkachi] {err}")),

            // Storage errors with DB guidance
            Error::Storage(msg) => PyRuntimeError::new_err(format!(
                "[kkachi] Storage error: {msg}. \
                 Hint: ensure kkachi was installed with storage support \
                 (pip install kkachi[storage]) and the database path is accessible."
            )),

            // Validation errors → ValueError
            Error::Validation(msg) => PyValueError::new_err(format!(
                "[kkachi] Invalid input: {msg}. \
                 Check the parameter types and value ranges in the documentation."
            )),

            Error::Signature(msg) => PyValueError::new_err(format!(
                "[kkachi] Signature error: {msg}. \
                 Verify your input/output field names match the expected signature."
            )),

            Error::Field(msg) => PyValueError::new_err(format!(
                "[kkachi] Field error: {msg}. \
                 Check that all required fields are provided and correctly named."
            )),

            // Module errors → RuntimeError with retry suggestion
            Error::Module(msg) => PyRuntimeError::new_err(format!(
                "[kkachi] Module execution failed: {msg}. \
                 If this is an LLM error, check your API key and network. \
                 Use .with_retry(3) for transient failures."
            )),

            // Optimization errors with score context
            Error::Optimization(details) => PyRuntimeError::new_err(format!(
                "[kkachi] Optimization did not converge: score={:.3} (target={:.3}) after {} iterations. \
                 {}. \
                 Hint: try increasing max_iterations, lowering the threshold, \
                 or providing more diverse examples.",
                details.score,
                details.threshold,
                details.iterations,
                details.feedback.as_deref().unwrap_or("No feedback available")
            )),

            // Parse errors with format guidance
            Error::Parse(msg) => PyValueError::new_err(format!(
                "[kkachi] Parse error: {msg}. \
                 Ensure the input is valid (JSON, YAML, or template syntax)."
            )),

            // Catch-all
            _ => PyRuntimeError::new_err(format!("[kkachi] {err}")),
        }
    }
}
