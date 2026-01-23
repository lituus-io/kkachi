// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the Checks validator.

use pyo3::prelude::*;

use kkachi::recursive::{checks, Checks, Validate};

use crate::compose::{extract_validator_node, PyScoreResult, PyValidator, ValidatorNode};

/// A pattern-based validator with fluent API.
///
/// Example:
/// ```python
/// validator = Checks() \
///     .require("fn ") \
///     .require("->") \
///     .forbid(".unwrap()") \
///     .forbid("panic!") \
///     .min_len(50)
/// ```
#[pyclass(name = "Checks")]
#[derive(Clone)]
pub struct PyChecks {
    inner: Checks,
}

#[pymethods]
impl PyChecks {
    /// Create a new empty Checks validator.
    #[new]
    fn new() -> Self {
        Self { inner: checks() }
    }

    /// Require a pattern to be present in the output.
    fn require(&self, pattern: String) -> Self {
        Self {
            inner: self.inner.clone().require(&pattern),
        }
    }

    /// Forbid a pattern from appearing in the output.
    fn forbid(&self, pattern: String) -> Self {
        Self {
            inner: self.inner.clone().forbid(&pattern),
        }
    }

    /// Set minimum output length.
    fn min_len(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().min_len(n),
        }
    }

    /// Set maximum output length.
    fn max_len(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().max_len(n),
        }
    }

    /// Add a regex pattern for validation.
    fn regex(&self, pattern: String) -> Self {
        Self {
            inner: self.inner.clone().regex(&pattern),
        }
    }

    /// Compose with AND semantics (both must pass).
    #[pyo3(name = "and_")]
    fn and_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValidator> {
        let self_node = ValidatorNode::Checks(self.inner.clone());
        let other_node = extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::And(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Compose with OR semantics (at least one must pass).
    #[pyo3(name = "or_")]
    fn or_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValidator> {
        let self_node = ValidatorNode::Checks(self.inner.clone());
        let other_node = extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::Or(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Validate text and return a ScoreResult.
    fn validate(&self, text: &str) -> PyScoreResult {
        PyScoreResult::from(self.inner.validate(text))
    }

    fn __repr__(&self) -> String {
        "Checks(...)".to_string()
    }
}

impl PyChecks {
    pub fn into_inner(self) -> Checks {
        self.inner
    }

    pub fn inner_ref(&self) -> &Checks {
        &self.inner
    }
}
