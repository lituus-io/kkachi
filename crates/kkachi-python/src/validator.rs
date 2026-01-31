// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for CLI validators.

use pyo3::prelude::*;

use kkachi::recursive::{cli, Cli};

/// A CLI command validator with fluent API.
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
    fn arg(&self, arg: String) -> Self {
        Self {
            inner: self.inner.clone().arg(&arg),
        }
    }

    /// Add multiple arguments.
    fn args(&self, args: Vec<String>) -> Self {
        let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        Self {
            inner: self.inner.clone().args(&args_refs),
        }
    }

    /// Set the weight for this validator stage (0.0 to 1.0).
    fn weight(&self, weight: f64) -> Self {
        Self {
            inner: self.inner.clone().weight(weight),
        }
    }

    /// Mark this stage as required (failure stops the pipeline).
    fn required(&self) -> Self {
        Self {
            inner: self.inner.clone().required(),
        }
    }

    /// Chain another command after this one.
    fn then(&self, command: String) -> Self {
        Self {
            inner: self.inner.clone().then(&command),
        }
    }

    /// Set the file extension for temp files.
    fn ext(&self, ext: String) -> Self {
        Self {
            inner: self.inner.clone().ext(&ext),
        }
    }

    /// Set an environment variable for command execution.
    fn env(&self, key: String, value: String) -> Self {
        Self {
            inner: self.inner.clone().env(&key, &value),
        }
    }

    /// Inherit an environment variable from the current process.
    fn env_from(&self, key: String) -> Self {
        Self {
            inner: self.inner.clone().env_from(&key),
        }
    }

    /// Set the working directory for command execution.
    fn workdir(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().workdir(&path),
        }
    }

    /// Set the timeout in seconds.
    fn timeout(&self, secs: u64) -> Self {
        Self {
            inner: self.inner.clone().timeout(secs),
        }
    }

    /// Enable output capture.
    fn capture(&self) -> Self {
        Self {
            inner: self.inner.clone().capture(),
        }
    }

    /// Compose with AND semantics (both must pass).
    ///
    /// # Example
    ///
    /// ```python
    /// from kkachi import CliValidator, Checks
    ///
    /// cli = CliValidator("rustfmt").args(["--check"])
    /// checks = Checks().require("fn ")
    /// combined = cli.and_(checks)
    /// ```
    #[pyo3(name = "and_")]
    fn and_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<crate::compose::PyValidator> {
        use crate::compose::{extract_validator_node, PyValidator, ValidatorNode};
        let self_node = ValidatorNode::Cli(Box::new(self.inner.clone()));
        let other_node = extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::And(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Compose with OR semantics (at least one must pass).
    ///
    /// # Example
    ///
    /// ```python
    /// from kkachi import CliValidator, Checks
    ///
    /// cli = CliValidator("rustfmt").args(["--check"])
    /// checks = Checks().forbid("panic!")
    /// combined = cli.or_(checks)
    /// ```
    #[pyo3(name = "or_")]
    fn or_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<crate::compose::PyValidator> {
        use crate::compose::{extract_validator_node, PyValidator, ValidatorNode};
        let self_node = ValidatorNode::Cli(Box::new(self.inner.clone()));
        let other_node = extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::Or(Box::new(self_node), Box::new(other_node)),
        })
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
