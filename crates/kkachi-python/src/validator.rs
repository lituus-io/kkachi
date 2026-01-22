// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for CLI validators (Cli and CliPipeline).

use pyo3::prelude::*;

use kkachi::recursive::{Cli, CliPipeline};

/// A CLI command validator.
///
/// Example:
/// ```python
/// validator = Cli("rustfmt") \
///     .args(["--check"]) \
///     .weight(0.1) \
///     .required()
/// ```
#[pyclass(name = "Cli")]
#[derive(Clone)]
pub struct PyCli {
    inner: Cli,
}

#[pymethods]
impl PyCli {
    /// Create a new CLI validator with the given command.
    #[new]
    fn new(command: String) -> Self {
        Self {
            inner: Cli::new(&command),
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
        Self {
            inner: self.inner.clone().args(args),
        }
    }

    /// Set the weight for this validator stage (0.0 to 1.0).
    fn weight(&self, weight: f32) -> Self {
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

    /// Read input from stdin instead of temp file.
    fn stdin(&self) -> Self {
        Self {
            inner: self.inner.clone().stdin(),
        }
    }

    /// Set the file extension for temp files.
    fn file_ext(&self, ext: String) -> Self {
        Self {
            inner: self.inner.clone().file_ext(&ext),
        }
    }

    /// Set an environment variable for command execution.
    fn env(&self, key: String, value: String) -> Self {
        Self {
            inner: self.inner.clone().env(&key, &value),
        }
    }

    /// Inherit an environment variable from the current process.
    fn env_inherit(&self, key: String) -> Self {
        Self {
            inner: self.inner.clone().env_inherit(&key),
        }
    }

    fn __repr__(&self) -> String {
        "Cli(...)".to_string()
    }
}

impl PyCli {
    pub fn into_inner(self) -> Cli {
        self.inner
    }
}

/// A pipeline of CLI validators with multiple stages.
///
/// Example:
/// ```python
/// pipeline = CliPipeline() \
///     .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \
///     .stage("compile", Cli("rustc").args(["--emit=metadata"]).required()) \
///     .stage("lint", Cli("cargo").args(["clippy"]).weight(0.3)) \
///     .file_ext("rs")
/// ```
#[pyclass(name = "CliPipeline")]
#[derive(Clone)]
pub struct PyCliPipeline {
    inner: CliPipeline,
}

#[pymethods]
impl PyCliPipeline {
    /// Create a new empty CLI pipeline.
    #[new]
    fn new() -> Self {
        Self {
            inner: CliPipeline::new(),
        }
    }

    /// Add a validation stage to the pipeline.
    fn stage(&self, name: String, cli: PyCli) -> Self {
        Self {
            inner: self.inner.clone().stage(&name, cli.into_inner()),
        }
    }

    /// Set the file extension for temp files used by all stages.
    fn file_ext(&self, ext: String) -> Self {
        Self {
            inner: self.inner.clone().file_ext(&ext),
        }
    }

    fn __repr__(&self) -> String {
        "CliPipeline(...)".to_string()
    }
}

impl PyCliPipeline {
    pub fn into_inner(self) -> CliPipeline {
        self.inner
    }
}
