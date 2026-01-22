// Copyright (c) 2025 Lituus-io <spicyzhug@gmail.com>
// Author: terekete
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for critic types.
//!
//! Note: ToolType is deprecated. Use Cli and CliPipeline instead for
//! composing custom validators.

use pyo3::prelude::*;

/// Tool types for built-in CLI critics.
///
/// DEPRECATED: Use `Cli` and `CliPipeline` to compose your own validators.
///
/// Example migration:
/// ```python
/// # Old (deprecated):
/// result = Kkachi.refine("q -> code").critic_rust().run(...)
///
/// # New (recommended):
/// validator = CliPipeline() \
///     .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \
///     .stage("compile", Cli("rustc").args(["--emit=metadata"]).required()) \
///     .file_ext("rs")
/// result = Kkachi.refine("q -> code").validate(validator).run(...)
/// ```
#[pyclass(name = "ToolType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyToolType {
    /// Rust (cargo check, cargo test, cargo clippy)
    Rust = 0,
    /// Python (python -m py_compile, pytest, ruff)
    Python = 1,
    /// Terraform (terraform fmt, validate, plan)
    Terraform = 2,
    /// Pulumi (pulumi preview, policy validate)
    Pulumi = 3,
    /// Kubernetes (kubectl apply --dry-run)
    Kubernetes = 4,
    /// JavaScript/TypeScript (tsc, eslint)
    JavaScript = 5,
    /// Go (go build, go test, go vet)
    Go = 6,
}

#[pymethods]
impl PyToolType {
    fn __repr__(&self) -> String {
        match self {
            PyToolType::Rust => "ToolType.Rust".to_string(),
            PyToolType::Python => "ToolType.Python".to_string(),
            PyToolType::Terraform => "ToolType.Terraform".to_string(),
            PyToolType::Pulumi => "ToolType.Pulumi".to_string(),
            PyToolType::Kubernetes => "ToolType.Kubernetes".to_string(),
            PyToolType::JavaScript => "ToolType.JavaScript".to_string(),
            PyToolType::Go => "ToolType.Go".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            PyToolType::Rust => "rust".to_string(),
            PyToolType::Python => "python".to_string(),
            PyToolType::Terraform => "terraform".to_string(),
            PyToolType::Pulumi => "pulumi".to_string(),
            PyToolType::Kubernetes => "kubernetes".to_string(),
            PyToolType::JavaScript => "javascript".to_string(),
            PyToolType::Go => "go".to_string(),
        }
    }
}
