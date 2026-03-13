// Copyright (c) 2024-2026 Lituus-io. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::useless_conversion,
    clippy::needless_pass_by_value,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::single_component_path_imports,
    clippy::upper_case_acronyms,
    clippy::wrong_self_convention,
    clippy::large_enum_variant,
    clippy::derivable_impls
)]

//! Python bindings for Kkachi - Recursive Language Prompting library.
//!
//! ## Example
//!
//! ```python
//! from kkachi import Kkachi, CliValidator, Checks, Memory
//!
//! # Pattern-based validation
//! validator = Checks().require("fn ").forbid(".unwrap()").min_len(50)
//!
//! # CLI-based validation
//! cli_validator = CliValidator("rustfmt").args(["--check"]).then("rustc").ext("rs")
//!
//! # Run refinement
//! result = Kkachi.refine("Write a URL parser") \
//!     .validate(cli_validator) \
//!     .max_iter(5) \
//!     .target(0.9) \
//!     .run(generate_fn)
//!
//! # Memory/RAG
//! mem = Memory()
//! mem.add("Example content")
//! results = mem.search("query", 3)
//! ```

use pyo3::prelude::*;

mod builder;
mod checks;
mod compose;
mod defaults;
mod dspy;
mod error;
mod evaluate_py;
mod jinja;
mod llm;
mod optimize_py;
mod pareto_py;
mod pipeline_py;
mod rewrite;
mod semantic;
pub(crate) mod skill;
mod step_def;
mod template;
mod types;
mod validator;

use builder::*;
use checks::*;
use compose::*;
use defaults::*;
use dspy::*;
use evaluate_py::*;
use jinja::*;
use llm::*;
use optimize_py::*;
use pareto_py::*;
use pipeline_py::*;
use rewrite::*;
use semantic::*;
use skill::*;
use step_def::*;
use template::*;
use types::*;
use validator::*;

/// Python module for Kkachi.
#[pymodule]
fn _kkachi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main entry point
    m.add_class::<PyKkachi>()?;

    // LLM implementations
    m.add_class::<PyApiLlm>()?;

    // Builder
    m.add_class::<PyRefineBuilder>()?;

    // Result types
    m.add_class::<PyRefineResult>()?;

    // Memory/RAG types
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRecall>()?;

    // Validators
    m.add_class::<PyCliValidator>()?;
    m.add_class::<PyCliCapture>()?;
    m.add_class::<PyChecks>()?;
    m.add_class::<PySemantic>()?;
    m.add_class::<PyValidator>()?;
    m.add_class::<PyScoreResult>()?;

    // Template system
    m.add_class::<PyTemplate>()?;
    m.add_class::<PyFormatType>()?;
    m.add_class::<PyPromptTone>()?;

    // Jinja template system
    m.add_class::<PyJinjaTemplate>()?;
    m.add_class::<PyJinjaFormatter>()?;

    // Rewrite utilities
    m.add_class::<PyRewrite>()?;
    m.add_function(wrap_pyfunction!(py_extract_code, m)?)?;
    m.add_function(wrap_pyfunction!(py_extract_all_code, m)?)?;

    // Runtime defaults
    m.add_class::<PyDefaults>()?;
    m.add_class::<PyDefaultAnnotation>()?;

    // Skills (persistent prompt context)
    m.add_class::<PySkill>()?;

    // Package result
    m.add_class::<PyPackageResult>()?;

    // DSPy-style modules - Builders
    m.add_class::<PyReasonBuilder>()?;
    m.add_class::<PyBestOfBuilder>()?;
    m.add_class::<PyEnsembleBuilder>()?;
    m.add_class::<PyAgentBuilder>()?;
    m.add_class::<PyProgramBuilder>()?;

    // DSPy-style modules - Result types
    m.add_class::<PyReasonResult>()?;
    m.add_class::<PyBestOfResult>()?;
    m.add_class::<PyScoredCandidate>()?;
    m.add_class::<PyPoolStats>()?;
    m.add_class::<PyCandidatePool>()?;
    m.add_class::<PyEnsembleResult>()?;
    m.add_class::<PyChainResult>()?;
    m.add_class::<PyConsensusPool>()?;
    m.add_class::<PyAgentResult>()?;
    m.add_class::<PyStep>()?;
    m.add_class::<PyProgramResult>()?;
    m.add_class::<PyExecutionResult>()?;

    // DSPy-style modules - Tool and Executor
    m.add_class::<PyToolDef>()?;
    m.add_class::<PyExecutor>()?;

    // Pipeline composition
    m.add_class::<PyPipelineBuilder>()?;
    m.add_class::<PyPipelineResult>()?;
    m.add_class::<PyConcurrentRunnerBuilder>()?;
    m.add_class::<PyConcurrentTaskResult>()?;

    // Multi-objective / Pareto optimization
    m.add_class::<PyObjective>()?;
    m.add_class::<PyMultiObjectiveBuilder>()?;
    m.add_class::<PyMultiObjectiveValidator>()?;
    m.add_class::<PyParetoCandidate>()?;
    m.add_class::<PyParetoFront>()?;
    m.add_class::<PyParetoRefineResult>()?;

    // Optimizer
    m.add_class::<PyDataset>()?;
    m.add_class::<PyOptimizeResult>()?;
    m.add_class::<PyOptimizerBuilder>()?;

    // Evaluation metrics
    m.add_class::<PyMetric>()?;

    // Step combinators
    m.add_class::<PyStepDef>()?;
    m.add_class::<PyStepResult>()?;

    // DSPy-style modules - Entry point functions
    m.add_function(wrap_pyfunction!(py_reason, m)?)?;
    m.add_function(wrap_pyfunction!(py_best_of, m)?)?;
    m.add_function(wrap_pyfunction!(py_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(py_agent, m)?)?;
    m.add_function(wrap_pyfunction!(py_program, m)?)?;
    m.add_function(wrap_pyfunction!(py_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(py_concurrent, m)?)?;
    m.add_function(wrap_pyfunction!(py_multi_objective, m)?)?;
    m.add_function(wrap_pyfunction!(py_refine_pareto, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimizer, m)?)?;
    m.add_function(wrap_pyfunction!(py_step_fn, m)?)?;
    m.add_function(wrap_pyfunction!(py_step_scored, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_all_steps, m)?)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
