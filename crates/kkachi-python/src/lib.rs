// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for Kkachi - Recursive Language Prompting library.
//!
//! ## Example
//!
//! ```python
//! from kkachi import refine, ApiLlm, CliValidator, Checks, Memory
//!
//! # Initialize LLM
//! llm = ApiLlm.from_env()
//!
//! # Pattern-based validation
//! validator = Checks().require("fn ").forbid(".unwrap()").min_len(50)
//!
//! # CLI-based validation
//! cli_validator = CliValidator("rustfmt").args(["--check"]).then("rustc").ext("rs")
//!
//! # Run refinement
//! result = refine(llm, "Write a URL parser") \
//!     .validate(cli_validator) \
//!     .max_iter(5) \
//!     .target(0.9) \
//!     .go()
//!
//! # Memory/RAG
//! mem = Memory()
//! mem.add("Example content")
//! results = mem.search("query", 3)
//! ```

use pyo3::prelude::*;

mod checks;
mod compose;
mod dspy;
mod jinja;
mod llm;
mod rewrite;
mod semantic;
mod template;
mod types;
mod validator;

use checks::*;
use compose::*;
use dspy::*;
use jinja::*;
use llm::*;
use rewrite::*;
use semantic::*;
use template::*;
use types::*;
use validator::*;

/// Python module for Kkachi.
#[pymodule]
fn _kkachi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // LLM implementations
    m.add_class::<PyApiLlm>()?;

    // Result types
    m.add_class::<PyRefineResult>()?;

    // Memory/RAG types
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRecall>()?;

    // Validators
    m.add_class::<PyCliValidator>()?;
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

    // DSPy-style modules - Builders
    m.add_class::<PyRefineBuilder>()?;
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

    // DSPy-style modules - Entry point functions
    m.add_function(wrap_pyfunction!(py_refine, m)?)?;
    m.add_function(wrap_pyfunction!(py_reason, m)?)?;
    m.add_function(wrap_pyfunction!(py_best_of, m)?)?;
    m.add_function(wrap_pyfunction!(py_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(py_agent, m)?)?;
    m.add_function(wrap_pyfunction!(py_program, m)?)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
