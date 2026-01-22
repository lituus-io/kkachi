// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the Kkachi LLM optimization library.
//!
//! This crate provides PyO3-based Python bindings for the core Kkachi functionality,
//! enabling Python users to leverage high-performance recursive language prompting.
//!
//! ## Usage
//!
//! ```python
//! from kkachi import Kkachi, Cli, CliPipeline
//!
//! # Compose your own validator
//! validator = CliPipeline() \
//!     .stage("syntax", Cli("python").args(["-m", "py_compile"]).required()) \
//!     .stage("lint", Cli("ruff").args(["check"])) \
//!     .file_ext("py")
//!
//! # Run refinement
//! result = Kkachi.refine("question -> code") \
//!     .domain("python") \
//!     .validate(validator) \
//!     .max_iterations(5) \
//!     .run("Write a URL parser", generate_fn)
//!
//! print(f"Score: {result.score}")
//! print(f"Answer: {result.answer}")
//! ```

use pyo3::prelude::*;

mod builder;
mod critic;
mod similarity;
mod types;
mod validator;

use builder::*;
use critic::*;
use similarity::*;
use types::*;
use validator::*;

/// Python module for Kkachi.
#[pymodule]
fn _kkachi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main entry point
    m.add_class::<PyKkachi>()?;

    // Builder
    m.add_class::<PyRefineBuilder>()?;

    // Result types
    m.add_class::<PyRefinementResult>()?;
    m.add_class::<PyRefineResult>()?;

    // Tool types (deprecated, use Cli/CliPipeline instead)
    m.add_class::<PyToolType>()?;

    // CLI Validators
    m.add_class::<PyCli>()?;
    m.add_class::<PyCliPipeline>()?;

    // Similarity
    m.add_class::<PySimilarityWeights>()?;

    // Few-shot config
    m.add_class::<PyFewShotConfig>()?;

    // VectorStore types
    m.add_class::<PyVectorSearchResult>()?;
    m.add_class::<PyInMemoryVectorStore>()?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
