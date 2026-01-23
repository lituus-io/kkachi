// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Recursive Language Prompting Module
//!
//! Enables iterative refinement where an LLM builds upon its previous responses
//! through feedback loops. Creates training-like convergence where output quality
//! improves over iterations.
//!
//! ## Key Features
//!
//! - **Self-Refinement**: LLM critiques and improves its own output
//! - **Generic Validation**: Composable validators without dynamic dispatch
//! - **Memory/RAG**: Built-in retrieval-augmented generation
//! - **Zero-Copy Design**: Lifetimes over Arc, GATs over async_trait
//!
//! ## Example
//!
//! ```ignore
//! use kkachi::recursive::prelude::*;
//!
//! let llm = MockLlm::new(|prompt, _| "fn add(a: i32, b: i32) -> i32 { a + b }".to_string());
//!
//! // Simple refinement with validation
//! let code = refine(&llm, "Write an add function")
//!     .validate(checks().require("fn ").require("->"))
//!     .max_iter(5)
//!     .go();
//!
//! // CLI validation pipeline
//! let validator = cli("rustfmt").arg("--check")
//!     .then("rustc").args(&["--emit=metadata"]).required()
//!     .ext("rs");
//!
//! let code = refine(&llm, "Write a parser")
//!     .validate(validator)
//!     .go();
//! ```

// DuckDB shim module - re-exports from either system or bundled crate
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub(crate) mod db {
    #[cfg(feature = "storage")]
    pub use duckdb::*;

    #[cfg(all(feature = "storage-bundled", not(feature = "storage")))]
    pub use duckdb_bundled::*;
}

// Core modules
pub mod checks;
pub mod cli;
pub mod compose;
pub mod llm;
pub mod memory;
pub mod refine;
pub mod result;
pub mod rewrite;
pub mod semantic;
pub mod validate;

// DSPy-style modules
pub mod agent;
pub mod best_of;
pub mod ensemble;
pub mod executor;
pub mod program;
pub mod reason;
pub mod tool;

// Prelude for convenient imports
pub mod prelude;

// Keep template module (minimal changes)
pub mod template;

// Re-exports from new modules
pub use checks::{checks, Check, CheckKind, Checks};
pub use cli::{cli, Cli, CliCapture, CliExecutor};
pub use compose::{all, any, All, And, Any, Or, ValidateExt};
pub use llm::{BoxedLlm, FailingLlm, IterativeMockLlm, Llm, LmOutput, MockLlm};
pub use memory::{
    cosine_similarity, memory, mmr_select, Document, Embedder, HashEmbedder, LinearIndex, Memory,
    Recall, VectorIndex,
};

// Feature-gated exports
#[cfg(feature = "embeddings-onnx")]
pub use memory::{OnnxEmbedder, OnnxEmbedderError};
#[cfg(feature = "hnsw")]
pub use memory::HnswIndex;
pub use refine::{refine, Config, Refine};
pub use result::{Compiled, ContextId, Correction, Example, Iteration, OptimizedPrompt, RefineResult, StopReason};
pub use semantic::{semantic, SemanticBuilder, SemanticValidator};
pub use rewrite::{extract_all_code, extract_code, extract_section, rewrite, Rewrite};
pub use validate::{AlwaysFail, BoolValidator, FnValidator, NoValidation, Score, ScoreValidator, Validate};

// Template re-exports
pub use template::{FormatSpec, FormatType, JsonSchema, PromptTone, Template, TemplateExample, TemplateOptions, ToneModifiers};

// DSPy-style module re-exports
pub use agent::{agent, Agent, AgentConfig, AgentResult, Step};
pub use best_of::{best_of, BestOf, BestOfConfig, BestOfResult, CandidatePool, DefaultScorer, FnScorer, PoolStats, ScoredCandidate, Scorer};
pub use ensemble::{ensemble, Aggregate, ChainResult, ConsensusPool, Ensemble, EnsembleConfig, EnsembleResult};
pub use executor::{bash_executor, node_executor, python_executor, ruby_executor, CodeExecutor, ExecutionResult, ProcessExecutor};
pub use program::{program, Program, ProgramConfig, ProgramResult};
pub use reason::{reason, Reason, ReasonConfig, ReasonResult};
pub use tool::{tool, AsyncFnTool, FnTool, MockTool, Tool, ToolBuilder};
