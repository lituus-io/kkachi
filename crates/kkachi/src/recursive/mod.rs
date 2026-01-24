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
//! ```no_run
//! use kkachi::recursive::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let llm = CliLlm::new()?;
//!
//!     // Simple refinement with validation
//!     let result = refine(&llm, "Write an add function in Rust")
//!         .validate(checks().require_all(["fn ", "->"]).forbid("todo!"))
//!         .max_iter(5)
//!         .go()?;
//!
//!     // CLI validation pipeline
//!     let validator = cli("rustfmt").arg("--check")
//!         .then("rustc").args(&["--emit=metadata"]).required()
//!         .ext("rs");
//!
//!     let result = refine(&llm, "Write a Rust parser")
//!         .validate(validator)
//!         .go()?;
//!     Ok(())
//! }
//! ```

// DuckDB module (requires system DuckDB: brew install duckdb on macOS)
#[cfg(feature = "storage")]
pub(crate) mod db {
    pub use duckdb::*;
}

// Core modules
pub mod checks;
pub mod cli;
pub mod compose;
pub mod conversation;
pub mod formatter;
pub mod llm;
pub mod memory;
pub mod refine;
pub mod result;
pub mod rewrite;
pub mod semantic;
pub mod typed;
pub mod validate;

// API client (feature-gated)
#[cfg(feature = "api")]
pub mod api_client;

// LLM wrappers
pub mod cache;
pub mod rate_limit;
pub mod retry;

// Multi-objective optimization
pub mod pareto;

// DSPy-style modules
pub mod agent;
pub mod best_of;
pub mod ensemble;
pub mod executor;
pub mod optimize;
pub mod pipeline;
pub mod program;
pub mod reason;
pub mod tool;

// Prelude for convenient imports
pub mod prelude;

// Keep template module (minimal changes)
pub mod template;

// Re-exports from new modules
#[cfg(feature = "api")]
pub use api_client::{ApiLlm, Provider};
pub use checks::{checks, Check, CheckKind, Checks};
pub use cli::{cli, Cli, CliCapture, CliTool};
pub use compose::{all, any, All, And, Any, Or, ValidateExt};
pub use llm::{BoxedLlm, CliLlm, FailingLlm, IterativeMockLlm, Llm, LmOutput, MockLlm};
pub use memory::{
    cosine_similarity, memory, mmr_select, Document, Embedder, HashEmbedder, LinearIndex, Memory,
    Recall, VectorIndex,
};

// Feature-gated exports
pub use conversation::{Conversation, Message, Role};
pub use formatter::{FeedbackFormatter, PassthroughFormatter, PromptFormatter};
#[cfg(feature = "hnsw")]
pub use memory::HnswIndex;
#[cfg(feature = "embeddings-onnx")]
pub use memory::{OnnxEmbedder, OnnxEmbedderError};
pub use refine::{refine, Config, Refine};
pub use result::{
    Compiled, ContextId, Correction, Example, Iteration, OptimizedPrompt, RefineEvent,
    RefineResult, StopReason,
};
pub use rewrite::{extract_all_code, extract_code, extract_section, rewrite, Rewrite};
pub use semantic::{semantic, SemanticBuilder, SemanticValidator};
pub use typed::{
    extract_json, parse_output, typed, DefaultFormat, FormatInstruction, SchemaFormat,
    TypedValidator,
};
pub use validate::{
    AlwaysFail, BoolValidator, FnValidator, NoValidation, Score, ScoreValidator, Validate,
};

// Template re-exports
pub use template::{
    FormatSpec, FormatType, JsonSchema, PromptTone, Template, TemplateExample, TemplateOptions,
    ToneModifiers,
};

// DSPy-style module re-exports
pub use agent::{agent, Agent, AgentConfig, AgentResult, Step};
pub use best_of::{
    best_of, BestOf, BestOfConfig, BestOfResult, CandidatePool, DefaultScorer, FnScorer, PoolStats,
    ScoredCandidate, Scorer,
};
pub use cache::{CacheExt, CachedLlm};
pub use ensemble::{
    ensemble, Aggregate, ChainResult, ConsensusPool, Ensemble, EnsembleConfig, EnsembleResult,
};
pub use executor::{
    bash_executor, node_executor, python_executor, ruby_executor, CodeExecutor, ExecutionResult,
    ProcessExecutor,
};
pub use optimize::{Dataset, MetricFn, OptimizeResult, Optimizer, Strategy, TrainingExample};
pub use pareto::{
    multi_objective, refine_pareto, refine_pareto_sync, Direction, MultiObjective, MultiObjective2,
    MultiObjective3, MultiObjectiveBuilder, MultiObjectiveValidate, MultiScore, Objective,
    ObjectiveScore, ParetoCandidate, ParetoFront, ParetoRefineResult, ParetoScored, Scalarization,
};
pub use pipeline::{
    pipeline, BranchBuilder, FanOutBranchResult, FanOutCollector, MergeStrategy, Pipeline,
    PipelineEvent, PipelineResult, StepResult,
};
pub use program::{program, Program, ProgramConfig, ProgramResult};
pub use rate_limit::{RateLimitConfig, RateLimitExt, RateLimitedLlm};
pub use reason::{reason, Reason, ReasonConfig, ReasonResult};
pub use retry::{LlmExt, RetryConfig, RetryLlm};
pub use tool::{tool, AsyncFnTool, FnTool, MockTool, Tool, ToolBuilder};
