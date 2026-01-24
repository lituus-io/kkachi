// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Public re-exports for convenient imports.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::prelude::*;
//!
//! // Now you have access to all common types and functions
//! let validator = checks().require("fn ");
//! ```

// Core refinement
pub use crate::recursive::refine::{refine, Config, Refine};

// Validation
pub use crate::recursive::checks::{checks, Check, CheckKind, Checks};
pub use crate::recursive::cli::{cli, Cli, CliCapture, CliTool};
pub use crate::recursive::compose::{all, any, All, And, Any, Or, ValidateExt};
pub use crate::recursive::validate::{
    AlwaysFail, BoolValidator, FnValidator, NoValidation, Score, ScoreValidator, Validate,
};

// Memory/RAG
pub use crate::recursive::memory::{
    cosine_similarity, memory, mmr_select, Document, Embedder, HashEmbedder, LinearIndex, Memory,
    Recall, VectorIndex,
};

// Feature-gated exports
#[cfg(feature = "api")]
pub use crate::recursive::api_client::{ApiLlm, Provider};
#[cfg(feature = "hnsw")]
pub use crate::recursive::memory::HnswIndex;
#[cfg(feature = "embeddings-onnx")]
pub use crate::recursive::memory::{OnnxEmbedder, OnnxEmbedderError};

// LLM
pub use crate::recursive::llm::{
    BoxedLlm, CliLlm, FailingLlm, IterativeMockLlm, Llm, LmOutput, MockLlm,
};

// Results
pub use crate::recursive::result::{
    Compiled, ContextId, Correction, Example, Iteration, OptimizedPrompt, RefineEvent,
    RefineResult, StopReason,
};

// Semantic validation
pub use crate::recursive::semantic::{semantic, SemanticBuilder, SemanticValidator};

// Markdown rewriting
pub use crate::recursive::rewrite::{
    extract_all_code, extract_code, extract_section, rewrite, Rewrite,
};

// DSPy-style patterns
pub use crate::recursive::agent::{agent, AgentResult, Step};
pub use crate::recursive::best_of::{
    best_of, BestOfResult, CandidatePool, PoolStats, ScoredCandidate, Scorer,
};
pub use crate::recursive::cache::{CacheExt, CachedLlm};
pub use crate::recursive::ensemble::{
    ensemble, Aggregate, ChainResult, ConsensusPool, EnsembleResult,
};
pub use crate::recursive::executor::{
    bash_executor, node_executor, python_executor, CodeExecutor, ExecutionResult,
};
pub use crate::recursive::optimize::{
    Dataset, OptimizeResult, Optimizer, Strategy, TrainingExample,
};
pub use crate::recursive::pipeline::{
    pipeline, BranchBuilder, FanOutBranchResult, MergeStrategy, Pipeline, PipelineEvent,
    PipelineResult, StepResult,
};
pub use crate::recursive::program::{program, ProgramResult};
pub use crate::recursive::rate_limit::{RateLimitConfig, RateLimitExt, RateLimitedLlm};
pub use crate::recursive::reason::{reason, ReasonResult};
pub use crate::recursive::retry::{LlmExt, RetryConfig, RetryLlm};
pub use crate::recursive::template::{FormatType, Template};
pub use crate::recursive::tool::{tool, MockTool, Tool};

// Multi-turn conversation
pub use crate::recursive::conversation::{Conversation, Message, Role};

// Typed/structured output
pub use crate::recursive::typed::{parse_output, typed, FormatInstruction, TypedValidator};

// Prompt formatting
pub use crate::recursive::formatter::{FeedbackFormatter, PassthroughFormatter, PromptFormatter};

// Multi-objective / Pareto
pub use crate::recursive::pareto::{
    multi_objective, refine_pareto, refine_pareto_sync, Direction, MultiObjective,
    MultiObjectiveValidate, MultiScore, Objective, ParetoCandidate, ParetoFront,
    ParetoRefineResult, Scalarization,
};

// Builder configs
pub use crate::recursive::agent::AgentConfig;
pub use crate::recursive::best_of::BestOfConfig;
pub use crate::recursive::ensemble::EnsembleConfig;
pub use crate::recursive::reason::ReasonConfig;
