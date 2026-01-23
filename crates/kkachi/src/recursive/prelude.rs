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
pub use crate::recursive::cli::{cli, Cli, CliCapture, CliExecutor};
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
#[cfg(feature = "embeddings-onnx")]
pub use crate::recursive::memory::{OnnxEmbedder, OnnxEmbedderError};
#[cfg(feature = "hnsw")]
pub use crate::recursive::memory::HnswIndex;

// LLM
pub use crate::recursive::llm::{
    BoxedLlm, FailingLlm, IterativeMockLlm, Llm, LmOutput, MockLlm,
};

// Results
pub use crate::recursive::result::{
    Compiled, ContextId, Correction, Example, Iteration, OptimizedPrompt, RefineResult, StopReason,
};

// Semantic validation
pub use crate::recursive::semantic::{semantic, SemanticBuilder, SemanticValidator};

// Markdown rewriting
pub use crate::recursive::rewrite::{
    extract_all_code, extract_code, extract_section, rewrite, Rewrite,
};

// DSPy-style patterns
pub use crate::recursive::tool::{tool, Tool, MockTool};
pub use crate::recursive::executor::{CodeExecutor, ExecutionResult, python_executor, node_executor, bash_executor};
pub use crate::recursive::reason::{reason, ReasonResult};
pub use crate::recursive::best_of::{best_of, Scorer, CandidatePool, ScoredCandidate, PoolStats};
pub use crate::recursive::ensemble::{ensemble, Aggregate, ConsensusPool, ChainResult};
pub use crate::recursive::agent::{agent, AgentResult, Step};
pub use crate::recursive::program::{program, ProgramResult};
