// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

// Allow manual modulo check since `is_multiple_of` is unstable (requires nightly)
#![allow(clippy::manual_is_multiple_of)]

//! # Kkachi - High-Performance LM Optimization Library
//!
//! Zero-copy, embeddable library for optimizing language model prompts and programs.
//! Designed for production use with focus on performance, safety, and ease of integration.
//!
//! ## Architecture
//!
//! Kkachi is built on several key principles:
//!
//! - **TRUE Zero-Copy**: `StrView<'a>` and `BufferView<'a>` for zero-allocation string handling
//! - **String Interning**: 4-byte `Sym` symbols for field names instead of 24-byte Strings
//! - **GATs**: Generic Associated Types for zero-cost async without boxing
//! - **Streaming Pipelines**: Data flows between modules without full materialization
//!
//! ## Quick Start
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

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
// Allow common patterns that trigger clippy warnings but are intentional
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::int_plus_one)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::manual_pattern_char_comparison)]

#[cfg(feature = "std")]
extern crate std;

// Phase 0: Zero-copy foundation
pub mod buffer;
pub mod intern;
pub mod str_view;

// Phase 1: Core infrastructure
pub mod bootstrap;
pub mod error;
pub mod example;
pub mod field;
pub mod module;
pub mod optimizer;
pub mod predict;
pub mod prediction;
pub mod signature;
pub mod types;

// Phase 3: DSPy modules
pub mod modules;

// Phase 4: Advanced optimizers
pub mod optimizers;

// Phase 5: Zero-copy adapters
pub mod adapter;

// Phase 6: Assertions
pub mod assertion;

// Phase 8: Hybrid executor
pub mod executor;

// Phase 10: Recursive Language Prompting (simplified API)
pub mod recursive;

// Phase 11: Diff visualization
pub mod diff;

// Phase 12: Human-in-the-Loop
pub mod hitl;

// Phase 13: Declarative API (thin re-export of recursive)
pub mod declarative;

// Recall/Precision tuning
pub mod recall_precision;

// Re-exports for convenience
pub use bootstrap::{BootstrapFewShot, BootstrapFewShotWithRandomSearch};
pub use error::{Error, OptimizationDetails, Result};
pub use example::Example;
pub use field::{Field, FieldType, InputField, OutputField};
pub use module::Module;
pub use optimizer::{ExampleMeta, ExampleSet, OptimizationResult, Optimizer, OptimizerConfig, Rng};
pub use predict::{
    predict_with_lm, DemoMeta, FieldRange, LMClient, LMOutput, Predict, PredictOutput,
};
pub use prediction::Prediction;
pub use signature::{Signature, SignatureBuilder};

// Advanced optimizers
pub use optimizers::{
    COPROConfig, COPROResult, CombineStrategy, Embedder as OptimizerEmbedder, EmbeddingIndex,
    Ensemble, EnsembleConfig, EnsembleResult, ErasedOptimizer, FailureCase, Improvement,
    ImprovementKind, KNNConfig, KNNFewShot, KNNSelector, LabeledConfig, LabeledFewShot,
    LabeledFewShotBuilder, MIPROConfig, MIPROResult, OptimizeInto, SIMBAConfig, SIMBAResult,
    SelectionStrategy, TPESampler, COPRO, MIPRO, SIMBA,
};

// DSPy modules
pub use modules::{
    bon_with_pool, multi_chain_with_pool, BestOfN, CandidatePool, CandidatePoolStats,
    ChainOfThought, ChainPool, ChainPoolStats, CodeExecutor, ExecutionResult, MultiChain,
    ProgramOfThought, ReAct, Refine as ModuleRefine, RefineConfig, ScoredCandidate, ScoredChain,
    Tool, ToolResult,
};

// Recall/Precision mode
pub use recall_precision::RecallPrecisionMode;

// Adapters
pub use adapter::{
    Adapter, ChatAdapter, ChatConfig, DemoData, JSONAdapter, JSONConfig, XMLAdapter, XMLConfig,
};

// Assertions
pub use assertion::{
    Assertion, AssertionLevel, AssertionResult, AssertionRunner, Contains, Custom, EndsWith,
    JsonValid, LengthBounds, NotEmpty, OneOf, RegexMatch, StartsWith,
};

// Executor
pub use executor::{
    BatchRunner, BufferPool, BufferPoolStats, ExecutorConfig, ExecutorStats, HybridExecutor,
    ScopedBuffer,
};

// Recursive Language Prompting (new simplified API)
pub use recursive::{
    agent,
    all,
    any,
    bash_executor,
    best_of,
    // Validation
    checks,
    cli,
    cosine_similarity,
    ensemble,
    extract_all_code,
    extract_code,
    extract_section,
    // Memory/RAG
    memory,
    mmr_select,
    node_executor,
    program,
    python_executor,
    reason,
    // Core refinement
    refine,
    // Markdown rewriting
    rewrite,
    ruby_executor,
    // DSPy-style patterns (new API)
    tool,
    Agent,
    AgentConfig,
    AgentResult,
    Aggregate,
    All,
    AlwaysFail,
    And,
    Any,
    AsyncFnTool,
    BestOf,
    BestOfConfig,
    BestOfResult,
    BoolValidator,
    BoxedLlm,
    CandidatePool as DspCandidatePool,
    ChainResult,
    Checks,
    Cli,
    CliCapture,
    CliExecutor,
    CodeExecutor as DspCodeExecutor,
    // Results
    Compiled,
    Config as RefineConfig2,
    ConsensusPool,
    ContextId,
    Correction,
    DefaultScorer,
    Document,
    Embedder,
    Ensemble as DspEnsemble,
    EnsembleConfig as DspEnsembleConfig,
    EnsembleResult as DspEnsembleResult,
    Example as RefineExample,
    ExecutionResult as DspExecutionResult,
    FailingLlm,
    FnScorer,
    FnTool,
    FnValidator,
    // Template
    FormatSpec,
    FormatType,
    HashEmbedder,
    Iteration,
    IterativeMockLlm,
    JsonSchema,
    LinearIndex,
    // LLM trait
    Llm,
    LmOutput,
    Memory,
    MockLlm,
    MockTool as DspMockTool,
    NoValidation,
    OptimizedPrompt,
    Or,
    PoolStats,
    ProcessExecutor,
    Program,
    ProgramConfig,
    ProgramResult,
    PromptTone,
    Reason,
    ReasonConfig as ReasonCfg,
    ReasonResult,
    Recall,
    Refine as RefineBuilder2,
    RefineResult,
    Rewrite,
    Score,
    ScoreValidator,
    ScoredCandidate as DspScoredCandidate,
    Scorer,
    Step,
    Template,
    TemplateExample,
    TemplateOptions,
    ToneModifiers,
    Tool as DspTool,
    ToolBuilder,
    Validate,
    ValidateExt,
    VectorIndex,
};

// Feature-gated recursive exports
#[cfg(feature = "hnsw")]
pub use recursive::HnswIndex;
#[cfg(feature = "embeddings-onnx")]
pub use recursive::{OnnxEmbedder, OnnxEmbedderError};

// Diff visualization
pub use diff::{
    Change, ChangeKind, DemoSnapshot, DemosDiff, DiffAlgorithm, DiffColors, DiffHunk, DiffRenderer,
    DiffStats, DiffStyle, FieldsDiff, IterationDiffBuilder, ModuleDiff, TextDiff,
};

// Human-in-the-Loop
pub use hitl::{
    AsyncHumanReviewer, AutoAcceptReviewer, CallbackReviewer, HITLConfig, HumanReviewer,
    RecordingReviewer, ReviewContext, ReviewDecision, ReviewTrigger, TerminalReviewer,
    ThresholdReviewer,
};

// Declarative API (thin re-export of recursive + Jinja)
pub use declarative::{JinjaTemplate, JinjaValue};

// Zero-copy types
pub use buffer::{Buffer, BufferView};
pub use intern::{resolve, sym, Sym};
pub use str_view::StrView;
pub use types::{FieldMap, Inputs};

/// Prelude module for convenient imports.
pub mod prelude {
    // New simplified recursive API
    pub use crate::recursive::prelude::*;

    // Error handling
    pub use crate::{Error, OptimizationDetails, Result};

    // Zero-copy types
    pub use crate::{resolve, sym, Buffer, BufferView, StrView, Sym};

    // Core types
    pub use crate::{
        Example, Field, FieldMap, InputField, Inputs, Module, OutputField, Predict, Prediction,
        Signature, SignatureBuilder,
    };

    // Optimizer system
    pub use crate::{
        BootstrapFewShot, ExampleSet, Optimizer, OptimizerConfig, COPRO, MIPRO, SIMBA,
    };

    // DSPy modules (old API - kept for compatibility)
    pub use crate::{
        bon_with_pool, multi_chain_with_pool, BestOfN, CandidatePool as OldCandidatePool,
        ChainOfThought, ChainPool, CodeExecutor as OldCodeExecutor, ExecutionResult, ModuleRefine,
        MultiChain, ProgramOfThought, ReAct, RefineConfig, ScoredCandidate as OldScoredCandidate,
        ScoredChain, Tool as OldTool, ToolResult,
    };

    // Adapters
    pub use crate::{Adapter, ChatAdapter, JSONAdapter, XMLAdapter};

    // Assertions
    pub use crate::{Assertion, AssertionLevel, AssertionRunner};

    // Executor
    pub use crate::{BatchRunner, BufferPool, ExecutorConfig, HybridExecutor};

    // Diff
    pub use crate::DiffStyle;

    // HITL
    pub use crate::{HITLConfig, ReviewDecision};

    // Recall/Precision
    pub use crate::RecallPrecisionMode;

    // LM Client
    pub use crate::{predict_with_lm, LMClient};
}

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
