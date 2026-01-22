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
//! - **DAG Composition**: Functions compose as directed acyclic graphs for flexible pipelines
//! - **Streaming Pipelines**: Data flows between modules without full materialization
//!
//! ## Quick Start
//!
//! ```ignore
//! use kkachi::prelude::*;
//!
//! // Create a simple chain of thought module
//! let cot = ChainOfThought::new(signature!("question -> answer"));
//!
//! // Build a DAG pipeline
//! let pipeline = graph!(cot >> validate >> format_output);
//!
//! // Execute
//! let result = pipeline.execute(inputs).await?;
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
pub mod dag;
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

// Phase 10: Recursive Language Prompting
pub mod recursive;

// Phase 11: Diff visualization
pub mod diff;

// Phase 12: Human-in-the-Loop
pub mod hitl;

// Phase 13: Declarative API
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
    COPROConfig, COPROResult, CombineStrategy, Embedder, EmbeddingIndex, Ensemble, EnsembleConfig,
    EnsembleResult, ErasedOptimizer, FailureCase, Improvement, ImprovementKind, KNNConfig,
    KNNFewShot, KNNSelector, LabeledConfig, LabeledFewShot, LabeledFewShotBuilder, MIPROConfig,
    MIPROResult, OptimizeInto, SIMBAConfig, SIMBAResult, SelectionStrategy, TPESampler, COPRO,
    MIPRO, SIMBA,
};

// DSPy modules
pub use modules::{
    bon_with_pool,
    multi_chain_with_pool,
    BestOfN,
    CandidatePool,
    CandidatePoolStats,
    ChainOfThought,
    ChainPool,
    ChainPoolStats,
    CodeExecutor,
    ExecutionResult,
    MultiChain,
    ProgramOfThought,
    ReAct,
    Refine,
    RefineConfig,
    // Candidate pool for recall/precision tuning
    ScoredCandidate,
    // Chain pool for recall/precision tuning
    ScoredChain,
    Tool,
    ToolResult,
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

// Recursive Language Prompting
#[cfg(feature = "huggingface")]
pub use recursive::HuggingFaceTokenizer;
#[cfg(feature = "tiktoken")]
pub use recursive::TiktokenTokenizer;
pub use recursive::{
    refine_loop,
    All,
    Any,
    BatchCondenseResult,
    BinaryCritic,
    CachedContext,
    ChecklistCritic,
    // ChunkStore
    ChunkSearchResult,
    ChunkStore,
    Cli,
    CliBinaryCritic,
    // CLI
    CliExecutor,
    CliPipeline,
    ClusterConfig,
    ClusterEngine,
    ClusterInfo,
    ClusterStats,
    CommandResult,
    CompositeVectorStore,
    CondenseConfig,
    CondenseEngine,
    CondenseResult,
    CondensedDocument,
    // Storage
    ContextId,
    ContextUpdate,
    ContextView,
    // Convergence criteria
    ConvergenceCriterion,
    // Critics
    Critic,
    CriticConfig,
    CriticResult,
    DocSource,
    DocumentCluster,
    DocumentFeatures,
    EmbeddingRef,
    ErrorCountCritic,
    EvalResult,
    FewShotConfig,
    FormatSpec,
    FormatType,
    FreshRetrieval,
    HashEmbedder,
    HeuristicCritic,
    HybridRetriever,
    InMemoryVectorStore,
    IterationRecord,
    JsonSchema,
    KeywordExtractor,
    // Declarative API
    Kkachi,
    LocalSimilarity,
    MaxIterations,
    NGramExtractor,
    NoFeedback,
    NoProgress,
    PrintProgress,
    ProgressCallback,
    // Prompt tone for recall/precision tuning
    PromptTone,
    // Module and builder
    Recursive,
    RecursiveBuilder,
    RecursiveConfig,
    RecursiveOutput,
    // State and config
    RecursiveState,
    RefineBuilder,
    RefineResult,
    // Runner
    RefinementResult,
    // Retrieval
    RetrievalConfig,
    RetrievalResult,
    RetrievalStats,
    RetrievedDoc,
    RunnerConfig,
    ScorePlateau,
    ScoreThreshold,
    // Similarity and clustering
    SimilarityWeights,
    // Tokenization
    SimpleTokenizer,
    SmartConvergence,
    SmartRetriever,
    StagedCritic,
    StandaloneRunner,
    StandardConvergence,
    StoredChunk,
    // Template
    Template,
    TemplateCritic,
    TemplateExample,
    TemplateOptions,
    Tokenizer,
    ToneModifiers,
    UpsertResult,
    ValidationResult,
    Validator,
    ValidatorCritic,
    VectorSearchResult,
    // VectorStore
    VectorStore,
    ZeroEmbedder,
};
// Chunking
#[cfg(any(feature = "tiktoken", feature = "huggingface", feature = "chunking"))]
pub use recursive::{
    chain, ChainBuilder, ChainedOutput, ChunkConfig, ChunkMetadata, ChunkStrategy, DependencyRules,
    OutputChunk, SectionType, TextChunk, TextChunker,
};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub use recursive::{ContextStore, RecursiveRunner};

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

// Declarative API (new zero-copy fluent builder)
pub use declarative::{
    pipe,
    pipeline,
    Check,
    CheckBuilder,
    CheckKind,
    FluentPipeline,
    LiveRag,
    NoCritic,
    NoStrategy,
    PipelineOutput,
    PipelineOutputOwned,
    RagAnalyzer,
    // RAG
    RagExample,
    StageCorrection,
    // Steps
    Step,
    StepBuilder,
    StepResult,
    Steps,
    StepsOutput,
    WithBestOfN,
    WithCoT,
    WithMultiChain,
    // Re-exports from declarative
    LLM as DeclLLM,
};

// Zero-copy types
pub use buffer::{Buffer, BufferView};
pub use intern::{resolve, sym, Sym};
pub use str_view::StrView;
pub use types::{FieldMap, Inputs};

// DAG types
pub use dag::{Graph, Node};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::{
        bon_with_pool,
        multi_chain_with_pool,
        pipe,
        // Declarative API (primary entry point)
        pipeline,
        predict_with_lm,
        resolve,
        sym,
        // Adapters
        Adapter,
        // Assertions
        Assertion,
        AssertionLevel,
        AssertionRunner,
        BatchRunner,
        BestOfN,
        BinaryCritic,
        BootstrapFewShot,
        // Zero-copy types
        Buffer,
        BufferPool,
        BufferView,
        CandidatePool,
        CandidatePoolStats,
        // DSPy modules
        ChainOfThought,
        ChainPool,
        ChainPoolStats,
        ChatAdapter,
        ChecklistCritic,
        CodeExecutor,
        Contains,
        ConvergenceCriterion,
        Critic,
        CriticResult,
        DeclLLM,
        // Diff
        DiffStyle,
        Ensemble,
        // Error handling
        Error,
        Example,
        ExampleSet,
        ExecutionResult,
        ExecutorConfig,
        Field,
        FluentPipeline,
        FormatType,
        Graph,
        // HITL
        HITLConfig,
        HeuristicCritic,
        // Executor
        HybridExecutor,
        InputField,
        Inputs,
        JSONAdapter,
        JsonValid,
        KNNFewShot,
        LMClient,
        LabeledFewShot,
        LengthBounds,
        LiveRag,
        MaxIterations,
        // Module system
        Module,
        MultiChain,
        NoCritic,
        NoStrategy,
        // DAG system
        Node,
        NotEmpty,
        OptimizationDetails,
        // Basic optimizer system
        Optimizer,
        OptimizerConfig,
        OutputField,
        // Pipeline outputs
        PipelineOutput,
        PipelineOutputOwned,
        Predict,
        Prediction,
        ProgramOfThought,
        // Prompt tone for recall/precision
        PromptTone,
        RagExample,
        ReAct,
        // Recall/Precision mode
        RecallPrecisionMode,
        RecursiveConfig,
        RecursiveOutput,
        // Recursive Language Prompting
        RecursiveState,
        Refine,
        RefineConfig,
        RegexMatch,
        Result,
        ReviewDecision,
        ScoreThreshold,
        // Candidate pool for recall/precision
        ScoredCandidate,
        // Chain pool for recall/precision
        ScoredChain,
        // Core types
        Signature,
        SignatureBuilder,
        StandardConvergence,
        Step,
        Steps,
        StrView,
        Sym,
        // Template
        Template,
        TemplateCritic,
        TemplateExample,
        Tool,
        ToolResult,
        XMLAdapter,
        // Advanced optimizers
        COPRO,
        MIPRO,
        SIMBA,
    };
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
