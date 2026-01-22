// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Recursive Language Prompting Module
//!
//! Enables iterative refinement where an LLM builds upon its previous responses
//! through feedback loops. Creates training-like convergence where output quality
//! improves over iterations.
//!
//! ## Key Techniques
//!
//! - **Self-Refinement**: LLM critiques and improves its own output
//! - **Recursion of Thought (RoT)**: Divide-and-conquer for complex problems
//! - **Prompt Chaining**: Multi-step pipelines with state accumulation
//! - **Human-in-the-Loop**: Manual intervention points
//! - **Convergence Detection**: Automatic stopping when quality plateaus
//!
//! ## Zero-Copy Design
//!
//! - `StrView<'a>` for all string references
//! - `SmallVec` for inline storage
//! - GATs for zero-cost async
//! - Reference-based ownership (lifetimes over Arc)
//!
//! ## Example
//!
//! ```ignore
//! use kkachi::recursive::*;
//!
//! // Compose your own validator using Cli and CliPipeline
//! let rust_validator = CliPipeline::new()
//!     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
//!     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
//!     .file_ext("rs");
//!
//! let result = Kkachi::refine("question -> code")
//!     .domain("rust_codegen")
//!     .validate(rust_validator)
//!     .max_iterations(5)
//!     .run("Write a URL parser", &lm)
//!     .await?;
//! ```

use crate::str_view::StrView;

// DuckDB shim module - re-exports from either system or bundled crate
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub(crate) mod db {
    #[cfg(feature = "storage")]
    pub use duckdb::*;

    #[cfg(all(feature = "storage-bundled", not(feature = "storage")))]
    pub use duckdb_bundled::*;
}

pub mod api;
pub mod cli;
pub mod cluster;
pub mod condense;
pub mod criterion;
pub mod critic;
pub mod keywords;
pub mod module;
pub mod retrieve;
pub mod runner;
pub mod similarity;
pub mod state;
pub mod storage;
pub mod template;
pub mod tokenize;
pub mod training;

// Chunking (requires tokenization)
#[cfg(any(feature = "tiktoken", feature = "huggingface", feature = "chunking"))]
pub mod chain;
#[cfg(any(feature = "tiktoken", feature = "huggingface", feature = "chunking"))]
pub mod chunk;

// Re-exports
pub use api::{CriticConfig, FewShotConfig, Kkachi, RefineBuilder, RefineResult};
pub use cli::{
    Cli, CliBinaryCritic, CliExecutor, CliPipeline, CommandResult, ValidationResult, Validator,
    ValidatorCritic,
};
pub use cluster::{ClusterConfig, ClusterEngine, ClusterStats, DocumentCluster};
pub use condense::{
    BatchCondenseResult, CondenseConfig, CondenseEngine, CondenseResult, CondensedDocument,
};
pub use criterion::{
    All, Any, ConvergenceCriterion, MaxIterations, NoFeedback, ScorePlateau, ScoreThreshold,
    SmartConvergence, StandardConvergence,
};
pub use critic::{
    BinaryCritic, ChecklistCritic, Critic, CriticResult, ErrorCountCritic, HeuristicCritic,
    StagedCritic, TemplateCritic,
};
pub use keywords::{KeywordExtractor, NGramExtractor};
pub use module::{refine_loop, EvalResult, Recursive, RecursiveBuilder};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub use retrieve::DuckDBVectorStore;
pub use retrieve::{
    CachedContext,
    // ChunkStore types
    ChunkSearchResult,
    ChunkStore,
    ClusterInfo,
    CompositeVectorStore,
    DocSource,
    Embedder,
    FreshRetrieval,
    HashEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStats,
    RetrievedDoc,
    SmartRetriever,
    StoredChunk,
    VectorSearchResult,
    // VectorStore types
    VectorStore,
    ZeroEmbedder,
};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub use runner::RecursiveRunner;
pub use runner::{
    NoProgress, PrintProgress, ProgressCallback, RefinementResult, RunnerConfig, StandaloneRunner,
};
pub use similarity::{DocumentFeatures, EmbeddingRef, LocalSimilarity, SimilarityWeights};
pub use state::{IterationRecord, RecursiveConfig, RecursiveState};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub use storage::ContextStore;
pub use storage::{ContextId, ContextUpdate, ContextView, UpsertResult};
pub use template::{
    FormatSpec,
    FormatType,
    JsonSchema,
    // Prompt tone for recall/precision tuning
    PromptTone,
    Template,
    TemplateExample,
    TemplateOptions,
    ToneModifiers,
};
// Tokenization
#[cfg(feature = "huggingface")]
pub use tokenize::HuggingFaceTokenizer;
#[cfg(feature = "tiktoken")]
pub use tokenize::TiktokenTokenizer;
pub use tokenize::{SimpleTokenizer, Tokenizer};
// Chunking
#[cfg(any(feature = "tiktoken", feature = "huggingface", feature = "chunking"))]
pub use chunk::{ChunkConfig, ChunkStrategy, SectionType, TextChunk, TextChunker};
// Chaining
#[cfg(any(feature = "tiktoken", feature = "huggingface", feature = "chunking"))]
pub use chain::{chain, ChainBuilder, ChainedOutput, ChunkMetadata, DependencyRules, OutputChunk};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub use training::TrainingRunner;
pub use training::{MutableVectorStore, TrainingConfig, TrainingExample, TrainingStats};

/// Output from recursive execution.
#[derive(Debug)]
pub struct RecursiveOutput<'a> {
    /// Final output
    pub output: StrView<'a>,
    /// Number of iterations taken
    pub iterations: u32,
    /// Final score
    pub score: f64,
    /// Whether converged or hit max iterations
    pub converged: bool,
}
