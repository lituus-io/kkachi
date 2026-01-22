// Copyright 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Declarative Pipeline API
//!
//! Zero-copy, fluent pipeline builder using generics and type-state patterns.
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::*;
//!
//! // Define your own validator
//! let validator = CliPipeline::new()
//!     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
//!     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
//!     .file_ext("rs");
//!
//! let result = pipeline("question -> code")
//!     .validate(validator)
//!     .refine(5, 0.9)
//!     .run(input, &llm)
//!     .await?;
//! ```

mod doc_template;
mod jinja;
mod oneshot;
mod pipeline;
mod rag;
mod rag_optimizer;
mod steps;

// Re-export main types
pub use pipeline::{
    pipe,
    pipeline,
    Check,
    CheckBuilder,
    CheckKind,
    FluentPipeline,
    NoCritic,
    NoStrategy,
    PipelineOutput,
    PipelineOutputOwned,
    // RAG-optimized pipeline integration
    RagOptimizedPipeline,
    RagPipelineConfig,
    RagPipelineOutput,
    StageCorrection,
    WithBestOfN,
    WithCoT,
    WithMultiChain,
    LLM,
};

pub use rag::{LiveRag, RagAnalyzer, RagExample};

pub use steps::{Step, StepBuilder, StepResult, Steps, StepsOutput};

// Re-export CLI validation primitives (generic, user-defined)
pub use crate::recursive::{
    BinaryCritic,
    ChecklistCritic,
    // Generic CLI primitives
    Cli,
    CliBinaryCritic,
    CliExecutor,
    CliPipeline,
    CommandResult,
    // Critic traits
    Critic,
    CriticResult,
    HeuristicCritic,
    ValidationResult,
    Validator,
    ValidatorCritic,
};

pub use crate::str_view::StrView;

// Document template types
pub use doc_template::{DocumentMetadata, RagDocumentTemplate, TemplateSection};

// One-shot prompt types
pub use oneshot::{ErrorCorrection, OneShotFailure, OneShotPrompt, OneShotTestResult};

// RAG optimizer
pub use rag_optimizer::{
    OptimizationResult, RagOptimizer, RagOptimizerConfig, Stage, StageConfig, StageResult,
};

// Jinja template support
pub use jinja::JinjaTemplate;
// Re-export minijinja::Value for template rendering
pub use minijinja::Value as JinjaValue;
