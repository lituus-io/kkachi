// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! REPL (Read-Eval-Print-Loop) for interactive prompt engineering.
//!
//! This module provides an interactive environment for:
//! - Configuring modules (signature, instruction, demos)
//! - Running predictions and refinements
//! - Viewing diffs between iterations
//! - Human-in-the-loop review
//! - Loading and executing pipelines
//! - Saving and loading sessions

#![allow(dead_code)] // REPL infrastructure is still being developed
#![allow(unused_imports)] // Public API re-exports for external consumers

mod commands;
mod completer;
mod engine;
mod pipeline;
mod prompt;
mod state;

pub use commands::CommandRegistry;
pub use engine::Repl;
pub use pipeline::{
    Pipeline, PipelineError, PipelineResult, PipelineStage, StageConfig, StageType,
};
pub use state::{DemoData, IterationSnapshot, ProviderType, SessionState, StateHistory};
