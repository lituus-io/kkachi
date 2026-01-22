// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Human-in-the-Loop (HITL) system for interactive refinement.
//!
//! This module provides:
//! - `HITLConfig` - Configuration for when to trigger human review
//! - `ReviewContext` - Context provided to the reviewer
//! - `ReviewDecision` - Human's decision after review
//! - `HumanReviewer` trait - Interface for review implementations
//! - `TerminalReviewer` - Interactive terminal-based reviewer

mod config;
mod review;
mod reviewer;
mod terminal;

pub use config::HITLConfig;
pub use review::{ReviewContext, ReviewDecision, ReviewTrigger};
pub use reviewer::{
    AsyncHumanReviewer, AutoAcceptReviewer, CallbackReviewer, HumanReviewer, RecordingReviewer,
    ThresholdReviewer,
};
pub use terminal::{ProgressReviewer, TerminalReviewer};

/// Re-export for convenience.
pub use crate::diff::{DiffRenderer, DiffStyle, ModuleDiff};
