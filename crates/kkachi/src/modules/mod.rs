// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! DSPy-style modules for LM-based prediction
//!
//! This module provides composable, zero-copy modules that implement
//! various reasoning strategies like Chain of Thought, ReAct, etc.
//!
//! All modules use GATs for zero-cost async and references for zero-copy.

pub mod best_of_n;
pub mod chain_of_thought;
pub mod multi_chain;
pub mod program_of_thought;
pub mod react;
pub mod refine;

// Re-exports
pub use best_of_n::{
    bon_with_pool,
    BestOfN,
    CandidatePool,
    CandidatePoolStats,
    // Candidate pool for recall/precision tuning
    ScoredCandidate,
};
pub use chain_of_thought::ChainOfThought;
pub use multi_chain::{
    multi_chain_with_pool,
    ChainPool,
    ChainPoolStats,
    MultiChain,
    // ChainPool for recall/precision tuning
    ScoredChain,
};
pub use program_of_thought::{CodeExecutor, ExecutionResult, ProgramOfThought};
pub use react::{ReAct, Tool, ToolResult};
pub use refine::{Refine, RefineConfig};
