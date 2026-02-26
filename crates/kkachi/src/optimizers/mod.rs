// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Advanced optimizers for prompt and demonstration optimization
//!
//! This module provides DSPy-style optimizers beyond basic bootstrap:
//! - COPRO: Coordinate prompt optimization
//! - MIPRO: Multi-instruction prompt optimization
//! - KNNFewShot: K-nearest neighbor demonstration selection
//! - LabeledFewShot: Direct labeled example selection
//! - SIMBA: Self-improving modular boosting
//! - Ensemble: Combines multiple optimizers

pub mod copro;
pub mod ensemble;
pub mod knn;
pub mod labeled;
pub mod mipro;
pub mod simba;

// Re-exports - configs
pub use copro::{COPROConfig, COPROResult, COPRO};
pub use ensemble::{
    CombineStrategy, Ensemble, EnsembleConfig, EnsembleResult, ErasedOptimizer, OptimizeInto,
};
pub use knn::{Embedder, EmbeddingIndex, KNNConfig, KNNFewShot, KNNSelector};
pub use labeled::{LabeledConfig, LabeledFewShot, LabeledFewShotBuilder, SelectionStrategy};
pub use mipro::{MIPROConfig, MIPROResult, TPESampler, Trial, MIPRO};
pub use simba::{FailureCase, Improvement, ImprovementKind, SIMBAConfig, SIMBAResult, SIMBA};
