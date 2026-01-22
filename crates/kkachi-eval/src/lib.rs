// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Evaluation framework for Kkachi

#![allow(clippy::needless_lifetimes)]
#![allow(clippy::should_implement_trait)]
//!
//! Provides high-performance metrics with Rayon parallelism.

pub mod evaluator;
pub mod metric;
pub mod parallel;

pub use evaluator::Evaluator;
pub use metric::{
    AnswerRelevancy, BatchMetricEvaluator, CompleteAndGrounded, ExactMatch, F1Score, Metric,
    MetricResult, SemanticF1,
};
pub use parallel::ParallelEvaluator;
