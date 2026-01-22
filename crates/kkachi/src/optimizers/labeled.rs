// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! LabeledFewShot - Direct Labeled Example Selection
//!
//! Simple optimizer that selects demonstrations directly from
//! labeled examples without any learning or optimization.
//!
//! ## Algorithm
//!
//! 1. Take k labeled examples from training set
//! 2. Optionally shuffle or sample
//! 3. Use directly as demonstrations

use crate::error::Result;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig, Rng};
use smallvec::SmallVec;

/// Selection strategy for labeled examples.
#[derive(Clone, Copy, Default)]
pub enum SelectionStrategy {
    /// Take first k examples
    #[default]
    First,
    /// Take last k examples
    Last,
    /// Random sample k examples
    Random,
    /// Stratified sampling (if labels available)
    Stratified,
}

/// LabeledFewShot configuration.
#[derive(Clone, Copy)]
pub struct LabeledConfig {
    /// Base optimizer config
    pub base: OptimizerConfig,
    /// Number of examples to select
    pub k: u8,
    /// Selection strategy
    pub strategy: SelectionStrategy,
    /// Random seed (for Random/Stratified strategies)
    pub seed: u64,
}

impl Default for LabeledConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig::default(),
            k: 5,
            strategy: SelectionStrategy::First,
            seed: 42,
        }
    }
}

impl LabeledConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            base: OptimizerConfig::new(),
            k: 5,
            strategy: SelectionStrategy::First,
            seed: 42,
        }
    }

    /// Set k (number of examples).
    pub const fn with_k(mut self, k: u8) -> Self {
        self.k = k;
        self
    }

    /// Set selection strategy.
    pub const fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// LabeledFewShot optimizer.
///
/// Selects demonstrations directly from labeled examples.
#[derive(Clone, Copy)]
pub struct LabeledFewShot {
    config: LabeledConfig,
}

impl LabeledFewShot {
    /// Create a new LabeledFewShot optimizer.
    pub const fn new(config: LabeledConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(LabeledConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &LabeledConfig {
        &self.config
    }

    /// Select k examples from trainset using configured strategy.
    pub fn select<'a>(&self, trainset: &ExampleSet<'a>) -> SmallVec<[u32; 8]> {
        let n = (self.config.k as usize).min(trainset.len());

        match self.config.strategy {
            SelectionStrategy::First => (0..n as u32).collect(),

            SelectionStrategy::Last => {
                let start = trainset.len().saturating_sub(n);
                (start as u32..trainset.len() as u32).collect()
            }

            SelectionStrategy::Random => {
                let mut rng = Rng::new(self.config.seed);
                let mut indices: SmallVec<[u32; 8]> = SmallVec::new();
                let mut available: Vec<u32> = (0..trainset.len() as u32).collect();

                for _ in 0..n {
                    if available.is_empty() {
                        break;
                    }
                    let idx = rng.gen_range(0, available.len() as u64) as usize;
                    indices.push(available.swap_remove(idx));
                }
                indices
            }

            SelectionStrategy::Stratified => {
                // Simplified stratified - just distribute evenly
                // Full impl would group by labels and sample from each
                let mut rng = Rng::new(self.config.seed);
                let mut indices: SmallVec<[u32; 8]> = SmallVec::new();

                if trainset.len() <= n {
                    return (0..trainset.len() as u32).collect();
                }

                // Evenly spaced with jitter
                let step = trainset.len() as f64 / n as f64;
                for i in 0..n {
                    let base = (i as f64 * step) as u32;
                    let jitter = rng.gen_range(0, (step as u64).max(1)) as u32;
                    let idx = (base + jitter).min(trainset.len() as u32 - 1);
                    if !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }

                // Fill remaining if needed
                while indices.len() < n {
                    let idx = rng.gen_range(0, trainset.len() as u64) as u32;
                    if !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }

                indices
            }
        }
    }
}

impl Optimizer for LabeledFewShot {
    type Output<'a> = OptimizationResult;
    type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        let indices = self.select(trainset);

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 0,
        }))
    }

    fn name(&self) -> &'static str {
        "LabeledFewShot"
    }
}

/// Builder for convenient LabeledFewShot creation.
pub struct LabeledFewShotBuilder {
    config: LabeledConfig,
}

impl LabeledFewShotBuilder {
    /// Start building.
    pub fn new() -> Self {
        Self {
            config: LabeledConfig::default(),
        }
    }

    /// Set k.
    pub fn k(mut self, k: u8) -> Self {
        self.config.k = k;
        self
    }

    /// Use first k strategy.
    pub fn first(mut self) -> Self {
        self.config.strategy = SelectionStrategy::First;
        self
    }

    /// Use last k strategy.
    pub fn last(mut self) -> Self {
        self.config.strategy = SelectionStrategy::Last;
        self
    }

    /// Use random strategy.
    pub fn random(mut self, seed: u64) -> Self {
        self.config.strategy = SelectionStrategy::Random;
        self.config.seed = seed;
        self
    }

    /// Use stratified strategy.
    pub fn stratified(mut self, seed: u64) -> Self {
        self.config.strategy = SelectionStrategy::Stratified;
        self.config.seed = seed;
        self
    }

    /// Build the optimizer.
    pub fn build(self) -> LabeledFewShot {
        LabeledFewShot::new(self.config)
    }
}

impl Default for LabeledFewShotBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_labeled_creation() {
        let labeled = LabeledFewShot::default();
        assert_eq!(labeled.name(), "LabeledFewShot");
        assert_eq!(labeled.config().k, 5);
    }

    #[test]
    fn test_labeled_config() {
        let config = LabeledConfig::new()
            .with_k(10)
            .with_strategy(SelectionStrategy::Random)
            .with_seed(123);
        assert_eq!(config.k, 10);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_builder() {
        let labeled = LabeledFewShotBuilder::new().k(7).random(999).build();

        assert_eq!(labeled.config().k, 7);
        assert_eq!(labeled.config().seed, 999);
        assert!(matches!(
            labeled.config().strategy,
            SelectionStrategy::Random
        ));
    }

    #[test]
    fn test_first_strategy() {
        let labeled = LabeledFewShot::new(
            LabeledConfig::new()
                .with_k(3)
                .with_strategy(SelectionStrategy::First),
        );

        // Create mock trainset
        let buffer = crate::buffer::Buffer::Static(
            b"input1\noutput1\ninput2\noutput2\ninput3\noutput3\ninput4\noutput4\ninput5\noutput5",
        );
        let trainset = ExampleSet::from_buffer(buffer, 5);

        let indices = labeled.select(&trainset);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        assert_eq!(indices[2], 2);
    }

    #[test]
    fn test_last_strategy() {
        let labeled = LabeledFewShot::new(
            LabeledConfig::new()
                .with_k(3)
                .with_strategy(SelectionStrategy::Last),
        );

        let buffer = crate::buffer::Buffer::Static(
            b"input1\noutput1\ninput2\noutput2\ninput3\noutput3\ninput4\noutput4\ninput5\noutput5",
        );
        let trainset = ExampleSet::from_buffer(buffer, 5);

        let indices = labeled.select(&trainset);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 2);
        assert_eq!(indices[1], 3);
        assert_eq!(indices[2], 4);
    }

    #[test]
    fn test_random_strategy() {
        let labeled = LabeledFewShot::new(
            LabeledConfig::new()
                .with_k(3)
                .with_strategy(SelectionStrategy::Random)
                .with_seed(42),
        );

        let buffer = crate::buffer::Buffer::Static(
            b"input1\noutput1\ninput2\noutput2\ninput3\noutput3\ninput4\noutput4\ninput5\noutput5",
        );
        let trainset = ExampleSet::from_buffer(buffer, 5);

        let indices = labeled.select(&trainset);
        assert_eq!(indices.len(), 3);

        // All indices should be valid
        for idx in &indices {
            assert!(*idx < 5);
        }

        // Should be deterministic with same seed
        let indices2 = labeled.select(&trainset);
        assert_eq!(indices, indices2);
    }
}
