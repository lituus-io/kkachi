// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Ensemble - Combines Multiple Optimizers
//!
//! Runs multiple optimizers and combines their results through
//! various strategies (voting, averaging, selection).
//!
//! ## Strategies
//!
//! - Best: Pick single best result
//! - Union: Combine demo sets (up to max)
//! - Voting: Weight by optimizer confidence

use crate::error::Result;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Strategy for combining optimizer results.
#[derive(Clone, Copy, Default)]
pub enum CombineStrategy {
    /// Pick the single best result
    #[default]
    Best,
    /// Union of all demo sets (truncated to max)
    Union,
    /// Weighted voting based on scores
    WeightedVote,
    /// Intersection of demo sets
    Intersection,
}

/// Ensemble optimizer configuration.
#[derive(Clone, Copy)]
pub struct EnsembleConfig {
    /// Base optimizer config
    pub base: OptimizerConfig,
    /// Combination strategy
    pub strategy: CombineStrategy,
    /// Run optimizers in parallel
    pub parallel: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig::default(),
            strategy: CombineStrategy::Best,
            parallel: true,
        }
    }
}

impl EnsembleConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            base: OptimizerConfig::new(),
            strategy: CombineStrategy::Best,
            parallel: true,
        }
    }

    /// Set combination strategy.
    pub const fn with_strategy(mut self, strategy: CombineStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set parallel execution.
    pub const fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Result from a single optimizer in ensemble.
#[derive(Clone)]
pub struct OptimizerResult {
    /// Optimizer name
    pub name: &'static str,
    /// Demo indices selected
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Score achieved
    pub score: f64,
    /// Weight for voting
    pub weight: f64,
}

/// Ensemble optimizer.
///
/// Combines multiple optimizers using various strategies.
pub struct Ensemble<'a> {
    config: EnsembleConfig,
    /// Boxed optimizers for heterogeneous storage
    optimizers: Vec<Box<dyn ErasedOptimizer + 'a>>,
    /// Optimizer names for tracking
    names: Vec<&'static str>,
}

/// Type-erased optimizer wrapper for heterogeneous storage.
/// Uses a boxed async function for type erasure.
pub trait ErasedOptimizer: Send + Sync {
    /// Run optimization and return OptimizationResult.
    fn optimize_erased<'a>(
        &'a self,
        trainset: &'a ExampleSet<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<OptimizationResult>> + Send + 'a>>;

    /// Get optimizer name.
    fn erased_name(&self) -> &'static str;
}

/// Wrapper to erase optimizer types.
struct OptimizerWrapper<O> {
    inner: O,
    name: &'static str,
}

impl<O> ErasedOptimizer for OptimizerWrapper<O>
where
    O: Send + Sync,
    for<'a> O: OptimizeInto<'a>,
{
    fn optimize_erased<'a>(
        &'a self,
        trainset: &'a ExampleSet<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<OptimizationResult>> + Send + 'a>>
    {
        self.inner.optimize_into(trainset)
    }

    fn erased_name(&self) -> &'static str {
        self.name
    }
}

/// Helper trait for type-erased optimization.
pub trait OptimizeInto<'a>: Send + Sync {
    /// Optimize and return result.
    fn optimize_into(
        &'a self,
        trainset: &'a ExampleSet<'a>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<OptimizationResult>> + Send + 'a>>;
}

impl<'a> Ensemble<'a> {
    /// Create a new ensemble.
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            optimizers: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Create with default config.
    pub fn default() -> Self {
        Self::new(EnsembleConfig::new())
    }

    /// Add an optimizer to the ensemble.
    /// The optimizer must implement `OptimizeInto` for type erasure.
    pub fn add<O>(mut self, optimizer: O) -> Self
    where
        O: Optimizer + Send + Sync + 'a,
        for<'b> O: OptimizeInto<'b>,
    {
        let name = <O as Optimizer>::name(&optimizer);
        self.optimizers.push(Box::new(OptimizerWrapper {
            inner: optimizer,
            name,
        }));
        self.names.push(name);
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }

    /// Get number of optimizers.
    pub fn len(&self) -> usize {
        self.optimizers.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.optimizers.is_empty()
    }

    /// Run all optimizers and combine results.
    pub async fn run<'b>(&'b self, trainset: &'b ExampleSet<'b>) -> Result<EnsembleResult> {
        if self.optimizers.is_empty() {
            return Ok(EnsembleResult {
                demo_indices: SmallVec::new(),
                score: 0.0,
                individual_results: Vec::new(),
                strategy: self.config.strategy,
            });
        }

        // Run optimizers (sequentially for simplicity - real impl could parallelize)
        let mut results = Vec::with_capacity(self.optimizers.len());

        for opt in self.optimizers.iter() {
            let name = opt.erased_name();
            let result = opt.optimize_erased(trainset).await?;
            results.push(OptimizerResult {
                name,
                demo_indices: result.demo_indices,
                score: result.score,
                weight: 1.0, // Equal weights by default
            });
        }

        // Combine based on strategy
        let combined = match self.config.strategy {
            CombineStrategy::Best => self.combine_best(&results),
            CombineStrategy::Union => self.combine_union(&results, self.config.base.max_demos),
            CombineStrategy::WeightedVote => {
                self.combine_weighted_vote(&results, self.config.base.max_demos)
            }
            CombineStrategy::Intersection => self.combine_intersection(&results),
        };

        Ok(EnsembleResult {
            demo_indices: combined.0,
            score: combined.1,
            individual_results: results,
            strategy: self.config.strategy,
        })
    }

    /// Combine using best strategy (pick highest score).
    fn combine_best(&self, results: &[OptimizerResult]) -> (SmallVec<[u32; 8]>, f64) {
        results
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .map(|r| (r.demo_indices.clone(), r.score))
            .unwrap_or((SmallVec::new(), 0.0))
    }

    /// Combine using union strategy.
    fn combine_union(&self, results: &[OptimizerResult], max: u8) -> (SmallVec<[u32; 8]>, f64) {
        let mut seen = std::collections::HashSet::new();
        let mut combined: SmallVec<[u32; 8]> = SmallVec::new();
        let mut total_score = 0.0;

        // Add demos in order of optimizer score
        let mut sorted: Vec<_> = results.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        for result in sorted {
            for &idx in &result.demo_indices {
                if seen.insert(idx) && combined.len() < max as usize {
                    combined.push(idx);
                }
            }
            total_score += result.score;
        }

        let avg_score = if results.is_empty() {
            0.0
        } else {
            total_score / results.len() as f64
        };

        (combined, avg_score)
    }

    /// Combine using weighted voting.
    fn combine_weighted_vote(
        &self,
        results: &[OptimizerResult],
        max: u8,
    ) -> (SmallVec<[u32; 8]>, f64) {
        // Count weighted votes for each demo index
        let mut votes: HashMap<u32, f64> = HashMap::new();

        for result in results {
            let weight = result.score.max(0.0); // Use score as weight
            for &idx in &result.demo_indices {
                *votes.entry(idx).or_insert(0.0) += weight;
            }
        }

        // Sort by votes
        let mut sorted: Vec<_> = votes.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top max
        let combined: SmallVec<[u32; 8]> = sorted
            .iter()
            .take(max as usize)
            .map(|(idx, _)| *idx)
            .collect();

        let avg_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
        };

        (combined, avg_score)
    }

    /// Combine using intersection strategy.
    fn combine_intersection(&self, results: &[OptimizerResult]) -> (SmallVec<[u32; 8]>, f64) {
        if results.is_empty() {
            return (SmallVec::new(), 0.0);
        }

        // Start with first result's indices
        let mut intersection: std::collections::HashSet<u32> =
            results[0].demo_indices.iter().copied().collect();

        // Intersect with each subsequent result
        for result in &results[1..] {
            let other: std::collections::HashSet<u32> =
                result.demo_indices.iter().copied().collect();
            intersection = intersection.intersection(&other).copied().collect();
        }

        let combined: SmallVec<[u32; 8]> = intersection.iter().copied().collect();

        let avg_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
        };

        (combined, avg_score)
    }
}

/// Result of ensemble optimization.
#[derive(Clone)]
pub struct EnsembleResult {
    /// Combined demo indices
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Combined score
    pub score: f64,
    /// Individual optimizer results
    pub individual_results: Vec<OptimizerResult>,
    /// Strategy used
    pub strategy: CombineStrategy,
}

impl<'a> Optimizer for Ensemble<'a> {
    type Output<'b>
        = OptimizationResult
    where
        'a: 'b;
    type OptimizeFut<'b>
        =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<OptimizationResult>> + Send + 'b>>
    where
        'a: 'b;

    fn optimize<'b>(&'b self, trainset: &'b ExampleSet<'b>) -> Self::OptimizeFut<'b> {
        Box::pin(async move {
            let result = self.run(trainset).await?;
            Ok(OptimizationResult {
                demo_indices: result.demo_indices,
                score: result.score,
                iterations: result.individual_results.len() as u16,
            })
        })
    }

    fn name(&self) -> &'static str {
        "Ensemble"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple optimizer for testing.
    struct MockOptimizer {
        demo_indices: SmallVec<[u32; 8]>,
        score: f64,
        name: &'static str,
    }

    impl Optimizer for MockOptimizer {
        type Output<'a> = OptimizationResult;
        type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

        fn optimize<'a>(&'a self, _trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
            std::future::ready(Ok(OptimizationResult {
                demo_indices: self.demo_indices.clone(),
                score: self.score,
                iterations: 1,
            }))
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    impl<'a> OptimizeInto<'a> for MockOptimizer {
        fn optimize_into(
            &'a self,
            trainset: &'a ExampleSet<'a>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<OptimizationResult>> + Send + 'a>,
        > {
            Box::pin(async move { self.optimize(trainset).await })
        }
    }

    #[test]
    fn test_ensemble_creation() {
        let ensemble: Ensemble<'_> = Ensemble::default();
        assert_eq!(ensemble.name(), "Ensemble");
        assert!(ensemble.is_empty());
    }

    #[test]
    fn test_ensemble_config() {
        let config = EnsembleConfig::new()
            .with_strategy(CombineStrategy::Union)
            .with_parallel(false);
        assert!(matches!(config.strategy, CombineStrategy::Union));
        assert!(!config.parallel);
    }

    #[tokio::test]
    async fn test_ensemble_best_strategy() {
        let ensemble = Ensemble::new(EnsembleConfig::new().with_strategy(CombineStrategy::Best))
            .add(MockOptimizer {
                demo_indices: SmallVec::from_slice(&[0, 1]),
                score: 0.7,
                name: "opt1",
            })
            .add(MockOptimizer {
                demo_indices: SmallVec::from_slice(&[2, 3]),
                score: 0.9,
                name: "opt2",
            });

        let buffer = crate::buffer::Buffer::Static(b"test data");
        let trainset = ExampleSet::from_buffer(buffer, 5);

        let result = ensemble.run(&trainset).await.unwrap();

        // Should pick opt2's result (higher score)
        assert_eq!(result.demo_indices.as_slice(), &[2, 3]);
        assert!((result.score - 0.9).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_ensemble_union_strategy() {
        let ensemble = Ensemble::new(EnsembleConfig::new().with_strategy(CombineStrategy::Union))
            .add(MockOptimizer {
                demo_indices: SmallVec::from_slice(&[0, 1]),
                score: 0.7,
                name: "opt1",
            })
            .add(MockOptimizer {
                demo_indices: SmallVec::from_slice(&[1, 2]),
                score: 0.8,
                name: "opt2",
            });

        let buffer = crate::buffer::Buffer::Static(b"test data");
        let trainset = ExampleSet::from_buffer(buffer, 5);

        let result = ensemble.run(&trainset).await.unwrap();

        // Should have union (0, 1, 2)
        assert!(result.demo_indices.contains(&0));
        assert!(result.demo_indices.contains(&1));
        assert!(result.demo_indices.contains(&2));
    }

    #[tokio::test]
    async fn test_ensemble_intersection_strategy() {
        let ensemble =
            Ensemble::new(EnsembleConfig::new().with_strategy(CombineStrategy::Intersection))
                .add(MockOptimizer {
                    demo_indices: SmallVec::from_slice(&[0, 1, 2]),
                    score: 0.7,
                    name: "opt1",
                })
                .add(MockOptimizer {
                    demo_indices: SmallVec::from_slice(&[1, 2, 3]),
                    score: 0.8,
                    name: "opt2",
                });

        let buffer = crate::buffer::Buffer::Static(b"test data");
        let trainset = ExampleSet::from_buffer(buffer, 5);

        let result = ensemble.run(&trainset).await.unwrap();

        // Should have intersection (1, 2)
        assert!(result.demo_indices.contains(&1));
        assert!(result.demo_indices.contains(&2));
        assert!(!result.demo_indices.contains(&0));
        assert!(!result.demo_indices.contains(&3));
    }
}
