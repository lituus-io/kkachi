// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multi-chain ensemble with aggregation.
//!
//! This module provides the [`ensemble`] entry point for generating multiple
//! responses and aggregating them using various strategies like majority vote.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, ensemble, Aggregate};
//!
//! let llm = MockLlm::new(|_, _| "Paris".to_string());
//!
//! let result = ensemble(&llm, "What is the capital of France?").n(5)
//!     .aggregate(Aggregate::MajorityVote)
//!     .go();
//!
//! assert!(result.output.contains("Paris"));
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::validate::{NoValidation, Validate};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Entry point for ensemble generation.
///
/// Creates a builder that generates N responses and aggregates them.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, ensemble};
///
/// let llm = MockLlm::new(|_, _| "42".to_string());
///
/// let result = ensemble(&llm, "What is the answer?").go();
/// ```
pub fn ensemble<'a, L: Llm>(llm: &'a L, prompt: &'a str) -> Ensemble<'a, L, NoValidation> {
    Ensemble::new(llm, prompt)
}

/// Aggregation strategy for combining multiple responses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Aggregate {
    /// Select the most common answer (default).
    #[default]
    MajorityVote,
    /// Select the longest answer.
    LongestAnswer,
    /// Select the first answer that passes validation.
    FirstSuccess,
    /// Require unanimous agreement.
    Unanimous,
    /// Select the shortest answer.
    ShortestAnswer,
}

/// Configuration for ensemble generation.
#[derive(Clone)]
pub struct EnsembleConfig {
    /// Whether to use Chain of Thought for each chain.
    pub with_reasoning: bool,
    /// Whether to normalize answers before comparison.
    pub normalize: bool,
    /// Minimum agreement ratio to consider result valid.
    pub min_agreement: f64,
    /// Whether to inject diversity hints for each chain.
    pub diverse: bool,
    /// Language to extract from code fences before validation (e.g., "rust").
    pub extract_lang: Option<String>,
    /// Whether to generate chains in parallel using threads.
    pub parallel: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            with_reasoning: false,
            normalize: true,
            min_agreement: 0.0,
            diverse: true,
            extract_lang: None,
            parallel: false,
        }
    }
}

/// Ensemble generation builder.
///
/// Generates multiple responses and aggregates them using the specified strategy.
pub struct Ensemble<'a, L: Llm, V: Validate> {
    llm: &'a L,
    prompt: &'a str,
    n: usize,
    validator: V,
    aggregate: Aggregate,
    config: EnsembleConfig,
}

impl<'a, L: Llm> Ensemble<'a, L, NoValidation> {
    /// Create a new ensemble builder with default N=3.
    pub fn new(llm: &'a L, prompt: &'a str) -> Self {
        Self {
            llm,
            prompt,
            n: 3,
            validator: NoValidation,
            aggregate: Aggregate::default(),
            config: EnsembleConfig::default(),
        }
    }
}

impl<'a, L: Llm, V: Validate> Ensemble<'a, L, V> {
    /// Set the number of chains to generate (default: 3).
    pub fn n(mut self, n: usize) -> Self {
        self.n = n.max(1);
        self
    }

    /// Set a validator for responses.
    pub fn validate<V2: Validate>(self, validator: V2) -> Ensemble<'a, L, V2> {
        Ensemble {
            llm: self.llm,
            prompt: self.prompt,
            n: self.n,
            validator,
            aggregate: self.aggregate,
            config: self.config,
        }
    }

    /// Set the aggregation strategy.
    pub fn aggregate(mut self, strategy: Aggregate) -> Self {
        self.aggregate = strategy;
        self
    }

    /// Enable Chain of Thought for each chain.
    pub fn with_reasoning(mut self) -> Self {
        self.config.with_reasoning = true;
        self
    }

    /// Disable answer normalization.
    ///
    /// By default, answers are normalized (trimmed, lowercased) before
    /// comparison. Disabling this compares exact strings.
    pub fn no_normalize(mut self) -> Self {
        self.config.normalize = false;
        self
    }

    /// Set minimum agreement ratio for valid result.
    ///
    /// If agreement ratio is below this threshold, the result is marked
    /// as low confidence.
    pub fn min_agreement(mut self, ratio: f64) -> Self {
        self.config.min_agreement = ratio.clamp(0.0, 1.0);
        self
    }

    /// Disable diversity hints between chains.
    pub fn no_diversity(mut self) -> Self {
        self.config.diverse = false;
        self
    }

    /// Enable diversity hints between chains (the default).
    ///
    /// Use this to explicitly re-enable diversity after `.no_diversity()`.
    pub fn diverse(mut self) -> Self {
        self.config.diverse = true;
        self
    }

    /// Generate chains in parallel using threads.
    ///
    /// When enabled, all N chains are generated concurrently using
    /// `std::thread::scope`.
    pub fn parallel(mut self) -> Self {
        self.config.parallel = true;
        self
    }

    /// Extract code from markdown fences before validation/comparison.
    pub fn extract(mut self, lang: impl Into<String>) -> Self {
        self.config.extract_lang = Some(lang.into());
        self
    }

    /// Execute synchronously and return the result.
    #[cfg(feature = "native")]
    pub fn go(self) -> EnsembleResult {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.run()))
        } else {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime")
                .block_on(self.run())
        }
    }

    /// Execute synchronously and return the result (fallback without tokio).
    #[cfg(not(feature = "native"))]
    pub fn go(self) -> EnsembleResult {
        futures::executor::block_on(self.run())
    }

    /// Execute synchronously and return result with consensus pool.
    #[cfg(feature = "native")]
    pub fn go_with_consensus(self) -> (EnsembleResult, ConsensusPool) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.run_with_consensus()))
        } else {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime")
                .block_on(self.run_with_consensus())
        }
    }

    /// Execute synchronously and return result with consensus pool (fallback).
    #[cfg(not(feature = "native"))]
    pub fn go_with_consensus(self) -> (EnsembleResult, ConsensusPool) {
        futures::executor::block_on(self.run_with_consensus())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> EnsembleResult {
        let (result, _) = self.run_with_consensus().await;
        result
    }

    /// Execute asynchronously with consensus pool.
    pub async fn run_with_consensus(self) -> (EnsembleResult, ConsensusPool) {
        #[cfg(feature = "tracing")]
        let _span =
            tracing::info_span!("ensemble", n = self.n, aggregate = ?self.aggregate).entered();

        let mut chains: SmallVec<[ChainResult; 8]> = SmallVec::new();
        let mut total_tokens = 0u32;
        let mut error: Option<String> = None;

        let prompt = if self.config.with_reasoning {
            format!("{}\n\nLet's think step by step.", self.prompt)
        } else {
            self.prompt.to_string()
        };

        if self.config.parallel {
            // Parallel mode: generate all chains concurrently using FuturesUnordered
            use futures::stream::{FuturesUnordered, StreamExt};

            let contexts: Vec<String> = (0..self.n)
                .map(|i| {
                    if self.config.diverse && self.n > 1 {
                        format!(
                            "Response {} of {}. Provide your independent answer.",
                            i + 1,
                            self.n
                        )
                    } else {
                        String::new()
                    }
                })
                .collect();

            let mut futs = FuturesUnordered::new();
            for (i, ctx) in contexts.iter().enumerate() {
                let fut = self.llm.generate(&prompt, ctx, None);
                futs.push(async move { (i, fut.await) });
            }

            let mut outputs: Vec<(usize, crate::error::Result<crate::recursive::llm::LmOutput>)> =
                Vec::with_capacity(self.n);
            while let Some(result) = futs.next().await {
                outputs.push(result);
            }

            for (i, result) in outputs {
                let output = match result {
                    Ok(out) => out,
                    Err(e) => {
                        error = Some(e.to_string());
                        continue;
                    }
                };

                total_tokens += output.prompt_tokens + output.completion_tokens;

                let raw_answer = if let Some(ref lang) = self.config.extract_lang {
                    use crate::recursive::rewrite::extract_code;
                    extract_code(&output.text, lang)
                        .map(|s| s.to_string())
                        .unwrap_or(output.text)
                } else {
                    output.text
                };

                let normalized = if self.config.normalize {
                    Self::normalize_answer(&raw_answer)
                } else {
                    raw_answer.clone()
                };

                let validator_score = self.validator.validate(&raw_answer);

                chains.push(ChainResult {
                    index: i,
                    raw_answer,
                    normalized_answer: normalized,
                    validator_score: validator_score.value,
                    agrees_with_majority: false,
                });
            }
        } else {
            // Sequential mode: generate chains one at a time
            for i in 0..self.n {
                // Build diversity context for this chain
                let context = if self.config.diverse && self.n > 1 {
                    format!(
                        "Response {} of {}. Provide your independent answer.",
                        i + 1,
                        self.n
                    )
                } else {
                    String::new()
                };

                let output = match self.llm.generate(&prompt, &context, None).await {
                    Ok(out) => out,
                    Err(e) => {
                        error = Some(e.to_string());
                        continue;
                    }
                };

                total_tokens += output.prompt_tokens + output.completion_tokens;

                // Extract code from markdown if configured
                let raw_answer = if let Some(ref lang) = self.config.extract_lang {
                    use crate::recursive::rewrite::extract_code;
                    extract_code(&output.text, lang)
                        .map(|s| s.to_string())
                        .unwrap_or(output.text)
                } else {
                    output.text
                };

                let normalized = if self.config.normalize {
                    Self::normalize_answer(&raw_answer)
                } else {
                    raw_answer.clone()
                };

                let validator_score = self.validator.validate(&raw_answer);

                chains.push(ChainResult {
                    index: i,
                    raw_answer,
                    normalized_answer: normalized,
                    validator_score: validator_score.value,
                    agrees_with_majority: false, // Will be set later
                });
            }
        }

        if chains.is_empty() {
            return (
                EnsembleResult {
                    output: String::new(),
                    chains_generated: 0,
                    tokens: total_tokens,
                    agreement_ratio: 0.0,
                    error,
                },
                ConsensusPool {
                    chains: SmallVec::new(),
                    selected_answer: String::new(),
                    total_tokens,
                },
            );
        }

        // Count votes for normalized answers
        let mut votes: HashMap<&str, usize> = HashMap::new();
        for chain in &chains {
            *votes.entry(&chain.normalized_answer).or_default() += 1;
        }

        // Select winner based on strategy
        let (selected, agreement_count) = match self.aggregate {
            Aggregate::MajorityVote => {
                let (answer, count) = votes
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(a, c)| (*a, *c))
                    .unwrap_or(("", 0));
                // Find the original answer (not normalized)
                let original = chains
                    .iter()
                    .find(|c| c.normalized_answer == answer)
                    .map(|c| c.raw_answer.clone())
                    .unwrap_or_default();
                (original, count)
            }
            Aggregate::LongestAnswer => {
                let chain = chains.iter().max_by_key(|c| c.raw_answer.len());
                match chain {
                    Some(c) => (c.raw_answer.clone(), 1),
                    None => (String::new(), 0),
                }
            }
            Aggregate::ShortestAnswer => {
                let chain = chains.iter().min_by_key(|c| c.raw_answer.len());
                match chain {
                    Some(c) => (c.raw_answer.clone(), 1),
                    None => (String::new(), 0),
                }
            }
            Aggregate::FirstSuccess => {
                let chain = chains.iter().find(|c| c.validator_score >= 1.0);
                match chain {
                    Some(c) => (c.raw_answer.clone(), 1),
                    None => {
                        // Fall back to first answer
                        let first = chains
                            .first()
                            .map(|c| c.raw_answer.clone())
                            .unwrap_or_default();
                        (first, 1)
                    }
                }
            }
            Aggregate::Unanimous => {
                // Check if all answers are the same
                let first_normalized = chains.first().map(|c| &c.normalized_answer);
                let all_same = first_normalized
                    .map(|first| chains.iter().all(|c| &c.normalized_answer == first))
                    .unwrap_or(false);

                if all_same {
                    let answer = chains
                        .first()
                        .map(|c| c.raw_answer.clone())
                        .unwrap_or_default();
                    (answer, chains.len())
                } else {
                    // Return empty if not unanimous
                    (String::new(), 0)
                }
            }
        };

        let agreement_ratio = agreement_count as f64 / chains.len() as f64;

        // Mark which chains agree with the selected answer
        let selected_normalized = if self.config.normalize {
            Self::normalize_answer(&selected)
        } else {
            selected.clone()
        };

        for chain in &mut chains {
            chain.agrees_with_majority = chain.normalized_answer == selected_normalized;
        }

        let pool = ConsensusPool {
            chains,
            selected_answer: selected.clone(),
            total_tokens,
        };

        #[cfg(feature = "tracing")]
        tracing::info!(
            chains = self.n,
            agreement = agreement_ratio,
            tokens = total_tokens,
            "ensemble complete"
        );

        (
            EnsembleResult {
                output: selected,
                chains_generated: self.n,
                tokens: total_tokens,
                agreement_ratio,
                error,
            },
            pool,
        )
    }

    /// Normalize an answer for comparison.
    fn normalize_answer(answer: &str) -> String {
        answer.trim().to_lowercase()
    }
}

/// Result of ensemble generation.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// The selected answer.
    pub output: String,
    /// Number of chains generated.
    pub chains_generated: usize,
    /// Total tokens used.
    pub tokens: u32,
    /// Agreement ratio (0.0-1.0).
    pub agreement_ratio: f64,
    /// Error message if some generations failed.
    pub error: Option<String>,
}

impl EnsembleResult {
    /// Check if there was strong agreement (> 50%).
    pub fn has_consensus(&self) -> bool {
        self.agreement_ratio > 0.5
    }

    /// Check if the result succeeded.
    pub fn success(&self) -> bool {
        !self.output.is_empty() && self.error.is_none()
    }
}

/// Result for a single chain in the ensemble.
#[derive(Debug, Clone)]
pub struct ChainResult {
    /// Index in generation order.
    pub index: usize,
    /// Raw (unnormalized) answer.
    pub raw_answer: String,
    /// Normalized answer for comparison.
    pub normalized_answer: String,
    /// Validation score.
    pub validator_score: f64,
    /// Whether this chain agrees with the selected answer.
    pub agrees_with_majority: bool,
}

/// Pool of all chains for consensus analysis.
#[derive(Debug, Clone)]
pub struct ConsensusPool {
    chains: SmallVec<[ChainResult; 8]>,
    selected_answer: String,
    total_tokens: u32,
}

impl ConsensusPool {
    /// Get all chains.
    pub fn chains(&self) -> &[ChainResult] {
        &self.chains
    }

    /// Get the selected answer.
    pub fn selected(&self) -> &str {
        &self.selected_answer
    }

    /// Get total tokens used.
    pub fn total_tokens(&self) -> u32 {
        self.total_tokens
    }

    /// Get the agreement ratio.
    pub fn agreement_ratio(&self) -> f64 {
        if self.chains.is_empty() {
            return 0.0;
        }
        let agreeing = self
            .chains
            .iter()
            .filter(|c| c.agrees_with_majority)
            .count();
        agreeing as f64 / self.chains.len() as f64
    }

    /// Check if there is unanimous agreement.
    pub fn has_unanimous_agreement(&self) -> bool {
        (self.agreement_ratio() - 1.0).abs() < f64::EPSILON
    }

    /// Get chains that disagree with the majority.
    pub fn dissenting_chains(&self) -> impl Iterator<Item = &ChainResult> {
        self.chains.iter().filter(|c| !c.agrees_with_majority)
    }

    /// Get chains that agree with the majority.
    pub fn agreeing_chains(&self) -> impl Iterator<Item = &ChainResult> {
        self.chains.iter().filter(|c| c.agrees_with_majority)
    }

    /// Get unique answers and their counts.
    pub fn vote_counts(&self) -> HashMap<&str, usize> {
        let mut counts = HashMap::new();
        for chain in &self.chains {
            *counts.entry(chain.normalized_answer.as_str()).or_default() += 1;
        }
        counts
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_ensemble_majority_vote() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n % 3 {
                0 | 1 => "Paris".to_string(), // 2/3 = majority
                _ => "London".to_string(),
            }
        });

        let result = ensemble(&llm, "Capital of France?")
            .aggregate(Aggregate::MajorityVote)
            .go();

        assert!(result.output.to_lowercase().contains("paris"));
        assert!(result.agreement_ratio > 0.5);
    }

    #[test]
    fn test_ensemble_with_consensus() {
        let llm = MockLlm::new(|_, _| "42".to_string());

        let (result, pool) = ensemble(&llm, "Answer?").n(5).go_with_consensus();

        assert_eq!(result.chains_generated, 5);
        assert!(pool.has_unanimous_agreement());
        assert!((pool.agreement_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ensemble_longest_answer() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "short".to_string(),
                1 => "medium length".to_string(),
                2 => "this is the longest answer of them all".to_string(),
                _ => "x".to_string(),
            }
        });

        let result = ensemble(&llm, "Answer")
            .aggregate(Aggregate::LongestAnswer)
            .go();

        assert!(result.output.contains("longest"));
    }

    #[test]
    fn test_ensemble_unanimous() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n < 2 { "same" } else { "different" }.to_string()
        });

        let result = ensemble(&llm, "Answer")
            .aggregate(Aggregate::Unanimous)
            .go();

        // Should be empty because not unanimous
        assert!(result.output.is_empty());
    }

    #[test]
    fn test_ensemble_normalization() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "PARIS".to_string(),
                1 => "paris".to_string(),
                2 => "Paris".to_string(),
                _ => "london".to_string(),
            }
        });

        let (result, pool) = ensemble(&llm, "Capital?").go_with_consensus();

        // All should be treated as same answer after normalization
        assert!(pool.has_unanimous_agreement());
        assert!((result.agreement_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ensemble_no_normalize() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "PARIS".to_string(),
                1 => "paris".to_string(),
                2 => "Paris".to_string(),
                _ => "x".to_string(),
            }
        });

        let (_, pool) = ensemble(&llm, "Capital?")
            .no_normalize()
            .go_with_consensus();

        // Without normalization, all answers are different
        assert!(!pool.has_unanimous_agreement());
    }

    #[test]
    fn test_consensus_pool_methods() {
        let llm = MockLlm::new(|_, _| "test".to_string());

        let (_, pool) = ensemble(&llm, "Test").go_with_consensus();

        assert_eq!(pool.chains().len(), 3);
        assert_eq!(pool.selected(), "test");
        assert_eq!(pool.agreeing_chains().count(), 3);
        assert_eq!(pool.dissenting_chains().count(), 0);

        let votes = pool.vote_counts();
        assert_eq!(votes.get("test"), Some(&3));
    }

    #[test]
    fn test_aggregate_default() {
        assert_eq!(Aggregate::default(), Aggregate::MajorityVote);
    }
}
