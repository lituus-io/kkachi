// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Best of N candidate generation with scoring.
//!
//! This module provides the [`best_of`] entry point for generating multiple
//! candidates and selecting the best one based on a scoring function.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, best_of};
//!
//! let llm = MockLlm::new(|_, _| "Generated response".to_string());
//!
//! let result = best_of(&llm, "Write a haiku")
//!     .metric(|output| if output.lines().count() == 3 { 1.0 } else { 0.0 })
//!     .go();
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::validate::{NoValidation, Validate};
use smallvec::SmallVec;

/// Entry point for Best of N generation.
///
/// Creates a builder that generates N candidates and selects the best one.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, best_of, checks};
///
/// let llm = MockLlm::new(|_, _| "fn main() {}".to_string());
///
/// let result = best_of(&llm, "Write code").n(5)
///     .validate(checks().require("fn "))
///     .go();
/// ```
pub fn best_of<'a, L: Llm>(
    llm: &'a L,
    prompt: &'a str,
) -> BestOf<'a, L, NoValidation, DefaultScorer> {
    BestOf::new(llm, prompt)
}

/// Scorer trait for evaluating candidates.
///
/// Implementations should return a score between 0.0 and 1.0.
pub trait Scorer: Send + Sync {
    /// Score the given output.
    fn score(&self, output: &str) -> f64;
}

/// Default scorer that returns 0.5 for all outputs.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultScorer;

impl Scorer for DefaultScorer {
    fn score(&self, _output: &str) -> f64 {
        0.5
    }
}

/// Scorer created from a closure.
pub struct FnScorer<F>(pub F);

impl<F: Fn(&str) -> f64 + Send + Sync> Scorer for FnScorer<F> {
    fn score(&self, output: &str) -> f64 {
        (self.0)(output)
    }
}

/// Configuration for Best of N generation.
#[derive(Clone)]
pub struct BestOfConfig {
    /// Whether to use Chain of Thought for each candidate.
    pub with_reasoning: bool,
    /// Weight for scorer vs validator (scorer_weight).
    pub scorer_weight: f64,
    /// Weight for validator (1.0 - scorer_weight).
    pub validator_weight: f64,
    /// Whether to inject diversity hints for each candidate.
    pub diverse: bool,
    /// Language to extract from code fences before validation (e.g., "rust").
    pub extract_lang: Option<String>,
    /// Whether to generate candidates in parallel using threads.
    pub parallel: bool,
}

impl Default for BestOfConfig {
    fn default() -> Self {
        Self {
            with_reasoning: false,
            scorer_weight: 0.5,
            validator_weight: 0.5,
            diverse: true,
            extract_lang: None,
            parallel: false,
        }
    }
}

/// Best of N generation builder.
///
/// Generates N candidates and selects the best based on combined
/// scorer and validator scores.
pub struct BestOf<'a, L: Llm, V: Validate, S: Scorer> {
    llm: &'a L,
    prompt: &'a str,
    n: usize,
    validator: V,
    scorer: S,
    config: BestOfConfig,
}

impl<'a, L: Llm> BestOf<'a, L, NoValidation, DefaultScorer> {
    /// Create a new Best of N builder with default N=3.
    pub fn new(llm: &'a L, prompt: &'a str) -> Self {
        Self {
            llm,
            prompt,
            n: 3,
            validator: NoValidation,
            scorer: DefaultScorer,
            config: BestOfConfig::default(),
        }
    }
}

impl<'a, L: Llm, V: Validate, S: Scorer> BestOf<'a, L, V, S> {
    /// Set the number of candidates to generate (default: 3).
    pub fn n(mut self, n: usize) -> Self {
        self.n = n.max(1);
        self
    }

    /// Set a validator for candidates.
    ///
    /// The validator score is combined with the scorer to determine
    /// the best candidate.
    pub fn validate<V2: Validate>(self, validator: V2) -> BestOf<'a, L, V2, S> {
        BestOf {
            llm: self.llm,
            prompt: self.prompt,
            n: self.n,
            validator,
            scorer: self.scorer,
            config: self.config,
        }
    }

    /// Set a custom scoring metric.
    ///
    /// The scorer evaluates each candidate and returns a score between
    /// 0.0 and 1.0.
    pub fn metric<F: Fn(&str) -> f64 + Send + Sync>(self, f: F) -> BestOf<'a, L, V, FnScorer<F>> {
        BestOf {
            llm: self.llm,
            prompt: self.prompt,
            n: self.n,
            validator: self.validator,
            scorer: FnScorer(f),
            config: self.config,
        }
    }

    /// Enable Chain of Thought for each candidate.
    pub fn with_reasoning(mut self) -> Self {
        self.config.with_reasoning = true;
        self
    }

    /// Set the weight for the scorer vs validator.
    ///
    /// Default is 0.5 (equal weight).
    pub fn scorer_weight(mut self, weight: f64) -> Self {
        self.config.scorer_weight = weight.clamp(0.0, 1.0);
        self.config.validator_weight = 1.0 - self.config.scorer_weight;
        self
    }

    /// Disable diversity hints between candidates.
    ///
    /// By default, each candidate receives a context hint encouraging
    /// a different approach. Disable this to generate candidates identically.
    pub fn no_diversity(mut self) -> Self {
        self.config.diverse = false;
        self
    }

    /// Enable diversity hints between candidates (the default).
    ///
    /// Each candidate receives a context hint encouraging a different approach.
    /// Use this to explicitly re-enable diversity after `.no_diversity()`.
    pub fn diverse(mut self) -> Self {
        self.config.diverse = true;
        self
    }

    /// Generate candidates in parallel using threads.
    ///
    /// When enabled, all N candidates are generated concurrently using
    /// `std::thread::scope`. Diversity hints are still applied but cannot
    /// reference previous candidates' outputs.
    pub fn parallel(mut self) -> Self {
        self.config.parallel = true;
        self
    }

    /// Extract code from markdown fences before validation.
    ///
    /// When set, the validator and scorer receive only the extracted code
    /// (from the first matching code fence), not the full LLM response.
    pub fn extract(mut self, lang: impl Into<String>) -> Self {
        self.config.extract_lang = Some(lang.into());
        self
    }

    /// Execute synchronously and return the best result.
    #[cfg(feature = "native")]
    pub fn go(self) -> BestOfResult {
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

    /// Execute synchronously and return the best result (fallback without tokio).
    #[cfg(not(feature = "native"))]
    pub fn go(self) -> BestOfResult {
        futures::executor::block_on(self.run())
    }

    /// Execute synchronously and return both result and candidate pool.
    #[cfg(feature = "native")]
    pub fn go_with_pool(self) -> (BestOfResult, CandidatePool) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.run_with_pool()))
        } else {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime")
                .block_on(self.run_with_pool())
        }
    }

    /// Execute synchronously and return both result and candidate pool (fallback).
    #[cfg(not(feature = "native"))]
    pub fn go_with_pool(self) -> (BestOfResult, CandidatePool) {
        futures::executor::block_on(self.run_with_pool())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> BestOfResult {
        let (result, _) = self.run_with_pool().await;
        result
    }

    /// Execute asynchronously with candidate pool.
    pub async fn run_with_pool(self) -> (BestOfResult, CandidatePool) {
        use crate::recursive::rewrite::extract_code;

        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("best_of", n = self.n).entered();

        let mut candidates: SmallVec<[ScoredCandidate; 8]> = SmallVec::new();
        let mut best_idx = 0;
        let mut best_score = f64::MIN;
        let mut total_tokens = 0u32;
        let mut error: Option<String> = None;

        let prompt = if self.config.with_reasoning {
            format!("{}\n\nLet's think step by step.", self.prompt)
        } else {
            self.prompt.to_string()
        };

        if self.config.parallel {
            // Parallel mode: generate all candidates concurrently using FuturesUnordered
            use futures::stream::{FuturesUnordered, StreamExt};

            let style_hints = [
                "concise and minimal",
                "explicit and well-documented",
                "using a different algorithm or technique",
                "optimized for readability",
                "optimized for performance",
                "using standard library idioms",
                "using a creative or unconventional approach",
                "with extensive error handling",
            ];

            let contexts: Vec<String> = (0..self.n)
                .map(|i| {
                    if self.config.diverse && self.n > 1 {
                        let hint = style_hints[i % style_hints.len()];
                        format!(
                            "Generate candidate {} of {}. Style: {}.",
                            i + 1,
                            self.n,
                            hint
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

                let text_to_score = if let Some(ref lang) = self.config.extract_lang {
                    extract_code(&output.text, lang)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| output.text.clone())
                } else {
                    output.text.clone()
                };

                let scorer_score = self.scorer.score(&text_to_score).clamp(0.0, 1.0);
                let validator_score = self.validator.validate(&text_to_score);
                let combined = scorer_score * self.config.scorer_weight
                    + validator_score.value * self.config.validator_weight;

                #[cfg(feature = "tracing")]
                tracing::debug!(candidate = i, score = combined, "best_of candidate scored");

                candidates.push(ScoredCandidate {
                    index: i,
                    output: output.text,
                    scorer_score,
                    validator_score: validator_score.value,
                    combined_score: combined,
                    feedback: validator_score.feedback_str().map(|s| s.to_string()),
                });

                if combined > best_score {
                    best_score = combined;
                    best_idx = candidates.len() - 1;
                }
            }
        } else {
            // Sequential mode: generate candidates one at a time
            for i in 0..self.n {
                // Build diversity context for this candidate
                let context = if self.config.diverse && self.n > 1 {
                    let style_hints = [
                        "concise and minimal",
                        "explicit and well-documented",
                        "using a different algorithm or technique",
                        "optimized for readability",
                        "optimized for performance",
                        "using standard library idioms",
                        "using a creative or unconventional approach",
                        "with extensive error handling",
                    ];
                    let hint = style_hints[i % style_hints.len()];
                    let mut ctx = format!(
                        "Generate candidate {} of {}. Style: {}.",
                        i + 1,
                        self.n,
                        hint
                    );
                    // Include previous outputs so the LLM avoids duplicates
                    if i > 0 && !candidates.is_empty() {
                        ctx.push_str("\n\nAvoid repeating these previous approaches:\n");
                        for prev in candidates.iter().take(3) {
                            let snippet = if prev.output.len() > 80 {
                                // Find a valid char boundary at or before byte 80
                                let mut end = 80;
                                while end > 0 && !prev.output.is_char_boundary(end) {
                                    end -= 1;
                                }
                                &prev.output[..end]
                            } else {
                                &prev.output
                            };
                            ctx.push_str(&format!("- {}\n", snippet.replace('\n', " ")));
                        }
                    }
                    ctx
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
                let text_to_score = if let Some(ref lang) = self.config.extract_lang {
                    extract_code(&output.text, lang)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| output.text.clone())
                } else {
                    output.text.clone()
                };

                let scorer_score = self.scorer.score(&text_to_score).clamp(0.0, 1.0);
                let validator_score = self.validator.validate(&text_to_score);
                let combined = scorer_score * self.config.scorer_weight
                    + validator_score.value * self.config.validator_weight;

                #[cfg(feature = "tracing")]
                tracing::debug!(candidate = i, score = combined, "best_of candidate scored");

                candidates.push(ScoredCandidate {
                    index: i,
                    output: output.text,
                    scorer_score,
                    validator_score: validator_score.value,
                    combined_score: combined,
                    feedback: validator_score.feedback_str().map(|s| s.to_string()),
                });

                if combined > best_score {
                    best_score = combined;
                    best_idx = candidates.len() - 1;
                }
            }
        }

        if candidates.is_empty() {
            return (
                BestOfResult {
                    output: String::new(),
                    score: 0.0,
                    candidates_generated: 0,
                    tokens: total_tokens,
                    error,
                },
                CandidatePool {
                    candidates: SmallVec::new(),
                    total_tokens,
                },
            );
        }

        let best = candidates[best_idx].output.clone();
        let pool = CandidatePool {
            candidates,
            total_tokens,
        };

        #[cfg(feature = "tracing")]
        tracing::info!(
            best_score,
            candidates = self.n,
            tokens = total_tokens,
            "best_of complete"
        );

        (
            BestOfResult {
                output: best,
                score: best_score,
                candidates_generated: self.n,
                tokens: total_tokens,
                error,
            },
            pool,
        )
    }
}

/// Result of Best of N generation.
#[derive(Debug, Clone)]
pub struct BestOfResult {
    /// The best candidate output.
    pub output: String,
    /// Combined score of the best candidate.
    pub score: f64,
    /// Number of candidates that were generated.
    pub candidates_generated: usize,
    /// Total tokens used.
    pub tokens: u32,
    /// Error message if some generations failed.
    pub error: Option<String>,
}

impl BestOfResult {
    /// Check if the generation succeeded.
    pub fn success(&self) -> bool {
        !self.output.is_empty() && self.error.is_none()
    }
}

/// A scored candidate from the generation pool.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// Index in generation order.
    pub index: usize,
    /// The generated output.
    pub output: String,
    /// Score from the scorer function.
    pub scorer_score: f64,
    /// Score from the validator.
    pub validator_score: f64,
    /// Combined score.
    pub combined_score: f64,
    /// Validation feedback if any.
    pub feedback: Option<String>,
}

/// Pool of all generated candidates for recall/precision tuning.
#[derive(Debug, Clone)]
pub struct CandidatePool {
    candidates: SmallVec<[ScoredCandidate; 8]>,
    total_tokens: u32,
}

impl CandidatePool {
    /// Get all candidates.
    pub fn candidates(&self) -> &[ScoredCandidate] {
        &self.candidates
    }

    /// Get total tokens used.
    pub fn total_tokens(&self) -> u32 {
        self.total_tokens
    }

    /// Filter candidates by minimum combined score threshold.
    pub fn filter_by_threshold(&self, threshold: f64) -> Vec<&ScoredCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.combined_score >= threshold)
            .collect()
    }

    /// Get the best candidate.
    pub fn best(&self) -> Option<&ScoredCandidate> {
        self.candidates
            .iter()
            .max_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap())
    }

    /// Get the top K candidates sorted by score.
    pub fn top_k(&self, k: usize) -> Vec<&ScoredCandidate> {
        let mut sorted: Vec<_> = self.candidates.iter().collect();
        sorted.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        sorted.truncate(k);
        sorted
    }

    /// Get statistics about the candidate pool.
    pub fn stats(&self) -> PoolStats {
        if self.candidates.is_empty() {
            return PoolStats {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let scores: Vec<f64> = self.candidates.iter().map(|c| c.combined_score).collect();
        let count = scores.len();
        let mean = scores.iter().sum::<f64>() / count as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / count as f64;

        PoolStats {
            count,
            mean,
            std_dev: variance.sqrt(),
            min: scores.iter().copied().fold(f64::MAX, f64::min),
            max: scores.iter().copied().fold(f64::MIN, f64::max),
        }
    }

    /// Check if any candidate passed validation.
    pub fn has_valid(&self) -> bool {
        self.candidates.iter().any(|c| c.validator_score >= 1.0)
    }

    /// Get all passing candidates.
    pub fn passing(&self) -> Vec<&ScoredCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.validator_score >= 1.0)
            .collect()
    }
}

/// Statistics about a candidate pool.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Number of candidates.
    pub count: usize,
    /// Mean combined score.
    pub mean: f64,
    /// Standard deviation of scores.
    pub std_dev: f64,
    /// Minimum score.
    pub min: f64,
    /// Maximum score.
    pub max: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::checks::checks;
    use crate::recursive::llm::MockLlm;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_best_of_basic() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            format!("Response {}", n)
        });

        let result = best_of(&llm, "Generate").go();

        assert!(!result.output.is_empty());
        assert_eq!(result.candidates_generated, 3);
    }

    #[test]
    fn test_best_of_with_scorer() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "short".to_string(),
                1 => "medium length".to_string(),
                2 => "this is the longest response".to_string(),
                _ => "default".to_string(),
            }
        });

        let result = best_of(&llm, "Generate")
            .metric(|output| output.len() as f64 / 30.0)
            .go();

        // Best should be the longest
        assert!(result.output.contains("longest"));
    }

    #[test]
    fn test_best_of_with_pool() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            format!("fn test{}() {{}}", n)
        });

        let (result, pool) = best_of(&llm, "Write code")
            .n(5)
            .validate(checks().require("fn "))
            .go_with_pool();

        assert!(result.score >= 0.5);
        assert_eq!(pool.candidates().len(), 5);

        let stats = pool.stats();
        assert_eq!(stats.count, 5);
        assert!(stats.mean > 0.0);
    }

    #[test]
    fn test_pool_filtering() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n % 2 == 0 { "good" } else { "bad" }.to_string()
        });

        let (_, pool) = best_of(&llm, "Generate")
            .n(4)
            .metric(|output| if output == "good" { 1.0 } else { 0.0 })
            .go_with_pool();

        // Combined score = scorer * 0.5 + validator * 0.5
        // "good": 1.0 * 0.5 + 1.0 * 0.5 = 1.0
        // "bad": 0.0 * 0.5 + 1.0 * 0.5 = 0.5
        // So threshold 0.75 filters to only "good" outputs
        let high_scorers = pool.filter_by_threshold(0.75);
        assert_eq!(high_scorers.len(), 2);

        let top_2 = pool.top_k(2);
        assert_eq!(top_2.len(), 2);
    }

    #[test]
    fn test_scorer_weight() {
        let llm = MockLlm::new(|_, _| "test".to_string());

        let builder = best_of(&llm, "test").n(1).scorer_weight(0.8);

        assert!((builder.config.scorer_weight - 0.8).abs() < f64::EPSILON);
        assert!((builder.config.validator_weight - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_pool_stats() {
        let pool = CandidatePool {
            candidates: SmallVec::new(),
            total_tokens: 0,
        };

        let stats = pool.stats();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_default_scorer() {
        let scorer = DefaultScorer;
        assert!((scorer.score("anything") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fn_scorer() {
        let scorer = FnScorer(|s: &str| s.len() as f64 / 10.0);
        assert!((scorer.score("hello") - 0.5).abs() < f64::EPSILON);
    }
}
