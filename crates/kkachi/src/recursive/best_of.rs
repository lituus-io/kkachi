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

use crate::recursive::engine::{GenerationConfig, GenerationEngine};
use crate::recursive::llm::Llm;
use crate::recursive::shared;
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
    /// Shared generation config (diversity, parallelism, extraction, etc.).
    pub gen: GenerationConfig,
    /// Weight for scorer vs validator (scorer_weight).
    pub scorer_weight: f64,
    /// Weight for validator (1.0 - scorer_weight).
    pub validator_weight: f64,
}

impl Default for BestOfConfig {
    fn default() -> Self {
        Self {
            gen: GenerationConfig {
                diverse: true,
                ..Default::default()
            },
            scorer_weight: 0.5,
            validator_weight: 0.5,
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
    /// Configuration for Best of N generation.
    pub config: BestOfConfig,
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
        self.config.gen.with_reasoning = true;
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
    pub fn no_diversity(mut self) -> Self {
        self.config.gen.diverse = false;
        self
    }

    /// Enable diversity hints between candidates (the default).
    pub fn diverse(mut self) -> Self {
        self.config.gen.diverse = true;
        self
    }

    /// Generate candidates in parallel using threads.
    pub fn parallel(mut self) -> Self {
        self.config.gen.parallel = true;
        self
    }

    /// Extract code from markdown fences before validation.
    pub fn extract(mut self, lang: impl Into<String>) -> Self {
        self.config.gen.extract_lang = Some(lang.into());
        self
    }

    /// Attach a skill (persistent prompt context) to this builder.
    pub fn skill(mut self, skill: &crate::recursive::skill::Skill<'_>) -> Self {
        let rendered = skill.render();
        if rendered.is_empty() {
            self.config.gen.skill_text = None;
        } else {
            self.config.gen.skill_text = Some(rendered);
        }
        self
    }

    /// Set runtime defaults applied via regex substitution before scoring.
    pub fn defaults(mut self, defaults: crate::recursive::defaults::Defaults) -> Self {
        self.config.gen.defaults = Some(defaults);
        self
    }

    /// Execute synchronously and return the best result.
    pub fn go(self) -> BestOfResult {
        shared::block_on(self.run())
    }

    /// Execute synchronously and return both result and candidate pool.
    pub fn go_with_pool(self) -> (BestOfResult, CandidatePool) {
        shared::block_on(self.run_with_pool())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> BestOfResult {
        let (result, _) = self.run_with_pool().await;
        result
    }

    /// Execute asynchronously with candidate pool.
    pub async fn run_with_pool(self) -> (BestOfResult, CandidatePool) {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("best_of", n = self.n).entered();

        // Use shared engine for candidate generation
        let engine = GenerationEngine::new(self.llm, self.prompt, self.n, &self.config.gen);
        let (raw_candidates, error) = engine.generate_candidates().await;

        let total_tokens: u32 = raw_candidates.iter().map(|c| c.tokens).sum();

        if raw_candidates.is_empty() {
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

        // Score each candidate
        let mut candidates: SmallVec<[ScoredCandidate; 8]> = SmallVec::new();
        let mut best_idx = 0;
        let mut best_score = f64::MIN;

        for raw in &raw_candidates {
            let scorer_score = self.scorer.score(&raw.transformed_text).clamp(0.0, 1.0);
            let validator_score = self.validator.validate(&raw.transformed_text);
            let combined = scorer_score * self.config.scorer_weight
                + validator_score.value * self.config.validator_weight;

            #[cfg(feature = "tracing")]
            tracing::debug!(
                candidate = raw.index,
                score = combined,
                "best_of candidate scored"
            );

            candidates.push(ScoredCandidate {
                index: raw.index,
                output: raw.raw_text.clone(),
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

        let best = shared::transform_output(
            &candidates[best_idx].output,
            None,
            self.config.gen.defaults.as_ref(),
        );
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

impl std::fmt::Display for BestOfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BestOf({} candidates, score={:.2}, tokens={})",
            self.candidates_generated, self.score, self.tokens
        )
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

    #[test]
    fn test_best_of_with_skill() {
        use crate::recursive::skill::Skill;

        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("deletionProtection") {
                "skill applied".to_string()
            } else {
                "no skill".to_string()
            }
        });

        let skill = Skill::new().instruct(
            "deletionProtection",
            "Always set deletionProtection: false.",
        );

        let result = best_of(&llm, "Generate config").n(1).skill(&skill).go();

        assert!(result.output.contains("skill applied"));
    }

    #[test]
    fn test_best_of_with_defaults() {
        use crate::recursive::defaults::Defaults;

        let llm = MockLlm::new(|_, _| "user:admin@example.com".to_string());

        let defaults = Defaults::new().set("email", r"admin@example\.com", "real@company.com");

        let result = best_of(&llm, "Generate IAM").n(1).defaults(defaults).go();

        assert!(result.output.contains("real@company.com"));
        assert!(!result.output.contains("admin@example.com"));
    }
}
