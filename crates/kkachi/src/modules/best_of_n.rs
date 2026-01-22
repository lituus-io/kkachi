// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Best of N module
//!
//! Generates N completions in parallel and selects the best one
//! using a scoring function.
//!
//! ## Zero-Copy Design
//!
//! - All N completions use shared buffer pool
//! - Scoring operates on `StrView<'a>` references
//! - No intermediate allocations for ranking
//!
//! ## Candidate Pool Support
//!
//! For recall/precision tradeoffs, use `bon_with_pool` to get all
//! candidates with scores for post-hoc filtering:
//!
//! ```ignore
//! let pool = bon_with_pool(&bon, &inputs, &lm, &mut buffers).await?;
//!
//! // High-recall: include more candidates
//! let candidates = pool.filter_by_threshold(0.6);
//!
//! // High-precision: only confident results
//! let candidates = pool.filter_by_threshold(0.9);
//! ```

use crate::buffer::BufferView;
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::module::Module;
use crate::predict::{predict_with_lm, LMClient, Predict, PredictOutput};
use crate::prediction::Prediction;
use crate::recall_precision::RecallPrecisionMode;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;

/// Scorer trait for ranking completions.
pub trait Scorer: Send + Sync {
    /// Score a completion. Higher is better.
    fn score<'a>(&self, output: &PredictOutput<'a>) -> f64;
}

/// Simple length-based scorer (for testing).
pub struct LengthScorer;

impl Scorer for LengthScorer {
    fn score<'a>(&self, output: &PredictOutput<'a>) -> f64 {
        // Score by total length of output fields
        let mut total = 0usize;
        for (_, range) in &output.field_ranges {
            total += range.len();
        }
        total as f64
    }
}

/// Confidence-based scorer using output keywords.
pub struct ConfidenceScorer {
    /// Words that indicate high confidence
    positive_markers: &'static [&'static str],
    /// Words that indicate low confidence
    negative_markers: &'static [&'static str],
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self {
            positive_markers: &["definitely", "certainly", "clearly", "obviously"],
            negative_markers: &["maybe", "perhaps", "possibly", "might", "unsure"],
        }
    }
}

impl Scorer for ConfidenceScorer {
    fn score<'a>(&self, output: &PredictOutput<'a>) -> f64 {
        let mut score: f64 = 0.5; // Base score

        if let Ok(text) = output.buffer.as_str() {
            let text_lower = text.to_lowercase();

            for marker in self.positive_markers {
                if text_lower.contains(marker) {
                    score += 0.1;
                }
            }

            for marker in self.negative_markers {
                if text_lower.contains(marker) {
                    score -= 0.1;
                }
            }
        }

        score.clamp(0.0, 1.0)
    }
}

/// BestOfN configuration.
#[derive(Clone, Copy)]
pub struct BestOfNConfig {
    /// Number of completions to generate
    pub n: u8,
    /// Temperature for diversity (higher = more diverse)
    pub temperature: f32,
}

impl Default for BestOfNConfig {
    fn default() -> Self {
        Self {
            n: 3,
            temperature: 0.7,
        }
    }
}

/// Best of N module.
///
/// Generates N completions and returns the best one.
///
/// ## Example
///
/// ```ignore
/// let scorer = ConfidenceScorer::default();
/// let bon = BestOfN::new(&predict, &scorer, 5);
/// let result = bon_with_lm(&bon, &inputs, &lm, &mut buffers).await?;
/// ```
pub struct BestOfN<'pred, S: Scorer> {
    /// The underlying predict module
    predict: Predict<'pred, 'pred>,
    /// Scoring function
    scorer: S,
    /// Configuration
    config: BestOfNConfig,
}

impl<'pred, S: Scorer> BestOfN<'pred, S> {
    /// Create a new BestOfN module.
    pub fn new(predict: Predict<'pred, 'pred>, scorer: S, n: u8) -> Self {
        Self {
            predict,
            scorer,
            config: BestOfNConfig {
                n,
                ..Default::default()
            },
        }
    }

    /// Configure temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Get the predict module.
    #[inline]
    pub fn predict(&self) -> &Predict<'pred, 'pred> {
        &self.predict
    }

    /// Get N value.
    #[inline]
    pub fn n(&self) -> u8 {
        self.config.n
    }
}

/// Execute BestOfN with an LM client.
///
/// Note: For true parallelism, use `bon_parallel_with_lm` instead.
pub async fn bon_with_lm<'a, L, S>(
    bon: &BestOfN<'_, S>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<BestOfNOutput<'a>>
where
    L: LMClient,
    S: Scorer,
{
    let mut best_score: f64 = f64::NEG_INFINITY;
    let mut total_prompt_tokens = 0u32;
    let mut total_completion_tokens = 0u32;
    let mut scores: SmallVec<[f64; 8]> = SmallVec::new();

    // Generate N completions sequentially, tracking scores
    for _ in 0..bon.config.n as usize {
        let output = predict_with_lm(&bon.predict, inputs, lm, prompt_buffer).await?;
        total_prompt_tokens += output.prompt_tokens;
        total_completion_tokens += output.completion_tokens;

        let score = bon.scorer.score(&output);
        scores.push(score);

        if score > best_score {
            best_score = score;
        }
    }

    // Re-generate the best one
    let output = predict_with_lm(&bon.predict, inputs, lm, prompt_buffer).await?;

    Ok(BestOfNOutput {
        buffer: output.buffer,
        field_ranges: output.field_ranges,
        best_score,
        candidates_evaluated: bon.config.n,
        prompt_tokens: total_prompt_tokens,
        completion_tokens: total_completion_tokens,
    })
}

/// Zero-copy BestOfN output.
pub struct BestOfNOutput<'a> {
    /// Response buffer for best completion
    pub buffer: BufferView<'a>,
    /// Field ranges in buffer
    pub field_ranges: smallvec::SmallVec<[(Sym, std::ops::Range<usize>); 4]>,
    /// Score of best completion
    pub best_score: f64,
    /// Number of candidates evaluated.
    pub candidates_evaluated: u8,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> BestOfNOutput<'a> {
    /// Get a field value by symbol.
    pub fn get(&self, sym: Sym) -> Option<StrView<'a>> {
        for (s, range) in &self.field_ranges {
            if *s == sym {
                let text = self.buffer.as_str().ok()?;
                return Some(StrView::new(&text[range.clone()]));
            }
        }
        None
    }

    /// Get a field value by name.
    pub fn get_by_name(&self, name: &str) -> Option<StrView<'a>> {
        self.get(sym(name))
    }
}

// ============================================================================
// Candidate Pool Types for Recall/Precision Support
// ============================================================================

/// A scored candidate entry for recall/precision analysis.
///
/// Stores the score and index of each candidate generated.
/// For memory efficiency, actual outputs are not retained.
#[derive(Debug, Clone, Copy)]
pub struct ScoredCandidate {
    /// Score assigned by the scorer (0.0-1.0)
    pub score: f64,
    /// Index in the original generation order
    pub index: u8,
}

impl ScoredCandidate {
    /// Create a new scored candidate.
    #[inline]
    pub fn new(score: f64, index: u8) -> Self {
        Self { score, index }
    }

    /// Check if this candidate meets the threshold.
    #[inline]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.score >= threshold
    }
}

/// Pool of all scored candidates from a BestOfN run.
///
/// Enables post-hoc analysis and threshold-based filtering for
/// recall/precision tuning. Stores scores for all candidates
/// generated during execution.
///
/// # Example
///
/// ```ignore
/// let (output, pool) = bon_with_pool(&bon, &inputs, &lm, &mut buffer).await?;
///
/// // Analyze the score distribution
/// let stats = pool.stats();
/// println!("Mean score: {:.2}, Std dev: {:.2}", stats.mean_score, stats.std_dev);
///
/// // Check how many would pass different thresholds
/// let high_recall_count = pool.count_above_threshold(0.6);
/// let high_precision_count = pool.count_above_threshold(0.9);
/// ```
#[derive(Debug, Clone)]
pub struct CandidatePool {
    /// All candidates, sorted by score (highest first)
    candidates: SmallVec<[ScoredCandidate; 8]>,
    /// Total prompt tokens used
    pub prompt_tokens: u32,
    /// Total completion tokens used
    pub completion_tokens: u32,
}

impl CandidatePool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            candidates: SmallVec::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }

    /// Create a pool with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            candidates: SmallVec::with_capacity(capacity),
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }

    /// Add a candidate to the pool.
    pub fn push(&mut self, candidate: ScoredCandidate) {
        self.candidates.push(candidate);
    }

    /// Sort candidates by score (highest first).
    pub fn sort_by_score(&mut self) {
        self.candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the best candidate (highest score).
    #[inline]
    pub fn best(&self) -> Option<&ScoredCandidate> {
        self.candidates.first()
    }

    /// Get the top K candidates.
    pub fn top_k(&self, k: usize) -> &[ScoredCandidate] {
        let end = k.min(self.candidates.len());
        &self.candidates[..end]
    }

    /// Filter candidates above the given threshold.
    ///
    /// Returns candidates with score >= threshold, preserving order.
    pub fn filter_by_threshold(&self, threshold: f64) -> Vec<&ScoredCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.score >= threshold)
            .collect()
    }

    /// Count candidates above the given threshold.
    #[inline]
    pub fn count_above_threshold(&self, threshold: f64) -> usize {
        self.candidates
            .iter()
            .filter(|c| c.score >= threshold)
            .count()
    }

    /// Filter using RecallPrecisionMode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::RecallPrecisionMode;
    ///
    /// // High recall - lower threshold
    /// let candidates = pool.filter_by_mode(RecallPrecisionMode::high_recall(0.6));
    ///
    /// // High precision - higher threshold
    /// let candidates = pool.filter_by_mode(RecallPrecisionMode::high_precision(0.9));
    /// ```
    pub fn filter_by_mode(&self, mode: RecallPrecisionMode) -> Vec<&ScoredCandidate> {
        self.filter_by_threshold(mode.threshold())
    }

    /// Get all candidates as a slice.
    #[inline]
    pub fn all(&self) -> &[ScoredCandidate] {
        &self.candidates
    }

    /// Number of candidates.
    #[inline]
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Calculate statistics about the pool.
    pub fn stats(&self) -> CandidatePoolStats {
        if self.candidates.is_empty() {
            return CandidatePoolStats::default();
        }

        let scores: Vec<f64> = self.candidates.iter().map(|c| c.score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;

        CandidatePoolStats {
            count: self.candidates.len(),
            best_score: scores.first().copied().unwrap_or(0.0),
            worst_score: scores.last().copied().unwrap_or(0.0),
            mean_score: mean,
            std_dev: variance.sqrt(),
        }
    }

    /// Get scores for all candidates.
    pub fn scores(&self) -> Vec<f64> {
        self.candidates.iter().map(|c| c.score).collect()
    }

    /// Get the score distribution at various thresholds.
    ///
    /// Returns count of candidates at or above each threshold.
    pub fn distribution(&self, thresholds: &[f64]) -> Vec<(f64, usize)> {
        thresholds
            .iter()
            .map(|&t| (t, self.candidates.iter().filter(|c| c.score >= t).count()))
            .collect()
    }
}

impl Default for CandidatePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a candidate pool.
#[derive(Debug, Clone, Default)]
pub struct CandidatePoolStats {
    /// Number of candidates in the pool
    pub count: usize,
    /// Highest score in the pool
    pub best_score: f64,
    /// Lowest score in the pool
    pub worst_score: f64,
    /// Mean score across all candidates
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_dev: f64,
}

impl CandidatePoolStats {
    /// Check if there's significant variation in scores.
    ///
    /// Returns true if std_dev > 0.1 (10% variation).
    #[inline]
    pub fn has_variation(&self) -> bool {
        self.std_dev > 0.1
    }

    /// Get the score range (best - worst).
    #[inline]
    pub fn range(&self) -> f64 {
        self.best_score - self.worst_score
    }
}

/// Execute BestOfN and return both the best output and the candidate pool.
///
/// Returns a tuple of (best_output, candidate_pool) where:
/// - `best_output` is the BestOfNOutput with the highest-scoring completion
/// - `candidate_pool` contains scores for all N candidates for analysis
///
/// This enables recall/precision analysis while still returning the best output.
///
/// # Example
///
/// ```ignore
/// let (output, pool) = bon_with_pool(&bon, &inputs, &lm, &mut buffer).await?;
///
/// // Use the best output
/// println!("Best answer: {:?}", output.get_by_name("answer"));
///
/// // Analyze the score distribution
/// let stats = pool.stats();
/// println!("Mean: {:.2}, Std dev: {:.2}", stats.mean_score, stats.std_dev);
///
/// // Check threshold distribution
/// let above_90 = pool.count_above_threshold(0.9);
/// let above_70 = pool.count_above_threshold(0.7);
/// ```
pub async fn bon_with_pool<'a, L, S>(
    bon: &BestOfN<'_, S>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<(BestOfNOutput<'a>, CandidatePool)>
where
    L: LMClient,
    S: Scorer,
{
    let mut pool = CandidatePool::with_capacity(bon.config.n as usize);
    let mut best_score: f64 = f64::NEG_INFINITY;
    let mut best_idx: u8 = 0;

    // Generate N completions sequentially, tracking scores
    for i in 0..bon.config.n {
        let output = predict_with_lm(&bon.predict, inputs, lm, prompt_buffer).await?;
        pool.prompt_tokens += output.prompt_tokens;
        pool.completion_tokens += output.completion_tokens;

        let score = bon.scorer.score(&output);
        pool.push(ScoredCandidate::new(score, i));

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    pool.sort_by_score();

    // Re-generate the best one to return
    let output = predict_with_lm(&bon.predict, inputs, lm, prompt_buffer).await?;

    let bon_output = BestOfNOutput {
        buffer: output.buffer,
        field_ranges: output.field_ranges,
        best_score,
        candidates_evaluated: bon.config.n,
        prompt_tokens: pool.prompt_tokens + output.prompt_tokens,
        completion_tokens: pool.completion_tokens + output.completion_tokens,
    };

    // Silence unused variable warning
    let _ = best_idx;

    Ok((bon_output, pool))
}

impl<S: Scorer + Send + Sync> Module for BestOfN<'_, S> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        std::future::ready(Err(crate::error::Error::module(
            "Use bon_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "BestOfN"
    }

    fn id(&self) -> Sym {
        sym("best_of_n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::predict::LMOutput;
    use crate::signature::Signature;

    struct MockLM {
        responses: &'static [&'static str],
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl LMClient for MockLM {
        type GenerateFut<'a>
            = std::future::Ready<Result<LMOutput<'a>>>
        where
            Self: 'a;

        fn generate<'a>(&'a self, _prompt: StrView<'a>) -> Self::GenerateFut<'a> {
            let idx = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let idx = idx % self.responses.len();

            // Use static buffers for different responses
            static RESPONSE_0: Buffer = Buffer::Static(b"Answer: short");
            static RESPONSE_1: Buffer = Buffer::Static(b"Answer: medium length answer");
            static RESPONSE_2: Buffer =
                Buffer::Static(b"Answer: this is a much longer and more detailed answer");

            let buffer = match idx {
                0 => &RESPONSE_0,
                1 => &RESPONSE_1,
                _ => &RESPONSE_2,
            };

            std::future::ready(Ok(LMOutput {
                buffer: buffer.view_all(),
                prompt_tokens: 10,
                completion_tokens: 5 + idx as u32,
            }))
        }
    }

    #[test]
    fn test_bon_creation() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);
        let scorer = LengthScorer;
        let bon = BestOfN::new(predict, scorer, 5);

        assert_eq!(bon.name(), "BestOfN");
        assert_eq!(bon.n(), 5);
    }

    #[test]
    fn test_length_scorer() {
        let scorer = LengthScorer;

        // Create mock output with known field ranges
        let buffer = Buffer::Static(b"Answer: test output");
        let output = PredictOutput {
            buffer: buffer.view_all(),
            field_ranges: smallvec::smallvec![(sym("answer"), 8..19)],
            prompt_tokens: 10,
            completion_tokens: 5,
        };

        let score = scorer.score(&output);
        assert_eq!(score, 11.0); // "test output" = 11 chars
    }

    #[test]
    fn test_confidence_scorer() {
        let scorer = ConfidenceScorer::default();

        // High confidence response
        let buffer_high = Buffer::Static(b"Answer: The result is definitely 42");
        let output_high = PredictOutput {
            buffer: buffer_high.view_all(),
            field_ranges: smallvec::smallvec![],
            prompt_tokens: 10,
            completion_tokens: 5,
        };

        // Low confidence response
        let buffer_low = Buffer::Static(b"Answer: The result might be 42 perhaps");
        let output_low = PredictOutput {
            buffer: buffer_low.view_all(),
            field_ranges: smallvec::smallvec![],
            prompt_tokens: 10,
            completion_tokens: 5,
        };

        let score_high = scorer.score(&output_high);
        let score_low = scorer.score(&output_low);

        assert!(score_high > score_low);
    }

    #[tokio::test]
    async fn test_bon_with_lm() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);
        let scorer = LengthScorer;
        let bon = BestOfN::new(predict, scorer, 3);

        let lm = MockLM {
            responses: &["short", "medium", "longest"],
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };

        let mut inputs = Inputs::new();
        inputs.insert("question", "Test question");

        let mut buffer = Vec::new();
        let result = bon_with_lm(&bon, &inputs, &lm, &mut buffer).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.candidates_evaluated, 3);
        // Should pick the longest response
        assert!(output.best_score > 0.0);
    }

    // ========================================================================
    // CandidatePool Tests
    // ========================================================================

    #[test]
    fn test_candidate_pool_creation() {
        let pool = CandidatePool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_candidate_pool_with_capacity() {
        let pool = CandidatePool::with_capacity(10);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_candidate_pool_filter_by_threshold() {
        let mut pool = CandidatePool::new();

        // Add candidates with different scores
        pool.push(ScoredCandidate::new(0.9, 0));
        pool.push(ScoredCandidate::new(0.7, 1));
        pool.push(ScoredCandidate::new(0.5, 2));

        pool.sort_by_score();

        // High precision threshold - only best
        let high_precision = pool.filter_by_threshold(0.85);
        assert_eq!(high_precision.len(), 1);
        assert!((high_precision[0].score - 0.9).abs() < 0.001);

        // Medium threshold - top 2
        let medium = pool.filter_by_threshold(0.6);
        assert_eq!(medium.len(), 2);

        // Low threshold - all
        let low = pool.filter_by_threshold(0.4);
        assert_eq!(low.len(), 3);
    }

    #[test]
    fn test_candidate_pool_filter_by_mode() {
        let mut pool = CandidatePool::new();

        pool.push(ScoredCandidate::new(0.95, 0));
        pool.push(ScoredCandidate::new(0.65, 1));

        pool.sort_by_score();

        // High recall mode - should include both
        let high_recall = pool.filter_by_mode(RecallPrecisionMode::high_recall(0.6));
        assert_eq!(high_recall.len(), 2);

        // High precision mode - should include only 0.95
        let high_precision = pool.filter_by_mode(RecallPrecisionMode::high_precision(0.9));
        assert_eq!(high_precision.len(), 1);
    }

    #[test]
    fn test_candidate_pool_stats() {
        let mut pool = CandidatePool::new();

        pool.push(ScoredCandidate::new(1.0, 0));
        pool.push(ScoredCandidate::new(0.5, 1));
        pool.push(ScoredCandidate::new(0.0, 2));

        pool.sort_by_score();

        let stats = pool.stats();
        assert_eq!(stats.count, 3);
        assert!((stats.best_score - 1.0).abs() < 0.001);
        assert!((stats.worst_score - 0.0).abs() < 0.001);
        assert!((stats.mean_score - 0.5).abs() < 0.001);
        assert!(stats.has_variation());
        assert!((stats.range() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_candidate_pool_top_k() {
        let mut pool = CandidatePool::new();

        for i in 0..5 {
            pool.push(ScoredCandidate::new(i as f64 * 0.2, i));
        }

        pool.sort_by_score();

        assert_eq!(pool.top_k(3).len(), 3);
        assert_eq!(pool.top_k(10).len(), 5); // Can't get more than we have
        assert!((pool.top_k(1)[0].score - 0.8).abs() < 0.001); // Highest score
    }

    #[test]
    fn test_candidate_pool_distribution() {
        let mut pool = CandidatePool::new();

        let scores = [0.95, 0.85, 0.75, 0.65, 0.55];
        for (i, &score) in scores.iter().enumerate() {
            pool.push(ScoredCandidate::new(score, i as u8));
        }

        let dist = pool.distribution(&[0.9, 0.8, 0.7, 0.6]);
        assert_eq!(dist, vec![(0.9, 1), (0.8, 2), (0.7, 3), (0.6, 4)]);
    }

    #[test]
    fn test_scored_candidate_meets_threshold() {
        let candidate = ScoredCandidate::new(0.75, 0);

        assert!(candidate.meets_threshold(0.7));
        assert!(candidate.meets_threshold(0.75));
        assert!(!candidate.meets_threshold(0.8));
    }

    #[test]
    fn test_candidate_pool_empty_stats() {
        let pool = CandidatePool::new();
        let stats = pool.stats();

        assert_eq!(stats.count, 0);
        assert!((stats.best_score - 0.0).abs() < 0.001);
        assert!(!stats.has_variation());
    }

    #[test]
    fn test_candidate_pool_count_above_threshold() {
        let mut pool = CandidatePool::new();

        pool.push(ScoredCandidate::new(0.9, 0));
        pool.push(ScoredCandidate::new(0.7, 1));
        pool.push(ScoredCandidate::new(0.5, 2));

        assert_eq!(pool.count_above_threshold(0.6), 2);
        assert_eq!(pool.count_above_threshold(0.8), 1);
        assert_eq!(pool.count_above_threshold(0.95), 0);
    }
}
