// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multi-Chain Comparison module
//!
//! Generates multiple reasoning chains in parallel and compares them
//! to select the most consistent or highest-quality answer.
//!
//! ## Zero-Copy Design
//!
//! - All chains use shared buffer pool
//! - Comparison operates on `StrView<'a>` references
//! - Supports majority voting and consistency checking

use crate::buffer::BufferView;
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::module::Module;
use crate::modules::chain_of_thought::{cot_with_lm, ChainOfThought, CoTOutput};
use crate::predict::LMClient;
use crate::prediction::Prediction;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ops::Range;

/// Strategy for selecting the final answer from multiple chains.
#[derive(Clone, Copy, Default)]
pub enum SelectionStrategy {
    /// Select by majority voting on the answer
    #[default]
    MajorityVote,
    /// Select the answer with longest reasoning
    LongestReasoning,
    /// Select the first successful answer
    FirstSuccess,
    /// Use custom scoring (requires scorer)
    Custom,
}

/// MultiChain configuration.
#[derive(Clone, Copy)]
pub struct MultiChainConfig {
    /// Number of chains to generate
    pub num_chains: u8,
    /// Selection strategy
    pub strategy: SelectionStrategy,
    /// Whether to include all reasoning chains in output
    pub include_all_chains: bool,
}

impl Default for MultiChainConfig {
    fn default() -> Self {
        Self {
            num_chains: 3,
            strategy: SelectionStrategy::MajorityVote,
            include_all_chains: false,
        }
    }
}

/// Multi-Chain Comparison module.
///
/// Generates multiple reasoning chains and selects the best answer.
///
/// ## Example
///
/// ```ignore
/// let cot = ChainOfThought::new(&signature);
/// let multi = MultiChain::new(cot, 5);
/// let result = multi_chain_with_lm(&multi, &inputs, &lm, &mut buffers).await?;
/// ```
#[derive(Clone, Copy)]
pub struct MultiChain<'sig, 'demo> {
    /// The underlying CoT module
    cot: ChainOfThought<'sig, 'demo>,
    /// Configuration
    config: MultiChainConfig,
}

impl<'sig, 'demo> MultiChain<'sig, 'demo> {
    /// Create a new MultiChain module.
    pub fn new(cot: ChainOfThought<'sig, 'demo>, num_chains: u8) -> Self {
        Self {
            cot,
            config: MultiChainConfig {
                num_chains,
                ..Default::default()
            },
        }
    }

    /// Configure selection strategy.
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Configure whether to include all chains.
    pub fn with_all_chains(mut self, include: bool) -> Self {
        self.config.include_all_chains = include;
        self
    }

    /// Get the CoT module.
    #[inline]
    pub fn cot(&self) -> &ChainOfThought<'sig, 'demo> {
        &self.cot
    }

    /// Get number of chains.
    #[inline]
    pub fn num_chains(&self) -> u8 {
        self.config.num_chains
    }
}

/// Single chain result.
pub struct ChainResult<'a> {
    /// The chain output
    pub output: CoTOutput<'a>,
    /// Normalized answer (for voting)
    pub normalized_answer: String,
}

// ===== Recall/Precision: ScoredChain and ChainPool =====

/// A chain result with consensus score for recall/precision filtering.
///
/// The consensus score indicates how much this chain agrees with the
/// majority answer. Higher scores mean stronger agreement.
#[derive(Debug, Clone)]
pub struct ScoredChain {
    /// Index of this chain (0-indexed)
    pub index: u8,
    /// The normalized answer this chain produced
    pub answer: String,
    /// Consensus score: fraction of chains that agree with this answer (0.0-1.0)
    pub consensus_score: f64,
    /// Whether this chain voted for the selected (majority) answer
    pub agrees_with_majority: bool,
}

/// Pool of all chain results with voting/consensus information.
///
/// Enables recall/precision filtering by consensus threshold.
/// High recall: include all chains (low threshold)
/// High precision: only chains with strong consensus (high threshold)
#[derive(Debug, Clone)]
pub struct ChainPool {
    /// All scored chains
    chains: SmallVec<[ScoredChain; 8]>,
    /// The selected majority answer
    pub selected_answer: String,
    /// Total number of chains generated
    pub chains_generated: u8,
    /// Number of chains that succeeded
    pub chains_succeeded: u8,
    /// Total prompt tokens used across all chains
    pub prompt_tokens: u32,
    /// Total completion tokens used across all chains
    pub completion_tokens: u32,
}

/// Statistics about the chain pool.
#[derive(Debug, Clone, Copy)]
pub struct ChainPoolStats {
    /// Mean consensus score across all chains
    pub mean_consensus: f64,
    /// Standard deviation of consensus scores
    pub std_dev: f64,
    /// Number of distinct answers
    pub distinct_answers: usize,
    /// Fraction of chains agreeing with majority
    pub agreement_ratio: f64,
}

impl ChainPool {
    /// Create a new chain pool from answers.
    pub fn from_answers(
        answers: &[String],
        selected_answer: String,
        chains_generated: u8,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> Self {
        // Count votes for each answer
        let mut vote_counts: HashMap<&str, usize> = HashMap::new();
        for answer in answers {
            *vote_counts.entry(answer.as_str()).or_insert(0) += 1;
        }

        let total = answers.len() as f64;

        // Build scored chains
        let chains: SmallVec<[ScoredChain; 8]> = answers
            .iter()
            .enumerate()
            .map(|(idx, answer)| {
                let count = *vote_counts.get(answer.as_str()).unwrap_or(&0) as f64;
                let consensus_score = count / total;
                let agrees_with_majority = answer == &selected_answer;

                ScoredChain {
                    index: idx as u8,
                    answer: answer.clone(),
                    consensus_score,
                    agrees_with_majority,
                }
            })
            .collect();

        Self {
            chains,
            selected_answer,
            chains_generated,
            chains_succeeded: answers.len() as u8,
            prompt_tokens,
            completion_tokens,
        }
    }

    /// Get all chains.
    #[inline]
    pub fn all_chains(&self) -> &[ScoredChain] {
        &self.chains
    }

    /// Get chains that agree with the majority answer.
    pub fn agreeing_chains(&self) -> impl Iterator<Item = &ScoredChain> {
        self.chains.iter().filter(|c| c.agrees_with_majority)
    }

    /// Get chains that disagree with the majority answer.
    pub fn dissenting_chains(&self) -> impl Iterator<Item = &ScoredChain> {
        self.chains.iter().filter(|c| !c.agrees_with_majority)
    }

    /// Filter chains by consensus threshold.
    ///
    /// Returns chains with consensus score >= threshold.
    /// Use high threshold (e.g., 0.5+) for high precision.
    /// Use low threshold (e.g., 0.2+) for high recall.
    pub fn filter_by_consensus(&self, threshold: f64) -> Vec<&ScoredChain> {
        self.chains
            .iter()
            .filter(|c| c.consensus_score >= threshold)
            .collect()
    }

    /// Filter chains using a RecallPrecisionMode.
    pub fn filter_by_mode(
        &self,
        mode: crate::recall_precision::RecallPrecisionMode,
    ) -> Vec<&ScoredChain> {
        self.filter_by_consensus(mode.threshold())
    }

    /// Get the agreement ratio (fraction of chains agreeing with majority).
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

    /// Count chains above a consensus threshold.
    pub fn count_above_consensus(&self, threshold: f64) -> usize {
        self.chains
            .iter()
            .filter(|c| c.consensus_score >= threshold)
            .count()
    }

    /// Get distinct answers in the pool.
    pub fn distinct_answers(&self) -> Vec<&str> {
        let mut seen: HashMap<&str, ()> = HashMap::new();
        let mut result = Vec::new();
        for chain in &self.chains {
            if !seen.contains_key(chain.answer.as_str()) {
                seen.insert(&chain.answer, ());
                result.push(chain.answer.as_str());
            }
        }
        result
    }

    /// Calculate pool statistics.
    pub fn stats(&self) -> ChainPoolStats {
        if self.chains.is_empty() {
            return ChainPoolStats {
                mean_consensus: 0.0,
                std_dev: 0.0,
                distinct_answers: 0,
                agreement_ratio: 0.0,
            };
        }

        let n = self.chains.len() as f64;
        let sum: f64 = self.chains.iter().map(|c| c.consensus_score).sum();
        let mean = sum / n;

        let variance: f64 = self
            .chains
            .iter()
            .map(|c| (c.consensus_score - mean).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        ChainPoolStats {
            mean_consensus: mean,
            std_dev,
            distinct_answers: self.distinct_answers().len(),
            agreement_ratio: self.agreement_ratio(),
        }
    }

    /// Get the best chain (highest consensus score).
    pub fn best_chain(&self) -> Option<&ScoredChain> {
        self.chains
            .iter()
            .max_by(|a, b| a.consensus_score.partial_cmp(&b.consensus_score).unwrap())
    }

    /// Check if all chains agree (perfect consensus).
    pub fn has_perfect_consensus(&self) -> bool {
        self.agreement_ratio() >= 1.0 - f64::EPSILON
    }

    /// Check if there's no clear winner (high disagreement).
    ///
    /// Returns true if the agreement ratio is below the given threshold.
    pub fn has_high_disagreement(&self, threshold: f64) -> bool {
        self.agreement_ratio() < threshold
    }
}

/// Execute MultiChain and return a ChainPool with all chain results.
///
/// This version returns the full pool of chains for recall/precision filtering,
/// unlike `multi_chain_with_lm` which only returns the selected answer.
///
/// # Example
///
/// ```ignore
/// let pool = multi_chain_with_pool(&multi, &inputs, &lm, &mut buffer).await?;
///
/// // High precision: only chains with strong consensus
/// let high_precision = pool.filter_by_consensus(0.5);
///
/// // Check for disagreement
/// if pool.has_high_disagreement(0.4) {
///     println!("Warning: chains have high disagreement");
/// }
/// ```
pub async fn multi_chain_with_pool<'a, L>(
    multi: &MultiChain<'_, '_>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<ChainPool>
where
    L: LMClient,
{
    let mut answers: SmallVec<[String; 8]> = SmallVec::new();
    let mut total_prompt_tokens = 0u32;
    let mut total_completion_tokens = 0u32;

    // Generate all chains, collecting normalized answers
    for _ in 0..multi.config.num_chains {
        match cot_with_lm(&multi.cot, inputs, lm, prompt_buffer).await {
            Ok(output) => {
                total_prompt_tokens += output.prompt_tokens;
                total_completion_tokens += output.completion_tokens;

                // Extract and normalize answer
                let answer = output.get_by_name("answer");
                let normalized = answer
                    .map(|a| normalize_answer(a.as_str()))
                    .unwrap_or_default();

                answers.push(normalized);
            }
            Err(_) => {
                // Skip failed chains
                continue;
            }
        }
    }

    if answers.is_empty() {
        return Err(crate::error::Error::module("All reasoning chains failed"));
    }

    // Find majority answer
    let selected_answer = select_majority_answer(&answers);

    Ok(ChainPool::from_answers(
        &answers,
        selected_answer,
        multi.config.num_chains,
        total_prompt_tokens,
        total_completion_tokens,
    ))
}

/// Execute MultiChain with an LM client.
///
/// This simplified version runs chains sequentially with a single buffer.
/// The majority voting happens on normalized answers tracked separately.
pub async fn multi_chain_with_lm<'a, L>(
    multi: &MultiChain<'_, '_>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<MultiChainOutput<'a>>
where
    L: LMClient,
{
    let mut answers: SmallVec<[String; 8]> = SmallVec::new();
    let mut total_prompt_tokens = 0u32;
    let mut total_completion_tokens = 0u32;
    let mut chains_succeeded = 0u8;

    // Generate all chains, collecting normalized answers
    for _ in 0..multi.config.num_chains {
        match cot_with_lm(&multi.cot, inputs, lm, prompt_buffer).await {
            Ok(output) => {
                total_prompt_tokens += output.prompt_tokens;
                total_completion_tokens += output.completion_tokens;
                chains_succeeded += 1;

                // Extract and normalize answer
                let answer = output.get_by_name("answer");
                let normalized = answer
                    .map(|a| normalize_answer(a.as_str()))
                    .unwrap_or_default();

                answers.push(normalized);
            }
            Err(_) => {
                // Skip failed chains
                continue;
            }
        }
    }

    if answers.is_empty() {
        return Err(crate::error::Error::module("All reasoning chains failed"));
    }

    // Find majority answer
    let selected_answer = select_majority_answer(&answers);

    // Re-generate to get final output
    let final_output = cot_with_lm(&multi.cot, inputs, lm, prompt_buffer).await?;

    Ok(MultiChainOutput {
        buffer: final_output.buffer,
        field_ranges: final_output.field_ranges,
        reasoning_range: final_output.reasoning_range,
        selected_answer,
        chains_generated: multi.config.num_chains,
        chains_succeeded,
        prompt_tokens: total_prompt_tokens,
        completion_tokens: total_completion_tokens,
    })
}

/// Select the answer that appears most frequently.
fn select_majority_answer(answers: &[String]) -> String {
    use std::collections::HashMap;
    let mut votes: HashMap<&str, usize> = HashMap::new();

    for answer in answers {
        *votes.entry(answer.as_str()).or_insert(0) += 1;
    }

    votes
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(answer, _)| answer.to_string())
        .unwrap_or_default()
}

/// Normalize answer for voting comparison.
fn normalize_answer(answer: &str) -> String {
    answer
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Select answer by majority vote.
#[allow(dead_code)] // Reserved for future SelectionStrategy implementation
fn select_by_vote(chains: &[ChainResult<'_>]) -> usize {
    let mut votes: HashMap<&str, (usize, usize)> = HashMap::new();

    for (idx, chain) in chains.iter().enumerate() {
        let entry = votes.entry(&chain.normalized_answer).or_insert((0, idx));
        entry.0 += 1;
    }

    // Find answer with most votes
    votes
        .into_iter()
        .max_by_key(|(_, (count, _))| *count)
        .map(|(_, (_, idx))| idx)
        .unwrap_or(0)
}

/// Select answer with longest reasoning.
#[allow(dead_code)] // Reserved for future SelectionStrategy implementation
fn select_by_reasoning_length(chains: &[ChainResult<'_>]) -> usize {
    chains
        .iter()
        .enumerate()
        .max_by_key(|(_, chain)| {
            chain
                .output
                .reasoning_range
                .as_ref()
                .map(|r| r.len())
                .unwrap_or(0)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Zero-copy MultiChain output.
pub struct MultiChainOutput<'a> {
    /// Response buffer for selected chain
    pub buffer: BufferView<'a>,
    /// Field ranges in buffer
    pub field_ranges: SmallVec<[(Sym, Range<usize>); 4]>,
    /// Reasoning range
    pub reasoning_range: Option<Range<usize>>,
    /// The selected answer (normalized)
    pub selected_answer: String,
    /// Number of chains generated.
    pub chains_generated: u8,
    /// Number of chains that succeeded.
    pub chains_succeeded: u8,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> MultiChainOutput<'a> {
    /// Get the reasoning text.
    pub fn reasoning(&self) -> Option<StrView<'a>> {
        let range = self.reasoning_range.as_ref()?;
        let text = self.buffer.as_str().ok()?;
        Some(StrView::new(&text[range.clone()]))
    }

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

impl Module for MultiChain<'_, '_> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        std::future::ready(Err(crate::error::Error::module(
            "Use multi_chain_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "MultiChain"
    }

    fn id(&self) -> Sym {
        sym("multi_chain")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::predict::LMOutput;
    use crate::signature::Signature;

    struct MockLM {
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

            // Different responses - 2 agree, 1 differs
            static RESPONSE_A: Buffer = Buffer::Static(b"Reasoning: Step 1.\n\nAnswer: 42");
            static RESPONSE_B: Buffer =
                Buffer::Static(b"Reasoning: Different reasoning.\n\nAnswer: 42");
            static RESPONSE_C: Buffer = Buffer::Static(b"Reasoning: Wrong path.\n\nAnswer: 41");

            let buffer = match idx % 3 {
                0 => &RESPONSE_A,
                1 => &RESPONSE_B,
                _ => &RESPONSE_C,
            };

            std::future::ready(Ok(LMOutput {
                buffer: buffer.view_all(),
                prompt_tokens: 10,
                completion_tokens: 5,
            }))
        }
    }

    #[test]
    fn test_multi_chain_creation() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        let multi = MultiChain::new(cot, 5);

        assert_eq!(multi.name(), "MultiChain");
        assert_eq!(multi.num_chains(), 5);
    }

    #[test]
    fn test_normalize_answer() {
        assert_eq!(normalize_answer("  The Answer  "), "the answer");
        assert_eq!(normalize_answer("42!"), "42");
        assert_eq!(normalize_answer("Yes."), "yes");
    }

    #[test]
    fn test_multi_chain_copy() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        let multi = MultiChain::new(cot, 3);
        let copy = multi; // Copy, not move
        assert_eq!(multi.num_chains(), copy.num_chains());
    }

    #[tokio::test]
    async fn test_multi_chain_with_lm() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        let multi = MultiChain::new(cot, 3);

        let lm = MockLM {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };

        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 6 * 7?");

        let mut buffer = Vec::new();
        let result = multi_chain_with_lm(&multi, &inputs, &lm, &mut buffer).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.chains_generated, 3);
        // Chains ran successfully
        assert!(output.chains_succeeded > 0);
    }

    // ===== ChainPool Tests =====

    #[test]
    fn test_chain_pool_from_answers() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        assert_eq!(pool.chains_generated, 3);
        assert_eq!(pool.chains_succeeded, 3);
        assert_eq!(pool.selected_answer, "42");
        assert_eq!(pool.all_chains().len(), 3);
    }

    #[test]
    fn test_chain_pool_agreement_ratio() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        // 2 out of 3 agree
        assert!((pool.agreement_ratio() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_chain_pool_agreeing_chains() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        let agreeing: Vec<_> = pool.agreeing_chains().collect();
        assert_eq!(agreeing.len(), 2);
        for chain in agreeing {
            assert!(chain.agrees_with_majority);
            assert_eq!(chain.answer, "42");
        }
    }

    #[test]
    fn test_chain_pool_dissenting_chains() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        let dissenting: Vec<_> = pool.dissenting_chains().collect();
        assert_eq!(dissenting.len(), 1);
        assert!(!dissenting[0].agrees_with_majority);
        assert_eq!(dissenting[0].answer, "41");
    }

    #[test]
    fn test_chain_pool_filter_by_consensus() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        // "42" appears 2/3 times = 0.667 consensus
        // "41" appears 1/3 times = 0.333 consensus

        // High threshold: only "42" chains
        let high = pool.filter_by_consensus(0.5);
        assert_eq!(high.len(), 2);

        // Low threshold: all chains
        let low = pool.filter_by_consensus(0.3);
        assert_eq!(low.len(), 3);
    }

    #[test]
    fn test_chain_pool_stats() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);
        let stats = pool.stats();

        assert_eq!(stats.distinct_answers, 2);
        assert!((stats.agreement_ratio - 2.0 / 3.0).abs() < 0.001);
        assert!(stats.mean_consensus > 0.0);
    }

    #[test]
    fn test_chain_pool_perfect_consensus() {
        let answers = vec!["42".to_string(), "42".to_string(), "42".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        assert!(pool.has_perfect_consensus());
        assert!(!pool.has_high_disagreement(0.5));
    }

    #[test]
    fn test_chain_pool_high_disagreement() {
        let answers = vec!["42".to_string(), "41".to_string(), "40".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        // Only 1/3 agreement
        assert!(!pool.has_perfect_consensus());
        assert!(pool.has_high_disagreement(0.5));
    }

    #[test]
    fn test_chain_pool_distinct_answers() {
        let answers = vec![
            "42".to_string(),
            "42".to_string(),
            "41".to_string(),
            "40".to_string(),
        ];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 4, 100, 50);

        let distinct = pool.distinct_answers();
        assert_eq!(distinct.len(), 3);
        assert!(distinct.contains(&"42"));
        assert!(distinct.contains(&"41"));
        assert!(distinct.contains(&"40"));
    }

    #[test]
    fn test_chain_pool_best_chain() {
        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        let best = pool.best_chain().unwrap();
        assert_eq!(best.answer, "42");
        assert!((best.consensus_score - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_chain_pool_filter_by_mode() {
        use crate::recall_precision::RecallPrecisionMode;

        let answers = vec!["42".to_string(), "42".to_string(), "41".to_string()];
        let pool = ChainPool::from_answers(&answers, "42".to_string(), 3, 100, 50);

        // High recall mode (threshold 0.6) - "42" chains have 0.667 consensus
        let high_recall = pool.filter_by_mode(RecallPrecisionMode::high_recall(0.6));
        assert_eq!(high_recall.len(), 2);

        // Lower threshold to get all
        let very_high_recall = pool.filter_by_mode(RecallPrecisionMode::high_recall(0.3));
        assert_eq!(very_high_recall.len(), 3);
    }

    #[tokio::test]
    async fn test_multi_chain_with_pool() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        let multi = MultiChain::new(cot, 3);

        let lm = MockLM {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };

        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 6 * 7?");

        let mut buffer = Vec::new();
        let result = multi_chain_with_pool(&multi, &inputs, &lm, &mut buffer).await;

        assert!(result.is_ok());
        let pool = result.unwrap();
        assert_eq!(pool.chains_generated, 3);
        assert!(pool.chains_succeeded > 0);
        // Pool should have chains
        assert!(!pool.all_chains().is_empty());
    }
}
