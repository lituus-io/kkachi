// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Recursive state management
//!
//! Zero-copy state tracking across recursive iterations.

use crate::recall_precision::RecallPrecisionMode;
use crate::str_view::StrView;
use smallvec::SmallVec;

/// Record of a single iteration.
#[derive(Clone, Copy, Debug)]
pub struct IterationRecord<'a> {
    /// Output from this iteration
    pub output: StrView<'a>,
    /// Feedback provided (if any)
    pub feedback: Option<StrView<'a>>,
    /// Quality score (0.0 - 1.0)
    pub score: f64,
    /// Duration in milliseconds
    pub duration_ms: u32,
}

/// State accumulated across recursive iterations.
///
/// Uses `SmallVec` for inline storage of small histories,
/// avoiding heap allocation for typical use cases.
#[derive(Debug)]
pub struct RecursiveState<'a> {
    /// Current iteration number (0-indexed)
    pub iteration: u32,
    /// History of outputs (zero-copy views into shared buffer)
    pub history: SmallVec<[IterationRecord<'a>; 8]>,
    /// Current best output (iteration index, score)
    pub best: Option<(u32, f64)>,
    /// Quality scores per iteration
    pub scores: SmallVec<[f64; 8]>,
    /// Whether human intervention was requested
    pub awaiting_human: bool,
}

impl<'a> RecursiveState<'a> {
    /// Create new empty state.
    #[inline]
    pub fn new() -> Self {
        Self {
            iteration: 0,
            history: SmallVec::new(),
            best: None,
            scores: SmallVec::new(),
            awaiting_human: false,
        }
    }

    /// Create state with existing scores (for evaluation context).
    #[inline]
    pub fn with_scores(scores: &[f64]) -> Self {
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &s)| (i as u32, s));

        Self {
            iteration: scores.len() as u32,
            history: SmallVec::new(),
            best,
            scores: scores.iter().copied().collect(),
            awaiting_human: false,
        }
    }

    /// Record a new iteration.
    #[inline]
    pub fn record(
        &mut self,
        output: StrView<'a>,
        feedback: Option<StrView<'a>>,
        score: f64,
        duration_ms: u32,
    ) {
        let record = IterationRecord {
            output,
            feedback,
            score,
            duration_ms,
        };
        self.history.push(record);
        self.scores.push(score);

        // Update best if this is better
        match self.best {
            None => self.best = Some((self.iteration, score)),
            Some((_, best_score)) if score > best_score => {
                self.best = Some((self.iteration, score));
            }
            _ => {}
        }
    }

    /// Record just a score (simplified version).
    #[inline]
    pub fn record_score(&mut self, score: f64) {
        self.scores.push(score);
        match self.best {
            None => self.best = Some((self.iteration, score)),
            Some((_, best_score)) if score > best_score => {
                self.best = Some((self.iteration, score));
            }
            _ => {}
        }
    }

    /// Get the best score achieved.
    #[inline]
    pub fn best_score(&self) -> f64 {
        self.best.map(|(_, s)| s).unwrap_or(0.0)
    }

    /// Get the best output.
    #[inline]
    pub fn best_output(&self) -> Option<StrView<'a>> {
        self.best
            .and_then(|(idx, _)| self.history.get(idx as usize).map(|r| r.output))
    }

    /// Get the most recent output.
    #[inline]
    pub fn latest_output(&self) -> Option<StrView<'a>> {
        self.history.last().map(|r| r.output)
    }

    /// Get the most recent score.
    #[inline]
    pub fn latest_score(&self) -> Option<f64> {
        self.scores.last().copied()
    }

    /// Check if scores are improving.
    #[inline]
    pub fn is_improving(&self, window: usize) -> bool {
        if self.scores.len() < 2 {
            return true;
        }
        let len = self.scores.len();
        let start = len.saturating_sub(window);
        let recent = &self.scores[start..];

        if recent.len() < 2 {
            return true;
        }

        // Check if the trend is positive
        let first_half: f64 =
            recent[..recent.len() / 2].iter().sum::<f64>() / (recent.len() / 2) as f64;
        let second_half: f64 = recent[recent.len() / 2..].iter().sum::<f64>()
            / (recent.len() - recent.len() / 2) as f64;

        second_half >= first_half
    }

    /// Get improvement from previous iteration.
    #[inline]
    pub fn improvement(&self) -> f64 {
        if self.scores.len() < 2 {
            return 0.0;
        }
        let len = self.scores.len();
        self.scores[len - 1] - self.scores[len - 2]
    }

    /// Calculate moving average of recent scores.
    #[inline]
    pub fn moving_average(&self, window: usize) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let start = self.scores.len().saturating_sub(window);
        let slice = &self.scores[start..];
        slice.iter().sum::<f64>() / slice.len() as f64
    }
}

impl Default for RecursiveState<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for recursive refinement.
#[derive(Clone, Copy, Debug)]
pub struct RecursiveConfig {
    /// Maximum iterations (safety limit)
    pub max_iterations: u32,
    /// Score threshold to consider "good enough"
    pub score_threshold: f64,
    /// Minimum score to store in context
    pub min_store_score: f64,
    /// Include full history in prompt
    pub include_history: bool,
    /// Number of history entries to include
    pub history_window: u8,
    /// Temperature decay per iteration (1.0 = no decay)
    pub temperature_decay: f32,
    /// Buffer size for accumulated state
    pub buffer_capacity: usize,
    /// Use chain of thought reasoning
    pub use_cot: bool,
    /// Number of variations to generate (BestOfN)
    pub best_of_n: Option<u8>,
    /// Plateau detection threshold (minimum improvement to continue)
    pub plateau_threshold: Option<f64>,
    /// Window size for plateau detection
    pub plateau_window: Option<usize>,
    /// Recall/precision mode for threshold tuning.
    ///
    /// Controls the tradeoff between:
    /// - High recall: more permissive, catches more results
    /// - High precision: more strict, only confident results
    pub recall_precision_mode: RecallPrecisionMode,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            score_threshold: 0.9,
            min_store_score: 0.7,
            include_history: true,
            history_window: 3,
            temperature_decay: 0.95,
            buffer_capacity: 32 * 1024,
            use_cot: false,
            best_of_n: None,
            plateau_threshold: None,
            plateau_window: None,
            recall_precision_mode: RecallPrecisionMode::default(),
        }
    }
}

impl RecursiveConfig {
    /// Set the recall/precision mode.
    ///
    /// This also updates the score_threshold to match the mode.
    #[inline]
    pub fn with_recall_precision_mode(mut self, mode: RecallPrecisionMode) -> Self {
        self.recall_precision_mode = mode;
        self.score_threshold = mode.threshold();
        self
    }

    /// Use high-recall mode (permissive, favors catching all results).
    ///
    /// Sets threshold to 0.6.
    #[inline]
    pub fn high_recall(self) -> Self {
        self.with_recall_precision_mode(RecallPrecisionMode::high_recall(
            RecallPrecisionMode::HIGH_RECALL_DEFAULT,
        ))
    }

    /// Use high-precision mode (strict, favors accuracy).
    ///
    /// Sets threshold to 0.9.
    #[inline]
    pub fn high_precision(self) -> Self {
        self.with_recall_precision_mode(RecallPrecisionMode::high_precision(
            RecallPrecisionMode::HIGH_PRECISION_DEFAULT,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = RecursiveState::new();
        assert_eq!(state.iteration, 0);
        assert!(state.history.is_empty());
        assert!(state.best.is_none());
    }

    #[test]
    fn test_record_score() {
        let mut state = RecursiveState::new();
        state.record_score(0.5);
        assert_eq!(state.best_score(), 0.5);

        state.record_score(0.8);
        assert_eq!(state.best_score(), 0.8);

        state.record_score(0.6);
        assert_eq!(state.best_score(), 0.8); // Best unchanged
    }

    #[test]
    fn test_moving_average() {
        let mut state = RecursiveState::new();
        state.record_score(0.5);
        state.record_score(0.6);
        state.record_score(0.7);
        state.record_score(0.8);

        let avg = state.moving_average(2);
        assert!((avg - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_improvement() {
        let mut state = RecursiveState::new();
        state.record_score(0.5);
        assert_eq!(state.improvement(), 0.0);

        state.record_score(0.7);
        assert!((state.improvement() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_config_defaults() {
        let config = RecursiveConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.score_threshold, 0.9);
        assert!(!config.use_cot);
        assert_eq!(config.recall_precision_mode, RecallPrecisionMode::Balanced);
    }

    #[test]
    fn test_config_high_recall() {
        let config = RecursiveConfig::default().high_recall();
        assert!((config.score_threshold - 0.6).abs() < 0.001);
        assert!(config.recall_precision_mode.favors_recall());
    }

    #[test]
    fn test_config_high_precision() {
        let config = RecursiveConfig::default().high_precision();
        assert!((config.score_threshold - 0.9).abs() < 0.001);
        assert!(config.recall_precision_mode.favors_precision());
    }

    #[test]
    fn test_config_with_recall_precision_mode() {
        let config = RecursiveConfig::default()
            .with_recall_precision_mode(RecallPrecisionMode::custom(0.75));
        assert!((config.score_threshold - 0.75).abs() < 0.001);
    }
}
