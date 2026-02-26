// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Metric trait and built-in implementations for evaluating LLM outputs.
//!
//! This module provides the [`Metric`] trait, which defines a standard interface
//! for scoring prediction quality against expected output. Built-in metrics cover
//! common evaluation patterns:
//!
//! - [`ExactMatch`]: Exact string equality (trimmed)
//! - [`Contains`]: Substring containment check
//! - [`F1Token`]: Word-level precision/recall/F1 score
//! - [`FnMetric`]: Closure wrapper for ad-hoc metrics
//!
//! All metrics return scores in the range `[0.0, 1.0]`.
//!
//! # Examples
//!
//! ```
//! use kkachi::metric::{Metric, ExactMatch, Contains, F1Token, FnMetric};
//!
//! // Exact match
//! let m = ExactMatch;
//! assert_eq!(m.evaluate("hello", "hello"), 1.0);
//! assert_eq!(m.evaluate("hello", "world"), 0.0);
//!
//! // Contains
//! let m = Contains;
//! assert_eq!(m.evaluate("the answer is 42", "42"), 1.0);
//! assert_eq!(m.evaluate("the answer is 41", "42"), 0.0);
//!
//! // F1 token overlap
//! let m = F1Token;
//! let score = m.evaluate("the quick brown fox", "the slow brown fox");
//! assert!(score > 0.5);
//!
//! // Custom closure metric
//! let m = FnMetric::new("length_ratio", |pred, expected| {
//!     let ratio = pred.len() as f64 / expected.len().max(1) as f64;
//!     ratio.min(1.0)
//! });
//! assert!(m.evaluate("abc", "abcd") > 0.5);
//! ```

use std::collections::HashMap;

/// Trait for evaluating prediction quality against expected output.
///
/// Implementations must be `Send + Sync` so they can be shared across threads
/// in async evaluation harnesses.
///
/// # Contract
///
/// - `evaluate` must return a value in `[0.0, 1.0]`.
/// - `1.0` indicates a perfect match.
/// - `0.0` indicates no match.
pub trait Metric: Send + Sync {
    /// Evaluate the prediction against the expected output.
    /// Returns a score between 0.0 and 1.0.
    fn evaluate(&self, prediction: &str, expected: &str) -> f64;

    /// Get the metric name.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Blanket implementations
// ---------------------------------------------------------------------------

impl Metric for Box<dyn Metric> {
    #[inline]
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        (**self).evaluate(prediction, expected)
    }

    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }
}

impl<M: Metric> Metric for &M {
    #[inline]
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        (**self).evaluate(prediction, expected)
    }

    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }
}

// ---------------------------------------------------------------------------
// ExactMatch
// ---------------------------------------------------------------------------

/// Exact string match metric (trimmed).
///
/// Scores 1.0 when `prediction.trim() == expected.trim()`, 0.0 otherwise.
///
/// # Examples
///
/// ```
/// use kkachi::metric::{Metric, ExactMatch};
///
/// let m = ExactMatch;
/// assert_eq!(m.evaluate("  hello  ", "hello"), 1.0);
/// assert_eq!(m.evaluate("hello", "world"), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactMatch;

impl Metric for ExactMatch {
    #[inline]
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        if prediction.trim() == expected.trim() {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn name(&self) -> &'static str {
        "exact_match"
    }
}

// ---------------------------------------------------------------------------
// Contains
// ---------------------------------------------------------------------------

/// Substring containment metric.
///
/// Scores 1.0 when `prediction.contains(expected)`, 0.0 otherwise.
/// Both sides are trimmed before the check.
///
/// # Examples
///
/// ```
/// use kkachi::metric::{Metric, Contains};
///
/// let m = Contains;
/// assert_eq!(m.evaluate("the answer is 42", "42"), 1.0);
/// assert_eq!(m.evaluate("the answer is 41", "42"), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Contains;

impl Metric for Contains {
    #[inline]
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        if prediction.trim().contains(expected.trim()) {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn name(&self) -> &'static str {
        "contains"
    }
}

// ---------------------------------------------------------------------------
// F1Token
// ---------------------------------------------------------------------------

/// Word-level F1 score metric.
///
/// Tokenizes both strings by whitespace, computes precision and recall
/// over the token multisets, and returns their harmonic mean (F1 score).
///
/// This is commonly used in QA evaluation (e.g., SQuAD).
///
/// # Examples
///
/// ```
/// use kkachi::metric::{Metric, F1Token};
///
/// let m = F1Token;
///
/// // Identical strings
/// assert_eq!(m.evaluate("the quick brown fox", "the quick brown fox"), 1.0);
///
/// // Partial overlap
/// let score = m.evaluate("the quick brown fox", "the slow brown fox");
/// assert!(score > 0.5 && score < 1.0);
///
/// // No overlap
/// assert_eq!(m.evaluate("hello world", "foo bar"), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct F1Token;

impl F1Token {
    /// Tokenize a string into lowercased words.
    fn tokenize(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    /// Build a word frequency map.
    fn word_counts<'a>(tokens: &[&'a str]) -> HashMap<&'a str, u32> {
        let mut counts = HashMap::new();
        for &token in tokens {
            *counts.entry(token).or_insert(0) += 1;
        }
        counts
    }
}

impl Metric for F1Token {
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        let pred_tokens = Self::tokenize(prediction.trim());
        let exp_tokens = Self::tokenize(expected.trim());

        if pred_tokens.is_empty() && exp_tokens.is_empty() {
            return 1.0;
        }
        if pred_tokens.is_empty() || exp_tokens.is_empty() {
            return 0.0;
        }

        let pred_counts = Self::word_counts(&pred_tokens);
        let exp_counts = Self::word_counts(&exp_tokens);

        // Count shared tokens (intersection of multisets)
        let mut shared = 0u32;
        for (token, &pred_count) in &pred_counts {
            if let Some(&exp_count) = exp_counts.get(token) {
                shared += pred_count.min(exp_count);
            }
        }

        if shared == 0 {
            return 0.0;
        }

        let precision = shared as f64 / pred_tokens.len() as f64;
        let recall = shared as f64 / exp_tokens.len() as f64;

        // Harmonic mean (F1)
        2.0 * precision * recall / (precision + recall)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "f1_token"
    }
}

// ---------------------------------------------------------------------------
// FnMetric
// ---------------------------------------------------------------------------

/// Closure-based metric for ad-hoc evaluation functions.
///
/// Wraps any `Fn(&str, &str) -> f64` closure as a [`Metric`] implementation.
/// Useful for one-off metrics or rapid prototyping.
///
/// # Examples
///
/// ```
/// use kkachi::metric::{Metric, FnMetric};
///
/// let m = FnMetric::new("starts_with", |pred, expected| {
///     if pred.starts_with(expected) { 1.0 } else { 0.0 }
/// });
///
/// assert_eq!(m.evaluate("hello world", "hello"), 1.0);
/// assert_eq!(m.evaluate("world hello", "hello"), 0.0);
/// ```
pub struct FnMetric<F>
where
    F: Fn(&str, &str) -> f64 + Send + Sync,
{
    name: &'static str,
    func: F,
}

impl<F> FnMetric<F>
where
    F: Fn(&str, &str) -> f64 + Send + Sync,
{
    /// Create a new closure-based metric with the given name.
    pub fn new(name: &'static str, func: F) -> Self {
        Self { name, func }
    }
}

impl<F> Metric for FnMetric<F>
where
    F: Fn(&str, &str) -> f64 + Send + Sync,
{
    #[inline]
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        (self.func)(prediction, expected)
    }

    #[inline]
    fn name(&self) -> &'static str {
        self.name
    }
}

impl<F> std::fmt::Debug for FnMetric<F>
where
    F: Fn(&str, &str) -> f64 + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnMetric")
            .field("name", &self.name)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ExactMatch ----

    #[test]
    fn test_exact_match_identical() {
        let m = ExactMatch;
        assert_eq!(m.evaluate("hello", "hello"), 1.0);
    }

    #[test]
    fn test_exact_match_with_whitespace() {
        let m = ExactMatch;
        assert_eq!(m.evaluate("  hello  ", "hello"), 1.0);
        assert_eq!(m.evaluate("hello", "  hello  "), 1.0);
        assert_eq!(m.evaluate("\thello\n", "hello"), 1.0);
    }

    #[test]
    fn test_exact_match_different() {
        let m = ExactMatch;
        assert_eq!(m.evaluate("hello", "world"), 0.0);
        assert_eq!(m.evaluate("Hello", "hello"), 0.0); // case-sensitive
    }

    #[test]
    fn test_exact_match_empty() {
        let m = ExactMatch;
        assert_eq!(m.evaluate("", ""), 1.0);
        assert_eq!(m.evaluate("  ", ""), 1.0);
        assert_eq!(m.evaluate("", "hello"), 0.0);
    }

    #[test]
    fn test_exact_match_name() {
        assert_eq!(ExactMatch.name(), "exact_match");
    }

    // ---- Contains ----

    #[test]
    fn test_contains_substring() {
        let m = Contains;
        assert_eq!(m.evaluate("the answer is 42", "42"), 1.0);
        assert_eq!(m.evaluate("42", "42"), 1.0);
    }

    #[test]
    fn test_contains_no_match() {
        let m = Contains;
        assert_eq!(m.evaluate("the answer is 41", "42"), 0.0);
    }

    #[test]
    fn test_contains_with_whitespace() {
        let m = Contains;
        assert_eq!(m.evaluate("  the answer is 42  ", "42"), 1.0);
        assert_eq!(m.evaluate("the answer is 42", "  42  "), 1.0);
    }

    #[test]
    fn test_contains_empty_expected() {
        let m = Contains;
        assert_eq!(m.evaluate("anything", ""), 1.0);
    }

    #[test]
    fn test_contains_name() {
        assert_eq!(Contains.name(), "contains");
    }

    // ---- F1Token ----

    #[test]
    fn test_f1_identical() {
        let m = F1Token;
        assert_eq!(m.evaluate("the quick brown fox", "the quick brown fox"), 1.0);
    }

    #[test]
    fn test_f1_partial_overlap() {
        let m = F1Token;
        // 3 out of 4 tokens match: "the", "brown", "fox"
        // precision = 3/4, recall = 3/4, F1 = 3/4
        let score = m.evaluate("the quick brown fox", "the slow brown fox");
        assert!((score - 0.75).abs() < 1e-9, "score = {}", score);
    }

    #[test]
    fn test_f1_no_overlap() {
        let m = F1Token;
        assert_eq!(m.evaluate("hello world", "foo bar"), 0.0);
    }

    #[test]
    fn test_f1_empty_both() {
        let m = F1Token;
        assert_eq!(m.evaluate("", ""), 1.0);
    }

    #[test]
    fn test_f1_one_empty() {
        let m = F1Token;
        assert_eq!(m.evaluate("hello", ""), 0.0);
        assert_eq!(m.evaluate("", "hello"), 0.0);
    }

    #[test]
    fn test_f1_different_lengths() {
        let m = F1Token;
        // pred: "a b c d", exp: "a b"
        // shared = 2, precision = 2/4 = 0.5, recall = 2/2 = 1.0
        // F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3
        let score = m.evaluate("a b c d", "a b");
        assert!((score - 2.0 / 3.0).abs() < 1e-9, "score = {}", score);
    }

    #[test]
    fn test_f1_repeated_tokens() {
        let m = F1Token;
        // pred: "a a b", exp: "a b b"
        // pred_counts: a=2, b=1; exp_counts: a=1, b=2
        // shared: min(2,1) + min(1,2) = 1 + 1 = 2
        // precision = 2/3, recall = 2/3, F1 = 2/3
        let score = m.evaluate("a a b", "a b b");
        assert!((score - 2.0 / 3.0).abs() < 1e-9, "score = {}", score);
    }

    #[test]
    fn test_f1_name() {
        assert_eq!(F1Token.name(), "f1_token");
    }

    // ---- FnMetric ----

    #[test]
    fn test_fn_metric_basic() {
        let m = FnMetric::new("custom", |pred, exp| {
            if pred == exp {
                1.0
            } else {
                0.0
            }
        });
        assert_eq!(m.evaluate("hello", "hello"), 1.0);
        assert_eq!(m.evaluate("hello", "world"), 0.0);
    }

    #[test]
    fn test_fn_metric_continuous() {
        let m = FnMetric::new("length_ratio", |pred, expected| {
            let ratio = pred.len() as f64 / expected.len().max(1) as f64;
            ratio.min(1.0)
        });
        assert_eq!(m.evaluate("abcd", "abcd"), 1.0);
        assert!((m.evaluate("ab", "abcd") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_fn_metric_name() {
        let m = FnMetric::new("my_metric", |_, _| 0.5);
        assert_eq!(m.name(), "my_metric");
    }

    #[test]
    fn test_fn_metric_debug() {
        let m = FnMetric::new("test", |_, _| 0.0);
        let debug = format!("{:?}", m);
        assert!(debug.contains("FnMetric"));
        assert!(debug.contains("test"));
    }

    // ---- Blanket impls ----

    #[test]
    fn test_box_dyn_metric() {
        let m: Box<dyn Metric> = Box::new(ExactMatch);
        assert_eq!(m.evaluate("hello", "hello"), 1.0);
        assert_eq!(m.evaluate("hello", "world"), 0.0);
        assert_eq!(m.name(), "exact_match");
    }

    #[test]
    fn test_ref_metric() {
        let m = ExactMatch;
        let r: &ExactMatch = &m;
        assert_eq!(r.evaluate("hello", "hello"), 1.0);
        assert_eq!(r.name(), "exact_match");
    }

    #[test]
    fn test_metric_via_box_dyn_dispatch() {
        let metrics: Vec<Box<dyn Metric>> = vec![
            Box::new(ExactMatch),
            Box::new(Contains),
            Box::new(F1Token),
            Box::new(FnMetric::new("always_half", |_, _| 0.5)),
        ];

        for m in &metrics {
            let _ = m.evaluate("hello", "hello");
            let _ = m.name();
        }

        assert_eq!(metrics[0].evaluate("hello", "hello"), 1.0);
        assert_eq!(metrics[1].evaluate("hello world", "world"), 1.0);
        assert_eq!(metrics[2].evaluate("hello", "hello"), 1.0);
        assert_eq!(metrics[3].evaluate("anything", "anything"), 0.5);
    }

    // ---- Edge cases ----

    #[test]
    fn test_unicode_handling() {
        let m = ExactMatch;
        assert_eq!(m.evaluate("cafe\u{0301}", "cafe\u{0301}"), 1.0);

        let m = Contains;
        assert_eq!(m.evaluate("I ate at the cafe\u{0301}", "cafe\u{0301}"), 1.0);

        let m = F1Token;
        assert_eq!(m.evaluate("cafe\u{0301} latte", "cafe\u{0301} latte"), 1.0);
    }

    #[test]
    fn test_multiline_strings() {
        let m = ExactMatch;
        assert_eq!(m.evaluate("line1\nline2", "line1\nline2"), 1.0);

        let m = Contains;
        assert_eq!(m.evaluate("line1\nline2\nline3", "line2"), 1.0);
    }

    #[test]
    fn test_f1_single_token() {
        let m = F1Token;
        assert_eq!(m.evaluate("hello", "hello"), 1.0);
        assert_eq!(m.evaluate("hello", "world"), 0.0);
    }

    #[test]
    fn test_f1_superset() {
        let m = F1Token;
        // pred is superset of expected
        // shared = 2, precision = 2/4 = 0.5, recall = 2/2 = 1.0
        // F1 = 2 * 0.5 * 1.0 / 1.5 = 2/3
        let score = m.evaluate("a b c d", "a b");
        assert!((score - 2.0 / 3.0).abs() < 1e-9);

        // expected is superset of pred
        // shared = 2, precision = 2/2 = 1.0, recall = 2/4 = 0.5
        // F1 = 2 * 1.0 * 0.5 / 1.5 = 2/3
        let score = m.evaluate("a b", "a b c d");
        assert!((score - 2.0 / 3.0).abs() < 1e-9);
    }
}
