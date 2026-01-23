// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Validation trait for the refinement loop.
//!
//! The [`Validate`] trait provides a generic interface for validating LLM outputs.
//! Unlike the old `Critic` trait, this uses generics and lifetimes for zero-copy
//! operation and no dynamic dispatch.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{Validate, Score, checks};
//!
//! // Use a closure as a validator
//! let v = |s: &str| s.contains("fn ");
//!
//! // Use the checks() builder
//! let v = checks().require("fn ").forbid(".unwrap()");
//! ```

use crate::str_view::StrView;
use smallvec::SmallVec;
use std::borrow::Cow;

/// Validation score with optional feedback.
///
/// This is a zero-copy result type that borrows feedback strings from
/// the validator when possible.
#[derive(Debug, Clone)]
pub struct Score<'a> {
    /// The validation score between 0.0 and 1.0.
    pub value: f64,
    /// Optional feedback explaining why the score was given.
    pub feedback: Option<Cow<'a, str>>,
    /// Optional breakdown of individual check scores.
    pub breakdown: Option<SmallVec<[(&'static str, f64); 8]>>,
    /// Confidence level for this score (0.0-1.0).
    ///
    /// For deterministic validators (pattern matching, CLI tools), this is 1.0.
    /// For semantic validators (LLM-as-judge), this reflects the judge's certainty.
    pub confidence: f64,
}

impl<'a> Score<'a> {
    /// Create a passing score (1.0).
    #[inline]
    pub fn pass() -> Self {
        Self {
            value: 1.0,
            feedback: None,
            breakdown: None,
            confidence: 1.0,
        }
    }

    /// Create a failing score (0.0) with feedback.
    #[inline]
    pub fn fail(msg: &'static str) -> Self {
        Self {
            value: 0.0,
            feedback: Some(Cow::Borrowed(msg)),
            breakdown: None,
            confidence: 1.0,
        }
    }

    /// Create a partial score with feedback.
    #[inline]
    pub fn partial(value: f64, msg: &'static str) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            feedback: Some(Cow::Borrowed(msg)),
            breakdown: None,
            confidence: 1.0,
        }
    }

    /// Create a score with a specific value.
    #[inline]
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            feedback: None,
            breakdown: None,
            confidence: 1.0,
        }
    }

    /// Create a score with a specific value and owned feedback.
    #[inline]
    pub fn with_feedback(value: f64, feedback: String) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            feedback: Some(Cow::Owned(feedback)),
            breakdown: None,
            confidence: 1.0,
        }
    }

    /// Add a breakdown of individual check scores.
    #[inline]
    pub fn with_breakdown(mut self, breakdown: SmallVec<[(&'static str, f64); 8]>) -> Self {
        self.breakdown = Some(breakdown);
        self
    }

    /// Set the confidence level for this score.
    ///
    /// Confidence indicates how certain the validator is about its assessment.
    /// For deterministic validators, this should be 1.0 (default).
    /// For semantic/probabilistic validators, this may be lower.
    #[inline]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Check if the score passes (>= threshold).
    #[inline]
    pub fn passes(&self, threshold: f64) -> bool {
        self.value >= threshold
    }

    /// Check if the score is a perfect 1.0.
    #[inline]
    pub fn is_perfect(&self) -> bool {
        (self.value - 1.0).abs() < f64::EPSILON
    }

    /// Get the feedback as a string slice.
    #[inline]
    pub fn feedback_str(&self) -> Option<&str> {
        self.feedback.as_deref()
    }

    /// Convert to an owned version with 'static lifetime.
    pub fn into_owned(self) -> Score<'static> {
        Score {
            value: self.value,
            feedback: self.feedback.map(|f| Cow::Owned(f.into_owned())),
            breakdown: self.breakdown,
            confidence: self.confidence,
        }
    }
}

impl Default for Score<'_> {
    fn default() -> Self {
        Self::pass()
    }
}

/// Trait for validating LLM outputs.
///
/// This is the core abstraction for scoring generated text. Validators can be:
/// - Simple closures (`|s| s.contains("fn ")`)
/// - The [`checks()`] builder for pattern matching
/// - The [`cli()`] builder for external tool validation
/// - Custom implementations
///
/// # Generic Implementation
///
/// The trait uses `&str` input for simplicity while the refinement loop
/// can use [`StrView`] internally for zero-copy operation.
pub trait Validate: Send + Sync {
    /// Validate the given text and return a score.
    ///
    /// The returned score should be between 0.0 (complete failure) and
    /// 1.0 (perfect pass). Intermediate values indicate partial success.
    fn validate(&self, text: &str) -> Score<'static>;

    /// Get the name of this validator (for logging/debugging).
    fn name(&self) -> &'static str {
        "validator"
    }
}

/// A validator that always passes.
///
/// Use this when you want to run the refinement loop without validation,
/// for example when testing the LLM output directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoValidation;

impl Validate for NoValidation {
    #[inline]
    fn validate(&self, _text: &str) -> Score<'static> {
        Score::pass()
    }

    fn name(&self) -> &'static str {
        "no_validation"
    }
}

/// A validator that always fails.
///
/// Use this for testing or when you want to force refinement.
#[derive(Debug, Clone, Copy)]
pub struct AlwaysFail(&'static str);

impl AlwaysFail {
    /// Create a new always-fail validator with the given feedback.
    pub fn new(feedback: &'static str) -> Self {
        Self(feedback)
    }
}

impl Default for AlwaysFail {
    fn default() -> Self {
        Self("Validation always fails")
    }
}

impl Validate for AlwaysFail {
    fn validate(&self, _text: &str) -> Score<'static> {
        Score::fail(self.0)
    }

    fn name(&self) -> &'static str {
        "always_fail"
    }
}

// ============================================================================
// Blanket Implementations for Closures
// ============================================================================

/// Wrapper for closure validators that return bool.
pub struct BoolValidator<F>(pub F)
where
    F: Fn(&str) -> bool + Send + Sync;

impl<F> Validate for BoolValidator<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    fn validate(&self, text: &str) -> Score<'static> {
        if (self.0)(text) {
            Score::pass()
        } else {
            Score::fail("Validation check failed")
        }
    }

    fn name(&self) -> &'static str {
        "bool_validator"
    }
}

/// Wrapper for closure validators that return f64.
pub struct ScoreValidator<F>(pub F)
where
    F: Fn(&str) -> f64 + Send + Sync;

impl<F> Validate for ScoreValidator<F>
where
    F: Fn(&str) -> f64 + Send + Sync,
{
    fn validate(&self, text: &str) -> Score<'static> {
        let value = (self.0)(text);
        if value >= 1.0 {
            Score::pass()
        } else if value <= 0.0 {
            Score::fail("Score is zero")
        } else {
            Score::partial(value, "Partial validation score")
        }
    }

    fn name(&self) -> &'static str {
        "score_validator"
    }
}

/// Wrapper for closure validators that return Score.
pub struct FnValidator<F>(pub F)
where
    F: Fn(&str) -> Score<'static> + Send + Sync;

impl<F> Validate for FnValidator<F>
where
    F: Fn(&str) -> Score<'static> + Send + Sync,
{
    fn validate(&self, text: &str) -> Score<'static> {
        (self.0)(text)
    }

    fn name(&self) -> &'static str {
        "fn_validator"
    }
}

// ============================================================================
// StrView-based validation helper
// ============================================================================

impl<'a> StrView<'a> {
    /// Validate this string view with the given validator.
    pub fn validate<V: Validate>(&self, validator: &V) -> Score<'static> {
        validator.validate(self.as_str())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_pass() {
        let score = Score::pass();
        assert!((score.value - 1.0).abs() < f64::EPSILON);
        assert!(score.feedback.is_none());
        assert!(score.is_perfect());
        assert!(score.passes(0.5));
    }

    #[test]
    fn test_score_fail() {
        let score = Score::fail("Test failure");
        assert!((score.value - 0.0).abs() < f64::EPSILON);
        assert_eq!(score.feedback_str(), Some("Test failure"));
        assert!(!score.is_perfect());
        assert!(!score.passes(0.5));
    }

    #[test]
    fn test_score_partial() {
        let score = Score::partial(0.7, "Partial");
        assert!((score.value - 0.7).abs() < f64::EPSILON);
        assert!(score.passes(0.5));
        assert!(!score.passes(0.8));
    }

    #[test]
    fn test_score_clamping() {
        let score = Score::new(1.5);
        assert!((score.value - 1.0).abs() < f64::EPSILON);

        let score = Score::new(-0.5);
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_no_validation() {
        let v = NoValidation;
        let score = v.validate("anything");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_always_fail() {
        let v = AlwaysFail::new("Custom message");
        let score = v.validate("anything");
        assert!((score.value - 0.0).abs() < f64::EPSILON);
        assert_eq!(score.feedback_str(), Some("Custom message"));
    }

    #[test]
    fn test_bool_validator() {
        let v = BoolValidator(|s: &str| s.contains("fn "));
        assert!(v.validate("fn main() {}").is_perfect());
        assert!(!v.validate("let x = 1").is_perfect());
    }

    #[test]
    fn test_score_validator() {
        let v = ScoreValidator(|s: &str| {
            let len = s.len() as f64;
            (len / 100.0).min(1.0)
        });
        let score = v.validate("hello"); // len=5
        assert!((score.value - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fn_validator() {
        let v = FnValidator(|s: &str| {
            if s.contains("fn ") && s.contains("->") {
                Score::pass()
            } else if s.contains("fn ") {
                Score::partial(0.5, "Missing return type")
            } else {
                Score::fail("Missing function")
            }
        });

        assert!(v.validate("fn foo() -> i32 {}").is_perfect());
        assert!((v.validate("fn foo() {}").value - 0.5).abs() < f64::EPSILON);
        assert!((v.validate("let x = 1").value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_strview_validate() {
        let text = StrView::new("fn main() {}");
        let v = BoolValidator(|s: &str| s.contains("fn "));
        let score = text.validate(&v);
        assert!(score.is_perfect());
    }

    #[test]
    fn test_score_into_owned() {
        let feedback = "test".to_string();
        let score = Score::with_feedback(0.5, feedback);
        let owned: Score<'static> = score.into_owned();
        assert!((owned.value - 0.5).abs() < f64::EPSILON);
    }
}
