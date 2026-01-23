// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Validator composition using generics.
//!
//! This module provides type-safe composition of validators using generics
//! instead of dynamic dispatch. The [`And`] and [`Or`] types combine validators
//! while preserving static typing.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{checks, cli, ValidateExt};
//!
//! let heuristic = checks().require("fn ").forbid(".unwrap()");
//! let cli_check = cli("echo").arg("ok");
//!
//! // Combine with And (both must pass)
//! let combined = heuristic.and(cli_check);
//! ```

use crate::recursive::validate::{Score, Validate};

/// Compose two validators with AND semantics.
///
/// Both validators must pass for the combined validator to pass.
/// The score is the minimum of the two validators' scores.
#[derive(Debug, Clone, Copy)]
pub struct And<A, B>(pub A, pub B);

impl<A: Validate, B: Validate> Validate for And<A, B> {
    fn validate(&self, text: &str) -> Score<'static> {
        let score_a = self.0.validate(text);
        let score_b = self.1.validate(text);

        // Take the minimum score
        let min_score = score_a.value.min(score_b.value);

        // Combine feedback
        let feedback = match (score_a.feedback_str(), score_b.feedback_str()) {
            (Some(a), Some(b)) => Some(format!("{}; {}", a, b)),
            (Some(a), None) => Some(a.to_string()),
            (None, Some(b)) => Some(b.to_string()),
            (None, None) => None,
        };

        match feedback {
            Some(f) => Score::with_feedback(min_score, f),
            None => Score::new(min_score),
        }
    }

    fn name(&self) -> &'static str {
        "and"
    }
}

/// Compose two validators with OR semantics.
///
/// At least one validator must pass for the combined validator to pass.
/// The score is the maximum of the two validators' scores.
#[derive(Debug, Clone, Copy)]
pub struct Or<A, B>(pub A, pub B);

impl<A: Validate, B: Validate> Validate for Or<A, B> {
    fn validate(&self, text: &str) -> Score<'static> {
        let score_a = self.0.validate(text);
        let score_b = self.1.validate(text);

        // Take the maximum score
        let max_score = score_a.value.max(score_b.value);

        // Only include feedback if both failed
        if score_a.value >= 1.0 || score_b.value >= 1.0 {
            Score::new(max_score)
        } else {
            let feedback = match (score_a.feedback_str(), score_b.feedback_str()) {
                (Some(a), Some(b)) => format!("{} OR {}", a, b),
                (Some(a), None) => a.to_string(),
                (None, Some(b)) => b.to_string(),
                (None, None) => "Validation failed".to_string(),
            };
            Score::with_feedback(max_score, feedback)
        }
    }

    fn name(&self) -> &'static str {
        "or"
    }
}

/// Extension trait for composing validators.
///
/// This trait adds `.and()` and `.or()` methods to any type implementing
/// [`Validate`], enabling fluent composition.
pub trait ValidateExt: Validate + Sized {
    /// Combine with another validator using AND semantics.
    ///
    /// Both validators must pass for the result to pass.
    fn and<V: Validate>(self, other: V) -> And<Self, V> {
        And(self, other)
    }

    /// Combine with another validator using OR semantics.
    ///
    /// At least one validator must pass for the result to pass.
    fn or<V: Validate>(self, other: V) -> Or<Self, V> {
        Or(self, other)
    }
}

// Blanket implementation for all validators
impl<V: Validate> ValidateExt for V {}

/// Compose multiple validators with AND semantics.
///
/// All validators must pass for the result to pass.
/// The score is the minimum of all validators' scores.
#[derive(Clone)]
pub struct All<V: Validate>(pub Vec<V>);

impl<V: Validate> Validate for All<V> {
    fn validate(&self, text: &str) -> Score<'static> {
        if self.0.is_empty() {
            return Score::pass();
        }

        let mut min_score = 1.0f64;
        let mut feedbacks = Vec::new();

        for v in &self.0 {
            let score = v.validate(text);
            min_score = min_score.min(score.value);
            if let Some(fb) = score.feedback_str() {
                feedbacks.push(fb.to_string());
            }
        }

        if feedbacks.is_empty() {
            Score::new(min_score)
        } else {
            Score::with_feedback(min_score, feedbacks.join("; "))
        }
    }

    fn name(&self) -> &'static str {
        "all"
    }
}

/// Compose multiple validators with OR semantics.
///
/// At least one validator must pass for the result to pass.
/// The score is the maximum of all validators' scores.
#[derive(Clone)]
pub struct Any<V: Validate>(pub Vec<V>);

impl<V: Validate> Validate for Any<V> {
    fn validate(&self, text: &str) -> Score<'static> {
        if self.0.is_empty() {
            return Score::pass();
        }

        let mut max_score = 0.0f64;
        let mut feedbacks = Vec::new();

        for v in &self.0 {
            let score = v.validate(text);
            max_score = max_score.max(score.value);
            if let Some(fb) = score.feedback_str() {
                feedbacks.push(fb.to_string());
            }
        }

        if max_score >= 1.0 {
            Score::pass()
        } else if feedbacks.is_empty() {
            Score::new(max_score)
        } else {
            Score::with_feedback(max_score, feedbacks.join(" OR "))
        }
    }

    fn name(&self) -> &'static str {
        "any"
    }
}

/// Create an All validator from an iterator.
pub fn all<V: Validate, I: IntoIterator<Item = V>>(validators: I) -> All<V> {
    All(validators.into_iter().collect())
}

/// Create an Any validator from an iterator.
pub fn any<V: Validate, I: IntoIterator<Item = V>>(validators: I) -> Any<V> {
    Any(validators.into_iter().collect())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::validate::{BoolValidator, NoValidation};

    #[test]
    fn test_and_both_pass() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("->"));
        let combined = v1.and(v2);

        let score = combined.validate("fn foo() -> i32 {}");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_and_one_fails() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("->"));
        let combined = v1.and(v2);

        let score = combined.validate("fn foo() {}");
        assert!(!score.is_perfect());
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_or_both_pass() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("struct "));
        let combined = v1.or(v2);

        let score = combined.validate("fn foo() {} struct Bar {}");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_or_one_passes() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("struct "));
        let combined = v1.or(v2);

        let score = combined.validate("fn foo() {}");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_or_both_fail() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("struct "));
        let combined = v1.or(v2);

        let score = combined.validate("let x = 1");
        assert!(!score.is_perfect());
    }

    #[test]
    fn test_all_empty() {
        let validators: Vec<NoValidation> = vec![];
        let combined = all(validators);
        assert!(combined.validate("anything").is_perfect());
    }

    #[test]
    fn test_all_passes() {
        // Use chained .and() since different closures have different types
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("->"));
        let v3 = BoolValidator(|s: &str| s.len() > 10);
        let combined = v1.and(v2).and(v3);

        let score = combined.validate("fn foo() -> i32 {}");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_all_one_fails() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("->"));
        let combined = v1.and(v2);

        let score = combined.validate("fn foo() {}");
        assert!(!score.is_perfect());
    }

    #[test]
    fn test_any_empty() {
        let validators: Vec<NoValidation> = vec![];
        let combined = any(validators);
        assert!(combined.validate("anything").is_perfect());
    }

    #[test]
    fn test_any_one_passes() {
        // Use chained .or() since different closures have different types
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("struct "));
        let combined = v1.or(v2);

        let score = combined.validate("fn foo() {}");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_chained_composition() {
        let v1 = BoolValidator(|s: &str| s.contains("fn "));
        let v2 = BoolValidator(|s: &str| s.contains("->"));
        let v3 = BoolValidator(|s: &str| !s.contains(".unwrap()"));

        // (fn AND ->) AND !unwrap
        let combined = v1.and(v2).and(v3);

        assert!(combined.validate("fn foo() -> i32 {}").is_perfect());
        assert!(!combined.validate("fn foo() -> i32 { x.unwrap() }").is_perfect());
    }
}
