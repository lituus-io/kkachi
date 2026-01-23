// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Stack-allocated check builder for heuristic validation.
//!
//! The [`Checks`] builder provides a fluent API for building heuristic validators
//! that check for required patterns, forbidden patterns, length constraints, etc.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::checks;
//!
//! let v = checks()
//!     .require("fn ")
//!     .require("->")
//!     .forbid(".unwrap()")
//!     .min_len(20);
//! ```

use crate::recursive::validate::{Score, Validate};
use regex::Regex;
use smallvec::SmallVec;
use std::sync::OnceLock;

/// Create a new checks builder.
///
/// This is the entry point for building heuristic validators.
#[inline]
pub fn checks() -> Checks {
    Checks::new()
}

/// Kind of check to perform.
#[derive(Clone)]
pub enum CheckKind {
    /// Require a substring pattern.
    Require(String),
    /// Forbid a substring pattern.
    Forbid(String),
    /// Require a regex pattern match.
    Regex(String, OnceLock<Regex>),
    /// Minimum length requirement.
    MinLen(usize),
    /// Maximum length requirement.
    MaxLen(usize),
    /// Maximum error count (for output containing error lines).
    MaxErrors(usize),
    /// Custom predicate.
    Predicate(fn(&str) -> bool, &'static str),
}

/// A single check with its metadata.
#[derive(Clone)]
pub struct Check {
    name: &'static str,
    kind: CheckKind,
    weight: f64,
    feedback: &'static str,
}

/// Fluent builder for heuristic validation checks.
///
/// This builder accumulates checks and implements [`Validate`] to score
/// text based on how many checks pass. Weights determine the contribution
/// of each check to the final score.
///
/// # Stack Allocation
///
/// Checks are stored in a SmallVec with inline capacity for common cases,
/// avoiding heap allocation for typical usage patterns.
#[derive(Clone)]
pub struct Checks {
    checks: SmallVec<[Check; 8]>,
    total_weight: f64,
}

impl Default for Checks {
    fn default() -> Self {
        Self::new()
    }
}

impl Checks {
    /// Create a new empty checks builder.
    pub fn new() -> Self {
        Self {
            checks: SmallVec::new(),
            total_weight: 0.0,
        }
    }

    /// Add a check with the specified weight.
    fn add_check(mut self, check: Check) -> Self {
        self.total_weight += check.weight;
        self.checks.push(check);
        self
    }

    /// Require a substring pattern to be present.
    ///
    /// The check fails if the text does not contain the pattern.
    pub fn require(self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        let feedback = "Missing required pattern";
        self.add_check(Check {
            name: "require",
            kind: CheckKind::Require(pattern),
            weight: 1.0,
            feedback,
        })
    }

    /// Require a substring pattern with custom weight.
    pub fn require_weighted(self, pattern: impl Into<String>, weight: f64) -> Self {
        let pattern = pattern.into();
        self.add_check(Check {
            name: "require",
            kind: CheckKind::Require(pattern),
            weight,
            feedback: "Missing required pattern",
        })
    }

    /// Forbid a substring pattern.
    ///
    /// The check fails if the text contains the pattern.
    pub fn forbid(self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        self.add_check(Check {
            name: "forbid",
            kind: CheckKind::Forbid(pattern),
            weight: 1.0,
            feedback: "Contains forbidden pattern",
        })
    }

    /// Forbid a substring pattern with custom weight.
    pub fn forbid_weighted(self, pattern: impl Into<String>, weight: f64) -> Self {
        let pattern = pattern.into();
        self.add_check(Check {
            name: "forbid",
            kind: CheckKind::Forbid(pattern),
            weight,
            feedback: "Contains forbidden pattern",
        })
    }

    /// Require a regex pattern to match.
    ///
    /// The regex is lazily compiled on first use.
    pub fn regex(self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        self.add_check(Check {
            name: "regex",
            kind: CheckKind::Regex(pattern, OnceLock::new()),
            weight: 1.0,
            feedback: "Regex pattern not matched",
        })
    }

    /// Require a regex pattern with custom weight.
    pub fn regex_weighted(self, pattern: impl Into<String>, weight: f64) -> Self {
        let pattern = pattern.into();
        self.add_check(Check {
            name: "regex",
            kind: CheckKind::Regex(pattern, OnceLock::new()),
            weight,
            feedback: "Regex pattern not matched",
        })
    }

    /// Require minimum text length.
    pub fn min_len(self, n: usize) -> Self {
        self.add_check(Check {
            name: "min_len",
            kind: CheckKind::MinLen(n),
            weight: 1.0,
            feedback: "Text too short",
        })
    }

    /// Require minimum text length with custom weight.
    pub fn min_len_weighted(self, n: usize, weight: f64) -> Self {
        self.add_check(Check {
            name: "min_len",
            kind: CheckKind::MinLen(n),
            weight,
            feedback: "Text too short",
        })
    }

    /// Require maximum text length.
    pub fn max_len(self, n: usize) -> Self {
        self.add_check(Check {
            name: "max_len",
            kind: CheckKind::MaxLen(n),
            weight: 1.0,
            feedback: "Text too long",
        })
    }

    /// Require maximum text length with custom weight.
    pub fn max_len_weighted(self, n: usize, weight: f64) -> Self {
        self.add_check(Check {
            name: "max_len",
            kind: CheckKind::MaxLen(n),
            weight,
            feedback: "Text too long",
        })
    }

    /// Limit the number of error-like lines.
    ///
    /// Counts lines containing "error", "Error", or "ERROR".
    pub fn max_errors(self, n: usize) -> Self {
        self.add_check(Check {
            name: "max_errors",
            kind: CheckKind::MaxErrors(n),
            weight: 1.0,
            feedback: "Too many error lines",
        })
    }

    /// Add a custom predicate check.
    ///
    /// The predicate receives the text and returns true if it passes.
    pub fn pred(self, name: &'static str, f: fn(&str) -> bool) -> Self {
        self.add_check(Check {
            name,
            kind: CheckKind::Predicate(f, name),
            weight: 1.0,
            feedback: "Predicate check failed",
        })
    }

    /// Add a custom predicate check with custom weight.
    pub fn pred_weighted(self, name: &'static str, f: fn(&str) -> bool, weight: f64) -> Self {
        self.add_check(Check {
            name,
            kind: CheckKind::Predicate(f, name),
            weight,
            feedback: "Predicate check failed",
        })
    }

    /// Set the weight for the most recently added check.
    ///
    /// # Panics
    ///
    /// Panics if no checks have been added yet.
    pub fn weight(mut self, w: f64) -> Self {
        if let Some(check) = self.checks.last_mut() {
            self.total_weight -= check.weight;
            check.weight = w;
            self.total_weight += w;
        }
        self
    }

    /// Set custom feedback for the most recently added check.
    pub fn feedback(mut self, msg: &'static str) -> Self {
        if let Some(check) = self.checks.last_mut() {
            check.feedback = msg;
        }
        self
    }

    /// Evaluate a single check against the text.
    fn evaluate_check(check: &Check, text: &str) -> bool {
        match &check.kind {
            CheckKind::Require(pattern) => text.contains(pattern.as_str()),
            CheckKind::Forbid(pattern) => !text.contains(pattern.as_str()),
            CheckKind::Regex(pattern, compiled) => {
                let regex = compiled.get_or_init(|| {
                    Regex::new(pattern).unwrap_or_else(|_| Regex::new("^$").unwrap())
                });
                regex.is_match(text)
            }
            CheckKind::MinLen(n) => text.len() >= *n,
            CheckKind::MaxLen(n) => text.len() <= *n,
            CheckKind::MaxErrors(n) => {
                let count = text
                    .lines()
                    .filter(|line| {
                        line.contains("error") || line.contains("Error") || line.contains("ERROR")
                    })
                    .count();
                count <= *n
            }
            CheckKind::Predicate(f, _) => f(text),
        }
    }
}

impl Validate for Checks {
    fn validate(&self, text: &str) -> Score<'static> {
        if self.checks.is_empty() {
            return Score::pass();
        }

        let mut weighted_sum = 0.0;
        let mut failed_checks = Vec::new();
        let mut breakdown = SmallVec::new();

        for check in &self.checks {
            let passed = Self::evaluate_check(check, text);
            let check_score = if passed { 1.0 } else { 0.0 };
            weighted_sum += check_score * check.weight;
            breakdown.push((check.name, check_score));

            if !passed {
                failed_checks.push(check.feedback);
            }
        }

        let final_score = if self.total_weight > 0.0 {
            weighted_sum / self.total_weight
        } else {
            1.0
        };

        if failed_checks.is_empty() {
            Score::pass().with_breakdown(breakdown)
        } else {
            let feedback = failed_checks.join("; ");
            Score::with_feedback(final_score, feedback).with_breakdown(breakdown)
        }
    }

    fn name(&self) -> &'static str {
        "checks"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_checks() {
        let v = checks();
        let score = v.validate("anything");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_require() {
        let v = checks().require("fn ");
        assert!(v.validate("fn main() {}").is_perfect());
        assert!(!v.validate("let x = 1").is_perfect());
    }

    #[test]
    fn test_forbid() {
        let v = checks().forbid(".unwrap()");
        assert!(v.validate("let x = 1").is_perfect());
        assert!(!v.validate("x.unwrap()").is_perfect());
    }

    #[test]
    fn test_min_len() {
        let v = checks().min_len(10);
        assert!(v.validate("hello world").is_perfect()); // 11 chars
        assert!(!v.validate("hello").is_perfect()); // 5 chars
    }

    #[test]
    fn test_max_len() {
        let v = checks().max_len(10);
        assert!(v.validate("hello").is_perfect()); // 5 chars
        assert!(!v.validate("hello world").is_perfect()); // 11 chars
    }

    #[test]
    fn test_regex() {
        let v = checks().regex(r"fn\s+\w+");
        assert!(v.validate("fn main() {}").is_perfect());
        assert!(!v.validate("let x = 1").is_perfect());
    }

    #[test]
    fn test_max_errors() {
        let v = checks().max_errors(1);
        assert!(v.validate("line1\nline2").is_perfect());
        assert!(v.validate("error: one\nline2").is_perfect()); // 1 error
        assert!(!v.validate("error: one\nerror: two").is_perfect()); // 2 errors
    }

    #[test]
    fn test_pred() {
        let v = checks().pred("has_return", |s| s.contains("return"));
        assert!(v.validate("return 42;").is_perfect());
        assert!(!v.validate("let x = 42;").is_perfect());
    }

    #[test]
    fn test_weighted_checks() {
        let v = checks()
            .require_weighted("fn ", 0.5)
            .require_weighted("->", 0.5);

        // Both pass -> 1.0
        let score = v.validate("fn foo() -> i32 {}");
        assert!((score.value - 1.0).abs() < f64::EPSILON);

        // Only fn passes -> 0.5
        let score = v.validate("fn foo() {}");
        assert!((score.value - 0.5).abs() < f64::EPSILON);

        // Neither passes -> 0.0
        let score = v.validate("let x = 1");
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weight_modifier() {
        let v = checks()
            .require("fn ")
            .weight(2.0)
            .require("->")
            .weight(1.0);

        // fn passes (2.0), -> fails (0) -> 2.0/3.0
        let score = v.validate("fn foo() {}");
        assert!((score.value - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_breakdown() {
        let v = checks().require("fn ").forbid(".unwrap()");

        let score = v.validate("fn foo() { x.unwrap() }");
        assert!(score.breakdown.is_some());

        let breakdown = score.breakdown.unwrap();
        assert_eq!(breakdown.len(), 2);
        assert_eq!(breakdown[0], ("require", 1.0)); // fn passes
        assert_eq!(breakdown[1], ("forbid", 0.0)); // unwrap fails
    }

    #[test]
    fn test_combined_checks() {
        let v = checks()
            .require("fn ")
            .require("->")
            .forbid(".unwrap()")
            .forbid("panic!")
            .min_len(20);

        // All pass
        let code = "fn parse(s: &str) -> Option<i32> { s.parse().ok() }";
        let score = v.validate(code);
        assert!(score.is_perfect());

        // Has unwrap
        let code = "fn parse(s: &str) -> i32 { s.parse().unwrap() }";
        let score = v.validate(code);
        assert!(score.value < 1.0);
    }

    #[test]
    fn test_feedback_modifier() {
        let v = checks().require("fn ").feedback("Missing function keyword");

        let score = v.validate("let x = 1");
        assert_eq!(score.feedback_str(), Some("Missing function keyword"));
    }
}
