// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Zero-Copy Assertions
//!
//! Assertions validate module outputs with zero-allocation operations.
//! Supports both hard assertions (failures stop execution) and soft suggestions.
//!
//! ## Assertion Levels
//!
//! - `Assert`: Hard constraint - failure is an error
//! - `Suggest`: Soft constraint - failure is a warning
//!
//! ## Built-in Assertions
//!
//! - `FieldExists` - Check field presence
//! - `LengthBounds` - Min/max length validation
//! - `RegexMatch` - Pattern matching
//! - `JsonValid` - JSON structure validation
//! - `NotEmpty` - Non-empty string check
//! - `Contains` - Substring check

use crate::error::{Error, Result};
use crate::intern::Sym;
use crate::str_view::StrView;
use smallvec::SmallVec;

/// Assertion severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertionLevel {
    /// Hard assertion - failure is an error.
    Assert,
    /// Soft suggestion - failure is a warning.
    Suggest,
}

/// Result of an assertion check.
#[derive(Debug, Clone, Copy)]
pub struct AssertionResult {
    /// Whether the assertion passed.
    pub passed: bool,
    /// The severity level.
    pub level: AssertionLevel,
    /// The field that was checked.
    pub field: Sym,
}

impl AssertionResult {
    /// Create a new passing result.
    #[inline]
    pub const fn pass(field: Sym, level: AssertionLevel) -> Self {
        Self {
            passed: true,
            level,
            field,
        }
    }

    /// Create a new failing result.
    #[inline]
    pub const fn fail(field: Sym, level: AssertionLevel) -> Self {
        Self {
            passed: false,
            level,
            field,
        }
    }

    /// Check if this is a hard failure (Assert level that didn't pass).
    #[inline]
    pub const fn is_hard_failure(&self) -> bool {
        !self.passed && matches!(self.level, AssertionLevel::Assert)
    }
}

/// Zero-copy assertion trait.
///
/// Assertions operate directly on `StrView` without allocation.
pub trait Assertion: Send + Sync {
    /// Check the assertion against a field value.
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult;

    /// Get the assertion level.
    fn level(&self) -> AssertionLevel;

    /// Get a description of this assertion.
    fn describe(&self) -> &'static str;
}

/// Assertion runner that processes multiple assertions.
///
/// Short-circuits on hard assertion failures.
pub struct AssertionRunner<'a> {
    assertions: &'a [&'a dyn Assertion],
}

impl<'a> AssertionRunner<'a> {
    /// Create a new assertion runner.
    pub const fn new(assertions: &'a [&'a dyn Assertion]) -> Self {
        Self { assertions }
    }

    /// Run all assertions on a field value.
    ///
    /// Returns early on hard assertion failure.
    pub fn run(&self, field: Sym, value: StrView<'_>) -> Result<SmallVec<[AssertionResult; 4]>> {
        let mut results = SmallVec::new();

        for assertion in self.assertions {
            let result = assertion.check(field, value);
            results.push(result);

            // Short-circuit on hard failure
            if result.is_hard_failure() {
                return Err(Error::AssertionFailed {
                    field,
                    description: assertion.describe(),
                });
            }
        }

        Ok(results)
    }

    /// Run all assertions and collect warnings (non-fatal).
    pub fn run_soft(&self, field: Sym, value: StrView<'_>) -> SmallVec<[AssertionResult; 4]> {
        self.assertions
            .iter()
            .map(|a| a.check(field, value))
            .collect()
    }

    /// Check if all assertions pass.
    pub fn all_pass(&self, field: Sym, value: StrView<'_>) -> bool {
        self.assertions.iter().all(|a| a.check(field, value).passed)
    }

    /// Count passing assertions.
    pub fn count_passing(&self, field: Sym, value: StrView<'_>) -> usize {
        self.assertions
            .iter()
            .filter(|a| a.check(field, value).passed)
            .count()
    }
}

// ============================================================================
// Built-in Assertions
// ============================================================================

/// Check that a field value is not empty.
#[derive(Debug, Clone, Copy)]
pub struct NotEmpty {
    level: AssertionLevel,
}

impl NotEmpty {
    /// Create a new NotEmpty assertion.
    pub const fn new(level: AssertionLevel) -> Self {
        Self { level }
    }

    /// Create as a hard assertion.
    pub const fn assert() -> Self {
        Self::new(AssertionLevel::Assert)
    }

    /// Create as a soft suggestion.
    pub const fn suggest() -> Self {
        Self::new(AssertionLevel::Suggest)
    }
}

impl Assertion for NotEmpty {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        if value.as_str().trim().is_empty() {
            AssertionResult::fail(field, self.level)
        } else {
            AssertionResult::pass(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must not be empty"
    }
}

/// Check that a field value has length within bounds.
#[derive(Debug, Clone, Copy)]
pub struct LengthBounds {
    min: Option<usize>,
    max: Option<usize>,
    level: AssertionLevel,
}

impl LengthBounds {
    /// Create a new LengthBounds assertion.
    pub const fn new(min: Option<usize>, max: Option<usize>, level: AssertionLevel) -> Self {
        Self { min, max, level }
    }

    /// Create with minimum length.
    pub const fn min(min: usize, level: AssertionLevel) -> Self {
        Self::new(Some(min), None, level)
    }

    /// Create with maximum length.
    pub const fn max(max: usize, level: AssertionLevel) -> Self {
        Self::new(None, Some(max), level)
    }

    /// Create with both bounds.
    pub const fn between(min: usize, max: usize, level: AssertionLevel) -> Self {
        Self::new(Some(min), Some(max), level)
    }
}

impl Assertion for LengthBounds {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        let len = value.as_str().len();

        let min_ok = self.min.map_or(true, |m| len >= m);
        let max_ok = self.max.map_or(true, |m| len <= m);

        if min_ok && max_ok {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        match (self.min, self.max) {
            (Some(_), Some(_)) => "value length must be within bounds",
            (Some(_), None) => "value length must meet minimum",
            (None, Some(_)) => "value length must not exceed maximum",
            (None, None) => "length check (no bounds set)",
        }
    }
}

/// Check that a field value contains a substring.
#[derive(Debug, Clone)]
pub struct Contains {
    substring: &'static str,
    level: AssertionLevel,
}

impl Contains {
    /// Create a new Contains assertion.
    pub const fn new(substring: &'static str, level: AssertionLevel) -> Self {
        Self { substring, level }
    }

    /// Create as a hard assertion.
    pub const fn assert(substring: &'static str) -> Self {
        Self::new(substring, AssertionLevel::Assert)
    }

    /// Create as a soft suggestion.
    pub const fn suggest(substring: &'static str) -> Self {
        Self::new(substring, AssertionLevel::Suggest)
    }
}

impl Assertion for Contains {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        if value.as_str().contains(self.substring) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must contain required substring"
    }
}

/// Check that a field value matches a regex pattern.
#[derive(Debug)]
pub struct RegexMatch {
    pattern: regex::Regex,
    level: AssertionLevel,
}

impl RegexMatch {
    /// Create a new RegexMatch assertion.
    ///
    /// Panics if the pattern is invalid.
    pub fn new(pattern: &str, level: AssertionLevel) -> Self {
        Self {
            pattern: regex::Regex::new(pattern).expect("invalid regex pattern"),
            level,
        }
    }

    /// Try to create a RegexMatch assertion.
    pub fn try_new(pattern: &str, level: AssertionLevel) -> Result<Self> {
        let pattern =
            regex::Regex::new(pattern).map_err(|e| Error::InvalidPattern(e.to_string()))?;
        Ok(Self { pattern, level })
    }

    /// Create as a hard assertion.
    pub fn assert(pattern: &str) -> Self {
        Self::new(pattern, AssertionLevel::Assert)
    }

    /// Create as a soft suggestion.
    pub fn suggest(pattern: &str) -> Self {
        Self::new(pattern, AssertionLevel::Suggest)
    }
}

impl Assertion for RegexMatch {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        if self.pattern.is_match(value.as_str()) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must match pattern"
    }
}

/// Check that a field value is valid JSON.
#[derive(Debug, Clone, Copy)]
pub struct JsonValid {
    level: AssertionLevel,
}

impl JsonValid {
    /// Create a new JsonValid assertion.
    pub const fn new(level: AssertionLevel) -> Self {
        Self { level }
    }

    /// Create as a hard assertion.
    pub const fn assert() -> Self {
        Self::new(AssertionLevel::Assert)
    }

    /// Create as a soft suggestion.
    pub const fn suggest() -> Self {
        Self::new(AssertionLevel::Suggest)
    }

    /// Check if a string is valid JSON using serde_json.
    fn is_valid_json(s: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(s).is_ok()
    }
}

impl Assertion for JsonValid {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        let text = value.as_str().trim();

        if Self::is_valid_json(text) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must be valid JSON"
    }
}

/// Check that a field value starts with a prefix.
#[derive(Debug, Clone)]
pub struct StartsWith {
    prefix: &'static str,
    level: AssertionLevel,
}

impl StartsWith {
    /// Create a new StartsWith assertion.
    pub const fn new(prefix: &'static str, level: AssertionLevel) -> Self {
        Self { prefix, level }
    }

    /// Create as a hard assertion.
    pub const fn assert(prefix: &'static str) -> Self {
        Self::new(prefix, AssertionLevel::Assert)
    }
}

impl Assertion for StartsWith {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        if value.as_str().starts_with(self.prefix) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must start with prefix"
    }
}

/// Check that a field value ends with a suffix.
#[derive(Debug, Clone)]
pub struct EndsWith {
    suffix: &'static str,
    level: AssertionLevel,
}

impl EndsWith {
    /// Create a new EndsWith assertion.
    pub const fn new(suffix: &'static str, level: AssertionLevel) -> Self {
        Self { suffix, level }
    }

    /// Create as a hard assertion.
    pub const fn assert(suffix: &'static str) -> Self {
        Self::new(suffix, AssertionLevel::Assert)
    }
}

impl Assertion for EndsWith {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        if value.as_str().ends_with(self.suffix) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must end with suffix"
    }
}

/// Check that a field value is one of allowed values.
#[derive(Debug, Clone)]
pub struct OneOf {
    allowed: &'static [&'static str],
    level: AssertionLevel,
}

impl OneOf {
    /// Create a new OneOf assertion.
    pub const fn new(allowed: &'static [&'static str], level: AssertionLevel) -> Self {
        Self { allowed, level }
    }

    /// Create as a hard assertion.
    pub const fn assert(allowed: &'static [&'static str]) -> Self {
        Self::new(allowed, AssertionLevel::Assert)
    }
}

impl Assertion for OneOf {
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        let text = value.as_str().trim();
        if self.allowed.contains(&text) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        "value must be one of allowed values"
    }
}

/// Custom assertion using a function.
pub struct Custom<F> {
    check_fn: F,
    description: &'static str,
    level: AssertionLevel,
}

impl<F> Custom<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    /// Create a new custom assertion.
    pub const fn new(check_fn: F, description: &'static str, level: AssertionLevel) -> Self {
        Self {
            check_fn,
            description,
            level,
        }
    }
}

impl<F> Assertion for Custom<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    fn check<'a>(&self, field: Sym, value: StrView<'a>) -> AssertionResult {
        if (self.check_fn)(value.as_str()) {
            AssertionResult::pass(field, self.level)
        } else {
            AssertionResult::fail(field, self.level)
        }
    }

    fn level(&self) -> AssertionLevel {
        self.level
    }

    fn describe(&self) -> &'static str {
        self.description
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    #[test]
    fn test_assertion_level() {
        assert_eq!(AssertionLevel::Assert, AssertionLevel::Assert);
        assert_ne!(AssertionLevel::Assert, AssertionLevel::Suggest);
    }

    #[test]
    fn test_assertion_result() {
        let field = sym("test");

        let pass = AssertionResult::pass(field, AssertionLevel::Assert);
        assert!(pass.passed);
        assert!(!pass.is_hard_failure());

        let fail = AssertionResult::fail(field, AssertionLevel::Assert);
        assert!(!fail.passed);
        assert!(fail.is_hard_failure());

        let soft_fail = AssertionResult::fail(field, AssertionLevel::Suggest);
        assert!(!soft_fail.passed);
        assert!(!soft_fail.is_hard_failure());
    }

    #[test]
    fn test_not_empty() {
        let field = sym("answer");
        let assertion = NotEmpty::assert();

        let result = assertion.check(field, StrView::new("hello"));
        assert!(result.passed);

        let result = assertion.check(field, StrView::new(""));
        assert!(!result.passed);

        let result = assertion.check(field, StrView::new("   "));
        assert!(!result.passed);
    }

    #[test]
    fn test_length_bounds() {
        let field = sym("answer");

        let min_check = LengthBounds::min(5, AssertionLevel::Assert);
        assert!(min_check.check(field, StrView::new("hello")).passed);
        assert!(!min_check.check(field, StrView::new("hi")).passed);

        let max_check = LengthBounds::max(5, AssertionLevel::Assert);
        assert!(max_check.check(field, StrView::new("hello")).passed);
        assert!(!max_check.check(field, StrView::new("hello world")).passed);

        let range_check = LengthBounds::between(3, 10, AssertionLevel::Assert);
        assert!(range_check.check(field, StrView::new("hello")).passed);
        assert!(!range_check.check(field, StrView::new("hi")).passed);
        assert!(
            !range_check
                .check(field, StrView::new("hello world!"))
                .passed
        );
    }

    #[test]
    fn test_contains() {
        let field = sym("answer");
        let assertion = Contains::assert("world");

        assert!(assertion.check(field, StrView::new("hello world")).passed);
        assert!(!assertion.check(field, StrView::new("hello")).passed);
    }

    #[test]
    fn test_regex_match() {
        let field = sym("answer");
        let assertion = RegexMatch::assert(r"^\d+$");

        assert!(assertion.check(field, StrView::new("12345")).passed);
        assert!(!assertion.check(field, StrView::new("hello")).passed);
        assert!(!assertion.check(field, StrView::new("123abc")).passed);
    }

    #[test]
    fn test_json_valid() {
        let field = sym("data");
        let assertion = JsonValid::assert();

        assert!(
            assertion
                .check(field, StrView::new(r#"{"key": "value"}"#))
                .passed
        );
        assert!(assertion.check(field, StrView::new(r#"[1, 2, 3]"#)).passed);
        assert!(
            assertion
                .check(field, StrView::new(r#"{"nested": {"a": 1}}"#))
                .passed
        );

        assert!(!assertion.check(field, StrView::new("not json")).passed);
        assert!(!assertion.check(field, StrView::new("{incomplete")).passed);
        assert!(
            !assertion
                .check(field, StrView::new(r#"{"unclosed": "string}"#))
                .passed
        );
    }

    #[test]
    fn test_starts_with() {
        let field = sym("answer");
        let assertion = StartsWith::assert("Answer:");

        assert!(assertion.check(field, StrView::new("Answer: 42")).passed);
        assert!(
            !assertion
                .check(field, StrView::new("The answer is 42"))
                .passed
        );
    }

    #[test]
    fn test_ends_with() {
        let field = sym("answer");
        let assertion = EndsWith::assert(".");

        assert!(
            assertion
                .check(field, StrView::new("This is a sentence."))
                .passed
        );
        assert!(!assertion.check(field, StrView::new("No period")).passed);
    }

    #[test]
    fn test_one_of() {
        let field = sym("choice");
        let assertion = OneOf::assert(&["yes", "no", "maybe"]);

        assert!(assertion.check(field, StrView::new("yes")).passed);
        assert!(assertion.check(field, StrView::new("no")).passed);
        assert!(assertion.check(field, StrView::new("  maybe  ")).passed); // trimmed
        assert!(!assertion.check(field, StrView::new("perhaps")).passed);
    }

    #[test]
    fn test_custom() {
        let field = sym("number");
        let assertion = Custom::new(
            |s| s.parse::<i32>().map_or(false, |n| n > 0),
            "must be positive integer",
            AssertionLevel::Assert,
        );

        assert!(assertion.check(field, StrView::new("42")).passed);
        assert!(!assertion.check(field, StrView::new("-5")).passed);
        assert!(!assertion.check(field, StrView::new("abc")).passed);
    }

    #[test]
    fn test_runner_all_pass() {
        let field = sym("answer");
        let not_empty = NotEmpty::assert();
        let min_len = LengthBounds::min(3, AssertionLevel::Assert);

        let assertions: &[&dyn Assertion] = &[&not_empty, &min_len];
        let runner = AssertionRunner::new(assertions);

        assert!(runner.all_pass(field, StrView::new("hello")));
        assert!(!runner.all_pass(field, StrView::new("hi")));
    }

    #[test]
    fn test_runner_count_passing() {
        let field = sym("answer");
        let not_empty = NotEmpty::assert();
        let min_len = LengthBounds::min(10, AssertionLevel::Suggest);

        let assertions: &[&dyn Assertion] = &[&not_empty, &min_len];
        let runner = AssertionRunner::new(assertions);

        assert_eq!(runner.count_passing(field, StrView::new("hello")), 1);
        assert_eq!(runner.count_passing(field, StrView::new("hello world")), 2);
    }

    #[test]
    fn test_runner_soft() {
        let field = sym("answer");
        let not_empty = NotEmpty::suggest();
        let min_len = LengthBounds::min(10, AssertionLevel::Suggest);

        let assertions: &[&dyn Assertion] = &[&not_empty, &min_len];
        let runner = AssertionRunner::new(assertions);

        let results = runner.run_soft(field, StrView::new("hi"));
        assert_eq!(results.len(), 2);
        assert!(results[0].passed); // not empty
        assert!(!results[1].passed); // too short
    }
}
