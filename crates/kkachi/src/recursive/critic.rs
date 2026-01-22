// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Critic trait and implementations
//!
//! Critics evaluate output quality and provide feedback for improvement.

use super::state::RecursiveState;
use crate::str_view::StrView;

/// Result of critic evaluation.
#[derive(Clone, Debug)]
pub struct CriticResult<'a> {
    /// Quality score (0.0 - 1.0)
    pub score: f64,
    /// Feedback for improvement (None if satisfactory)
    pub feedback: Option<String>,
    /// Detailed breakdown (optional)
    pub breakdown: Option<Vec<(String, f64)>>,
    /// Marker for lifetime
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> CriticResult<'a> {
    /// Create a new critic result.
    #[inline]
    pub fn new(score: f64) -> Self {
        Self {
            score,
            feedback: None,
            breakdown: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Add feedback to the result.
    #[inline]
    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    /// Add score breakdown.
    #[inline]
    pub fn with_breakdown(mut self, breakdown: Vec<(String, f64)>) -> Self {
        self.breakdown = Some(breakdown);
        self
    }

    /// Check if the output is satisfactory (no feedback needed).
    #[inline]
    pub fn is_satisfactory(&self) -> bool {
        self.feedback.is_none()
    }
}

/// Critic that evaluates output quality.
///
/// Critics are responsible for:
/// 1. Scoring outputs (0.0 - 1.0)
/// 2. Generating feedback for improvement
pub trait Critic: Send + Sync {
    /// Evaluate the output and return a score with optional feedback.
    fn evaluate<'a>(&self, output: StrView<'a>, state: &RecursiveState<'a>) -> CriticResult<'a>;
}

/// Binary critic - pass/fail based on a predicate.
#[derive(Clone)]
pub struct BinaryCritic<F> {
    check: F,
    failure_feedback: &'static str,
}

impl<F> BinaryCritic<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    /// Create a new binary critic.
    pub fn new(check: F, failure_feedback: &'static str) -> Self {
        Self {
            check,
            failure_feedback,
        }
    }
}

impl<F> Critic for BinaryCritic<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        if (self.check)(output.as_str()) {
            CriticResult::new(1.0)
        } else {
            CriticResult::new(0.0).with_feedback(self.failure_feedback)
        }
    }
}

/// Checklist critic - weighted checks.
pub struct ChecklistCritic {
    checks: Vec<ChecklistItem>,
}

struct ChecklistItem {
    name: String,
    check: Box<dyn Fn(&str) -> bool + Send + Sync>,
    weight: f32,
    feedback: String,
}

impl ChecklistCritic {
    /// Create a new checklist critic.
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Add a check to the checklist.
    pub fn add_check<F>(
        mut self,
        name: impl Into<String>,
        check: F,
        weight: f32,
        feedback: impl Into<String>,
    ) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.checks.push(ChecklistItem {
            name: name.into(),
            check: Box::new(check),
            weight,
            feedback: feedback.into(),
        });
        self
    }
}

impl Default for ChecklistCritic {
    fn default() -> Self {
        Self::new()
    }
}

impl Critic for ChecklistCritic {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        let text = output.as_str();
        let mut passed_weight = 0.0f32;
        let mut total_weight = 0.0f32;
        let mut failed_checks = Vec::new();
        let mut breakdown = Vec::new();

        for item in &self.checks {
            total_weight += item.weight;
            let passed = (item.check)(text);

            breakdown.push((item.name.clone(), if passed { 1.0 } else { 0.0 }));

            if passed {
                passed_weight += item.weight;
            } else {
                failed_checks.push(item.feedback.as_str());
            }
        }

        let score = if total_weight == 0.0 {
            1.0
        } else {
            (passed_weight / total_weight) as f64
        };

        let mut result = CriticResult::new(score).with_breakdown(breakdown);

        if !failed_checks.is_empty() {
            let feedback = failed_checks.join("\n");
            result = result.with_feedback(feedback);
        }

        result
    }
}

/// Error count critic - score based on number of errors.
#[derive(Clone)]
pub struct ErrorCountCritic<F> {
    count_errors: F,
    max_errors: usize,
}

impl<F> ErrorCountCritic<F>
where
    F: Fn(&str) -> Vec<String> + Send + Sync,
{
    /// Create a new error count critic.
    pub fn new(count_errors: F, max_errors: usize) -> Self {
        Self {
            count_errors,
            max_errors,
        }
    }
}

impl<F> Critic for ErrorCountCritic<F>
where
    F: Fn(&str) -> Vec<String> + Send + Sync,
{
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        let errors = (self.count_errors)(output.as_str());

        if errors.is_empty() {
            CriticResult::new(1.0)
        } else {
            let score = 1.0 - (errors.len() as f64 / self.max_errors as f64).min(1.0);
            let feedback = format!("Found {} errors:\n{}", errors.len(), errors.join("\n"));
            CriticResult::new(score).with_feedback(feedback)
        }
    }
}

/// Staged critic - run critics in order, stop on first required failure.
pub struct StagedCritic {
    stages: Vec<StagedItem>,
}

struct StagedItem {
    name: String,
    critic: Box<dyn Critic>,
    weight: f32,
    required: bool,
}

impl StagedCritic {
    /// Create a new staged critic.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage.
    pub fn add_stage(
        mut self,
        name: impl Into<String>,
        critic: impl Critic + 'static,
        weight: f32,
        required: bool,
    ) -> Self {
        self.stages.push(StagedItem {
            name: name.into(),
            critic: Box::new(critic),
            weight,
            required,
        });
        self
    }
}

impl Default for StagedCritic {
    fn default() -> Self {
        Self::new()
    }
}

impl Critic for StagedCritic {
    fn evaluate<'a>(&self, output: StrView<'a>, state: &RecursiveState<'a>) -> CriticResult<'a> {
        let mut total_score = 0.0f64;
        let mut total_weight = 0.0f32;
        let mut feedback_parts = Vec::new();
        let mut breakdown = Vec::new();

        for stage in &self.stages {
            let result = stage.critic.evaluate(output, state);

            breakdown.push((stage.name.clone(), result.score));
            total_score += result.score * stage.weight as f64;
            total_weight += stage.weight;

            if let Some(fb) = result.feedback {
                feedback_parts.push(format!("[{}] {}", stage.name, fb));
            }

            // Stop on required failure
            if stage.required && result.score < 1.0 {
                break;
            }
        }

        let final_score = if total_weight == 0.0 {
            0.0
        } else {
            total_score / total_weight as f64
        };

        let mut result = CriticResult::new(final_score).with_breakdown(breakdown);

        if !feedback_parts.is_empty() {
            result = result.with_feedback(feedback_parts.join("\n"));
        }

        result
    }
}

/// Heuristic critic with configurable rules.
#[derive(Clone)]
pub struct HeuristicCritic {
    /// Minimum output length
    pub min_length: Option<usize>,
    /// Maximum output length
    pub max_length: Option<usize>,
    /// Required substrings
    pub required_contains: Vec<String>,
    /// Forbidden substrings
    pub forbidden_contains: Vec<String>,
}

impl HeuristicCritic {
    /// Create a new heuristic critic with default settings.
    pub fn new() -> Self {
        Self {
            min_length: None,
            max_length: None,
            required_contains: Vec::new(),
            forbidden_contains: Vec::new(),
        }
    }

    /// Set minimum length.
    pub fn min_length(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    /// Set maximum length.
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Add required substring.
    pub fn require(mut self, s: impl Into<String>) -> Self {
        self.required_contains.push(s.into());
        self
    }

    /// Add forbidden substring.
    pub fn forbid(mut self, s: impl Into<String>) -> Self {
        self.forbidden_contains.push(s.into());
        self
    }
}

impl Default for HeuristicCritic {
    fn default() -> Self {
        Self::new()
    }
}

impl Critic for HeuristicCritic {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        let text = output.as_str();
        let mut issues = Vec::new();
        let mut checks_passed = 0;
        let mut checks_total = 0;

        // Length checks
        if let Some(min) = self.min_length {
            checks_total += 1;
            if text.len() >= min {
                checks_passed += 1;
            } else {
                issues.push(format!(
                    "Output too short (min: {}, got: {})",
                    min,
                    text.len()
                ));
            }
        }

        if let Some(max) = self.max_length {
            checks_total += 1;
            if text.len() <= max {
                checks_passed += 1;
            } else {
                issues.push(format!(
                    "Output too long (max: {}, got: {})",
                    max,
                    text.len()
                ));
            }
        }

        // Required contains
        for required in &self.required_contains {
            checks_total += 1;
            if text.contains(required) {
                checks_passed += 1;
            } else {
                issues.push(format!("Missing required content: '{}'", required));
            }
        }

        // Forbidden contains
        for forbidden in &self.forbidden_contains {
            checks_total += 1;
            if !text.contains(forbidden) {
                checks_passed += 1;
            } else {
                issues.push(format!("Contains forbidden content: '{}'", forbidden));
            }
        }

        let score = if checks_total == 0 {
            1.0
        } else {
            checks_passed as f64 / checks_total as f64
        };

        if issues.is_empty() {
            CriticResult::new(score)
        } else {
            CriticResult::new(score).with_feedback(issues.join("\n"))
        }
    }
}

/// Template-aware critic that validates output format before delegating.
///
/// This critic first checks if the output matches the template's format specification,
/// then delegates to an inner critic for additional validation.
///
/// # Example
///
/// ```ignore
/// use kkachi::recursive::{Template, TemplateCritic, HeuristicCritic};
///
/// let template = Template::from_str(r#"
/// ---
/// format:
///   type: json
///   schema:
///     type: object
///     required: [answer]
/// options:
///   strict: true
/// ---
/// Answer questions.
/// "#)?;
///
/// let inner = HeuristicCritic::new().min_length(10);
/// let critic = TemplateCritic::new(&template, inner);
/// ```
pub struct TemplateCritic<'t, C> {
    /// Reference to the template.
    template: &'t super::template::Template<'t>,
    /// Inner critic for additional validation.
    inner: C,
    /// Weight for format validation (0.0 - 1.0).
    format_weight: f32,
}

impl<'t, C: Critic> TemplateCritic<'t, C> {
    /// Create a new template critic.
    ///
    /// By default, format validation has 30% weight.
    pub fn new(template: &'t super::template::Template<'t>, inner: C) -> Self {
        Self {
            template,
            inner,
            format_weight: 0.3,
        }
    }

    /// Set the weight for format validation.
    ///
    /// The weight determines how much the format score contributes to the final score.
    /// Default is 0.3 (30%).
    pub fn with_format_weight(mut self, weight: f32) -> Self {
        self.format_weight = weight.clamp(0.0, 1.0);
        self
    }
}

impl<'t, C: Critic> Critic for TemplateCritic<'t, C> {
    fn evaluate<'a>(&self, output: StrView<'a>, state: &RecursiveState<'a>) -> CriticResult<'a> {
        let text = output.as_str();

        // First, validate format if strict mode is enabled
        let (format_score, format_feedback) = if self.template.options.strict {
            match self.template.validate_output(text) {
                Ok(()) => (1.0, None),
                Err(e) => (0.0, Some(format!("Format error: {}", e))),
            }
        } else {
            // Non-strict: try to validate but don't fail
            match self.template.validate_output(text) {
                Ok(()) => (1.0, None),
                Err(e) => (0.5, Some(format!("Format warning: {}", e))),
            }
        };

        // If format validation failed in strict mode, return immediately
        if self.template.options.strict && format_score < 1.0 {
            return CriticResult::new(0.0).with_feedback(format_feedback.unwrap_or_default());
        }

        // Delegate to inner critic
        let inner_result = self.inner.evaluate(output, state);

        // Combine scores
        let inner_weight = 1.0 - self.format_weight;
        let combined_score =
            (format_score * self.format_weight as f64) + (inner_result.score * inner_weight as f64);

        // Combine feedback
        let combined_feedback = match (format_feedback, inner_result.feedback) {
            (Some(f), Some(i)) => Some(format!("{}\n{}", f, i)),
            (Some(f), None) => Some(f),
            (None, Some(i)) => Some(i),
            (None, None) => None,
        };

        // Build breakdown
        let mut breakdown = vec![
            ("format".to_string(), format_score),
            ("content".to_string(), inner_result.score),
        ];
        if let Some(inner_breakdown) = inner_result.breakdown {
            for (name, score) in inner_breakdown {
                breakdown.push((format!("content.{}", name), score));
            }
        }

        let mut result = CriticResult::new(combined_score).with_breakdown(breakdown);

        if let Some(feedback) = combined_feedback {
            result = result.with_feedback(feedback);
        }

        result
    }
}

/// Create a fail result with a message.
impl CriticResult<'_> {
    /// Create a failing result with the given feedback.
    pub fn fail(feedback: &str) -> Self {
        Self::new(0.0).with_feedback(feedback.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_critic_pass() {
        let critic = BinaryCritic::new(|s| s.contains("hello"), "Must contain 'hello'");
        let state = RecursiveState::new();
        let result = critic.evaluate(StrView::new("hello world"), &state);

        assert_eq!(result.score, 1.0);
        assert!(result.feedback.is_none());
    }

    #[test]
    fn test_binary_critic_fail() {
        let critic = BinaryCritic::new(|s| s.contains("hello"), "Must contain 'hello'");
        let state = RecursiveState::new();
        let result = critic.evaluate(StrView::new("goodbye world"), &state);

        assert_eq!(result.score, 0.0);
        assert!(result.feedback.is_some());
    }

    #[test]
    fn test_checklist_critic() {
        let critic = ChecklistCritic::new()
            .add_check("has_hello", |s| s.contains("hello"), 1.0, "Missing 'hello'")
            .add_check("has_world", |s| s.contains("world"), 1.0, "Missing 'world'");

        let state = RecursiveState::new();

        let result = critic.evaluate(StrView::new("hello world"), &state);
        assert_eq!(result.score, 1.0);

        let result = critic.evaluate(StrView::new("hello"), &state);
        assert_eq!(result.score, 0.5);
    }

    #[test]
    fn test_heuristic_critic() {
        let critic = HeuristicCritic::new()
            .min_length(5)
            .max_length(100)
            .require("test");

        let state = RecursiveState::new();

        let result = critic.evaluate(StrView::new("this is a test"), &state);
        assert_eq!(result.score, 1.0);

        let result = critic.evaluate(StrView::new("hi"), &state);
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_error_count_critic() {
        let critic = ErrorCountCritic::new(
            |s| {
                s.lines()
                    .filter(|l| l.starts_with("ERROR:"))
                    .map(|l| l.to_string())
                    .collect()
            },
            10,
        );

        let state = RecursiveState::new();

        let result = critic.evaluate(StrView::new("OK\nOK\n"), &state);
        assert_eq!(result.score, 1.0);

        let result = critic.evaluate(StrView::new("ERROR: something\nERROR: else\n"), &state);
        assert_eq!(result.score, 0.8); // 2/10 errors = 0.2 penalty
    }
}
