// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! One-shot prompt formation and testing.
//!
//! This module provides types for:
//! - Forming optimized prompts from successful refinement runs
//! - Testing prompts in fresh contexts (no history)
//! - Capturing error corrections for RAG storage
//!
//! # Zero-Copy Design
//!
//! - Uses `StrView<'a>` for string references where possible
//! - Uses `SmallVec` for inline storage
//! - Only allocates when producing output strings
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::{OneShotPrompt, ErrorCorrection};
//!
//! // Form prompt from refinement history
//! let prompt = OneShotPrompt::from_history(
//!     "How do I create a bucket?",
//!     &iteration_history,
//!     0.95,
//! );
//!
//! // Render as prompt string
//! let text = prompt.render();
//!
//! // Test in fresh context
//! let output = llm.generate_fresh(&text).await?;
//! ```

use std::fmt::Write;

use smallvec::SmallVec;

use crate::diff::{DiffRenderer, DiffStyle, TextDiff};
use crate::recursive::IterationRecord;
use crate::str_view::StrView;

// =============================================================================
// Error Correction
// =============================================================================

/// A single error correction learned during refinement.
///
/// Captures an error message and its fix (typically as a diff).
#[derive(Clone, Debug)]
pub struct ErrorCorrection {
    /// The error message from the validator/critic.
    pub error: String,
    /// The fix applied (diff summary between iterations).
    pub fix: String,
    /// Which iteration this error occurred.
    pub iteration: u32,
}

impl ErrorCorrection {
    /// Create a new error correction.
    pub fn new(error: String, fix: String, iteration: u32) -> Self {
        Self {
            error,
            fix,
            iteration,
        }
    }
}

// =============================================================================
// One-Shot Prompt
// =============================================================================

/// An optimized one-shot prompt formed from a successful refinement run.
///
/// Contains the question, answer, and error corrections learned during
/// iterative refinement. Can be rendered as a prompt string and tested
/// in a fresh context to verify it works without conversation history.
///
/// # Zero-Copy Design
///
/// - `question` uses `StrView<'a>` when possible
/// - `error_corrections` uses `SmallVec` for inline storage
/// - `render()` is the only method that allocates
#[derive(Clone, Debug)]
pub struct OneShotPrompt<'a> {
    /// The original question/task.
    pub question: StrView<'a>,
    /// The successful answer.
    pub answer: String,
    /// Error corrections learned during refinement.
    pub error_corrections: SmallVec<[ErrorCorrection; 8]>,
    /// Final validation score.
    pub score: f64,
    /// Number of iterations to converge.
    pub iterations: u32,
}

impl<'a> OneShotPrompt<'a> {
    /// Create a new one-shot prompt.
    pub fn new(question: &'a str, answer: String, score: f64, iterations: u32) -> Self {
        Self {
            question: StrView::new(question),
            answer,
            error_corrections: SmallVec::new(),
            score,
            iterations,
        }
    }

    /// Add an error correction.
    pub fn add_correction(&mut self, correction: ErrorCorrection) {
        self.error_corrections.push(correction);
    }

    /// Form a one-shot prompt from iteration history.
    ///
    /// Extracts error corrections by comparing consecutive iterations:
    /// - If iteration N has feedback (error), iteration N+1 contains the fix
    /// - The fix is computed as a diff between the two iterations
    ///
    /// # Arguments
    ///
    /// * `question` - The original question/task
    /// * `history` - Iteration records from refinement
    /// * `final_score` - The final validation score
    pub fn from_history(
        question: &'a str,
        history: &[IterationRecord<'a>],
        final_score: f64,
    ) -> Self {
        let mut corrections = SmallVec::new();

        // Extract error corrections by diffing consecutive iterations
        for (idx, window) in history.windows(2).enumerate() {
            let prev = &window[0];
            let curr = &window[1];

            // If previous iteration had feedback (error), the current iteration is the fix
            if let Some(ref feedback) = prev.feedback {
                let fix = compute_diff_summary(prev.output.as_str(), curr.output.as_str());
                corrections.push(ErrorCorrection {
                    error: feedback.as_str().to_string(),
                    fix,
                    iteration: idx as u32,
                });
            }
        }

        // Get the final answer from the last iteration
        let answer = history
            .last()
            .map(|h| h.output.as_str().to_string())
            .unwrap_or_default();

        Self {
            question: StrView::new(question),
            answer,
            error_corrections: corrections,
            score: final_score,
            iterations: history.len() as u32,
        }
    }

    /// Render the prompt as a string for LLM generation.
    ///
    /// Produces a prompt that includes:
    /// - The question
    /// - An example answer (the successful output)
    /// - Common pitfalls to avoid (from error corrections)
    ///
    /// This is used for one-shot testing in a fresh context.
    pub fn render(&self) -> String {
        let mut prompt = String::with_capacity(2048);

        writeln!(prompt, "Question: {}\n", self.question.as_str()).unwrap();
        writeln!(prompt, "Example Answer:\n{}\n", self.answer).unwrap();

        if !self.error_corrections.is_empty() {
            prompt.push_str("Common pitfalls to avoid:\n");
            for ec in &self.error_corrections {
                writeln!(prompt, "- {}", ec.error).unwrap();
            }
            prompt.push('\n');
        }

        prompt.push_str("Now generate the answer:");
        prompt
    }

    /// Get the question.
    pub fn question(&self) -> &str {
        self.question.as_str()
    }

    /// Get the answer.
    pub fn answer(&self) -> &str {
        &self.answer
    }

    /// Get the error corrections.
    pub fn corrections(&self) -> &[ErrorCorrection] {
        &self.error_corrections
    }

    /// Check if there are any error corrections.
    pub fn has_corrections(&self) -> bool {
        !self.error_corrections.is_empty()
    }
}

// =============================================================================
// One-Shot Failure
// =============================================================================

/// Captured one-shot failure for retry context.
///
/// When one-shot testing fails, this struct captures the failure
/// information so it can be fed back into the refinement loop.
#[derive(Clone, Debug)]
pub struct OneShotFailure {
    /// Which optimization attempt this was.
    pub attempt: u32,
    /// The prompt that was used.
    pub prompt_used: String,
    /// The output generated by the LLM.
    pub output_generated: String,
    /// Errors from validation.
    pub errors: Vec<String>,
}

impl OneShotFailure {
    /// Create a new one-shot failure.
    pub fn new(
        attempt: u32,
        prompt_used: String,
        output_generated: String,
        errors: Vec<String>,
    ) -> Self {
        Self {
            attempt,
            prompt_used,
            output_generated,
            errors,
        }
    }

    /// Get a summary of the failure for feeding back into refinement.
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "One-shot attempt {} failed with {} errors:\n",
            self.attempt,
            self.errors.len()
        );
        for error in &self.errors {
            writeln!(summary, "- {}", error).unwrap();
        }
        summary
    }
}

// =============================================================================
// One-Shot Test Result
// =============================================================================

/// Result of testing a one-shot prompt in a fresh context.
#[derive(Clone, Debug)]
pub struct OneShotTestResult {
    /// The output generated by the LLM.
    pub output: String,
    /// Validation score.
    pub score: f64,
    /// Whether the test passed.
    pub passed: bool,
    /// Errors from validation (if any).
    pub errors: Vec<String>,
}

impl OneShotTestResult {
    /// Create a passing result.
    pub fn pass(output: String, score: f64) -> Self {
        Self {
            output,
            score,
            passed: true,
            errors: Vec::new(),
        }
    }

    /// Create a failing result.
    pub fn fail(output: String, score: f64, errors: Vec<String>) -> Self {
        Self {
            output,
            score,
            passed: false,
            errors,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute a human-readable diff summary between two outputs.
fn compute_diff_summary(before: &str, after: &str) -> String {
    let diff = TextDiff::new(before, after);
    let renderer = DiffRenderer::new().with_style(DiffStyle::Unified);
    renderer.render_text(&diff)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oneshot_prompt_creation() {
        let prompt = OneShotPrompt::new(
            "How do I parse JSON?",
            "import json\ndata = json.loads(text)".to_string(),
            0.95,
            3,
        );

        assert_eq!(prompt.question(), "How do I parse JSON?");
        assert_eq!(prompt.answer(), "import json\ndata = json.loads(text)");
        assert_eq!(prompt.score, 0.95);
        assert_eq!(prompt.iterations, 3);
        assert!(!prompt.has_corrections());
    }

    #[test]
    fn test_oneshot_prompt_render() {
        let mut prompt = OneShotPrompt::new(
            "How do I create a list?",
            "my_list = []".to_string(),
            0.9,
            2,
        );

        prompt.add_correction(ErrorCorrection::new(
            "NameError: 'lst' is not defined".to_string(),
            "Changed 'lst' to 'my_list'".to_string(),
            0,
        ));

        let rendered = prompt.render();

        assert!(rendered.contains("Question: How do I create a list?"));
        assert!(rendered.contains("my_list = []"));
        assert!(rendered.contains("Common pitfalls to avoid:"));
        assert!(rendered.contains("NameError"));
        assert!(rendered.contains("Now generate the answer:"));
    }

    #[test]
    fn test_oneshot_failure() {
        let failure = OneShotFailure::new(
            1,
            "Question: test".to_string(),
            "bad output".to_string(),
            vec!["Error 1".to_string(), "Error 2".to_string()],
        );

        let summary = failure.summary();
        assert!(summary.contains("attempt 1 failed"));
        assert!(summary.contains("2 errors"));
        assert!(summary.contains("Error 1"));
        assert!(summary.contains("Error 2"));
    }

    #[test]
    fn test_oneshot_test_result() {
        let pass = OneShotTestResult::pass("good output".to_string(), 0.95);
        assert!(pass.passed);
        assert!(pass.errors.is_empty());

        let fail = OneShotTestResult::fail(
            "bad output".to_string(),
            0.3,
            vec!["syntax error".to_string()],
        );
        assert!(!fail.passed);
        assert_eq!(fail.errors.len(), 1);
    }
}
