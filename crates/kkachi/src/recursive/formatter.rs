// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Prompt formatting trait for use in refinement loops.
//!
//! The [`PromptFormatter`] trait allows transforming prompts at each iteration
//! of the refinement loop, incorporating feedback and iteration context.
//! This enables template-based prompt construction without dynamic dispatch.
//!
//! # Example
//!
//! ```
//! use kkachi::recursive::formatter::{PromptFormatter, PassthroughFormatter};
//! use std::borrow::Cow;
//!
//! // The default passthrough formatter returns the prompt unchanged
//! let fmt = PassthroughFormatter;
//! let result = fmt.format("my prompt", None, 0);
//! assert_eq!(result.as_ref(), "my prompt");
//! ```

use std::borrow::Cow;

/// Trait for formatting prompts in the refinement loop.
///
/// Implementations can transform the prompt at each iteration, incorporating
/// feedback from previous iterations and the current iteration number.
/// This enables template-based prompt construction.
///
/// # Type Parameters
///
/// Uses `Cow<'a, str>` for zero-copy when no transformation is needed
/// (the common case with `PassthroughFormatter`).
pub trait PromptFormatter: Send + Sync {
    /// Format the prompt for a given iteration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The base prompt/task description
    /// * `feedback` - Feedback from the previous iteration (if any)
    /// * `iteration` - Current iteration number (0-indexed)
    ///
    /// # Returns
    ///
    /// A `Cow<str>` containing the formatted prompt. Returns `Cow::Borrowed`
    /// when no transformation is needed for zero-copy operation.
    fn format<'a>(
        &'a self,
        prompt: &'a str,
        feedback: Option<&str>,
        iteration: u32,
    ) -> Cow<'a, str>;
}

/// Default formatter that passes the prompt through unchanged.
///
/// This is the default `PromptFormatter` used when no formatter is specified.
/// It returns a borrowed reference to the original prompt with no allocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PassthroughFormatter;

impl PromptFormatter for PassthroughFormatter {
    #[inline]
    fn format<'a>(
        &'a self,
        prompt: &'a str,
        _feedback: Option<&str>,
        _iteration: u32,
    ) -> Cow<'a, str> {
        Cow::Borrowed(prompt)
    }
}

/// A formatter that appends feedback to the prompt.
///
/// This simple formatter includes any available feedback as a suffix
/// to the base prompt, separated by a newline.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedbackFormatter;

impl PromptFormatter for FeedbackFormatter {
    fn format<'a>(
        &'a self,
        prompt: &'a str,
        feedback: Option<&str>,
        _iteration: u32,
    ) -> Cow<'a, str> {
        match feedback {
            Some(fb) => Cow::Owned(format!(
                "{}\n\n[Feedback from previous attempt: {}]",
                prompt, fb
            )),
            None => Cow::Borrowed(prompt),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_formatter() {
        let fmt = PassthroughFormatter;
        let result = fmt.format("hello world", None, 0);
        assert_eq!(result.as_ref(), "hello world");
        // Verify it's borrowed (zero-copy)
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_passthrough_with_feedback() {
        let fmt = PassthroughFormatter;
        let result = fmt.format("prompt", Some("feedback"), 3);
        assert_eq!(result.as_ref(), "prompt");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_feedback_formatter_no_feedback() {
        let fmt = FeedbackFormatter;
        let result = fmt.format("prompt", None, 0);
        assert_eq!(result.as_ref(), "prompt");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_feedback_formatter_with_feedback() {
        let fmt = FeedbackFormatter;
        let result = fmt.format("my task", Some("needs improvement"), 1);
        assert!(result.contains("my task"));
        assert!(result.contains("needs improvement"));
        assert!(matches!(result, Cow::Owned(_)));
    }
}
