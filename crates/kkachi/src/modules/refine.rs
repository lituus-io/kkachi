// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Refine module (Retry with Reflection)
//!
//! Implements iterative refinement where the model reflects on its
//! output and improves it based on feedback or self-assessment.
//!
//! ## Zero-Copy Design
//!
//! - All iterations share buffers
//! - Feedback operates on `StrView<'a>` references
//! - Configurable stopping conditions

use crate::buffer::BufferView;
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::module::Module;
use crate::predict::{predict_with_lm, LMClient, Predict, PredictOutput};
use crate::prediction::Prediction;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;
use std::ops::Range;

/// Feedback provider trait.
pub trait FeedbackProvider: Send + Sync {
    /// Evaluate an output and provide feedback.
    ///
    /// Returns None if the output is satisfactory (no refinement needed).
    /// Returns Some(feedback) if refinement is needed.
    fn evaluate<'a>(&self, output: &PredictOutput<'a>) -> Option<String>;
}

/// Simple feedback that always requests refinement.
pub struct AlwaysRefineFeedback {
    /// Maximum number of refinement iterations.
    pub max_iterations: u8,
    current: std::sync::atomic::AtomicU8,
}

impl AlwaysRefineFeedback {
    /// Create a new feedback provider that always requests refinement.
    pub fn new(max_iterations: u8) -> Self {
        Self {
            max_iterations,
            current: std::sync::atomic::AtomicU8::new(0),
        }
    }
}

impl FeedbackProvider for AlwaysRefineFeedback {
    fn evaluate<'a>(&self, _output: &PredictOutput<'a>) -> Option<String> {
        let current = self
            .current
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if current < self.max_iterations {
            Some("Please refine and improve your answer.".to_string())
        } else {
            None
        }
    }
}

/// Self-reflection feedback using the LM itself.
pub struct SelfReflectionFeedback {
    /// Minimum confidence threshold (0.0-1.0)
    pub threshold: f32,
}

impl Default for SelfReflectionFeedback {
    fn default() -> Self {
        Self { threshold: 0.8 }
    }
}

impl FeedbackProvider for SelfReflectionFeedback {
    fn evaluate<'a>(&self, output: &PredictOutput<'a>) -> Option<String> {
        // Simple heuristic: check for uncertainty markers
        if let Ok(text) = output.buffer.as_str() {
            let text_lower = text.to_lowercase();
            let uncertain = text_lower.contains("uncertain")
                || text_lower.contains("not sure")
                || text_lower.contains("might be")
                || text_lower.contains("possibly");

            if uncertain {
                return Some(
                    "Your answer seems uncertain. Please reconsider and provide a more confident response."
                        .to_string(),
                );
            }

            // Check for very short answers
            let answer_len = output
                .field_ranges
                .iter()
                .map(|(_, r)| r.len())
                .sum::<usize>();

            if answer_len < 10 {
                return Some(
                    "Your answer is very brief. Please elaborate with more detail.".to_string(),
                );
            }
        }

        None
    }
}

/// Refine module configuration.
#[derive(Clone, Copy)]
pub struct RefineConfig {
    /// Maximum refinement iterations
    pub max_iterations: u8,
    /// Whether to include refinement history
    pub include_history: bool,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            include_history: true,
        }
    }
}

/// Refine module.
///
/// Iteratively refines outputs based on feedback.
///
/// ## Example
///
/// ```ignore
/// let feedback = SelfReflectionFeedback::default();
/// let refine = Refine::new(&predict, feedback);
/// let result = refine_with_lm(&refine, &inputs, &lm, &mut buffer).await?;
/// ```
pub struct Refine<'pred, F: FeedbackProvider> {
    /// The underlying predict module
    predict: Predict<'pred, 'pred>,
    /// Feedback provider (reserved for future refinement logic)
    #[allow(dead_code)]
    feedback: F,
    /// Configuration
    config: RefineConfig,
}

impl<'pred, F: FeedbackProvider> Refine<'pred, F> {
    /// Create a new Refine module.
    pub fn new(predict: Predict<'pred, 'pred>, feedback: F) -> Self {
        Self {
            predict,
            feedback,
            config: RefineConfig::default(),
        }
    }

    /// Configure max iterations.
    pub fn with_max_iterations(mut self, n: u8) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Configure history inclusion.
    pub fn with_history(mut self, include: bool) -> Self {
        self.config.include_history = include;
        self
    }

    /// Get the predict module.
    #[inline]
    pub fn predict(&self) -> &Predict<'pred, 'pred> {
        &self.predict
    }

    /// Build refinement prompt with previous output and feedback.
    pub fn build_refine_prompt_into<'buf>(
        &self,
        inputs: &Inputs<'_>,
        previous_output: &str,
        feedback: &str,
        buffer: &'buf mut Vec<u8>,
    ) -> StrView<'buf> {
        buffer.clear();

        // Original instructions
        buffer.extend_from_slice(self.predict.signature().instructions.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Previous context
        buffer.extend_from_slice(b"Your previous response:\n");
        buffer.extend_from_slice(previous_output.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Feedback
        buffer.extend_from_slice(b"Feedback:\n");
        buffer.extend_from_slice(feedback.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Refinement prompt
        buffer.extend_from_slice(
            b"Please provide an improved response based on the feedback above.\n\n",
        );

        // Current input
        for field in &self.predict.signature().input_fields {
            if let Some(value) = inputs.get(&field.name) {
                buffer.extend_from_slice(field.prefix.as_bytes());
                buffer.push(b' ');
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
        }

        // Prompt for output
        for field in &self.predict.signature().output_fields {
            buffer.extend_from_slice(field.prefix.as_bytes());
            buffer.push(b' ');
        }

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }
}

/// Execute Refine with an LM client.
///
/// Note: This simplified version runs initial prediction only.
/// For full refinement loop, the caller should manage the loop externally.
pub async fn refine_with_lm<'a, L, F>(
    refine: &Refine<'_, F>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<RefineOutput<'a>>
where
    L: LMClient,
    F: FeedbackProvider,
{
    // Initial prediction
    let output = predict_with_lm(&refine.predict, inputs, lm, prompt_buffer).await?;
    let iterations = 1u8;

    // Note: Full refinement loop requires separate buffer management
    // to avoid borrow conflicts. Callers wanting multiple iterations
    // should use refine_iteration() in a loop with their own buffer management.

    Ok(RefineOutput {
        buffer: output.buffer,
        field_ranges: output.field_ranges,
        iterations,
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
    })
}

/// Run a single refinement iteration.
///
/// Returns the refined output and feedback. Returns None for feedback
/// if no further refinement is needed.
pub async fn refine_iteration<'a, L, F>(
    refine: &Refine<'_, F>,
    inputs: &Inputs<'_>,
    previous_response: &str,
    feedback_text: &str,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<RefineOutput<'a>>
where
    L: LMClient,
    F: FeedbackProvider,
{
    // Build refinement prompt
    let prompt =
        refine.build_refine_prompt_into(inputs, previous_response, feedback_text, prompt_buffer);

    // Generate refined output
    let lm_output = lm.generate(prompt).await?;
    let text = lm_output.text()?;
    let ranges = refine.predict.parse_response_ranges(text);

    Ok(RefineOutput {
        buffer: lm_output.buffer,
        field_ranges: ranges,
        iterations: 1,
        prompt_tokens: lm_output.prompt_tokens,
        completion_tokens: lm_output.completion_tokens,
    })
}

/// Zero-copy Refine output.
pub struct RefineOutput<'a> {
    /// Response buffer for final output.
    pub buffer: BufferView<'a>,
    /// Field ranges in buffer (symbol to byte range mapping).
    pub field_ranges: SmallVec<[(Sym, Range<usize>); 4]>,
    /// Number of refinement iterations.
    pub iterations: u8,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> RefineOutput<'a> {
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

impl<F: FeedbackProvider + Send + Sync> Module for Refine<'_, F> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        std::future::ready(Err(crate::error::Error::module(
            "Use refine_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "Refine"
    }

    fn id(&self) -> Sym {
        sym("refine")
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

            // Each iteration gives longer answer
            static RESPONSE_1: Buffer = Buffer::Static(b"Answer: Short");
            static RESPONSE_2: Buffer = Buffer::Static(b"Answer: Medium length answer");
            static RESPONSE_3: Buffer =
                Buffer::Static(b"Answer: This is a comprehensive and detailed answer");

            let buffer = match idx {
                0 => &RESPONSE_1,
                1 => &RESPONSE_2,
                _ => &RESPONSE_3,
            };

            std::future::ready(Ok(LMOutput {
                buffer: buffer.view_all(),
                prompt_tokens: 10,
                completion_tokens: 5 + idx as u32,
            }))
        }
    }

    #[test]
    fn test_refine_creation() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);
        let feedback = SelfReflectionFeedback::default();
        let refine = Refine::new(predict, feedback);

        assert_eq!(refine.name(), "Refine");
    }

    #[test]
    fn test_self_reflection_feedback() {
        let feedback = SelfReflectionFeedback::default();

        // Short answer should trigger refinement
        let short_buffer = Buffer::Static(b"Answer: 42");
        let short_output = PredictOutput {
            buffer: short_buffer.view_all(),
            field_ranges: smallvec::smallvec![(sym("answer"), 8..10)],
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        assert!(feedback.evaluate(&short_output).is_some());

        // Uncertain answer should trigger refinement
        let uncertain_buffer = Buffer::Static(b"Answer: I'm not sure but it might be 42");
        let uncertain_output = PredictOutput {
            buffer: uncertain_buffer.view_all(),
            field_ranges: smallvec::smallvec![(sym("answer"), 8..40)],
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        assert!(feedback.evaluate(&uncertain_output).is_some());

        // Good answer should not trigger refinement
        let good_buffer = Buffer::Static(b"Answer: The capital of France is definitely Paris");
        let good_output = PredictOutput {
            buffer: good_buffer.view_all(),
            field_ranges: smallvec::smallvec![(sym("answer"), 8..50)],
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        assert!(feedback.evaluate(&good_output).is_none());
    }

    #[test]
    fn test_always_refine_feedback() {
        let feedback = AlwaysRefineFeedback::new(2);

        let buffer = Buffer::Static(b"Answer: test");
        let output = PredictOutput {
            buffer: buffer.view_all(),
            field_ranges: smallvec::smallvec![],
            prompt_tokens: 10,
            completion_tokens: 5,
        };

        // First two should request refinement
        assert!(feedback.evaluate(&output).is_some());
        assert!(feedback.evaluate(&output).is_some());
        // Third should not
        assert!(feedback.evaluate(&output).is_none());
    }

    #[tokio::test]
    async fn test_refine_with_lm() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);
        let feedback = AlwaysRefineFeedback::new(2);
        let refine = Refine::new(predict, feedback);

        let lm = MockLM {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };

        let mut inputs = Inputs::new();
        inputs.insert("question", "Test question");

        let mut buffer = Vec::new();
        let result = refine_with_lm(&refine, &inputs, &lm, &mut buffer).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        // Initial prediction runs once
        assert_eq!(output.iterations, 1);
    }
}
