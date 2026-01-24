// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Chain of Thought module
//!
//! Implements the Chain of Thought (CoT) reasoning strategy where the LM
//! is prompted to show its reasoning steps before providing an answer.
//!
//! ## Zero-Copy Design
//!
//! - Uses `StrView<'a>` for all string references
//! - Module is Copy (just references + config)
//! - No intermediate allocations during execution

use crate::buffer::BufferView;
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::module::Module;
use crate::predict::{LMClient, Predict};
use crate::prediction::Prediction;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;
use std::ops::Range;

/// Chain of Thought module.
///
/// Wraps a Predict module and adds a "reasoning" or "thought" field
/// to the signature, prompting the LM to show its work before answering.
///
/// ## Example
///
/// ```ignore
/// let sig = Signature::parse("question -> answer").unwrap();
/// let cot = ChainOfThought::new(&sig);
/// // Signature becomes: question -> reasoning, answer
/// ```
#[derive(Clone, Copy)]
pub struct ChainOfThought<'sig, 'demo> {
    /// The underlying predict module
    predict: Predict<'sig, 'demo>,
    /// Symbol for the reasoning field
    reasoning_sym: Sym,
    /// Whether to include rationale in output
    include_rationale: bool,
}

impl<'sig, 'demo> ChainOfThought<'sig, 'demo> {
    /// Create a new ChainOfThought module.
    ///
    /// This wraps the signature to include a "reasoning" output field
    /// before the original output fields.
    pub fn new(signature: &'sig Signature<'sig>) -> Self {
        Self {
            predict: Predict::without_demos(signature),
            reasoning_sym: sym("reasoning"),
            include_rationale: true,
        }
    }

    /// Create with custom predict module (with demos).
    pub fn with_predict(predict: Predict<'sig, 'demo>) -> Self {
        Self {
            predict,
            reasoning_sym: sym("reasoning"),
            include_rationale: true,
        }
    }

    /// Set the reasoning field name.
    pub fn with_reasoning_field(mut self, name: &str) -> Self {
        self.reasoning_sym = sym(name);
        self
    }

    /// Configure whether to include rationale in output.
    pub fn with_rationale(mut self, include: bool) -> Self {
        self.include_rationale = include;
        self
    }

    /// Get the underlying predict module.
    #[inline]
    pub fn predict(&self) -> &Predict<'sig, 'demo> {
        &self.predict
    }

    /// Get the reasoning field symbol.
    #[inline]
    pub fn reasoning_sym(&self) -> Sym {
        self.reasoning_sym
    }

    /// Build CoT prompt into provided buffer.
    ///
    /// Modifies the standard prompt to include reasoning instructions.
    pub fn build_prompt_into<'buf>(
        &self,
        inputs: &Inputs<'_>,
        buffer: &'buf mut Vec<u8>,
    ) -> StrView<'buf> {
        buffer.clear();

        // Add CoT prefix
        buffer.extend_from_slice(b"Let's think step by step.\n\n");

        // Add instructions
        buffer.extend_from_slice(self.predict.signature().instructions.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Add current input
        buffer.extend_from_slice(b"Now:\n");
        for field in &self.predict.signature().input_fields {
            if let Some(value) = inputs.get(&field.name) {
                buffer.extend_from_slice(field.prefix.as_bytes());
                buffer.push(b' ');
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
        }

        // Add reasoning field prompt
        buffer.extend_from_slice(b"\nReasoning: ");

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    /// Parse CoT response, extracting reasoning and answer.
    pub fn parse_response<'a>(&self, response: StrView<'a>) -> SmallVec<[(Sym, Range<usize>); 4]> {
        let mut ranges = SmallVec::new();
        let text = response.as_str();

        // Find reasoning section
        if let Some(reasoning_start) = text.find("Reasoning:") {
            let value_start = reasoning_start + "Reasoning:".len();
            let value_start = skip_whitespace(text, value_start);

            // Reasoning ends at "Answer:" or end of text
            let value_end = text[value_start..]
                .find("Answer:")
                .map(|i| value_start + i)
                .unwrap_or(text.len());

            ranges.push((self.reasoning_sym, value_start..value_end.saturating_sub(1)));
        }

        // Parse other fields from underlying predict
        let other_ranges = self.predict.parse_response_ranges(response);
        ranges.extend(other_ranges);

        ranges
    }
}

/// Skip whitespace in text starting from pos.
#[inline]
fn skip_whitespace(text: &str, mut pos: usize) -> usize {
    let bytes = text.as_bytes();
    while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n') {
        pos += 1;
    }
    pos
}

/// Execute ChainOfThought with an LM client.
pub async fn cot_with_lm<'a, L>(
    cot: &ChainOfThought<'_, '_>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<CoTOutput<'a>>
where
    L: LMClient,
{
    // Build CoT prompt
    let prompt = cot.build_prompt_into(inputs, prompt_buffer);

    // Call LM
    let output = lm.generate(prompt).await?;

    // Parse response
    let text = output.text()?;
    let ranges = cot.parse_response(text);

    // Extract reasoning
    let reasoning_range = ranges
        .iter()
        .find(|(s, _)| *s == cot.reasoning_sym)
        .map(|(_, r)| r.clone());

    Ok(CoTOutput {
        buffer: output.buffer,
        field_ranges: ranges,
        reasoning_range,
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
    })
}

/// Zero-copy Chain of Thought output.
pub struct CoTOutput<'a> {
    /// Response buffer
    pub buffer: BufferView<'a>,
    /// All field ranges into buffer.
    pub field_ranges: SmallVec<[(Sym, Range<usize>); 4]>,
    /// Range for reasoning specifically.
    pub reasoning_range: Option<Range<usize>>,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> CoTOutput<'a> {
    /// Get the reasoning text.
    pub fn reasoning(&self) -> Option<StrView<'a>> {
        let range = self.reasoning_range.as_ref()?;
        let text = self.buffer.as_str().ok()?;
        Some(StrView::new(&text[range.clone()]))
    }

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

impl Module for ChainOfThought<'_, '_> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        // Placeholder - real usage should use cot_with_lm
        std::future::ready(Err(crate::error::Error::module(
            "Use cot_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "ChainOfThought"
    }

    fn id(&self) -> Sym {
        sym("chain_of_thought")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::predict::LMOutput;

    struct MockLM;

    impl LMClient for MockLM {
        type GenerateFut<'a>
            = std::future::Ready<Result<LMOutput<'a>>>
        where
            Self: 'a;

        fn generate<'a>(&'a self, _prompt: StrView<'a>) -> Self::GenerateFut<'a> {
            static BUFFER: Buffer = Buffer::Static(
                b"Reasoning: First, I need to add 2 + 2 which equals 4.\n\nAnswer: 4",
            );
            std::future::ready(Ok(LMOutput {
                buffer: BUFFER.view_all(),
                prompt_tokens: 20,
                completion_tokens: 15,
            }))
        }
    }

    #[test]
    fn test_cot_creation() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        assert_eq!(cot.name(), "ChainOfThought");
        assert_eq!(cot.reasoning_sym(), sym("reasoning"));
    }

    #[test]
    fn test_cot_copy() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        let copy = cot; // Copy, not move
        assert_eq!(cot.name(), copy.name());
    }

    #[test]
    fn test_cot_size() {
        // ChainOfThought should be small
        assert!(std::mem::size_of::<ChainOfThought>() <= 48);
    }

    #[tokio::test]
    async fn test_cot_with_lm() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);
        let lm = MockLM;

        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2 + 2?");

        let mut buffer = Vec::new();
        let result = cot_with_lm(&cot, &inputs, &lm, &mut buffer).await;

        assert!(result.is_ok());
        let output = result.unwrap();

        // Check we got reasoning
        let reasoning = output.reasoning();
        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().as_str().contains("add 2 + 2"));
    }

    #[test]
    fn test_parse_response() {
        let sig = Signature::parse("question -> answer").unwrap();
        let cot = ChainOfThought::new(&sig);

        let response = StrView::new(
            "Reasoning: Step 1: Consider the question.\nStep 2: Calculate.\n\nAnswer: 42",
        );
        let ranges = cot.parse_response(response);

        assert!(!ranges.is_empty());

        // Check reasoning was extracted
        let reasoning_range = ranges.iter().find(|(s, _)| *s == sym("reasoning"));
        assert!(reasoning_range.is_some());
    }
}
