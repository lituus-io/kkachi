// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Predict module for LM-based predictions with demo management
//!
//! This module provides the [`Predict`] type, which is the core DSPy-style
//! prediction module. Uses lifetimes over Arc and GATs for zero-cost async.

use crate::buffer::{Buffer, BufferView};
use crate::error::Result;
use crate::intern::Sym;
use crate::module::Module;
use crate::prediction::Prediction;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;
use std::future::Future;
use std::ops::Range;

/// Predict module that uses an LM to generate predictions.
///
/// This is the fundamental DSPy-style module. It uses references and lifetimes
/// instead of Arc for zero-cost abstraction. The LM client is passed by reference
/// at call time rather than stored as a trait object.
///
/// ## Type Parameters
///
/// - `'sig`: Lifetime of the signature
/// - `'demo`: Lifetime of demonstration buffer
///
/// ## Example
///
/// ```ignore
/// let predict = Predict::new(&signature, &demo_buffer, &demo_indices);
/// let prediction = predict.call_with_lm(inputs, &lm_client).await?;
/// ```
pub struct Predict<'sig, 'demo> {
    /// The signature defining inputs/outputs
    signature: &'sig Signature<'sig>,
    /// Shared buffer containing all demo data
    demo_buffer: &'demo Buffer,
    /// Demo indices into the shared buffer
    demo_indices: &'demo [DemoMeta],
}

/// Copy-able range - unlike std::ops::Range, this implements Copy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FieldRange {
    /// Start byte offset (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
}

impl FieldRange {
    /// Create a new field range.
    #[inline]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Convert to std Range.
    #[inline]
    pub const fn as_range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    /// Get length.
    #[inline]
    pub const fn len(&self) -> u32 {
        self.end - self.start
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Metadata for a single demonstration - indices into shared buffer.
#[derive(Clone, Copy, Debug)]
pub struct DemoMeta {
    /// Input field ranges [(symbol, start, end)]
    pub input_ranges: [(Sym, FieldRange); 4],
    /// Number of valid input ranges
    pub input_count: u8,
    /// Output field ranges [(symbol, start, end)]
    pub output_ranges: [(Sym, FieldRange); 2],
    /// Number of valid output ranges
    pub output_count: u8,
}

impl DemoMeta {
    /// Create empty demo metadata.
    pub const fn empty() -> Self {
        Self {
            input_ranges: [(Sym::EMPTY, FieldRange::new(0, 0)); 4],
            input_count: 0,
            output_ranges: [(Sym::EMPTY, FieldRange::new(0, 0)); 2],
            output_count: 0,
        }
    }

    /// Get input fields as iterator.
    pub fn inputs(&self) -> impl Iterator<Item = (Sym, FieldRange)> + '_ {
        self.input_ranges[..self.input_count as usize]
            .iter()
            .copied()
    }

    /// Get output fields as iterator.
    pub fn outputs(&self) -> impl Iterator<Item = (Sym, FieldRange)> + '_ {
        self.output_ranges[..self.output_count as usize]
            .iter()
            .copied()
    }
}

impl Sym {
    /// Empty symbol constant for initialization.
    pub const EMPTY: Sym = unsafe { Sym::from_raw(u32::MAX) };
}

impl<'sig, 'demo> Predict<'sig, 'demo> {
    /// Create a new Predict module with references.
    pub const fn new(
        signature: &'sig Signature<'sig>,
        demo_buffer: &'demo Buffer,
        demo_indices: &'demo [DemoMeta],
    ) -> Self {
        Self {
            signature,
            demo_buffer,
            demo_indices,
        }
    }

    /// Empty buffer for use in without_demos
    const EMPTY_BUFFER: &'static Buffer = &Buffer::Empty;

    /// Create without demos.
    pub const fn without_demos(signature: &'sig Signature<'sig>) -> Self {
        Self {
            signature,
            demo_buffer: Self::EMPTY_BUFFER,
            demo_indices: &[],
        }
    }

    /// Get the signature.
    #[inline]
    pub const fn signature(&self) -> &'sig Signature<'sig> {
        self.signature
    }

    /// Get demo count.
    #[inline]
    pub const fn demo_count(&self) -> usize {
        self.demo_indices.len()
    }

    /// Build prompt into provided buffer, returning view.
    ///
    /// This avoids allocation by writing directly to the provided buffer.
    pub fn build_prompt_into<'buf>(
        &self,
        inputs: &Inputs<'_>,
        buffer: &'buf mut Vec<u8>,
    ) -> StrView<'buf> {
        buffer.clear();

        // Add instructions
        buffer.extend_from_slice(self.signature.instructions.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Add demonstrations
        for demo in self.demo_indices {
            buffer.extend_from_slice(b"Example:\n");

            // Input fields from demo buffer
            for (sym, fr) in demo.inputs() {
                if let Ok(prefix) = self.get_field_prefix(sym) {
                    buffer.extend_from_slice(prefix.as_bytes());
                    buffer.push(b' ');
                    let value = &self.demo_buffer.as_slice()[fr.as_range()];
                    buffer.extend_from_slice(value);
                    buffer.push(b'\n');
                }
            }

            // Output fields from demo buffer
            for (sym, fr) in demo.outputs() {
                if let Ok(prefix) = self.get_field_prefix(sym) {
                    buffer.extend_from_slice(prefix.as_bytes());
                    buffer.push(b' ');
                    let value = &self.demo_buffer.as_slice()[fr.as_range()];
                    buffer.extend_from_slice(value);
                    buffer.push(b'\n');
                }
            }
            buffer.push(b'\n');
        }

        // Add current input
        buffer.extend_from_slice(b"Now:\n");
        for field in &self.signature.input_fields {
            if let Some(value) = inputs.get(&field.name) {
                buffer.extend_from_slice(field.prefix.as_bytes());
                buffer.push(b' ');
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
        }

        // Prompt for output
        for field in &self.signature.output_fields {
            buffer.extend_from_slice(field.prefix.as_bytes());
            buffer.push(b' ');
        }

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    /// Get field prefix by symbol.
    fn get_field_prefix(&self, sym: Sym) -> Result<&str> {
        for field in &self.signature.input_fields {
            if crate::intern::sym(&field.name) == sym {
                return Ok(field.prefix.as_ref());
            }
        }
        for field in &self.signature.output_fields {
            if crate::intern::sym(&field.name) == sym {
                return Ok(field.prefix.as_ref());
            }
        }
        Err(crate::error::Error::module("Unknown field symbol"))
    }

    /// Parse LM response into field ranges.
    ///
    /// Returns ranges into the response buffer rather than copying strings.
    pub fn parse_response_ranges(
        &self,
        response: StrView<'_>,
    ) -> SmallVec<[(Sym, Range<usize>); 4]> {
        let mut ranges = SmallVec::new();
        let text = response.as_str();

        for field in &self.signature.output_fields {
            let prefix = field.prefix.as_ref();
            if let Some(start) = text.find(prefix) {
                let mut value_start = start + prefix.len();

                // Skip colon and whitespace
                let remaining = &text[value_start..];
                if remaining.starts_with(':') {
                    value_start += 1;
                }
                while value_start < text.len() && text.as_bytes()[value_start] == b' ' {
                    value_start += 1;
                }

                let value_end = text[value_start..]
                    .find('\n')
                    .map(|i| value_start + i)
                    .unwrap_or(text.len());

                let sym = crate::intern::sym(&field.name);
                ranges.push((sym, value_start..value_end));
            }
        }

        ranges
    }
}

impl Clone for Predict<'_, '_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Predict<'_, '_> {}

/// Trait for LM clients with GATs.
///
/// No dynamic dispatch - implementations are monomorphized at compile time.
pub trait LMClient: Send + Sync {
    /// Future type for generate - each impl provides its own
    type GenerateFut<'a>: Future<Output = Result<LMOutput<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Generate a completion.
    ///
    /// Returns a view into the response buffer rather than allocating.
    fn generate<'a>(&'a self, prompt: StrView<'a>) -> Self::GenerateFut<'a>;
}

/// Zero-copy LM output.
pub struct LMOutput<'a> {
    /// Response buffer (owned by the LM client).
    pub buffer: BufferView<'a>,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> LMOutput<'a> {
    /// Get response text as string view.
    pub fn text(&self) -> Result<StrView<'a>> {
        Ok(StrView::new(self.buffer.as_str()?))
    }
}

/// Execute predict with a specific LM client.
///
/// This is an async function that builds the prompt and calls the LM.
/// Uses static dispatch - no dynamic allocation or trait objects.
///
/// # Arguments
///
/// * `predict` - The Predict module configuration
/// * `inputs` - Input fields for this prediction
/// * `lm` - The LM client implementation
/// * `prompt_buffer` - Reusable buffer for prompt construction
///
/// # Returns
///
/// The LM output and parsed field ranges, with lifetime tied to the LM's response buffer.
pub async fn predict_with_lm<'a, L>(
    predict: &Predict<'_, '_>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<PredictOutput<'a>>
where
    L: LMClient,
{
    // Build the prompt
    let prompt = predict.build_prompt_into(inputs, prompt_buffer);

    // Call LM
    let output = lm.generate(prompt).await?;

    // Parse response
    let text = output.text()?;
    let ranges = predict.parse_response_ranges(text);

    Ok(PredictOutput {
        buffer: output.buffer,
        field_ranges: ranges,
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
    })
}

/// Zero-copy predict output.
pub struct PredictOutput<'a> {
    /// Response buffer.
    pub buffer: BufferView<'a>,
    /// Field ranges into buffer (symbol to byte range mapping).
    pub field_ranges: SmallVec<[(Sym, Range<usize>); 4]>,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> PredictOutput<'a> {
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
        self.get(crate::intern::sym(name))
    }
}

// Keep the old Module impl for backwards compatibility with simple cases
impl Module for Predict<'_, '_> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        // This is a placeholder - real usage should use predict_with_lm
        std::future::ready(Err(crate::error::Error::module(
            "Use predict_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "Predict"
    }

    fn id(&self) -> Sym {
        crate::intern::sym("predict")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::Signature;

    // MockLM defined for potential future LM integration tests
    #[allow(dead_code)]
    struct MockLM {
        #[allow(dead_code)]
        response: &'static str,
    }

    #[allow(dead_code)]
    impl LMClient for MockLM {
        type GenerateFut<'a>
            = std::future::Ready<Result<LMOutput<'a>>>
        where
            Self: 'a;

        fn generate<'a>(&'a self, _prompt: StrView<'a>) -> Self::GenerateFut<'a> {
            // Create a static buffer for testing
            static BUFFER: Buffer = Buffer::Static(b"Answer: 42");
            std::future::ready(Ok(LMOutput {
                buffer: BUFFER.view_all(),
                prompt_tokens: 10,
                completion_tokens: 5,
            }))
        }
    }

    #[test]
    fn test_predict_size() {
        // Predict should be small (3 references: 2 thin pointers + 1 fat pointer for slice)
        // &Signature (8) + &Buffer (8) + &[DemoMeta] (16) = 32 bytes on 64-bit
        assert!(std::mem::size_of::<Predict>() <= 32);
    }

    #[test]
    fn test_predict_copy() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);

        // Should be Copy
        let copy = predict;
        assert_eq!(copy.demo_count(), 0);
    }

    #[test]
    fn test_build_prompt() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);

        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let mut buffer = Vec::new();
        let prompt = predict.build_prompt_into(&inputs, &mut buffer);

        assert!(prompt.contains("Question"));
        assert!(prompt.contains("What is 2+2?"));
    }

    #[test]
    fn test_demo_meta() {
        let demo = DemoMeta::empty();
        assert_eq!(demo.input_count, 0);
        assert_eq!(demo.output_count, 0);
    }

    #[tokio::test]
    async fn test_predict_with_lm() {
        let sig = Signature::parse("question -> answer").unwrap();
        let predict = Predict::without_demos(&sig);

        let lm = MockLM {
            response: "Answer: 42",
        };
        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let mut prompt_buffer = Vec::new();
        let result = predict_with_lm(&predict, &inputs, &lm, &mut prompt_buffer).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        let answer = output.get_by_name("answer");
        assert!(answer.is_some());
    }
}
