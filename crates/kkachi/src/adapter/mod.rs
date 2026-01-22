// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Zero-Copy Adapters
//!
//! Adapters format prompts and parse responses for different LM interfaces.
//! Uses zero-copy string handling for efficiency.
//!
//! ## Available Adapters
//!
//! - `ChatAdapter`: Chat-style prompt formatting
//! - `JSONAdapter`: JSON response parsing
//! - `XMLAdapter`: XML response parsing

pub mod chat;
pub mod json;
pub mod xml;

use crate::error::Result;
use crate::intern::Sym;
use crate::predict::FieldRange;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;

/// Zero-copy adapter trait.
///
/// Adapters handle:
/// 1. Formatting prompts for LM input
/// 2. Parsing responses to extract fields
pub trait Adapter: Send + Sync {
    /// Format a prompt for the LM.
    ///
    /// Writes to the provided buffer and returns a view into it.
    fn format<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        signature: &Signature<'_>,
        inputs: &Inputs<'_>,
        demos: &[DemoData<'_>],
    ) -> StrView<'a>;

    /// Parse a response from the LM.
    ///
    /// Returns field ranges into the response buffer.
    fn parse<'a>(
        &self,
        response: StrView<'a>,
        signature: &Signature<'_>,
    ) -> Result<SmallVec<[(Sym, FieldRange); 4]>>;

    /// Adapter name.
    fn name(&self) -> &'static str;
}

/// Demonstration data for prompt formatting.
#[derive(Clone, Copy)]
pub struct DemoData<'a> {
    /// Input fields as (symbol, value) pairs
    pub inputs: &'a [(Sym, StrView<'a>)],
    /// Output fields as (symbol, value) pairs
    pub outputs: &'a [(Sym, StrView<'a>)],
}

impl<'a> DemoData<'a> {
    /// Create new demo data.
    pub const fn new(inputs: &'a [(Sym, StrView<'a>)], outputs: &'a [(Sym, StrView<'a>)]) -> Self {
        Self { inputs, outputs }
    }

    /// Get input by symbol.
    pub fn get_input(&self, sym: Sym) -> Option<StrView<'a>> {
        self.inputs.iter().find(|(s, _)| *s == sym).map(|(_, v)| *v)
    }

    /// Get output by symbol.
    pub fn get_output(&self, sym: Sym) -> Option<StrView<'a>> {
        self.outputs
            .iter()
            .find(|(s, _)| *s == sym)
            .map(|(_, v)| *v)
    }
}

/// Format a field name for display.
#[inline]
pub fn format_field_name(buffer: &mut Vec<u8>, name: &str) {
    // Capitalize first letter
    let mut chars = name.chars();
    if let Some(first) = chars.next() {
        for c in first.to_uppercase() {
            buffer.push(c as u8);
        }
    }
    for c in chars {
        if c == '_' {
            buffer.push(b' ');
        } else {
            buffer.push(c as u8);
        }
    }
}

/// Capitalize first letter and replace underscores with spaces.
fn formatted_field_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    let mut chars = name.chars();
    if let Some(first) = chars.next() {
        for c in first.to_uppercase() {
            result.push(c);
        }
    }
    for c in chars {
        if c == '_' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

/// Find field value in response text.
///
/// Looks for patterns like "Field Name: value" or "[Field Name] value".
pub fn find_field_value(text: &str, field_name: &str) -> Option<std::ops::Range<usize>> {
    // Get capitalized version (e.g., "answer" -> "Answer")
    let formatted = formatted_field_name(field_name);

    // Try "Field Name: value" pattern with both versions
    let patterns = [
        format!("{}: ", formatted),
        format!("{}:", formatted),
        format!("[{}] ", formatted),
        format!("[{}]", formatted),
        format!("{}: ", field_name),
        format!("{}:", field_name),
        format!("[{}] ", field_name),
        format!("[{}]", field_name),
    ];

    for pattern in &patterns {
        if let Some(start) = text.find(pattern.as_str()) {
            let value_start = start + pattern.len();
            // Find end of value (next field or end of text)
            let value_end = find_value_end(text, value_start);
            if value_start < value_end {
                return Some(value_start..value_end);
            }
        }
    }

    None
}

/// Find the end of a field value.
fn find_value_end(text: &str, start: usize) -> usize {
    let remaining = &text[start..];

    // Look for next field marker or end
    let markers = ["\n\n", "\n[", "\nAnswer:", "\nQuestion:", "\nReasoning:"];

    let mut end = remaining.len();
    for marker in markers {
        if let Some(pos) = remaining.find(marker) {
            end = end.min(pos);
        }
    }

    // Trim trailing whitespace
    let value = &remaining[..end];
    start + value.trim_end().len()
}

// Re-exports
pub use chat::{ChatAdapter, ChatConfig};
pub use json::{JSONAdapter, JSONConfig};
pub use xml::{XMLAdapter, XMLConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_field_name() {
        let mut buffer = Vec::new();
        format_field_name(&mut buffer, "question");
        assert_eq!(String::from_utf8(buffer).unwrap(), "Question");

        let mut buffer = Vec::new();
        format_field_name(&mut buffer, "chain_of_thought");
        assert_eq!(String::from_utf8(buffer).unwrap(), "Chain of thought");
    }

    #[test]
    fn test_find_field_value() {
        let text = "Question: What is 2+2?\nAnswer: 4";

        let range = find_field_value(text, "Question");
        assert!(range.is_some());
        assert_eq!(&text[range.unwrap()], "What is 2+2?");

        let range = find_field_value(text, "Answer");
        assert!(range.is_some());
        assert_eq!(&text[range.unwrap()], "4");
    }

    #[test]
    fn test_demo_data() {
        use crate::intern::sym;

        let q_sym = sym("question");
        let a_sym = sym("answer");

        let inputs = [(q_sym, StrView::new("What is 2+2?"))];
        let outputs = [(a_sym, StrView::new("4"))];

        let demo = DemoData::new(&inputs, &outputs);

        assert_eq!(
            demo.get_input(q_sym).map(|v| v.as_str()),
            Some("What is 2+2?")
        );
        assert_eq!(demo.get_output(a_sym).map(|v| v.as_str()), Some("4"));
    }
}
