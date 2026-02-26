// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Typed adapter trait for formatting and parsing typed signatures.
//!
//! Bridges the existing [`Adapter`] system with the new [`TypedSignature`] system,
//! enabling automatic prompt formatting and response parsing based on field types.

use crate::error::Result;
use crate::intern::{resolve, Sym};
use crate::predict::FieldRange;
use crate::str_view::StrView;
use crate::typed_sig::{ParsedOutput, TypedDemo, TypedSignature, ValueKind};
use smallvec::SmallVec;

/// Trait for adapters that understand typed signatures.
///
/// Extends the existing adapter system to work with [`TypedSignature`]
/// for automatic type-aware formatting and parsing.
pub trait TypedAdapter: Send + Sync {
    /// Format a prompt from a typed signature and inputs.
    ///
    /// Writes to the provided buffer and returns a view into it.
    fn format_typed<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        sig: &TypedSignature,
        inputs: &[(Sym, StrView<'_>)],
        demos: &[TypedDemo<'_>],
    ) -> StrView<'a>;

    /// Parse a response using the typed signature.
    ///
    /// Returns a zero-copy parsed output with field ranges.
    fn parse_typed<'a>(
        &self,
        response: StrView<'a>,
        sig: &TypedSignature,
    ) -> Result<ParsedOutput<'a>>;

    /// Adapter name.
    fn name(&self) -> &'static str;
}

/// Chat-style typed adapter.
///
/// Formats prompts as instruction + field labels and parses
/// responses by looking for field markers.
pub struct ChatTypedAdapter;

impl TypedAdapter for ChatTypedAdapter {
    fn format_typed<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        sig: &TypedSignature,
        inputs: &[(Sym, StrView<'_>)],
        demos: &[TypedDemo<'_>],
    ) -> StrView<'a> {
        buffer.clear();

        // Add instruction
        buffer.extend_from_slice(sig.instruction().as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Add demos if present
        for demo in demos {
            buffer.extend_from_slice(b"---\n");
            for (sym, value) in demo.inputs.iter() {
                let name = resolve(*sym);
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b": ");
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
            for (sym, value) in demo.outputs.iter() {
                let name = resolve(*sym);
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b": ");
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
        }

        // Add input fields
        if !demos.is_empty() {
            buffer.extend_from_slice(b"---\n");
        }

        for &(sym, view) in inputs {
            let name = resolve(sym);
            buffer.extend_from_slice(name.as_bytes());
            buffer.extend_from_slice(b": ");
            buffer.extend_from_slice(view.as_str().as_bytes());
            buffer.push(b'\n');
        }

        // Add output field markers
        buffer.push(b'\n');
        for field in sig.outputs() {
            let name = resolve(field.name);
            buffer.extend_from_slice(name.as_bytes());
            buffer.extend_from_slice(b": ");
            // Add type hint
            let hint = match field.kind {
                ValueKind::Int => "(integer) ",
                ValueKind::Float => "(number) ",
                ValueKind::Bool => "(true/false) ",
                ValueKind::JsonObject => "(JSON object) ",
                ValueKind::JsonArray => "(JSON array) ",
                ValueKind::Enum => "(one of the options) ",
                ValueKind::Str => "",
            };
            buffer.extend_from_slice(hint.as_bytes());
            buffer.push(b'\n');
        }

        // Safety: we only wrote valid UTF-8
        let s = std::str::from_utf8(buffer).expect("buffer contains valid UTF-8");
        StrView::new(s)
    }

    fn parse_typed<'a>(
        &self,
        response: StrView<'a>,
        sig: &TypedSignature,
    ) -> Result<ParsedOutput<'a>> {
        let text = response.as_str();
        let mut fields = SmallVec::new();

        for field in sig.outputs() {
            let name = resolve(field.name);
            // Look for "field_name: value" pattern
            if let Some(range) = find_typed_field(text, name) {
                fields.push((field.name, FieldRange::new(range.start as u32, range.end as u32)));
            }
        }

        Ok(ParsedOutput::with_fields(response, fields))
    }

    fn name(&self) -> &'static str {
        "chat_typed"
    }
}

/// JSON-style typed adapter.
///
/// Formats prompts requesting JSON output and parses JSON responses.
pub struct JsonTypedAdapter;

impl TypedAdapter for JsonTypedAdapter {
    fn format_typed<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        sig: &TypedSignature,
        inputs: &[(Sym, StrView<'_>)],
        demos: &[TypedDemo<'_>],
    ) -> StrView<'a> {
        buffer.clear();

        // Add instruction
        buffer.extend_from_slice(sig.instruction().as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Add demos
        for demo in demos {
            buffer.extend_from_slice(b"Example:\n");
            for (sym, value) in demo.inputs.iter() {
                let name = resolve(*sym);
                buffer.extend_from_slice(b"  ");
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b": ");
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
            buffer.extend_from_slice(b"  Output: {");
            for (i, (sym, value)) in demo.outputs.iter().enumerate() {
                if i > 0 {
                    buffer.extend_from_slice(b", ");
                }
                let name = resolve(*sym);
                buffer.push(b'"');
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b"\": \"");
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'"');
            }
            buffer.extend_from_slice(b"}\n\n");
        }

        // Add input
        for &(sym, view) in inputs {
            let name = resolve(sym);
            buffer.extend_from_slice(name.as_bytes());
            buffer.extend_from_slice(b": ");
            buffer.extend_from_slice(view.as_str().as_bytes());
            buffer.push(b'\n');
        }

        // Request JSON output
        buffer.extend_from_slice(b"\nRespond with a JSON object containing: {");
        for (i, field) in sig.outputs().iter().enumerate() {
            if i > 0 {
                buffer.extend_from_slice(b", ");
            }
            let name = resolve(field.name);
            buffer.push(b'"');
            buffer.extend_from_slice(name.as_bytes());
            buffer.extend_from_slice(b"\": ");
            let type_str = match field.kind {
                ValueKind::Str => "\"string\"",
                ValueKind::Int => "integer",
                ValueKind::Float => "number",
                ValueKind::Bool => "boolean",
                ValueKind::JsonObject => "object",
                ValueKind::JsonArray => "array",
                ValueKind::Enum => "\"string\"",
            };
            buffer.extend_from_slice(type_str.as_bytes());
        }
        buffer.extend_from_slice(b"}\n");

        let s = std::str::from_utf8(buffer).expect("buffer contains valid UTF-8");
        StrView::new(s)
    }

    fn parse_typed<'a>(
        &self,
        response: StrView<'a>,
        sig: &TypedSignature,
    ) -> Result<ParsedOutput<'a>> {
        let text = response.as_str();
        let mut fields = SmallVec::new();

        // Find JSON object in response
        let json_start = text.find('{');
        let json_end = text.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_text = &text[start..=end];

            for field in sig.outputs() {
                let name = resolve(field.name);
                // Simple JSON field extraction: "field_name": "value" or "field_name": value
                let pattern = format!("\"{}\"", name);
                if let Some(key_pos) = json_text.find(&pattern) {
                    let after_key = key_pos + pattern.len();
                    // Skip colon and whitespace
                    let value_start_in_json =
                        json_text[after_key..].find(|c: char| c != ':' && c != ' ');
                    if let Some(vs) = value_start_in_json {
                        let abs_value_start = start + after_key + vs;
                        let remaining = &text[abs_value_start..];

                        let value_end = if remaining.starts_with('"') {
                            // String value: find closing quote
                            remaining[1..]
                                .find('"')
                                .map(|p| abs_value_start + 1 + p)
                                .unwrap_or(abs_value_start + remaining.len())
                        } else {
                            // Non-string value: find comma, }, or newline
                            remaining
                                .find(|c: char| c == ',' || c == '}' || c == '\n')
                                .map(|p| abs_value_start + p)
                                .unwrap_or(abs_value_start + remaining.len())
                        };

                        let field_start = if text[abs_value_start..].starts_with('"') {
                            abs_value_start + 1
                        } else {
                            abs_value_start
                        };

                        if field_start < value_end {
                            fields.push((
                                field.name,
                                FieldRange::new(field_start as u32, value_end as u32),
                            ));
                        }
                    }
                }
            }
        }

        Ok(ParsedOutput::with_fields(response, fields))
    }

    fn name(&self) -> &'static str {
        "json_typed"
    }
}

/// Find a typed field value in text (chat-style "field: value" format).
fn find_typed_field(text: &str, field_name: &str) -> Option<std::ops::Range<usize>> {
    // Try various patterns
    let patterns = [
        format!("{}: ", field_name),
        format!("{}:", field_name),
    ];

    for pattern in &patterns {
        if let Some(start) = text.find(pattern.as_str()) {
            let value_start = start + pattern.len();
            // Value ends at next newline containing a colon (next field) or end of text
            let remaining = &text[value_start..];
            let value_len = remaining
                .find('\n')
                .and_then(|nl| {
                    // Check if next line looks like another field
                    let after_nl = &remaining[nl + 1..];
                    if after_nl.contains(':') || after_nl.is_empty() {
                        Some(nl)
                    } else {
                        // Multi-line value: find next field marker
                        after_nl
                            .find('\n')
                            .map(|nl2| nl + 1 + nl2)
                            .or(Some(remaining.len()))
                    }
                })
                .unwrap_or(remaining.len());

            let trimmed = remaining[..value_len].trim_end();
            if !trimmed.is_empty() {
                return Some(value_start..value_start + trimmed.len());
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    fn make_qa_sig() -> TypedSignature {
        TypedSignature::new("Answer questions.")
            .input(sym("question"), ValueKind::Str)
            .output(sym("answer"), ValueKind::Str)
    }

    #[test]
    fn test_chat_format() {
        let sig = make_qa_sig();
        let mut buffer = Vec::new();
        let q_sym = sym("question");

        let inputs = [(q_sym, StrView::new("What is 2+2?"))];
        let result = ChatTypedAdapter.format_typed(&mut buffer, &sig, &inputs, &[]);

        let text = result.as_str();
        assert!(text.contains("Answer questions."));
        assert!(text.contains("question: What is 2+2?"));
        assert!(text.contains("answer:"));
    }

    #[test]
    fn test_chat_parse() {
        let sig = make_qa_sig();
        let response = StrView::new("answer: 4");
        let parsed = ChatTypedAdapter.parse_typed(response, &sig).unwrap();

        let answer_sym = sym("answer");
        assert_eq!(parsed.get_str(answer_sym), Some("4"));
    }

    #[test]
    fn test_json_format() {
        let sig = make_qa_sig();
        let mut buffer = Vec::new();
        let q_sym = sym("question");

        let inputs = [(q_sym, StrView::new("What is 2+2?"))];
        let result = JsonTypedAdapter.format_typed(&mut buffer, &sig, &inputs, &[]);

        let text = result.as_str();
        assert!(text.contains("Answer questions."));
        assert!(text.contains("question: What is 2+2?"));
        assert!(text.contains("\"answer\""));
    }

    #[test]
    fn test_json_parse() {
        let sig = make_qa_sig();
        let response = StrView::new("Here is the answer: {\"answer\": \"4\"}");
        let parsed = JsonTypedAdapter.parse_typed(response, &sig).unwrap();

        let answer_sym = sym("answer");
        assert_eq!(parsed.get_str(answer_sym), Some("4"));
    }

    #[test]
    fn test_json_parse_numeric() {
        let sig = TypedSignature::new("Calculate.")
            .input(sym("expression"), ValueKind::Str)
            .output(sym("result"), ValueKind::Int);

        let response = StrView::new("{\"result\": 42}");
        let parsed = JsonTypedAdapter.parse_typed(response, &sig).unwrap();

        let result_sym = sym("result");
        assert_eq!(parsed.get_str(result_sym), Some("42"));
    }
}
