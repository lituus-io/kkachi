// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! JSON Adapter
//!
//! Formats prompts for JSON output and parses JSON responses.
//! Uses streaming parsing for efficiency.

use crate::adapter::{Adapter, DemoData};
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::predict::FieldRange;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;

/// JSON adapter configuration.
#[derive(Clone, Copy)]
pub struct JSONConfig {
    /// Pretty print JSON in prompts
    pub pretty: bool,
    /// Include schema in prompt
    pub include_schema: bool,
    /// Allow partial/streaming JSON
    pub allow_partial: bool,
}

impl Default for JSONConfig {
    fn default() -> Self {
        Self {
            pretty: true,
            include_schema: true,
            allow_partial: true,
        }
    }
}

impl JSONConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            pretty: true,
            include_schema: true,
            allow_partial: true,
        }
    }

    /// Set pretty printing.
    pub const fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Set schema inclusion.
    pub const fn with_schema(mut self, include: bool) -> Self {
        self.include_schema = include;
        self
    }

    /// Set partial JSON allowance.
    pub const fn with_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }
}

/// JSON adapter for LM prompts.
///
/// Formats prompts to request JSON output:
/// ```text
/// [Instructions]
///
/// Respond with JSON in this format:
/// {
///   "field1": "...",
///   "field2": "..."
/// }
///
/// Input:
/// {
///   "question": "..."
/// }
///
/// Output:
/// ```
#[derive(Clone, Copy)]
pub struct JSONAdapter {
    config: JSONConfig,
}

impl JSONAdapter {
    /// Create a new JSON adapter.
    pub const fn new(config: JSONConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(JSONConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &JSONConfig {
        &self.config
    }

    /// Write JSON schema for output fields.
    fn write_schema(&self, buffer: &mut Vec<u8>, signature: &Signature<'_>) {
        buffer.extend_from_slice(b"{\n");

        for (i, field) in signature.output_fields.iter().enumerate() {
            let name = &field.name;
            if self.config.pretty {
                buffer.extend_from_slice(b"  ");
            }
            buffer.push(b'"');
            buffer.extend_from_slice(name.as_bytes());
            buffer.extend_from_slice(b"\": ");

            // Type hint
            buffer.extend_from_slice(b"\"<");
            if !field.desc.is_empty() {
                buffer.extend_from_slice(field.desc.as_bytes());
            } else {
                buffer.extend_from_slice(b"string");
            }
            buffer.extend_from_slice(b">\"");

            if i < signature.output_fields.len() - 1 {
                buffer.push(b',');
            }
            buffer.push(b'\n');
        }

        buffer.push(b'}');
    }

    /// Write JSON object for inputs.
    fn write_input_json(
        &self,
        buffer: &mut Vec<u8>,
        inputs: &Inputs<'_>,
        signature: &Signature<'_>,
    ) {
        buffer.extend_from_slice(b"{\n");

        let mut first = true;
        for field in &signature.input_fields {
            let name = &field.name;
            if let Some(value) = inputs.get(name.as_ref()) {
                if !first {
                    buffer.extend_from_slice(b",\n");
                }
                first = false;

                if self.config.pretty {
                    buffer.extend_from_slice(b"  ");
                }
                buffer.push(b'"');
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b"\": \"");
                // Escape JSON string
                write_escaped_json_string(buffer, value);
                buffer.push(b'"');
            }
        }

        buffer.push(b'\n');
        buffer.push(b'}');
    }

    /// Write demo as JSON.
    fn write_demo_json(
        &self,
        buffer: &mut Vec<u8>,
        demo: &DemoData<'_>,
        signature: &Signature<'_>,
    ) {
        use crate::intern::sym;

        // Input
        buffer.extend_from_slice(b"Input:\n{\n");
        let mut first = true;
        for field in &signature.input_fields {
            let field_sym = sym(&field.name);
            if let Some(value) = demo.get_input(field_sym) {
                if !first {
                    buffer.extend_from_slice(b",\n");
                }
                first = false;

                if self.config.pretty {
                    buffer.extend_from_slice(b"  ");
                }
                buffer.push(b'"');
                buffer.extend_from_slice(field.name.as_bytes());
                buffer.extend_from_slice(b"\": \"");
                write_escaped_json_string(buffer, value.as_str());
                buffer.push(b'"');
            }
        }
        buffer.extend_from_slice(b"\n}\n\n");

        // Output
        buffer.extend_from_slice(b"Output:\n{\n");
        first = true;
        for field in &signature.output_fields {
            let field_sym = sym(&field.name);
            if let Some(value) = demo.get_output(field_sym) {
                if !first {
                    buffer.extend_from_slice(b",\n");
                }
                first = false;

                if self.config.pretty {
                    buffer.extend_from_slice(b"  ");
                }
                buffer.push(b'"');
                buffer.extend_from_slice(field.name.as_bytes());
                buffer.extend_from_slice(b"\": \"");
                write_escaped_json_string(buffer, value.as_str());
                buffer.push(b'"');
            }
        }
        buffer.extend_from_slice(b"\n}\n");
    }
}

/// Write an escaped JSON string value.
fn write_escaped_json_string(buffer: &mut Vec<u8>, s: &str) {
    for c in s.chars() {
        match c {
            '"' => buffer.extend_from_slice(b"\\\""),
            '\\' => buffer.extend_from_slice(b"\\\\"),
            '\n' => buffer.extend_from_slice(b"\\n"),
            '\r' => buffer.extend_from_slice(b"\\r"),
            '\t' => buffer.extend_from_slice(b"\\t"),
            c if c.is_control() => {
                buffer.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c => {
                let mut buf = [0u8; 4];
                buffer.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
}

/// Parse a JSON string value, handling escapes.
#[allow(dead_code)] // Reserved for future use in response parsing
fn parse_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('u') => {
                    // Unicode escape
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(code) {
                            result.push(c);
                        }
                    }
                }
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Find a JSON field value in text.
/// Returns the range of the value (without quotes for strings).
fn find_json_field(text: &str, field_name: &str) -> Option<std::ops::Range<usize>> {
    let pattern = format!("\"{}\"", field_name);
    let field_start = text.find(&pattern)?;
    let after_field = field_start + pattern.len();

    // Skip whitespace and colon
    let rest = &text[after_field..];
    let colon_pos = rest.find(':')?;
    let after_colon = after_field + colon_pos + 1;

    let rest = &text[after_colon..];
    let value_start = rest.find(|c: char| !c.is_whitespace())?;
    let value_start = after_colon + value_start;

    // Determine value type
    let rest = &text[value_start..];
    let first_char = rest.chars().next()?;

    match first_char {
        '"' => {
            // String value
            let content_start = value_start + 1;
            let mut i = 0;
            let mut escaped = false;
            for c in rest[1..].chars() {
                if escaped {
                    escaped = false;
                    i += c.len_utf8();
                } else if c == '\\' {
                    escaped = true;
                    i += 1;
                } else if c == '"' {
                    return Some(content_start..content_start + i);
                } else {
                    i += c.len_utf8();
                }
            }
            None
        }
        '{' | '[' => {
            // Object or array - find matching bracket
            let open = first_char;
            let close = if open == '{' { '}' } else { ']' };
            let mut depth = 1;
            let mut i = 1;
            let mut in_string = false;
            let mut escaped = false;

            for c in rest[1..].chars() {
                if escaped {
                    escaped = false;
                } else if c == '\\' && in_string {
                    escaped = true;
                } else if c == '"' {
                    in_string = !in_string;
                } else if !in_string {
                    if c == open {
                        depth += 1;
                    } else if c == close {
                        depth -= 1;
                        if depth == 0 {
                            return Some(value_start..value_start + i + 1);
                        }
                    }
                }
                i += c.len_utf8();
            }
            None
        }
        _ => {
            // Number, boolean, or null
            let end = rest
                .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
                .unwrap_or(rest.len());
            Some(value_start..value_start + end)
        }
    }
}

impl Adapter for JSONAdapter {
    fn format<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        signature: &Signature<'_>,
        inputs: &Inputs<'_>,
        demos: &[DemoData<'_>],
    ) -> StrView<'a> {
        buffer.clear();

        // Instructions
        if !signature.instructions.is_empty() {
            buffer.extend_from_slice(signature.instructions.as_bytes());
            buffer.extend_from_slice(b"\n\n");
        }

        // Schema
        if self.config.include_schema {
            buffer.extend_from_slice(b"Respond with JSON in this format:\n");
            self.write_schema(buffer, signature);
            buffer.extend_from_slice(b"\n\n");
        }

        // Demos
        for (i, demo) in demos.iter().enumerate() {
            if i > 0 {
                buffer.extend_from_slice(b"\n---\n\n");
            }
            self.write_demo_json(buffer, demo, signature);
        }

        // Separator
        if !demos.is_empty() {
            buffer.extend_from_slice(b"\n---\n\n");
        }

        // Input
        buffer.extend_from_slice(b"Input:\n");
        self.write_input_json(buffer, inputs, signature);
        buffer.extend_from_slice(b"\n\nOutput:\n");

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    fn parse<'a>(
        &self,
        response: StrView<'a>,
        signature: &Signature<'_>,
    ) -> Result<SmallVec<[(Sym, FieldRange); 4]>> {
        let text = response.as_str();
        let mut fields = SmallVec::new();

        for field in &signature.output_fields {
            let field_name = &field.name;
            if let Some(range) = find_json_field(text, field_name) {
                fields.push((
                    sym(field_name),
                    FieldRange::new(range.start as u32, range.end as u32),
                ));
            }
        }

        Ok(fields)
    }

    fn name(&self) -> &'static str {
        "JSON"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    #[test]
    fn test_json_adapter_creation() {
        let adapter = JSONAdapter::default();
        assert_eq!(adapter.name(), "JSON");
        assert!(adapter.config().pretty);
    }

    #[test]
    fn test_json_config() {
        let config = JSONConfig::new().with_pretty(false).with_schema(false);
        assert!(!config.pretty);
        assert!(!config.include_schema);
    }

    #[test]
    fn test_escape_json_string() {
        let mut buffer = Vec::new();
        write_escaped_json_string(&mut buffer, "Hello\nWorld");
        assert_eq!(String::from_utf8(buffer).unwrap(), "Hello\\nWorld");

        let mut buffer = Vec::new();
        write_escaped_json_string(&mut buffer, "Say \"Hi\"");
        assert_eq!(String::from_utf8(buffer).unwrap(), "Say \\\"Hi\\\"");
    }

    #[test]
    fn test_parse_json_string() {
        assert_eq!(parse_json_string("Hello\\nWorld"), "Hello\nWorld");
        assert_eq!(parse_json_string("Say \\\"Hi\\\""), "Say \"Hi\"");
        assert_eq!(parse_json_string("Tab\\there"), "Tab\there");
    }

    #[test]
    fn test_find_json_field_string() {
        let json = r#"{"answer": "42", "other": "foo"}"#;
        let range = find_json_field(json, "answer");
        assert!(range.is_some());
        assert_eq!(&json[range.unwrap()], "42");
    }

    #[test]
    fn test_find_json_field_number() {
        let json = r#"{"count": 42, "name": "test"}"#;
        let range = find_json_field(json, "count");
        assert!(range.is_some());
        assert_eq!(&json[range.unwrap()], "42");
    }

    #[test]
    fn test_find_json_field_object() {
        let json = r#"{"data": {"a": 1, "b": 2}, "name": "test"}"#;
        let range = find_json_field(json, "data");
        assert!(range.is_some());
        assert_eq!(&json[range.unwrap()], r#"{"a": 1, "b": 2}"#);
    }

    #[test]
    fn test_format_basic() {
        let adapter = JSONAdapter::new(JSONConfig::new().with_schema(false));
        let sig = Signature::parse("question -> answer").unwrap();

        let mut buffer = Vec::new();
        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);

        assert!(prompt.as_str().contains("\"question\": \"What is 2+2?\""));
        assert!(prompt.as_str().contains("Output:"));
    }

    #[test]
    fn test_parse_response() {
        let adapter = JSONAdapter::default();
        let sig = Signature::parse("question -> answer").unwrap();

        let response = StrView::new(r#"{"answer": "4"}"#);
        let fields = adapter.parse(response, &sig).unwrap();

        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, sym("answer"));

        let range = fields[0].1.as_range();
        assert_eq!(&response.as_str()[range], "4");
    }
}
