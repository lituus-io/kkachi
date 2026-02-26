// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Structured output enforcement via typed validation.
//!
//! Provides a [`TypedValidator`] that validates LLM outputs by attempting to
//! deserialize them into a target type `T`. Combined with format instructions,
//! this enables reliable structured output extraction from language models.
//!
//! # Example
//!
//! ```
//! use kkachi::recursive::typed::{typed, TypedValidator};
//! use kkachi::recursive::validate::Validate;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Answer {
//!     answer: String,
//!     confidence: f64,
//! }
//!
//! let validator = typed::<Answer>()
//!     .schema(r#"{"answer": "string", "confidence": "number"}"#);
//!
//! // Passes when output is valid JSON matching the type
//! let score = validator.validate(r#"{"answer": "Paris", "confidence": 0.95}"#);
//! assert!(score.is_perfect());
//!
//! // Fails with helpful feedback when output is invalid
//! let score = validator.validate("not json at all");
//! assert!(!score.is_perfect());
//! assert!(score.feedback_str().unwrap().contains("Invalid JSON"));
//! ```

use crate::error::{Error, Result};
use crate::recursive::validate::{Score, Validate};
use serde::de::DeserializeOwned;
use std::borrow::Cow;
use std::marker::PhantomData;

/// Trait for generating format instructions to include in prompts.
///
/// Implementations provide instructions that tell the LLM what format
/// to use for its output (e.g., "respond with valid JSON matching this schema").
pub trait FormatInstruction: Send + Sync {
    /// Generate a format instruction string.
    fn instruction(&self) -> Cow<'_, str>;
}

/// Default format instruction that requests plain JSON output.
#[derive(Debug, Clone, Copy)]
pub struct DefaultFormat;

impl FormatInstruction for DefaultFormat {
    fn instruction(&self) -> Cow<'_, str> {
        Cow::Borrowed("Respond with valid JSON only. No markdown, no explanation.")
    }
}

/// Format instruction that includes a JSON schema.
#[derive(Debug, Clone)]
pub struct SchemaFormat {
    schema: Cow<'static, str>,
}

impl SchemaFormat {
    /// Create a new schema format with a static schema string.
    pub fn new(schema: &'static str) -> Self {
        Self {
            schema: Cow::Borrowed(schema),
        }
    }

    /// Create a new schema format with an owned schema string.
    pub fn new_owned(schema: String) -> Self {
        Self {
            schema: Cow::Owned(schema),
        }
    }
}

impl FormatInstruction for SchemaFormat {
    fn instruction(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "Respond with valid JSON matching this schema:\n{}",
            self.schema
        ))
    }
}

/// A validator that enforces JSON-parseable output matching type `T`.
///
/// Combines deserialization validation with format instructions to guide
/// the LLM toward producing correctly structured output.
///
/// # Type Parameters
///
/// * `T` - The target type to deserialize into (must implement `DeserializeOwned`).
/// * `F` - The format instruction generator (defaults to [`DefaultFormat`]).
pub struct TypedValidator<T, F = DefaultFormat> {
    format: F,
    _phantom: PhantomData<fn() -> T>,
}

// Manual Send + Sync since PhantomData<fn() -> T> is always Send + Sync
unsafe impl<T, F: Send> Send for TypedValidator<T, F> {}
unsafe impl<T, F: Sync> Sync for TypedValidator<T, F> {}

/// Create a typed validator for type `T` with default format instructions.
///
/// # Example
///
/// ```
/// use kkachi::recursive::typed::typed;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Config {
///     name: String,
///     value: i32,
/// }
///
/// let validator = typed::<Config>();
/// ```
pub fn typed<T: DeserializeOwned>() -> TypedValidator<T, DefaultFormat> {
    TypedValidator {
        format: DefaultFormat,
        _phantom: PhantomData,
    }
}

impl<T: DeserializeOwned> TypedValidator<T, DefaultFormat> {
    /// Add a JSON schema for format instructions.
    ///
    /// This replaces the default format instruction with one that includes
    /// the schema, helping the LLM produce correctly structured output.
    pub fn schema(self, schema: &'static str) -> TypedValidator<T, SchemaFormat> {
        TypedValidator {
            format: SchemaFormat::new(schema),
            _phantom: PhantomData,
        }
    }

    /// Add a JSON schema from an owned string.
    pub fn schema_owned(self, schema: String) -> TypedValidator<T, SchemaFormat> {
        TypedValidator {
            format: SchemaFormat::new_owned(schema),
            _phantom: PhantomData,
        }
    }
}

impl<T: DeserializeOwned, F: FormatInstruction> TypedValidator<T, F> {
    /// Set a custom format instruction generator.
    pub fn with_format<F2: FormatInstruction>(self, format: F2) -> TypedValidator<T, F2> {
        TypedValidator {
            format,
            _phantom: PhantomData,
        }
    }

    /// Get the format instruction to include in prompts.
    ///
    /// Append this to your prompt so the LLM knows what format to use.
    pub fn instruction(&self) -> Cow<'_, str> {
        self.format.instruction()
    }
}

impl<T: DeserializeOwned + Send + Sync, F: FormatInstruction> Validate for TypedValidator<T, F> {
    fn validate(&self, text: &str) -> Score<'static> {
        let json_str = extract_json(text);
        match serde_json::from_str::<T>(json_str) {
            Ok(_) => Score::pass(),
            Err(e) => Score::with_feedback(
                0.0,
                format!("Invalid JSON: {}. {}", e, self.format.instruction()),
            ),
        }
    }

    fn name(&self) -> &'static str {
        "typed_validator"
    }
}

/// Extract JSON from text, handling markdown code fences.
///
/// If the text contains a JSON code block (```json ... ```), extracts the
/// content within. Otherwise, tries to find raw JSON by looking for
/// the first `{` or `[` and matching the last `}` or `]`.
pub fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // Try to extract from ```json ... ``` code block
    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + 7..];
        // Skip to the newline after ```json
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        if let Some(end) = content.find("```") {
            return content[..end].trim();
        }
    }

    // Try to extract from generic ``` ... ``` code block
    if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        // Skip to the newline after ```
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        if let Some(end) = content.find("```") {
            let extracted = content[..end].trim();
            // Only use if it looks like JSON
            if extracted.starts_with('{') || extracted.starts_with('[') {
                return extracted;
            }
        }
    }

    // Try to find raw JSON object or array
    let first_brace = trimmed.find('{');
    let first_bracket = trimmed.find('[');

    let json_start = match (first_brace, first_bracket) {
        (Some(b), Some(k)) => Some(b.min(k)),
        (Some(b), None) => Some(b),
        (None, Some(k)) => Some(k),
        (None, None) => None,
    };

    if let Some(start) = json_start {
        let open_char = trimmed.as_bytes()[start];
        let close_char = if open_char == b'{' { b'}' } else { b']' };

        // Find the matching close from the end
        if let Some(end) = trimmed.rfind(close_char as char) {
            if end > start {
                return &trimmed[start..=end];
            }
        }
    }

    // Fall back to the full text
    trimmed
}

/// Parse a JSON string into type `T`, extracting JSON from markdown if needed.
///
/// This is a convenience function for parsing LLM output that may contain
/// JSON within markdown code fences or surrounding text.
///
/// # Example
///
/// ```
/// use kkachi::recursive::typed::parse_output;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Answer { value: i32 }
///
/// let result: Answer = parse_output(r#"```json
/// {"value": 42}
/// ```"#).unwrap();
/// assert_eq!(result.value, 42);
/// ```
pub fn parse_output<T: DeserializeOwned>(text: &str) -> Result<T> {
    let json_str = extract_json(text);
    serde_json::from_str(json_str)
        .map_err(|e| Error::module(format!("Failed to parse output as JSON: {}", e)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, Debug, PartialEq)]
    struct SimpleStruct {
        name: String,
        value: i32,
    }

    #[derive(Deserialize, Debug)]
    #[allow(dead_code)]
    struct NestedStruct {
        items: Vec<String>,
        count: usize,
    }

    #[test]
    fn test_typed_validator_valid_json() {
        let v = typed::<SimpleStruct>();
        let score = v.validate(r#"{"name": "test", "value": 42}"#);
        assert!(score.is_perfect());
    }

    #[test]
    fn test_typed_validator_invalid_json() {
        let v = typed::<SimpleStruct>();
        let score = v.validate("not json at all");
        assert!((score.value - 0.0).abs() < f64::EPSILON);
        assert!(score.feedback_str().unwrap().contains("Invalid JSON"));
    }

    #[test]
    fn test_typed_validator_wrong_schema() {
        let v = typed::<SimpleStruct>();
        // Valid JSON but wrong structure (missing required fields)
        let score = v.validate(r#"{"wrong_field": true}"#);
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_typed_validator_with_schema() {
        let v = typed::<SimpleStruct>().schema(r#"{"name": "string", "value": "number"}"#);
        let score = v.validate(r#"{"name": "hello", "value": 1}"#);
        assert!(score.is_perfect());

        let score = v.validate("bad");
        let feedback = score.feedback_str().unwrap();
        assert!(feedback.contains("schema"));
    }

    #[test]
    fn test_extract_json_code_fence() {
        let text = r#"Here is the JSON:
```json
{"name": "test", "value": 42}
```
That's the answer."#;
        let extracted = extract_json(text);
        assert_eq!(extracted, r#"{"name": "test", "value": 42}"#);
    }

    #[test]
    fn test_extract_json_generic_fence() {
        let text = r#"```
{"items": ["a", "b"], "count": 2}
```"#;
        let extracted = extract_json(text);
        assert_eq!(extracted, r#"{"items": ["a", "b"], "count": 2}"#);
    }

    #[test]
    fn test_extract_json_raw() {
        let text = r#"The answer is {"name": "raw", "value": 99} and that's it."#;
        let extracted = extract_json(text);
        assert_eq!(extracted, r#"{"name": "raw", "value": 99}"#);
    }

    #[test]
    fn test_extract_json_array() {
        let text = r#"[1, 2, 3]"#;
        let extracted = extract_json(text);
        assert_eq!(extracted, "[1, 2, 3]");
    }

    #[test]
    fn test_extract_json_no_json() {
        let text = "just plain text";
        let extracted = extract_json(text);
        assert_eq!(extracted, "just plain text");
    }

    #[test]
    fn test_parse_output_from_fence() {
        let text = r#"```json
{"name": "parsed", "value": 7}
```"#;
        let result: SimpleStruct = parse_output(text).unwrap();
        assert_eq!(result.name, "parsed");
        assert_eq!(result.value, 7);
    }

    #[test]
    fn test_parse_output_raw_json() {
        let result: SimpleStruct = parse_output(r#"{"name": "raw", "value": 1}"#).unwrap();
        assert_eq!(result.name, "raw");
        assert_eq!(result.value, 1);
    }

    #[test]
    fn test_parse_output_invalid() {
        let result = parse_output::<SimpleStruct>("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_typed_nested() {
        let v = typed::<NestedStruct>();
        let score = v.validate(r#"{"items": ["hello", "world"], "count": 2}"#);
        assert!(score.is_perfect());
    }

    #[test]
    fn test_format_instruction_default() {
        let v = typed::<SimpleStruct>();
        let inst = v.instruction();
        assert!(inst.contains("JSON"));
    }

    #[test]
    fn test_format_instruction_schema() {
        let v = typed::<SimpleStruct>().schema(r#"{"name": "string"}"#);
        let inst = v.instruction();
        assert!(inst.contains("schema"));
        assert!(inst.contains(r#""name""#));
    }

    #[test]
    fn test_typed_with_surrounding_text() {
        let v = typed::<SimpleStruct>();
        let text = r#"Sure! Here is the answer:

{"name": "extracted", "value": 100}

Hope this helps!"#;
        let score = v.validate(text);
        assert!(score.is_perfect());
    }

    #[test]
    fn test_schema_owned() {
        let schema = format!(r#"{{"name": "{}"}}"#, "string");
        let v = typed::<SimpleStruct>().schema_owned(schema);
        let inst = v.instruction();
        assert!(inst.contains("name"));
    }
}
