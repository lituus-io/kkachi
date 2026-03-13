// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Typed Signature System for statically-declared, const-constructible signatures.
//!
//! This module provides a lightweight, stack-allocated alternative to [`crate::signature::Signature`]
//! for cases where the signature shape is known at compile time. All types are `Copy` and
//! const-constructible, enabling zero-cost signature declarations as `const` items.
//!
//! # Design Principles
//!
//! - **Const-constructible**: [`TypedSignature`] can be declared as `const` with builder methods.
//! - **Zero-copy parsing**: [`ParsedOutput`] borrows from the raw LLM response via [`StrView`].
//! - **Type-safe access**: [`ValueKind`] annotations enable typed getters (`get_int`, `get_float`, etc.).
//! - **Stack-allocated**: Fixed-size arrays (max 4 inputs, 4 outputs) avoid heap allocation.
//!
//! # Example
//!
//! ```
//! use kkachi::intern::{QUESTION, ANSWER};
//! use kkachi::typed_sig::{TypedSignature, ValueKind};
//!
//! const QA: TypedSignature = TypedSignature::new("Answer questions.")
//!     .input(QUESTION, ValueKind::Str)
//!     .output(ANSWER, ValueKind::Str);
//!
//! assert_eq!(QA.instruction(), "Answer questions.");
//! assert_eq!(QA.input_count(), 1);
//! assert_eq!(QA.output_count(), 1);
//! ```

use crate::intern::Sym;
use crate::predict::FieldRange;
use crate::recursive::validate::{Score, Validate};
use crate::str_view::StrView;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// ValueKind
// ---------------------------------------------------------------------------

/// The expected value type for a typed field.
///
/// Used to declare the kind of data a field carries, enabling typed accessors
/// on [`ParsedOutput`] and validation via [`TypedFieldValidator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueKind {
    /// Free-form string.
    Str = 0,
    /// Integer (parseable via `i64::from_str`).
    Int = 1,
    /// Floating-point number (parseable via `f64::from_str`).
    Float = 2,
    /// Boolean (`true` / `false`, case-insensitive).
    Bool = 3,
    /// JSON object (starts with `{`).
    JsonObject = 4,
    /// JSON array (starts with `[`).
    JsonArray = 5,
    /// One of a fixed set of string values (validated externally).
    Enum = 6,
}

impl ValueKind {
    /// Return a human-readable label for this kind.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Str => "string",
            Self::Int => "integer",
            Self::Float => "float",
            Self::Bool => "boolean",
            Self::JsonObject => "JSON object",
            Self::JsonArray => "JSON array",
            Self::Enum => "enum",
        }
    }

    /// Check whether `text` is a valid representation of this kind.
    pub fn matches(self, text: &str) -> bool {
        let trimmed = text.trim();
        match self {
            Self::Str => true,
            Self::Int => trimmed.parse::<i64>().is_ok(),
            Self::Float => trimmed.parse::<f64>().is_ok(),
            Self::Bool => matches!(
                trimmed.to_ascii_lowercase().as_str(),
                "true" | "false" | "yes" | "no" | "1" | "0"
            ),
            Self::JsonObject => trimmed.starts_with('{') && trimmed.ends_with('}'),
            Self::JsonArray => trimmed.starts_with('[') && trimmed.ends_with(']'),
            Self::Enum => !trimmed.is_empty(),
        }
    }
}

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

/// Whether a field is an input or an output of the signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    /// Supplied by the caller.
    Input = 0,
    /// Produced by the LLM.
    Output = 1,
}

// ---------------------------------------------------------------------------
// TypedField
// ---------------------------------------------------------------------------

/// A single field declaration within a [`TypedSignature`].
///
/// Exactly 8 bytes: `Sym` (4) + `ValueKind` (1) + `required` (1) + `Direction` (1) + padding (1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedField {
    /// Interned field name.
    pub name: Sym,
    /// Expected value type.
    pub kind: ValueKind,
    /// Whether the field must be present.
    pub required: bool,
    /// Input or output.
    pub direction: Direction,
}

impl TypedField {
    /// Create a new typed field (const-compatible).
    #[inline]
    pub const fn new(name: Sym, kind: ValueKind, direction: Direction) -> Self {
        Self {
            name,
            kind,
            required: true,
            direction,
        }
    }

    /// Create an empty / placeholder field for array initialization.
    const fn empty() -> Self {
        Self {
            name: Sym::EMPTY,
            kind: ValueKind::Str,
            required: false,
            direction: Direction::Input,
        }
    }

    /// Return a copy with `required` set to `false`.
    #[inline]
    pub const fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

// ---------------------------------------------------------------------------
// TypedSignature
// ---------------------------------------------------------------------------

/// Maximum number of input (or output) fields in a single signature.
const MAX_FIELDS: usize = 4;

/// A const-constructible, entirely stack-allocated signature.
///
/// Stores up to [`MAX_FIELDS`] input fields and [`MAX_FIELDS`] output fields
/// in fixed-size arrays. The `input_count` / `output_count` bytes track how
/// many slots are occupied.
///
/// # Const Construction
///
/// ```
/// use kkachi::intern::{QUESTION, ANSWER};
/// use kkachi::typed_sig::{TypedSignature, ValueKind};
///
/// const QA: TypedSignature = TypedSignature::new("Answer questions.")
///     .input(QUESTION, ValueKind::Str)
///     .output(ANSWER, ValueKind::Str);
/// ```
///
/// Note: For `const` contexts you must use pre-interned symbols (e.g. `QUESTION`,
/// `ANSWER`). The runtime [`crate::intern::sym`] function works in non-const contexts.
#[derive(Debug, Clone, Copy)]
pub struct TypedSignature {
    instruction: &'static str,
    inputs: [TypedField; MAX_FIELDS],
    input_count: u8,
    outputs: [TypedField; MAX_FIELDS],
    output_count: u8,
}

impl TypedSignature {
    /// Create a new typed signature with the given instruction.
    #[inline]
    pub const fn new(instruction: &'static str) -> Self {
        Self {
            instruction,
            inputs: [TypedField::empty(); MAX_FIELDS],
            input_count: 0,
            outputs: [TypedField::empty(); MAX_FIELDS],
            output_count: 0,
        }
    }

    /// Add a required input field (const builder).
    ///
    /// # Panics
    ///
    /// Panics at compile time (const evaluation) if more than [`MAX_FIELDS`] inputs are added.
    #[inline]
    pub const fn input(mut self, name: Sym, kind: ValueKind) -> Self {
        assert!(
            (self.input_count as usize) < MAX_FIELDS,
            "TypedSignature: exceeded maximum number of input fields"
        );
        self.inputs[self.input_count as usize] = TypedField::new(name, kind, Direction::Input);
        self.input_count += 1;
        self
    }

    /// Add an optional input field (const builder).
    #[inline]
    pub const fn input_optional(mut self, name: Sym, kind: ValueKind) -> Self {
        assert!(
            (self.input_count as usize) < MAX_FIELDS,
            "TypedSignature: exceeded maximum number of input fields"
        );
        let mut field = TypedField::new(name, kind, Direction::Input);
        field.required = false;
        self.inputs[self.input_count as usize] = field;
        self.input_count += 1;
        self
    }

    /// Add a required output field (const builder).
    ///
    /// # Panics
    ///
    /// Panics at compile time (const evaluation) if more than [`MAX_FIELDS`] outputs are added.
    #[inline]
    pub const fn output(mut self, name: Sym, kind: ValueKind) -> Self {
        assert!(
            (self.output_count as usize) < MAX_FIELDS,
            "TypedSignature: exceeded maximum number of output fields"
        );
        self.outputs[self.output_count as usize] = TypedField::new(name, kind, Direction::Output);
        self.output_count += 1;
        self
    }

    /// Add an optional output field (const builder).
    #[inline]
    pub const fn output_optional(mut self, name: Sym, kind: ValueKind) -> Self {
        assert!(
            (self.output_count as usize) < MAX_FIELDS,
            "TypedSignature: exceeded maximum number of output fields"
        );
        let mut field = TypedField::new(name, kind, Direction::Output);
        field.required = false;
        self.outputs[self.output_count as usize] = field;
        self.output_count += 1;
        self
    }

    // -- Accessors ----------------------------------------------------------

    /// Get the instruction text.
    #[inline]
    pub const fn instruction(&self) -> &'static str {
        self.instruction
    }

    /// Number of declared input fields.
    #[inline]
    pub const fn input_count(&self) -> usize {
        self.input_count as usize
    }

    /// Number of declared output fields.
    #[inline]
    pub const fn output_count(&self) -> usize {
        self.output_count as usize
    }

    /// Slice of active input fields.
    #[inline]
    pub fn inputs(&self) -> &[TypedField] {
        &self.inputs[..self.input_count as usize]
    }

    /// Slice of active output fields.
    #[inline]
    pub fn outputs(&self) -> &[TypedField] {
        &self.outputs[..self.output_count as usize]
    }

    /// Look up an input field by name.
    pub fn get_input(&self, name: Sym) -> Option<&TypedField> {
        self.inputs().iter().find(|f| f.name == name)
    }

    /// Look up an output field by name.
    pub fn get_output(&self, name: Sym) -> Option<&TypedField> {
        self.outputs().iter().find(|f| f.name == name)
    }

    /// Create a [`TypedFieldValidator`] from this signature.
    pub fn validator(&self) -> TypedFieldValidator {
        TypedFieldValidator { sig: *self }
    }
}

// ---------------------------------------------------------------------------
// ParsedOutput
// ---------------------------------------------------------------------------

/// Zero-copy parsed output from an LLM response.
///
/// Holds a [`StrView`] into the raw response together with [`FieldRange`] entries
/// that delimit individual field values. All accessor methods borrow from the
/// original response buffer without copying.
#[derive(Debug, Clone)]
pub struct ParsedOutput<'a> {
    /// The full raw output text.
    raw: StrView<'a>,
    /// Parsed (field-name, byte-range) pairs.
    fields: SmallVec<[(Sym, FieldRange); 4]>,
}

impl<'a> ParsedOutput<'a> {
    /// Create an empty parsed output wrapping `raw`.
    pub fn new(raw: StrView<'a>) -> Self {
        Self {
            raw,
            fields: SmallVec::new(),
        }
    }

    /// Create a parsed output with pre-computed field ranges.
    pub fn with_fields(raw: StrView<'a>, fields: SmallVec<[(Sym, FieldRange); 4]>) -> Self {
        Self { raw, fields }
    }

    /// Add a field range.
    pub fn push(&mut self, name: Sym, range: FieldRange) {
        self.fields.push((name, range));
    }

    /// The full raw response text.
    #[inline]
    pub fn raw(&self) -> StrView<'a> {
        self.raw
    }

    /// Iterate over `(Sym, FieldRange)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(Sym, FieldRange)> {
        self.fields.iter()
    }

    /// Number of parsed fields.
    #[inline]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    // -- Typed getters ------------------------------------------------------

    /// Get the raw string slice for a field (zero-copy).
    pub fn get_raw(&self, name: Sym) -> Option<&'a str> {
        for (sym, fr) in &self.fields {
            if *sym == name {
                return self.raw.try_slice(fr.as_range()).map(|v| v.as_str());
            }
        }
        None
    }

    /// Get a field value as a string slice.
    ///
    /// Equivalent to [`get_raw`](Self::get_raw) but trims surrounding whitespace.
    pub fn get_str(&self, name: Sym) -> Option<&'a str> {
        self.get_raw(name).map(|s| s.trim())
    }

    /// Get a field value parsed as `i64`.
    pub fn get_int(&self, name: Sym) -> Option<i64> {
        self.get_str(name).and_then(|s| s.parse().ok())
    }

    /// Get a field value parsed as `f64`.
    pub fn get_float(&self, name: Sym) -> Option<f64> {
        self.get_str(name).and_then(|s| s.parse().ok())
    }

    /// Get a field value parsed as `bool`.
    ///
    /// Recognises `true`, `false`, `yes`, `no`, `1`, `0` (case-insensitive).
    pub fn get_bool(&self, name: Sym) -> Option<bool> {
        self.get_str(name)
            .and_then(|s| match s.to_ascii_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" => Some(false),
                _ => None,
            })
    }

    /// Convenience: parse a simple `"Field: value\n"` format into this output.
    ///
    /// For each output field in `sig`, looks for `"FieldName: "` (capitalised) in the
    /// raw text and records the byte range of the value up to the next newline.
    pub fn parse_from_sig(raw: StrView<'a>, sig: &TypedSignature) -> Self {
        let text = raw.as_str();
        let mut out = Self::new(raw);

        for field in sig.outputs() {
            let field_name = field.name.as_str();
            // Try "FieldName:" pattern (capitalised first letter)
            let mut prefix = String::with_capacity(field_name.len() + 2);
            let mut chars = field_name.chars();
            if let Some(first) = chars.next() {
                prefix.push(first.to_ascii_uppercase());
                prefix.extend(chars);
            }
            prefix.push(':');

            if let Some(idx) = text.find(&prefix) {
                let mut start = idx + prefix.len();
                // Skip whitespace after colon
                while start < text.len() && text.as_bytes()[start] == b' ' {
                    start += 1;
                }
                let end = text[start..]
                    .find('\n')
                    .map(|i| start + i)
                    .unwrap_or(text.len());

                out.push(field.name, FieldRange::new(start as u32, end as u32));
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// TypedFieldValidator
// ---------------------------------------------------------------------------

/// Validates that a parsed LLM output conforms to a [`TypedSignature`].
///
/// For each required output field in the signature, the validator checks:
/// 1. The field is present in the text (matched by capitalised name prefix).
/// 2. The field value matches the declared [`ValueKind`].
///
/// The overall score is the fraction of required output fields that pass.
///
/// # Example
///
/// ```
/// use kkachi::intern::{QUESTION, ANSWER, SCORE};
/// use kkachi::typed_sig::{TypedSignature, ValueKind};
/// use kkachi::recursive::validate::Validate;
///
/// const SIG: TypedSignature = TypedSignature::new("Score an answer.")
///     .input(QUESTION, ValueKind::Str)
///     .output(ANSWER, ValueKind::Str)
///     .output(SCORE, ValueKind::Float);
///
/// let validator = SIG.validator();
///
/// // Both fields present and correctly typed
/// assert!(validator.validate("Answer: Paris\nScore: 0.95").is_perfect());
///
/// // Missing score field
/// let s = validator.validate("Answer: Paris");
/// assert!(s.value < 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TypedFieldValidator {
    sig: TypedSignature,
}

impl TypedFieldValidator {
    /// Create a validator for the given signature.
    pub const fn new(sig: TypedSignature) -> Self {
        Self { sig }
    }

    /// Get a reference to the underlying signature.
    pub const fn signature(&self) -> &TypedSignature {
        &self.sig
    }
}

impl Validate for TypedFieldValidator {
    fn validate(&self, text: &str) -> Score<'static> {
        let view = StrView::new(text);
        let parsed = ParsedOutput::parse_from_sig(view, &self.sig);

        let outputs = self.sig.outputs();
        if outputs.is_empty() {
            return Score::pass();
        }

        let mut passed = 0usize;
        let mut total_required = 0usize;
        let mut feedback_parts: SmallVec<[String; 4]> = SmallVec::new();

        for field in outputs {
            if !field.required {
                continue;
            }
            total_required += 1;

            match parsed.get_raw(field.name) {
                Some(raw_value) => {
                    if field.kind.matches(raw_value) {
                        passed += 1;
                    } else {
                        feedback_parts.push(format!(
                            "Field '{}': expected {}, got {:?}",
                            field.name.as_str(),
                            field.kind.label(),
                            raw_value.trim(),
                        ));
                    }
                }
                None => {
                    feedback_parts
                        .push(format!("Missing required field '{}'", field.name.as_str(),));
                }
            }
        }

        if total_required == 0 {
            return Score::pass();
        }

        let value = passed as f64 / total_required as f64;
        if feedback_parts.is_empty() {
            Score::pass()
        } else {
            Score::with_feedback(value, feedback_parts.join("; "))
        }
    }

    fn name(&self) -> &'static str {
        "typed_field_validator"
    }
}

// ---------------------------------------------------------------------------
// TypedDemo
// ---------------------------------------------------------------------------

/// A typed demonstration (few-shot example) for a [`TypedSignature`].
///
/// Stores input and output field values as `(Sym, &str)` pairs. This is
/// intended for constructing few-shot prompts where both the question
/// and the expected answer are known ahead of time.
#[derive(Debug, Clone)]
pub struct TypedDemo<'a> {
    /// Input field values.
    pub inputs: SmallVec<[(Sym, &'a str); 4]>,
    /// Output field values.
    pub outputs: SmallVec<[(Sym, &'a str); 4]>,
}

impl<'a> TypedDemo<'a> {
    /// Create an empty demonstration.
    pub fn new() -> Self {
        Self {
            inputs: SmallVec::new(),
            outputs: SmallVec::new(),
        }
    }

    /// Add an input field value.
    pub fn input(mut self, name: Sym, value: &'a str) -> Self {
        self.inputs.push((name, value));
        self
    }

    /// Add an output field value.
    pub fn output(mut self, name: Sym, value: &'a str) -> Self {
        self.outputs.push((name, value));
        self
    }

    /// Format this demo as a string block (e.g. for prompt construction).
    pub fn format(&self) -> String {
        let mut buf = String::new();
        for (sym, val) in &self.inputs {
            let name = sym.as_str();
            // Capitalise first letter
            let mut chars = name.chars();
            if let Some(first) = chars.next() {
                buf.push(first.to_ascii_uppercase());
                buf.extend(chars);
            }
            buf.push_str(": ");
            buf.push_str(val);
            buf.push('\n');
        }
        for (sym, val) in &self.outputs {
            let name = sym.as_str();
            let mut chars = name.chars();
            if let Some(first) = chars.next() {
                buf.push(first.to_ascii_uppercase());
                buf.extend(chars);
            }
            buf.push_str(": ");
            buf.push_str(val);
            buf.push('\n');
        }
        buf
    }
}

impl<'a> Default for TypedDemo<'a> {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::{sym, ANSWER, QUESTION, SCORE as SYM_SCORE};

    // -- Layout assertions --------------------------------------------------

    #[test]
    fn typed_field_is_8_bytes() {
        assert_eq!(std::mem::size_of::<TypedField>(), 8);
    }

    #[test]
    fn value_kind_is_1_byte() {
        assert_eq!(std::mem::size_of::<ValueKind>(), 1);
    }

    #[test]
    fn direction_is_1_byte() {
        assert_eq!(std::mem::size_of::<Direction>(), 1);
    }

    #[test]
    fn typed_signature_is_copy() {
        const SIG: TypedSignature = TypedSignature::new("test");
        let a = SIG;
        let b = a; // Copy
        assert_eq!(a.instruction(), b.instruction());
    }

    // -- Const construction -------------------------------------------------

    #[test]
    fn const_construction_basic() {
        const SIG: TypedSignature = TypedSignature::new("Answer questions.")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str);

        assert_eq!(SIG.instruction(), "Answer questions.");
        assert_eq!(SIG.input_count(), 1);
        assert_eq!(SIG.output_count(), 1);

        let inp = SIG.inputs();
        assert_eq!(inp[0].name, QUESTION);
        assert_eq!(inp[0].kind, ValueKind::Str);
        assert!(inp[0].required);
        assert_eq!(inp[0].direction, Direction::Input);

        let out = SIG.outputs();
        assert_eq!(out[0].name, ANSWER);
        assert_eq!(out[0].kind, ValueKind::Str);
        assert!(out[0].required);
        assert_eq!(out[0].direction, Direction::Output);
    }

    #[test]
    fn const_construction_multi_field() {
        const SIG: TypedSignature = TypedSignature::new("Score an answer.")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(SYM_SCORE, ValueKind::Float);

        assert_eq!(SIG.input_count(), 1);
        assert_eq!(SIG.output_count(), 2);
        assert_eq!(SIG.outputs()[1].kind, ValueKind::Float);
    }

    #[test]
    fn const_construction_optional_fields() {
        const SIG: TypedSignature = TypedSignature::new("Optional test")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output_optional(SYM_SCORE, ValueKind::Float);

        assert!(SIG.outputs()[0].required);
        assert!(!SIG.outputs()[1].required);
    }

    #[test]
    fn const_construction_max_fields() {
        let context = sym("context");
        let reasoning = sym("reasoning");
        let evidence = sym("evidence");

        const SIG: TypedSignature = TypedSignature::new("Full")
            .input(QUESTION, ValueKind::Str)
            .input(ANSWER, ValueKind::Str) // reuse just for testing max slots
            .input(QUESTION, ValueKind::Str)
            .input(ANSWER, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(QUESTION, ValueKind::Int)
            .output(ANSWER, ValueKind::Float)
            .output(QUESTION, ValueKind::Bool);

        assert_eq!(SIG.input_count(), 4);
        assert_eq!(SIG.output_count(), 4);

        // Also test runtime builder with dynamic syms
        let rt_sig = TypedSignature::new("Runtime")
            .input(context, ValueKind::Str)
            .output(reasoning, ValueKind::Str)
            .output(evidence, ValueKind::Str);
        assert_eq!(rt_sig.input_count(), 1);
        assert_eq!(rt_sig.output_count(), 2);
    }

    // -- ValueKind::matches -------------------------------------------------

    #[test]
    fn value_kind_str_matches_anything() {
        assert!(ValueKind::Str.matches("hello"));
        assert!(ValueKind::Str.matches(""));
        assert!(ValueKind::Str.matches("42"));
    }

    #[test]
    fn value_kind_int_matches() {
        assert!(ValueKind::Int.matches("42"));
        assert!(ValueKind::Int.matches("-7"));
        assert!(ValueKind::Int.matches(" 100 "));
        assert!(!ValueKind::Int.matches("3.14"));
        assert!(!ValueKind::Int.matches("abc"));
    }

    #[test]
    fn value_kind_float_matches() {
        assert!(ValueKind::Float.matches("3.14"));
        assert!(ValueKind::Float.matches("-0.5"));
        assert!(ValueKind::Float.matches("42")); // ints are valid floats
        assert!(ValueKind::Float.matches(" 1e10 "));
        assert!(!ValueKind::Float.matches("abc"));
    }

    #[test]
    fn value_kind_bool_matches() {
        assert!(ValueKind::Bool.matches("true"));
        assert!(ValueKind::Bool.matches("False"));
        assert!(ValueKind::Bool.matches("YES"));
        assert!(ValueKind::Bool.matches("no"));
        assert!(ValueKind::Bool.matches("1"));
        assert!(ValueKind::Bool.matches("0"));
        assert!(!ValueKind::Bool.matches("maybe"));
        assert!(!ValueKind::Bool.matches(""));
    }

    #[test]
    fn value_kind_json_object_matches() {
        assert!(ValueKind::JsonObject.matches(r#"{"key": "value"}"#));
        assert!(ValueKind::JsonObject.matches("{}"));
        assert!(!ValueKind::JsonObject.matches("[1,2]"));
        assert!(!ValueKind::JsonObject.matches("hello"));
    }

    #[test]
    fn value_kind_json_array_matches() {
        assert!(ValueKind::JsonArray.matches("[1, 2, 3]"));
        assert!(ValueKind::JsonArray.matches("[]"));
        assert!(!ValueKind::JsonArray.matches("{}"));
        assert!(!ValueKind::JsonArray.matches("hello"));
    }

    #[test]
    fn value_kind_enum_matches() {
        assert!(ValueKind::Enum.matches("Option1"));
        assert!(!ValueKind::Enum.matches(""));
        assert!(!ValueKind::Enum.matches("   "));
    }

    // -- ParsedOutput -------------------------------------------------------

    #[test]
    fn parsed_output_basic() {
        let text = "Answer: Paris\nScore: 0.95\n";
        let view = StrView::new(text);

        let mut parsed = ParsedOutput::new(view);
        parsed.push(ANSWER, FieldRange::new(8, 13)); // "Paris"
        parsed.push(SYM_SCORE, FieldRange::new(21, 25)); // "0.95"

        assert_eq!(parsed.get_str(ANSWER), Some("Paris"));
        assert_eq!(parsed.get_str(SYM_SCORE), Some("0.95"));
        assert_eq!(parsed.get_float(SYM_SCORE), Some(0.95));
        assert_eq!(parsed.get_str(QUESTION), None);
        assert_eq!(parsed.field_count(), 2);
    }

    #[test]
    fn parsed_output_get_int() {
        let text = "Count: 42";
        let view = StrView::new(text);
        let count_sym = sym("count");

        let mut parsed = ParsedOutput::new(view);
        parsed.push(count_sym, FieldRange::new(7, 9));

        assert_eq!(parsed.get_int(count_sym), Some(42));
    }

    #[test]
    fn parsed_output_get_bool() {
        let text = "Valid: true";
        let view = StrView::new(text);
        let valid_sym = sym("valid");

        let mut parsed = ParsedOutput::new(view);
        parsed.push(valid_sym, FieldRange::new(7, 11));

        assert_eq!(parsed.get_bool(valid_sym), Some(true));
    }

    #[test]
    fn parsed_output_parse_from_sig() {
        const SIG: TypedSignature = TypedSignature::new("Score an answer.")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(SYM_SCORE, ValueKind::Float);

        let text = "Answer: The capital of France is Paris\nScore: 0.95\n";
        let view = StrView::new(text);
        let parsed = ParsedOutput::parse_from_sig(view, &SIG);

        assert_eq!(parsed.field_count(), 2);
        assert_eq!(
            parsed.get_str(ANSWER),
            Some("The capital of France is Paris")
        );
        assert_eq!(parsed.get_float(SYM_SCORE), Some(0.95));
    }

    #[test]
    fn parsed_output_parse_from_sig_missing_field() {
        const SIG: TypedSignature = TypedSignature::new("Score an answer.")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(SYM_SCORE, ValueKind::Float);

        let text = "Answer: Paris\n";
        let view = StrView::new(text);
        let parsed = ParsedOutput::parse_from_sig(view, &SIG);

        assert_eq!(parsed.field_count(), 1);
        assert_eq!(parsed.get_str(ANSWER), Some("Paris"));
        assert_eq!(parsed.get_float(SYM_SCORE), None);
    }

    #[test]
    fn parsed_output_raw_accessor() {
        let text = "hello world";
        let view = StrView::new(text);
        let parsed = ParsedOutput::new(view);
        assert_eq!(parsed.raw().as_str(), "hello world");
    }

    #[test]
    fn parsed_output_with_fields() {
        let text = "Answer: yes";
        let view = StrView::new(text);
        let fields: SmallVec<[(Sym, FieldRange); 4]> =
            smallvec::smallvec![(ANSWER, FieldRange::new(8, 11))];
        let parsed = ParsedOutput::with_fields(view, fields);
        assert_eq!(parsed.get_str(ANSWER), Some("yes"));
    }

    // -- TypedFieldValidator ------------------------------------------------

    #[test]
    fn validator_all_fields_present_and_correct() {
        const SIG: TypedSignature = TypedSignature::new("Score an answer.")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(SYM_SCORE, ValueKind::Float);

        let v = SIG.validator();
        let score = v.validate("Answer: Paris\nScore: 0.95");
        assert!(score.is_perfect(), "score = {:?}", score);
    }

    #[test]
    fn validator_missing_required_field() {
        const SIG: TypedSignature = TypedSignature::new("Score an answer.")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(SYM_SCORE, ValueKind::Float);

        let v = SIG.validator();
        let score = v.validate("Answer: Paris");
        // 1 of 2 required fields passed
        assert!((score.value - 0.5).abs() < f64::EPSILON);
        assert!(score
            .feedback_str()
            .unwrap()
            .contains("Missing required field"));
    }

    #[test]
    fn validator_wrong_type() {
        const SIG: TypedSignature =
            TypedSignature::new("Count things.").output(SYM_SCORE, ValueKind::Int);

        let v = SIG.validator();
        let score = v.validate("Score: not_a_number");
        assert!((score.value - 0.0).abs() < f64::EPSILON);
        assert!(score.feedback_str().unwrap().contains("expected integer"));
    }

    #[test]
    fn validator_optional_field_not_required() {
        const SIG: TypedSignature = TypedSignature::new("Optional test")
            .output(ANSWER, ValueKind::Str)
            .output_optional(SYM_SCORE, ValueKind::Float);

        let v = SIG.validator();
        // Only required field (answer) is present
        let score = v.validate("Answer: Paris");
        assert!(score.is_perfect());
    }

    #[test]
    fn validator_no_outputs() {
        const SIG: TypedSignature =
            TypedSignature::new("Input only").input(QUESTION, ValueKind::Str);

        let v = SIG.validator();
        let score = v.validate("anything");
        assert!(score.is_perfect());
    }

    #[test]
    fn validator_name() {
        const SIG: TypedSignature = TypedSignature::new("test");
        let v = SIG.validator();
        assert_eq!(v.name(), "typed_field_validator");
    }

    #[test]
    fn validator_via_constructor() {
        const SIG: TypedSignature = TypedSignature::new("test").output(ANSWER, ValueKind::Str);
        let v = TypedFieldValidator::new(SIG);
        assert_eq!(v.signature().instruction(), "test");
        assert!(v.validate("Answer: hello").is_perfect());
    }

    // -- TypedDemo ----------------------------------------------------------

    #[test]
    fn typed_demo_basic() {
        let demo = TypedDemo::new()
            .input(QUESTION, "What is the capital of France?")
            .output(ANSWER, "Paris");

        assert_eq!(demo.inputs.len(), 1);
        assert_eq!(demo.outputs.len(), 1);

        let formatted = demo.format();
        assert!(formatted.contains("Question: What is the capital of France?"));
        assert!(formatted.contains("Answer: Paris"));
    }

    #[test]
    fn typed_demo_default() {
        let demo = TypedDemo::default();
        assert!(demo.inputs.is_empty());
        assert!(demo.outputs.is_empty());
    }

    #[test]
    fn typed_demo_multi_field() {
        let demo = TypedDemo::new()
            .input(QUESTION, "2+2?")
            .output(ANSWER, "4")
            .output(SYM_SCORE, "1.0");

        let formatted = demo.format();
        assert!(formatted.contains("Question: 2+2?"));
        assert!(formatted.contains("Answer: 4"));
        assert!(formatted.contains("Score: 1.0"));
    }

    // -- Lookup helpers -----------------------------------------------------

    #[test]
    fn get_input_output_lookup() {
        const SIG: TypedSignature = TypedSignature::new("Lookup test")
            .input(QUESTION, ValueKind::Str)
            .output(ANSWER, ValueKind::Str)
            .output(SYM_SCORE, ValueKind::Float);

        assert!(SIG.get_input(QUESTION).is_some());
        assert!(SIG.get_input(ANSWER).is_none());
        assert!(SIG.get_output(ANSWER).is_some());
        assert!(SIG.get_output(SYM_SCORE).is_some());
        assert!(SIG.get_output(QUESTION).is_none());
    }

    // -- ValueKind::label ---------------------------------------------------

    #[test]
    fn value_kind_labels() {
        assert_eq!(ValueKind::Str.label(), "string");
        assert_eq!(ValueKind::Int.label(), "integer");
        assert_eq!(ValueKind::Float.label(), "float");
        assert_eq!(ValueKind::Bool.label(), "boolean");
        assert_eq!(ValueKind::JsonObject.label(), "JSON object");
        assert_eq!(ValueKind::JsonArray.label(), "JSON array");
        assert_eq!(ValueKind::Enum.label(), "enum");
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn parsed_output_whitespace_trimming() {
        let text = "Answer:   Paris   \n";
        let view = StrView::new(text);
        let mut parsed = ParsedOutput::new(view);
        parsed.push(ANSWER, FieldRange::new(7, 18));

        // get_raw preserves whitespace
        assert_eq!(parsed.get_raw(ANSWER), Some("   Paris   "));
        // get_str trims
        assert_eq!(parsed.get_str(ANSWER), Some("Paris"));
    }

    #[test]
    fn parsed_output_iter() {
        let text = "Answer: a\nScore: 1";
        let view = StrView::new(text);
        let mut parsed = ParsedOutput::new(view);
        parsed.push(ANSWER, FieldRange::new(8, 9));
        parsed.push(SYM_SCORE, FieldRange::new(17, 18));

        let pairs: Vec<_> = parsed.iter().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, ANSWER);
        assert_eq!(pairs[1].0, SYM_SCORE);
    }

    #[test]
    fn field_range_as_range() {
        let fr = FieldRange::new(5, 10);
        assert_eq!(fr.as_range(), 5..10);
        assert_eq!(fr.len(), 5);
        assert!(!fr.is_empty());

        let empty_fr = FieldRange::new(3, 3);
        assert!(empty_fr.is_empty());
    }
}
