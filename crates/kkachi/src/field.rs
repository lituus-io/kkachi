// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Field definitions for signatures

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Type of field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Input field
    Input,
    /// Output field
    Output,
}

/// A field in a signature with zero-copy strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field<'a> {
    /// Field name
    #[serde(borrow)]
    pub name: Cow<'a, str>,

    /// Field description
    #[serde(borrow)]
    pub desc: Cow<'a, str>,

    /// Field prefix for formatting
    #[serde(borrow)]
    pub prefix: Cow<'a, str>,

    /// Field type (input or output)
    pub field_type: FieldType,

    /// Optional format specification
    #[serde(borrow)]
    pub format: Option<Cow<'a, str>>,
}

impl<'a> Field<'a> {
    /// Create a new field
    pub fn new(
        name: impl Into<Cow<'a, str>>,
        desc: impl Into<Cow<'a, str>>,
        field_type: FieldType,
    ) -> Self {
        let name = name.into();
        let prefix = Self::infer_prefix(&name);

        Self {
            name,
            desc: desc.into(),
            prefix: Cow::Owned(prefix),
            field_type,
            format: None,
        }
    }

    /// Set the prefix
    pub fn with_prefix(mut self, prefix: impl Into<Cow<'a, str>>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the format
    pub fn with_format(mut self, format: impl Into<Cow<'a, str>>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Convert to owned version
    pub fn into_owned(self) -> Field<'static> {
        Field {
            name: Cow::Owned(self.name.into_owned()),
            desc: Cow::Owned(self.desc.into_owned()),
            prefix: Cow::Owned(self.prefix.into_owned()),
            field_type: self.field_type,
            format: self.format.map(|f| Cow::Owned(f.into_owned())),
        }
    }

    /// Infer prefix from field name (camelCase -> Camel Case)
    fn infer_prefix(name: &str) -> String {
        let mut result = String::with_capacity(name.len() + 5);
        let mut prev_lower = false;

        for (i, ch) in name.chars().enumerate() {
            if i > 0 && ch.is_uppercase() && prev_lower {
                result.push(' ');
            }

            if i == 0 {
                result.push(ch.to_ascii_uppercase());
            } else {
                result.push(ch);
            }

            prev_lower = ch.is_lowercase();
        }

        result
    }
}

/// Helper to create an input field
pub struct InputField;

impl InputField {
    /// Create an input field.
    pub fn create<'a>(name: impl Into<Cow<'a, str>>, desc: impl Into<Cow<'a, str>>) -> Field<'a> {
        Field::new(name, desc, FieldType::Input)
    }
}

/// Helper to create an output field
pub struct OutputField;

impl OutputField {
    /// Create an output field.
    pub fn create<'a>(name: impl Into<Cow<'a, str>>, desc: impl Into<Cow<'a, str>>) -> Field<'a> {
        Field::new(name, desc, FieldType::Output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_prefix() {
        assert_eq!(Field::infer_prefix("question"), "Question");
        assert_eq!(Field::infer_prefix("answerText"), "Answer Text");
        assert_eq!(Field::infer_prefix("someValue"), "Some Value");
    }

    #[test]
    fn test_field_creation() {
        let field = InputField::create("query", "User query");
        assert_eq!(field.name, "query");
        assert_eq!(field.desc, "User query");
        assert_eq!(field.prefix, "Query");
        assert_eq!(field.field_type, FieldType::Input);
    }
}
