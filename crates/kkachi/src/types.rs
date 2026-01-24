// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Core type definitions

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// A value that can be either owned or borrowed
pub type Value<'a> = Cow<'a, str>;

/// Input/Output field map using zero-copy strings
pub type FieldMap<'a> = std::collections::HashMap<Cow<'a, str>, Cow<'a, str>>;

/// Represents structured input data with lifetime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inputs<'a> {
    #[serde(borrow)]
    fields: FieldMap<'a>,
}

impl<'a> Inputs<'a> {
    /// Create new inputs
    pub fn new() -> Self {
        Self {
            fields: std::collections::HashMap::new(),
        }
    }

    /// Insert a field
    pub fn insert(&mut self, key: impl Into<Cow<'a, str>>, value: impl Into<Cow<'a, str>>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Get a field
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|v| v.as_ref())
    }

    /// Get all fields
    pub fn fields(&self) -> &FieldMap<'a> {
        &self.fields
    }

    /// Convert to owned version
    pub fn into_owned(self) -> Inputs<'static> {
        Inputs {
            fields: self
                .fields
                .into_iter()
                .map(|(k, v)| (Cow::Owned(k.into_owned()), Cow::Owned(v.into_owned())))
                .collect(),
        }
    }
}

impl<'a> Default for Inputs<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> FromIterator<(Cow<'a, str>, Cow<'a, str>)> for Inputs<'a> {
    fn from_iter<T: IntoIterator<Item = (Cow<'a, str>, Cow<'a, str>)>>(iter: T) -> Self {
        Self {
            fields: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inputs_creation() {
        let inputs = Inputs::new();
        assert_eq!(inputs.fields().len(), 0);
    }

    #[test]
    fn test_inputs_insert_and_get() {
        let mut inputs = Inputs::new();
        inputs.insert("key1", "value1");
        inputs.insert("key2", "value2");

        assert_eq!(inputs.get("key1"), Some("value1"));
        assert_eq!(inputs.get("key2"), Some("value2"));
        assert_eq!(inputs.get("key3"), None);
    }

    #[test]
    fn test_inputs_from_iter() {
        let data = vec![
            (Cow::Borrowed("a"), Cow::Borrowed("1")),
            (Cow::Borrowed("b"), Cow::Borrowed("2")),
        ];

        let inputs: Inputs = data.into_iter().collect();
        assert_eq!(inputs.get("a"), Some("1"));
        assert_eq!(inputs.get("b"), Some("2"));
    }

    #[test]
    fn test_inputs_into_owned() {
        let mut inputs = Inputs::new();
        inputs.insert("test", "value");

        let owned = inputs.into_owned();
        assert_eq!(owned.get("test"), Some("value"));
    }

    #[test]
    fn test_inputs_borrowed_vs_owned() {
        let mut inputs = Inputs::new();
        inputs.insert(Cow::Borrowed("borrowed"), Cow::Owned("owned".to_string()));

        assert_eq!(inputs.get("borrowed"), Some("owned"));
    }
}
