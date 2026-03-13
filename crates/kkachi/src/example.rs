// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Training and evaluation examples

use crate::types::FieldMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// An example for training or evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example<'a> {
    /// Input fields
    #[serde(borrow)]
    pub inputs: FieldMap<'a>,

    /// Optional ground truth outputs
    #[serde(borrow)]
    pub outputs: Option<FieldMap<'a>>,
}

impl<'a> Example<'a> {
    /// Create a new example
    pub fn new() -> Self {
        Self {
            inputs: FieldMap::new(),
            outputs: None,
        }
    }

    /// Create with inputs
    pub fn with_inputs(inputs: FieldMap<'a>) -> Self {
        Self {
            inputs,
            outputs: None,
        }
    }

    /// Add outputs (ground truth)
    pub fn with_outputs(mut self, outputs: FieldMap<'a>) -> Self {
        self.outputs = Some(outputs);
        self
    }

    /// Get an input field
    pub fn get_input(&self, key: &str) -> Option<&str> {
        self.inputs.get(key).map(|v| v.as_ref())
    }

    /// Get an output field
    pub fn get_output(&self, key: &str) -> Option<&str> {
        self.outputs.as_ref()?.get(key).map(|v| v.as_ref())
    }

    /// Insert an input field
    pub fn insert_input(&mut self, key: impl Into<Cow<'a, str>>, value: impl Into<Cow<'a, str>>) {
        self.inputs.insert(key.into(), value.into());
    }

    /// Insert an output field
    pub fn insert_output(&mut self, key: impl Into<Cow<'a, str>>, value: impl Into<Cow<'a, str>>) {
        self.outputs
            .get_or_insert_with(FieldMap::new)
            .insert(key.into(), value.into());
    }

    /// Convert to owned version
    pub fn into_owned(self) -> Example<'static> {
        Example {
            inputs: self
                .inputs
                .into_iter()
                .map(|(k, v)| (Cow::Owned(k.into_owned()), Cow::Owned(v.into_owned())))
                .collect(),
            outputs: self.outputs.map(|outputs| {
                outputs
                    .into_iter()
                    .map(|(k, v)| (Cow::Owned(k.into_owned()), Cow::Owned(v.into_owned())))
                    .collect()
            }),
        }
    }
}

impl<'a> Default for Example<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_creation() {
        let example = Example::new();
        assert!(example.inputs.is_empty());
        assert!(example.outputs.is_none());
    }

    #[test]
    fn test_example_insert_input() {
        let mut example = Example::new();
        example.insert_input("question", "What is 2+2?");

        assert_eq!(example.get_input("question"), Some("What is 2+2?"));
        assert_eq!(example.inputs.len(), 1);
    }

    #[test]
    fn test_example_insert_output() {
        let mut example = Example::new();
        example.insert_output("answer", "4");

        assert_eq!(example.get_output("answer"), Some("4"));
        assert!(example.outputs.is_some());
        assert_eq!(example.outputs.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_example_with_inputs() {
        let mut inputs = FieldMap::new();
        inputs.insert("q1".into(), "value1".into());

        let example = Example::with_inputs(inputs);
        assert_eq!(example.get_input("q1"), Some("value1"));
    }

    #[test]
    fn test_example_with_outputs() {
        let mut outputs = FieldMap::new();
        outputs.insert("a1".into(), "answer1".into());

        let example = Example::new().with_outputs(outputs);
        assert_eq!(example.get_output("a1"), Some("answer1"));
    }

    #[test]
    fn test_example_into_owned() {
        let mut example = Example::new();
        example.insert_input("test", "value");
        example.insert_output("result", "output");

        let owned = example.into_owned();
        assert_eq!(owned.get_input("test"), Some("value"));
        assert_eq!(owned.get_output("result"), Some("output"));
    }

    #[test]
    fn test_example_multiple_fields() {
        let mut example = Example::new();
        example.insert_input("question", "Q1");
        example.insert_input("context", "C1");
        example.insert_output("answer", "A1");
        example.insert_output("confidence", "0.95");

        assert_eq!(example.inputs.len(), 2);
        assert_eq!(example.outputs.as_ref().unwrap().len(), 2);
    }
}
