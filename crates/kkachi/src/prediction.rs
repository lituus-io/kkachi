// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Prediction results from module execution

use crate::types::FieldMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Result of a module prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction<'a> {
    /// Output fields
    #[serde(borrow)]
    pub outputs: FieldMap<'a>,

    /// Optional metadata
    #[serde(borrow)]
    pub metadata: Option<PredictionMetadata<'a>>,
}

/// Metadata about a prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata<'a> {
    /// Model used
    #[serde(borrow)]
    pub model: Option<Cow<'a, str>>,

    /// Token usage
    pub tokens: Option<TokenUsage>,

    /// Latency in milliseconds
    pub latency_ms: Option<u64>,

    /// Confidence score
    pub confidence: Option<f64>,
}

/// Token usage statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt tokens
    pub prompt_tokens: u32,

    /// Completion tokens
    pub completion_tokens: u32,

    /// Total tokens
    pub total_tokens: u32,
}

impl<'a> Prediction<'a> {
    /// Create a new prediction
    pub fn new() -> Self {
        Self {
            outputs: FieldMap::new(),
            metadata: None,
        }
    }

    /// Create with outputs
    pub fn with_outputs(outputs: FieldMap<'a>) -> Self {
        Self {
            outputs,
            metadata: None,
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: PredictionMetadata<'a>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get an output field
    pub fn get(&self, key: &str) -> Option<&str> {
        self.outputs.get(key).map(|v| v.as_ref())
    }

    /// Insert an output field
    pub fn insert(&mut self, key: impl Into<Cow<'a, str>>, value: impl Into<Cow<'a, str>>) {
        self.outputs.insert(key.into(), value.into());
    }

    /// Convert to owned version
    pub fn into_owned(self) -> Prediction<'static> {
        Prediction {
            outputs: self
                .outputs
                .into_iter()
                .map(|(k, v)| (Cow::Owned(k.into_owned()), Cow::Owned(v.into_owned())))
                .collect(),
            metadata: self.metadata.map(|m| PredictionMetadata {
                model: m.model.map(|s| Cow::Owned(s.into_owned())),
                tokens: m.tokens,
                latency_ms: m.latency_ms,
                confidence: m.confidence,
            }),
        }
    }
}

impl<'a> Default for Prediction<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenUsage {
    /// Create new token usage
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(10, 5);
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_prediction_creation() {
        let pred = Prediction::new();
        assert!(pred.outputs.is_empty());
        assert!(pred.metadata.is_none());
    }

    #[test]
    fn test_prediction_insert_and_get() {
        let mut pred = Prediction::new();
        pred.insert("answer", "42");

        assert_eq!(pred.get("answer"), Some("42"));
        assert_eq!(pred.get("missing"), None);
    }

    #[test]
    fn test_prediction_with_metadata() {
        let metadata = PredictionMetadata {
            model: Some(Cow::Borrowed("gpt-4")),
            tokens: Some(TokenUsage::new(10, 5)),
            latency_ms: Some(250),
            confidence: Some(0.95),
        };

        let pred = Prediction::new().with_metadata(metadata);
        assert!(pred.metadata.is_some());

        let meta = pred.metadata.unwrap();
        assert_eq!(meta.model, Some(Cow::Borrowed("gpt-4")));
        assert_eq!(meta.latency_ms, Some(250));
        assert_eq!(meta.confidence, Some(0.95));
    }

    #[test]
    fn test_prediction_into_owned() {
        let mut pred = Prediction::new();
        pred.insert("key", "value");

        let owned = pred.into_owned();
        assert_eq!(owned.get("key"), Some("value"));
    }
}
