// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! LM response types

use serde::{Deserialize, Serialize};

/// Response from language model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMResponse {
    /// Generated text
    pub text: String,

    /// Token usage
    pub usage: Option<Usage>,

    /// Model that generated the response
    pub model: String,

    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Token usage statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: u32,

    /// Completion tokens
    pub completion_tokens: u32,

    /// Total tokens
    #[allow(dead_code)]
    pub total_tokens: u32,
}

impl Usage {
    /// Create new usage stats
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

impl LMResponse {
    /// Create a new response
    pub fn new(text: String, model: String) -> Self {
        Self {
            text,
            usage: None,
            model,
            finish_reason: None,
        }
    }

    /// Add usage stats
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, reason: String) -> Self {
        self.finish_reason = Some(reason);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_new() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_lm_response_new() {
        let resp = LMResponse::new("test".to_string(), "gpt-4".to_string());
        assert_eq!(resp.text, "test");
        assert_eq!(resp.model, "gpt-4");
        assert!(resp.usage.is_none());
        assert!(resp.finish_reason.is_none());
    }

    #[test]
    fn test_lm_response_builder() {
        let usage = Usage::new(10, 5);
        let resp = LMResponse::new("test".to_string(), "gpt-3.5".to_string())
            .with_usage(usage)
            .with_finish_reason("stop".to_string());

        assert!(resp.usage.is_some());
        assert_eq!(resp.finish_reason, Some("stop".to_string()));
    }
}
