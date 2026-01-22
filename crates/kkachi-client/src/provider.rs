// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Provider abstraction for different LM backends

use crate::lm::LMConfig;
use crate::request::LMRequest;
use crate::response::{LMResponse, Usage};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Type of LM provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderType {
    /// OpenAI
    OpenAI,
    /// Anthropic
    Anthropic,
    /// Local model
    Local,
    /// Custom provider
    Custom,
}

/// Provider trait for LM backends
#[async_trait]
pub trait Provider: Send + Sync {
    /// Complete a request
    async fn complete(
        &self,
        request: LMRequest<'_>,
        config: &LMConfig,
    ) -> anyhow::Result<LMResponse>;

    /// Get provider type
    fn provider_type(&self) -> ProviderType;
}

/// OpenAI provider implementation
pub struct OpenAIProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn complete(
        &self,
        request: LMRequest<'_>,
        config: &LMConfig,
    ) -> anyhow::Result<LMResponse> {
        #[derive(Serialize)]
        struct OpenAIRequest<'a> {
            model: &'a str,
            messages: &'a [crate::request::Message<'a>],
            temperature: f32,
            max_tokens: u32,
        }

        #[derive(Deserialize)]
        struct OpenAIResponse {
            choices: Vec<Choice>,
            usage: Option<OpenAIUsage>,
            model: String,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: OpenAIMessage,
            finish_reason: Option<String>,
        }

        #[derive(Deserialize)]
        struct OpenAIMessage {
            content: String,
        }

        #[derive(Deserialize)]
        struct OpenAIUsage {
            prompt_tokens: u32,
            completion_tokens: u32,
            #[allow(dead_code)] // Present in API response but not currently used
            total_tokens: u32,
        }

        let req = OpenAIRequest {
            model: &config.model,
            messages: &request.messages,
            temperature: request.temperature.unwrap_or(config.temperature),
            max_tokens: request.max_tokens.unwrap_or(config.max_tokens),
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await?
            .json::<OpenAIResponse>()
            .await?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

        let mut lm_response = LMResponse::new(choice.message.content.clone(), response.model);

        if let Some(usage) = response.usage {
            lm_response =
                lm_response.with_usage(Usage::new(usage.prompt_tokens, usage.completion_tokens));
        }

        if let Some(reason) = &choice.finish_reason {
            lm_response = lm_response.with_finish_reason(reason.clone());
        }

        Ok(lm_response)
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type() {
        let provider = OpenAIProvider::new("test-key".to_string());
        assert_eq!(provider.provider_type(), ProviderType::OpenAI);
    }
}
