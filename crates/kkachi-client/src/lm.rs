// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Language model client abstraction

use crate::provider::Provider;
use crate::request::LMRequest;
use crate::response::LMResponse;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Configuration for LM client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMConfig {
    /// Model name
    pub model: String,

    /// Temperature
    pub temperature: f32,

    /// Max tokens
    pub max_tokens: u32,

    /// Whether to cache responses
    pub cache: bool,

    /// Number of retries
    pub num_retries: u32,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            temperature: 0.0,
            max_tokens: 4000,
            cache: true,
            num_retries: 3,
        }
    }
}

/// Language model client trait
#[async_trait]
pub trait LM: Send + Sync {
    /// Generate a completion
    async fn generate(&self, request: LMRequest<'_>) -> anyhow::Result<LMResponse>;

    /// Get the model name
    fn model(&self) -> &str;

    /// Get configuration
    fn config(&self) -> &LMConfig;
}

/// Concrete LM implementation
pub struct LMClient {
    config: LMConfig,
    provider: Box<dyn Provider>,
}

impl LMClient {
    /// Create a new LM client
    pub fn new(config: LMConfig, provider: Box<dyn Provider>) -> Self {
        Self { config, provider }
    }

    /// Create with default config
    pub fn with_provider(provider: Box<dyn Provider>) -> Self {
        Self::new(LMConfig::default(), provider)
    }
}

#[async_trait]
impl LM for LMClient {
    async fn generate(&self, request: LMRequest<'_>) -> anyhow::Result<LMResponse> {
        self.provider.complete(request, &self.config).await
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    fn config(&self) -> &LMConfig {
        &self.config
    }
}
