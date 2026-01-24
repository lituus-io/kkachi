// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Real LLM API client supporting Anthropic, OpenAI, and Claude Code CLI.
//!
//! This module provides [`ApiLlm`], a generic LLM client that can talk to
//! multiple providers. It implements the [`Llm`] trait using blocking HTTP
//! requests (via `reqwest::blocking`), keeping the same sync pattern as
//! [`MockLlm`].
//!
//! # Providers
//!
//! - **Anthropic**: Claude API (Messages format)
//! - **OpenAI**: Chat Completions API (also compatible endpoints like Together, Groq, etc.)
//! - **ClaudeCode**: Local `claude` CLI subprocess (no API key needed)
//!
//! # Examples
//!
//! ```ignore
//! use kkachi::recursive::ApiLlm;
//!
//! // Auto-detect from environment (ANTHROPIC_API_KEY or OPENAI_API_KEY or claude CLI)
//! let llm = ApiLlm::from_env().unwrap();
//!
//! // Explicit providers
//! let llm = ApiLlm::anthropic("sk-...", "claude-sonnet-4-20250514");
//! let llm = ApiLlm::openai("sk-...", "gpt-4o");
//! let llm = ApiLlm::claude_code();
//! ```

use crate::error::{Error, Result};
use crate::recursive::llm::{Llm, LmOutput};
use reqwest::blocking::Client;
use serde_json::Value;
use std::sync::Arc;

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Claude API (Messages format).
    Anthropic {
        /// API key for authentication.
        api_key: String,
        /// Model identifier (e.g., "claude-sonnet-4-20250514").
        model: String,
        /// Base URL for the API endpoint.
        base_url: String,
    },
    /// OpenAI Chat Completions API (also compatible endpoints).
    OpenAI {
        /// API key for authentication.
        api_key: String,
        /// Model identifier (e.g., "gpt-4o").
        model: String,
        /// Base URL for the API endpoint.
        base_url: String,
    },
    /// Local Claude Code CLI (subprocess invocation, no API key needed).
    ClaudeCode {
        /// Path to the `claude` binary.
        path: String,
    },
}

/// Real LLM client implementing the [`Llm`] trait.
///
/// Supports three modes:
/// 1. Anthropic API (requires `ANTHROPIC_API_KEY`)
/// 2. OpenAI API (requires `OPENAI_API_KEY`)
/// 3. Claude Code CLI (uses locally installed `claude` binary, no key needed)
pub struct ApiLlm {
    client: Option<Client>,
    provider: Provider,
    temperature: f64,
    max_tokens: u32,
    on_token: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

impl ApiLlm {
    /// Auto-detect provider from environment.
    ///
    /// Checked in order:
    /// 1. `ANTHROPIC_API_KEY` → Anthropic Claude
    /// 2. `OPENAI_API_KEY` → OpenAI
    /// 3. `claude` binary in PATH → ClaudeCode
    ///
    /// Override defaults with:
    /// - `KKACHI_MODEL` — model name (API providers only)
    /// - `KKACHI_BASE_URL` — endpoint URL (API providers only)
    pub fn from_env() -> Result<Self> {
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            let model = std::env::var("KKACHI_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
            let base_url = std::env::var("KKACHI_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
            return Ok(Self::anthropic_with_url(key, model, base_url));
        }

        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let model = std::env::var("KKACHI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
            let base_url = std::env::var("KKACHI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".to_string());
            return Ok(Self::openai_with_url(key, model, base_url));
        }

        // Check for claude CLI
        if which_claude().is_some() {
            return Ok(Self::claude_code());
        }

        Err(Error::module(
            "No LLM provider found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or install claude CLI.",
        ))
    }

    /// Create an Anthropic Claude client.
    pub fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::anthropic_with_url(api_key, model, "https://api.anthropic.com")
    }

    /// Create an Anthropic Claude client with a custom base URL.
    pub fn anthropic_with_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: Some(Client::new()),
            provider: Provider::Anthropic {
                api_key: api_key.into(),
                model: model.into(),
                base_url: base_url.into(),
            },
            temperature: 0.7,
            max_tokens: 4096,
            on_token: None,
        }
    }

    /// Create an OpenAI client.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::openai_with_url(api_key, model, "https://api.openai.com")
    }

    /// Create an OpenAI client with a custom base URL.
    pub fn openai_with_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: Some(Client::new()),
            provider: Provider::OpenAI {
                api_key: api_key.into(),
                model: model.into(),
                base_url: base_url.into(),
            },
            temperature: 0.7,
            max_tokens: 4096,
            on_token: None,
        }
    }

    /// Create a Claude Code CLI client (no API key needed).
    ///
    /// Uses the locally installed `claude` binary. Falls back to searching
    /// common installation paths if not found in PATH.
    pub fn claude_code() -> Self {
        let path = which_claude().unwrap_or_else(|| "claude".to_string());
        Self {
            client: None,
            provider: Provider::ClaudeCode { path },
            temperature: 0.7,
            max_tokens: 4096,
            on_token: None,
        }
    }

    /// Set the temperature for generation.
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the maximum tokens for generation.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set a callback that receives each token as it's generated.
    ///
    /// When set, the API client uses streaming mode and calls this
    /// function with each token/chunk as it arrives. The full response
    /// is still accumulated and returned as normal.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let llm = ApiLlm::from_env().unwrap()
    ///     .on_token(|token| print!("{}", token));
    /// ```
    pub fn on_token(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.on_token = Some(Arc::new(f));
        self
    }

    /// Get the provider being used.
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    #[cfg(feature = "tracing")]
    fn provider_name(&self) -> &str {
        match &self.provider {
            Provider::Anthropic { .. } => "anthropic",
            Provider::OpenAI { .. } => "openai",
            Provider::ClaudeCode { .. } => "claude_code",
        }
    }

    fn call_api(&self, prompt: &str, context: &str, feedback: Option<&str>) -> Result<LmOutput> {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!(
            "llm_call",
            provider = self.provider_name(),
            model = self.model_name(),
        )
        .entered();

        let result = self.call_api_inner(prompt, context, feedback);

        #[cfg(feature = "tracing")]
        if let Ok(ref output) = result {
            tracing::info!(
                prompt_tokens = output.prompt_tokens,
                completion_tokens = output.completion_tokens,
                "llm call complete"
            );
        }

        result
    }

    fn call_api_inner(
        &self,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<LmOutput> {
        match &self.provider {
            Provider::Anthropic {
                api_key,
                model,
                base_url,
            } => {
                if self.on_token.is_some() {
                    self.call_anthropic_streaming(
                        api_key, model, base_url, prompt, context, feedback,
                    )
                } else {
                    self.call_anthropic(api_key, model, base_url, prompt, context, feedback)
                }
            }
            Provider::OpenAI {
                api_key,
                model,
                base_url,
            } => {
                if self.on_token.is_some() {
                    self.call_openai_streaming(api_key, model, base_url, prompt, context, feedback)
                } else {
                    self.call_openai(api_key, model, base_url, prompt, context, feedback)
                }
            }
            Provider::ClaudeCode { path } => self.call_claude_code(path, prompt, context, feedback),
        }
    }

    fn build_user_message(&self, prompt: &str, feedback: Option<&str>) -> String {
        match feedback {
            Some(fb) => format!("{}\n\n[Previous attempt feedback: {}]", prompt, fb),
            None => prompt.to_string(),
        }
    }

    fn call_anthropic(
        &self,
        api_key: &str,
        model: &str,
        base_url: &str,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<LmOutput> {
        let client = self.client.as_ref().unwrap();
        let user_message = self.build_user_message(prompt, feedback);

        let mut body = serde_json::json!({
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user_message}]
        });

        if !context.is_empty() {
            body["system"] = Value::String(context.to_string());
        }

        if self.temperature != 0.7 {
            body["temperature"] = Value::from(self.temperature);
        }

        let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| Error::module(format!("Anthropic API request failed: {}", e)))?;

        let status = response.status();
        let response_text = response
            .text()
            .map_err(|e| Error::module(format!("Failed to read response body: {}", e)))?;

        if !status.is_success() {
            return Err(Error::module(format!(
                "Anthropic API error ({}): {}",
                status, response_text
            )));
        }

        let json: Value = serde_json::from_str(&response_text)
            .map_err(|e| Error::module(format!("Failed to parse Anthropic response: {}", e)))?;

        let text = json["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .ok_or_else(|| {
                Error::module(format!("Unexpected Anthropic response format: {}", json))
            })?
            .to_string();

        let prompt_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LmOutput::with_tokens(
            text,
            prompt_tokens,
            completion_tokens,
        ))
    }

    fn call_openai(
        &self,
        api_key: &str,
        model: &str,
        base_url: &str,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<LmOutput> {
        let client = self.client.as_ref().unwrap();
        let user_message = self.build_user_message(prompt, feedback);

        let mut messages = Vec::new();
        if !context.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": context}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user_message}));

        let body = serde_json::json!({
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        });

        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| Error::module(format!("OpenAI API request failed: {}", e)))?;

        let status = response.status();
        let response_text = response
            .text()
            .map_err(|e| Error::module(format!("Failed to read response body: {}", e)))?;

        if !status.is_success() {
            return Err(Error::module(format!(
                "OpenAI API error ({}): {}",
                status, response_text
            )));
        }

        let json: Value = serde_json::from_str(&response_text)
            .map_err(|e| Error::module(format!("Failed to parse OpenAI response: {}", e)))?;

        let text = json["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .ok_or_else(|| Error::module(format!("Unexpected OpenAI response format: {}", json)))?
            .to_string();

        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LmOutput::with_tokens(
            text,
            prompt_tokens,
            completion_tokens,
        ))
    }

    fn call_anthropic_streaming(
        &self,
        api_key: &str,
        model: &str,
        base_url: &str,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<LmOutput> {
        let client = self.client.as_ref().unwrap();
        let user_message = self.build_user_message(prompt, feedback);
        let on_token = self.on_token.as_ref().unwrap();

        let mut body = serde_json::json!({
            "model": model,
            "max_tokens": self.max_tokens,
            "stream": true,
            "messages": [{"role": "user", "content": user_message}]
        });

        if !context.is_empty() {
            body["system"] = Value::String(context.to_string());
        }

        if self.temperature != 0.7 {
            body["temperature"] = Value::from(self.temperature);
        }

        let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| Error::module(format!("Anthropic API request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().unwrap_or_default();
            return Err(Error::module(format!(
                "Anthropic API error ({}): {}",
                status, error_text
            )));
        }

        // Parse SSE stream
        let mut accumulated = String::new();
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let text = response
            .text()
            .map_err(|e| Error::module(format!("Failed to read streaming response: {}", e)))?;

        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(json) = serde_json::from_str::<Value>(data) {
                    match json["type"].as_str() {
                        Some("content_block_delta") => {
                            if let Some(text) = json["delta"]["text"].as_str() {
                                on_token(text);
                                accumulated.push_str(text);
                            }
                        }
                        Some("message_delta") => {
                            if let Some(t) = json["usage"]["output_tokens"].as_u64() {
                                completion_tokens = t as u32;
                            }
                        }
                        Some("message_start") => {
                            if let Some(t) = json["message"]["usage"]["input_tokens"].as_u64() {
                                prompt_tokens = t as u32;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(LmOutput::with_tokens(
            accumulated,
            prompt_tokens,
            completion_tokens,
        ))
    }

    fn call_openai_streaming(
        &self,
        api_key: &str,
        model: &str,
        base_url: &str,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<LmOutput> {
        let client = self.client.as_ref().unwrap();
        let user_message = self.build_user_message(prompt, feedback);
        let on_token = self.on_token.as_ref().unwrap();

        let mut messages = Vec::new();
        if !context.is_empty() {
            messages.push(serde_json::json!({"role": "system", "content": context}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user_message}));

        let body = serde_json::json!({
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": true,
            "messages": messages
        });

        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| Error::module(format!("OpenAI API request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().unwrap_or_default();
            return Err(Error::module(format!(
                "OpenAI API error ({}): {}",
                status, error_text
            )));
        }

        // Parse SSE stream
        let mut accumulated = String::new();
        let text = response
            .text()
            .map_err(|e| Error::module(format!("Failed to read streaming response: {}", e)))?;

        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }
                if let Ok(json) = serde_json::from_str::<Value>(data) {
                    if let Some(content) = json["choices"]
                        .as_array()
                        .and_then(|arr| arr.first())
                        .and_then(|choice| choice["delta"]["content"].as_str())
                    {
                        on_token(content);
                        accumulated.push_str(content);
                    }
                }
            }
        }

        // Estimate tokens for streaming (not provided in chunks)
        let est_prompt_tokens = (prompt.split_whitespace().count() as f64 * 1.3) as u32;
        let est_completion_tokens = (accumulated.split_whitespace().count() as f64 * 1.3) as u32;

        Ok(LmOutput::with_tokens(
            accumulated,
            est_prompt_tokens,
            est_completion_tokens,
        ))
    }

    fn call_claude_code(
        &self,
        path: &str,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<LmOutput> {
        let mut combined = String::new();
        if !context.is_empty() {
            combined.push_str(context);
            combined.push_str("\n\n");
        }
        combined.push_str(prompt);
        if let Some(fb) = feedback {
            combined.push_str("\n\n[Previous attempt feedback: ");
            combined.push_str(fb);
            combined.push(']');
        }

        let output = std::process::Command::new(path)
            .args(["-p", &combined, "--output-format", "text"])
            .output()
            .map_err(|e| Error::module(format!("Failed to execute claude CLI: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::module(format!("claude CLI failed: {}", stderr)));
        }

        let text = String::from_utf8(output.stdout)
            .map_err(|e| Error::module(format!("claude CLI output is not valid UTF-8: {}", e)))?
            .trim()
            .to_string();

        // Estimate tokens from word count (rough approximation)
        let word_count = text.split_whitespace().count() as u32;
        let prompt_word_count = combined.split_whitespace().count() as u32;
        let est_prompt_tokens = (prompt_word_count as f64 * 1.3) as u32;
        let est_completion_tokens = (word_count as f64 * 1.3) as u32;

        Ok(LmOutput::with_tokens(
            text,
            est_prompt_tokens,
            est_completion_tokens,
        ))
    }
}

impl Llm for ApiLlm {
    type GenerateFut<'a> = std::future::Ready<Result<LmOutput>>;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        std::future::ready(self.call_api(prompt, context, feedback))
    }

    fn model_name(&self) -> &str {
        match &self.provider {
            Provider::Anthropic { model, .. } => model,
            Provider::OpenAI { model, .. } => model,
            Provider::ClaudeCode { .. } => "claude-code",
        }
    }

    fn max_context(&self) -> usize {
        match &self.provider {
            Provider::Anthropic { .. } => 200_000,
            Provider::OpenAI { .. } => 128_000,
            Provider::ClaudeCode { .. } => 200_000,
        }
    }
}

/// Find the `claude` binary in PATH or common locations.
fn which_claude() -> Option<String> {
    // Check PATH first
    if let Ok(output) = std::process::Command::new("which").arg("claude").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    // Common installation paths
    let common_paths = ["/usr/local/bin/claude", "/opt/homebrew/bin/claude"];

    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let llm = ApiLlm::anthropic("test-key", "claude-sonnet-4-20250514");
        assert_eq!(llm.model_name(), "claude-sonnet-4-20250514");
        assert_eq!(llm.max_context(), 200_000);

        let llm = ApiLlm::openai("test-key", "gpt-4o");
        assert_eq!(llm.model_name(), "gpt-4o");
        assert_eq!(llm.max_context(), 128_000);
    }

    #[test]
    fn test_builder_methods() {
        let llm = ApiLlm::anthropic("key", "model")
            .temperature(0.5)
            .max_tokens(2048);
        assert!((llm.temperature - 0.5).abs() < f64::EPSILON);
        assert_eq!(llm.max_tokens, 2048);
    }

    #[test]
    fn test_build_user_message() {
        let llm = ApiLlm::anthropic("key", "model");

        let msg = llm.build_user_message("Write code", None);
        assert_eq!(msg, "Write code");

        let msg = llm.build_user_message("Write code", Some("Add error handling"));
        assert!(msg.contains("Write code"));
        assert!(msg.contains("[Previous attempt feedback: Add error handling]"));
    }

    #[test]
    fn test_provider_enum() {
        let p = Provider::ClaudeCode {
            path: "/usr/local/bin/claude".to_string(),
        };
        assert_eq!(
            p,
            Provider::ClaudeCode {
                path: "/usr/local/bin/claude".to_string()
            }
        );
    }

    #[test]
    fn test_claude_code_creation() {
        let llm = ApiLlm::claude_code();
        assert_eq!(llm.model_name(), "claude-code");
        assert!(llm.client.is_none());
    }

    #[test]
    fn test_custom_base_url() {
        let llm = ApiLlm::openai_with_url("key", "model", "https://custom.api.com");
        match &llm.provider {
            Provider::OpenAI { base_url, .. } => {
                assert_eq!(base_url, "https://custom.api.com");
            }
            _ => panic!("Wrong provider"),
        }
    }
}
