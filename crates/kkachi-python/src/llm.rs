// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for LLM implementations.
//!
//! Provides ApiLlm for real LLM API access (Anthropic, OpenAI, Claude Code CLI)
//! and support for custom endpoints, with optimization wrappers (cache, rate limiting, retry).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kkachi::recursive::{
    ApiLlm, CacheExt, CachedLlm, LlmExt, RateLimitExt, RateLimitedLlm, RetryLlm,
};

/// Internal enum holding different LLM variants with optimizations applied.
///
/// This allows Python users to chain optimizations like:
/// `llm.with_cache(100).with_rate_limit(10.0).with_retry(3)`
///
/// Valid chaining order: cache → rate_limit → retry (outermost)
/// Any order works, but this enum only stores the resulting types.
pub enum LlmVariant {
    /// Plain API LLM
    Plain(ApiLlm),

    // Single optimization
    /// Cached LLM
    Cached(CachedLlm<ApiLlm>),
    /// Rate-limited LLM
    RateLimited(RateLimitedLlm<ApiLlm>),
    /// Retry LLM
    Retry(RetryLlm<ApiLlm>),

    // Two optimizations
    /// Rate Limited wrapping Cached (cache -> rate_limit)
    RateLimitedCached(RateLimitedLlm<CachedLlm<ApiLlm>>),
    /// Retry wrapping Cached (cache -> retry)
    RetryCached(RetryLlm<CachedLlm<ApiLlm>>),
    /// Retry wrapping Rate Limited (rate_limit -> retry)
    RetryRateLimited(RetryLlm<RateLimitedLlm<ApiLlm>>),

    // All three optimizations (cache -> rate_limit -> retry)
    /// Retry wrapping Rate Limited wrapping Cached
    RetryRateLimitedCached(RetryLlm<RateLimitedLlm<CachedLlm<ApiLlm>>>),
}

// Implement Llm trait for LlmVariant to forward calls to the wrapped LLM
impl kkachi::recursive::llm::Llm for LlmVariant {
    type GenerateFut<'a>
        = std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = kkachi::error::Result<kkachi::recursive::llm::LmOutput>,
                > + Send
                + 'a,
        >,
    >
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        match self {
            LlmVariant::Plain(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::Cached(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::RateLimited(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::Retry(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::RateLimitedCached(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::RetryCached(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::RetryRateLimited(llm) => Box::pin(llm.generate(prompt, context, feedback)),
            LlmVariant::RetryRateLimitedCached(llm) => {
                Box::pin(llm.generate(prompt, context, feedback))
            }
        }
    }

    fn model_name(&self) -> &str {
        match self {
            LlmVariant::Plain(llm) => llm.model_name(),
            LlmVariant::Cached(llm) => llm.model_name(),
            LlmVariant::RateLimited(llm) => llm.model_name(),
            LlmVariant::Retry(llm) => llm.model_name(),
            LlmVariant::RateLimitedCached(llm) => llm.model_name(),
            LlmVariant::RetryCached(llm) => llm.model_name(),
            LlmVariant::RetryRateLimited(llm) => llm.model_name(),
            LlmVariant::RetryRateLimitedCached(llm) => llm.model_name(),
        }
    }

    fn max_context(&self) -> usize {
        match self {
            LlmVariant::Plain(llm) => llm.max_context(),
            LlmVariant::Cached(llm) => llm.max_context(),
            LlmVariant::RateLimited(llm) => llm.max_context(),
            LlmVariant::Retry(llm) => llm.max_context(),
            LlmVariant::RateLimitedCached(llm) => llm.max_context(),
            LlmVariant::RetryCached(llm) => llm.max_context(),
            LlmVariant::RetryRateLimited(llm) => llm.max_context(),
            LlmVariant::RetryRateLimitedCached(llm) => llm.max_context(),
        }
    }
}

/// Real LLM API client supporting Anthropic, OpenAI, and Claude Code CLI.
///
/// Examples:
/// ```python
/// from kkachi import ApiLlm
///
/// # Auto-detect from environment variables
/// llm = ApiLlm.from_env()
///
/// # Explicit providers
/// llm = ApiLlm.anthropic("sk-ant-...", "claude-sonnet-4-20250514")
/// llm = ApiLlm.openai("sk-...", "gpt-4o")
/// llm = ApiLlm.claude_code()
///
/// # Custom endpoints (for proxies, self-hosted, etc.)
/// llm = ApiLlm.anthropic_with_url("key", "model", "https://custom.api.com")
/// llm = ApiLlm.openai_with_url("key", "model", "https://custom.api.com")
///
/// # Optimization methods
/// llm = llm.with_cache(100).with_rate_limit(10.0).with_retry(3)
/// ```
#[pyclass(name = "ApiLlm")]
pub struct PyApiLlm {
    pub(crate) inner: LlmVariant,
}

#[pymethods]
impl PyApiLlm {
    /// Auto-detect LLM provider from environment variables.
    ///
    /// Checks in order:
    /// 1. ANTHROPIC_API_KEY → Anthropic Claude
    /// 2. OPENAI_API_KEY → OpenAI
    /// 3. claude binary in PATH → Claude Code CLI
    ///
    /// Override defaults with:
    /// - KKACHI_MODEL — model name (API providers only)
    /// - KKACHI_BASE_URL — endpoint URL (API providers only)
    ///
    /// Returns:
    ///     ApiLlm: Configured LLM client
    ///
    /// Raises:
    ///     RuntimeError: If no provider found
    ///
    /// Example:
    ///     ```python
    ///     import os
    ///     os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    ///     llm = ApiLlm.from_env()
    ///     ```
    #[staticmethod]
    fn from_env() -> PyResult<Self> {
        ApiLlm::from_env()
            .map(|inner| Self {
                inner: LlmVariant::Plain(inner),
            })
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create LLM from environment: {}", e))
            })
    }

    /// Create an Anthropic Claude API client.
    ///
    /// Args:
    ///     api_key (str): Anthropic API key (starts with "sk-ant-")
    ///     model (str): Model identifier (e.g., "claude-sonnet-4-20250514")
    ///
    /// Returns:
    ///     ApiLlm: Anthropic LLM client
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.anthropic("sk-ant-...", "claude-sonnet-4-20250514")
    ///     ```
    #[staticmethod]
    fn anthropic(api_key: String, model: String) -> Self {
        Self {
            inner: LlmVariant::Plain(ApiLlm::anthropic(api_key, model)),
        }
    }

    /// Create an Anthropic Claude API client with custom base URL.
    ///
    /// Useful for:
    /// - Proxy servers
    /// - Self-hosted Anthropic-compatible endpoints
    /// - Testing with mock servers
    ///
    /// Args:
    ///     api_key (str): Anthropic API key
    ///     model (str): Model identifier
    ///     base_url (str): Custom API endpoint (e.g., "https://proxy.example.com")
    ///
    /// Returns:
    ///     ApiLlm: Anthropic LLM client with custom endpoint
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.anthropic_with_url(
    ///         "sk-ant-...",
    ///         "claude-sonnet-4-20250514",
    ///         "https://my-proxy.com"
    ///     )
    ///     ```
    #[staticmethod]
    fn anthropic_with_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            inner: LlmVariant::Plain(ApiLlm::anthropic_with_url(api_key, model, base_url)),
        }
    }

    /// Create an OpenAI API client.
    ///
    /// Also works with OpenAI-compatible endpoints like:
    /// - Together.ai
    /// - Groq
    /// - Anyscale
    /// - vLLM servers
    ///
    /// Args:
    ///     api_key (str): OpenAI API key (starts with "sk-")
    ///     model (str): Model identifier (e.g., "gpt-4o", "gpt-4-turbo")
    ///
    /// Returns:
    ///     ApiLlm: OpenAI LLM client
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.openai("sk-...", "gpt-4o")
    ///     ```
    #[staticmethod]
    fn openai(api_key: String, model: String) -> Self {
        Self {
            inner: LlmVariant::Plain(ApiLlm::openai(api_key, model)),
        }
    }

    /// Create an OpenAI API client with custom base URL.
    ///
    /// Useful for:
    /// - OpenAI-compatible providers (Together, Groq, Anyscale)
    /// - Self-hosted vLLM, text-generation-inference, etc.
    /// - Proxy servers
    /// - Testing with mock servers
    ///
    /// Args:
    ///     api_key (str): API key for the endpoint
    ///     model (str): Model identifier
    ///     base_url (str): Custom API endpoint
    ///
    /// Returns:
    ///     ApiLlm: OpenAI-compatible LLM client
    ///
    /// Example:
    ///     ```python
    ///     # Together.ai
    ///     llm = ApiLlm.openai_with_url(
    ///         "your-together-key",
    ///         "meta-llama/Llama-3-70b-chat-hf",
    ///         "https://api.together.xyz"
    ///     )
    ///
    ///     # Groq
    ///     llm = ApiLlm.openai_with_url(
    ///         "your-groq-key",
    ///         "llama3-70b-8192",
    ///         "https://api.groq.com/openai"
    ///     )
    ///     ```
    #[staticmethod]
    fn openai_with_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            inner: LlmVariant::Plain(ApiLlm::openai_with_url(api_key, model, base_url)),
        }
    }

    /// Create a Claude Code CLI client.
    ///
    /// Uses the locally installed `claude` binary. No API key required.
    /// Falls back to searching common installation paths if not in PATH.
    ///
    /// Returns:
    ///     ApiLlm: Claude Code CLI client
    ///
    /// Raises:
    ///     RuntimeError: If claude binary not found
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.claude_code()
    ///     ```
    #[staticmethod]
    fn claude_code() -> Self {
        Self {
            inner: LlmVariant::Plain(ApiLlm::claude_code()),
        }
    }

    // Note: temperature() and max_tokens() configuration methods are not exposed
    // in Python bindings due to Rust ownership semantics. ApiLlm uses a consuming
    // builder pattern that doesn't translate well to Python.
    //
    // To configure temperature and max_tokens, use the environment variables:
    // - KKACHI_TEMPERATURE
    // - KKACHI_MAX_TOKENS
    //
    // Or modify the Rust crate to make ApiLlm cloneable, then uncomment below.

    /// Enable LRU caching for LLM responses.
    ///
    /// Caches identical (prompt, context, feedback) tuples to avoid redundant
    /// API calls. Uses an LRU (Least Recently Used) eviction strategy.
    ///
    /// Args:
    ///     capacity (int): Maximum number of responses to cache
    ///
    /// Returns:
    ///     ApiLlm: Self with caching enabled
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.from_env().with_cache(100)
    ///     # First call hits the API
    ///     response1 = llm.generate("Hello")
    ///     # Second identical call returns cached result
    ///     response2 = llm.generate("Hello")  # instant, no API call
    ///     ```
    fn with_cache(mut self_: PyRefMut<'_, Self>, capacity: usize) -> Py<Self> {
        use std::mem;
        let old_inner = mem::replace(&mut self_.inner, LlmVariant::Plain(ApiLlm::claude_code()));

        let new_inner = match old_inner {
            // Can only apply cache to Plain - cache should be applied first
            LlmVariant::Plain(llm) => LlmVariant::Cached(llm.with_cache(capacity)),
            // Already has cache or cache must be applied first - keep as-is
            other => other,
        };

        self_.inner = new_inner;
        self_.into()
    }

    /// Enable proactive rate limiting using a token bucket algorithm.
    ///
    /// Prevents 429 rate limit errors by pacing requests before they're sent.
    /// Uses a token bucket algorithm allowing controlled bursts.
    ///
    /// Args:
    ///     requests_per_second (float): Maximum sustained request rate
    ///
    /// Returns:
    ///     ApiLlm: Self with rate limiting enabled
    ///
    /// Example:
    ///     ```python
    ///     # Limit to 10 requests per second
    ///     llm = ApiLlm.from_env().with_rate_limit(10.0)
    ///
    ///     # These calls will be automatically paced
    ///     for i in range(20):
    ///         response = llm.generate(f"Question {i}")
    ///     ```
    fn with_rate_limit(mut self_: PyRefMut<'_, Self>, requests_per_second: f64) -> Py<Self> {
        use std::mem;
        let old_inner = mem::replace(&mut self_.inner, LlmVariant::Plain(ApiLlm::claude_code()));

        let new_inner = match old_inner {
            LlmVariant::Plain(llm) => {
                LlmVariant::RateLimited(llm.with_rate_limit(requests_per_second))
            }
            LlmVariant::Cached(llm) => {
                LlmVariant::RateLimitedCached(llm.with_rate_limit(requests_per_second))
            }
            // Already rate limited or must be applied before retry - keep as-is
            other => other,
        };

        self_.inner = new_inner;
        self_.into()
    }

    /// Enable automatic retry with exponential backoff.
    ///
    /// Automatically retries transient errors (rate limits, server errors,
    /// timeouts) with exponential backoff delays.
    ///
    /// Args:
    ///     max_retries (int): Maximum number of retry attempts
    ///
    /// Returns:
    ///     ApiLlm: Self with retry enabled
    ///
    /// Example:
    ///     ```python
    ///     # Retry up to 3 times on transient errors
    ///     llm = ApiLlm.from_env().with_retry(3)
    ///
    ///     # Automatically retries on 429, 500, 502, 503, timeouts
    ///     response = llm.generate("Your prompt")
    ///     ```
    fn with_retry(mut self_: PyRefMut<'_, Self>, max_retries: u32) -> Py<Self> {
        use std::mem;
        let old_inner = mem::replace(&mut self_.inner, LlmVariant::Plain(ApiLlm::claude_code()));

        let new_inner = match old_inner {
            LlmVariant::Plain(llm) => LlmVariant::Retry(llm.with_retry(max_retries)),
            LlmVariant::Cached(llm) => LlmVariant::RetryCached(llm.with_retry(max_retries)),
            LlmVariant::RateLimited(llm) => {
                LlmVariant::RetryRateLimited(llm.with_retry(max_retries))
            }
            LlmVariant::RateLimitedCached(llm) => {
                LlmVariant::RetryRateLimitedCached(llm.with_retry(max_retries))
            }
            // Already has retry - keep as-is (ignore new config)
            other => other,
        };

        self_.inner = new_inner;
        self_.into()
    }

    /// Get the model name being used.
    ///
    /// Returns:
    ///     str: Model identifier
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.anthropic("key", "claude-sonnet-4-20250514")
    ///     print(llm.model_name())  # "claude-sonnet-4-20250514"
    ///     ```
    fn model_name(&self) -> &str {
        use kkachi::recursive::llm::Llm;
        self.inner.model_name()
    }

    /// Get the maximum context length for this model.
    ///
    /// Returns:
    ///     int: Maximum context length in tokens
    ///
    /// Example:
    ///     ```python
    ///     llm = ApiLlm.anthropic("key", "claude-sonnet-4-20250514")
    ///     print(llm.max_context())  # 200000
    ///     ```
    fn max_context(&self) -> usize {
        use kkachi::recursive::llm::Llm;
        self.inner.max_context()
    }

    fn __repr__(&self) -> String {
        use kkachi::recursive::llm::Llm;
        format!(
            "ApiLlm(model='{}', max_context={})",
            self.inner.model_name(),
            self.inner.max_context()
        )
    }
}

impl PyApiLlm {
    /// Get a reference to the inner LLM variant
    pub fn inner_ref(&self) -> &LlmVariant {
        &self.inner
    }
}
