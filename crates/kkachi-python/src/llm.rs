// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for LLM implementations.
//!
//! Provides ApiLlm for real LLM API access (Anthropic, OpenAI, Claude Code CLI)
//! and support for custom endpoints.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kkachi::recursive::ApiLlm;

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
/// ```
#[pyclass(name = "ApiLlm")]
pub struct PyApiLlm {
    pub(crate) inner: ApiLlm,
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
            .map(|inner| Self { inner })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create LLM from environment: {}", e)))
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
            inner: ApiLlm::anthropic(api_key, model),
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
            inner: ApiLlm::anthropic_with_url(api_key, model, base_url),
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
            inner: ApiLlm::openai(api_key, model),
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
            inner: ApiLlm::openai_with_url(api_key, model, base_url),
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
            inner: ApiLlm::claude_code(),
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
    /// Get a reference to the inner Rust ApiLlm
    pub fn inner_ref(&self) -> &ApiLlm {
        &self.inner
    }
}
