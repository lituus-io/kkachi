// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for LLM implementations.
//!
//! Uses a config-based approach with lazy construction instead of the previous
//! 8-variant enum. Optimization wrappers (cache, rate limiting, retry) are stored
//! as config fields and the final `BoxedLlm` is built lazily on first use.
//!
//! Dynamic dispatch via `BoxedLlm` is acceptable at the Python FFI boundary
//! since GIL overhead dominates.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kkachi::recursive::llm::Llm;
use kkachi::recursive::{ApiLlm, BoxedLlm, CacheExt, LlmExt, RateLimitExt};

/// Configuration for LLM optimization wrappers.
///
/// Stored as plain data fields, built lazily into a `BoxedLlm` on first use.
struct LlmConfig {
    cache_capacity: Option<usize>,
    rate_limit_rps: Option<f64>,
    max_retries: Option<u32>,
    timeout_secs: Option<u64>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            cache_capacity: None,
            rate_limit_rps: None,
            max_retries: None,
            timeout_secs: None,
        }
    }
}

/// Real LLM API client supporting Anthropic, OpenAI, and Claude Code CLI.
///
/// Uses a config-based approach: `with_cache()`, `with_rate_limit()`, `with_retry()`
/// just set config fields. The actual LLM wrapper is built lazily on first use.
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
/// # Optimization methods (config-based, no cloning)
/// llm = llm.with_cache(100).with_rate_limit(10.0).with_retry(3)
/// ```
#[pyclass(name = "ApiLlm")]
pub struct PyApiLlm {
    /// The base ApiLlm, consumed on first build.
    base: Option<ApiLlm>,
    /// Display-only metadata, stored separately from the consumed base.
    model: String,
    max_ctx: usize,
    config: LlmConfig,
    /// Lazily built BoxedLlm wrapping base + optimizations.
    built: Option<BoxedLlm>,
}

impl PyApiLlm {
    fn new_plain(base: ApiLlm) -> Self {
        let model = base.model_name().to_string();
        let max_ctx = base.max_context();
        Self {
            base: Some(base),
            model,
            max_ctx,
            config: LlmConfig::default(),
            built: None,
        }
    }

    /// Build the BoxedLlm from base + config if not already built.
    ///
    /// Consumes the `ApiLlm` from `self.base` on first call.
    fn ensure_built(&mut self) {
        if self.built.is_some() {
            return;
        }

        let base = match self.base.take() {
            Some(b) => b,
            None => {
                // Base was already consumed but built was invalidated.
                // This shouldn't happen in normal usage since config changes
                // are only allowed before first use.
                return;
            }
        };

        // Build the LLM stack based on config
        // Order: base -> cache -> rate_limit -> retry (innermost to outermost)
        //
        // We use Box::leak to get 'static references for the wrapper chain.
        // This is acceptable because PyApiLlm lives for the duration of the
        // Python process (GC'd by Python).
        // BoxedLlm::new takes L: Llm by value and wraps in Arc internally.
        // We build the wrapper chain and pass the final type to BoxedLlm::new.
        match (
            self.config.cache_capacity,
            self.config.rate_limit_rps,
            self.config.max_retries,
        ) {
            (None, None, None) => {
                self.built = Some(BoxedLlm::new(base));
            }
            (Some(cap), None, None) => {
                self.built = Some(BoxedLlm::new(base.with_cache(cap)));
            }
            (None, Some(rps), None) => {
                self.built = Some(BoxedLlm::new(base.with_rate_limit(rps)));
            }
            (None, None, Some(retries)) => {
                self.built = Some(BoxedLlm::new(base.with_retry(retries)));
            }
            (Some(cap), Some(rps), None) => {
                self.built = Some(BoxedLlm::new(base.with_cache(cap).with_rate_limit(rps)));
            }
            (Some(cap), None, Some(retries)) => {
                self.built = Some(BoxedLlm::new(base.with_cache(cap).with_retry(retries)));
            }
            (None, Some(rps), Some(retries)) => {
                self.built = Some(BoxedLlm::new(base.with_rate_limit(rps).with_retry(retries)));
            }
            (Some(cap), Some(rps), Some(retries)) => {
                self.built = Some(BoxedLlm::new(
                    base.with_cache(cap)
                        .with_rate_limit(rps)
                        .with_retry(retries),
                ));
            }
        }
    }

    /// Get a reference to the built LLM for use by DSPy modules.
    pub fn get_llm(&mut self) -> &BoxedLlm {
        self.ensure_built();
        self.built.as_ref().unwrap()
    }
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
    #[staticmethod]
    fn from_env() -> PyResult<Self> {
        ApiLlm::from_env().map(Self::new_plain).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create LLM from environment: {}", e))
        })
    }

    /// Create an Anthropic Claude API client.
    #[staticmethod]
    fn anthropic(api_key: String, model: String) -> Self {
        Self::new_plain(ApiLlm::anthropic(api_key, model))
    }

    /// Create an Anthropic Claude API client with custom base URL.
    #[staticmethod]
    fn anthropic_with_url(api_key: String, model: String, base_url: String) -> Self {
        Self::new_plain(ApiLlm::anthropic_with_url(api_key, model, base_url))
    }

    /// Create an OpenAI API client.
    #[staticmethod]
    fn openai(api_key: String, model: String) -> Self {
        Self::new_plain(ApiLlm::openai(api_key, model))
    }

    /// Create an OpenAI API client with custom base URL.
    #[staticmethod]
    fn openai_with_url(api_key: String, model: String, base_url: String) -> Self {
        Self::new_plain(ApiLlm::openai_with_url(api_key, model, base_url))
    }

    /// Create a Claude Code CLI client.
    #[staticmethod]
    fn claude_code() -> Self {
        Self::new_plain(ApiLlm::claude_code())
    }

    /// Enable LRU caching for LLM responses.
    ///
    /// Caches identical (prompt, context, feedback) tuples to avoid redundant
    /// API calls. Uses an LRU (Least Recently Used) eviction strategy.
    fn with_cache(mut self_: PyRefMut<'_, Self>, capacity: usize) -> Py<Self> {
        self_.config.cache_capacity = Some(capacity);
        self_.built = None; // Invalidate built LLM
        self_.into()
    }

    /// Enable proactive rate limiting using a token bucket algorithm.
    ///
    /// Prevents 429 rate limit errors by pacing requests before they're sent.
    fn with_rate_limit(mut self_: PyRefMut<'_, Self>, requests_per_second: f64) -> Py<Self> {
        self_.config.rate_limit_rps = Some(requests_per_second);
        self_.built = None;
        self_.into()
    }

    /// Enable automatic retry with exponential backoff.
    ///
    /// Automatically retries transient errors (rate limits, server errors,
    /// timeouts) with exponential backoff delays.
    fn with_retry(mut self_: PyRefMut<'_, Self>, max_retries: u32) -> Py<Self> {
        self_.config.max_retries = Some(max_retries);
        self_.built = None;
        self_.into()
    }

    /// Set the HTTP request timeout in seconds (default: 300).
    ///
    /// Controls how long to wait for an LLM API response before timing out.
    /// Increase for large prompts that generate long responses.
    ///
    /// Example:
    ///     llm = ApiLlm.openai("key", "gpt-4o").with_timeout(600)  # 10 minutes
    fn with_timeout(mut self_: PyRefMut<'_, Self>, secs: u64) -> Py<Self> {
        self_.config.timeout_secs = Some(secs);
        // Rebuild the base ApiLlm with new timeout
        if let Some(base) = self_.base.take() {
            self_.base = Some(base.timeout(secs));
        }
        self_.built = None;
        self_.into()
    }

    /// Get the model name being used.
    fn model_name(&self) -> &str {
        &self.model
    }

    /// Get the maximum context length for this model.
    fn max_context(&self) -> usize {
        self.max_ctx
    }

    /// Call the LLM directly: `llm(prompt, feedback=None) -> str`.
    ///
    /// This makes `ApiLlm` satisfy the `Callable[[str, Optional[str]], str]`
    /// protocol so it can be passed directly to `reason(llm, prompt)` without
    /// wrapping in a lambda.
    #[pyo3(signature = (prompt, feedback=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        prompt: String,
        feedback: Option<String>,
    ) -> PyResult<String> {
        self.ensure_built();
        let llm = self
            .built
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("LLM not initialized"))?;
        // Use block_on to run the future with a proper Tokio runtime so that
        // retry (tokio::time::sleep) and rate-limit wrappers work correctly.
        py.allow_threads(|| {
            let result =
                kkachi::recursive::block_on(llm.generate(&prompt, "", feedback.as_deref()));
            match result {
                Ok(output) => Ok(output.text),
                Err(e) => Err(PyRuntimeError::new_err(format!("{e}"))),
            }
        })
    }

    fn __repr__(&self) -> String {
        let mut opts = Vec::new();
        if let Some(cap) = self.config.cache_capacity {
            opts.push(format!("cache={}", cap));
        }
        if let Some(rps) = self.config.rate_limit_rps {
            opts.push(format!("rate_limit={:.1}", rps));
        }
        if let Some(retries) = self.config.max_retries {
            opts.push(format!("retry={}", retries));
        }
        if let Some(secs) = self.config.timeout_secs {
            opts.push(format!("timeout={}s", secs));
        }
        let opts_str = if opts.is_empty() {
            String::new()
        } else {
            format!(", {}", opts.join(", "))
        };
        format!(
            "ApiLlm(model='{}', max_context={}{})",
            self.model, self.max_ctx, opts_str
        )
    }
}
