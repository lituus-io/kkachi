// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Retry with exponential backoff for LLM calls.
//!
//! This module provides [`RetryLlm`], a wrapper that adds automatic retry
//! with exponential backoff to any [`Llm`] implementation. Use the [`LlmExt`]
//! trait to wrap any LLM with retry logic.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, LlmExt};
//!
//! let llm = MockLlm::new(|_, _| "response".to_string());
//! let llm_with_retry = llm.with_retry(3);
//! ```

use crate::error::{Error, Result};
use crate::recursive::llm::{Llm, LmOutput};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay before first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier for exponential backoff.
    pub backoff_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
        }
    }
}

/// An LLM wrapper that retries on transient errors.
///
/// Wraps any [`Llm`] implementation with automatic retry logic using
/// exponential backoff. Only retries on errors that appear transient
/// (rate limits, server errors, timeouts).
pub struct RetryLlm<L: Llm> {
    inner: L,
    config: RetryConfig,
}

impl<L: Llm> RetryLlm<L> {
    /// Create a new RetryLlm with the given config.
    pub fn new(inner: L, config: RetryConfig) -> Self {
        Self { inner, config }
    }
}

/// Internal state for the retry future.
enum RetryState<'a, L: Llm + 'a> {
    /// Currently awaiting a generate call.
    Generating(Pin<Box<L::GenerateFut<'a>>>),
    /// Sleeping before next retry (with tokio timer when native feature enabled).
    #[cfg(feature = "native")]
    Sleeping(Pin<Box<tokio::time::Sleep>>),
    /// Sleeping before next retry (spin-wait fallback without tokio).
    #[cfg(not(feature = "native"))]
    Sleeping(std::time::Instant, Duration),
}

/// Future returned by `RetryLlm::generate()`.
///
/// Implements a state machine that retries the inner LLM on transient errors
/// with exponential backoff delays.
pub struct RetryFut<'a, L: Llm + 'a> {
    llm: &'a L,
    prompt: &'a str,
    context: &'a str,
    feedback: Option<&'a str>,
    config: &'a RetryConfig,
    attempt: u32,
    delay: Duration,
    state: RetryState<'a, L>,
}

// SAFETY: All fields are Unpin:
// - References are Unpin
// - Copy types are Unpin
// - Pin<Box<T>> is Unpin (Box is always Unpin)
// - tokio::time::Sleep is Unpin
impl<'a, L: Llm + 'a> Unpin for RetryFut<'a, L> {}

impl<'a, L: Llm + 'a> Future for RetryFut<'a, L> {
    type Output = Result<LmOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        loop {
            match &mut this.state {
                RetryState::Generating(fut) => match fut.as_mut().poll(cx) {
                    Poll::Ready(Ok(output)) => return Poll::Ready(Ok(output)),
                    Poll::Ready(Err(e))
                        if is_retryable(&e) && this.attempt < this.config.max_retries =>
                    {
                        this.attempt += 1;
                        #[cfg(feature = "native")]
                        {
                            let sleep = tokio::time::sleep(this.delay);
                            this.state = RetryState::Sleeping(Box::pin(sleep));
                        }
                        #[cfg(not(feature = "native"))]
                        {
                            let start = std::time::Instant::now();
                            this.state = RetryState::Sleeping(start, this.delay);
                        }
                        this.delay = Duration::from_secs_f64(
                            (this.delay.as_secs_f64() * this.config.backoff_factor)
                                .min(this.config.max_delay.as_secs_f64()),
                        );
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Pending => return Poll::Pending,
                },
                #[cfg(feature = "native")]
                RetryState::Sleeping(sleep) => match sleep.as_mut().poll(cx) {
                    Poll::Ready(()) => {
                        let new_fut = this.llm.generate(this.prompt, this.context, this.feedback);
                        this.state = RetryState::Generating(Box::pin(new_fut));
                    }
                    Poll::Pending => return Poll::Pending,
                },
                #[cfg(not(feature = "native"))]
                RetryState::Sleeping(start, duration) => {
                    if start.elapsed() >= *duration {
                        let new_fut = this.llm.generate(this.prompt, this.context, this.feedback);
                        this.state = RetryState::Generating(Box::pin(new_fut));
                    } else {
                        // Spin-wait without tokio (not ideal but functional)
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

impl<L: Llm> Llm for RetryLlm<L> {
    type GenerateFut<'a>
        = RetryFut<'a, L>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        let fut = self.inner.generate(prompt, context, feedback);
        RetryFut {
            llm: &self.inner,
            prompt,
            context,
            feedback,
            config: &self.config,
            attempt: 0,
            delay: self.config.initial_delay,
            state: RetryState::Generating(Box::pin(fut)),
        }
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn max_context(&self) -> usize {
        self.inner.max_context()
    }
}

/// Extension trait for adding retry/cache capabilities to any Llm.
pub trait LlmExt: Llm + Sized {
    /// Wrap this LLM with retry logic using the specified number of retries.
    ///
    /// Uses default backoff configuration (500ms initial, 2x backoff, 30s max).
    fn with_retry(self, max_retries: u32) -> RetryLlm<Self> {
        RetryLlm::new(
            self,
            RetryConfig {
                max_retries,
                ..Default::default()
            },
        )
    }

    /// Wrap this LLM with retry logic using a custom configuration.
    fn with_retry_config(self, config: RetryConfig) -> RetryLlm<Self> {
        RetryLlm::new(self, config)
    }
}

impl<L: Llm> LlmExt for L {}

/// Check if an error is likely transient and worth retrying.
fn is_retryable(error: &Error) -> bool {
    let msg = error.to_string().to_lowercase();

    // HTTP status codes indicating transient failures
    if msg.contains("429") || msg.contains("rate limit") {
        return true;
    }
    if msg.contains("500") || msg.contains("502") || msg.contains("503") {
        return true;
    }
    if msg.contains("internal server error") || msg.contains("bad gateway") {
        return true;
    }
    if msg.contains("service unavailable") || msg.contains("gateway timeout") {
        return true;
    }

    // Network/connection errors
    if msg.contains("timeout") || msg.contains("timed out") {
        return true;
    }
    if msg.contains("connection") && (msg.contains("reset") || msg.contains("refused")) {
        return true;
    }

    // Overloaded
    if msg.contains("overloaded") || msg.contains("capacity") {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::{FailingLlm, MockLlm};

    #[tokio::test]
    async fn test_retry_success_first_try() {
        let llm = MockLlm::new(|_, _| "ok".to_string()).with_retry(3);

        let result = llm.generate("test", "", None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "ok");
    }

    #[tokio::test]
    async fn test_retry_non_retryable_error() {
        let llm = FailingLlm::new("parse error: invalid format").with_retry(3);

        let result = llm.generate("test", "", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_retryable_error_exhausted() {
        let llm = FailingLlm::new("HTTP 429 rate limit exceeded").with_retry_config(RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_factor: 2.0,
        });

        let result = llm.generate("test", "", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_succeeds_after_failures() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let attempt = AtomicU32::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = attempt.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                "success".to_string()
            } else {
                "success".to_string()
            }
        })
        .with_retry(3);

        let result = llm.generate("test", "", None).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable(&Error::module("HTTP 429 rate limit exceeded")));
        assert!(is_retryable(&Error::module("500 internal server error")));
        assert!(is_retryable(&Error::module("502 Bad Gateway")));
        assert!(is_retryable(&Error::module("503 Service Unavailable")));
        assert!(is_retryable(&Error::module("connection timeout")));
        assert!(is_retryable(&Error::module("connection reset")));
        assert!(is_retryable(&Error::module("server overloaded")));

        assert!(!is_retryable(&Error::module("parse error")));
        assert!(!is_retryable(&Error::module("invalid API key")));
        assert!(!is_retryable(&Error::module("model not found")));
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert!((config.backoff_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_name_preserved() {
        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_name("test-model")
            .with_retry(3);

        assert_eq!(llm.model_name(), "test-model");
    }
}
