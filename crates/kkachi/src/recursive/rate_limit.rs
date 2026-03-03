// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Proactive rate limiting for LLM calls.
//!
//! This module provides [`RateLimitedLlm`], a wrapper that proactively rate-limits
//! requests using a token bucket algorithm. Unlike [`RetryLlm`](super::retry::RetryLlm)
//! which reacts to 429 errors, this wrapper prevents them by pacing requests.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, RateLimitExt};
//!
//! let llm = MockLlm::new(|_, _| "response".to_string());
//! let limited = llm.with_rate_limit(10.0); // 10 requests/second
//! ```

use crate::error::Result;
use crate::recursive::llm::{Llm, LmOutput};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Configuration for token-bucket rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum sustained requests per second.
    pub requests_per_second: f64,
    /// Burst capacity — how many tokens the bucket holds.
    /// Allows short bursts above the steady-state rate.
    /// Defaults to 1 (strict pacing).
    pub burst: u32,
}

impl RateLimitConfig {
    /// Create a new config with the given RPS and burst=1.
    pub fn new(requests_per_second: f64) -> Self {
        Self {
            requests_per_second,
            burst: 1,
        }
    }

    /// Set the burst capacity.
    pub fn with_burst(mut self, burst: u32) -> Self {
        self.burst = burst;
        self
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 10.0,
            burst: 1,
        }
    }
}

/// Internal token bucket state.
#[derive(Debug)]
struct TokenBucketState {
    /// Current available tokens.
    tokens: f64,
    /// Maximum tokens (burst capacity).
    max_tokens: f64,
    /// Refill rate (tokens per second).
    refill_rate: f64,
    /// Last refill timestamp.
    last_refill: Instant,
}

impl TokenBucketState {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            tokens: config.burst as f64,
            max_tokens: config.burst as f64,
            refill_rate: config.requests_per_second,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time, then try to acquire one.
    /// Returns `Some(Duration)` if caller must wait, or `None` if acquired.
    fn try_acquire(&mut self) -> Option<Duration> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);

        // Refill tokens
        self.tokens += elapsed.as_secs_f64() * self.refill_rate;
        if self.tokens > self.max_tokens {
            self.tokens = self.max_tokens;
        }
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            None
        } else {
            let deficit = 1.0 - self.tokens;
            let wait_secs = deficit / self.refill_rate;
            Some(Duration::from_secs_f64(wait_secs))
        }
    }
}

/// Shared token bucket handle.
#[derive(Debug, Clone)]
struct TokenBucket {
    state: Arc<Mutex<TokenBucketState>>,
}

impl TokenBucket {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(TokenBucketState::new(config))),
        }
    }

    fn try_acquire(&self) -> Option<Duration> {
        self.state.lock().unwrap().try_acquire()
    }
}

/// An LLM wrapper that proactively rate-limits requests using a token bucket.
///
/// Before each `generate()` call, the wrapper checks if a token is available.
/// If not, it sleeps for the required duration before proceeding.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, RateLimitExt, RateLimitConfig};
///
/// let llm = MockLlm::new(|_, _| "response".to_string());
/// let limited = llm.with_rate_limit_config(
///     RateLimitConfig::new(5.0).with_burst(3)
/// );
/// ```
pub struct RateLimitedLlm<L: Llm> {
    inner: L,
    bucket: TokenBucket,
}

impl<L: Llm> RateLimitedLlm<L> {
    /// Create a new RateLimitedLlm with the given config.
    pub fn new(inner: L, config: RateLimitConfig) -> Self {
        Self {
            bucket: TokenBucket::new(&config),
            inner,
        }
    }

    /// Get a reference to the inner LLM.
    pub fn inner(&self) -> &L {
        &self.inner
    }
}

/// Internal state for the rate-limit future.
enum RateLimitState<'a, L: Llm + 'a> {
    /// Waiting for a token to become available.
    #[cfg(feature = "native")]
    WaitingForSlot(Pin<Box<tokio::time::Sleep>>),
    /// Waiting for a token (spin-wait fallback without tokio).
    #[cfg(not(feature = "native"))]
    WaitingForSlot(Instant, Duration),
    /// Token acquired, generating.
    Generating(Pin<Box<L::GenerateFut<'a>>>),
}

/// Future returned by `RateLimitedLlm::generate()`.
///
/// State machine: WaitingForSlot → Generating → Done.
/// If a token is immediately available, starts in Generating.
pub struct RateLimitFut<'a, L: Llm + 'a> {
    llm: &'a L,
    prompt: &'a str,
    context: &'a str,
    feedback: Option<&'a str>,
    bucket: TokenBucket,
    state: RateLimitState<'a, L>,
}

// SAFETY: All fields are Unpin:
// - References are Unpin
// - Pin<Box<T>> is always Unpin (Box is Unpin)
// - TokenBucket contains Arc (Unpin)
// - Instant, Duration: Copy types, Unpin
impl<'a, L: Llm + 'a> Unpin for RateLimitFut<'a, L> {}

impl<'a, L: Llm + 'a> Future for RateLimitFut<'a, L> {
    type Output = Result<LmOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        loop {
            match &mut this.state {
                #[cfg(feature = "native")]
                RateLimitState::WaitingForSlot(sleep) => match sleep.as_mut().poll(cx) {
                    Poll::Ready(()) => match this.bucket.try_acquire() {
                        None => {
                            let fut = this.llm.generate(this.prompt, this.context, this.feedback);
                            this.state = RateLimitState::Generating(Box::pin(fut));
                        }
                        Some(wait) => {
                            this.state =
                                RateLimitState::WaitingForSlot(Box::pin(tokio::time::sleep(wait)));
                        }
                    },
                    Poll::Pending => return Poll::Pending,
                },
                #[cfg(not(feature = "native"))]
                RateLimitState::WaitingForSlot(start, duration) => {
                    if start.elapsed() >= *duration {
                        match this.bucket.try_acquire() {
                            None => {
                                let fut =
                                    this.llm.generate(this.prompt, this.context, this.feedback);
                                this.state = RateLimitState::Generating(Box::pin(fut));
                            }
                            Some(wait) => {
                                *start = Instant::now();
                                *duration = wait;
                                cx.waker().wake_by_ref();
                                return Poll::Pending;
                            }
                        }
                    } else {
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                }
                RateLimitState::Generating(fut) => {
                    return fut.as_mut().poll(cx);
                }
            }
        }
    }
}

impl<L: Llm> Llm for RateLimitedLlm<L> {
    type GenerateFut<'a>
        = RateLimitFut<'a, L>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        match self.bucket.try_acquire() {
            None => {
                // Token available — go directly to Generating
                let fut = self.inner.generate(prompt, context, feedback);
                RateLimitFut {
                    llm: &self.inner,
                    prompt,
                    context,
                    feedback,
                    bucket: self.bucket.clone(),
                    state: RateLimitState::Generating(Box::pin(fut)),
                }
            }
            Some(wait) => {
                // Must wait
                #[cfg(feature = "native")]
                let state = RateLimitState::WaitingForSlot(Box::pin(tokio::time::sleep(wait)));
                #[cfg(not(feature = "native"))]
                let state = RateLimitState::WaitingForSlot(Instant::now(), wait);
                RateLimitFut {
                    llm: &self.inner,
                    prompt,
                    context,
                    feedback,
                    bucket: self.bucket.clone(),
                    state,
                }
            }
        }
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn max_context(&self) -> usize {
        self.inner.max_context()
    }
}

/// Extension trait for adding rate limiting to any Llm.
pub trait RateLimitExt: Llm + Sized {
    /// Wrap this LLM with a token-bucket rate limiter.
    ///
    /// `rps` is the sustained requests-per-second rate.
    /// Burst defaults to 1 (strict pacing).
    fn with_rate_limit(self, rps: f64) -> RateLimitedLlm<Self> {
        RateLimitedLlm::new(self, RateLimitConfig::new(rps))
    }

    /// Wrap this LLM with a rate limiter using a custom configuration.
    fn with_rate_limit_config(self, config: RateLimitConfig) -> RateLimitedLlm<Self> {
        RateLimitedLlm::new(self, config)
    }
}

impl<L: Llm> RateLimitExt for L {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;

    #[test]
    fn test_config_defaults() {
        let config = RateLimitConfig::default();
        assert!((config.requests_per_second - 10.0).abs() < f64::EPSILON);
        assert_eq!(config.burst, 1);
    }

    #[test]
    fn test_config_builder() {
        let config = RateLimitConfig::new(5.0).with_burst(10);
        assert!((config.requests_per_second - 5.0).abs() < f64::EPSILON);
        assert_eq!(config.burst, 10);
    }

    #[test]
    fn test_token_bucket_immediate_acquire() {
        let bucket = TokenBucket::new(&RateLimitConfig::new(10.0).with_burst(5));
        for _ in 0..5 {
            assert!(bucket.try_acquire().is_none());
        }
        // 6th should require waiting
        assert!(bucket.try_acquire().is_some());
    }

    #[test]
    fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(&RateLimitConfig::new(1000.0).with_burst(1));
        // Consume the one token
        assert!(bucket.try_acquire().is_none());
        // Next should require wait
        let wait = bucket.try_acquire();
        assert!(wait.is_some());
        // Wait should be approximately 1ms (1/1000s)
        let w = wait.unwrap();
        assert!(w < Duration::from_millis(5));
    }

    #[test]
    fn test_model_name_preserved() {
        let llm = MockLlm::new(|_, _| "ok".to_string());
        let limited = llm.with_rate_limit(10.0);
        assert_eq!(limited.model_name(), "mock");
    }

    #[test]
    fn test_inner_accessible() {
        let llm = MockLlm::new(|_, _| "ok".to_string());
        let limited = llm.with_rate_limit(10.0);
        assert_eq!(limited.inner().model_name(), "mock");
    }

    #[tokio::test]
    async fn test_rate_limit_allows_burst() {
        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_rate_limit_config(RateLimitConfig::new(10.0).with_burst(3));

        let start = Instant::now();
        for _ in 0..3 {
            llm.generate("test", "", None).await.unwrap();
        }
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_rate_limit_paces_after_burst() {
        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_rate_limit_config(RateLimitConfig::new(20.0).with_burst(1));

        // First call: immediate (uses burst token)
        llm.generate("test", "", None).await.unwrap();

        // Second call: should wait ~50ms (1/20s)
        let start = Instant::now();
        llm.generate("test", "", None).await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(30));
    }

    #[tokio::test]
    async fn test_rate_limit_composable_with_retry() {
        use crate::recursive::retry::LlmExt;

        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_rate_limit(10.0)
            .with_retry(3);

        let result = llm.generate("test", "", None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limit_composable_with_cache() {
        use crate::recursive::cache::CacheExt;

        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_cache(10)
            .with_rate_limit(10.0);

        let result = llm.generate("test", "", None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limit_full_chain() {
        use crate::recursive::cache::CacheExt;
        use crate::recursive::retry::LlmExt;

        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_cache(100)
            .with_rate_limit(10.0)
            .with_retry(3);

        let result = llm.generate("test", "", None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "ok");
    }
}
