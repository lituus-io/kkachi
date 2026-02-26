// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! LRU caching layer for LLM calls.
//!
//! This module provides [`CachedLlm`], a wrapper that caches LLM responses
//! using an LRU eviction strategy. Identical prompts return cached results
//! without making additional API calls.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, CacheExt};
//!
//! let llm = MockLlm::new(|_, _| "response".to_string());
//! let cached = llm.with_cache(100);
//! ```

use crate::error::Result;
use crate::recursive::llm::{Llm, LmOutput};
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{ready, Context, Poll};

/// An LLM wrapper that caches responses using an LRU cache.
///
/// Identical (prompt, context, feedback) tuples return the cached
/// response without calling the underlying LLM again.
pub struct CachedLlm<L: Llm> {
    inner: L,
    cache: Mutex<LruCache<u64, LmOutput>>,
}

impl<L: Llm> CachedLlm<L> {
    /// Create a new CachedLlm with the given capacity.
    pub fn new(inner: L, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap())),
        }
    }

    /// Get the number of cached entries.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }
}

/// Future returned by `CachedLlm::generate()`.
///
/// Either returns a cached result immediately (Hit) or awaits the
/// inner LLM's future and caches the result on success (Miss).
pub enum CachedFut<'a, L: Llm + 'a> {
    /// Cache hit — result is ready immediately.
    Hit(Option<Result<LmOutput>>),
    /// Cache miss — awaiting inner LLM future.
    Miss {
        /// The inner LLM future being awaited.
        inner: Pin<Box<L::GenerateFut<'a>>>,
        /// Reference to the shared cache for storing the result.
        cache: &'a Mutex<LruCache<u64, LmOutput>>,
        /// The cache key for this request.
        key: u64,
    },
}

// SAFETY: All fields are Unpin:
// - Option<Result<LmOutput>>: Unpin
// - Pin<Box<T>>: always Unpin (Box is Unpin)
// - &'a Mutex<...>: Unpin
// - u64: Unpin
impl<'a, L: Llm + 'a> Unpin for CachedFut<'a, L> {}

impl<'a, L: Llm + 'a> Future for CachedFut<'a, L> {
    type Output = Result<LmOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this {
            CachedFut::Hit(result) => Poll::Ready(
                result
                    .take()
                    .expect("CachedFut::Hit polled after completion"),
            ),
            CachedFut::Miss { inner, cache, key } => {
                let output = ready!(inner.as_mut().poll(cx));
                if let Ok(ref lm_out) = output {
                    cache.lock().unwrap().put(*key, lm_out.clone());
                }
                Poll::Ready(output)
            }
        }
    }
}

impl<L: Llm> Llm for CachedLlm<L> {
    type GenerateFut<'a>
        = CachedFut<'a, L>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        let key = cache_key(prompt, context, feedback);

        // Check cache
        if let Some(cached) = self.cache.lock().unwrap().get(&key) {
            return CachedFut::Hit(Some(Ok(cached.clone())));
        }

        // Cache miss - return future that will cache on completion
        CachedFut::Miss {
            inner: Box::pin(self.inner.generate(prompt, context, feedback)),
            cache: &self.cache,
            key,
        }
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn max_context(&self) -> usize {
        self.inner.max_context()
    }
}

/// Extension trait for adding caching to any Llm.
pub trait CacheExt: Llm + Sized {
    /// Wrap this LLM with an LRU cache of the given capacity.
    ///
    /// Identical prompts will return cached results without calling
    /// the underlying LLM again.
    fn with_cache(self, capacity: usize) -> CachedLlm<Self> {
        CachedLlm::new(self, capacity)
    }
}

impl<L: Llm> CacheExt for L {}

/// Compute a cache key from prompt, context, and feedback.
fn cache_key(prompt: &str, context: &str, feedback: Option<&str>) -> u64 {
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    context.hash(&mut hasher);
    feedback.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_cache_hit() {
        let call_count = AtomicU32::new(0);
        let llm = MockLlm::new(move |_, _| {
            call_count.fetch_add(1, Ordering::SeqCst);
            "response".to_string()
        })
        .with_cache(10);

        // First call - cache miss
        let r1 = llm.generate("hello", "", None).await.unwrap();
        assert_eq!(r1.text, "response");
        assert_eq!(llm.cache_len(), 1);

        // Second call - cache hit
        let r2 = llm.generate("hello", "", None).await.unwrap();
        assert_eq!(r2.text, "response");
        assert_eq!(llm.cache_len(), 1);
    }

    #[tokio::test]
    async fn test_cache_different_prompts() {
        let llm = MockLlm::new(|prompt, _| format!("reply to: {}", prompt)).with_cache(10);

        let r1 = llm.generate("a", "", None).await.unwrap();
        let r2 = llm.generate("b", "", None).await.unwrap();

        assert_eq!(r1.text, "reply to: a");
        assert_eq!(r2.text, "reply to: b");
        assert_eq!(llm.cache_len(), 2);
    }

    #[tokio::test]
    async fn test_cache_with_feedback() {
        let llm = MockLlm::new(|_, fb| {
            fb.map(|f| format!("with: {}", f))
                .unwrap_or("no feedback".to_string())
        })
        .with_cache(10);

        let r1 = llm.generate("p", "", None).await.unwrap();
        let r2 = llm.generate("p", "", Some("improve")).await.unwrap();

        assert_eq!(r1.text, "no feedback");
        assert_eq!(r2.text, "with: improve");
        assert_eq!(llm.cache_len(), 2); // Different keys
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let llm = MockLlm::new(|prompt, _| prompt.to_string()).with_cache(2);

        llm.generate("a", "", None).await.unwrap();
        llm.generate("b", "", None).await.unwrap();
        assert_eq!(llm.cache_len(), 2);

        // This should evict "a"
        llm.generate("c", "", None).await.unwrap();
        assert_eq!(llm.cache_len(), 2);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let llm = MockLlm::new(|_, _| "ok".to_string()).with_cache(10);

        llm.generate("a", "", None).await.unwrap();
        llm.generate("b", "", None).await.unwrap();
        assert_eq!(llm.cache_len(), 2);

        llm.clear_cache();
        assert_eq!(llm.cache_len(), 0);
    }

    #[test]
    fn test_model_name_preserved() {
        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_name("gpt-4")
            .with_cache(10);

        assert_eq!(llm.model_name(), "gpt-4");
    }

    #[test]
    fn test_cache_composable_with_retry() {
        use crate::recursive::retry::LlmExt;

        let llm = MockLlm::new(|_, _| "ok".to_string())
            .with_cache(10)
            .with_retry(3);

        // Just verify it compiles and the type works
        assert_eq!(llm.model_name(), "mock");
    }
}
