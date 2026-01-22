// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! In-memory LRU cache

use crate::{Cache, CacheKey};
use async_trait::async_trait;
use dashmap::DashMap;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::Mutex;

/// In-memory LRU cache
pub struct MemoryCache {
    cache: Arc<Mutex<LruCache<CacheKey, Vec<u8>>>>,
}

impl MemoryCache {
    /// Create a new memory cache with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("Capacity must be > 0"),
            ))),
        }
    }

    /// Create with default capacity (1000 items)
    pub fn default() -> Self {
        Self::new(1000)
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let mut cache = self.cache.lock().await;
        cache.get(key).cloned()
    }

    async fn set(&self, key: CacheKey, value: Vec<u8>) -> anyhow::Result<()> {
        let mut cache = self.cache.lock().await;
        cache.put(key, value);
        Ok(())
    }

    async fn contains(&self, key: &CacheKey) -> bool {
        let cache = self.cache.lock().await;
        cache.contains(key)
    }

    async fn clear(&self) -> anyhow::Result<()> {
        let mut cache = self.cache.lock().await;
        cache.clear();
        Ok(())
    }
}

/// Thread-safe concurrent cache using DashMap (no LRU)
pub struct ConcurrentCache {
    cache: DashMap<CacheKey, Vec<u8>>,
}

impl ConcurrentCache {
    /// Create a new concurrent cache
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }
}

impl Default for ConcurrentCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Cache for ConcurrentCache {
    async fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        self.cache.get(key).map(|v| v.clone())
    }

    async fn set(&self, key: CacheKey, value: Vec<u8>) -> anyhow::Result<()> {
        self.cache.insert(key, value);
        Ok(())
    }

    async fn contains(&self, key: &CacheKey) -> bool {
        self.cache.contains_key(key)
    }

    async fn clear(&self) -> anyhow::Result<()> {
        self.cache.clear();
        Ok(())
    }
}
