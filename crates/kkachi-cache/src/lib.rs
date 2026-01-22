// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Caching layer for Kkachi

#![allow(clippy::should_implement_trait)]
#![allow(clippy::inherent_to_string)]

#[cfg(feature = "disk")]
pub mod disk;
pub mod key;
pub mod memory;

#[cfg(feature = "disk")]
pub use disk::DiskCache;
pub use key::CacheKey;
pub use memory::MemoryCache;

use async_trait::async_trait;

/// Cache trait
#[async_trait]
pub trait Cache: Send + Sync {
    /// Get a value from cache
    async fn get(&self, key: &CacheKey) -> Option<Vec<u8>>;

    /// Set a value in cache
    async fn set(&self, key: CacheKey, value: Vec<u8>) -> anyhow::Result<()>;

    /// Check if key exists
    async fn contains(&self, key: &CacheKey) -> bool;

    /// Clear the cache
    async fn clear(&self) -> anyhow::Result<()>;
}
