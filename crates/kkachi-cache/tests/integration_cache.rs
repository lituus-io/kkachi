// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for caching system

use kkachi_cache::*;
use std::borrow::Cow;

#[tokio::test]
async fn test_memory_cache_basic() {
    let cache = MemoryCache::new(100);

    let key = CacheKey::from_request("gpt-4", "test prompt", 0.0);
    let value = b"cached response".to_vec();

    cache.set(key.clone(), value.clone()).await.unwrap();

    let retrieved = cache.get(&key).await;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), value);
}

#[tokio::test]
async fn test_memory_cache_lru_eviction() {
    let cache = MemoryCache::new(2); // Small capacity

    let key1 = CacheKey::from_request("gpt-4", "prompt1", 0.0);
    let key2 = CacheKey::from_request("gpt-4", "prompt2", 0.0);
    let key3 = CacheKey::from_request("gpt-4", "prompt3", 0.0);

    cache.set(key1.clone(), b"value1".to_vec()).await.unwrap();
    cache.set(key2.clone(), b"value2".to_vec()).await.unwrap();
    cache.set(key3.clone(), b"value3".to_vec()).await.unwrap();

    // key1 should be evicted
    assert!(cache.get(&key1).await.is_none());
    assert!(cache.get(&key2).await.is_some());
    assert!(cache.get(&key3).await.is_some());
}

#[tokio::test]
async fn test_concurrent_cache() {
    let cache = memory::ConcurrentCache::new();

    let key = CacheKey::from_request("gpt-4", "concurrent test", 0.0);
    let value = b"test value".to_vec();

    cache.set(key.clone(), value.clone()).await.unwrap();

    let retrieved = cache.get(&key).await;
    assert_eq!(retrieved.unwrap(), value);
}

#[tokio::test]
async fn test_cache_contains() {
    let cache = MemoryCache::new(10);

    let key = CacheKey::from_request("gpt-4", "test", 0.0);

    assert!(!cache.contains(&key).await);

    cache.set(key.clone(), b"data".to_vec()).await.unwrap();

    assert!(cache.contains(&key).await);
}

#[tokio::test]
async fn test_cache_clear() {
    let cache = MemoryCache::new(10);

    let key1 = CacheKey::from_request("gpt-4", "test1", 0.0);
    let key2 = CacheKey::from_request("gpt-4", "test2", 0.0);

    cache.set(key1.clone(), b"data1".to_vec()).await.unwrap();
    cache.set(key2.clone(), b"data2".to_vec()).await.unwrap();

    cache.clear().await.unwrap();

    assert!(!cache.contains(&key1).await);
    assert!(!cache.contains(&key2).await);
}

#[tokio::test]
async fn test_disk_cache() {
    let temp_dir = std::env::temp_dir().join("kkachi_test_cache");
    let cache = DiskCache::new(temp_dir.clone()).await.unwrap();

    let key = CacheKey::from_request("gpt-4", "disk test", 0.5);
    let value = b"stored on disk".to_vec();

    cache.set(key.clone(), value.clone()).await.unwrap();

    let retrieved = cache.get(&key).await;
    assert_eq!(retrieved.unwrap(), value);

    // Clean up
    tokio::fs::remove_dir_all(temp_dir).await.ok();
}

#[test]
fn test_cache_key_generation() {
    let key1 = CacheKey::from_request("gpt-4", "test message", 0.0);
    let key2 = CacheKey::from_request("gpt-4", "test message", 0.0);
    let key3 = CacheKey::from_request("gpt-4", "different message", 0.0);

    // Same inputs = same key
    assert_eq!(key1, key2);

    // Different inputs = different key
    assert_ne!(key1, key3);
}

#[test]
fn test_cache_key_temperature_sensitivity() {
    let key1 = CacheKey::from_request("gpt-4", "test", 0.0);
    let key2 = CacheKey::from_request("gpt-4", "test", 1.0);

    // Different temperatures = different keys
    assert_ne!(key1, key2);
}

#[test]
fn test_cache_key_string_representation() {
    let key = CacheKey::from_request("gpt-4", "test", 0.5);
    let key_str = key.to_string();

    assert!(key_str.contains("gpt-4"));
    assert!(key_str.contains("500")); // 0.5 * 1000 = 500
}
