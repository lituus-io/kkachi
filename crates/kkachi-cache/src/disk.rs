// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Disk-based cache using bincode serialization

use crate::{Cache, CacheKey};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Disk-based cache
pub struct DiskCache {
    cache_dir: PathBuf,
}

impl DiskCache {
    /// Create a new disk cache
    pub async fn new(cache_dir: PathBuf) -> anyhow::Result<Self> {
        fs::create_dir_all(&cache_dir).await?;
        Ok(Self { cache_dir })
    }

    /// Get file path for a cache key
    fn key_path(&self, key: &CacheKey) -> PathBuf {
        self.cache_dir.join(format!("{}.bin", key.to_string()))
    }
}

#[async_trait]
impl Cache for DiskCache {
    async fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let path = self.key_path(key);
        if !path.exists() {
            return None;
        }

        let mut file = fs::File::open(path).await.ok()?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await.ok()?;
        Some(data)
    }

    async fn set(&self, key: CacheKey, value: Vec<u8>) -> anyhow::Result<()> {
        let path = self.key_path(&key);
        let mut file = fs::File::create(path).await?;
        file.write_all(&value).await?;
        file.sync_all().await?;
        Ok(())
    }

    async fn contains(&self, key: &CacheKey) -> bool {
        self.key_path(key).exists()
    }

    async fn clear(&self) -> anyhow::Result<()> {
        let mut entries = fs::read_dir(&self.cache_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("bin") {
                fs::remove_file(entry.path()).await?;
            }
        }
        Ok(())
    }
}
