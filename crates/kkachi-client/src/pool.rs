// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Connection pooling for LM clients

use std::sync::Arc;
use tokio::sync::Semaphore;

/// Connection pool for rate limiting and concurrency control
pub struct LMPool {
    /// Semaphore for controlling concurrent requests
    semaphore: Arc<Semaphore>,

    /// Maximum concurrent requests
    max_concurrent: usize,
}

impl LMPool {
    /// Create a new pool
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// Acquire a permit to make a request
    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("Semaphore closed")
    }

    /// Get max concurrent requests
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

impl Default for LMPool {
    fn default() -> Self {
        Self::new(10) // Default: 10 concurrent requests
    }
}
