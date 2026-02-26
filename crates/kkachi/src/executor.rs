// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Hybrid Executor
//!
//! Combines Rayon (CPU-bound parallelism) with Tokio (async I/O)
//! for optimal performance in LM optimization pipelines.
//!
//! ## Architecture
//!
//! - **Tokio**: Handles async I/O (LM API calls, network)
//! - **Rayon**: Handles CPU-bound tasks (metric computation, parsing)
//! - **BufferPool**: Lock-free buffer pool for zero-copy response handling
//!
//! ## Usage
//!
//! ```ignore
//! let executor = HybridExecutor::new(4, 8);
//!
//! // Run batch predictions (I/O) then evaluate (CPU)
//! let results = executor.evaluate_batch(&module, &examples, &metric).await;
//! ```

use crate::error::Result;
use crossbeam::queue::ArrayQueue;
use rayon::prelude::*;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Lock-free buffer pool for zero-copy response handling.
///
/// Provides reusable buffers to avoid allocation during high-throughput
/// LM operations.
pub struct BufferPool {
    /// Pool of available buffers
    buffers: ArrayQueue<Vec<u8>>,
    /// Default buffer capacity
    capacity: usize,
}

impl BufferPool {
    /// Create a new buffer pool.
    pub fn new(pool_size: usize, buffer_capacity: usize) -> Self {
        let buffers = ArrayQueue::new(pool_size);

        // Pre-allocate buffers
        for _ in 0..pool_size {
            let _ = buffers.push(Vec::with_capacity(buffer_capacity));
        }

        Self {
            buffers,
            capacity: buffer_capacity,
        }
    }

    /// Create with default settings (32 buffers, 16KB each).
    pub fn default() -> Self {
        Self::new(32, 16 * 1024)
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns a pooled buffer if available, otherwise allocates a new one.
    #[inline]
    pub fn acquire(&self) -> Vec<u8> {
        self.buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    /// Release a buffer back to the pool.
    ///
    /// The buffer is cleared and returned to the pool for reuse.
    #[inline]
    pub fn release(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        // If pool is full, buffer is simply dropped
        let _ = self.buffers.push(buffer);
    }

    /// Get a scoped buffer that auto-returns on drop.
    pub fn scoped(&self) -> ScopedBuffer<'_> {
        ScopedBuffer {
            buffer: Some(self.acquire()),
            pool: self,
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            available: self.buffers.len(),
            capacity: self.buffers.capacity(),
            buffer_size: self.capacity,
        }
    }
}

/// Buffer pool statistics.
#[derive(Debug, Clone, Copy)]
pub struct BufferPoolStats {
    /// Number of available buffers.
    pub available: usize,
    /// Total pool capacity.
    pub capacity: usize,
    /// Size of each buffer.
    pub buffer_size: usize,
}

/// A buffer that automatically returns to the pool on drop.
pub struct ScopedBuffer<'a> {
    buffer: Option<Vec<u8>>,
    pool: &'a BufferPool,
}

impl<'a> ScopedBuffer<'a> {
    /// Get mutable access to the buffer.
    pub fn buffer(&mut self) -> &mut Vec<u8> {
        self.buffer.as_mut().unwrap()
    }

    /// Take ownership of the buffer (won't return to pool).
    pub fn take(mut self) -> Vec<u8> {
        self.buffer.take().unwrap()
    }
}

impl<'a> Drop for ScopedBuffer<'a> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

impl<'a> std::ops::Deref for ScopedBuffer<'a> {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl<'a> std::ops::DerefMut for ScopedBuffer<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().unwrap()
    }
}

/// Hybrid executor combining Rayon and Tokio.
///
/// Provides optimal performance for LM optimization by using:
/// - Tokio for concurrent I/O (API calls)
/// - Rayon for parallel CPU work (metrics, parsing)
pub struct HybridExecutor {
    /// Rayon thread pool for CPU-bound work
    rayon_pool: rayon::ThreadPool,
    /// Semaphore for LM concurrency control
    lm_semaphore: Arc<Semaphore>,
    /// Buffer pool for response handling
    buffer_pool: Arc<BufferPool>,
    /// Maximum concurrent LM calls
    max_lm_concurrency: usize,
}

/// Configuration for the hybrid executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Number of Rayon threads (0 = auto)
    pub rayon_threads: usize,
    /// Maximum concurrent LM API calls
    pub max_lm_concurrency: usize,
    /// Buffer pool size
    pub buffer_pool_size: usize,
    /// Buffer capacity (bytes)
    pub buffer_capacity: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            rayon_threads: 0, // Auto-detect
            max_lm_concurrency: 8,
            buffer_pool_size: 32,
            buffer_capacity: 16 * 1024,
        }
    }
}

impl ExecutorConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Rayon thread count.
    pub fn rayon_threads(mut self, threads: usize) -> Self {
        self.rayon_threads = threads;
        self
    }

    /// Set maximum LM concurrency.
    pub fn max_lm_concurrency(mut self, max: usize) -> Self {
        self.max_lm_concurrency = max;
        self
    }

    /// Set buffer pool size.
    pub fn buffer_pool_size(mut self, size: usize) -> Self {
        self.buffer_pool_size = size;
        self
    }

    /// Set buffer capacity (bytes per buffer).
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }
}

impl HybridExecutor {
    /// Create a new hybrid executor with default settings.
    pub fn new() -> Result<Self> {
        Self::with_config(ExecutorConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: ExecutorConfig) -> Result<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if config.rayon_threads > 0 {
            builder = builder.num_threads(config.rayon_threads);
        }

        let rayon_pool = builder
            .build()
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;

        Ok(Self {
            rayon_pool,
            lm_semaphore: Arc::new(Semaphore::new(config.max_lm_concurrency)),
            buffer_pool: Arc::new(BufferPool::new(
                config.buffer_pool_size,
                config.buffer_capacity,
            )),
            max_lm_concurrency: config.max_lm_concurrency,
        })
    }

    /// Get the buffer pool.
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Get the LM semaphore for concurrency control.
    pub fn lm_semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(&self.lm_semaphore)
    }

    /// Run CPU-bound work on the Rayon thread pool.
    pub fn run_cpu<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.rayon_pool.install(f)
    }

    /// Score predictions using Rayon parallelism.
    pub fn score_parallel<'a, F>(&self, predictions: &'a [String], f: F) -> Vec<f64>
    where
        F: Fn(&str) -> f64 + Sync,
    {
        self.rayon_pool
            .install(|| predictions.par_iter().map(|p| f(p)).collect())
    }

    /// Acquire LM permit for rate limiting.
    pub async fn acquire_lm_permit(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.lm_semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore closed")
    }

    /// Get executor statistics.
    pub fn stats(&self) -> ExecutorStats {
        ExecutorStats {
            rayon_threads: self.rayon_pool.current_num_threads(),
            lm_concurrency_max: self.max_lm_concurrency,
            lm_concurrency_available: self.lm_semaphore.available_permits(),
            buffer_pool: self.buffer_pool.stats(),
        }
    }
}

impl Default for HybridExecutor {
    fn default() -> Self {
        Self::new().expect("failed to create executor")
    }
}

/// Executor statistics.
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    /// Number of Rayon threads.
    pub rayon_threads: usize,
    /// Maximum LM concurrency.
    pub lm_concurrency_max: usize,
    /// Available LM permits.
    pub lm_concurrency_available: usize,
    /// Buffer pool statistics.
    pub buffer_pool: BufferPoolStats,
}

/// Batch runner that processes items with controlled concurrency.
pub struct BatchRunner<'a> {
    executor: &'a HybridExecutor,
}

impl<'a> BatchRunner<'a> {
    /// Create a new batch runner.
    pub fn new(executor: &'a HybridExecutor) -> Self {
        Self { executor }
    }

    /// Get the executor reference.
    pub fn executor(&self) -> &'a HybridExecutor {
        self.executor
    }

    /// Process items with CPU parallelism.
    pub fn process_parallel<T, F, R>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Sync,
        F: Fn(&T) -> R + Sync + Send,
        R: Send,
    {
        self.executor.run_cpu(|| items.par_iter().map(f).collect())
    }

    /// Score predictions in parallel.
    pub fn score_batch<F>(&self, predictions: &[String], scorer: F) -> Vec<f64>
    where
        F: Fn(&str) -> f64 + Sync,
    {
        self.executor.score_parallel(predictions, scorer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_create() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.stats().capacity, 4);
        assert_eq!(pool.stats().buffer_size, 1024);
    }

    #[test]
    fn test_buffer_pool_acquire_release() {
        let pool = BufferPool::new(2, 1024);

        // Acquire both buffers
        let mut buf1 = pool.acquire();
        let mut buf2 = pool.acquire();

        // Modify buffers
        buf1.extend_from_slice(b"hello");
        buf2.extend_from_slice(b"world");

        // Release
        pool.release(buf1);
        pool.release(buf2);

        // Acquire again - should be cleared
        let buf3 = pool.acquire();
        assert!(buf3.is_empty());
    }

    #[test]
    fn test_buffer_pool_scoped() {
        let pool = BufferPool::new(2, 1024);

        {
            let mut scoped = pool.scoped();
            scoped.extend_from_slice(b"test");
            assert_eq!(scoped.len(), 4);
        }
        // Buffer returned to pool on drop

        assert_eq!(pool.stats().available, 2);
    }

    #[test]
    fn test_executor_config() {
        let config = ExecutorConfig::new()
            .rayon_threads(4)
            .max_lm_concurrency(16)
            .buffer_pool_size(64);

        assert_eq!(config.rayon_threads, 4);
        assert_eq!(config.max_lm_concurrency, 16);
        assert_eq!(config.buffer_pool_size, 64);
    }

    #[test]
    fn test_executor_create() {
        let executor = HybridExecutor::new().unwrap();
        let stats = executor.stats();

        assert!(stats.rayon_threads > 0);
        assert_eq!(stats.lm_concurrency_max, 8); // default
    }

    #[test]
    fn test_executor_run_cpu() {
        let executor = HybridExecutor::new().unwrap();

        let result = executor.run_cpu(|| (0..100).map(|i| i * 2).sum::<i32>());

        assert_eq!(result, 9900);
    }

    #[test]
    fn test_executor_score_parallel() {
        let executor = HybridExecutor::new().unwrap();

        let predictions: Vec<String> = (0..100).map(|i| format!("prediction {}", i)).collect();

        let scores = executor.score_parallel(&predictions, |p| p.len() as f64 / 20.0);

        assert_eq!(scores.len(), 100);
        assert!(scores.iter().all(|&s| s > 0.0));
    }

    #[tokio::test]
    async fn test_executor_acquire_permit() {
        let executor =
            HybridExecutor::with_config(ExecutorConfig::new().max_lm_concurrency(2)).unwrap();

        let permit1 = executor.acquire_lm_permit().await;
        assert_eq!(executor.stats().lm_concurrency_available, 1);

        let permit2 = executor.acquire_lm_permit().await;
        assert_eq!(executor.stats().lm_concurrency_available, 0);

        drop(permit1);
        assert_eq!(executor.stats().lm_concurrency_available, 1);

        drop(permit2);
        assert_eq!(executor.stats().lm_concurrency_available, 2);
    }
}
