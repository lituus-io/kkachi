// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for hybrid executor
//!
//! Tests the buffer pool, hybrid executor, and batch processing.

use kkachi::*;

#[test]
fn test_buffer_pool_basic() {
    let pool = BufferPool::new(4, 1024);

    // Acquire buffers
    let mut buf1 = pool.acquire();
    let mut buf2 = pool.acquire();

    // Write to buffers
    buf1.extend_from_slice(b"hello");
    buf2.extend_from_slice(b"world");

    assert_eq!(&buf1[..], b"hello");
    assert_eq!(&buf2[..], b"world");

    // Release buffers
    pool.release(buf1);
    pool.release(buf2);

    // Acquire again - should be cleared
    let buf3 = pool.acquire();
    assert!(buf3.is_empty());
    assert!(buf3.capacity() >= 1024);
}

#[test]
fn test_buffer_pool_overflow() {
    let pool = BufferPool::new(2, 512);

    // Acquire all buffers
    let _buf1 = pool.acquire();
    let _buf2 = pool.acquire();

    // Pool is empty, but acquire should still work (allocates new)
    let buf3 = pool.acquire();
    assert!(buf3.capacity() >= 512);
}

#[test]
fn test_buffer_pool_scoped() {
    let pool = BufferPool::new(4, 1024);

    {
        let mut scoped = pool.scoped();
        scoped.extend_from_slice(b"test data");
        assert_eq!(scoped.len(), 9);
    } // Buffer returned to pool here

    let stats = pool.stats();
    assert_eq!(stats.available, 4); // All buffers back in pool
}

#[test]
fn test_buffer_pool_scoped_take() {
    let pool = BufferPool::new(4, 1024);

    let owned = {
        let mut scoped = pool.scoped();
        scoped.extend_from_slice(b"keep this");
        scoped.take() // Take ownership
    };

    assert_eq!(&owned[..], b"keep this");

    // Pool should have one less buffer available (taken, not returned)
    let stats = pool.stats();
    assert_eq!(stats.available, 3);
}

#[test]
fn test_executor_config_defaults() {
    let config = ExecutorConfig::default();

    assert_eq!(config.rayon_threads, 0); // auto-detect
    assert_eq!(config.max_lm_concurrency, 8);
    assert_eq!(config.buffer_pool_size, 32);
    assert_eq!(config.buffer_capacity, 16 * 1024);
}

#[test]
fn test_executor_config_builder() {
    let config = ExecutorConfig::new()
        .rayon_threads(4)
        .max_lm_concurrency(16)
        .buffer_pool_size(64);

    assert_eq!(config.rayon_threads, 4);
    assert_eq!(config.max_lm_concurrency, 16);
    assert_eq!(config.buffer_pool_size, 64);
}

#[test]
fn test_executor_creation() {
    let executor = HybridExecutor::new().unwrap();
    let stats = executor.stats();

    assert!(stats.rayon_threads > 0);
    assert_eq!(stats.lm_concurrency_max, 8);
    assert_eq!(stats.lm_concurrency_available, 8);
}

#[test]
fn test_executor_with_config() {
    let config = ExecutorConfig::new()
        .max_lm_concurrency(4)
        .buffer_pool_size(8);

    let executor = HybridExecutor::with_config(config).unwrap();
    let stats = executor.stats();

    assert_eq!(stats.lm_concurrency_max, 4);
    assert_eq!(stats.buffer_pool.capacity, 8);
}

#[test]
fn test_executor_run_cpu() {
    let executor = HybridExecutor::new().unwrap();

    // Run CPU-bound computation
    let result = executor.run_cpu(|| (0..1000).map(|i| i * i).sum::<i64>());

    assert_eq!(result, 332833500);
}

#[test]
fn test_executor_score_parallel() {
    let executor = HybridExecutor::new().unwrap();

    let predictions: Vec<String> = (0..100)
        .map(|i| format!("prediction number {}", i))
        .collect();

    let scores = executor.score_parallel(&predictions, |p| p.len() as f64 / 25.0);

    assert_eq!(scores.len(), 100);
    assert!(scores.iter().all(|&s| s > 0.0));
}

#[test]
fn test_batch_runner_process_parallel() {
    let executor = HybridExecutor::new().unwrap();
    let runner = BatchRunner::new(&executor);

    let items: Vec<i32> = (0..100).collect();

    let results = runner.process_parallel(&items, |x| x * x);

    assert_eq!(results.len(), 100);
    assert_eq!(results[0], 0);
    assert_eq!(results[10], 100);
    assert_eq!(results[99], 9801);
}

#[test]
fn test_batch_runner_score_batch() {
    let executor = HybridExecutor::new().unwrap();
    let runner = BatchRunner::new(&executor);

    let predictions: Vec<String> = vec![
        "short".to_string(),
        "medium length".to_string(),
        "this is a longer prediction".to_string(),
    ];

    let scores = runner.score_batch(&predictions, |p| p.len() as f64);

    assert_eq!(scores.len(), 3);
    assert!(scores[0] < scores[1]);
    assert!(scores[1] < scores[2]);
}

#[tokio::test]
async fn test_executor_acquire_permit() {
    let executor =
        HybridExecutor::with_config(ExecutorConfig::new().max_lm_concurrency(2)).unwrap();

    // Initially all permits available
    assert_eq!(executor.stats().lm_concurrency_available, 2);

    // Acquire first permit
    let permit1 = executor.acquire_lm_permit().await;
    assert_eq!(executor.stats().lm_concurrency_available, 1);

    // Acquire second permit
    let permit2 = executor.acquire_lm_permit().await;
    assert_eq!(executor.stats().lm_concurrency_available, 0);

    // Release permits
    drop(permit1);
    assert_eq!(executor.stats().lm_concurrency_available, 1);

    drop(permit2);
    assert_eq!(executor.stats().lm_concurrency_available, 2);
}

#[test]
fn test_executor_buffer_pool_integration() {
    let executor = HybridExecutor::new().unwrap();
    let pool = executor.buffer_pool();

    // Use pool through executor
    let mut buf = pool.acquire();
    buf.extend_from_slice(b"test");

    let stats = pool.stats();
    assert!(stats.buffer_size > 0);

    pool.release(buf);
}

#[test]
fn test_executor_parallel_scoring_correctness() {
    let executor = HybridExecutor::new().unwrap();

    // Create predictions with known properties
    let predictions: Vec<String> = vec![
        "a".repeat(10),
        "b".repeat(20),
        "c".repeat(30),
        "d".repeat(40),
        "e".repeat(50),
    ];

    let scores = executor.score_parallel(&predictions, |p| p.len() as f64);

    assert_eq!(scores[0], 10.0);
    assert_eq!(scores[1], 20.0);
    assert_eq!(scores[2], 30.0);
    assert_eq!(scores[3], 40.0);
    assert_eq!(scores[4], 50.0);
}

#[test]
fn test_batch_runner_large_batch() {
    let executor = HybridExecutor::new().unwrap();
    let runner = BatchRunner::new(&executor);

    // Process large batch
    let items: Vec<u64> = (0..10000).collect();

    let results = runner.process_parallel(&items, |&x| {
        // Some CPU work
        (0..100).fold(x, |acc, _| acc.wrapping_mul(17).wrapping_add(1))
    });

    assert_eq!(results.len(), 10000);
}

#[test]
fn test_executor_stats() {
    let config = ExecutorConfig::new()
        .rayon_threads(2)
        .max_lm_concurrency(4)
        .buffer_pool_size(16)
        .buffer_capacity(8192);

    let executor = HybridExecutor::with_config(config).unwrap();
    let stats = executor.stats();

    assert_eq!(stats.rayon_threads, 2);
    assert_eq!(stats.lm_concurrency_max, 4);
    assert_eq!(stats.lm_concurrency_available, 4);
    assert_eq!(stats.buffer_pool.capacity, 16);
    assert_eq!(stats.buffer_pool.buffer_size, 8192);
}
