// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Cache performance benchmarks
//!
//! Measures cache hit/miss performance to establish baseline before Arc<LmOutput> optimization.
//!
//! Target improvements (Phase 3):
//! - 30-50% reduction in allocations
//! - 10-100x faster cache hits (refcount vs clone)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kkachi::recursive::{llm::MockLlm, CacheExt, Llm};

fn benchmark_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hit");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Create mock LLM with cache
    let mock = MockLlm::new(|prompt, _| format!("Response to: {}", prompt));
    let cached_llm = mock.with_cache(100);

    // Pre-warm cache with a response
    rt.block_on(async {
        let _ = cached_llm.generate("test prompt", "", None).await;
    });

    // Benchmark cache hits
    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    // This should hit cache (already warmed)
                    let result = cached_llm
                        .generate(black_box("test prompt"), "", None)
                        .await;
                    black_box(result)
                })
            });
        });
    }

    group.finish();
}

fn benchmark_cache_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_miss");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mock = MockLlm::new(|prompt, _| format!("Response to: {}", prompt));
    let cached_llm = mock.with_cache(100);

    let mut counter = 0;

    group.throughput(Throughput::Elements(1));
    group.bench_function("miss_and_store", |b| {
        b.iter(|| {
            // Each iteration uses unique prompt to force cache miss
            let prompt = format!("unique prompt {}", counter);
            counter += 1;

            rt.block_on(async {
                let result = cached_llm.generate(black_box(&prompt), "", None).await;
                black_box(result)
            })
        });
    });

    group.finish();
}

fn benchmark_cache_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_capacity");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for capacity in [10, 50, 100, 500].iter() {
        let mock = MockLlm::new(|prompt, _| format!("Response to: {}", prompt));
        let cached_llm = mock.with_cache(*capacity);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            capacity,
            |b, _cap| {
                b.iter(|| {
                    rt.block_on(async {
                        // Hit same cache entry
                        let result = cached_llm
                            .generate(black_box("test prompt"), "", None)
                            .await;
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cache_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_eviction");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mock = MockLlm::new(|prompt, _| format!("Response to: {}", prompt));
    let cached_llm = mock.with_cache(10); // Small cache to force evictions

    let mut counter = 0;

    group.throughput(Throughput::Elements(1));
    group.bench_function("with_eviction", |b| {
        b.iter(|| {
            // Generate many unique prompts to force LRU evictions
            let prompt = format!("prompt_{}", counter);
            counter += 1;

            rt.block_on(async {
                let result = cached_llm.generate(black_box(&prompt), "", None).await;
                black_box(result)
            })
        });
    });

    group.finish();
}

criterion_group!(
    cache_benches,
    benchmark_cache_hit,
    benchmark_cache_miss,
    benchmark_cache_capacity,
    benchmark_cache_eviction,
);
criterion_main!(cache_benches);
