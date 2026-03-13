// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Benchmarks for core LLM operations with mock backends.
//!
//! These benchmarks measure the overhead of the kkachi framework itself
//! (validation, scoring, pipeline composition) without actual API latency.

use criterion::{criterion_group, criterion_main, Criterion};
use kkachi::recursive::validate::Validate;
use kkachi::recursive::{all, best_of, checks, ensemble, pipeline, refine, MockLlm};

fn bench_refine_mock(c: &mut Criterion) {
    let llm = MockLlm::new(|_, _| "fn add(a: i32, b: i32) -> i32 { a + b }".to_string());
    let v = checks().require("fn ").require("->").min_len(20);

    c.bench_function("refine_1iter", |b| {
        b.iter(|| {
            refine(&llm, "write add fn")
                .validate(v.clone())
                .max_iter(1)
                .go()
        })
    });

    c.bench_function("refine_5iter_pass", |b| {
        b.iter(|| {
            refine(&llm, "write add fn")
                .validate(v.clone())
                .max_iter(5)
                .go()
        })
    });

    // Test with a validator that always fails (forces all iterations)
    let v_fail = checks().require("nonexistent_token");
    c.bench_function("refine_5iter_fail", |b| {
        b.iter(|| {
            refine(&llm, "write add fn")
                .validate(v_fail.clone())
                .max_iter(5)
                .go()
        })
    });
}

fn bench_best_of_mock(c: &mut Criterion) {
    let llm = MockLlm::new(|_, _| "candidate output text".to_string());

    c.bench_function("best_of_3", |b| {
        b.iter(|| {
            best_of(&llm, "prompt")
                .n(3)
                .metric(|text: &str| text.len() as f64 / 100.0)
                .go()
        })
    });

    c.bench_function("best_of_10", |b| {
        b.iter(|| {
            best_of(&llm, "prompt")
                .n(10)
                .metric(|text: &str| text.len() as f64 / 100.0)
                .go()
        })
    });

    c.bench_function("best_of_3_with_validation", |b| {
        b.iter(|| {
            best_of(&llm, "prompt")
                .n(3)
                .validate(checks().require("candidate"))
                .metric(|text: &str| text.len() as f64 / 100.0)
                .go()
        })
    });
}

fn bench_ensemble_mock(c: &mut Criterion) {
    let llm = MockLlm::new(|_, _| "Paris".to_string());

    c.bench_function("ensemble_3", |b| {
        b.iter(|| ensemble(&llm, "capital of France").n(3).go())
    });

    c.bench_function("ensemble_5", |b| {
        b.iter(|| ensemble(&llm, "capital of France").n(5).go())
    });

    c.bench_function("ensemble_10", |b| {
        b.iter(|| ensemble(&llm, "capital of France").n(10).go())
    });
}

fn bench_validators(c: &mut Criterion) {
    let text =
        "fn fibonacci(n: u32) -> u32 { if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) } }";

    c.bench_function("checks_single_require", |b| {
        let v = checks().require("fn ");
        b.iter(|| v.validate(text))
    });

    c.bench_function("checks_multi_require", |b| {
        let v = checks()
            .require("fn ")
            .require("->")
            .require("{")
            .require("}");
        b.iter(|| v.validate(text))
    });

    c.bench_function("checks_require_forbid_min_len", |b| {
        let v = checks()
            .require("fn ")
            .require("->")
            .forbid("unsafe")
            .min_len(20);
        b.iter(|| v.validate(text))
    });

    c.bench_function("compose_all", |b| {
        let v = all([
            checks().require("fn "),
            checks().require("->"),
            checks().forbid("unsafe"),
        ]);
        b.iter(|| v.validate(text))
    });

    // Long text validation
    let long_text = text.repeat(100);
    c.bench_function("checks_long_text", |b| {
        let v = checks().require("fn ").forbid("unsafe").min_len(100);
        b.iter(|| v.validate(&long_text))
    });
}

fn bench_pipeline_mock(c: &mut Criterion) {
    let llm = MockLlm::new(|_, _| "```rust\nfn main() { println!(\"hello\"); }\n```".to_string());
    let v = checks().require("fn ");

    c.bench_function("pipeline_refine", |b| {
        b.iter(|| pipeline(&llm, "write main").refine(v.clone()).go())
    });

    c.bench_function("pipeline_refine_extract", |b| {
        b.iter(|| {
            pipeline(&llm, "write main")
                .refine(checks().require("```"))
                .extract("rust")
                .go()
        })
    });

    c.bench_function("pipeline_3step", |b| {
        b.iter(|| {
            pipeline(&llm, "write main")
                .refine(checks().require("```"))
                .extract("rust")
                .map(|s| s.trim().to_string())
                .go()
        })
    });
}

criterion_group!(
    benches,
    bench_refine_mock,
    bench_best_of_mock,
    bench_ensemble_mock,
    bench_validators,
    bench_pipeline_mock
);
criterion_main!(benches);
