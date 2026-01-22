// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Performance benchmarks for Kkachi

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kkachi::*;

fn benchmark_signature_creation(c: &mut Criterion) {
    c.bench_function("signature_from_str", |b| {
        b.iter(|| {
            Signature::from_str(black_box("question, context -> answer, confidence")).unwrap()
        })
    });
}

fn benchmark_field_inference(c: &mut Criterion) {
    c.bench_function("field_prefix_inference", |b| {
        b.iter(|| {
            field::InputField::new(black_box("userQueryText"), "desc")
        })
    });
}

fn benchmark_example_creation(c: &mut Criterion) {
    c.bench_function("example_with_data", |b| {
        b.iter(|| {
            let mut ex = Example::new();
            ex.insert_input("q", "test");
            ex.insert_output("a", "answer");
            ex
        })
    });
}

fn benchmark_prediction_creation(c: &mut Criterion) {
    c.bench_function("prediction_insert", |b| {
        b.iter(|| {
            let mut pred = Prediction::new();
            pred.insert("answer", "test response");
            pred.insert("confidence", "0.95");
            pred
        })
    });
}

criterion_group!(
    benches,
    benchmark_signature_creation,
    benchmark_field_inference,
    benchmark_example_creation,
    benchmark_prediction_creation
);
criterion_main!(benches);
