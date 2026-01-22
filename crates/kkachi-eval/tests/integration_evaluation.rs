// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for evaluation system

use kkachi::*;
use kkachi_eval::*;
use std::borrow::Cow;
use std::sync::Arc;

#[test]
fn test_exact_match_metric() {
    let mut example = Example::new();
    example.insert_output("answer", "42");

    let mut prediction = Prediction::new();
    prediction.insert("answer", Cow::Borrowed("42"));

    let metric = metric::ExactMatch;
    let result = metric.evaluate(&example, &prediction);

    assert!(result.passed);
    assert_eq!(result.score, 1.0);
}

#[test]
fn test_exact_match_failure() {
    let mut example = Example::new();
    example.insert_output("answer", "42");

    let mut prediction = Prediction::new();
    prediction.insert("answer", Cow::Borrowed("43"));

    let metric = metric::ExactMatch;
    let result = metric.evaluate(&example, &prediction);

    assert!(!result.passed);
    assert_eq!(result.score, 0.0);
}

#[test]
fn test_f1_score_metric() {
    let mut example = Example::new();
    example.insert_output("answer", "the quick brown fox");

    let mut prediction = Prediction::new();
    prediction.insert("answer", Cow::Borrowed("the quick red fox"));

    let metric = metric::F1Score;
    let result = metric.evaluate(&example, &prediction);

    assert!(result.score > 0.0);
    assert!(result.score < 1.0);
}

#[test]
fn test_parallel_evaluator() {
    let examples: Vec<Example> = (0..20)
        .map(|i| {
            let mut ex = Example::new();
            ex.insert_output("value", i.to_string());
            ex
        })
        .collect();

    let predictions: Vec<Prediction> = (0..20)
        .map(|i| {
            let mut pred = Prediction::new();
            pred.insert("value", Cow::Owned(i.to_string()));
            pred
        })
        .collect();

    let evaluator = ParallelEvaluator::new(Arc::new(metric::ExactMatch)).with_threads(4);

    let result = evaluator
        .evaluate_predictions(&examples, &predictions)
        .unwrap();

    assert_eq!(result.total, 20);
    assert_eq!(result.passed, 20);
    assert_eq!(result.score, 1.0);
    assert_eq!(result.accuracy(), 1.0);
}

#[test]
fn test_evaluator_mixed_results() {
    let examples: Vec<Example> = vec!["1", "2", "3", "4", "5"]
        .iter()
        .map(|&v| {
            let mut ex = Example::new();
            ex.insert_output("val", v);
            ex
        })
        .collect();

    let predictions: Vec<Prediction> = vec!["1", "2", "3", "wrong", "5"]
        .iter()
        .map(|&v| {
            let mut pred = Prediction::new();
            pred.insert("val", Cow::Borrowed(v));
            pred
        })
        .collect();

    let evaluator = ParallelEvaluator::new(Arc::new(metric::ExactMatch));
    let result = evaluator
        .evaluate_predictions(&examples, &predictions)
        .unwrap();

    assert_eq!(result.total, 5);
    assert_eq!(result.passed, 4);
    assert_eq!(result.accuracy(), 0.8);
}

#[test]
fn test_evaluation_result_statistics() {
    let examples: Vec<Example> = (0..10)
        .map(|i| {
            let mut ex = Example::new();
            ex.insert_output("num", i.to_string());
            ex
        })
        .collect();

    let predictions: Vec<Prediction> = (0..10)
        .map(|i| {
            let mut pred = Prediction::new();
            pred.insert("num", Cow::Owned((i / 2).to_string())); // Wrong half the time
            pred
        })
        .collect();

    let evaluator = ParallelEvaluator::new(Arc::new(metric::ExactMatch));
    let result = evaluator
        .evaluate_predictions(&examples, &predictions)
        .unwrap();

    assert!(result.score < 1.0);
    assert!(result.score > 0.0);
    assert_eq!(result.results.len(), 10);
}
