// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for optimizer functionality
//!
//! These tests verify end-to-end optimizer workflows with mock data.

use kkachi::buffer::Buffer;
use kkachi::optimizer::{ExampleMeta, ExampleSet, Optimizer};
use kkachi::optimizers::*;
use kkachi::recursive::optimize::*;

#[test]
fn test_dataset_construction_and_usage() {
    // Test Dataset API for optimization workflows
    let dataset = Dataset::new()
        .example("Input 1", "Expected 1")
        .example("Input 2", "Expected 2")
        .labeled_example("Input 3", "Expected 3", "category_a")
        .labeled_example("Input 4", "Expected 4", "category_b");

    assert_eq!(dataset.len(), 4);
    assert!(!dataset.is_empty());

    assert_eq!(dataset.examples[0].input, "Input 1");
    assert_eq!(dataset.examples[0].expected, "Expected 1");
    assert!(dataset.examples[0].label.is_none());

    assert_eq!(dataset.examples[2].label, Some("category_a".to_string()));
    assert_eq!(dataset.examples[3].label, Some("category_b".to_string()));
}

#[test]
fn test_training_example_with_labels() {
    let example = TrainingExample::new("question", "answer");
    assert_eq!(example.input, "question");
    assert_eq!(example.expected, "answer");
    assert!(example.label.is_none());

    let labeled = example.with_label("math");
    assert_eq!(labeled.label, Some("math".to_string()));
}

#[test]
fn test_optimizer_config_creation() {
    // Test all optimizer configs can be created with fluent API

    let labeled_cfg = LabeledConfig::new()
        .with_k(3)
        .with_strategy(SelectionStrategy::First)
        .with_seed(42);
    assert_eq!(labeled_cfg.k, 3);

    let knn_cfg = KNNConfig::new().with_k(5);
    assert_eq!(knn_cfg.k, 5);

    let copro_cfg = COPROConfig::new().with_breadth(3).with_depth(2);
    assert_eq!(copro_cfg.breadth, 3);

    let mipro_cfg = MIPROConfig::new()
        .with_num_instructions(10)
        .with_num_trials(50);
    assert_eq!(mipro_cfg.num_trials, 50);

    let simba_cfg = SIMBAConfig::new()
        .with_max_iterations(5)
        .with_min_improvement(0.05);
    assert_eq!(simba_cfg.max_iterations, 5);
}

#[test]
fn test_labeled_fewshot_creation() {
    let optimizer = LabeledFewShot::default();
    assert_eq!(optimizer.name(), "LabeledFewShot");
    assert_eq!(optimizer.config().k, 5);

    let custom = LabeledFewShot::new(
        LabeledConfig::new()
            .with_k(3)
            .with_strategy(SelectionStrategy::First),
    );
    assert_eq!(custom.config().k, 3);
}

#[test]
fn test_labeled_fewshot_builder() {
    let optimizer = LabeledFewShotBuilder::new().k(7).random(999).build();

    assert_eq!(optimizer.config().k, 7);
    assert_eq!(optimizer.config().seed, 999);
}

#[test]
fn test_labeled_fewshot_selection() {
    // Create a mock trainset
    let buffer = Buffer::Static(
        b"input1\noutput1\ninput2\noutput2\ninput3\noutput3\ninput4\noutput4\ninput5\noutput5",
    );
    let trainset = ExampleSet::from_buffer(buffer, 5);

    // Test First strategy
    let first = LabeledFewShot::new(
        LabeledConfig::new()
            .with_k(3)
            .with_strategy(SelectionStrategy::First),
    );
    let indices = first.select(&trainset);
    assert_eq!(indices.len(), 3);
    assert_eq!(indices[0], 0);
    assert_eq!(indices[1], 1);
    assert_eq!(indices[2], 2);

    // Test Last strategy
    let last = LabeledFewShot::new(
        LabeledConfig::new()
            .with_k(3)
            .with_strategy(SelectionStrategy::Last),
    );
    let indices = last.select(&trainset);
    assert_eq!(indices.len(), 3);
    assert_eq!(indices[0], 2);
    assert_eq!(indices[1], 3);
    assert_eq!(indices[2], 4);
}

#[test]
fn test_knn_fewshot_creation() {
    // Note: KNNFewShot requires an Embedder type parameter
    // This test just verifies the config works
    let config = KNNConfig::default();
    assert_eq!(config.k, 3);
}

#[test]
fn test_embedding_index() {
    // Test embedding index can be created
    let index = EmbeddingIndex::new(3); // 3 dimensions
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

#[test]
fn test_copro_creation() {
    let optimizer = COPRO::default();
    assert_eq!(optimizer.name(), "COPRO");

    let custom = COPRO::new(COPROConfig::new().with_breadth(2).with_depth(1));
    assert_eq!(custom.config().breadth, 2);
}

#[test]
fn test_mipro_creation() {
    let optimizer = MIPRO::default();
    assert_eq!(optimizer.name(), "MIPRO");

    let custom = MIPRO::new(
        MIPROConfig::new()
            .with_num_trials(25)
            .with_num_instructions(8),
    );
    assert_eq!(custom.config().num_trials, 25);
}

#[test]
fn test_mipro_tpe_sampler() {
    let mut sampler = TPESampler::new(0.25, 42);

    // Should return random with no history
    let first = sampler.suggest_instruction(5);
    assert!(first < 5);

    // Should be able to suggest multiple times
    let second = sampler.suggest_instruction(5);
    assert!(second < 5);
}

#[test]
fn test_simba_creation() {
    let optimizer = SIMBA::default();
    assert_eq!(optimizer.name(), "SIMBA");

    let custom = SIMBA::new(
        SIMBAConfig::new()
            .with_max_iterations(3)
            .with_min_improvement(0.05),
    );
    assert_eq!(custom.config().max_iterations, 3);
}

#[test]
fn test_ensemble_creation_and_strategies() {
    let config = EnsembleConfig::new().with_strategy(CombineStrategy::Best);

    let ensemble = Ensemble::new(config);
    assert_eq!(ensemble.name(), "Ensemble");

    // Test all combine strategies can be created
    let strategies = vec![
        CombineStrategy::Best,
        CombineStrategy::Union,
        CombineStrategy::Intersection,
    ];

    for strategy in strategies {
        let cfg = EnsembleConfig::new().with_strategy(strategy);
        let e = Ensemble::new(cfg);
        assert_eq!(e.name(), "Ensemble");
    }
}

#[test]
fn test_example_set_zero_copy() {
    // Verify ExampleSet uses zero-copy patterns
    let buffer = Buffer::from("Example data content");
    let meta = vec![
        ExampleMeta::empty(),
        ExampleMeta::empty(),
        ExampleMeta::empty(),
    ];

    let example_set = ExampleSet::new(&buffer, &meta);

    assert_eq!(example_set.len(), 3);
    assert!(!example_set.is_empty());
}

#[test]
fn test_example_meta_fields() {
    let meta = ExampleMeta::empty();

    // Initially no ranges
    assert_eq!(meta.inputs().count(), 0);
    assert_eq!(meta.outputs().count(), 0);

    // Meta can be constructed
    let mut meta2 = ExampleMeta::empty();
    meta2.input_count = 0;
    meta2.output_count = 0;
    assert_eq!(meta2.inputs().count(), 0);
}

#[test]
fn test_selection_strategies() {
    // Verify all selection strategies exist
    let strategies = vec![
        SelectionStrategy::First,
        SelectionStrategy::Last,
        SelectionStrategy::Random,
        SelectionStrategy::Stratified,
    ];

    for strategy in strategies {
        // Should be able to create config with each strategy
        let config = LabeledConfig::new().with_strategy(strategy);
        let _optimizer = LabeledFewShot::new(config);
    }
}

#[test]
fn test_combine_strategies() {
    // Verify all combine strategies exist
    let strategies = vec![
        CombineStrategy::Best,
        CombineStrategy::Union,
        CombineStrategy::Intersection,
    ];

    for strategy in strategies {
        // Should be able to create ensemble with each strategy
        let _ensemble = Ensemble::new(EnsembleConfig::new().with_strategy(strategy));
    }
}

#[test]
fn test_all_optimizers_have_names() {
    // Verify all optimizers implement Optimizer trait with name()
    assert_eq!(LabeledFewShot::default().name(), "LabeledFewShot");
    assert_eq!(COPRO::default().name(), "COPRO");
    assert_eq!(MIPRO::default().name(), "MIPRO");
    assert_eq!(SIMBA::default().name(), "SIMBA");
    assert_eq!(Ensemble::new(EnsembleConfig::default()).name(), "Ensemble");
}

#[test]
fn test_all_optimizers_have_configs() {
    // Verify all optimizers expose their configs
    let labeled = LabeledFewShot::default();
    assert_eq!(labeled.config().k, 5);

    let copro = COPRO::default();
    assert_eq!(copro.config().breadth, 5);

    let mipro = MIPRO::default();
    assert_eq!(mipro.config().num_trials, 50);

    let simba = SIMBA::default();
    assert_eq!(simba.config().max_iterations, 5);

    let _ensemble = Ensemble::new(EnsembleConfig::default());
    // Ensemble config created successfully
}

#[test]
fn test_metric_functions() {
    // Test various metric patterns work correctly

    // Exact match
    let exact = |output: &str, expected: &str| -> f64 {
        if output == expected {
            1.0
        } else {
            0.0
        }
    };
    assert_eq!(exact("hello", "hello"), 1.0);
    assert_eq!(exact("hello", "world"), 0.0);

    // Contains
    let contains = |output: &str, expected: &str| -> f64 {
        if output.contains(expected) {
            1.0
        } else {
            0.0
        }
    };
    assert_eq!(contains("hello world", "world"), 1.0);
    assert_eq!(contains("hello", "world"), 0.0);

    // Partial credit based on length
    let length_sim = |output: &str, expected: &str| -> f64 {
        let len_diff = (output.len() as i32 - expected.len() as i32).abs();
        1.0 / (1.0 + len_diff as f64 / 10.0)
    };
    assert!(length_sim("hello", "hello") > 0.9);
    assert!(length_sim("hello", "hi") < 0.8);
}
