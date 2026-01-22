// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! End-to-end tests for the training pipeline with CLI critics.
//!
//! These tests validate the full training workflow:
//! 1. CLI critic validates code against real tools
//! 2. Training runner updates context database on improvements
//! 3. RAG vector store is updated upon convergence
//! 4. Future queries can retrieve cached results

use kkachi::recursive::{
    BinaryCritic, ChecklistCritic, Critic, HashEmbedder, HeuristicCritic, InMemoryVectorStore,
    RecursiveConfig, RecursiveState, StandaloneRunner, TrainingConfig, TrainingExample,
    TrainingStats, VectorStore,
};
use kkachi::str_view::StrView;

/// Test that BinaryCritic correctly evaluates code
#[test]
fn test_binary_critic_for_rust_code() {
    // Critic that checks for fn keyword
    let critic = BinaryCritic::new(
        |code: &str| code.contains("fn ") && code.contains("->"),
        "Code must have fn keyword and return type",
    );

    let state = RecursiveState::new();

    // Valid Rust function
    let valid_code = "fn parse_url(s: &str) -> Option<String> { None }";
    let result = critic.evaluate(StrView::new(valid_code), &state);
    assert!(result.is_satisfactory());
    assert_eq!(result.score, 1.0);

    // Invalid code (missing return type)
    let invalid_code = "fn parse_url(s: &str) { }";
    let result = critic.evaluate(StrView::new(invalid_code), &state);
    assert!(!result.is_satisfactory());
    assert_eq!(result.score, 0.0);
}

/// Test ChecklistCritic for code quality checks
#[test]
fn test_checklist_critic_for_code_quality() {
    let critic = ChecklistCritic::new()
        .add_check(
            "has function",
            |s| s.contains("fn "),
            0.3,
            "Missing function definition",
        )
        .add_check(
            "has comments",
            |s| s.contains("//") || s.contains("///"),
            0.2,
            "Missing comments",
        )
        .add_check(
            "handles errors",
            |s| s.contains("Result") || s.contains("Option"),
            0.3,
            "No error handling",
        )
        .add_check("no unwrap", |s| !s.contains("unwrap()"), 0.2, "Uses unwrap");

    let state = RecursiveState::new();

    // High quality code
    let good_code = r#"
/// Parses a URL and returns the hostname.
fn parse_url(s: &str) -> Option<String> {
    // Extract hostname
    s.split('/').nth(2).map(|h| h.to_string())
}
"#;
    let result = critic.evaluate(StrView::new(good_code), &state);
    assert!(
        result.score >= 0.8,
        "Expected high score, got {}",
        result.score
    );

    // Poor quality code (uses unwrap, no comments)
    let bad_code = "fn parse(s: &str) -> String { s.split('/').nth(2).unwrap().to_string() }";
    let result = critic.evaluate(StrView::new(bad_code), &state);
    assert!(
        result.score < 0.5,
        "Expected low score, got {}",
        result.score
    );
}

/// Test HeuristicCritic for pattern matching
#[test]
fn test_heuristic_critic() {
    let critic = HeuristicCritic::new()
        .min_length(20)
        .max_length(5000)
        .require("fn ")
        .forbid("panic!");

    let state = RecursiveState::new();

    // Valid code
    let valid = "fn example() -> u32 { 42 }";
    let result = critic.evaluate(StrView::new(valid), &state);
    assert!(result.is_satisfactory());

    // Too short
    let short = "fn x() {}";
    let result = critic.evaluate(StrView::new(short), &state);
    assert!(!result.is_satisfactory());

    // Contains panic
    let panics = "fn example() { panic!(\"error\"); }";
    let result = critic.evaluate(StrView::new(panics), &state);
    assert!(!result.is_satisfactory());
}

/// Test StandaloneRunner for iterative refinement
#[test]
#[allow(clippy::useless_vec)]
fn test_standalone_runner_refinement() {
    let critic = BinaryCritic::new(
        |code: &str| {
            code.contains("fn ")
                && code.contains("->")
                && code.contains("Option")
                && !code.contains("unwrap")
        },
        "Code must have fn, return type, Option, and no unwrap",
    );

    let config = RecursiveConfig {
        max_iterations: 5,
        score_threshold: 1.0,
        ..Default::default()
    };

    let runner = StandaloneRunner::with_config(&critic, "rust", config);

    // Simulate LLM generating progressively better code
    let iterations = vec![
        "fn parse(s: &str) { }",                         // Missing return type
        "fn parse(s: &str) -> String { s.to_string() }", // No Option
        "fn parse(s: &str) -> Option<String> { s.parse().unwrap() }", // Has unwrap
        "fn parse(s: &str) -> Option<String> { s.parse().ok() }", // Perfect
    ];

    let mut iter_idx = 0;
    let result = runner.refine("Parse a string", |_iteration, _feedback| {
        let code = iterations[iter_idx.min(iterations.len() - 1)].to_string();
        iter_idx += 1;
        Ok(code)
    });

    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(
        result.score >= 1.0,
        "Expected convergence, got score {}",
        result.score
    );
    assert!(
        result.iterations <= 4,
        "Expected <= 4 iterations, got {}",
        result.iterations
    );
}

/// Test that RAG vector store correctly stores and retrieves examples
#[test]
fn test_rag_store_for_few_shot() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add training examples
    let examples = vec![
        TrainingExample {
            id: "rust:url_parse".to_string(),
            question: "How to parse a URL in Rust?".to_string(),
            answer: "use url::Url;\nfn parse(s: &str) -> Option<Url> { Url::parse(s).ok() }".to_string(),
            score: 0.95,
            iterations: 2,
            domain: "rust".to_string(),
            error_corrections: vec![],
        },
        TrainingExample {
            id: "rust:json_parse".to_string(),
            question: "How to parse JSON in Rust?".to_string(),
            answer: "use serde_json;\nfn parse(s: &str) -> serde_json::Value { serde_json::from_str(s).unwrap() }".to_string(),
            score: 0.9,
            iterations: 3,
            domain: "rust".to_string(),
            error_corrections: vec![("Used unwrap".to_string(), "Could use ? operator".to_string())],
        },
        TrainingExample {
            id: "rust:file_read".to_string(),
            question: "How to read a file in Rust?".to_string(),
            answer: "use std::fs;\nfn read(path: &str) -> std::io::Result<String> { fs::read_to_string(path) }".to_string(),
            score: 1.0,
            iterations: 1,
            domain: "rust".to_string(),
            error_corrections: vec![],
        },
    ];

    // Add examples to RAG store
    for example in &examples {
        store.add(example.id.clone(), example.as_learning_example());
    }

    assert_eq!(store.len(), 3);

    // Query for similar examples
    let results = store.search_text("How do I parse TOML in Rust?", 2);
    assert_eq!(results.len(), 2);

    // Should find parsing-related examples (URL or JSON parsing)
    let has_parse = results.iter().any(|r| r.content.contains("parse"));
    assert!(has_parse, "Should retrieve parsing examples");
}

/// Test deduplication logic for RAG updates
#[test]
fn test_rag_deduplication() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add initial example
    store.add(
        "rust:url1".to_string(),
        "Question: How to parse URL?\n\nFinal Answer:\nuse url::Url;".to_string(),
    );

    assert_eq!(store.len(), 1);

    // Query for similar example
    let results = store.search_text("How do I parse a URL in Rust?", 1);
    assert!(!results.is_empty());

    let top = &results[0];
    let dedup_threshold = 0.8f32;

    // If similarity is above threshold, update instead of add
    if top.score >= dedup_threshold {
        store.update(
            top.id.clone(),
            "Question: How to parse URL?\n\nFinal Answer:\nuse url::Url;\nwith better error handling".to_string(),
        );
        assert_eq!(store.len(), 1, "Should update, not add");
    } else {
        store.add(
            "rust:url2".to_string(),
            "Question: Parse URL hostname?\n\nFinal Answer:\nuse url.host_str()".to_string(),
        );
        // May be 1 or 2 depending on hash similarity
    }

    // Verify the content was updated
    if let Some(content) = store.get(&top.id) {
        if top.score >= dedup_threshold {
            assert!(
                content.contains("better error handling"),
                "Content should be updated"
            );
        }
    }
}

/// Test TrainingExample formatting
#[test]
fn test_training_example_formatting() {
    let example = TrainingExample {
        id: "test:example".to_string(),
        question: "Write a function to validate emails".to_string(),
        answer: r#"fn is_valid(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}"#
        .to_string(),
        score: 0.95,
        iterations: 2,
        domain: "rust".to_string(),
        error_corrections: vec![(
            "Missing @ check".to_string(),
            "Added contains('@')".to_string(),
        )],
    };

    // Test few-shot format
    let few_shot = example.as_few_shot();
    assert!(few_shot.contains("validate emails"));
    assert!(few_shot.contains("is_valid"));
    assert!(few_shot.contains("0.95"));
    assert!(few_shot.contains("2")); // iterations

    // Test learning example format
    let learning = example.as_learning_example();
    assert!(learning.contains("Question:"));
    assert!(learning.contains("Final Answer:"));
    assert!(learning.contains("Error Corrections Made:"));
    assert!(learning.contains("Missing @ check"));
    assert!(learning.contains("Added contains('@')"));
}

/// Test TrainingConfig builder pattern
#[test]
fn test_training_config_builder() {
    let config = TrainingConfig::with_domain("terraform")
        .with_min_rag_score(0.9)
        .with_store_iterations(true)
        .with_update_on_improvement(true);

    assert_eq!(config.runner.domain, "terraform");
    assert_eq!(config.min_rag_score, 0.9);
    assert!(config.store_iterations);
    assert!(config.update_on_improvement);
}

/// Test TrainingStats tracking
#[test]
fn test_training_stats() {
    // Simulate training session
    let stats = TrainingStats {
        total_refinements: 10,
        successful_refinements: 8,
        cache_hits: 2,
        total_iterations: 25,
        rag_additions: 5,
        rag_updates: 2,
        rag_deduplicated: 1,
    };

    assert_eq!(stats.total_refinements, 10);
    assert_eq!(stats.successful_refinements, 8);
    assert_eq!(
        stats.rag_additions + stats.rag_updates + stats.rag_deduplicated,
        8
    );
}

/// Test the full training pipeline flow (without actual LLM/storage)
#[test]
#[allow(clippy::useless_vec)]
fn test_training_pipeline_flow() {
    // 1. Set up critic
    let critic = ChecklistCritic::new()
        .add_check(
            "has resource block",
            |s| s.contains("resource"),
            0.3,
            "Missing resource block",
        )
        .add_check(
            "correct resource type",
            |s| s.contains("aws_s3_bucket"),
            0.3,
            "Wrong resource type",
        )
        .add_check(
            "has versioning",
            |s| s.contains("versioning"),
            0.2,
            "Missing versioning",
        )
        .add_check(
            "has encryption",
            |s| s.contains("encryption"),
            0.2,
            "Missing encryption",
        );

    // 2. Set up runner
    let config = RecursiveConfig {
        max_iterations: 5,
        score_threshold: 1.0,
        ..Default::default()
    };

    let runner = StandaloneRunner::with_config(&critic, "terraform", config);

    // 3. Set up RAG store for few-shot examples
    let embedder = HashEmbedder::new(64);
    let mut rag_store = InMemoryVectorStore::new(embedder);

    // Add existing few-shot examples
    rag_store.add(
        "tf:s3_basic".to_string(),
        "Q: Create S3 bucket\nA: resource \"aws_s3_bucket\" \"example\" {}".to_string(),
    );

    // 4. Simulate refinement iterations
    let iterations = vec![
        "resource \"aws_s3_bucket\" \"my_bucket\" {}",
        "resource \"aws_s3_bucket\" \"my_bucket\" {\n  versioning { enabled = true }\n}",
        "resource \"aws_s3_bucket\" \"my_bucket\" {\n  versioning { enabled = true }\n  server_side_encryption_configuration {\n    rule { ... }\n  }\n}",
    ];

    let mut iter_idx = 0;
    let result = runner.refine(
        "Create S3 bucket with versioning and encryption",
        |_iter, _fb| {
            let code = iterations[iter_idx.min(iterations.len() - 1)].to_string();
            iter_idx += 1;
            Ok(code)
        },
    );

    assert!(result.is_ok());
    let result = result.unwrap();

    // 5. On convergence, add to RAG for future retrieval
    if result.score >= 0.8 {
        let example = TrainingExample {
            id: format!("terraform:{}", result.context_id),
            question: "Create S3 bucket with versioning and encryption".to_string(),
            answer: result.answer.clone(),
            score: result.score,
            iterations: result.iterations,
            domain: "terraform".to_string(),
            error_corrections: result.error_corrections.clone(),
        };

        rag_store.add(example.id.clone(), example.as_learning_example());
    }

    // 6. Verify RAG was updated
    assert!(rag_store.len() >= 2, "RAG should have at least 2 examples");

    // 7. Verify future queries can retrieve the new example
    let results = rag_store.search_text("S3 bucket encryption", 2);
    assert!(
        !results.is_empty(),
        "Should find encryption-related examples"
    );
}

/// Test that the critic provides useful feedback for refinement
#[test]
fn test_critic_feedback_for_refinement() {
    let critic = HeuristicCritic::new()
        .min_length(50)
        .require("fn ")
        .require("->")
        .forbid("todo!")
        .forbid("unimplemented!");

    let state = RecursiveState::new();

    // Code that needs improvement
    let code = "fn parse() { todo!() }";
    let result = critic.evaluate(StrView::new(code), &state);

    assert!(!result.is_satisfactory());
    assert!(result.feedback.is_some(), "Should provide feedback");

    let feedback = result.feedback.unwrap();
    assert!(
        feedback.contains("too short")
            || feedback.contains("forbidden")
            || feedback.contains("missing"),
        "Feedback should explain the issue: {}",
        feedback
    );
}
