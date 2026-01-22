// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for training mode with RAG updates.

use kkachi::recursive::{
    HashEmbedder, InMemoryVectorStore, TrainingConfig, TrainingExample, TrainingStats, VectorStore,
};

#[test]
fn test_training_example_formatting() {
    let example = TrainingExample {
        id: "rust:abc123".to_string(),
        question: "How do I parse a URL in Rust?".to_string(),
        answer: r#"use url::Url;

fn parse_url(s: &str) -> Option<String> {
    Url::parse(s).ok().and_then(|u| u.host_str().map(|h| h.to_string()))
}"#
        .to_string(),
        score: 1.0,
        iterations: 3,
        domain: "rust".to_string(),
        error_corrections: vec![
            (
                "Missing Result type".to_string(),
                "Changed to Option".to_string(),
            ),
            (
                "Missing import".to_string(),
                "Added use url::Url".to_string(),
            ),
        ],
    };

    let few_shot = example.as_few_shot();
    assert!(few_shot.contains("parse a URL"));
    assert!(few_shot.contains("Url::parse"));
    assert!(few_shot.contains("1.00"));

    let learning = example.as_learning_example();
    assert!(learning.contains("Error Corrections Made"));
    assert!(learning.contains("Missing Result type"));
    assert!(learning.contains("Changed to Option"));
}

#[test]
fn test_mutable_vector_store_operations() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add documents
    store.add("doc1".to_string(), "First document about Rust".to_string());
    store.add(
        "doc2".to_string(),
        "Second document about Python".to_string(),
    );
    store.add("doc3".to_string(), "Third document about Go".to_string());

    assert_eq!(store.len(), 3);

    // Remove a document
    assert!(store.remove("doc2"));
    assert_eq!(store.len(), 2);

    // Remove non-existent
    assert!(!store.remove("doc2"));
    assert_eq!(store.len(), 2);

    // Update a document
    store.update(
        "doc1".to_string(),
        "Updated document about Rust programming".to_string(),
    );
    assert_eq!(store.len(), 2);

    // Get document
    let content = store.get("doc1");
    assert!(content.is_some());
    assert!(content.unwrap().contains("Updated"));

    // Get non-existent
    assert!(store.get("nonexistent").is_none());

    // Clear
    store.clear();
    assert!(store.is_empty());
}

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

#[test]
fn test_training_stats_tracking() {
    let stats = TrainingStats {
        total_refinements: 10,
        successful_refinements: 8,
        cache_hits: 2,
        total_iterations: 25,
        rag_additions: 6,
        rag_updates: 2,
        rag_deduplicated: 1,
    };

    assert_eq!(stats.total_refinements, 10);
    assert_eq!(stats.successful_refinements, 8);
    assert_eq!(stats.rag_additions + stats.rag_updates, 8); // Should match successful
}

#[test]
fn test_vector_store_deduplication_scenario() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add initial example
    store.add(
        "rust:url1".to_string(),
        "Q: How to parse URL?\nA: Use url::Url".to_string(),
    );

    // Search for similar
    let results = store.search_text("How do I parse a URL in Rust?", 1);
    assert!(!results.is_empty());

    // If similarity is high, we'd update instead of add
    let top = &results[0];
    if top.score > 0.8 {
        // Update existing
        store.update(
            top.id.clone(),
            "Q: How to parse URL?\nA: Use url::Url with better error handling".to_string(),
        );
        assert_eq!(store.len(), 1);
    } else {
        // Add new
        store.add(
            "rust:url2".to_string(),
            "Q: Parse URL hostname?\nA: Use url.host_str()".to_string(),
        );
        assert_eq!(store.len(), 2);
    }
}

#[test]
fn test_few_shot_retrieval_for_training() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Populate with training examples
    let examples = vec![
        ("ex1", "Q: Parse JSON in Rust\nA: use serde_json::from_str"),
        (
            "ex2",
            "Q: Read file in Rust\nA: use std::fs::read_to_string",
        ),
        ("ex3", "Q: HTTP request in Rust\nA: use reqwest::get"),
        ("ex4", "Q: Parse YAML in Rust\nA: use serde_yaml::from_str"),
    ];

    for (id, content) in examples {
        store.add(id.to_string(), content.to_string());
    }

    // Query for similar examples (few-shot retrieval)
    let results = store.search_text("How to parse TOML in Rust?", 3);

    assert_eq!(results.len(), 3);

    // Should find parsing-related examples
    let has_parse = results.iter().any(|r| r.content.contains("Parse"));
    assert!(has_parse, "Should find parsing examples for few-shot");
}

#[test]
fn test_rag_update_on_convergence() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Simulate convergence: add successful refinement to RAG
    let question = "Write a function to validate email addresses";
    let answer = r#"fn is_valid_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}"#;

    // Create training example
    let example = TrainingExample {
        id: "rust:email_validator".to_string(),
        question: question.to_string(),
        answer: answer.to_string(),
        score: 0.95,
        iterations: 2,
        domain: "rust".to_string(),
        error_corrections: vec![],
    };

    // Add to RAG on convergence
    store.add(example.id.clone(), example.as_learning_example());

    // Verify it's retrievable
    let results = store.search_text("email validation", 1);
    assert!(!results.is_empty());
    assert!(results[0].content.contains("validate email"));
}
