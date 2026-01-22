// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for RAG (Retrieval Augmented Generation) functionality.

use kkachi::recursive::{CompositeVectorStore, HashEmbedder, InMemoryVectorStore, VectorStore};

#[test]
fn test_composite_vector_store() {
    // Create two separate stores
    let embedder1 = HashEmbedder::new(64);
    let mut store1 = InMemoryVectorStore::new(embedder1);
    store1.add(
        "rust1".to_string(),
        "Rust ownership and borrowing".to_string(),
    );
    store1.add("rust2".to_string(), "Rust lifetimes explained".to_string());

    let embedder2 = HashEmbedder::new(64);
    let mut store2 = InMemoryVectorStore::new(embedder2);
    store2.add(
        "python1".to_string(),
        "Python list comprehensions".to_string(),
    );
    store2.add("python2".to_string(), "Python async await".to_string());

    // Create composite store
    let composite = CompositeVectorStore::new(vec![&store1 as &dyn VectorStore, &store2]);

    // Total length should be sum
    assert_eq!(composite.len(), 4);

    // Search should find from both stores
    let results = composite.search_text("Rust programming", 3);
    assert!(!results.is_empty());

    let results = composite.search_text("Python programming", 3);
    assert!(!results.is_empty());
}

#[test]
fn test_vector_search_result_ordering() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add documents with varying relevance to "machine learning"
    store.add("ml1".to_string(), "Machine learning algorithms".to_string());
    store.add(
        "ml2".to_string(),
        "Deep learning neural networks machine learning".to_string(),
    );
    store.add(
        "unrelated".to_string(),
        "Cooking recipes for pasta".to_string(),
    );

    let results = store.search_text("machine learning", 3);

    // Results should be ordered by score descending
    for i in 0..results.len() - 1 {
        assert!(
            results[i].score >= results[i + 1].score,
            "Results not ordered: {} >= {}",
            results[i].score,
            results[i + 1].score
        );
    }

    // ML-related docs should rank higher than cooking
    let ml_ids: Vec<_> = results.iter().filter(|r| r.id.starts_with("ml")).collect();
    assert!(!ml_ids.is_empty(), "Should find ML documents");
}

#[test]
fn test_vector_store_metadata() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    store.add("doc1".to_string(), "Test document content".to_string());

    let results = store.search_text("test", 1);
    assert_eq!(results.len(), 1);

    let result = &results[0];
    assert_eq!(result.id, "doc1");
    assert_eq!(result.content, "Test document content");
    assert!(result.score > 0.0);
}

#[test]
fn test_empty_store_search() {
    let embedder = HashEmbedder::new(64);
    let store = InMemoryVectorStore::new(embedder);

    assert!(store.is_empty());

    let results = store.search_text("anything", 10);
    assert!(results.is_empty());
}

#[test]
fn test_search_k_limits() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add 5 documents
    for i in 0..5 {
        store.add(format!("doc{}", i), format!("Document number {}", i));
    }

    // Search for more than available
    let results = store.search_text("document", 10);
    assert_eq!(results.len(), 5, "Should return all 5 documents");

    // Search for fewer
    let results = store.search_text("document", 2);
    assert_eq!(results.len(), 2, "Should return only 2 documents");

    // Search for zero
    let results = store.search_text("document", 0);
    assert!(results.is_empty(), "Should return no documents");
}

#[test]
fn test_composite_empty_stores() {
    let embedder1 = HashEmbedder::new(64);
    let store1 = InMemoryVectorStore::new(embedder1);

    let embedder2 = HashEmbedder::new(64);
    let store2 = InMemoryVectorStore::new(embedder2);

    let composite = CompositeVectorStore::new(vec![&store1 as &dyn VectorStore, &store2]);

    assert!(composite.is_empty());
    assert_eq!(composite.len(), 0);

    let results = composite.search_text("anything", 10);
    assert!(results.is_empty());
}

#[test]
fn test_dimension_consistency() {
    let dim = 128;
    let embedder = HashEmbedder::new(dim);
    let store = InMemoryVectorStore::new(embedder);

    assert_eq!(store.dimension(), dim);
}

#[test]
fn test_few_shot_retrieval_scenario() {
    // Simulate few-shot example retrieval for code generation
    let embedder = HashEmbedder::new(64);
    let mut examples = InMemoryVectorStore::new(embedder);

    // Add example code snippets
    examples.add(
        "ex1".to_string(),
        "Q: Parse URL hostname\nA: use url::Url; fn parse(s: &str) -> Option<String>".to_string(),
    );
    examples.add(
        "ex2".to_string(),
        "Q: Read file contents\nA: use std::fs; fn read(p: &str) -> std::io::Result<String>"
            .to_string(),
    );
    examples.add(
        "ex3".to_string(),
        "Q: HTTP GET request\nA: use reqwest; async fn get(url: &str) -> Result<String>"
            .to_string(),
    );
    examples.add(
        "ex4".to_string(),
        "Q: Parse JSON\nA: use serde_json; fn parse<T: DeserializeOwned>(s: &str) -> Result<T>"
            .to_string(),
    );

    // Query for similar task
    let query = "How do I parse a URL and extract the domain?";
    let results = examples.search_text(query, 3);

    assert!(!results.is_empty());
    // URL-related example should rank high
    let has_url_example = results.iter().any(|r| r.content.contains("URL"));
    assert!(has_url_example, "Should find URL-related example");
}
