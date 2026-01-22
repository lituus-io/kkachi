// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for DuckDB vector store.
//!
//! These tests require the `storage` feature to be enabled:
//! ```bash
//! cargo test -p kkachi --features storage --test integration_duckdb_vector_store
//! ```

#![cfg(feature = "storage")]

use kkachi::recursive::{DuckDBVectorStore, HashEmbedder, MutableVectorStore, VectorStore};

/// Test basic DuckDB vector store operations.
#[test]
fn test_duckdb_vector_store_in_memory() {
    let embedder = HashEmbedder::new(64);
    let mut store =
        DuckDBVectorStore::in_memory(embedder).expect("Failed to create in-memory store");

    // Store should be empty initially
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());

    // Add documents
    store
        .add("doc1", "First document about Rust programming")
        .expect("Failed to add doc1");
    store
        .add("doc2", "Second document about Python programming")
        .expect("Failed to add doc2");
    store
        .add("doc3", "Third document about JavaScript")
        .expect("Failed to add doc3");

    assert_eq!(store.len(), 3);
    assert!(!store.is_empty());

    // Search for similar documents
    let results = store.search_text("Rust code", 2);
    assert_eq!(results.len(), 2);

    // First result should be about Rust (highest similarity)
    assert!(results[0].content.contains("Rust"));
}

/// Test document retrieval by ID.
#[test]
fn test_duckdb_get_document() {
    let embedder = HashEmbedder::new(64);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    store
        .add("my-id", "Test document content")
        .expect("Failed to add");

    let content = store.get("my-id").expect("Failed to get");
    assert!(content.is_some());
    assert_eq!(content.unwrap(), "Test document content");

    // Non-existent document
    let missing = store.get("nonexistent").expect("Failed to get");
    assert!(missing.is_none());
}

/// Test document update (upsert).
#[test]
fn test_duckdb_update_document() {
    let embedder = HashEmbedder::new(64);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    store
        .add("doc1", "Original content")
        .expect("Failed to add");
    assert_eq!(store.len(), 1);

    // Update same ID
    store
        .update("doc1", "Updated content")
        .expect("Failed to update");
    assert_eq!(store.len(), 1);

    let content = store.get("doc1").expect("Failed to get");
    assert_eq!(content.unwrap(), "Updated content");
}

/// Test document removal.
#[test]
fn test_duckdb_remove_document() {
    let embedder = HashEmbedder::new(64);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    store.add("doc1", "Content 1").expect("Failed to add");
    store.add("doc2", "Content 2").expect("Failed to add");
    assert_eq!(store.len(), 2);

    let removed = store.remove("doc1").expect("Failed to remove");
    assert!(removed);
    assert_eq!(store.len(), 1);

    // Remove non-existent
    let not_removed = store.remove("doc1").expect("Failed to remove");
    assert!(!not_removed);
}

/// Test clearing the store.
#[test]
fn test_duckdb_clear_store() {
    let embedder = HashEmbedder::new(64);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    store.add("doc1", "Content 1").expect("Failed to add");
    store.add("doc2", "Content 2").expect("Failed to add");
    store.add("doc3", "Content 3").expect("Failed to add");
    assert_eq!(store.len(), 3);

    store.clear().expect("Failed to clear");
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
}

/// Test batch add operations.
#[test]
fn test_duckdb_add_batch() {
    let embedder = HashEmbedder::new(64);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    let docs = vec![
        ("id1".to_string(), "Document one".to_string()),
        ("id2".to_string(), "Document two".to_string()),
        ("id3".to_string(), "Document three".to_string()),
    ];

    store.add_batch(docs).expect("Failed to add batch");
    assert_eq!(store.len(), 3);
}

/// Test embedding dimension consistency.
#[test]
fn test_duckdb_dimension() {
    let embedder = HashEmbedder::new(128);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    assert_eq!(store.dimension(), 128);
}

/// Test MutableVectorStore trait implementation.
#[test]
fn test_duckdb_mutable_vector_store_trait() {
    let embedder = HashEmbedder::new(64);
    let mut store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    // Use trait methods (these take &mut self)
    MutableVectorStore::add(&mut store, "doc1".to_string(), "Content one".to_string());
    MutableVectorStore::add(&mut store, "doc2".to_string(), "Content two".to_string());

    assert_eq!(store.len(), 2);

    MutableVectorStore::update(
        &mut store,
        "doc1".to_string(),
        "Updated content".to_string(),
    );
    assert_eq!(store.len(), 2);

    let removed = MutableVectorStore::remove(&mut store, "doc2");
    assert!(removed);
    assert_eq!(store.len(), 1);

    MutableVectorStore::clear(&mut store);
    assert!(store.is_empty());
}

/// Test search returns sorted results.
#[test]
fn test_duckdb_search_ordering() {
    let embedder = HashEmbedder::new(64);
    let store = DuckDBVectorStore::in_memory(embedder).expect("Failed to create store");

    // Add documents with varying similarity to "Rust"
    store
        .add("rust_doc", "Rust programming language guide")
        .expect("Failed to add");
    store
        .add("python_doc", "Python programming basics")
        .expect("Failed to add");
    store
        .add("rust_tutorial", "Learn Rust today")
        .expect("Failed to add");
    store
        .add("js_doc", "JavaScript for beginners")
        .expect("Failed to add");

    let results = store.search_text("Rust programming", 4);
    assert_eq!(results.len(), 4);

    // Results should be sorted by score descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results not sorted: {} < {}",
            results[i - 1].score,
            results[i].score
        );
    }
}

/// Test with file-based storage (creates temp file).
#[test]
fn test_duckdb_file_storage() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("kkachi_test_vectors.duckdb");

    // Clean up any previous test file
    let _ = std::fs::remove_file(&db_path);

    let embedder = HashEmbedder::new(64);

    // Create and populate store
    {
        let store = DuckDBVectorStore::open(&db_path, embedder.clone())
            .expect("Failed to create file store");

        store
            .add("persistent_doc", "This document should persist")
            .expect("Failed to add");

        assert_eq!(store.len(), 1);
    }

    // Reopen and verify data persists
    {
        let store =
            DuckDBVectorStore::open(&db_path, embedder).expect("Failed to reopen file store");

        assert_eq!(store.len(), 1);

        let content = store.get("persistent_doc").expect("Failed to get");
        assert_eq!(content.unwrap(), "This document should persist");
    }

    // Clean up
    let _ = std::fs::remove_file(&db_path);
}

/// Test custom table configuration.
#[test]
fn test_duckdb_custom_table() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("kkachi_test_custom_table.duckdb");
    let _ = std::fs::remove_file(&db_path);

    let embedder = HashEmbedder::new(32);
    let store = DuckDBVectorStore::with_table(
        &db_path,
        embedder,
        "custom_vectors",
        "text_content",
        "vec_embedding",
    )
    .expect("Failed to create custom table store");

    store.add("test", "Test content").expect("Failed to add");
    assert_eq!(store.len(), 1);

    let _ = std::fs::remove_file(&db_path);
}
