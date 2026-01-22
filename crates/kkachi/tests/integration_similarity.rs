// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for similarity scoring and vector stores.

use kkachi::intern::sym;
use kkachi::recursive::{
    DocumentFeatures, Embedder, EmbeddingRef, HashEmbedder, InMemoryVectorStore, LocalSimilarity,
    SimilarityWeights, VectorStore,
};
use smallvec::smallvec;

#[test]
fn test_local_similarity_full_pipeline() {
    let similarity = LocalSimilarity::with_weights(SimilarityWeights::default());

    let emb_a: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
    let emb_b: [f32; 4] = [0.15, 0.22, 0.28, 0.42];

    // Create two similar documents using builder pattern
    let doc_a = DocumentFeatures::new(1)
        .with_keywords(smallvec![
            (sym("s3"), 0.9),
            (sym("bucket"), 0.8),
            (sym("terraform"), 0.85),
            (sym("create"), 0.5),
        ])
        .with_category_path(smallvec![sym("infrastructure"), sym("aws"), sym("storage"),])
        .with_tags(smallvec![sym("terraform"), sym("aws"), sym("s3")])
        .with_embedding(EmbeddingRef::new(&emb_a));

    let doc_b = DocumentFeatures::new(2)
        .with_keywords(smallvec![
            (sym("s3"), 0.88),
            (sym("bucket"), 0.82),
            (sym("terraform"), 0.9),
            (sym("aws"), 0.7),
        ])
        .with_category_path(smallvec![sym("infrastructure"), sym("aws"), sym("storage"),])
        .with_tags(smallvec![sym("terraform"), sym("aws"), sym("iac")])
        .with_embedding(EmbeddingRef::new(&emb_b));

    let score = similarity.score(&doc_a, &doc_b);

    // Should be highly similar
    assert!(score > 0.7, "Expected high similarity, got {}", score);
}

#[test]
fn test_local_similarity_different_domains() {
    let similarity = LocalSimilarity::with_weights(SimilarityWeights::default());

    let emb_terraform: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
    let emb_kubernetes: [f32; 4] = [0.8, 0.1, 0.05, 0.9];

    // Two documents from different domains
    let terraform_doc = DocumentFeatures::new(1)
        .with_keywords(smallvec![(sym("s3"), 0.9), (sym("terraform"), 0.85)])
        .with_category_path(smallvec![sym("infrastructure"), sym("aws")])
        .with_tags(smallvec![sym("terraform")])
        .with_embedding(EmbeddingRef::new(&emb_terraform));

    let kubernetes_doc = DocumentFeatures::new(2)
        .with_keywords(smallvec![
            (sym("kubernetes"), 0.9),
            (sym("deployment"), 0.85)
        ])
        .with_category_path(smallvec![sym("infrastructure"), sym("kubernetes")])
        .with_tags(smallvec![sym("k8s")])
        .with_embedding(EmbeddingRef::new(&emb_kubernetes));

    let score = similarity.score(&terraform_doc, &kubernetes_doc);

    // Should have low similarity
    assert!(score < 0.5, "Expected low similarity, got {}", score);
}

#[test]
fn test_similarity_weights_impact() {
    let emb_a: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    let emb_b: [f32; 4] = [0.0, 1.0, 0.0, 0.0]; // Orthogonal embedding

    // Test that different weights change the score
    let doc_a = DocumentFeatures::new(1)
        .with_keywords(smallvec![(sym("shared"), 0.9)])
        .with_category_path(smallvec![sym("cat")])
        .with_tags(smallvec![sym("tag")])
        .with_embedding(EmbeddingRef::new(&emb_a));

    let doc_b = DocumentFeatures::new(2)
        .with_keywords(smallvec![(sym("shared"), 0.9)]) // Same keywords
        .with_category_path(smallvec![sym("cat")]) // Same category
        .with_tags(smallvec![sym("tag")]) // Same tags
        .with_embedding(EmbeddingRef::new(&emb_b));

    // With default weights (embedding: 0.4)
    let default_sim = LocalSimilarity::with_weights(SimilarityWeights::default());
    let default_score = default_sim.score(&doc_a, &doc_b);

    // Default score should be non-negative
    assert!(
        default_score >= 0.0,
        "Default score {} should be non-negative",
        default_score
    );

    // With embedding-focused weights
    let embedding_sim = LocalSimilarity::with_weights(SimilarityWeights {
        embedding: 0.8,
        keyword: 0.1,
        metadata: 0.05,
        hierarchy: 0.05,
    });
    let embedding_score = embedding_sim.score(&doc_a, &doc_b);

    // With keyword-focused weights
    let keyword_sim = LocalSimilarity::with_weights(SimilarityWeights {
        embedding: 0.1,
        keyword: 0.7,
        metadata: 0.1,
        hierarchy: 0.1,
    });
    let keyword_score = keyword_sim.score(&doc_a, &doc_b);

    // Keyword-focused should score higher (keywords match, embeddings don't)
    assert!(
        keyword_score > embedding_score,
        "Keyword-focused {} should be higher than embedding-focused {}",
        keyword_score,
        embedding_score
    );
}

#[test]
fn test_in_memory_vector_store() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Add documents
    store.add("doc1".to_string(), "How to create S3 buckets".to_string());
    store.add("doc2".to_string(), "Creating S3 bucket in AWS".to_string());
    store.add("doc3".to_string(), "Kubernetes pod deployment".to_string());
    store.add("doc4".to_string(), "Docker container basics".to_string());

    assert_eq!(store.len(), 4);
    assert!(!store.is_empty());

    // Search for S3-related docs
    let results = store.search_text("S3 bucket creation", 2);

    assert_eq!(results.len(), 2);
    // Top results should be S3-related
    assert!(
        results[0].id == "doc1" || results[0].id == "doc2",
        "Expected S3 doc, got {}",
        results[0].id
    );
}

#[test]
fn test_vector_store_add_batch() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    let docs = vec![
        ("a".to_string(), "First document".to_string()),
        ("b".to_string(), "Second document".to_string()),
        ("c".to_string(), "Third document".to_string()),
    ];

    store.add_batch(docs);

    assert_eq!(store.len(), 3);
}

#[test]
fn test_vector_store_clear() {
    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    store.add("doc1".to_string(), "Content 1".to_string());
    store.add("doc2".to_string(), "Content 2".to_string());

    assert_eq!(store.len(), 2);

    store.clear();

    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
}

#[test]
fn test_hash_embedder_deterministic() {
    let embedder = HashEmbedder::new(64);

    let emb1 = embedder.embed("test query");
    let emb2 = embedder.embed("test query");

    // Same input should produce same embedding
    assert_eq!(emb1.as_slice(), emb2.as_slice());

    // Different input should produce different embedding
    let emb3 = embedder.embed("different query");
    assert_ne!(emb1.as_slice(), emb3.as_slice());
}

#[test]
fn test_embedding_cosine_similarity() {
    // Identical vectors
    let a = [1.0f32, 0.0, 0.0];
    let b = [1.0f32, 0.0, 0.0];
    let ref_a = EmbeddingRef::new(&a);
    let ref_b = EmbeddingRef::new(&b);
    let sim = ref_a.cosine_similarity(&ref_b);
    assert!((sim - 1.0).abs() < 0.001, "Expected 1.0, got {}", sim);

    // Orthogonal vectors
    let c = [1.0f32, 0.0, 0.0];
    let d = [0.0f32, 1.0, 0.0];
    let ref_c = EmbeddingRef::new(&c);
    let ref_d = EmbeddingRef::new(&d);
    let sim = ref_c.cosine_similarity(&ref_d);
    assert!(sim.abs() < 0.001, "Expected 0.0, got {}", sim);

    // Opposite vectors
    let e = [1.0f32, 0.0, 0.0];
    let f = [-1.0f32, 0.0, 0.0];
    let ref_e = EmbeddingRef::new(&e);
    let ref_f = EmbeddingRef::new(&f);
    let sim = ref_e.cosine_similarity(&ref_f);
    assert!((sim - (-1.0)).abs() < 0.001, "Expected -1.0, got {}", sim);
}
