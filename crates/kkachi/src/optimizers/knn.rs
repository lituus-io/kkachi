// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! KNN Few-Shot - K-Nearest Neighbor Demonstration Selection
//!
//! Selects demonstrations based on similarity to the input query
//! using embedding-based nearest neighbor search.
//!
//! ## Algorithm
//!
//! 1. Embed all training examples
//! 2. For each test input, embed the query
//! 3. Find k nearest neighbors by cosine similarity
//! 4. Use those as demonstrations

use crate::error::Result;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig};
use crate::str_view::StrView;
use smallvec::SmallVec;
use std::future::Future;
use std::pin::Pin;

/// Embedding provider trait.
pub trait Embedder: Send + Sync {
    /// Embed a single text.
    fn embed<'a>(
        &'a self,
        text: StrView<'a>,
        output: &'a mut Vec<f32>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Embedding dimension.
    fn dimension(&self) -> usize;
}

/// KNN Few-Shot configuration.
#[derive(Clone, Copy)]
pub struct KNNConfig {
    /// Base optimizer config
    pub base: OptimizerConfig,
    /// Number of neighbors to select
    pub k: u8,
    /// Whether to use weighted voting (by similarity)
    pub weighted: bool,
}

impl Default for KNNConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig::default(),
            k: 3,
            weighted: true,
        }
    }
}

impl KNNConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            base: OptimizerConfig::new(),
            k: 3,
            weighted: true,
        }
    }

    /// Set k (number of neighbors).
    pub const fn with_k(mut self, k: u8) -> Self {
        self.k = k;
        self
    }

    /// Set weighted mode.
    pub const fn with_weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }
}

/// Precomputed embedding index for fast retrieval.
pub struct EmbeddingIndex {
    /// Embeddings stored as contiguous f32 array
    embeddings: Vec<f32>,
    /// Embedding dimension
    dim: usize,
    /// Number of embeddings
    count: usize,
    /// L2 norms for each embedding (for fast cosine similarity)
    norms: Vec<f32>,
}

impl EmbeddingIndex {
    /// Create new empty index.
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: Vec::new(),
            dim,
            count: 0,
            norms: Vec::new(),
        }
    }

    /// Add an embedding to the index.
    pub fn add(&mut self, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);

        self.embeddings.extend_from_slice(embedding);

        // Compute L2 norm
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.norms.push(norm);

        self.count += 1;
    }

    /// Get embedding at index.
    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.embeddings[start..start + self.dim]
    }

    /// Number of embeddings.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Find k nearest neighbors using cosine similarity.
    pub fn find_nearest(&self, query: &[f32], k: usize) -> SmallVec<[(u32, f32); 8]> {
        debug_assert_eq!(query.len(), self.dim);

        // Compute query norm
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return SmallVec::new();
        }

        // Compute similarities to all embeddings
        let mut similarities: Vec<(u32, f32)> = (0..self.count)
            .map(|i| {
                let emb = self.get(i);
                let emb_norm = self.norms[i];

                if emb_norm == 0.0 {
                    return (i as u32, 0.0);
                }

                // Cosine similarity = dot(a, b) / (|a| * |b|)
                let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                let similarity = dot / (query_norm * emb_norm);

                (i as u32, similarity)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top k
        similarities.truncate(k);
        similarities.into_iter().collect()
    }

    /// Find k nearest neighbors using SIMD (when available).
    #[cfg(target_arch = "x86_64")]
    pub fn find_nearest_simd(&self, query: &[f32], k: usize) -> SmallVec<[(u32, f32); 8]> {
        // For now, fall back to scalar implementation
        // Full SIMD implementation would use AVX2/AVX-512
        self.find_nearest(query, k)
    }
}

/// KNN Few-Shot optimizer.
///
/// Selects demonstrations based on embedding similarity.
#[derive(Clone)]
pub struct KNNFewShot<E: Embedder> {
    config: KNNConfig,
    embedder: E,
}

impl<E: Embedder> KNNFewShot<E> {
    /// Create a new KNN Few-Shot optimizer.
    pub fn new(config: KNNConfig, embedder: E) -> Self {
        Self { config, embedder }
    }

    /// Get the configuration.
    pub fn config(&self) -> &KNNConfig {
        &self.config
    }

    /// Get the embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Build embedding index for a training set.
    pub async fn build_index<'a>(
        &self,
        trainset: &ExampleSet<'a>,
        buffer: &mut Vec<f32>,
    ) -> Result<EmbeddingIndex> {
        let dim = self.embedder.dimension();
        let mut index = EmbeddingIndex::new(dim);

        buffer.clear();
        buffer.resize(dim, 0.0);

        for example in trainset.iter() {
            // Combine input fields for embedding
            let text = example.input_text();

            self.embedder.embed(text, buffer).await?;
            index.add(buffer);
        }

        Ok(index)
    }

    /// Select demonstrations for a query.
    pub async fn select_demos<'a>(
        &self,
        query: StrView<'a>,
        index: &EmbeddingIndex,
        buffer: &mut Vec<f32>,
    ) -> Result<SmallVec<[u32; 8]>> {
        let dim = self.embedder.dimension();

        buffer.clear();
        buffer.resize(dim, 0.0);

        self.embedder.embed(query, buffer).await?;

        let neighbors = index.find_nearest(buffer, self.config.k as usize);
        Ok(neighbors.iter().map(|(idx, _)| *idx).collect())
    }
}

/// Prebuilt KNN selector (index already computed).
pub struct KNNSelector {
    config: KNNConfig,
    index: EmbeddingIndex,
}

impl KNNSelector {
    /// Create from prebuilt index.
    pub fn new(config: KNNConfig, index: EmbeddingIndex) -> Self {
        Self { config, index }
    }

    /// Select demonstrations for a query embedding.
    pub fn select(&self, query_embedding: &[f32]) -> SmallVec<[u32; 8]> {
        let neighbors = self
            .index
            .find_nearest(query_embedding, self.config.k as usize);
        neighbors.iter().map(|(idx, _)| *idx).collect()
    }

    /// Get the index.
    pub fn index(&self) -> &EmbeddingIndex {
        &self.index
    }
}

impl<E: Embedder> Optimizer for KNNFewShot<E> {
    type Output<'a>
        = OptimizationResult
    where
        E: 'a;
    type OptimizeFut<'a>
        = std::future::Ready<Result<OptimizationResult>>
    where
        E: 'a;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        // KNN is query-dependent, so basic optimize just returns first k
        let n = (self.config.k as usize).min(trainset.len());
        let indices: SmallVec<[u32; 8]> = (0..n as u32).collect();

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 0,
        }))
    }

    fn name(&self) -> &'static str {
        "KNNFewShot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEmbedder {
        dim: usize,
    }

    impl Embedder for MockEmbedder {
        fn embed<'a>(
            &'a self,
            text: StrView<'a>,
            output: &'a mut Vec<f32>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                output.clear();
                output.resize(self.dim, 0.0);

                // Simple hash-based embedding for testing
                let bytes = text.as_str().as_bytes();
                for (i, &b) in bytes.iter().enumerate() {
                    output[i % self.dim] += (b as f32) / 255.0;
                }

                // Normalize
                let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in output.iter_mut() {
                        *x /= norm;
                    }
                }

                Ok(())
            })
        }

        fn dimension(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn test_knn_creation() {
        let embedder = MockEmbedder { dim: 64 };
        let knn = KNNFewShot::new(KNNConfig::default(), embedder);
        assert_eq!(knn.name(), "KNNFewShot");
        assert_eq!(knn.config().k, 3);
    }

    #[test]
    fn test_knn_config() {
        let config = KNNConfig::new().with_k(5).with_weighted(false);
        assert_eq!(config.k, 5);
        assert!(!config.weighted);
    }

    #[test]
    fn test_embedding_index() {
        let mut index = EmbeddingIndex::new(4);

        // Add some embeddings
        index.add(&[1.0, 0.0, 0.0, 0.0]);
        index.add(&[0.0, 1.0, 0.0, 0.0]);
        index.add(&[0.7, 0.7, 0.0, 0.0]); // Between first two

        assert_eq!(index.len(), 3);

        // Query similar to first embedding
        let neighbors = index.find_nearest(&[0.9, 0.1, 0.0, 0.0], 2);
        assert_eq!(neighbors.len(), 2);

        // First neighbor should be embedding 0 (most similar)
        assert_eq!(neighbors[0].0, 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let mut index = EmbeddingIndex::new(3);

        // Orthogonal vectors
        index.add(&[1.0, 0.0, 0.0]);
        index.add(&[0.0, 1.0, 0.0]);
        index.add(&[0.0, 0.0, 1.0]);

        // Query exactly matches first
        let neighbors = index.find_nearest(&[1.0, 0.0, 0.0], 3);
        assert!(neighbors[0].1 > 0.99); // First should have similarity ~1.0
        assert!(neighbors[1].1.abs() < 0.01); // Others should be ~0.0
    }
}
