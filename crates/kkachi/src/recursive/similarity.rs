// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multi-signal similarity scoring for document retrieval.
//!
//! All computations are local (no external API calls):
//! - Embedding cosine similarity (SIMD-friendly)
//! - TF-IDF keyword Jaccard similarity
//! - Metadata tag overlap
//! - Hierarchical category distance

use std::cmp::Ordering;

use smallvec::SmallVec;

use crate::intern::Sym;

/// Weights for multi-signal similarity (all local computation).
#[derive(Clone, Copy, Debug)]
pub struct SimilarityWeights {
    /// Weight for embedding cosine similarity (default: 0.40).
    pub embedding: f32,
    /// Weight for TF-IDF keyword Jaccard (default: 0.25).
    pub keyword: f32,
    /// Weight for metadata tag overlap (default: 0.20).
    pub metadata: f32,
    /// Weight for hierarchical category distance (default: 0.15).
    pub hierarchy: f32,
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            embedding: 0.40,
            keyword: 0.25,
            metadata: 0.20,
            hierarchy: 0.15,
        }
    }
}

impl SimilarityWeights {
    /// Create weights emphasizing semantic similarity.
    pub fn semantic_focused() -> Self {
        Self {
            embedding: 0.60,
            keyword: 0.20,
            metadata: 0.10,
            hierarchy: 0.10,
        }
    }

    /// Create weights emphasizing keyword matching.
    pub fn keyword_focused() -> Self {
        Self {
            embedding: 0.25,
            keyword: 0.50,
            metadata: 0.15,
            hierarchy: 0.10,
        }
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalize(&mut self) {
        let sum = self.embedding + self.keyword + self.metadata + self.hierarchy;
        if sum > 0.0 {
            self.embedding /= sum;
            self.keyword /= sum;
            self.metadata /= sum;
            self.hierarchy /= sum;
        }
    }
}

/// Zero-copy reference to an embedding vector.
#[derive(Clone, Copy)]
pub struct EmbeddingRef<'a> {
    data: &'a [f32],
}

impl<'a> EmbeddingRef<'a> {
    /// Create a new embedding reference.
    #[inline]
    pub fn new(data: &'a [f32]) -> Self {
        Self { data }
    }

    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }

    /// Get the dimension of the embedding.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Compute cosine similarity with another embedding.
    #[inline]
    pub fn cosine_similarity(&self, other: &EmbeddingRef<'_>) -> f32 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }

        let (dot, norm_a, norm_b) = self.dot_and_norms(other);

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Compute dot product and squared norms in a single pass.
    #[inline]
    fn dot_and_norms(&self, other: &EmbeddingRef<'_>) -> (f32, f32, f32) {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // Process in chunks of 4 for better cache utilization
        let chunks = self.data.len() / 4;
        let remainder = self.data.len() % 4;

        for i in 0..chunks {
            let base = i * 4;
            let a0 = self.data[base];
            let a1 = self.data[base + 1];
            let a2 = self.data[base + 2];
            let a3 = self.data[base + 3];
            let b0 = other.data[base];
            let b1 = other.data[base + 1];
            let b2 = other.data[base + 2];
            let b3 = other.data[base + 3];

            dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
            norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        }

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            let a = self.data[base + i];
            let b = other.data[base + i];
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        (dot, norm_a, norm_b)
    }

    /// Compute L2 (Euclidean) distance.
    #[inline]
    pub fn l2_distance(&self, other: &EmbeddingRef<'_>) -> f32 {
        if self.data.len() != other.data.len() {
            return f32::MAX;
        }

        let mut sum = 0.0f32;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let diff = a - b;
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Document features for multi-signal similarity (zero-copy where possible).
pub struct DocumentFeatures<'a> {
    /// Document identifier.
    pub id: u64,
    /// Keywords with TF-IDF scores (sorted by Sym for fast Jaccard).
    pub keywords: SmallVec<[(Sym, f32); 16]>,
    /// Hierarchical category path (e.g., ["infrastructure", "aws", "storage"]).
    pub category_path: SmallVec<[Sym; 8]>,
    /// Flat metadata tags.
    pub tags: SmallVec<[Sym; 8]>,
    /// Optional embedding reference.
    pub embedding: Option<EmbeddingRef<'a>>,
}

impl<'a> DocumentFeatures<'a> {
    /// Create new document features.
    pub fn new(id: u64) -> Self {
        Self {
            id,
            keywords: SmallVec::new(),
            category_path: SmallVec::new(),
            tags: SmallVec::new(),
            embedding: None,
        }
    }

    /// Add a keyword with TF-IDF score.
    pub fn with_keyword(mut self, keyword: Sym, tfidf: f32) -> Self {
        self.keywords.push((keyword, tfidf));
        self
    }

    /// Set keywords (must be sorted by Sym).
    pub fn with_keywords(mut self, keywords: SmallVec<[(Sym, f32); 16]>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Set category path.
    pub fn with_category_path(mut self, path: SmallVec<[Sym; 8]>) -> Self {
        self.category_path = path;
        self
    }

    /// Set tags.
    pub fn with_tags(mut self, tags: SmallVec<[Sym; 8]>) -> Self {
        self.tags = tags;
        self
    }

    /// Set embedding.
    pub fn with_embedding(mut self, embedding: EmbeddingRef<'a>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Sort keywords by Sym for efficient Jaccard computation.
    pub fn sort_keywords(&mut self) {
        self.keywords.sort_by_key(|(sym, _)| *sym);
    }
}

/// Local similarity scorer (no external API calls).
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalSimilarity {
    weights: SimilarityWeights,
}

impl LocalSimilarity {
    /// Create with custom weights.
    pub fn with_weights(weights: SimilarityWeights) -> Self {
        Self { weights }
    }

    /// Compute multi-signal similarity score between two documents.
    #[inline]
    pub fn score(&self, a: &DocumentFeatures<'_>, b: &DocumentFeatures<'_>) -> f32 {
        let embedding_score = match (&a.embedding, &b.embedding) {
            (Some(ea), Some(eb)) => ea.cosine_similarity(eb),
            _ => 0.0,
        };

        self.weights.embedding * embedding_score
            + self.weights.keyword * self.keyword_jaccard(&a.keywords, &b.keywords)
            + self.weights.metadata * self.tag_overlap(&a.tags, &b.tags)
            + self.weights.hierarchy * self.hierarchy_similarity(&a.category_path, &b.category_path)
    }

    /// O(n) sorted-merge Jaccard (zero allocation).
    /// Keywords must be sorted by Sym.
    #[inline]
    pub fn keyword_jaccard(&self, a: &[(Sym, f32)], b: &[(Sym, f32)]) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let (mut i, mut j) = (0, 0);
        let mut intersection = 0u32;
        let mut union = 0u32;

        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                Ordering::Equal => {
                    intersection += 1;
                    union += 1;
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    union += 1;
                    i += 1;
                }
                Ordering::Greater => {
                    union += 1;
                    j += 1;
                }
            }
        }

        // Add remaining elements to union
        union += (a.len() - i + b.len() - j) as u32;

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Weighted Jaccard that considers TF-IDF scores.
    #[inline]
    pub fn weighted_keyword_jaccard(&self, a: &[(Sym, f32)], b: &[(Sym, f32)]) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let (mut i, mut j) = (0, 0);
        let mut intersection_sum = 0.0f32;
        let mut union_sum = 0.0f32;

        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                Ordering::Equal => {
                    // Take minimum for intersection, maximum for union
                    intersection_sum += a[i].1.min(b[j].1);
                    union_sum += a[i].1.max(b[j].1);
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    union_sum += a[i].1;
                    i += 1;
                }
                Ordering::Greater => {
                    union_sum += b[j].1;
                    j += 1;
                }
            }
        }

        // Add remaining elements to union
        while i < a.len() {
            union_sum += a[i].1;
            i += 1;
        }
        while j < b.len() {
            union_sum += b[j].1;
            j += 1;
        }

        if union_sum == 0.0 {
            0.0
        } else {
            intersection_sum / union_sum
        }
    }

    /// Hierarchy similarity: shared prefix depth / max depth.
    #[inline]
    pub fn hierarchy_similarity(&self, a: &[Sym], b: &[Sym]) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }

        let shared = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
        let max_depth = a.len().max(b.len());

        if max_depth == 0 {
            1.0
        } else {
            shared as f32 / max_depth as f32
        }
    }

    /// Tag overlap (Jaccard on unordered sets).
    #[inline]
    pub fn tag_overlap(&self, a: &[Sym], b: &[Sym]) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let intersection = a.iter().filter(|t| b.contains(t)).count();
        let union = a.len() + b.len() - intersection;

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    #[test]
    fn test_similarity_weights_default() {
        let weights = SimilarityWeights::default();
        let sum = weights.embedding + weights.keyword + weights.metadata + weights.hierarchy;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_weights_normalize() {
        let mut weights = SimilarityWeights {
            embedding: 2.0,
            keyword: 1.0,
            metadata: 1.0,
            hierarchy: 0.0,
        };
        weights.normalize();
        assert!((weights.embedding - 0.5).abs() < 0.001);
        assert!((weights.keyword - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_embedding_cosine_similarity() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let ea = EmbeddingRef::new(&a);
        let eb = EmbeddingRef::new(&b);
        assert!((ea.cosine_similarity(&eb) - 1.0).abs() < 0.001);

        let c = [0.0, 1.0, 0.0];
        let ec = EmbeddingRef::new(&c);
        assert!(ea.cosine_similarity(&ec).abs() < 0.001);

        let d = [-1.0, 0.0, 0.0];
        let ed = EmbeddingRef::new(&d);
        assert!((ea.cosine_similarity(&ed) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_l2_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let ea = EmbeddingRef::new(&a);
        let eb = EmbeddingRef::new(&b);
        assert!((ea.l2_distance(&eb) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_keyword_jaccard() {
        let scorer = LocalSimilarity::default();

        let a = vec![(sym("rust"), 0.8), (sym("code"), 0.5)];
        let b = vec![(sym("code"), 0.6), (sym("rust"), 0.7)];

        // Sort for Jaccard
        let mut a_sorted = a.clone();
        let mut b_sorted = b.clone();
        a_sorted.sort_by_key(|(s, _)| *s);
        b_sorted.sort_by_key(|(s, _)| *s);

        let jaccard = scorer.keyword_jaccard(&a_sorted, &b_sorted);
        assert!((jaccard - 1.0).abs() < 0.001); // Same keywords

        let c = vec![(sym("python"), 0.9)];
        let jaccard2 = scorer.keyword_jaccard(&a_sorted, &c);
        assert!(jaccard2 < 0.5); // Different keywords
    }

    #[test]
    fn test_hierarchy_similarity() {
        let scorer = LocalSimilarity::default();

        let a: SmallVec<[Sym; 8]> =
            smallvec::smallvec![sym("infrastructure"), sym("aws"), sym("storage")];
        let b: SmallVec<[Sym; 8]> =
            smallvec::smallvec![sym("infrastructure"), sym("aws"), sym("compute")];
        let c: SmallVec<[Sym; 8]> =
            smallvec::smallvec![sym("infrastructure"), sym("gcp"), sym("storage")];

        let sim_ab = scorer.hierarchy_similarity(&a, &b);
        let sim_ac = scorer.hierarchy_similarity(&a, &c);

        assert!(sim_ab > sim_ac); // Same first two levels vs same first level
        assert!((sim_ab - 2.0 / 3.0).abs() < 0.001);
        assert!((sim_ac - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_tag_overlap() {
        let scorer = LocalSimilarity::default();

        let a: SmallVec<[Sym; 8]> = smallvec::smallvec![sym("terraform"), sym("aws")];
        let b: SmallVec<[Sym; 8]> = smallvec::smallvec![sym("terraform"), sym("gcp")];
        let c: SmallVec<[Sym; 8]> = smallvec::smallvec![sym("kubernetes"), sym("helm")];

        let overlap_ab = scorer.tag_overlap(&a, &b);
        let overlap_ac = scorer.tag_overlap(&a, &c);

        assert!((overlap_ab - 1.0 / 3.0).abs() < 0.001); // 1 shared out of 3 unique
        assert!(overlap_ac.abs() < 0.001); // No overlap
    }

    #[test]
    fn test_document_features_builder() {
        let emb = [0.1, 0.2, 0.3];
        let doc = DocumentFeatures::new(42)
            .with_keyword(sym("rust"), 0.9)
            .with_keyword(sym("code"), 0.7)
            .with_category_path(smallvec::smallvec![sym("lang"), sym("systems")])
            .with_tags(smallvec::smallvec![sym("fast"), sym("safe")])
            .with_embedding(EmbeddingRef::new(&emb));

        assert_eq!(doc.id, 42);
        assert_eq!(doc.keywords.len(), 2);
        assert_eq!(doc.category_path.len(), 2);
        assert_eq!(doc.tags.len(), 2);
        assert!(doc.embedding.is_some());
    }

    #[test]
    fn test_full_similarity_score() {
        let scorer = LocalSimilarity::default();

        let emb_a = [1.0, 0.0, 0.0, 0.0];
        let emb_b = [0.9, 0.1, 0.0, 0.0];

        let mut doc_a = DocumentFeatures::new(1)
            .with_keywords(smallvec::smallvec![(sym("rust"), 0.8), (sym("async"), 0.6)])
            .with_category_path(smallvec::smallvec![sym("lang"), sym("systems")])
            .with_tags(smallvec::smallvec![sym("fast")])
            .with_embedding(EmbeddingRef::new(&emb_a));
        doc_a.sort_keywords();

        let mut doc_b = DocumentFeatures::new(2)
            .with_keywords(smallvec::smallvec![(sym("rust"), 0.7), (sym("tokio"), 0.5)])
            .with_category_path(smallvec::smallvec![sym("lang"), sym("systems")])
            .with_tags(smallvec::smallvec![sym("fast"), sym("concurrent")])
            .with_embedding(EmbeddingRef::new(&emb_b));
        doc_b.sort_keywords();

        let score = scorer.score(&doc_a, &doc_b);
        assert!(score > 0.3); // Should have decent similarity
        assert!(score < 1.0); // Not identical
    }
}
