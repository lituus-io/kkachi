// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Document clustering using agglomerative clustering.
//!
//! Local clustering with O(n²) complexity for small document sets
//! (typically after initial retrieval of ~20 documents).

use smallvec::SmallVec;

use super::similarity::{DocumentFeatures, LocalSimilarity};

/// Cluster configuration.
#[derive(Clone, Copy, Debug)]
pub struct ClusterConfig {
    /// Minimum similarity threshold to cluster documents (default: 0.85).
    pub similarity_threshold: f32,
    /// Minimum cluster size to consider for condensation (default: 3).
    pub min_cluster_size: usize,
    /// Maximum documents per cluster (default: 10).
    pub max_cluster_size: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            min_cluster_size: 3,
            max_cluster_size: 10,
        }
    }
}

impl ClusterConfig {
    /// Create with a specific similarity threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            similarity_threshold: threshold,
            ..Default::default()
        }
    }

    /// Set minimum cluster size.
    pub fn with_min_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Set maximum cluster size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_cluster_size = size;
        self
    }
}

/// A cluster of similar documents.
#[derive(Debug)]
pub struct DocumentCluster {
    /// Indices of documents in this cluster.
    pub doc_indices: SmallVec<[usize; 8]>,
    /// Index of the centroid document (most similar to others).
    pub centroid_idx: usize,
    /// Average pairwise similarity within the cluster.
    pub avg_similarity: f32,
}

impl DocumentCluster {
    /// Get the number of documents in the cluster.
    #[inline]
    pub fn size(&self) -> usize {
        self.doc_indices.len()
    }

    /// Check if the cluster meets minimum size requirements.
    #[inline]
    pub fn is_significant(&self, min_size: usize) -> bool {
        self.doc_indices.len() >= min_size
    }
}

/// Local clustering engine using agglomerative clustering.
#[derive(Clone, Default)]
pub struct ClusterEngine {
    similarity: LocalSimilarity,
    config: ClusterConfig,
}

impl ClusterEngine {
    /// Create with custom configuration.
    pub fn with_config(config: ClusterConfig) -> Self {
        Self {
            similarity: LocalSimilarity::default(),
            config,
        }
    }

    /// Create with custom similarity scorer.
    pub fn with_similarity(similarity: LocalSimilarity, config: ClusterConfig) -> Self {
        Self { similarity, config }
    }

    /// Cluster documents using single-linkage agglomerative clustering.
    ///
    /// Returns clusters that meet the minimum size threshold.
    pub fn cluster<'a>(&self, docs: &[DocumentFeatures<'a>]) -> Vec<DocumentCluster> {
        if docs.len() < self.config.min_cluster_size {
            return vec![];
        }

        // Compute pairwise similarity matrix (upper triangular)
        let n = docs.len();
        let mut sim_matrix = vec![0.0f32; n * n];

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.similarity.score(&docs[i], &docs[j]);
                sim_matrix[i * n + j] = sim;
                sim_matrix[j * n + i] = sim;
            }
            sim_matrix[i * n + i] = 1.0; // Self-similarity
        }

        // Union-find for clustering
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        // Find with path compression
        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        // Union by rank
        fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                if rank[pi] < rank[pj] {
                    parent[pi] = pj;
                } else if rank[pi] > rank[pj] {
                    parent[pj] = pi;
                } else {
                    parent[pj] = pi;
                    rank[pi] += 1;
                }
            }
        }

        // Cluster documents with similarity above threshold
        for i in 0..n {
            for j in (i + 1)..n {
                if sim_matrix[i * n + j] >= self.config.similarity_threshold {
                    union(&mut parent, &mut rank, i, j);
                }
            }
        }

        // Group documents by cluster
        let mut cluster_map: std::collections::HashMap<usize, SmallVec<[usize; 8]>> =
            std::collections::HashMap::new();

        for i in 0..n {
            let root = find(&mut parent, i);
            cluster_map.entry(root).or_default().push(i);
        }

        // Build cluster objects
        let mut clusters: Vec<DocumentCluster> = cluster_map
            .into_iter()
            .filter(|(_, indices)| indices.len() >= self.config.min_cluster_size)
            .map(|(_, indices)| {
                let (centroid_idx, avg_similarity) =
                    self.compute_centroid_and_avg(&indices, &sim_matrix, n);
                DocumentCluster {
                    doc_indices: indices,
                    centroid_idx,
                    avg_similarity,
                }
            })
            .collect();

        // Sort by average similarity descending
        clusters.sort_by(|a, b| {
            b.avg_similarity
                .partial_cmp(&a.avg_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        clusters
    }

    /// Compute centroid (document most similar to all others) and average similarity.
    fn compute_centroid_and_avg(
        &self,
        indices: &[usize],
        sim_matrix: &[f32],
        n: usize,
    ) -> (usize, f32) {
        let mut best_idx = indices[0];
        let mut best_total = 0.0f32;
        let mut total_sim = 0.0f32;
        let mut pair_count = 0u32;

        for &i in indices {
            let mut doc_total = 0.0f32;
            for &j in indices {
                if i != j {
                    let sim = sim_matrix[i * n + j];
                    doc_total += sim;
                    total_sim += sim;
                    pair_count += 1;
                }
            }
            if doc_total > best_total {
                best_total = doc_total;
                best_idx = i;
            }
        }

        let avg_similarity = if pair_count > 0 {
            total_sim / pair_count as f32
        } else {
            1.0
        };

        (best_idx, avg_similarity)
    }

    /// Find the nearest cluster for a new document.
    pub fn find_nearest_cluster<'a>(
        &self,
        doc: &DocumentFeatures<'a>,
        docs: &[DocumentFeatures<'a>],
        clusters: &[DocumentCluster],
    ) -> Option<(usize, f32)> {
        let mut best_cluster = None;
        let mut best_sim = 0.0f32;

        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            // Compute average similarity to cluster members
            let mut total_sim = 0.0f32;
            for &doc_idx in &cluster.doc_indices {
                total_sim += self.similarity.score(doc, &docs[doc_idx]);
            }
            let avg_sim = total_sim / cluster.doc_indices.len() as f32;

            if avg_sim > best_sim && avg_sim >= self.config.similarity_threshold {
                best_sim = avg_sim;
                best_cluster = Some(cluster_idx);
            }
        }

        best_cluster.map(|idx| (idx, best_sim))
    }

    /// Merge two clusters.
    pub fn merge_clusters(a: &DocumentCluster, b: &DocumentCluster) -> DocumentCluster {
        let mut doc_indices = a.doc_indices.clone();
        doc_indices.extend(b.doc_indices.iter().copied());

        // The centroid will need to be recomputed with the full similarity matrix
        // For now, use the centroid from the larger cluster
        let centroid_idx = if a.size() >= b.size() {
            a.centroid_idx
        } else {
            b.centroid_idx
        };

        // Approximate average similarity
        let total_pairs = (a.size() + b.size()) * (a.size() + b.size() - 1) / 2;
        let a_pairs = a.size() * (a.size() - 1) / 2;
        let b_pairs = b.size() * (b.size() - 1) / 2;

        let avg_similarity = if total_pairs > 0 {
            (a.avg_similarity * a_pairs as f32 + b.avg_similarity * b_pairs as f32)
                / total_pairs as f32
        } else {
            (a.avg_similarity + b.avg_similarity) / 2.0
        };

        DocumentCluster {
            doc_indices,
            centroid_idx,
            avg_similarity,
        }
    }
}

/// Cluster statistics for monitoring.
#[derive(Debug, Clone, Copy)]
pub struct ClusterStats {
    /// Total number of clusters.
    pub num_clusters: usize,
    /// Total documents clustered.
    pub docs_clustered: usize,
    /// Average cluster size.
    pub avg_cluster_size: f32,
    /// Average within-cluster similarity.
    pub avg_similarity: f32,
}

impl ClusterStats {
    /// Compute statistics from clusters.
    pub fn from_clusters(clusters: &[DocumentCluster]) -> Self {
        if clusters.is_empty() {
            return Self {
                num_clusters: 0,
                docs_clustered: 0,
                avg_cluster_size: 0.0,
                avg_similarity: 0.0,
            };
        }

        let docs_clustered: usize = clusters.iter().map(|c| c.size()).sum();
        let avg_cluster_size = docs_clustered as f32 / clusters.len() as f32;
        let avg_similarity: f32 =
            clusters.iter().map(|c| c.avg_similarity).sum::<f32>() / clusters.len() as f32;

        Self {
            num_clusters: clusters.len(),
            docs_clustered,
            avg_cluster_size,
            avg_similarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    fn create_test_doc(
        id: u64,
        keywords: &[&str],
        _embedding: &[f32],
    ) -> DocumentFeatures<'static> {
        let mut doc = DocumentFeatures::new(id);

        for (i, kw) in keywords.iter().enumerate() {
            doc.keywords.push((sym(kw), 1.0 - i as f32 * 0.1));
        }
        doc.sort_keywords();

        // We can't store the embedding reference because it needs a longer lifetime
        // For testing, we'll use a static slice approach
        doc
    }

    #[test]
    fn test_cluster_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.min_cluster_size, 3);
        assert_eq!(config.max_cluster_size, 10);
    }

    #[test]
    fn test_cluster_config_builder() {
        let config = ClusterConfig::with_threshold(0.9)
            .with_min_size(2)
            .with_max_size(5);

        assert_eq!(config.similarity_threshold, 0.9);
        assert_eq!(config.min_cluster_size, 2);
        assert_eq!(config.max_cluster_size, 5);
    }

    #[test]
    fn test_cluster_empty_docs() {
        let engine = ClusterEngine::default();
        let docs: Vec<DocumentFeatures<'_>> = vec![];
        let clusters = engine.cluster(&docs);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_too_few_docs() {
        let engine = ClusterEngine::default();
        let docs = vec![
            create_test_doc(1, &["rust", "code"], &[]),
            create_test_doc(2, &["rust", "programming"], &[]),
        ];
        let clusters = engine.cluster(&docs);
        assert!(clusters.is_empty()); // Default min_cluster_size is 3
    }

    #[test]
    fn test_cluster_similar_docs() {
        let config = ClusterConfig::with_threshold(0.3).with_min_size(2);
        let engine = ClusterEngine::with_config(config);

        // Create docs with overlapping keywords (should cluster)
        let docs = vec![
            create_test_doc(1, &["rust", "async", "tokio"], &[]),
            create_test_doc(2, &["rust", "async", "futures"], &[]),
            create_test_doc(3, &["python", "asyncio", "web"], &[]),
        ];

        let clusters = engine.cluster(&docs);

        // Docs 0 and 1 share "rust" and "async", should cluster
        // Doc 2 is different
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_document_cluster_size() {
        let cluster = DocumentCluster {
            doc_indices: smallvec::smallvec![0, 1, 2],
            centroid_idx: 0,
            avg_similarity: 0.9,
        };

        assert_eq!(cluster.size(), 3);
        assert!(cluster.is_significant(2));
        assert!(cluster.is_significant(3));
        assert!(!cluster.is_significant(4));
    }

    #[test]
    fn test_cluster_stats() {
        let clusters = vec![
            DocumentCluster {
                doc_indices: smallvec::smallvec![0, 1, 2],
                centroid_idx: 0,
                avg_similarity: 0.9,
            },
            DocumentCluster {
                doc_indices: smallvec::smallvec![3, 4],
                centroid_idx: 3,
                avg_similarity: 0.8,
            },
        ];

        let stats = ClusterStats::from_clusters(&clusters);
        assert_eq!(stats.num_clusters, 2);
        assert_eq!(stats.docs_clustered, 5);
        assert!((stats.avg_cluster_size - 2.5).abs() < 0.001);
        assert!((stats.avg_similarity - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_cluster_stats_empty() {
        let stats = ClusterStats::from_clusters(&[]);
        assert_eq!(stats.num_clusters, 0);
        assert_eq!(stats.docs_clustered, 0);
        assert_eq!(stats.avg_cluster_size, 0.0);
        assert_eq!(stats.avg_similarity, 0.0);
    }

    #[test]
    fn test_merge_clusters() {
        let a = DocumentCluster {
            doc_indices: smallvec::smallvec![0, 1, 2],
            centroid_idx: 1,
            avg_similarity: 0.9,
        };
        let b = DocumentCluster {
            doc_indices: smallvec::smallvec![3, 4],
            centroid_idx: 3,
            avg_similarity: 0.8,
        };

        let merged = ClusterEngine::merge_clusters(&a, &b);
        assert_eq!(merged.size(), 5);
        assert_eq!(merged.centroid_idx, 1); // From larger cluster
    }
}
