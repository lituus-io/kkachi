// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Smart retrieval pipeline with clustering and condensation.
//!
//! Integrates multi-signal similarity, clustering, and condensation
//! into a cohesive retrieval system.

use crate::str_view::StrView;

use super::cluster::{ClusterConfig, ClusterEngine};
use super::condense::{CondenseEngine, CondensedDocument};
use super::keywords::KeywordExtractor;
use super::similarity::{DocumentFeatures, LocalSimilarity, SimilarityWeights};
use super::storage::ContextId;

/// Smart retrieval configuration.
#[derive(Clone, Debug)]
pub struct RetrievalConfig {
    /// Initial number of documents to retrieve (default: 20).
    pub initial_k: usize,
    /// Final number of documents to return (default: 5).
    pub final_k: usize,
    /// Minimum similarity threshold for cache hit (default: 0.95).
    pub similarity_threshold: f32,
    /// Threshold for clustering similar documents (default: 0.80).
    pub cluster_threshold: f32,
    /// Minimum cluster size for condensation (default: 3).
    pub min_cluster_size: usize,
    /// Whether to automatically condense clusters (default: true).
    pub auto_condense: bool,
    /// Whether to use semantic cache (default: true).
    pub use_semantic_cache: bool,
    /// Similarity weights for multi-signal scoring.
    pub similarity_weights: SimilarityWeights,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            initial_k: 20,
            final_k: 5,
            similarity_threshold: 0.95,
            cluster_threshold: 0.80,
            min_cluster_size: 3,
            auto_condense: true,
            use_semantic_cache: true,
            similarity_weights: SimilarityWeights::default(),
        }
    }
}

impl RetrievalConfig {
    /// Create with a specific similarity threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            similarity_threshold: threshold,
            ..Default::default()
        }
    }

    /// Set initial retrieval count.
    pub fn with_initial_k(mut self, k: usize) -> Self {
        self.initial_k = k;
        self
    }

    /// Set final result count.
    pub fn with_final_k(mut self, k: usize) -> Self {
        self.final_k = k;
        self
    }

    /// Enable or disable auto-condensation.
    pub fn with_auto_condense(mut self, enabled: bool) -> Self {
        self.auto_condense = enabled;
        self
    }

    /// Enable or disable semantic cache.
    pub fn with_semantic_cache(mut self, enabled: bool) -> Self {
        self.use_semantic_cache = enabled;
        self
    }

    /// Set similarity weights.
    pub fn with_weights(mut self, weights: SimilarityWeights) -> Self {
        self.similarity_weights = weights;
        self
    }

    /// Set cluster threshold.
    pub fn with_cluster_threshold(mut self, threshold: f32) -> Self {
        self.cluster_threshold = threshold;
        self
    }
}

/// A cached context from semantic lookup.
#[derive(Clone, Debug)]
pub struct CachedContext {
    /// The stored answer.
    pub answer: String,
    /// Concise summary.
    pub summary: String,
    /// Quality score.
    pub score: f32,
    /// Number of iterations taken.
    pub iterations: u16,
    /// Similarity to the query.
    pub similarity: f32,
    /// Context ID.
    pub context_id: ContextId,
}

/// A single retrieved document.
#[derive(Clone, Debug)]
pub struct RetrievedDoc {
    /// Document identifier.
    pub id: u64,
    /// Document content.
    pub content: String,
    /// Relevance score to query.
    pub score: f32,
    /// Source type.
    pub source: DocSource,
}

/// Source of a retrieved document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocSource {
    /// From context storage.
    Storage,
    /// From external vector store.
    VectorStore,
    /// From file system.
    FileSystem,
    /// Condensed from multiple sources.
    Condensed,
}

/// Result from a retrieval operation.
#[derive(Debug)]
pub enum RetrievalResult {
    /// Found a high-quality cached result.
    Cached(CachedContext),
    /// Fresh retrieval results.
    Fresh(FreshRetrieval),
}

impl RetrievalResult {
    /// Check if result is from cache.
    pub fn is_cached(&self) -> bool {
        matches!(self, RetrievalResult::Cached(_))
    }

    /// Get documents (empty if cached).
    pub fn documents(&self) -> &[RetrievedDoc] {
        match self {
            RetrievalResult::Cached(_) => &[],
            RetrievalResult::Fresh(fresh) => &fresh.documents,
        }
    }

    /// Get cached answer if available.
    pub fn cached_answer(&self) -> Option<&str> {
        match self {
            RetrievalResult::Cached(ctx) => Some(&ctx.answer),
            RetrievalResult::Fresh(_) => None,
        }
    }
}

/// Fresh retrieval with documents and metadata.
#[derive(Debug)]
pub struct FreshRetrieval {
    /// Retrieved documents.
    pub documents: Vec<RetrievedDoc>,
    /// Condensed documents (if auto_condense enabled).
    pub condensed: Vec<CondensedDocument>,
    /// Clusters found (for debugging/analysis).
    pub clusters: Vec<ClusterInfo>,
    /// Retrieval statistics.
    pub stats: RetrievalStats,
}

/// Information about a cluster (for debugging).
#[derive(Clone, Debug)]
pub struct ClusterInfo {
    /// Number of documents in cluster.
    pub size: usize,
    /// Average similarity within cluster.
    pub avg_similarity: f32,
    /// Whether it was condensed.
    pub was_condensed: bool,
}

/// Statistics from retrieval operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RetrievalStats {
    /// Documents initially retrieved.
    pub initial_count: usize,
    /// Documents after filtering.
    pub filtered_count: usize,
    /// Clusters formed.
    pub cluster_count: usize,
    /// Documents condensed.
    pub condensed_count: usize,
    /// Final document count.
    pub final_count: usize,
    /// Time spent in retrieval (microseconds).
    pub retrieval_time_us: u64,
    /// Time spent in clustering (microseconds).
    pub cluster_time_us: u64,
}

/// Trait for embedding documents/queries.
pub trait Embedder: Send + Sync {
    /// Embed a text string.
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Embedding dimension.
    fn dimension(&self) -> usize;
}

/// Smart retriever with multi-signal search and auto-condensation.
pub struct SmartRetriever<'a, E: Embedder> {
    /// Embedder for semantic similarity (reserved for future semantic search).
    #[allow(dead_code)]
    embedder: &'a E,
    /// Keyword extractor.
    keyword_extractor: KeywordExtractor,
    /// Similarity scorer.
    similarity: LocalSimilarity,
    /// Cluster engine.
    cluster_engine: ClusterEngine,
    /// Condense engine.
    condense_engine: CondenseEngine,
    /// Configuration.
    config: RetrievalConfig,
}

impl<'a, E: Embedder> SmartRetriever<'a, E> {
    /// Create a new smart retriever.
    pub fn new(embedder: &'a E, config: RetrievalConfig) -> Self {
        let similarity = LocalSimilarity::with_weights(config.similarity_weights);
        let cluster_config = ClusterConfig::with_threshold(config.cluster_threshold)
            .with_min_size(config.min_cluster_size);

        Self {
            embedder,
            keyword_extractor: KeywordExtractor::new(),
            similarity,
            cluster_engine: ClusterEngine::with_config(cluster_config),
            condense_engine: CondenseEngine::default(),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(embedder: &'a E) -> Self {
        Self::new(embedder, RetrievalConfig::default())
    }

    /// Process retrieved documents with clustering and condensation.
    ///
    /// Takes raw retrieved documents and their contents, returns processed results.
    pub fn process(&self, docs: Vec<RetrievedDoc>, contents: &[&str]) -> FreshRetrieval {
        let initial_count = docs.len();

        if docs.is_empty() {
            return FreshRetrieval {
                documents: vec![],
                condensed: vec![],
                clusters: vec![],
                stats: RetrievalStats::default(),
            };
        }

        // Build document features for clustering (keywords only for now)
        // Note: For full embedding-based clustering, we'd need to store embeddings
        // alongside features with proper lifetime management
        let keyword_features: Vec<DocumentFeatures<'static>> = docs
            .iter()
            .zip(contents.iter())
            .map(|(doc, content)| {
                let keywords = self.keyword_extractor.extract_sorted(content);
                DocumentFeatures::new(doc.id).with_keywords(keywords)
            })
            .collect();

        // Cluster similar documents
        let clusters = self.cluster_engine.cluster(&keyword_features);
        let cluster_count = clusters.len();

        // Build cluster info
        let cluster_infos: Vec<ClusterInfo> = clusters
            .iter()
            .map(|c| ClusterInfo {
                size: c.size(),
                avg_similarity: c.avg_similarity,
                was_condensed: self.config.auto_condense
                    && c.size() >= self.config.min_cluster_size,
            })
            .collect();

        // Condense clusters if enabled
        let mut condensed = Vec::new();
        let mut condensed_doc_ids: std::collections::HashSet<u64> =
            std::collections::HashSet::new();

        if self.config.auto_condense {
            for cluster in &clusters {
                if cluster.size() >= self.config.min_cluster_size {
                    // Gather contents for this cluster
                    let cluster_contents: Vec<StrView<'_>> = cluster
                        .doc_indices
                        .iter()
                        .filter_map(|&i| contents.get(i).map(|c| StrView::new(c)))
                        .collect();

                    // Simple condensation (without LLM)
                    let merged_content = self.condense_engine.condense_simple(&cluster_contents);

                    let condensed_doc = self.condense_engine.create_condensed(
                        cluster,
                        &keyword_features,
                        merged_content,
                    );

                    // Track which docs were condensed
                    for &idx in &cluster.doc_indices {
                        if let Some(doc) = docs.get(idx) {
                            condensed_doc_ids.insert(doc.id);
                        }
                    }

                    condensed.push(condensed_doc);
                }
            }
        }

        // Filter out condensed documents from results
        let mut final_docs: Vec<RetrievedDoc> = docs
            .into_iter()
            .filter(|d| !condensed_doc_ids.contains(&d.id))
            .collect();

        // Add condensed documents as single entries
        for (i, c) in condensed.iter().enumerate() {
            final_docs.push(RetrievedDoc {
                id: u64::MAX - i as u64, // Use high IDs for condensed docs
                content: c.content.clone(),
                score: c.confidence,
                source: DocSource::Condensed,
            });
        }

        // Sort by score and truncate to final_k
        final_docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_docs.truncate(self.config.final_k);

        let final_count = final_docs.len();
        let condensed_count = condensed_doc_ids.len();

        FreshRetrieval {
            documents: final_docs,
            condensed,
            clusters: cluster_infos,
            stats: RetrievalStats {
                initial_count,
                filtered_count: initial_count,
                cluster_count,
                condensed_count,
                final_count,
                retrieval_time_us: 0,
                cluster_time_us: 0,
            },
        }
    }

    /// Score a document against a query.
    pub fn score_document(&self, query: &str, document: &str) -> f32 {
        let query_keywords = self.keyword_extractor.extract_sorted(query);
        let doc_keywords = self.keyword_extractor.extract_sorted(document);

        let query_features = DocumentFeatures::new(0).with_keywords(query_keywords);
        let doc_features = DocumentFeatures::new(1).with_keywords(doc_keywords);

        self.similarity.score(&query_features, &doc_features)
    }

    /// Re-rank documents by similarity to query.
    pub fn rerank(&self, query: &str, docs: &mut [RetrievedDoc], contents: &[&str]) {
        let query_keywords = self.keyword_extractor.extract_sorted(query);
        let query_features = DocumentFeatures::new(0).with_keywords(query_keywords);

        for (doc, content) in docs.iter_mut().zip(contents.iter()) {
            let doc_keywords = self.keyword_extractor.extract_sorted(content);
            let doc_features = DocumentFeatures::new(doc.id).with_keywords(doc_keywords);
            doc.score = self.similarity.score(&query_features, &doc_features);
        }

        docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get configuration.
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }
}

/// Simple embedder that returns zero vectors (for testing).
#[derive(Clone, Copy, Debug, Default)]
pub struct ZeroEmbedder {
    dim: usize,
}

impl ZeroEmbedder {
    /// Create with specific dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for ZeroEmbedder {
    fn embed(&self, _text: &str) -> Vec<f32> {
        vec![0.0; self.dim]
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Hash-based embedder for testing (deterministic, not semantic).
#[derive(Clone, Copy, Debug)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// Create with specific dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut result = vec![0.0f32; self.dim];

        for (i, word) in text.split_whitespace().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.to_lowercase().hash(&mut hasher);
            let hash = hasher.finish();

            // Distribute hash across embedding dimensions
            for j in 0..self.dim {
                let idx = (i + j) % self.dim;
                result[idx] += ((hash >> (j % 64)) & 0xFF) as f32 / 255.0;
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        result
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

// ============================================================================
// VectorStore Trait for External RAG Sources
// ============================================================================

/// Result from a vector store search.
#[derive(Clone, Debug)]
pub struct VectorSearchResult {
    /// Document identifier.
    pub id: String,
    /// Document content.
    pub content: String,
    /// Similarity score (0.0 - 1.0).
    pub score: f32,
    /// Optional metadata.
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

impl VectorSearchResult {
    /// Create a new search result.
    pub fn new(id: impl Into<String>, content: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            score,
            metadata: None,
        }
    }

    /// Add metadata.
    pub fn with_metadata(mut self, metadata: std::collections::HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Convert to RetrievedDoc.
    pub fn to_retrieved_doc(&self, numeric_id: u64) -> RetrievedDoc {
        RetrievedDoc {
            id: numeric_id,
            content: self.content.clone(),
            score: self.score,
            source: DocSource::VectorStore,
        }
    }
}

/// Trait for external vector stores (e.g., DuckDB, Pinecone, Qdrant).
///
/// This trait provides a unified interface for searching vector stores,
/// allowing the SmartRetriever to work with various backends.
///
/// # Example
///
/// ```ignore
/// use kkachi::recursive::VectorStore;
///
/// struct MyVectorStore { /* ... */ }
///
/// impl VectorStore for MyVectorStore {
///     fn search(&self, embedding: &[f32], k: usize) -> Vec<VectorSearchResult> {
///         // Perform search against your vector store
///     }
///     // ...
/// }
/// ```
pub trait VectorStore: Send + Sync {
    /// Search by embedding vector.
    ///
    /// Returns the top-k most similar documents.
    fn search(&self, embedding: &[f32], k: usize) -> Vec<VectorSearchResult>;

    /// Search by text query (embeds internally and searches).
    ///
    /// Returns the top-k most similar documents.
    fn search_text(&self, query: &str, k: usize) -> Vec<VectorSearchResult>;

    /// Get the embedding dimension expected by this store.
    fn dimension(&self) -> usize;

    /// Get the number of documents in the store.
    fn len(&self) -> usize;

    /// Check if the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory vector store for testing and small-scale use.
pub struct InMemoryVectorStore<E: Embedder> {
    /// Embedder for text queries.
    embedder: E,
    /// Stored documents: (id, content, embedding).
    documents: Vec<(String, String, Vec<f32>)>,
}

impl<E: Embedder> InMemoryVectorStore<E> {
    /// Create a new empty store.
    pub fn new(embedder: E) -> Self {
        Self {
            embedder,
            documents: Vec::new(),
        }
    }

    /// Add a document to the store.
    pub fn add(&mut self, id: impl Into<String>, content: impl Into<String>) {
        let content = content.into();
        let embedding = self.embedder.embed(&content);
        self.documents.push((id.into(), content, embedding));
    }

    /// Add multiple documents.
    pub fn add_batch(
        &mut self,
        docs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) {
        for (id, content) in docs {
            self.add(id, content);
        }
    }

    /// Clear all documents.
    pub fn clear(&mut self) {
        self.documents.clear();
    }

    /// Remove a document by ID.
    ///
    /// Returns true if a document was removed, false if not found.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self
            .documents
            .iter()
            .position(|(doc_id, _, _)| doc_id == id)
        {
            self.documents.remove(pos);
            true
        } else {
            false
        }
    }

    /// Update a document (remove + add).
    pub fn update(&mut self, id: impl Into<String>, content: impl Into<String>) {
        let id = id.into();
        self.remove(&id);
        self.add(id, content);
    }

    /// Get document by ID.
    pub fn get(&self, id: &str) -> Option<&str> {
        self.documents
            .iter()
            .find(|(doc_id, _, _)| doc_id == id)
            .map(|(_, content, _)| content.as_str())
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl<E: Embedder> VectorStore for InMemoryVectorStore<E> {
    fn search(&self, embedding: &[f32], k: usize) -> Vec<VectorSearchResult> {
        if self.documents.is_empty() {
            return vec![];
        }

        // Score all documents
        let mut scored: Vec<(usize, f32)> = self
            .documents
            .iter()
            .enumerate()
            .map(|(i, (_, _, emb))| (i, Self::cosine_similarity(embedding, emb)))
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        scored
            .into_iter()
            .take(k)
            .map(|(i, score)| {
                let (id, content, _) = &self.documents[i];
                VectorSearchResult::new(id.clone(), content.clone(), score)
            })
            .collect()
    }

    fn search_text(&self, query: &str, k: usize) -> Vec<VectorSearchResult> {
        let embedding = self.embedder.embed(query);
        self.search(&embedding, k)
    }

    fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    fn len(&self) -> usize {
        self.documents.len()
    }
}

/// A vector store that combines multiple stores.
pub struct CompositeVectorStore<'a> {
    stores: Vec<&'a dyn VectorStore>,
}

impl<'a> CompositeVectorStore<'a> {
    /// Create from multiple stores.
    pub fn new(stores: Vec<&'a dyn VectorStore>) -> Self {
        Self { stores }
    }

    /// Add a store.
    pub fn add_store(&mut self, store: &'a dyn VectorStore) {
        self.stores.push(store);
    }
}

impl VectorStore for CompositeVectorStore<'_> {
    fn search(&self, embedding: &[f32], k: usize) -> Vec<VectorSearchResult> {
        // Collect results from all stores
        let mut all_results: Vec<VectorSearchResult> = self
            .stores
            .iter()
            .flat_map(|store| store.search(embedding, k))
            .collect();

        // Sort by score descending
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k
        all_results.truncate(k);
        all_results
    }

    fn search_text(&self, query: &str, k: usize) -> Vec<VectorSearchResult> {
        // Use the first store's embedder
        if let Some(_store) = self.stores.first() {
            let mut all_results: Vec<VectorSearchResult> = self
                .stores
                .iter()
                .flat_map(|s| s.search_text(query, k))
                .collect();

            all_results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            all_results.truncate(k);
            all_results
        } else {
            vec![]
        }
    }

    fn dimension(&self) -> usize {
        self.stores.first().map(|s| s.dimension()).unwrap_or(0)
    }

    fn len(&self) -> usize {
        self.stores.iter().map(|s| s.len()).sum()
    }
}

/// Hybrid retriever that combines SmartRetriever with external VectorStore.
pub struct HybridRetriever<'a, E: Embedder, V: VectorStore> {
    /// Smart retriever for local processing.
    smart: SmartRetriever<'a, E>,
    /// External vector store.
    vector_store: &'a V,
    /// Weight for vector store results (0.0 - 1.0).
    vector_store_weight: f32,
}

impl<'a, E: Embedder, V: VectorStore> HybridRetriever<'a, E, V> {
    /// Create a new hybrid retriever.
    pub fn new(embedder: &'a E, vector_store: &'a V, config: RetrievalConfig) -> Self {
        Self {
            smart: SmartRetriever::new(embedder, config),
            vector_store,
            vector_store_weight: 0.5,
        }
    }

    /// Set the weight for vector store results.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.vector_store_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Retrieve from external vector store.
    pub fn retrieve_external(&self, query: &str, k: usize) -> Vec<RetrievedDoc> {
        self.vector_store
            .search_text(query, k)
            .into_iter()
            .enumerate()
            .map(|(i, r)| r.to_retrieved_doc(i as u64))
            .collect()
    }

    /// Process external results with clustering and condensation.
    pub fn process_external(&self, query: &str, k: usize) -> FreshRetrieval {
        let results = self.vector_store.search_text(query, k);
        let docs: Vec<RetrievedDoc> = results
            .iter()
            .enumerate()
            .map(|(i, r)| r.to_retrieved_doc(i as u64))
            .collect();
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();

        self.smart.process(docs, &contents)
    }

    /// Get the smart retriever configuration.
    pub fn config(&self) -> &RetrievalConfig {
        self.smart.config()
    }

    /// Get the vector store weight.
    pub fn vector_store_weight(&self) -> f32 {
        self.vector_store_weight
    }
}

// ============================================================================
// DuckDB Vector Store Implementation
// ============================================================================

/// DuckDB-backed vector store for persistent vector storage.
///
/// This store uses DuckDB's VSS (Vector Similarity Search) extension
/// for efficient cosine similarity queries. Requires the `storage` feature.
///
/// # Example
///
/// ```ignore
/// use kkachi::recursive::{DuckDBVectorStore, VectorStore, HashEmbedder};
///
/// // Open or create a DuckDB vector store
/// let store = DuckDBVectorStore::open(
///     "vectors.duckdb",
///     HashEmbedder::new(64),
/// )?;
///
/// // Search for similar documents
/// let results = store.search_text("How do I parse JSON?", 5);
/// ```
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub struct DuckDBVectorStore<E: Embedder> {
    /// DuckDB connection (wrapped in Mutex for thread safety).
    conn: std::sync::Mutex<super::db::Connection>,
    /// Embedder for text queries.
    embedder: E,
    /// Table name for storing vectors.
    table: String,
    /// Content column name.
    content_column: String,
    /// Embedding column name.
    embedding_column: String,
    /// Embedding dimension.
    dimension: usize,
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<E: Embedder> DuckDBVectorStore<E> {
    /// Open or create a DuckDB vector store at the given path.
    ///
    /// Creates the table and required indexes if they don't exist.
    pub fn open(path: impl AsRef<std::path::Path>, embedder: E) -> crate::error::Result<Self> {
        let conn = super::db::Connection::open(path.as_ref())
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Disable extension autoloading to avoid dependency on external extensions
        conn.execute_batch(
            "SET autoload_known_extensions = false; SET autoinstall_known_extensions = false;",
        )
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let dimension = embedder.dimension();
        let table = "kkachi_vectors".to_string();
        let content_column = "content".to_string();
        let embedding_column = "embedding".to_string();

        // Create table if not exists
        // Using TEXT for embeddings (stored as JSON) since DuckDB's Rust bindings
        // don't fully support array type deserialization
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id VARCHAR PRIMARY KEY,
                {} TEXT NOT NULL,
                {} TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            "#,
            table, content_column, embedding_column
        ))
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            embedder,
            table,
            content_column,
            embedding_column,
            dimension,
        })
    }

    /// Open an in-memory DuckDB vector store.
    pub fn in_memory(embedder: E) -> crate::error::Result<Self> {
        let conn = super::db::Connection::open_in_memory()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Disable extension autoloading to avoid dependency on external extensions
        conn.execute_batch(
            "SET autoload_known_extensions = false; SET autoinstall_known_extensions = false;",
        )
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let dimension = embedder.dimension();
        let table = "kkachi_vectors".to_string();
        let content_column = "content".to_string();
        let embedding_column = "embedding".to_string();

        // Using TEXT for embeddings (stored as JSON)
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id VARCHAR PRIMARY KEY,
                {} TEXT NOT NULL,
                {} TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            "#,
            table, content_column, embedding_column
        ))
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            embedder,
            table,
            content_column,
            embedding_column,
            dimension,
        })
    }

    /// Open with custom table configuration.
    pub fn with_table(
        path: impl AsRef<std::path::Path>,
        embedder: E,
        table: impl Into<String>,
        content_column: impl Into<String>,
        embedding_column: impl Into<String>,
    ) -> crate::error::Result<Self> {
        let conn = super::db::Connection::open(path.as_ref())
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Disable extension autoloading to avoid dependency on external extensions
        conn.execute_batch(
            "SET autoload_known_extensions = false; SET autoinstall_known_extensions = false;",
        )
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let dimension = embedder.dimension();
        let table = table.into();
        let content_column = content_column.into();
        let embedding_column = embedding_column.into();

        // Create table if not exists (using TEXT for embeddings stored as JSON)
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id VARCHAR PRIMARY KEY,
                {} TEXT NOT NULL,
                {} TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            "#,
            table, content_column, embedding_column
        ))
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            embedder,
            table,
            content_column,
            embedding_column,
            dimension,
        })
    }

    /// Add a document to the store.
    pub fn add(
        &self,
        id: impl Into<String>,
        content: impl Into<String>,
    ) -> crate::error::Result<()> {
        let id = id.into();
        let content = content.into();
        let embedding = self.embedder.embed(&content);

        // Serialize embedding as JSON string
        let embedding_json = serde_json::to_string(&embedding)
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {} (id, {}, {}, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                self.table, self.content_column, self.embedding_column
            ),
            super::db::params![id, content, embedding_json],
        ).map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(())
    }

    /// Add multiple documents in a batch.
    pub fn add_batch(
        &self,
        docs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> crate::error::Result<()> {
        for (id, content) in docs {
            self.add(id, content)?;
        }
        Ok(())
    }

    /// Remove a document by ID.
    pub fn remove(&self, id: &str) -> crate::error::Result<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
        let changes = conn
            .execute(
                &format!("DELETE FROM {} WHERE id = ?", self.table),
                super::db::params![id],
            )
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(changes > 0)
    }

    /// Update a document (remove + add).
    pub fn update(
        &self,
        id: impl Into<String>,
        content: impl Into<String>,
    ) -> crate::error::Result<()> {
        self.add(id, content)
    }

    /// Get document content by ID.
    pub fn get(&self, id: &str) -> crate::error::Result<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {} FROM {} WHERE id = ?",
                self.content_column, self.table
            ))
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let mut rows = stmt
            .query(super::db::params![id])
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?
        {
            let content: String = row
                .get(0)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            Ok(Some(content))
        } else {
            Ok(None)
        }
    }

    /// Clear all documents from the store.
    pub fn clear(&self) -> crate::error::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
        conn.execute(&format!("DELETE FROM {}", self.table), [])
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<E: Embedder> VectorStore for DuckDBVectorStore<E> {
    fn search(&self, embedding: &[f32], k: usize) -> Vec<VectorSearchResult> {
        // Lock the connection for thread safety
        let conn = match self.conn.lock() {
            Ok(conn) => conn,
            Err(_) => return vec![],
        };

        // Fetch all documents and compute similarity in Rust
        let mut stmt = match conn.prepare(&format!(
            "SELECT id, {}, {} FROM {}",
            self.content_column, self.embedding_column, self.table
        )) {
            Ok(stmt) => stmt,
            Err(_) => return vec![],
        };

        let rows = match stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            // Embedding is stored as JSON string
            let emb_json: String = row.get(2)?;
            Ok((id, content, emb_json))
        }) {
            Ok(rows) => rows,
            Err(_) => return vec![],
        };

        let mut scored: Vec<(String, String, f32)> = rows
            .filter_map(|r| r.ok())
            .filter_map(|(id, content, emb_json)| {
                // Parse embedding from JSON
                let emb: Vec<f32> = serde_json::from_str(&emb_json).ok()?;
                let score = Self::cosine_similarity(embedding, &emb);
                Some((id, content, score))
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        scored
            .into_iter()
            .take(k)
            .map(|(id, content, score)| VectorSearchResult::new(id, content, score))
            .collect()
    }

    fn search_text(&self, query: &str, k: usize) -> Vec<VectorSearchResult> {
        let embedding = self.embedder.embed(query);
        self.search(&embedding, k)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        // Lock the connection for thread safety
        let conn = match self.conn.lock() {
            Ok(conn) => conn,
            Err(_) => return 0,
        };

        let mut stmt = match conn.prepare(&format!("SELECT COUNT(*) FROM {}", self.table)) {
            Ok(stmt) => stmt,
            Err(_) => return 0,
        };

        stmt.query_row([], |row| row.get::<_, i64>(0))
            .map(|c| c as usize)
            .unwrap_or(0)
    }
}

/// Implement MutableVectorStore for DuckDBVectorStore.
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<E: Embedder> super::training::MutableVectorStore for DuckDBVectorStore<E> {
    fn add(&mut self, id: String, content: String) {
        let _ = DuckDBVectorStore::add(self, id, content);
    }

    fn add_batch(&mut self, docs: Vec<(String, String)>) {
        let _ = DuckDBVectorStore::add_batch(self, docs);
    }

    fn remove(&mut self, id: &str) -> bool {
        DuckDBVectorStore::remove(self, id).unwrap_or(false)
    }

    fn clear(&mut self) {
        let _ = DuckDBVectorStore::clear(self);
    }

    fn update(&mut self, id: String, content: String) {
        let _ = DuckDBVectorStore::update(self, id, content);
    }
}

// ============================================================================
// ChunkStore Trait for Chunk-based Storage
// ============================================================================

/// A stored chunk with parent document reference.
#[derive(Debug, Clone)]
pub struct StoredChunk {
    /// Chunk identifier.
    pub id: String,
    /// Parent document identifier.
    pub parent_id: String,
    /// Chunk content.
    pub content: String,
    /// Chunk index (0-based).
    pub index: u16,
    /// Total chunks in parent document.
    pub total: u16,
    /// Section type if detected.
    pub section_type: Option<String>,
    /// Embedding vector.
    pub embedding: Vec<f32>,
}

impl StoredChunk {
    /// Create a new stored chunk.
    pub fn new(
        id: impl Into<String>,
        parent_id: impl Into<String>,
        content: impl Into<String>,
        index: u16,
        total: u16,
    ) -> Self {
        Self {
            id: id.into(),
            parent_id: parent_id.into(),
            content: content.into(),
            index,
            total,
            section_type: None,
            embedding: Vec::new(),
        }
    }

    /// Set section type.
    pub fn with_section_type(mut self, section_type: impl Into<String>) -> Self {
        self.section_type = Some(section_type.into());
        self
    }

    /// Set embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
}

/// Result from a chunk search.
#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    /// The stored chunk.
    pub chunk: StoredChunk,
    /// Similarity score (0.0 - 1.0).
    pub score: f32,
}

impl ChunkSearchResult {
    /// Create a new search result.
    pub fn new(chunk: StoredChunk, score: f32) -> Self {
        Self { chunk, score }
    }
}

/// Trait for storing and retrieving chunks with parent document reference.
///
/// Extends VectorStore with chunk-specific operations for managing
/// document chunks with parent references and section metadata.
pub trait ChunkStore: VectorStore {
    /// Add a chunk with parent document reference.
    fn add_chunk(
        &self,
        chunk_id: &str,
        parent_id: &str,
        content: &str,
        index: u16,
        total: u16,
        section_type: Option<&str>,
    ) -> crate::error::Result<()>;

    /// Get all chunks for a parent document.
    fn get_chunks(&self, parent_id: &str) -> crate::error::Result<Vec<StoredChunk>>;

    /// Search chunks by similarity.
    fn search_chunks(
        &self,
        query: &str,
        k: usize,
        parent_filter: Option<&str>,
    ) -> Vec<ChunkSearchResult>;

    /// Get a specific chunk by ID.
    fn get_chunk(&self, chunk_id: &str) -> crate::error::Result<Option<StoredChunk>>;

    /// Remove all chunks for a parent document.
    fn remove_chunks(&self, parent_id: &str) -> crate::error::Result<usize>;

    /// Count chunks for a parent document.
    fn count_chunks(&self, parent_id: &str) -> usize;
}

/// Implement ChunkStore for DuckDBVectorStore.
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<E: Embedder> ChunkStore for DuckDBVectorStore<E> {
    fn add_chunk(
        &self,
        chunk_id: &str,
        parent_id: &str,
        content: &str,
        index: u16,
        total: u16,
        section_type: Option<&str>,
    ) -> crate::error::Result<()> {
        let embedding = self.embedder.embed(content);
        let embedding_json = serde_json::to_string(&embedding)
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Create chunks table if not exists
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {}_chunks (
                id VARCHAR PRIMARY KEY,
                parent_id VARCHAR NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_total INTEGER NOT NULL,
                section_type VARCHAR,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_{}_chunks_parent ON {}_chunks(parent_id);
            "#,
            self.table, self.table, self.table
        ))
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        conn.execute(
            &format!(
                r#"INSERT OR REPLACE INTO {}_chunks
                   (id, parent_id, content, chunk_index, chunk_total, section_type, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)"#,
                self.table
            ),
            super::db::params![
                chunk_id,
                parent_id,
                content,
                index as i32,
                total as i32,
                section_type,
                embedding_json
            ],
        )
        .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(())
    }

    fn get_chunks(&self, parent_id: &str) -> crate::error::Result<Vec<StoredChunk>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Check if table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [&format!("{}_chunks", self.table)],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return Ok(Vec::new());
        }

        let mut stmt = conn
            .prepare(&format!(
                r#"SELECT id, parent_id, content, chunk_index, chunk_total, section_type, embedding
                   FROM {}_chunks
                   WHERE parent_id = ?
                   ORDER BY chunk_index"#,
                self.table
            ))
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([parent_id], |row| {
                let id: String = row.get(0)?;
                let parent_id: String = row.get(1)?;
                let content: String = row.get(2)?;
                let index: i32 = row.get(3)?;
                let total: i32 = row.get(4)?;
                let section_type: Option<String> = row.get(5)?;
                let emb_json: String = row.get(6)?;
                Ok((id, parent_id, content, index, total, section_type, emb_json))
            })
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let mut chunks = Vec::new();
        for row in rows {
            let (id, parent_id, content, index, total, section_type, emb_json) =
                row.map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let embedding: Vec<f32> = serde_json::from_str(&emb_json).unwrap_or_default();

            let mut chunk = StoredChunk::new(id, parent_id, content, index as u16, total as u16);
            chunk.embedding = embedding;
            chunk.section_type = section_type;
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn search_chunks(
        &self,
        query: &str,
        k: usize,
        parent_filter: Option<&str>,
    ) -> Vec<ChunkSearchResult> {
        let query_embedding = self.embedder.embed(query);

        let conn = match self.conn.lock() {
            Ok(conn) => conn,
            Err(_) => return vec![],
        };

        // Check if table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [&format!("{}_chunks", self.table)],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return vec![];
        }

        let sql = match parent_filter {
            Some(_) => format!(
                r#"SELECT id, parent_id, content, chunk_index, chunk_total, section_type, embedding
                   FROM {}_chunks
                   WHERE parent_id = ?"#,
                self.table
            ),
            None => format!(
                r#"SELECT id, parent_id, content, chunk_index, chunk_total, section_type, embedding
                   FROM {}_chunks"#,
                self.table
            ),
        };

        let mut stmt = match conn.prepare(&sql) {
            Ok(stmt) => stmt,
            Err(_) => return vec![],
        };

        // Use query instead of query_map to avoid closure type mismatch
        let mut rows = if let Some(parent) = parent_filter {
            match stmt.query(super::db::params![parent]) {
                Ok(rows) => rows,
                Err(_) => return vec![],
            }
        } else {
            match stmt.query(super::db::params![]) {
                Ok(rows) => rows,
                Err(_) => return vec![],
            }
        };

        let mut scored: Vec<ChunkSearchResult> = Vec::new();
        while let Ok(Some(row)) = rows.next() {
            let id: String = row.get(0).unwrap_or_default();
            let parent_id: String = row.get(1).unwrap_or_default();
            let content: String = row.get(2).unwrap_or_default();
            let index: i32 = row.get(3).unwrap_or_default();
            let total: i32 = row.get(4).unwrap_or_default();
            let section_type: Option<String> = row.get(5).ok();
            let emb_json: String = row.get(6).unwrap_or_default();

            if let Ok(embedding) = serde_json::from_str::<Vec<f32>>(&emb_json) {
                let score = Self::cosine_similarity(&query_embedding, &embedding);
                let mut chunk =
                    StoredChunk::new(id, parent_id, content, index as u16, total as u16);
                chunk.embedding = embedding;
                chunk.section_type = section_type;
                scored.push(ChunkSearchResult::new(chunk, score));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(k);
        scored
    }

    fn get_chunk(&self, chunk_id: &str) -> crate::error::Result<Option<StoredChunk>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Check if table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [&format!("{}_chunks", self.table)],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return Ok(None);
        }

        let mut stmt = conn
            .prepare(&format!(
                r#"SELECT id, parent_id, content, chunk_index, chunk_total, section_type, embedding
                   FROM {}_chunks
                   WHERE id = ?"#,
                self.table
            ))
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        let mut rows = stmt
            .query([chunk_id])
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        if let Some(row) = rows
            .next()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?
        {
            let id: String = row
                .get(0)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let parent_id: String = row
                .get(1)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let content: String = row
                .get(2)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let index: i32 = row
                .get(3)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let total: i32 = row
                .get(4)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let section_type: Option<String> = row
                .get(5)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;
            let emb_json: String = row
                .get(6)
                .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

            let embedding: Vec<f32> = serde_json::from_str(&emb_json).unwrap_or_default();

            let mut chunk = StoredChunk::new(id, parent_id, content, index as u16, total as u16);
            chunk.embedding = embedding;
            chunk.section_type = section_type;

            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    fn remove_chunks(&self, parent_id: &str) -> crate::error::Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        // Check if table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [&format!("{}_chunks", self.table)],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return Ok(0);
        }

        let changes = conn
            .execute(
                &format!("DELETE FROM {}_chunks WHERE parent_id = ?", self.table),
                [parent_id],
            )
            .map_err(|e| crate::error::Error::Storage(e.to_string()))?;

        Ok(changes)
    }

    fn count_chunks(&self, parent_id: &str) -> usize {
        let conn = match self.conn.lock() {
            Ok(conn) => conn,
            Err(_) => return 0,
        };

        // Check if table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM information_schema.tables WHERE table_name = ?",
                [&format!("{}_chunks", self.table)],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return 0;
        }

        conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM {}_chunks WHERE parent_id = ?",
                self.table
            ),
            [parent_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c as usize)
        .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_config_default() {
        let config = RetrievalConfig::default();
        assert_eq!(config.initial_k, 20);
        assert_eq!(config.final_k, 5);
        assert!(config.auto_condense);
        assert!(config.use_semantic_cache);
    }

    #[test]
    fn test_retrieval_config_builder() {
        let config = RetrievalConfig::with_threshold(0.9)
            .with_initial_k(30)
            .with_final_k(10)
            .with_auto_condense(false)
            .with_semantic_cache(false);

        assert!((config.similarity_threshold - 0.9).abs() < 0.001);
        assert_eq!(config.initial_k, 30);
        assert_eq!(config.final_k, 10);
        assert!(!config.auto_condense);
        assert!(!config.use_semantic_cache);
    }

    #[test]
    fn test_zero_embedder() {
        let embedder = ZeroEmbedder::new(384);
        let embedding = embedder.embed("test text");
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_hash_embedder() {
        let embedder = HashEmbedder::new(64);
        let e1 = embedder.embed("hello world");
        let e2 = embedder.embed("hello world");
        let e3 = embedder.embed("different text");

        assert_eq!(e1.len(), 64);
        assert_eq!(e1, e2); // Deterministic
        assert_ne!(e1, e3); // Different for different text
    }

    #[test]
    fn test_smart_retriever_creation() {
        let embedder = ZeroEmbedder::new(384);
        let retriever = SmartRetriever::with_defaults(&embedder);
        assert_eq!(retriever.config().initial_k, 20);
    }

    #[test]
    fn test_smart_retriever_empty_docs() {
        let embedder = ZeroEmbedder::new(384);
        let retriever = SmartRetriever::with_defaults(&embedder);

        let result = retriever.process(vec![], &[]);
        assert!(result.documents.is_empty());
        assert!(result.condensed.is_empty());
    }

    #[test]
    fn test_smart_retriever_process() {
        let embedder = HashEmbedder::new(64);
        let config = RetrievalConfig::default()
            .with_auto_condense(false)
            .with_final_k(3);
        let retriever = SmartRetriever::new(&embedder, config);

        let contents = vec![
            "rust programming",
            "python programming",
            "javascript programming",
        ];
        let docs = vec![
            RetrievedDoc {
                id: 1,
                content: contents[0].to_string(),
                score: 0.9,
                source: DocSource::Storage,
            },
            RetrievedDoc {
                id: 2,
                content: contents[1].to_string(),
                score: 0.8,
                source: DocSource::Storage,
            },
            RetrievedDoc {
                id: 3,
                content: contents[2].to_string(),
                score: 0.7,
                source: DocSource::Storage,
            },
        ];

        let result = retriever.process(docs, &contents);
        assert_eq!(result.documents.len(), 3);
        assert_eq!(result.stats.initial_count, 3);
    }

    #[test]
    fn test_score_document() {
        let embedder = HashEmbedder::new(64);
        let retriever = SmartRetriever::with_defaults(&embedder);

        let score1 =
            retriever.score_document("rust async programming", "rust async tokio programming");
        let score2 = retriever.score_document("rust async programming", "python web development");

        // More similar documents should have higher scores
        assert!(score1 > score2);
    }

    #[test]
    fn test_rerank() {
        let embedder = HashEmbedder::new(64);
        let retriever = SmartRetriever::with_defaults(&embedder);

        let mut docs = vec![
            RetrievedDoc {
                id: 1,
                content: "python web".to_string(),
                score: 0.9,
                source: DocSource::Storage,
            },
            RetrievedDoc {
                id: 2,
                content: "rust async tokio".to_string(),
                score: 0.5,
                source: DocSource::Storage,
            },
        ];
        let contents = ["python web", "rust async tokio"];

        retriever.rerank("rust async programming", &mut docs, &contents);

        // After reranking for "rust async", the rust doc should be first
        assert_eq!(docs[0].id, 2);
    }

    #[test]
    fn test_retrieval_result() {
        let cached = CachedContext {
            answer: "cached answer".to_string(),
            summary: "summary".to_string(),
            score: 0.95,
            iterations: 2,
            similarity: 0.98,
            context_id: ContextId::from_question("test", "domain"),
        };

        let result = RetrievalResult::Cached(cached);
        assert!(result.is_cached());
        assert_eq!(result.cached_answer(), Some("cached answer"));
        assert!(result.documents().is_empty());
    }

    #[test]
    fn test_doc_source() {
        assert_eq!(DocSource::Storage, DocSource::Storage);
        assert_ne!(DocSource::Storage, DocSource::Condensed);
    }

    #[test]
    fn test_cluster_info() {
        let info = ClusterInfo {
            size: 5,
            avg_similarity: 0.9,
            was_condensed: true,
        };
        assert_eq!(info.size, 5);
        assert!(info.was_condensed);
    }

    // =========================================================================
    // VectorStore Tests
    // =========================================================================

    #[test]
    fn test_vector_search_result() {
        let result = VectorSearchResult::new("doc1", "test content", 0.95);
        assert_eq!(result.id, "doc1");
        assert_eq!(result.content, "test content");
        assert!((result.score - 0.95).abs() < 0.001);
        assert!(result.metadata.is_none());

        let mut meta = std::collections::HashMap::new();
        meta.insert("key".to_string(), "value".to_string());
        let result = result.with_metadata(meta);
        assert!(result.metadata.is_some());
    }

    #[test]
    fn test_vector_search_result_to_retrieved_doc() {
        let result = VectorSearchResult::new("doc1", "content", 0.85);
        let doc = result.to_retrieved_doc(42);

        assert_eq!(doc.id, 42);
        assert_eq!(doc.content, "content");
        assert!((doc.score - 0.85).abs() < 0.001);
        assert_eq!(doc.source, DocSource::VectorStore);
    }

    #[test]
    fn test_in_memory_vector_store_empty() {
        let embedder = HashEmbedder::new(64);
        let store = InMemoryVectorStore::new(embedder);

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.dimension(), 64);

        let results = store.search_text("test query", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_in_memory_vector_store_add() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add("doc1", "first document");
        store.add("doc2", "second document");

        assert!(!store.is_empty());
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_in_memory_vector_store_add_batch() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add_batch([("doc1", "first"), ("doc2", "second"), ("doc3", "third")]);

        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_in_memory_vector_store_search() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add("rust", "rust programming language");
        store.add("python", "python programming language");
        store.add("java", "java programming language");

        let results = store.search_text("rust programming", 2);
        assert_eq!(results.len(), 2);

        // The rust doc should be most similar to "rust programming"
        assert_eq!(results[0].id, "rust");
    }

    #[test]
    fn test_in_memory_vector_store_clear() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add("doc1", "content");
        assert_eq!(store.len(), 1);

        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn test_composite_vector_store() {
        let embedder1 = HashEmbedder::new(64);
        let embedder2 = HashEmbedder::new(64);

        let mut store1 = InMemoryVectorStore::new(embedder1);
        store1.add("store1_doc1", "rust async programming");

        let mut store2 = InMemoryVectorStore::new(embedder2);
        store2.add("store2_doc1", "rust tokio runtime");

        let composite = CompositeVectorStore::new(vec![&store1, &store2]);

        assert_eq!(composite.len(), 2);
        assert_eq!(composite.dimension(), 64);

        let results = composite.search_text("rust async", 5);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hybrid_retriever() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(HashEmbedder::new(64));
        store.add("doc1", "rust async programming");
        store.add("doc2", "python web development");

        let config = RetrievalConfig::default()
            .with_auto_condense(false)
            .with_final_k(5);

        let retriever = HybridRetriever::new(&embedder, &store, config).with_weight(0.7);

        assert!((retriever.vector_store_weight() - 0.7).abs() < 0.001);

        let docs = retriever.retrieve_external("rust programming", 2);
        assert!(!docs.is_empty());
        assert!(docs.iter().all(|d| d.source == DocSource::VectorStore));
    }

    #[test]
    fn test_hybrid_retriever_process() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(HashEmbedder::new(64));
        store.add("doc1", "rust async programming tokio");
        store.add("doc2", "rust async await futures");
        store.add("doc3", "python flask web framework");

        let config = RetrievalConfig::default()
            .with_auto_condense(false)
            .with_final_k(5);

        let retriever = HybridRetriever::new(&embedder, &store, config);
        let result = retriever.process_external("rust async", 5);

        assert!(!result.documents.is_empty());
        assert_eq!(result.stats.initial_count, 3);
    }
}
