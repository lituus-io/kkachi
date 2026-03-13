// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Memory (RAG) storage for semantic search and learning.
//!
//! This module provides the [`Memory`] type for storing and retrieving
//! documents using semantic similarity. It supports both in-memory and
//! persistent storage (with the `storage` feature).
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::memory;
//!
//! let mut mem = memory();
//! mem.add("Q: How to parse JSON in Rust? A: Use serde_json::from_str()").unwrap();
//!
//! let results = mem.search("parsing JSON", 3).unwrap();
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(feature = "storage")]
use crate::error::Error;
use crate::error::Result;

#[cfg(feature = "storage")]
use crate::recursive::db::Connection;

/// Create a new in-memory Memory instance.
///
/// This is the entry point for building a memory store.
#[inline]
pub fn memory() -> Memory<HashEmbedder> {
    Memory::new()
}

/// Trait for embedding text into vectors.
///
/// Embedders convert text to fixed-dimension vectors for similarity search.
pub trait Embedder: Send + Sync {
    /// Embed text into a vector.
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;
}

/// A simple hash-based embedder for development and testing.
///
/// This embedder uses hashing to create pseudo-embeddings. It's fast but
/// doesn't capture semantic similarity. Use a real embedding model
/// (e.g., sentence-transformers) for production.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dimension: usize,
}

impl HashEmbedder {
    /// Create a new hash embedder with the specified dimension.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];

        // Hash each word and use it to set embedding values
        for word in text.split_whitespace() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();

            // Use hash to determine position and value
            let pos = (hash as usize) % self.dimension;
            let value = ((hash >> 32) as f32) / (u32::MAX as f32);
            embedding[pos] += value;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// A document stored in memory.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier.
    pub id: String,
    /// The document content.
    pub content: String,
    /// Pre-computed embedding.
    pub embedding: Vec<f32>,
    /// Optional tag for categorization.
    pub tag: Option<String>,
}

/// A search result from memory.
#[derive(Debug, Clone)]
pub struct Recall {
    /// Document identifier.
    pub id: String,
    /// Document content.
    pub content: String,
    /// Similarity score (0.0 to 1.0).
    pub score: f64,
    /// Optional tag.
    pub tag: Option<String>,
}

/// In-memory document storage.
struct InMemoryStore {
    documents: Vec<Document>,
    next_id: u64,
}

impl InMemoryStore {
    fn new() -> Self {
        Self {
            documents: Vec::new(),
            next_id: 0,
        }
    }

    fn add(&mut self, content: String, embedding: Vec<f32>, tag: Option<String>) -> String {
        let id = format!("doc:{}", self.next_id);
        self.next_id += 1;
        self.documents.push(Document {
            id: id.clone(),
            content,
            embedding,
            tag,
        });
        id
    }

    fn add_with_id(
        &mut self,
        id: String,
        content: String,
        embedding: Vec<f32>,
        tag: Option<String>,
    ) {
        // Remove existing document with same ID
        self.documents.retain(|d| d.id != id);
        self.documents.push(Document {
            id,
            content,
            embedding,
            tag,
        });
    }

    fn get(&self, id: &str) -> Option<&Document> {
        self.documents.iter().find(|d| d.id == id)
    }

    fn search(&self, query_embedding: &[f32], k: usize) -> Vec<Recall> {
        let mut scored: Vec<_> = self
            .documents
            .iter()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                (doc, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(doc, score)| Recall {
                id: doc.id.clone(),
                content: doc.content.clone(),
                score,
                tag: doc.tag.clone(),
            })
            .collect()
    }

    fn update(&mut self, id: &str, content: String, embedding: Vec<f32>) -> bool {
        if let Some(doc) = self.documents.iter_mut().find(|d| d.id == id) {
            doc.content = content;
            doc.embedding = embedding;
            true
        } else {
            false
        }
    }

    fn remove(&mut self, id: &str) -> bool {
        let len_before = self.documents.len();
        self.documents.retain(|d| d.id != id);
        self.documents.len() < len_before
    }

    fn all(&self) -> Vec<Recall> {
        self.documents
            .iter()
            .map(|doc| Recall {
                id: doc.id.clone(),
                content: doc.content.clone(),
                score: 1.0,
                tag: doc.tag.clone(),
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.documents.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    fn tags(&self) -> Vec<String> {
        let mut tags: Vec<_> = self
            .documents
            .iter()
            .filter_map(|d| d.tag.clone())
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)) as f64
    }
}

// ============================================================================
// Vector Index Trait (for future HNSW support)
// ============================================================================

/// Trait for vector indexing and search.
///
/// This abstraction allows swapping the search strategy from linear scan
/// to more efficient approximate nearest neighbor algorithms like HNSW.
pub trait VectorIndex: Send + Sync {
    /// Insert a vector with the given ID.
    fn insert(&mut self, id: usize, embedding: &[f32]);

    /// Search for the k nearest neighbors.
    ///
    /// Returns a vector of (id, similarity_score) pairs.
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f64)>;

    /// Remove a vector by ID.
    fn remove(&mut self, id: usize);

    /// Get the number of vectors in the index.
    fn len(&self) -> usize;

    /// Check if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Linear scan index (default, exact search).
///
/// This is the simplest index that performs a brute-force linear scan.
/// Suitable for small collections (< 10,000 documents).
#[derive(Debug, Clone, Default)]
pub struct LinearIndex {
    embeddings: Vec<(usize, Vec<f32>)>,
}

impl LinearIndex {
    /// Create a new linear index.
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
        }
    }
}

impl VectorIndex for LinearIndex {
    fn insert(&mut self, id: usize, embedding: &[f32]) {
        // Remove existing if present
        self.embeddings.retain(|(i, _)| *i != id);
        self.embeddings.push((id, embedding.to_vec()));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f64)> {
        let mut scored: Vec<_> = self
            .embeddings
            .iter()
            .map(|(id, emb)| (*id, cosine_similarity(query, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn remove(&mut self, id: usize) {
        self.embeddings.retain(|(i, _)| *i != id);
    }

    fn len(&self) -> usize {
        self.embeddings.len()
    }
}

// ============================================================================
// MMR (Maximal Marginal Relevance) for Diverse Retrieval
// ============================================================================

/// Select documents using Maximal Marginal Relevance.
///
/// MMR balances relevance to the query with diversity among selected documents.
///
/// # Arguments
///
/// * `query_embedding` - The query vector
/// * `doc_embeddings` - Vector of (id, embedding, relevance_score) tuples
/// * `k` - Number of documents to select
/// * `lambda` - Balance parameter (0.0 = max diversity, 1.0 = max relevance)
///
/// # Returns
///
/// Vector of selected document IDs with their MMR scores.
pub fn mmr_select(
    _query_embedding: &[f32],
    doc_embeddings: &[(usize, Vec<f32>, f64)],
    k: usize,
    lambda: f64,
) -> Vec<(usize, f64)> {
    if doc_embeddings.is_empty() || k == 0 {
        return Vec::new();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let mut selected: Vec<(usize, f64)> = Vec::with_capacity(k);
    let mut remaining: Vec<_> = doc_embeddings.iter().collect();

    while selected.len() < k && !remaining.is_empty() {
        let best_idx = remaining
            .iter()
            .enumerate()
            .map(|(idx, (doc_id, doc_emb, relevance))| {
                // Calculate max similarity to already selected documents
                let max_sim_to_selected = if selected.is_empty() {
                    0.0
                } else {
                    selected
                        .iter()
                        .filter_map(|(sel_id, _)| {
                            doc_embeddings
                                .iter()
                                .find(|(id, _, _)| id == sel_id)
                                .map(|(_, sel_emb, _)| cosine_similarity(doc_emb, sel_emb))
                        })
                        .fold(0.0f64, |a, b| a.max(b))
                };

                // MMR score = λ * relevance - (1 - λ) * max_similarity
                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;
                (idx, *doc_id, mmr_score)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((idx, doc_id, score)) = best_idx {
            selected.push((doc_id, score));
            remaining.remove(idx);
        } else {
            break;
        }
    }

    selected
}

/// Storage backend enum.
enum MemoryStore {
    InMemory(InMemoryStore),
    #[cfg(feature = "storage")]
    Persistent {
        conn: Connection,
        dimension: usize,
        db_path: String,
    },
}

/// Compute a deterministic content-hash ID for upsert operations.
fn content_hash_id(content: &str) -> String {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("upsert:{:016x}", hasher.finish())
}

impl MemoryStore {
    /// Insert a document with the given ID. Replaces any existing document with the same ID.
    fn insert(
        &mut self,
        id: &str,
        content: &str,
        embedding: &[f32],
        tag: Option<&str>,
    ) -> Result<()> {
        match self {
            MemoryStore::InMemory(store) => {
                store.add_with_id(
                    id.to_string(),
                    content.to_string(),
                    embedding.to_vec(),
                    tag.map(|t| t.to_string()),
                );
                Ok(())
            }
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let embedding_bytes = embedding_to_bytes(embedding);
                conn.execute(
                    "INSERT OR REPLACE INTO documents (id, content, embedding, tag) VALUES (?, ?, ?, ?)",
                    duckdb::params![id, content, &embedding_bytes, &tag],
                )
                .map_err(|e| Error::memory("insert", e.to_string(), "Ensure the database file is not locked by another process and has write permissions."))?;
                Ok(())
            }
        }
    }

    /// Add a document with an auto-generated ID.
    fn add(&mut self, content: &str, embedding: &[f32], tag: Option<&str>) -> Result<String> {
        match self {
            MemoryStore::InMemory(store) => Ok(store.add(
                content.to_string(),
                embedding.to_vec(),
                tag.map(|t| t.to_string()),
            )),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let id = format!("doc:{}", uuid_v4());
                let embedding_bytes = embedding_to_bytes(embedding);
                conn.execute(
                    "INSERT OR REPLACE INTO documents (id, content, embedding, tag) VALUES (?, ?, ?, ?)",
                    duckdb::params![&id, content, &embedding_bytes, &tag],
                )
                .map_err(|e| Error::memory("add", e.to_string(), "Ensure the database file is not locked by another process and has write permissions."))?;
                Ok(id)
            }
        }
    }

    /// Fetch all documents with their embeddings (for MMR search).
    fn fetch_all_with_embeddings(&self) -> Result<Vec<(String, String, Vec<f32>, Option<String>)>> {
        match self {
            MemoryStore::InMemory(store) => Ok(store
                .documents
                .iter()
                .map(|d| {
                    (
                        d.id.clone(),
                        d.content.clone(),
                        d.embedding.clone(),
                        d.tag.clone(),
                    )
                })
                .collect()),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent {
                conn, dimension, ..
            } => {
                let mut stmt = conn
                    .prepare("SELECT id, content, embedding, tag FROM documents")
                    .map_err(|e| {
                        Error::memory("search", e.to_string(), "Check database integrity.")
                    })?;
                Ok(stmt
                    .query_map([], |row| {
                        let id: String = row.get(0)?;
                        let content: String = row.get(1)?;
                        let embedding_bytes: Vec<u8> = row.get(2)?;
                        let tag: Option<String> = row.get(3)?;
                        let embedding = bytes_to_embedding(&embedding_bytes, *dimension);
                        Ok((id, content, embedding, tag))
                    })
                    .map_err(|e| {
                        Error::memory("search", e.to_string(), "Check database integrity.")
                    })?
                    .filter_map(|r| r.ok())
                    .collect())
            }
        }
    }

    fn update_doc(&mut self, id: &str, content: &str, embedding: &[f32]) -> Result<bool> {
        match self {
            MemoryStore::InMemory(store) => {
                Ok(store.update(id, content.to_string(), embedding.to_vec()))
            }
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let embedding_bytes = embedding_to_bytes(embedding);
                conn.execute(
                    "UPDATE documents SET content = ?, embedding = ? WHERE id = ?",
                    duckdb::params![content, &embedding_bytes, id],
                )
                .map(|n| n > 0)
                .map_err(|e| {
                    Error::memory(
                        "update",
                        e.to_string(),
                        "The document may have been removed concurrently.",
                    )
                })
            }
        }
    }

    fn remove_doc(&mut self, id: &str) -> Result<bool> {
        match self {
            MemoryStore::InMemory(store) => Ok(store.remove(id)),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => conn
                .execute("DELETE FROM documents WHERE id = ?", [id])
                .map(|n| n > 0)
                .map_err(|e| {
                    Error::memory("remove", e.to_string(), "Ensure the database is writable.")
                }),
        }
    }

    fn all_docs(&self) -> Result<Vec<Recall>> {
        match self {
            MemoryStore::InMemory(store) => Ok(store.all()),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let mut stmt = conn
                    .prepare("SELECT id, content, tag FROM documents")
                    .map_err(|e| {
                        Error::memory("all", e.to_string(), "Check database integrity.")
                    })?;
                Ok(stmt
                    .query_map([], |row| {
                        Ok(Recall {
                            id: row.get(0)?,
                            content: row.get(1)?,
                            score: 1.0,
                            tag: row.get(2)?,
                        })
                    })
                    .map_err(|e| Error::memory("all", e.to_string(), "Check database integrity."))?
                    .filter_map(|r| r.ok())
                    .collect())
            }
        }
    }

    fn count(&self) -> Result<usize> {
        match self {
            MemoryStore::InMemory(store) => Ok(store.len()),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let mut stmt = conn
                    .prepare("SELECT COUNT(*) FROM documents")
                    .map_err(|e| {
                        Error::memory(
                            "len",
                            e.to_string(),
                            "The database may be corrupt; try recreating it.",
                        )
                    })?;
                let count = stmt
                    .query_row([], |row| row.get::<_, i64>(0))
                    .map_err(|e| {
                        Error::memory(
                            "len",
                            e.to_string(),
                            "The database may be corrupt; try recreating it.",
                        )
                    })?;
                Ok(count as usize)
            }
        }
    }

    fn unique_tags(&self) -> Result<Vec<String>> {
        match self {
            MemoryStore::InMemory(store) => Ok(store.tags()),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let mut stmt = conn
                    .prepare(
                        "SELECT DISTINCT tag FROM documents WHERE tag IS NOT NULL ORDER BY tag",
                    )
                    .map_err(|e| {
                        Error::memory("tags", e.to_string(), "Check database integrity.")
                    })?;
                Ok(stmt
                    .query_map([], |row| row.get(0))
                    .map_err(|e| Error::memory("tags", e.to_string(), "Check database integrity."))?
                    .filter_map(|r| r.ok())
                    .collect())
            }
        }
    }
}

/// Memory storage for RAG (Retrieval-Augmented Generation).
///
/// Provides semantic search over stored documents. Can be in-memory
/// or persistent (with the `storage` feature).
pub struct Memory<E: Embedder = HashEmbedder> {
    embedder: E,
    store: MemoryStore,
    k: usize,
    learn_threshold: Option<f64>,
    /// MMR diversity parameter (0.0 = max diversity, 1.0 = max relevance).
    /// If None, MMR is disabled and standard search is used.
    mmr_lambda: Option<f64>,
}

impl<E: Embedder> Memory<E> {
    /// Get a reference to the embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Get the number of results to return from search.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Set the number of results to return from search.
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }
}

impl Memory<HashEmbedder> {
    /// Create a new in-memory Memory with the default hash embedder.
    pub fn new() -> Self {
        Self {
            embedder: HashEmbedder::default(),
            store: MemoryStore::InMemory(InMemoryStore::new()),
            k: 3,
            learn_threshold: None,
            mmr_lambda: None,
        }
    }
}

impl Default for Memory<HashEmbedder> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Embedder> Memory<E> {
    /// Create a new Memory with a custom embedder.
    pub fn with_embedder(embedder: E) -> Self {
        Self {
            embedder,
            store: MemoryStore::InMemory(InMemoryStore::new()),
            k: 3,
            learn_threshold: None,
            mmr_lambda: None,
        }
    }

    /// Set a custom embedder, consuming self and returning a new Memory.
    pub fn embedder_with<E2: Embedder>(self, embedder: E2) -> Memory<E2> {
        Memory {
            embedder,
            store: MemoryStore::InMemory(InMemoryStore::new()),
            k: self.k,
            learn_threshold: self.learn_threshold,
            mmr_lambda: self.mmr_lambda,
        }
    }

    /// Enable MMR (Maximal Marginal Relevance) for diverse retrieval.
    ///
    /// The lambda parameter controls the balance between relevance and diversity:
    /// - `lambda = 1.0`: Pure relevance (same as standard search)
    /// - `lambda = 0.5`: Balanced (default recommendation)
    /// - `lambda = 0.0`: Maximum diversity
    ///
    /// # Example
    ///
    /// ```
    /// use kkachi::recursive::memory;
    ///
    /// let mem = memory().diversity(0.5);  // Balanced MMR
    /// ```
    pub fn diversity(mut self, lambda: f64) -> Self {
        self.mmr_lambda = Some(lambda.clamp(0.0, 1.0));
        self
    }

    /// Set the number of results to return from search (fluent).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Enable learning (write-back) above a score threshold.
    pub fn learn_above(mut self, threshold: f64) -> Self {
        self.learn_threshold = Some(threshold);
        self
    }

    /// Use persistent storage.
    #[cfg(feature = "storage")]
    pub fn persist(mut self, path: &str) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| Error::storage(e.to_string()))?;
        let dimension = self.embedder.dimension();

        // Create tables if they don't exist
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                tag TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_documents_tag ON documents(tag);
            "#,
        )
        .map_err(|e| Error::storage(e.to_string()))?;

        self.store = MemoryStore::Persistent {
            conn,
            dimension,
            db_path: path.to_string(),
        };
        Ok(self)
    }

    /// Seed the memory with initial examples if it's empty.
    pub fn seed_if_empty<I, S1, S2>(mut self, items: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        if self.is_empty()? {
            for (question, answer) in items {
                let content = format!("Q: {}\nA: {}", question.into(), answer.into());
                self.add(&content)?;
            }
        }
        Ok(self)
    }

    /// Internal: embed content and insert into the store.
    /// If `id` is None, an auto-generated ID is used.
    fn insert_doc(&mut self, id: Option<&str>, content: &str, tag: Option<&str>) -> Result<String> {
        let embedding = self.embedder.embed(content);
        match id {
            Some(id) => {
                self.store.insert(id, content, &embedding, tag)?;
                Ok(id.to_string())
            }
            None => self.store.add(content, &embedding, tag),
        }
    }

    /// Add a document to memory.
    pub fn add(&mut self, content: &str) -> Result<String> {
        self.insert_doc(None, content, None)
    }

    /// Add a document with a custom ID.
    pub fn add_with_id(&mut self, id: impl Into<String>, content: &str) -> Result<()> {
        let id = id.into();
        let embedding = self.embedder.embed(content);
        self.store.insert(&id, content, &embedding, None)
    }

    /// Add a tagged document to memory.
    pub fn add_tagged(&mut self, tag: &str, content: &str) -> Result<String> {
        self.insert_doc(None, content, Some(tag))
    }

    /// Get a document by ID.
    pub fn get(&self, id: &str) -> Option<String> {
        match &self.store {
            MemoryStore::InMemory(store) => store.get(id).map(|d| d.content.clone()),
            #[cfg(feature = "storage")]
            MemoryStore::Persistent { conn, .. } => {
                let mut stmt = conn
                    .prepare("SELECT content FROM documents WHERE id = ?")
                    .ok()?;
                stmt.query_row([id], |row| row.get(0)).ok()
            }
        }
    }

    /// Search for similar documents.
    ///
    /// If `diversity()` was called on this Memory, MMR will be used automatically.
    /// Otherwise, standard cosine similarity ranking is used.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<Recall>> {
        // Use MMR if diversity is enabled
        if let Some(lambda) = self.mmr_lambda {
            return self.search_diverse(query, k, lambda);
        }

        let query_embedding = self.embedder.embed(query);
        match &self.store {
            // Fast path for in-memory: use InMemoryStore's optimized search
            MemoryStore::InMemory(store) => Ok(store.search(&query_embedding, k)),
            // Generic path via fetch_all_with_embeddings (persistent storage)
            #[cfg(feature = "storage")]
            _ => {
                let all = self.store.fetch_all_with_embeddings()?;
                let mut results: Vec<Recall> = all
                    .into_iter()
                    .map(|(id, content, embedding, tag)| {
                        let score = cosine_similarity(&query_embedding, &embedding);
                        Recall {
                            id,
                            content,
                            score,
                            tag,
                        }
                    })
                    .collect();
                results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(k);
                Ok(results)
            }
        }
    }

    /// Search using the configured k value.
    pub fn search_default(&self, query: &str) -> Result<Vec<Recall>> {
        self.search(query, self.k)
    }

    /// Search with MMR (Maximal Marginal Relevance) for diverse results.
    ///
    /// This method explicitly uses MMR regardless of the `diversity` setting.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `k` - Number of results to return
    /// * `lambda` - Balance parameter (0.0 = max diversity, 1.0 = max relevance)
    ///
    /// # Example
    ///
    /// ```
    /// use kkachi::recursive::memory;
    ///
    /// let mut mem = memory();
    /// mem.add("Document about Rust").unwrap();
    /// mem.add("Another Rust document").unwrap();
    /// mem.add("Python programming guide").unwrap();
    ///
    /// // Get diverse results
    /// let results = mem.search_diverse("programming languages", 3, 0.5).unwrap();
    /// ```
    pub fn search_diverse(&self, query: &str, k: usize, lambda: f64) -> Result<Vec<Recall>> {
        let query_embedding = self.embedder.embed(query);
        let docs = self.store.fetch_all_with_embeddings()?;

        // Build doc_data for MMR: (index, embedding, relevance)
        let doc_data: Vec<_> = docs
            .iter()
            .enumerate()
            .map(|(idx, (_, _, emb, _))| {
                let relevance = cosine_similarity(&query_embedding, emb);
                (idx, emb.clone(), relevance)
            })
            .collect();

        let selected = mmr_select(&query_embedding, &doc_data, k, lambda);

        Ok(selected
            .into_iter()
            .filter_map(|(idx, score)| {
                docs.get(idx).map(|(id, content, _, tag)| Recall {
                    id: id.clone(),
                    content: content.clone(),
                    score,
                    tag: tag.clone(),
                })
            })
            .collect())
    }

    /// Update an existing document.
    pub fn update(&mut self, id: &str, content: &str) -> Result<bool> {
        let embedding = self.embedder.embed(content);
        self.store.update_doc(id, content, &embedding)
    }

    /// Insert or update a document by content hash.
    ///
    /// If a document with the same content hash already exists, it is updated.
    /// Otherwise, a new document is added. Useful for idempotent ingestion.
    ///
    /// Returns the document ID.
    pub fn upsert(&mut self, content: &str) -> Result<String> {
        let id = content_hash_id(content);
        let embedding = self.embedder.embed(content);
        self.store.insert(&id, content, &embedding, None)?;
        Ok(id)
    }

    /// Insert or update a tagged document by content hash.
    ///
    /// Same as [`upsert`](Memory::upsert) but also sets a tag.
    pub fn upsert_tagged(&mut self, tag: &str, content: &str) -> Result<String> {
        let id = content_hash_id(content);
        let embedding = self.embedder.embed(content);
        self.store.insert(&id, content, &embedding, Some(tag))?;
        Ok(id)
    }

    /// Search for similar documents, filtering by minimum similarity score.
    ///
    /// Only returns results with score >= `min_score`.
    pub fn search_above(&self, query: &str, k: usize, min_score: f64) -> Result<Vec<Recall>> {
        Ok(self
            .search(query, k)?
            .into_iter()
            .filter(|r| r.score >= min_score)
            .collect())
    }

    /// Remove a document by ID.
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        self.store.remove_doc(id)
    }

    /// Get all documents.
    pub fn all(&self) -> Result<Vec<Recall>> {
        self.store.all_docs()
    }

    /// Get the number of documents.
    pub fn len(&self) -> Result<usize> {
        self.store.count()
    }

    /// Check if memory is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.store.count()? == 0)
    }

    /// Get all unique tags.
    pub fn tags(&self) -> Result<Vec<String>> {
        self.store.unique_tags()
    }

    /// Learn from a successful refinement (write-back).
    ///
    /// This is called by the refinement loop when a result meets the
    /// learning threshold.
    pub fn learn(&mut self, question: &str, output: &str, score: f64) -> Result<()> {
        if let Some(threshold) = self.learn_threshold {
            if score >= threshold {
                let content = format!("Q: {}\nA: {}", question, output);
                self.add(&content)?;
            }
        }
        Ok(())
    }

    /// Get the database path if using persistent storage.
    #[cfg(feature = "storage")]
    pub fn db_path(&self) -> Option<&str> {
        match &self.store {
            MemoryStore::Persistent { db_path, .. } => Some(db_path.as_str()),
            _ => None,
        }
    }

    /// Create a packager builder for this memory's persistent DB.
    ///
    /// Returns an error if the memory is not using persistent storage.
    #[cfg(feature = "storage")]
    pub fn package(&self, name: &str) -> Result<crate::recursive::packager::PackagerBuilder<'_>> {
        match &self.store {
            MemoryStore::Persistent { db_path, .. } => Ok(
                crate::recursive::packager::PackagerBuilder::new(std::path::Path::new(
                    db_path.as_str(),
                ))
                .name_owned(name.to_string()),
            ),
            _ => Err(Error::storage(
                "Cannot package in-memory store. Call .persist(path) first.",
            )),
        }
    }
}

// Helper functions for persistent storage
#[cfg(feature = "storage")]
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(feature = "storage")]
fn bytes_to_embedding(bytes: &[u8], dimension: usize) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .take(dimension)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(feature = "storage")]
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:032x}", nanos)
}

// ============================================================================
// ONNX Embedder (feature-gated)
// ============================================================================

/// ONNX-based semantic embedder using HuggingFace models.
///
/// This embedder runs ONNX models locally for generating semantic embeddings,
/// without requiring a Python runtime or external API calls.
///
/// # Supported Models
///
/// Any sentence-transformers model exported to ONNX format:
/// - `all-MiniLM-L6-v2` (384 dimensions, fast)
/// - `all-mpnet-base-v2` (768 dimensions, higher quality)
/// - `multi-qa-MiniLM-L6-cos-v1` (384 dimensions, QA optimized)
///
/// # Example
///
/// ```ignore
/// use kkachi::recursive::memory::{Memory, OnnxEmbedder};
///
/// let embedder = OnnxEmbedder::from_dir("./models/all-MiniLM-L6-v2")?;
/// let mem = Memory::with_embedder(embedder);
/// mem.add("Hello world");
/// ```
#[cfg(feature = "embeddings-onnx")]
pub struct OnnxEmbedder {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    dimension: usize,
    max_length: usize,
    normalize: bool,
}

#[cfg(feature = "embeddings-onnx")]
impl std::fmt::Debug for OnnxEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbedder")
            .field("dimension", &self.dimension)
            .field("max_length", &self.max_length)
            .field("normalize", &self.normalize)
            .finish()
    }
}

#[cfg(feature = "embeddings-onnx")]
impl OnnxEmbedder {
    /// Load an ONNX embedder from a directory containing model files.
    ///
    /// The directory should contain:
    /// - `model.onnx` - The ONNX model file
    /// - `tokenizer.json` - The HuggingFace tokenizer
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model directory
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedder = OnnxEmbedder::from_dir("./models/all-MiniLM-L6-v2")?;
    /// ```
    pub fn from_dir(path: impl AsRef<std::path::Path>) -> Result<Self, OnnxEmbedderError> {
        let path = path.as_ref();
        let model_path = path.join("model.onnx");
        let tokenizer_path = path.join("tokenizer.json");

        if !model_path.exists() {
            return Err(OnnxEmbedderError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }
        if !tokenizer_path.exists() {
            return Err(OnnxEmbedderError::TokenizerNotFound(
                tokenizer_path.display().to_string(),
            ));
        }

        // Load ONNX session
        let session = ort::session::Session::builder()
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?;

        // Default dimension - will be determined from first inference
        let dimension = 384; // Common default for MiniLM models

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| OnnxEmbedderError::TokenizerError(e.to_string()))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            dimension,
            max_length: 512,
            normalize: true,
        })
    }

    /// Set the maximum sequence length for tokenization.
    ///
    /// Default is 512 tokens. Longer sequences will be truncated.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Disable L2 normalization of embeddings.
    ///
    /// By default, embeddings are normalized to unit length for cosine similarity.
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Run inference to get embeddings.
    fn run_inference(&self, text: &str) -> Result<Vec<f32>, OnnxEmbedderError> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| OnnxEmbedderError::TokenizerError(e.to_string()))?;

        let input_ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .take(self.max_length)
            .map(|&id| id as i64)
            .collect();

        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .take(self.max_length)
            .map(|&m| m as i64)
            .collect();

        let seq_len = input_ids.len();

        // Create input tensors with shape [1, seq_len]
        let input_ids_array =
            ndarray::Array2::from_shape_vec((1, seq_len), input_ids).map_err(|e| {
                OnnxEmbedderError::OrtError(format!("Failed to create input_ids array: {}", e))
            })?;

        let attention_mask_array =
            ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.clone()).map_err(|e| {
                OnnxEmbedderError::OrtError(format!("Failed to create attention_mask array: {}", e))
            })?;

        // Create token_type_ids (all zeros for single sentence)
        let token_type_ids: Vec<i64> = vec![0; seq_len];
        let token_type_ids_array = ndarray::Array2::from_shape_vec((1, seq_len), token_type_ids)
            .map_err(|e| {
                OnnxEmbedderError::OrtError(format!("Failed to create token_type_ids array: {}", e))
            })?;

        // Create ort Values from arrays
        let input_ids_value = ort::value::Tensor::from_array(input_ids_array)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?;
        let attention_mask_value = ort::value::Tensor::from_array(attention_mask_array)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?;
        let token_type_ids_value = ort::value::Tensor::from_array(token_type_ids_array)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?;

        // Lock the session for inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| OnnxEmbedderError::OrtError(format!("Session lock poisoned: {}", e)))?;

        // Get output name before running inference
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| OnnxEmbedderError::OrtError("No output in model".to_string()))?;

        // Run inference using SessionInputs
        let inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = vec![
            (
                std::borrow::Cow::Borrowed("input_ids"),
                input_ids_value.into(),
            ),
            (
                std::borrow::Cow::Borrowed("attention_mask"),
                attention_mask_value.into(),
            ),
            (
                std::borrow::Cow::Borrowed("token_type_ids"),
                token_type_ids_value.into(),
            ),
        ];
        let outputs = session
            .run(inputs)
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?;

        // Extract embeddings - output shape is [1, seq_len, hidden_size]
        let output_value = outputs
            .get(&output_name)
            .ok_or_else(|| OnnxEmbedderError::OrtError("No output tensor".to_string()))?;

        let tensor_data = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| OnnxEmbedderError::OrtError(e.to_string()))?;

        // Get shape and data from the tensor - copy to owned data to release borrow
        let shape: Vec<i64> = tensor_data.0.iter().map(|&x| x as i64).collect();
        let data: Vec<f32> = tensor_data.1.to_vec();

        // Mean pooling with attention mask
        let embedding = Self::mean_pool_static(
            self.dimension,
            self.normalize,
            &shape,
            &data,
            &attention_mask,
        );

        Ok(embedding)
    }

    /// Mean pooling of token embeddings with attention mask (static version).
    fn mean_pool_static(
        dimension: usize,
        normalize: bool,
        shape: &[i64],
        data: &[f32],
        mask: &[i64],
    ) -> Vec<f32> {
        // Shape is [batch, seq_len, hidden_size]
        if shape.len() != 3 {
            // Fallback: just take the first dimension elements
            return data.iter().take(dimension).copied().collect();
        }

        let seq_len = shape[1] as usize;
        let hidden_size = shape[2] as usize;
        let mut pooled = vec![0.0f32; hidden_size];
        let mut total_weight = 0.0f32;

        for (t, &m) in mask.iter().enumerate().take(seq_len) {
            if m > 0 {
                let weight = m as f32;
                for h in 0..hidden_size {
                    // Index into flattened [1, seq_len, hidden_size] tensor
                    let idx = t * hidden_size + h;
                    if let Some(&val) = data.get(idx) {
                        pooled[h] += val * weight;
                    }
                }
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            for p in &mut pooled {
                *p /= total_weight;
            }
        }

        // Normalize if enabled
        if normalize {
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for p in &mut pooled {
                    *p /= norm;
                }
            }
        }

        pooled
    }
}

#[cfg(feature = "embeddings-onnx")]
impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.run_inference(text).unwrap_or_else(|_| {
            // Fallback to zeros on error (shouldn't happen in practice)
            vec![0.0; self.dimension]
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Errors that can occur when using the ONNX embedder.
#[cfg(feature = "embeddings-onnx")]
#[derive(Debug, Clone)]
pub enum OnnxEmbedderError {
    /// The model file was not found.
    ModelNotFound(String),
    /// The tokenizer file was not found.
    TokenizerNotFound(String),
    /// ONNX Runtime error.
    OrtError(String),
    /// Tokenizer error.
    TokenizerError(String),
}

#[cfg(feature = "embeddings-onnx")]
impl std::fmt::Display for OnnxEmbedderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound(path) => write!(f, "Model file not found: {}", path),
            Self::TokenizerNotFound(path) => write!(f, "Tokenizer file not found: {}", path),
            Self::OrtError(msg) => write!(f, "ONNX Runtime error: {}", msg),
            Self::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
        }
    }
}

#[cfg(feature = "embeddings-onnx")]
impl std::error::Error for OnnxEmbedderError {}

// ============================================================================
// HNSW Index (Hierarchical Navigable Small World)
// ============================================================================

/// HNSW index for approximate nearest neighbor search.
///
/// HNSW (Hierarchical Navigable Small World) is a graph-based algorithm that
/// provides fast approximate nearest neighbor search with high recall.
///
/// # Performance
///
/// - Build time: O(n log n)
/// - Search time: O(log n)
/// - Memory: O(n * M) where M is the number of connections per node
///
/// # Example
///
/// ```
/// use kkachi::recursive::memory::{HnswIndex, VectorIndex};
///
/// let mut index = HnswIndex::new(64);  // 64-dimensional vectors
/// index.insert(0, &vec![1.0; 64]);
/// index.insert(1, &vec![0.5; 64]);
///
/// let results = index.search(&vec![0.9; 64], 2);
/// assert_eq!(results[0].0, 0);  // Most similar
/// ```
#[cfg(feature = "hnsw")]
#[derive(Debug, Clone)]
pub struct HnswIndex {
    /// Stored vectors: id -> embedding
    vectors: Vec<(usize, Vec<f32>)>,
    /// Graph layers: layer -> node_id -> neighbors
    layers: Vec<Vec<Vec<usize>>>,
    /// Entry point for search
    entry_point: Option<usize>,
    /// Maximum number of connections per node (M parameter)
    m: usize,
    /// Maximum connections at layer 0 (M0 = 2 * M)
    m0: usize,
    /// Level multiplier (1 / ln(M))
    ml: f64,
    /// Search beam width during construction
    ef_construction: usize,
    /// Search beam width during query
    ef_search: usize,
    /// Dimension of vectors
    dimension: usize,
    /// ID to internal index mapping
    id_to_index: std::collections::HashMap<usize, usize>,
    /// Random state for level generation
    rng_state: u64,
}

#[cfg(feature = "hnsw")]
impl HnswIndex {
    /// Create a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimensionality of vectors to index
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            layers: Vec::new(),
            entry_point: None,
            m: 16,  // Standard default
            m0: 32, // 2 * M
            ml: 1.0 / (16.0_f64).ln(),
            ef_construction: 200,
            ef_search: 50,
            dimension,
            id_to_index: std::collections::HashMap::new(),
            rng_state: 42,
        }
    }

    /// Set the M parameter (max connections per node).
    ///
    /// Higher values improve recall but increase memory usage.
    /// Default is 16.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m0 = 2 * m;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Set the ef_construction parameter (beam width during build).
    ///
    /// Higher values improve index quality but slow down construction.
    /// Default is 200.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the ef_search parameter (beam width during query).
    ///
    /// Higher values improve recall but slow down queries.
    /// Default is 50.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Generate a random level for a new node.
    fn random_level(&mut self) -> usize {
        // Simple xorshift RNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        let random = (self.rng_state as f64) / (u64::MAX as f64);
        let level = (-random.ln() * self.ml).floor() as usize;
        level.min(15) // Cap at 16 layers
    }

    /// Get the embedding for an internal index.
    fn get_embedding(&self, idx: usize) -> Option<&[f32]> {
        self.vectors.get(idx).map(|(_, v)| v.as_slice())
    }

    /// Search a single layer for nearest neighbors.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f64)> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashSet};

        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();

        // Initialize with entry points
        for &ep in entry_points {
            if let Some(emb) = self.get_embedding(ep) {
                let dist = 1.0 - cosine_similarity(query, emb);
                visited.insert(ep);
                candidates.push(Reverse((OrderedFloat(dist), ep)));
                results.push((OrderedFloat(dist), ep));
            }
        }

        while let Some(Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            // Get furthest result distance
            let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f64::MAX);
            if c_dist > f_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            if let Some(neighbors) = self.layers.get(layer).and_then(|l| l.get(c_idx)) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        if let Some(emb) = self.get_embedding(neighbor) {
                            let dist = 1.0 - cosine_similarity(query, emb);
                            let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f64::MAX);

                            if dist < f_dist || results.len() < ef {
                                candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                                results.push((OrderedFloat(dist), neighbor));
                                while results.len() > ef {
                                    results.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to similarity scores
        results
            .into_sorted_vec()
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (idx, 1.0 - dist))
            .collect()
    }

    /// Select neighbors using simple heuristic.
    fn select_neighbors(&self, candidates: &[(usize, f64)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|(idx, _)| *idx).collect()
    }

    /// Connect a new node to its neighbors.
    fn connect_neighbors(
        &mut self,
        node_idx: usize,
        neighbors: &[usize],
        layer: usize,
        max_connections: usize,
    ) {
        // Ensure layer exists
        while self.layers.len() <= layer {
            self.layers.push(Vec::new());
        }

        // Ensure node exists in layer
        while self.layers[layer].len() <= node_idx {
            self.layers[layer].push(Vec::new());
        }

        // Add neighbors to node
        self.layers[layer][node_idx] = neighbors.to_vec();

        // Add bidirectional connections
        for &neighbor in neighbors {
            while self.layers[layer].len() <= neighbor {
                self.layers[layer].push(Vec::new());
            }

            // Check if we need to add and potentially prune
            let needs_add = !self.layers[layer][neighbor].contains(&node_idx);
            if needs_add {
                self.layers[layer][neighbor].push(node_idx);

                // Prune if too many connections
                if self.layers[layer][neighbor].len() > max_connections {
                    // Get embeddings needed for scoring
                    let neighbor_emb = self.vectors.get(neighbor).map(|(_, v)| v.clone());

                    if let Some(emb) = neighbor_emb {
                        // Score all connections and keep best
                        let current_connections = self.layers[layer][neighbor].clone();
                        let mut scored: Vec<_> = current_connections
                            .iter()
                            .filter_map(|&n| {
                                self.vectors
                                    .get(n)
                                    .map(|(_, e)| (n, cosine_similarity(&emb, e)))
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        self.layers[layer][neighbor] = scored
                            .into_iter()
                            .take(max_connections)
                            .map(|(n, _)| n)
                            .collect();
                    }
                }
            }
        }
    }
}

/// Wrapper for f64 that implements Ord for use in BinaryHeap.
#[cfg(feature = "hnsw")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

#[cfg(feature = "hnsw")]
impl Eq for OrderedFloat {}

#[cfg(feature = "hnsw")]
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "hnsw")]
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(feature = "hnsw")]
impl VectorIndex for HnswIndex {
    fn insert(&mut self, id: usize, embedding: &[f32]) {
        // Check dimension
        if embedding.len() != self.dimension {
            return;
        }

        // Remove existing if present
        if self.id_to_index.contains_key(&id) {
            self.remove(id);
        }

        let node_idx = self.vectors.len();
        self.vectors.push((id, embedding.to_vec()));
        self.id_to_index.insert(id, node_idx);

        // Generate random level for this node
        let node_level = self.random_level();

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            // Initialize layers
            for l in 0..=node_level {
                while self.layers.len() <= l {
                    self.layers.push(Vec::new());
                }
                while self.layers[l].len() <= node_idx {
                    self.layers[l].push(Vec::new());
                }
            }
            return;
        }

        let entry = self.entry_point.unwrap();
        let max_layer = self.layers.len().saturating_sub(1);

        // Search from top layer down to node_level+1
        let mut curr_entry = vec![entry];
        for layer in (node_level + 1..=max_layer).rev() {
            let results = self.search_layer(embedding, &curr_entry, 1, layer);
            if let Some((best, _)) = results.first() {
                curr_entry = vec![*best];
            }
        }

        // Insert into layers node_level down to 0
        for layer in (0..=node_level.min(max_layer)).rev() {
            let max_conn = if layer == 0 { self.m0 } else { self.m };

            let candidates = self.search_layer(embedding, &curr_entry, self.ef_construction, layer);
            let neighbors = self.select_neighbors(&candidates, max_conn);

            self.connect_neighbors(node_idx, &neighbors, layer, max_conn);

            curr_entry = neighbors;
        }

        // Update entry point if new node has higher level
        if node_level > max_layer {
            self.entry_point = Some(node_idx);
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f64)> {
        if self.vectors.is_empty() || self.entry_point.is_none() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap();
        let max_layer = self.layers.len().saturating_sub(1);

        // Search from top layer down to layer 1
        let mut curr_entry = vec![entry];
        for layer in (1..=max_layer).rev() {
            let results = self.search_layer(query, &curr_entry, 1, layer);
            if let Some((best, _)) = results.first() {
                curr_entry = vec![*best];
            }
        }

        // Search layer 0 with ef_search
        let mut results = self.search_layer(query, &curr_entry, self.ef_search.max(k), 0);

        // Convert internal indices to external IDs
        results.truncate(k);
        results
            .into_iter()
            .filter_map(|(idx, score)| self.vectors.get(idx).map(|(id, _)| (*id, score)))
            .collect()
    }

    fn remove(&mut self, id: usize) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            // Remove from all layers
            for layer in &mut self.layers {
                if idx < layer.len() {
                    layer[idx].clear();
                }
                // Remove references to this node
                for neighbors in layer.iter_mut() {
                    neighbors.retain(|&n| n != idx);
                }
            }

            // Mark as removed (we don't actually remove to keep indices stable)
            self.id_to_index.remove(&id);

            // Update entry point if needed
            if self.entry_point == Some(idx) {
                // Find another valid entry point
                self.entry_point = self.id_to_index.values().copied().next();
            }
        }
    }

    fn len(&self) -> usize {
        self.id_to_index.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedder() {
        let embedder = HashEmbedder::new(64);
        let embedding = embedder.embed("hello world");
        assert_eq!(embedding.len(), 64);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < f64::EPSILON);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_add_and_get() {
        let mut mem = memory();
        let id = mem.add("test content").unwrap();
        assert!(mem.get(&id).is_some());
        assert_eq!(mem.get(&id).unwrap(), "test content");
    }

    #[test]
    fn test_memory_search() {
        let mut mem = memory();
        mem.add("How to parse JSON in Rust? Use serde_json")
            .unwrap();
        mem.add("How to read a file? Use std::fs").unwrap();
        mem.add("How to make HTTP requests? Use reqwest").unwrap();

        // Hash-based embedding works on exact word matches
        let results = mem.search("parse JSON Rust", 2).unwrap();
        assert_eq!(results.len(), 2);
        // The JSON one should be most relevant with exact word match
        assert!(results[0].content.contains("JSON"));
    }

    #[test]
    fn test_memory_update() {
        let mut mem = memory();
        let id = mem.add("original").unwrap();
        assert!(mem.update(&id, "updated").unwrap());
        assert_eq!(mem.get(&id).unwrap(), "updated");
    }

    #[test]
    fn test_memory_remove() {
        let mut mem = memory();
        let id = mem.add("to delete").unwrap();
        assert!(mem.remove(&id).unwrap());
        assert!(mem.get(&id).is_none());
    }

    #[test]
    fn test_memory_tagged() {
        let mut mem = memory();
        mem.add_tagged("rust", "Rust content").unwrap();
        mem.add_tagged("python", "Python content").unwrap();

        let tags = mem.tags().unwrap();
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"python".to_string()));
    }

    #[test]
    fn test_memory_seed_if_empty() {
        let mem = memory()
            .seed_if_empty([("What is Rust?", "A systems programming language")])
            .unwrap();

        assert_eq!(mem.len().unwrap(), 1);
        let results = mem.search("Rust", 1).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_memory_learn() {
        let mut mem = memory().learn_above(0.8);

        // Below threshold - should not add
        mem.learn("question", "bad answer", 0.5).unwrap();
        assert_eq!(mem.len().unwrap(), 0);

        // Above threshold - should add
        mem.learn("question", "good answer", 0.9).unwrap();
        assert_eq!(mem.len().unwrap(), 1);
    }

    #[test]
    fn test_memory_len_and_empty() {
        let mut mem = memory();
        assert!(mem.is_empty().unwrap());
        assert_eq!(mem.len().unwrap(), 0);

        mem.add("doc1").unwrap();
        assert!(!mem.is_empty().unwrap());
        assert_eq!(mem.len().unwrap(), 1);
    }

    #[test]
    fn test_memory_all() {
        let mut mem = memory();
        mem.add("doc1").unwrap();
        mem.add("doc2").unwrap();
        mem.add("doc3").unwrap();

        let all = mem.all().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_memory_with_k() {
        let mem = memory().with_k(5);
        assert_eq!(mem.k(), 5);
    }

    // ========================================================================
    // Phase 2 Tests: MMR, VectorIndex
    // ========================================================================

    #[test]
    fn test_linear_index() {
        let mut index = LinearIndex::new();
        assert!(index.is_empty());

        index.insert(0, &[1.0, 0.0, 0.0]);
        index.insert(1, &[0.0, 1.0, 0.0]);
        index.insert(2, &[0.5, 0.5, 0.0]);

        assert_eq!(index.len(), 3);

        // Search should return in order of similarity
        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Most similar
    }

    #[test]
    fn test_linear_index_remove() {
        let mut index = LinearIndex::new();
        index.insert(0, &[1.0, 0.0]);
        index.insert(1, &[0.0, 1.0]);

        assert_eq!(index.len(), 2);
        index.remove(0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_mmr_select_basic() {
        // Three similar documents
        let doc_embeddings = vec![
            (0, vec![1.0, 0.0, 0.0], 0.9),  // High relevance
            (1, vec![0.9, 0.1, 0.0], 0.85), // Similar to 0
            (2, vec![0.0, 1.0, 0.0], 0.5),  // Different
        ];

        // With high lambda (pure relevance), should select in relevance order
        let selected = mmr_select(&[1.0, 0.0, 0.0], &doc_embeddings, 2, 1.0);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 0); // Highest relevance first

        // With low lambda (diversity), should prefer different documents
        let selected_diverse = mmr_select(&[1.0, 0.0, 0.0], &doc_embeddings, 2, 0.3);
        assert_eq!(selected_diverse.len(), 2);
        // After selecting doc 0, doc 2 should be preferred for diversity
    }

    #[test]
    fn test_mmr_select_empty() {
        let selected = mmr_select(&[1.0, 0.0], &[], 3, 0.5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_memory_diversity() {
        let mut mem = memory().diversity(0.5);
        mem.add("Document about Rust programming").unwrap();
        mem.add("Another document about Rust language").unwrap();
        mem.add("Python is great for ML").unwrap();

        let results = mem.search("Rust programming language", 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_memory_search_diverse() {
        let mut mem = memory();
        mem.add("Rust systems programming").unwrap();
        mem.add("Rust memory safety").unwrap();
        mem.add("Python data science").unwrap();

        // Explicit diverse search
        let results = mem.search_diverse("programming language", 2, 0.5).unwrap();
        assert_eq!(results.len(), 2);
    }

    // ========================================================================
    // HNSW Index Tests
    // ========================================================================

    #[cfg(feature = "hnsw")]
    mod hnsw_tests {
        use super::*;

        #[test]
        fn test_hnsw_basic() {
            let mut index = HnswIndex::new(3);
            assert!(index.is_empty());

            index.insert(0, &[1.0, 0.0, 0.0]);
            index.insert(1, &[0.0, 1.0, 0.0]);
            index.insert(2, &[0.0, 0.0, 1.0]);

            assert_eq!(index.len(), 3);
        }

        #[test]
        fn test_hnsw_search() {
            let mut index = HnswIndex::new(3);

            index.insert(0, &[1.0, 0.0, 0.0]);
            index.insert(1, &[0.9, 0.1, 0.0]);
            index.insert(2, &[0.0, 1.0, 0.0]);

            let results = index.search(&[1.0, 0.0, 0.0], 2);
            assert_eq!(results.len(), 2);
            // Most similar should be ID 0
            assert_eq!(results[0].0, 0);
        }

        #[test]
        fn test_hnsw_remove() {
            let mut index = HnswIndex::new(3);

            index.insert(0, &[1.0, 0.0, 0.0]);
            index.insert(1, &[0.0, 1.0, 0.0]);

            assert_eq!(index.len(), 2);
            index.remove(0);
            assert_eq!(index.len(), 1);

            // Should still be able to search
            let results = index.search(&[0.0, 1.0, 0.0], 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, 1);
        }

        #[test]
        fn test_hnsw_update() {
            let mut index = HnswIndex::new(3);

            index.insert(0, &[1.0, 0.0, 0.0]);
            index.insert(1, &[0.0, 1.0, 0.0]);

            // Update ID 0 to be similar to ID 1
            index.insert(0, &[0.0, 0.9, 0.1]);

            let results = index.search(&[0.0, 1.0, 0.0], 2);
            // Now both should be similar to the query
            assert!(results.iter().any(|(id, _)| *id == 0));
            assert!(results.iter().any(|(id, _)| *id == 1));
        }

        #[test]
        fn test_hnsw_builder_pattern() {
            let index = HnswIndex::new(64)
                .with_m(32)
                .with_ef_construction(400)
                .with_ef_search(100);

            assert_eq!(index.dimension, 64);
            assert_eq!(index.m, 32);
            assert_eq!(index.ef_construction, 400);
            assert_eq!(index.ef_search, 100);
        }

        #[test]
        fn test_hnsw_many_vectors() {
            let mut index = HnswIndex::new(16);

            // Insert 100 random-ish vectors
            for i in 0..100 {
                let mut vec = vec![0.0f32; 16];
                vec[i % 16] = 1.0;
                vec[(i + 1) % 16] = 0.5;
                index.insert(i, &vec);
            }

            assert_eq!(index.len(), 100);

            // Search should return results
            let query = vec![1.0f32; 16];
            let results = index.search(&query, 10);
            assert_eq!(results.len(), 10);
        }

        #[test]
        fn test_hnsw_empty_search() {
            let index = HnswIndex::new(3);
            let results = index.search(&[1.0, 0.0, 0.0], 5);
            assert!(results.is_empty());
        }

        #[test]
        fn test_hnsw_wrong_dimension() {
            let mut index = HnswIndex::new(3);

            // Wrong dimension should be silently ignored
            index.insert(0, &[1.0, 0.0]); // Only 2 dimensions
            assert_eq!(index.len(), 0);

            // Correct dimension should work
            index.insert(0, &[1.0, 0.0, 0.0]);
            assert_eq!(index.len(), 1);
        }
    }
}
