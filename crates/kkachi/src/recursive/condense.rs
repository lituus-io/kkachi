// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Document condensation engine.
//!
//! Merges clusters of similar documents into single comprehensive documents,
//! eliminating redundancy while preserving all unique information.

use smallvec::SmallVec;

use crate::intern::Sym;
use crate::str_view::StrView;

use super::cluster::DocumentCluster;
use super::keywords::KeywordExtractor;
use super::similarity::DocumentFeatures;

/// Configuration for document condensation.
#[derive(Clone, Debug)]
pub struct CondenseConfig {
    /// Maximum documents to condense in one batch.
    pub max_batch_size: usize,
    /// Whether to preserve all unique facts.
    pub preserve_all_facts: bool,
    /// Whether to note contradictions.
    pub note_contradictions: bool,
    /// Target length ratio (condensed / original).
    pub target_ratio: f32,
}

impl Default for CondenseConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10,
            preserve_all_facts: true,
            note_contradictions: true,
            target_ratio: 0.5,
        }
    }
}

/// A condensed document created from multiple similar sources.
#[derive(Clone, Debug)]
pub struct CondensedDocument {
    /// The merged content.
    pub content: String,
    /// IDs of source documents.
    pub source_ids: SmallVec<[u64; 8]>,
    /// Number of source documents.
    pub source_count: u16,
    /// Merged keywords with combined TF-IDF scores.
    pub keywords: SmallVec<[(Sym, f32); 16]>,
    /// Merged category path (longest common prefix + most specific).
    pub category_path: SmallVec<[Sym; 8]>,
    /// Union of all tags.
    pub tags: SmallVec<[Sym; 8]>,
    /// Confidence score based on cluster similarity.
    pub confidence: f32,
}

impl CondensedDocument {
    /// Create a new condensed document.
    pub fn new(content: String, source_ids: SmallVec<[u64; 8]>, confidence: f32) -> Self {
        Self {
            content,
            source_count: source_ids.len() as u16,
            source_ids,
            keywords: SmallVec::new(),
            category_path: SmallVec::new(),
            tags: SmallVec::new(),
            confidence,
        }
    }

    /// Set keywords.
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
}

/// Result from a condensation operation.
#[derive(Debug)]
pub struct CondenseResult {
    /// The condensed document.
    pub document: CondensedDocument,
    /// Number of unique facts preserved.
    pub facts_preserved: usize,
    /// Number of duplicates removed.
    pub duplicates_removed: usize,
    /// Contradictions noted (if enabled).
    pub contradictions: Vec<String>,
}

/// Engine for condensing clusters of similar documents.
///
/// This provides the logic for merging documents, but the actual
/// LLM call for generating condensed content is done by the caller.
#[derive(Default)]
pub struct CondenseEngine {
    config: CondenseConfig,
    keyword_extractor: KeywordExtractor,
}

impl CondenseEngine {
    /// Create with custom configuration.
    pub fn with_config(config: CondenseConfig) -> Self {
        Self {
            config,
            keyword_extractor: KeywordExtractor::new(),
        }
    }

    /// Build a condensation prompt for the LLM.
    ///
    /// Returns the prompt text that should be sent to an LLM to generate
    /// the condensed document.
    pub fn build_prompt<'a>(&self, contents: &[StrView<'a>]) -> String {
        let mut prompt =
            String::with_capacity(contents.iter().map(|c| c.len()).sum::<usize>() + 512);

        prompt.push_str("Merge these similar documents into one comprehensive document:\n\n");

        for (i, content) in contents.iter().enumerate() {
            prompt.push_str(&format!(
                "--- Document {} ---\n{}\n\n",
                i + 1,
                content.as_str()
            ));
        }

        prompt.push_str("Create a single document that:\n");
        if self.config.preserve_all_facts {
            prompt.push_str("1. Preserves ALL unique facts and information\n");
        }
        prompt.push_str("2. Eliminates redundancy and repetition\n");
        if self.config.note_contradictions {
            prompt.push_str("3. Notes any contradictions between sources\n");
        }
        prompt.push_str("4. Maintains a clear, coherent structure\n");
        prompt.push_str("\nCondensed document:\n");

        prompt
    }

    /// Merge keywords from multiple documents.
    ///
    /// Combines TF-IDF scores, keeping highest score for duplicates.
    pub fn merge_keywords(&self, docs: &[DocumentFeatures<'_>]) -> SmallVec<[(Sym, f32); 16]> {
        use std::collections::HashMap;

        let mut merged: HashMap<Sym, f32> = HashMap::new();

        for doc in docs {
            for &(sym, score) in &doc.keywords {
                merged
                    .entry(sym)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
        }

        let mut result: SmallVec<[(Sym, f32); 16]> = merged.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(16);
        result
    }

    /// Merge category paths from multiple documents.
    ///
    /// Returns the longest path that is a prefix of all paths, or the
    /// most common path if no common prefix exists.
    pub fn merge_category_paths(&self, docs: &[DocumentFeatures<'_>]) -> SmallVec<[Sym; 8]> {
        if docs.is_empty() {
            return SmallVec::new();
        }

        // Find longest common prefix
        let first = &docs[0].category_path;
        if first.is_empty() {
            // Find the first non-empty path
            for doc in docs {
                if !doc.category_path.is_empty() {
                    return doc.category_path.clone();
                }
            }
            return SmallVec::new();
        }

        let mut prefix_len = first.len();
        for doc in &docs[1..] {
            let common = first
                .iter()
                .zip(doc.category_path.iter())
                .take_while(|(a, b)| a == b)
                .count();
            prefix_len = prefix_len.min(common);
        }

        if prefix_len > 0 {
            first[..prefix_len].iter().copied().collect()
        } else {
            // No common prefix, return the most specific (longest) path
            docs.iter()
                .max_by_key(|d| d.category_path.len())
                .map(|d| d.category_path.clone())
                .unwrap_or_default()
        }
    }

    /// Merge tags from multiple documents (union).
    pub fn merge_tags(&self, docs: &[DocumentFeatures<'_>]) -> SmallVec<[Sym; 8]> {
        use std::collections::HashSet;

        let all_tags: HashSet<Sym> = docs.iter().flat_map(|d| d.tags.iter().copied()).collect();

        all_tags.into_iter().collect()
    }

    /// Create a condensed document from cluster.
    ///
    /// The `condensed_content` should be the LLM-generated merged text.
    pub fn create_condensed(
        &self,
        cluster: &DocumentCluster,
        docs: &[DocumentFeatures<'_>],
        condensed_content: String,
    ) -> CondensedDocument {
        // Get documents in this cluster
        let cluster_docs: Vec<DocumentFeatures<'_>> = cluster
            .doc_indices
            .iter()
            .filter_map(|&i| docs.get(i))
            .map(|d| DocumentFeatures {
                id: d.id,
                keywords: d.keywords.clone(),
                category_path: d.category_path.clone(),
                tags: d.tags.clone(),
                embedding: d.embedding,
            })
            .collect();

        let source_ids: SmallVec<[u64; 8]> = cluster_docs.iter().map(|d| d.id).collect();
        let keywords = self.merge_keywords(&cluster_docs);
        let category_path = self.merge_category_paths(&cluster_docs);
        let tags = self.merge_tags(&cluster_docs);

        CondensedDocument {
            content: condensed_content,
            source_ids,
            source_count: cluster.doc_indices.len() as u16,
            keywords,
            category_path,
            tags,
            confidence: cluster.avg_similarity,
        }
    }

    /// Simple text-based condensation without LLM.
    ///
    /// Extracts unique sentences and combines them. This is a fallback
    /// when no LLM is available.
    pub fn condense_simple<'a>(&self, contents: &[StrView<'a>]) -> String {
        use std::collections::HashSet;

        let mut seen_sentences: HashSet<String> = HashSet::new();
        let mut result = String::new();

        for content in contents {
            for sentence in content.as_str().split(|c| c == '.' || c == '!' || c == '?') {
                let trimmed = sentence.trim();
                if !trimmed.is_empty() {
                    let normalized = trimmed.to_lowercase();
                    if seen_sentences.insert(normalized) {
                        if !result.is_empty() {
                            result.push_str(". ");
                        }
                        result.push_str(trimmed);
                    }
                }
            }
        }

        if !result.is_empty() && !result.ends_with('.') {
            result.push('.');
        }

        result
    }

    /// Extract keywords from condensed content.
    pub fn extract_keywords(&self, content: &str) -> SmallVec<[(Sym, f32); 16]> {
        self.keyword_extractor.extract_sorted(content)
    }
}

/// Batch condensation result.
#[derive(Debug)]
pub struct BatchCondenseResult {
    /// Successfully condensed documents.
    pub condensed: Vec<CondensedDocument>,
    /// Clusters that were too small to condense.
    pub skipped_clusters: usize,
    /// Total source documents processed.
    pub total_sources: usize,
}

impl BatchCondenseResult {
    /// Get compression ratio (condensed / original count).
    pub fn compression_ratio(&self) -> f32 {
        if self.total_sources == 0 {
            1.0
        } else {
            self.condensed.len() as f32 / self.total_sources as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    #[test]
    fn test_condense_config_default() {
        let config = CondenseConfig::default();
        assert_eq!(config.max_batch_size, 10);
        assert!(config.preserve_all_facts);
        assert!(config.note_contradictions);
    }

    #[test]
    fn test_condensed_document_builder() {
        let doc = CondensedDocument::new(
            "merged content".to_string(),
            smallvec::smallvec![1, 2, 3],
            0.9,
        )
        .with_keywords(smallvec::smallvec![(sym("rust"), 0.8)])
        .with_category_path(smallvec::smallvec![sym("lang")])
        .with_tags(smallvec::smallvec![sym("fast")]);

        assert_eq!(doc.content, "merged content");
        assert_eq!(doc.source_count, 3);
        assert_eq!(doc.keywords.len(), 1);
        assert!((doc.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_build_prompt() {
        let engine = CondenseEngine::default();
        let contents = vec![
            StrView::new("Document one content"),
            StrView::new("Document two content"),
        ];

        let prompt = engine.build_prompt(&contents);

        assert!(prompt.contains("Document 1"));
        assert!(prompt.contains("Document 2"));
        assert!(prompt.contains("Document one content"));
        assert!(prompt.contains("Document two content"));
        assert!(prompt.contains("Preserves ALL unique facts"));
    }

    #[test]
    fn test_merge_keywords() {
        let engine = CondenseEngine::default();

        let mut doc1 = DocumentFeatures::new(1);
        doc1.keywords = smallvec::smallvec![(sym("rust"), 0.9), (sym("code"), 0.5)];

        let mut doc2 = DocumentFeatures::new(2);
        doc2.keywords = smallvec::smallvec![(sym("rust"), 0.7), (sym("async"), 0.6)];

        let merged = engine.merge_keywords(&[doc1, doc2]);

        // "rust" should have max score of 0.9
        let rust_score = merged
            .iter()
            .find(|(s, _)| *s == sym("rust"))
            .map(|(_, score)| *score);
        assert!((rust_score.unwrap() - 0.9).abs() < 0.001);

        // Should have all 3 unique keywords
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merge_category_paths() {
        let engine = CondenseEngine::default();

        let mut doc1 = DocumentFeatures::new(1);
        doc1.category_path = smallvec::smallvec![sym("lang"), sym("systems"), sym("rust")];

        let mut doc2 = DocumentFeatures::new(2);
        doc2.category_path = smallvec::smallvec![sym("lang"), sym("systems"), sym("cpp")];

        let merged = engine.merge_category_paths(&[doc1, doc2]);

        // Common prefix is ["lang", "systems"]
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], sym("lang"));
        assert_eq!(merged[1], sym("systems"));
    }

    #[test]
    fn test_merge_tags() {
        let engine = CondenseEngine::default();

        let mut doc1 = DocumentFeatures::new(1);
        doc1.tags = smallvec::smallvec![sym("fast"), sym("safe")];

        let mut doc2 = DocumentFeatures::new(2);
        doc2.tags = smallvec::smallvec![sym("safe"), sym("concurrent")];

        let merged = engine.merge_tags(&[doc1, doc2]);

        // Should be union: fast, safe, concurrent
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_condense_simple() {
        let engine = CondenseEngine::default();
        let contents = vec![
            StrView::new("Rust is fast. Rust is safe."),
            StrView::new("Rust is safe. Rust is concurrent."),
        ];

        let condensed = engine.condense_simple(&contents);

        // Should deduplicate "Rust is safe"
        assert!(condensed.contains("Rust is fast"));
        assert!(condensed.contains("Rust is safe"));
        assert!(condensed.contains("Rust is concurrent"));

        // Count occurrences of "safe" - should only appear once
        let safe_count = condensed.matches("safe").count();
        assert_eq!(safe_count, 1);
    }

    #[test]
    fn test_batch_condense_result() {
        let result = BatchCondenseResult {
            condensed: vec![CondensedDocument::new(
                "a".to_string(),
                smallvec::smallvec![1, 2, 3],
                0.9,
            )],
            skipped_clusters: 2,
            total_sources: 10,
        };

        assert!((result.compression_ratio() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_extract_keywords() {
        let engine = CondenseEngine::default();
        let keywords = engine.extract_keywords("Rust programming language for systems");

        assert!(!keywords.is_empty());
        let syms: Vec<Sym> = keywords.iter().map(|(s, _)| *s).collect();
        assert!(syms.contains(&sym("rust")));
    }
}
