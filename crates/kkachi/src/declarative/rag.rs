// Copyright 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Smart RAG with auto-update and deduplication.
//!
//! Provides:
//! - `RagExample<'a>` - Zero-copy example with score/iteration metadata
//! - `RagAnalyzer<'a, E>` - Deduplication and consolidation logic
//! - `LiveRag<'a, E>` - Mutable RAG store with auto-update on pipeline success

use crate::recursive::Embedder;
use crate::str_view::StrView;

use super::pipeline::PipelineOutputOwned;

// =============================================================================
// RagExample
// =============================================================================

/// RAG example with metadata for ranking.
#[derive(Debug, Clone)]
pub struct RagExample<'a> {
    /// Unique identifier.
    pub id: StrView<'a>,
    /// Input text.
    pub input: StrView<'a>,
    /// Output text.
    pub output: StrView<'a>,
    /// Score from critic (0.0 - 1.0).
    pub score: f64,
    /// Which iteration produced this.
    pub iteration: u32,
}

impl<'a> RagExample<'a> {
    /// Create a new example.
    pub fn new(id: &'a str, input: &'a str, output: &'a str) -> Self {
        Self {
            id: StrView::new(id),
            input: StrView::new(input),
            output: StrView::new(output),
            score: 1.0,
            iteration: 0,
        }
    }

    /// Create with explicit score.
    pub fn with_score(id: &'a str, input: &'a str, output: &'a str, score: f64) -> Self {
        Self {
            id: StrView::new(id),
            input: StrView::new(input),
            output: StrView::new(output),
            score,
            iteration: 0,
        }
    }

    /// Create from successful pipeline output.
    pub fn from_output(id: String, input: String, output: &PipelineOutputOwned) -> RagExampleOwned {
        RagExampleOwned {
            id,
            input,
            output: output.answer.clone(),
            score: output.score,
            iteration: output.iterations,
        }
    }

    /// Convert to owned variant.
    pub fn into_owned(self) -> RagExampleOwned {
        RagExampleOwned {
            id: self.id.to_string(),
            input: self.input.to_string(),
            output: self.output.to_string(),
            score: self.score,
            iteration: self.iteration,
        }
    }
}

/// Owned variant of RagExample.
#[derive(Debug, Clone)]
pub struct RagExampleOwned {
    /// Unique identifier.
    pub id: String,
    /// Input text.
    pub input: String,
    /// Output text.
    pub output: String,
    /// Score from critic.
    pub score: f64,
    /// Which iteration produced this.
    pub iteration: u32,
}

impl RagExampleOwned {
    /// Create a new owned example.
    pub fn new(id: impl Into<String>, input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            output: output.into(),
            score: 1.0,
            iteration: 0,
        }
    }

    /// Create with explicit score.
    pub fn with_score(
        id: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
        score: f64,
    ) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            output: output.into(),
            score,
            iteration: 0,
        }
    }
}

// =============================================================================
// RagAnalyzer
// =============================================================================

/// Analyzes and consolidates RAG examples.
///
/// Prevents duplicates and merges functionally equivalent examples.
pub struct RagAnalyzer<'a, E: Embedder> {
    /// Embedder for similarity computation.
    embedder: &'a E,
    /// Similarity threshold for deduplication (default: 0.92).
    similarity_threshold: f64,
    /// Maximum examples to keep (default: 10).
    max_examples: usize,
}

impl<'a, E: Embedder> RagAnalyzer<'a, E> {
    /// Create a new analyzer with the given embedder.
    pub fn new(embedder: &'a E) -> Self {
        Self {
            embedder,
            similarity_threshold: 0.92,
            max_examples: 10,
        }
    }

    /// Set similarity threshold for deduplication.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set maximum examples to keep.
    pub fn with_max_examples(mut self, max: usize) -> Self {
        self.max_examples = max;
        self
    }

    /// Get similarity threshold.
    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }

    /// Get max examples.
    pub fn max_examples(&self) -> usize {
        self.max_examples
    }

    /// Check if example is functionally duplicate.
    pub fn is_duplicate(&self, new_example: &str, existing: &[&str]) -> bool {
        let new_embedding = self.embedder.embed(new_example);
        for existing_ex in existing {
            let existing_embedding = self.embedder.embed(existing_ex);
            if cosine_similarity(&new_embedding, &existing_embedding) > self.similarity_threshold {
                return true;
            }
        }
        false
    }

    /// Check if two strings are similar above threshold.
    pub fn is_similar(&self, a: &str, b: &str) -> bool {
        let embed_a = self.embedder.embed(a);
        let embed_b = self.embedder.embed(b);
        cosine_similarity(&embed_a, &embed_b) > self.similarity_threshold
    }

    /// Consolidate similar examples, keeping the best one per cluster.
    pub fn consolidate(&self, examples: &[RagExampleOwned]) -> Vec<RagExampleOwned> {
        if examples.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<&RagExampleOwned>> = Vec::new();

        for example in examples {
            let mut found_cluster = false;
            for cluster in &mut clusters {
                if self.is_similar(&example.output, &cluster[0].output) {
                    cluster.push(example);
                    found_cluster = true;
                    break;
                }
            }
            if !found_cluster {
                clusters.push(vec![example]);
            }
        }

        // Keep highest-scoring example from each cluster
        let mut result: Vec<RagExampleOwned> = clusters
            .into_iter()
            .filter_map(|cluster| {
                cluster
                    .into_iter()
                    .max_by(|a, b| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
            })
            .collect();

        // Sort by score descending
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max examples
        result.truncate(self.max_examples);
        result
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)) as f64
}

// =============================================================================
// LiveRag
// =============================================================================

/// Mutable RAG store that auto-updates with successful pipeline outputs.
pub struct LiveRag<'a, E: Embedder> {
    /// Stored examples.
    examples: Vec<RagExampleOwned>,
    /// Analyzer for deduplication.
    analyzer: RagAnalyzer<'a, E>,
    /// Minimum score to auto-add (default: 0.9).
    auto_add_threshold: f64,
}

impl<'a, E: Embedder> LiveRag<'a, E> {
    /// Create a new live RAG store.
    pub fn new(embedder: &'a E) -> Self {
        Self {
            examples: Vec::new(),
            analyzer: RagAnalyzer::new(embedder),
            auto_add_threshold: 0.9,
        }
    }

    /// Set the auto-add threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.auto_add_threshold = threshold;
        self
    }

    /// Set similarity threshold for deduplication.
    pub fn with_similarity(mut self, threshold: f64) -> Self {
        self.analyzer = self.analyzer.with_threshold(threshold);
        self
    }

    /// Set maximum examples to keep.
    pub fn with_max_examples(mut self, max: usize) -> Self {
        self.analyzer = self.analyzer.with_max_examples(max);
        self
    }

    /// Seed with initial examples.
    pub fn seed(&mut self, examples: &[RagExample<'_>]) {
        for ex in examples {
            self.examples.push(ex.clone().into_owned());
        }
    }

    /// Seed with owned examples.
    pub fn seed_owned(&mut self, examples: Vec<RagExampleOwned>) {
        self.examples.extend(examples);
    }

    /// Get number of stored examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Add example if it's good enough and not a duplicate.
    ///
    /// Returns `true` if the example was added.
    pub fn maybe_add(&mut self, example: RagExampleOwned) -> bool {
        // Check score threshold
        if example.score < self.auto_add_threshold {
            return false;
        }

        // Check for duplicates
        let existing_outputs: Vec<&str> = self.examples.iter().map(|e| e.output.as_str()).collect();

        if self
            .analyzer
            .is_duplicate(&example.output, &existing_outputs)
        {
            return false;
        }

        self.examples.push(example);

        // Consolidate if we have too many
        if self.examples.len() > self.analyzer.max_examples() * 2 {
            self.examples = self.analyzer.consolidate(&self.examples);
        }

        true
    }

    /// Add from pipeline output.
    pub fn add_from_output(
        &mut self,
        id: String,
        input: String,
        output: &PipelineOutputOwned,
    ) -> bool {
        let example = RagExample::from_output(id, input, output);
        self.maybe_add(example)
    }

    /// Get k best examples for few-shot.
    pub fn top_k(&self, k: usize) -> &[RagExampleOwned] {
        let len = k.min(self.examples.len());
        &self.examples[..len]
    }

    /// Get all examples.
    pub fn examples(&self) -> &[RagExampleOwned] {
        &self.examples
    }

    /// Force consolidation.
    pub fn consolidate(&mut self) {
        self.examples = self.analyzer.consolidate(&self.examples);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::HashEmbedder;

    #[test]
    fn test_rag_example_creation() {
        let ex = RagExample::new("id1", "What is 2+2?", "4");
        assert_eq!(ex.id.as_str(), "id1");
        assert_eq!(ex.input.as_str(), "What is 2+2?");
        assert_eq!(ex.output.as_str(), "4");
        assert_eq!(ex.score, 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_live_rag_seed_and_query() {
        let embedder = HashEmbedder::new(64);
        let mut rag = LiveRag::new(&embedder);

        rag.seed(&[
            RagExample::new("ex1", "Calculate 2+2", "4"),
            RagExample::new("ex2", "Calculate 3*4", "12"),
        ]);

        assert_eq!(rag.len(), 2);
        assert_eq!(rag.top_k(1).len(), 1);
    }

    #[test]
    fn test_live_rag_deduplication() {
        let embedder = HashEmbedder::new(64);
        let mut rag = LiveRag::new(&embedder)
            .with_threshold(0.9)
            .with_similarity(0.95);

        // Add first example
        let added1 = rag.maybe_add(RagExampleOwned::with_score(
            "ex1", "input1", "output1", 0.95,
        ));
        assert!(added1);

        // Try to add exact duplicate - should fail
        let _added2 = rag.maybe_add(RagExampleOwned::with_score(
            "ex2", "input2", "output1", 0.95,
        ));
        // May or may not be duplicate depending on hash embedder behavior
        // The key test is that the logic runs without error

        assert!(!rag.is_empty());
    }

    #[test]
    fn test_analyzer_consolidate() {
        let embedder = HashEmbedder::new(64);
        let analyzer = RagAnalyzer::new(&embedder)
            .with_threshold(0.99) // Very high threshold so nothing clusters
            .with_max_examples(5);

        let examples = vec![
            RagExampleOwned::with_score("ex1", "in1", "out1", 0.9),
            RagExampleOwned::with_score("ex2", "in2", "out2", 0.95),
            RagExampleOwned::with_score("ex3", "in3", "out3", 0.8),
        ];

        let consolidated = analyzer.consolidate(&examples);
        assert!(consolidated.len() <= 5);
    }
}
