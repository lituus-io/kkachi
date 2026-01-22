// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Local keyword extraction using TF-IDF.
//!
//! No LLM calls required - purely statistical keyword extraction
//! with sub-millisecond performance.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::intern::{sym, Sym};

/// Stop words for English text (common words to filter out).
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also", "now", "here", "there",
    "then",
];

/// Local keyword extraction using TF-IDF.
pub struct KeywordExtractor {
    /// Pre-computed IDF scores from corpus.
    idf: HashMap<Sym, f32>,
    /// Stop words as interned symbols for O(1) lookup.
    stop_words: Vec<Sym>,
    /// Minimum word length to consider.
    min_word_length: usize,
    /// Maximum keywords to extract per document.
    max_keywords: usize,
}

impl Default for KeywordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordExtractor {
    /// Create a new keyword extractor with default settings.
    pub fn new() -> Self {
        let stop_words: Vec<Sym> = STOP_WORDS.iter().map(|w| sym(w)).collect();

        Self {
            idf: HashMap::new(),
            stop_words,
            min_word_length: 2,
            max_keywords: 16,
        }
    }

    /// Set minimum word length.
    pub fn with_min_word_length(mut self, len: usize) -> Self {
        self.min_word_length = len;
        self
    }

    /// Set maximum keywords to extract.
    pub fn with_max_keywords(mut self, max: usize) -> Self {
        self.max_keywords = max;
        self
    }

    /// Add custom stop words.
    pub fn with_stop_words(mut self, words: &[&str]) -> Self {
        for word in words {
            self.stop_words.push(sym(word));
        }
        self
    }

    /// Update IDF scores from a corpus of documents.
    ///
    /// Call this periodically to update keyword importance based on corpus statistics.
    pub fn update_idf_from_corpus(&mut self, documents: &[&str]) {
        if documents.is_empty() {
            return;
        }

        let total_docs = documents.len() as f32;
        let mut doc_freq: HashMap<Sym, u32> = HashMap::new();

        // Count document frequency for each term
        for doc in documents {
            let mut seen: std::collections::HashSet<Sym> = std::collections::HashSet::new();
            for token in self.tokenize(doc) {
                if seen.insert(token) {
                    *doc_freq.entry(token).or_insert(0) += 1;
                }
            }
        }

        // Compute IDF: log(N / df)
        self.idf.clear();
        for (term, df) in doc_freq {
            self.idf.insert(term, (total_docs / df as f32).ln());
        }
    }

    /// Set IDF scores directly (e.g., loaded from database).
    pub fn set_idf(&mut self, idf: HashMap<Sym, f32>) {
        self.idf = idf;
    }

    /// Extract keywords with TF-IDF scores (no LLM call, ~100μs).
    ///
    /// Returns keywords sorted by TF-IDF score descending.
    pub fn extract(&self, text: &str) -> SmallVec<[(Sym, f32); 16]> {
        let tokens = self.tokenize(text);
        if tokens.is_empty() {
            return SmallVec::new();
        }

        // Compute term frequency
        let mut tf: HashMap<Sym, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(*token).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as f32;

        // Compute TF-IDF scores
        let mut scores: SmallVec<[(Sym, f32); 16]> = tf
            .iter()
            .map(|(&term, &count)| {
                let tf_score = count as f32 / doc_len;
                // Use IDF if available, otherwise default to 1.0
                let idf_score = self.idf.get(&term).copied().unwrap_or(1.0);
                (term, tf_score * idf_score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to max keywords
        scores.truncate(self.max_keywords);

        scores
    }

    /// Extract keywords and sort by Sym for efficient Jaccard computation.
    pub fn extract_sorted(&self, text: &str) -> SmallVec<[(Sym, f32); 16]> {
        let mut keywords = self.extract(text);
        keywords.sort_by_key(|(s, _)| *s);
        keywords
    }

    /// Tokenize text into words, filtering stop words and short words.
    fn tokenize(&self, text: &str) -> Vec<Sym> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|word| {
                let word_lower = word.to_lowercase();
                word_lower.len() >= self.min_word_length
                    && !self.stop_words.contains(&sym(&word_lower))
            })
            .map(|word| sym(&word.to_lowercase()))
            .collect()
    }

    /// Check if a word is a stop word.
    #[inline]
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(&sym(&word.to_lowercase()))
    }
}

/// Simple n-gram extractor for phrase-level keywords.
pub struct NGramExtractor {
    /// N-gram sizes to extract (e.g., [2, 3] for bigrams and trigrams).
    sizes: Vec<usize>,
    /// Minimum frequency to include.
    min_freq: u32,
    /// Maximum n-grams to return.
    max_ngrams: usize,
}

impl Default for NGramExtractor {
    fn default() -> Self {
        Self {
            sizes: vec![2, 3],
            min_freq: 2,
            max_ngrams: 10,
        }
    }
}

impl NGramExtractor {
    /// Create with specific n-gram sizes.
    pub fn with_sizes(sizes: Vec<usize>) -> Self {
        Self {
            sizes,
            ..Default::default()
        }
    }

    /// Set minimum frequency threshold.
    pub fn with_min_freq(mut self, freq: u32) -> Self {
        self.min_freq = freq;
        self
    }

    /// Extract n-grams from text.
    pub fn extract(&self, text: &str) -> Vec<(String, u32)> {
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        let mut freq: HashMap<String, u32> = HashMap::new();

        for &n in &self.sizes {
            if words.len() < n {
                continue;
            }

            for window in words.windows(n) {
                let ngram = window.join(" ").to_lowercase();
                *freq.entry(ngram).or_insert(0) += 1;
            }
        }

        let mut ngrams: Vec<(String, u32)> = freq
            .into_iter()
            .filter(|(_, count)| *count >= self.min_freq)
            .collect();

        ngrams.sort_by(|a, b| b.1.cmp(&a.1));
        ngrams.truncate(self.max_ngrams);
        ngrams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_extractor_basic() {
        let extractor = KeywordExtractor::new();
        let text = "Rust is a systems programming language focused on safety and performance.";
        let keywords = extractor.extract(text);

        assert!(!keywords.is_empty());
        // Should include meaningful words like "rust", "systems", "programming"
        let keyword_syms: Vec<Sym> = keywords.iter().map(|(s, _)| *s).collect();
        assert!(keyword_syms.contains(&sym("rust")));
        assert!(keyword_syms.contains(&sym("systems")));
        assert!(keyword_syms.contains(&sym("programming")));
    }

    #[test]
    fn test_keyword_extractor_filters_stop_words() {
        let extractor = KeywordExtractor::new();
        let text = "The quick brown fox jumps and the lazy dog.";
        let keywords = extractor.extract(text);

        // Should not include common stop words like "the", "and"
        let keyword_syms: Vec<Sym> = keywords.iter().map(|(s, _)| *s).collect();
        assert!(!keyword_syms.contains(&sym("the")));
        assert!(!keyword_syms.contains(&sym("and")));

        // Should include content words
        assert!(keyword_syms.contains(&sym("quick")));
        assert!(keyword_syms.contains(&sym("brown")));
        assert!(keyword_syms.contains(&sym("fox")));
    }

    #[test]
    fn test_keyword_extractor_with_idf() {
        let mut extractor = KeywordExtractor::new();

        // Build corpus for IDF
        let corpus = vec![
            "rust programming language",
            "python programming language",
            "javascript programming language",
            "rust systems programming",
        ];
        extractor.update_idf_from_corpus(&corpus);

        let keywords = extractor.extract("rust systems programming");

        // Verify keywords are extracted
        assert!(!keywords.is_empty());

        // Find "rust" - should be present
        let rust_score = keywords
            .iter()
            .find(|(s, _)| *s == sym("rust"))
            .map(|(_, score)| *score);
        assert!(rust_score.is_some(), "rust should be extracted as keyword");
        assert!(rust_score.unwrap() > 0.0);

        // "rust" appears in 2/4 docs, "programming" appears in 4/4 docs
        // So "rust" should have higher IDF
        let programming_score = keywords
            .iter()
            .find(|(s, _)| *s == sym("programming"))
            .map(|(_, score)| *score);
        if let (Some(r), Some(p)) = (rust_score, programming_score) {
            // "rust" has IDF = ln(4/2) = 0.693, "programming" has IDF = ln(4/4) = 0
            // So rust should have higher score
            assert!(r > p, "rust should have higher TF-IDF than programming");
        }
    }

    #[test]
    fn test_keyword_extractor_max_keywords() {
        let extractor = KeywordExtractor::new().with_max_keywords(3);
        let text = "one two three four five six seven eight nine ten";
        let keywords = extractor.extract(text);

        assert!(keywords.len() <= 3);
    }

    #[test]
    fn test_keyword_extractor_min_word_length() {
        let extractor = KeywordExtractor::new().with_min_word_length(4);
        let text = "a an the rust code go py";
        let keywords = extractor.extract(text);

        // Should only include "rust" and "code"
        let keyword_syms: Vec<Sym> = keywords.iter().map(|(s, _)| *s).collect();
        assert!(keyword_syms.contains(&sym("rust")));
        assert!(keyword_syms.contains(&sym("code")));
        assert!(!keyword_syms.contains(&sym("go")));
        assert!(!keyword_syms.contains(&sym("py")));
    }

    #[test]
    fn test_keyword_extractor_sorted() {
        let extractor = KeywordExtractor::new();
        let text = "Rust async tokio programming concurrency";
        let keywords = extractor.extract_sorted(text);

        // Verify sorted by Sym
        for i in 1..keywords.len() {
            assert!(keywords[i - 1].0 <= keywords[i].0);
        }
    }

    #[test]
    fn test_is_stop_word() {
        let extractor = KeywordExtractor::new();
        assert!(extractor.is_stop_word("the"));
        assert!(extractor.is_stop_word("The"));
        assert!(extractor.is_stop_word("and"));
        assert!(!extractor.is_stop_word("rust"));
        assert!(!extractor.is_stop_word("programming"));
    }

    #[test]
    fn test_ngram_extractor() {
        let extractor = NGramExtractor::with_sizes(vec![2]).with_min_freq(1);
        let text = "machine learning is great machine learning works well";
        let ngrams = extractor.extract(text);

        assert!(!ngrams.is_empty());
        // "machine learning" should appear twice
        let ml = ngrams.iter().find(|(s, _)| s == "machine learning");
        assert!(ml.is_some());
        assert_eq!(ml.unwrap().1, 2);
    }

    #[test]
    fn test_empty_text() {
        let extractor = KeywordExtractor::new();
        let keywords = extractor.extract("");
        assert!(keywords.is_empty());

        let keywords2 = extractor.extract("   ");
        assert!(keywords2.is_empty());
    }

    #[test]
    fn test_custom_stop_words() {
        let extractor = KeywordExtractor::new().with_stop_words(&["rust", "code"]);
        let text = "Rust code programming";
        let keywords = extractor.extract(text);

        let keyword_syms: Vec<Sym> = keywords.iter().map(|(s, _)| *s).collect();
        assert!(!keyword_syms.contains(&sym("rust")));
        assert!(!keyword_syms.contains(&sym("code")));
        assert!(keyword_syms.contains(&sym("programming")));
    }
}
