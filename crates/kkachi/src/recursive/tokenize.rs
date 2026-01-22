// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Token counting and encoding for LLM text processing.
//!
//! This module provides a unified interface for tokenization across different
//! LLM providers (OpenAI, Anthropic, HuggingFace models).
//!
//! # Features
//!
//! - `tiktoken` - Enable tiktoken-rs for OpenAI/Anthropic models (MIT license)
//! - `huggingface` - Enable HuggingFace tokenizers for open models (Apache 2.0)
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::tokenize::{Tokenizer, TiktokenTokenizer};
//!
//! let tokenizer = TiktokenTokenizer::claude();
//! let count = tokenizer.count_tokens("Hello, world!");
//! println!("Token count: {}", count);
//! ```

#[cfg(any(feature = "tiktoken", feature = "huggingface"))]
use crate::error::{Error, Result};

// ============================================================================
// Tokenizer Trait
// ============================================================================

/// Zero-copy tokenizer trait for token counting and encoding.
///
/// This trait provides a unified interface for different tokenizer backends.
/// Implementations should be thread-safe (`Send + Sync`).
pub trait Tokenizer: Send + Sync {
    /// Count tokens in text without allocating token IDs.
    ///
    /// This is the most efficient operation for just getting token counts.
    fn count_tokens(&self, text: &str) -> usize;

    /// Encode text to token IDs.
    ///
    /// Returns a vector of token IDs that can be decoded back to text.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode token IDs back to text.
    ///
    /// May return lossy results for invalid token sequences.
    fn decode(&self, tokens: &[u32]) -> String;

    /// Get the model name/identifier for this tokenizer.
    fn model(&self) -> &str;

    /// Get the vocabulary size.
    fn vocab_size(&self) -> usize {
        0 // Default unknown
    }

    /// Check if a special token exists.
    fn has_special_token(&self, token: &str) -> bool {
        let _ = token;
        false
    }
}

// ============================================================================
// Tiktoken Implementation (OpenAI/Anthropic models)
// ============================================================================

#[cfg(feature = "tiktoken")]
pub use tiktoken_impl::*;

#[cfg(feature = "tiktoken")]
mod tiktoken_impl {
    use super::*;

    /// Tiktoken-based tokenizer for OpenAI/Anthropic models.
    ///
    /// Supports GPT-2, GPT-3.5, GPT-4, and Claude models via cl100k_base encoding.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::recursive::tokenize::TiktokenTokenizer;
    ///
    /// // For Claude models
    /// let tokenizer = TiktokenTokenizer::claude();
    ///
    /// // For GPT-4
    /// let tokenizer = TiktokenTokenizer::gpt4();
    ///
    /// // Custom encoding
    /// let tokenizer = TiktokenTokenizer::with_encoding("cl100k_base")?;
    /// ```
    pub struct TiktokenTokenizer {
        bpe: std::sync::Arc<tiktoken_rs::CoreBPE>,
        model: &'static str,
        vocab_size: usize,
    }

    impl Clone for TiktokenTokenizer {
        fn clone(&self) -> Self {
            Self {
                bpe: std::sync::Arc::clone(&self.bpe),
                model: self.model,
                vocab_size: self.vocab_size,
            }
        }
    }

    impl TiktokenTokenizer {
        /// Create tokenizer for Claude models (cl100k_base encoding).
        ///
        /// Claude uses a similar tokenization to GPT-4.
        pub fn claude() -> Self {
            let bpe = tiktoken_rs::cl100k_base().expect("cl100k_base should always be available");
            Self {
                bpe: std::sync::Arc::new(bpe),
                model: "claude",
                vocab_size: 100277,
            }
        }

        /// Create tokenizer for GPT-4 models (cl100k_base encoding).
        pub fn gpt4() -> Self {
            let bpe = tiktoken_rs::cl100k_base().expect("cl100k_base should always be available");
            Self {
                bpe: std::sync::Arc::new(bpe),
                model: "gpt-4",
                vocab_size: 100277,
            }
        }

        /// Create tokenizer for GPT-3.5 models (cl100k_base encoding).
        pub fn gpt35() -> Self {
            let bpe = tiktoken_rs::cl100k_base().expect("cl100k_base should always be available");
            Self {
                bpe: std::sync::Arc::new(bpe),
                model: "gpt-3.5-turbo",
                vocab_size: 100277,
            }
        }

        /// Create tokenizer for GPT-2 models (r50k_base encoding).
        pub fn gpt2() -> Self {
            let bpe = tiktoken_rs::r50k_base().expect("r50k_base should always be available");
            Self {
                bpe: std::sync::Arc::new(bpe),
                model: "gpt-2",
                vocab_size: 50257,
            }
        }

        /// Create tokenizer with a specific encoding.
        ///
        /// Supported encodings: "cl100k_base", "p50k_base", "r50k_base", "o200k_base"
        pub fn with_encoding(encoding: &str) -> Result<Self> {
            let (bpe, vocab_size) = match encoding {
                "cl100k_base" => (
                    tiktoken_rs::cl100k_base().map_err(|e| Error::parse(e.to_string()))?,
                    100277,
                ),
                "p50k_base" => (
                    tiktoken_rs::p50k_base().map_err(|e| Error::parse(e.to_string()))?,
                    50281,
                ),
                "r50k_base" => (
                    tiktoken_rs::r50k_base().map_err(|e| Error::parse(e.to_string()))?,
                    50257,
                ),
                "o200k_base" => (
                    tiktoken_rs::o200k_base().map_err(|e| Error::parse(e.to_string()))?,
                    200019,
                ),
                _ => return Err(Error::parse(format!("Unknown encoding: {}", encoding))),
            };

            Ok(Self {
                bpe: std::sync::Arc::new(bpe),
                model: "custom",
                vocab_size,
            })
        }

        /// Get the underlying BPE encoder.
        pub fn bpe(&self) -> &tiktoken_rs::CoreBPE {
            &self.bpe
        }
    }

    impl Tokenizer for TiktokenTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            self.bpe.encode_ordinary(text).len()
        }

        fn encode(&self, text: &str) -> Vec<u32> {
            self.bpe
                .encode_ordinary(text)
                .into_iter()
                .map(|t| t as u32)
                .collect()
        }

        fn decode(&self, tokens: &[u32]) -> String {
            self.bpe
                .decode(tokens.to_vec())
                .unwrap_or_else(|_| String::new())
        }

        fn model(&self) -> &str {
            self.model
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    impl std::fmt::Debug for TiktokenTokenizer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TiktokenTokenizer")
                .field("model", &self.model)
                .field("vocab_size", &self.vocab_size)
                .finish()
        }
    }
}

// ============================================================================
// HuggingFace Tokenizers Implementation
// ============================================================================

#[cfg(feature = "huggingface")]
pub use huggingface_impl::*;

#[cfg(feature = "huggingface")]
mod huggingface_impl {
    use super::*;
    use std::sync::Arc;

    /// HuggingFace tokenizer for open models (Llama, Mistral, etc.).
    ///
    /// Supports loading from HuggingFace Hub or local files.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::recursive::tokenize::HuggingFaceTokenizer;
    ///
    /// // From HuggingFace Hub
    /// let tokenizer = HuggingFaceTokenizer::from_pretrained("meta-llama/Llama-2-7b")?;
    ///
    /// // From local file
    /// let tokenizer = HuggingFaceTokenizer::from_file("tokenizer.json")?;
    /// ```
    #[derive(Clone)]
    pub struct HuggingFaceTokenizer {
        tokenizer: Arc<tokenizers::Tokenizer>,
        model: String,
    }

    impl HuggingFaceTokenizer {
        /// Load tokenizer from HuggingFace Hub.
        ///
        /// # Arguments
        /// * `identifier` - Model identifier (e.g., "meta-llama/Llama-2-7b")
        pub fn from_pretrained(identifier: &str) -> Result<Self> {
            let tokenizer = tokenizers::Tokenizer::from_pretrained(identifier, None)
                .map_err(|e| Error::parse(format!("Failed to load tokenizer: {}", e)))?;

            Ok(Self {
                tokenizer: Arc::new(tokenizer),
                model: identifier.to_string(),
            })
        }

        /// Load tokenizer from a local JSON file.
        pub fn from_file(path: &str) -> Result<Self> {
            let tokenizer = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| Error::parse(format!("Failed to load tokenizer: {}", e)))?;

            Ok(Self {
                tokenizer: Arc::new(tokenizer),
                model: path.to_string(),
            })
        }

        /// Get the underlying tokenizer.
        pub fn inner(&self) -> &tokenizers::Tokenizer {
            &self.tokenizer
        }
    }

    impl Tokenizer for HuggingFaceTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            self.tokenizer
                .encode(text, false)
                .map(|e| e.get_ids().len())
                .unwrap_or(0)
        }

        fn encode(&self, text: &str) -> Vec<u32> {
            self.tokenizer
                .encode(text, false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default()
        }

        fn decode(&self, tokens: &[u32]) -> String {
            self.tokenizer
                .decode(tokens, false)
                .unwrap_or_else(|_| String::new())
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn vocab_size(&self) -> usize {
            self.tokenizer.get_vocab_size(true)
        }

        fn has_special_token(&self, token: &str) -> bool {
            self.tokenizer.token_to_id(token).is_some()
        }
    }

    impl std::fmt::Debug for HuggingFaceTokenizer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HuggingFaceTokenizer")
                .field("model", &self.model)
                .field("vocab_size", &self.vocab_size())
                .finish()
        }
    }
}

// ============================================================================
// Simple Tokenizer (Fallback)
// ============================================================================

/// Simple whitespace-based tokenizer for testing and fallback.
///
/// This tokenizer splits on whitespace and punctuation. It's not suitable
/// for production use but works without external dependencies.
#[derive(Debug, Clone, Default)]
pub struct SimpleTokenizer {
    model: &'static str,
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer.
    pub fn new() -> Self {
        Self { model: "simple" }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        // Split on whitespace and count non-empty tokens
        text.split_whitespace().count()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // Simple hash-based encoding
        text.split_whitespace()
            .enumerate()
            .map(|(i, _)| i as u32)
            .collect()
    }

    fn decode(&self, _tokens: &[u32]) -> String {
        // Cannot decode without storing original text
        String::new()
    }

    fn model(&self) -> &str {
        self.model
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_count() {
        let tokenizer = SimpleTokenizer::new();
        assert_eq!(tokenizer.count_tokens("Hello world"), 2);
        assert_eq!(tokenizer.count_tokens(""), 0);
        assert_eq!(tokenizer.count_tokens("one"), 1);
        assert_eq!(tokenizer.count_tokens("  multiple   spaces  "), 2);
    }

    #[test]
    fn test_simple_tokenizer_model() {
        let tokenizer = SimpleTokenizer::new();
        assert_eq!(tokenizer.model(), "simple");
    }

    #[cfg(feature = "tiktoken")]
    mod tiktoken_tests {
        use super::*;

        #[test]
        fn test_tiktoken_count_tokens() {
            let tokenizer = TiktokenTokenizer::claude();
            let count = tokenizer.count_tokens("Hello, world!");
            assert!(count > 0);
            assert!(count < 10); // Should be around 4 tokens
        }

        #[test]
        fn test_tiktoken_encode_decode_roundtrip() {
            let tokenizer = TiktokenTokenizer::claude();
            let text = "Hello, world! This is a test.";
            let tokens = tokenizer.encode(text);
            let decoded = tokenizer.decode(&tokens);
            assert_eq!(decoded, text);
        }

        #[test]
        fn test_tiktoken_claude_model() {
            let tokenizer = TiktokenTokenizer::claude();
            assert_eq!(tokenizer.model(), "claude");
            assert_eq!(tokenizer.vocab_size(), 100277);
        }

        #[test]
        fn test_tiktoken_gpt4_model() {
            let tokenizer = TiktokenTokenizer::gpt4();
            assert_eq!(tokenizer.model(), "gpt-4");
            assert_eq!(tokenizer.vocab_size(), 100277);
        }

        #[test]
        fn test_tiktoken_empty_text() {
            let tokenizer = TiktokenTokenizer::claude();
            assert_eq!(tokenizer.count_tokens(""), 0);
            assert!(tokenizer.encode("").is_empty());
        }

        #[test]
        fn test_tiktoken_unicode_handling() {
            let tokenizer = TiktokenTokenizer::claude();
            let text = "Hello 世界! 🌍";
            let tokens = tokenizer.encode(text);
            let decoded = tokenizer.decode(&tokens);
            assert_eq!(decoded, text);
        }

        #[test]
        fn test_tiktoken_with_encoding() {
            let tokenizer = TiktokenTokenizer::with_encoding("cl100k_base").unwrap();
            assert!(tokenizer.count_tokens("test") > 0);
        }

        #[test]
        fn test_tiktoken_invalid_encoding() {
            let result = TiktokenTokenizer::with_encoding("invalid_encoding");
            assert!(result.is_err());
        }
    }

    #[cfg(feature = "huggingface")]
    mod huggingface_tests {
        use super::*;

        // Note: These tests require network access to HuggingFace Hub
        // They are marked as ignore by default

        #[test]
        #[ignore = "requires network access"]
        fn test_huggingface_from_pretrained() {
            let tokenizer = HuggingFaceTokenizer::from_pretrained("bert-base-uncased").unwrap();
            let count = tokenizer.count_tokens("Hello, world!");
            assert!(count > 0);
        }

        #[test]
        #[ignore = "requires network access"]
        fn test_huggingface_encode_decode() {
            let tokenizer = HuggingFaceTokenizer::from_pretrained("bert-base-uncased").unwrap();
            let text = "Hello world";
            let tokens = tokenizer.encode(text);
            assert!(!tokens.is_empty());
        }
    }
}
