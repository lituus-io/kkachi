// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Token-aware text chunking for LLM processing.
//!
//! This module provides semantic text chunking with token-based limits,
//! supporting various splitting strategies and section detection.
//!
//! # Features
//!
//! - Token-based chunk limits (not character-based)
//! - Semantic boundary detection (sentences, paragraphs, sections)
//! - Section type detection for structured output
//! - Overlap support for context continuity
//! - Zero-copy where possible via `Cow<'a, str>`
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::{TextChunker, ChunkConfig, TiktokenTokenizer};
//!
//! let tokenizer = TiktokenTokenizer::claude();
//! let chunker = TextChunker::new(tokenizer)
//!     .with_config(ChunkConfig {
//!         max_tokens: 2048,
//!         overlap_tokens: 128,
//!         ..Default::default()
//!     });
//!
//! let chunks = chunker.chunk("Long document text...");
//! for chunk in chunks {
//!     println!("Chunk {}/{}: {} tokens", chunk.index, chunk.total, chunk.token_count);
//! }
//! ```

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::tokenize::Tokenizer;

// ============================================================================
// Chunk Configuration
// ============================================================================

/// Configuration for text chunking.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Maximum tokens per chunk.
    pub max_tokens: usize,
    /// Overlap tokens between chunks for context continuity.
    pub overlap_tokens: usize,
    /// Strategy for splitting text.
    pub strategy: ChunkStrategy,
    /// Whether to preserve semantic boundaries when splitting.
    pub semantic_split: bool,
    /// Minimum tokens per chunk (avoid tiny chunks).
    pub min_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            overlap_tokens: 128,
            strategy: ChunkStrategy::Sentence,
            semantic_split: true,
            min_tokens: 64,
        }
    }
}

impl ChunkConfig {
    /// Create config for small chunks (suitable for embeddings).
    pub fn for_embeddings() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 64,
            strategy: ChunkStrategy::Sentence,
            semantic_split: true,
            min_tokens: 32,
        }
    }

    /// Create config for large chunks (suitable for context windows).
    pub fn for_context() -> Self {
        Self {
            max_tokens: 8192,
            overlap_tokens: 256,
            strategy: ChunkStrategy::Paragraph,
            semantic_split: true,
            min_tokens: 128,
        }
    }

    /// Create config for code blocks.
    pub fn for_code() -> Self {
        Self {
            max_tokens: 4096,
            overlap_tokens: 0,
            strategy: ChunkStrategy::CodeBlock,
            semantic_split: true,
            min_tokens: 16,
        }
    }

    /// Set maximum tokens per chunk.
    #[inline]
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set overlap tokens between chunks.
    #[inline]
    pub fn overlap(mut self, tokens: usize) -> Self {
        self.overlap_tokens = tokens;
        self
    }

    /// Set splitting strategy.
    #[inline]
    pub fn strategy(mut self, strategy: ChunkStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable or disable semantic splitting.
    #[inline]
    pub fn semantic(mut self, enabled: bool) -> Self {
        self.semantic_split = enabled;
        self
    }
}

// ============================================================================
// Chunk Strategy
// ============================================================================

/// Strategy for splitting text into chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    /// Split on sentence boundaries (period, exclamation, question).
    #[default]
    Sentence,
    /// Split on paragraph boundaries (double newlines).
    Paragraph,
    /// Split on section headers (## or ###).
    Section,
    /// Split on code blocks (``` markers).
    CodeBlock,
    /// Fixed token count (no semantic awareness).
    Fixed,
}

impl ChunkStrategy {
    /// Get the primary delimiter for this strategy.
    pub fn delimiter(&self) -> &'static str {
        match self {
            Self::Sentence => ". ",
            Self::Paragraph => "\n\n",
            Self::Section => "\n## ",
            Self::CodeBlock => "```",
            Self::Fixed => "",
        }
    }

    /// Check if this strategy preserves semantic boundaries.
    #[inline]
    pub fn is_semantic(&self) -> bool {
        !matches!(self, Self::Fixed)
    }
}

// ============================================================================
// Section Type
// ============================================================================

/// Section types for structured output (code + document + custom).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionType {
    // Code-focused sections
    /// Import statements and dependencies.
    Imports,
    /// Main code block.
    CodeBlock,
    /// Test code.
    Tests,
    /// Code documentation (docstrings, comments).
    Documentation,
    /// Example usage code.
    Example,

    // Document-focused sections
    /// Introduction or overview.
    Introduction,
    /// Main body content.
    Body,
    /// Detailed explanation.
    Explanation,
    /// Summary or conclusion.
    Summary,
    /// References or citations.
    References,
    /// Appendix or supplementary content.
    Appendix,

    // Custom user-defined section
    /// Custom section type with user-defined name.
    Custom(Cow<'static, str>),
}

impl SectionType {
    /// Create a custom section type.
    pub fn custom(name: impl Into<Cow<'static, str>>) -> Self {
        Self::Custom(name.into())
    }

    /// Check if this is a code-related section.
    #[inline]
    pub fn is_code_section(&self) -> bool {
        matches!(
            self,
            Self::Imports | Self::CodeBlock | Self::Tests | Self::Example
        )
    }

    /// Check if this is a documentation section.
    #[inline]
    pub fn is_doc_section(&self) -> bool {
        matches!(
            self,
            Self::Documentation
                | Self::Introduction
                | Self::Body
                | Self::Explanation
                | Self::Summary
                | Self::References
                | Self::Appendix
        )
    }

    /// Check if this is a custom section.
    #[inline]
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    /// Get the section name as a string.
    pub fn name(&self) -> &str {
        match self {
            Self::Imports => "imports",
            Self::CodeBlock => "code",
            Self::Tests => "tests",
            Self::Documentation => "documentation",
            Self::Example => "example",
            Self::Introduction => "introduction",
            Self::Body => "body",
            Self::Explanation => "explanation",
            Self::Summary => "summary",
            Self::References => "references",
            Self::Appendix => "appendix",
            Self::Custom(name) => name.as_ref(),
        }
    }

    /// Try to detect section type from header text.
    pub fn from_header(header: &str) -> Option<Self> {
        let lower = header.to_lowercase();
        let trimmed = lower.trim_start_matches('#').trim();

        match trimmed {
            s if s.starts_with("import") || s.starts_with("depend") => Some(Self::Imports),
            s if s.starts_with("code") || s.starts_with("implementation") => Some(Self::CodeBlock),
            s if s.starts_with("test") => Some(Self::Tests),
            s if s.starts_with("doc") || s.starts_with("api") => Some(Self::Documentation),
            s if s.starts_with("example") || s.starts_with("usage") => Some(Self::Example),
            s if s.starts_with("intro") || s.starts_with("overview") => Some(Self::Introduction),
            s if s.starts_with("body") || s.starts_with("main") => Some(Self::Body),
            s if s.starts_with("explain") || s.starts_with("detail") => Some(Self::Explanation),
            s if s.starts_with("summar") || s.starts_with("conclusion") => Some(Self::Summary),
            s if s.starts_with("refer") || s.starts_with("citation") => Some(Self::References),
            s if s.starts_with("append") || s.starts_with("supplement") => Some(Self::Appendix),
            _ => None,
        }
    }
}

impl Default for SectionType {
    fn default() -> Self {
        Self::Body
    }
}

// ============================================================================
// Text Chunk
// ============================================================================

/// A chunk of text with metadata.
#[derive(Debug, Clone)]
pub struct TextChunk<'a> {
    /// The chunk content (zero-copy reference when possible).
    pub content: Cow<'a, str>,
    /// Token count for this chunk.
    pub token_count: usize,
    /// Chunk index (0-based).
    pub index: u16,
    /// Total chunks in sequence.
    pub total: u16,
    /// Byte offset in original text.
    pub byte_offset: usize,
    /// Section type if detected.
    pub section_type: Option<SectionType>,
    /// Dependencies on other chunks (for context continuity).
    pub depends_on: SmallVec<[u16; 4]>,
}

impl<'a> TextChunk<'a> {
    /// Create a new text chunk.
    pub fn new(
        content: impl Into<Cow<'a, str>>,
        token_count: usize,
        index: u16,
        total: u16,
    ) -> Self {
        Self {
            content: content.into(),
            token_count,
            index,
            total,
            byte_offset: 0,
            section_type: None,
            depends_on: SmallVec::new(),
        }
    }

    /// Set the byte offset.
    #[inline]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.byte_offset = offset;
        self
    }

    /// Set the section type.
    #[inline]
    pub fn with_section(mut self, section: SectionType) -> Self {
        self.section_type = Some(section);
        self
    }

    /// Add a dependency on another chunk.
    #[inline]
    pub fn depends_on_chunk(mut self, chunk_index: u16) -> Self {
        if !self.depends_on.contains(&chunk_index) {
            self.depends_on.push(chunk_index);
        }
        self
    }

    /// Check if this is the first chunk.
    #[inline]
    pub fn is_first(&self) -> bool {
        self.index == 0
    }

    /// Check if this is the last chunk.
    #[inline]
    pub fn is_last(&self) -> bool {
        self.index + 1 == self.total
    }

    /// Check if this chunk has dependencies.
    #[inline]
    pub fn has_dependencies(&self) -> bool {
        !self.depends_on.is_empty()
    }

    /// Get the content as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.content
    }

    /// Convert to owned chunk (useful for storage).
    pub fn into_owned(self) -> TextChunk<'static> {
        TextChunk {
            content: Cow::Owned(self.content.into_owned()),
            token_count: self.token_count,
            index: self.index,
            total: self.total,
            byte_offset: self.byte_offset,
            section_type: self.section_type,
            depends_on: self.depends_on,
        }
    }
}

// ============================================================================
// Text Chunker
// ============================================================================

/// Token-aware text chunker.
///
/// Splits text into chunks based on token limits while respecting
/// semantic boundaries when possible.
pub struct TextChunker<T: Tokenizer> {
    tokenizer: T,
    config: ChunkConfig,
}

impl<T: Tokenizer> TextChunker<T> {
    /// Create a new chunker with default configuration.
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            config: ChunkConfig::default(),
        }
    }

    /// Create a chunker with custom configuration.
    pub fn with_config(tokenizer: T, config: ChunkConfig) -> Self {
        Self { tokenizer, config }
    }

    /// Get the tokenizer reference.
    #[inline]
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get the configuration reference.
    #[inline]
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Count tokens in text.
    #[inline]
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.count_tokens(text)
    }

    /// Estimate number of chunks without materializing them.
    pub fn estimate_chunks(&self, text: &str) -> usize {
        let total_tokens = self.count_tokens(text);
        if total_tokens == 0 {
            return 0;
        }

        let effective_chunk_size = self
            .config
            .max_tokens
            .saturating_sub(self.config.overlap_tokens);
        if effective_chunk_size == 0 {
            return 1;
        }

        (total_tokens + effective_chunk_size - 1) / effective_chunk_size
    }

    /// Chunk text into token-limited pieces.
    pub fn chunk<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        if text.is_empty() {
            return Vec::new();
        }

        let total_tokens = self.count_tokens(text);
        if total_tokens <= self.config.max_tokens {
            // Single chunk - no splitting needed
            return vec![TextChunk::new(Cow::Borrowed(text), total_tokens, 0, 1)];
        }

        match self.config.strategy {
            ChunkStrategy::Sentence => self.chunk_by_sentences(text),
            ChunkStrategy::Paragraph => self.chunk_by_paragraphs(text),
            ChunkStrategy::Section => self.chunk_by_sections(text),
            ChunkStrategy::CodeBlock => self.chunk_by_code_blocks(text),
            ChunkStrategy::Fixed => self.chunk_fixed(text),
        }
    }

    /// Chunk with section detection for templates.
    pub fn chunk_with_sections<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        let mut chunks = self.chunk_by_sections(text);

        // Detect and assign section types
        for chunk in &mut chunks {
            if chunk.section_type.is_none() {
                chunk.section_type = self.detect_section_type(&chunk.content);
            }
        }

        // Set up dependencies based on section types
        self.resolve_dependencies(&mut chunks);

        chunks
    }

    // ========================================================================
    // Private chunking methods
    // ========================================================================

    fn chunk_by_sentences<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        self.chunk_by_delimiter(text, &[". ", "! ", "? ", ".\n", "!\n", "?\n"])
    }

    fn chunk_by_paragraphs<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        self.chunk_by_delimiter(text, &["\n\n", "\r\n\r\n"])
    }

    fn chunk_by_sections<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        // Split on markdown headers
        let mut sections = Vec::new();
        let mut current_start = 0;
        let mut current_section_type = None;

        for (idx, line) in text.lines().enumerate() {
            if line.starts_with("## ") || line.starts_with("### ") {
                if current_start < text.len() {
                    let end = text[current_start..]
                        .find(line)
                        .map(|i| current_start + i)
                        .unwrap_or(text.len());

                    if end > current_start {
                        let section_text = &text[current_start..end];
                        if !section_text.trim().is_empty() {
                            sections.push((section_text, current_section_type.take()));
                        }
                    }
                }

                current_section_type = SectionType::from_header(line);
                current_start = text[current_start..]
                    .find(line)
                    .map(|i| current_start + i)
                    .unwrap_or(current_start);
            }
        }

        // Don't forget the last section
        if current_start < text.len() {
            let section_text = &text[current_start..];
            if !section_text.trim().is_empty() {
                sections.push((section_text, current_section_type));
            }
        }

        if sections.is_empty() {
            // No sections found, fall back to paragraph splitting
            return self.chunk_by_paragraphs(text);
        }

        // Now chunk each section if needed
        let mut result = Vec::new();
        for (section_text, section_type) in sections {
            let section_chunks = self.chunk_section(section_text, section_type);
            result.extend(section_chunks);
        }

        // Update total count and indices
        let total = result.len() as u16;
        for (idx, chunk) in result.iter_mut().enumerate() {
            chunk.index = idx as u16;
            chunk.total = total;
        }

        result
    }

    fn chunk_section<'a>(
        &self,
        text: &'a str,
        section_type: Option<SectionType>,
    ) -> Vec<TextChunk<'a>> {
        let tokens = self.count_tokens(text);
        if tokens <= self.config.max_tokens {
            let mut chunk = TextChunk::new(Cow::Borrowed(text), tokens, 0, 1);
            chunk.section_type = section_type;
            return vec![chunk];
        }

        // Section is too large, sub-chunk it
        let mut chunks = self.chunk_by_sentences(text);
        for chunk in &mut chunks {
            chunk.section_type = section_type.clone();
        }
        chunks
    }

    fn chunk_by_code_blocks<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        let mut chunks = Vec::new();
        let mut current_start = 0;
        let mut in_code_block = false;
        let mut code_block_start = 0;

        for (idx, line) in text.lines().enumerate() {
            let line_start = text[current_start..]
                .find(line)
                .map(|i| current_start + i)
                .unwrap_or(current_start);

            if line.starts_with("```") {
                if !in_code_block {
                    // Starting a code block - save any text before it
                    if line_start > current_start {
                        let before_text = &text[current_start..line_start];
                        if !before_text.trim().is_empty() {
                            let tokens = self.count_tokens(before_text);
                            let chunk = TextChunk::new(Cow::Borrowed(before_text), tokens, 0, 0)
                                .with_offset(current_start);
                            chunks.push(chunk);
                        }
                    }
                    code_block_start = line_start;
                    in_code_block = true;
                } else {
                    // Ending a code block
                    let line_end = line_start + line.len() + 1; // +1 for newline
                    let code_text = &text[code_block_start..line_end.min(text.len())];
                    let tokens = self.count_tokens(code_text);
                    let chunk = TextChunk::new(Cow::Borrowed(code_text), tokens, 0, 0)
                        .with_offset(code_block_start)
                        .with_section(SectionType::CodeBlock);
                    chunks.push(chunk);
                    current_start = line_end.min(text.len());
                    in_code_block = false;
                }
            }
        }

        // Handle remaining text
        if current_start < text.len() {
            let remaining = &text[current_start..];
            if !remaining.trim().is_empty() {
                let tokens = self.count_tokens(remaining);
                let chunk = TextChunk::new(Cow::Borrowed(remaining), tokens, 0, 0)
                    .with_offset(current_start);
                chunks.push(chunk);
            }
        }

        // Update indices
        let total = chunks.len() as u16;
        for (idx, chunk) in chunks.iter_mut().enumerate() {
            chunk.index = idx as u16;
            chunk.total = total;
        }

        // Now check if any chunks exceed max_tokens and sub-chunk them
        let mut final_chunks = Vec::new();
        for chunk in chunks {
            if chunk.token_count > self.config.max_tokens {
                let sub_chunks = self.chunk_fixed(&chunk.content);
                for mut sub in sub_chunks {
                    sub.section_type = chunk.section_type.clone();
                    final_chunks.push(sub.into_owned());
                }
            } else {
                final_chunks.push(chunk.into_owned());
            }
        }

        // Re-update indices
        let total = final_chunks.len() as u16;
        for (idx, chunk) in final_chunks.iter_mut().enumerate() {
            chunk.index = idx as u16;
            chunk.total = total;
        }

        final_chunks
    }

    fn chunk_fixed<'a>(&self, text: &'a str) -> Vec<TextChunk<'a>> {
        let total_tokens = self.count_tokens(text);
        if total_tokens <= self.config.max_tokens {
            return vec![TextChunk::new(Cow::Borrowed(text), total_tokens, 0, 1)];
        }

        // Encode to get token boundaries
        let tokens = self.tokenizer.encode(text);
        let effective_size = self
            .config
            .max_tokens
            .saturating_sub(self.config.overlap_tokens);
        if effective_size == 0 {
            return vec![TextChunk::new(Cow::Borrowed(text), total_tokens, 0, 1)];
        }

        let mut chunks = Vec::new();
        let mut start_token = 0;

        while start_token < tokens.len() {
            let end_token = (start_token + self.config.max_tokens).min(tokens.len());
            let chunk_tokens: Vec<u32> = tokens[start_token..end_token].to_vec();
            let chunk_text = self.tokenizer.decode(&chunk_tokens);

            chunks.push(TextChunk::new(
                Cow::Owned(chunk_text),
                chunk_tokens.len(),
                0,
                0,
            ));

            start_token += effective_size;
        }

        let total = chunks.len() as u16;
        for (idx, chunk) in chunks.iter_mut().enumerate() {
            chunk.index = idx as u16;
            chunk.total = total;
        }

        chunks
    }

    fn chunk_by_delimiter<'a>(&self, text: &'a str, delimiters: &[&str]) -> Vec<TextChunk<'a>> {
        // Find all split points
        let mut split_points = vec![0];
        for delim in delimiters {
            let mut search_start = 0;
            while let Some(pos) = text[search_start..].find(delim) {
                let absolute_pos = search_start + pos + delim.len();
                if absolute_pos < text.len() {
                    split_points.push(absolute_pos);
                }
                search_start = absolute_pos;
            }
        }
        split_points.sort_unstable();
        split_points.dedup();
        split_points.push(text.len());

        // Group split points into chunks that fit max_tokens
        let mut chunks = Vec::new();
        let mut chunk_start = 0;
        let mut chunk_end = 0;
        let mut current_tokens = 0;

        for &point in &split_points[1..] {
            let segment = &text[chunk_end..point];
            let segment_tokens = self.count_tokens(segment);

            if current_tokens + segment_tokens > self.config.max_tokens && current_tokens > 0 {
                // Current chunk is full, save it
                let chunk_text = &text[chunk_start..chunk_end];
                if !chunk_text.trim().is_empty() {
                    chunks.push(
                        TextChunk::new(Cow::Borrowed(chunk_text), current_tokens, 0, 0)
                            .with_offset(chunk_start),
                    );
                }

                // Start new chunk with overlap
                if self.config.overlap_tokens > 0 && !chunks.is_empty() {
                    // Find overlap start point
                    let overlap_start = self.find_overlap_start(&text[..chunk_end], chunk_start);
                    chunk_start = overlap_start;
                    current_tokens = self.count_tokens(&text[chunk_start..chunk_end]);
                } else {
                    chunk_start = chunk_end;
                    current_tokens = 0;
                }
            }

            current_tokens += segment_tokens;
            chunk_end = point;
        }

        // Don't forget the last chunk
        if chunk_start < text.len() {
            let chunk_text = &text[chunk_start..];
            if !chunk_text.trim().is_empty() {
                chunks.push(
                    TextChunk::new(
                        Cow::Borrowed(chunk_text),
                        self.count_tokens(chunk_text),
                        0,
                        0,
                    )
                    .with_offset(chunk_start),
                );
            }
        }

        // Update indices
        let total = chunks.len() as u16;
        for (idx, chunk) in chunks.iter_mut().enumerate() {
            chunk.index = idx as u16;
            chunk.total = total;
        }

        chunks
    }

    fn find_overlap_start(&self, text: &str, min_start: usize) -> usize {
        // Find a good overlap start point that's roughly overlap_tokens back
        let target_tokens = self.config.overlap_tokens;
        let mut best_pos = min_start;
        let mut tokens_from_end = 0;

        // Walk backwards from end to find overlap point
        for (idx, c) in text.char_indices().rev() {
            if idx < min_start {
                break;
            }
            tokens_from_end = self.count_tokens(&text[idx..]);
            if tokens_from_end >= target_tokens {
                best_pos = idx;
                break;
            }
        }

        best_pos
    }

    fn detect_section_type(&self, content: &str) -> Option<SectionType> {
        let trimmed = content.trim();

        // Check for code block
        if trimmed.starts_with("```") || trimmed.contains("\n```") {
            return Some(SectionType::CodeBlock);
        }

        // Check for import statements
        if trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("#include")
        {
            return Some(SectionType::Imports);
        }

        // Check for test patterns
        if trimmed.contains("#[test]")
            || trimmed.contains("fn test_")
            || trimmed.contains("def test_")
            || trimmed.contains("@Test")
            || trimmed.contains("describe(")
            || trimmed.contains("it(")
        {
            return Some(SectionType::Tests);
        }

        // Check for header-based detection
        let first_line = trimmed.lines().next().unwrap_or("");
        if first_line.starts_with('#') {
            return SectionType::from_header(first_line);
        }

        None
    }

    fn resolve_dependencies(&self, chunks: &mut [TextChunk<'_>]) {
        // Build index of chunks by section type
        let mut imports_idx: Option<u16> = None;
        let mut code_idx: Option<u16> = None;
        let mut docs_idx: Option<u16> = None;

        for chunk in chunks.iter() {
            match &chunk.section_type {
                Some(SectionType::Imports) => imports_idx = Some(chunk.index),
                Some(SectionType::CodeBlock) => code_idx = Some(chunk.index),
                Some(SectionType::Documentation) => docs_idx = Some(chunk.index),
                _ => {}
            }
        }

        // Apply dependency rules
        for chunk in chunks.iter_mut() {
            match &chunk.section_type {
                // CodeBlock depends on Imports
                Some(SectionType::CodeBlock) => {
                    if let Some(idx) = imports_idx {
                        if idx != chunk.index && !chunk.depends_on.contains(&idx) {
                            chunk.depends_on.push(idx);
                        }
                    }
                }
                // Tests depend on CodeBlock
                Some(SectionType::Tests) => {
                    if let Some(idx) = code_idx {
                        if idx != chunk.index && !chunk.depends_on.contains(&idx) {
                            chunk.depends_on.push(idx);
                        }
                    }
                }
                // Example depends on CodeBlock and Documentation
                Some(SectionType::Example) => {
                    if let Some(idx) = code_idx {
                        if idx != chunk.index && !chunk.depends_on.contains(&idx) {
                            chunk.depends_on.push(idx);
                        }
                    }
                    if let Some(idx) = docs_idx {
                        if idx != chunk.index && !chunk.depends_on.contains(&idx) {
                            chunk.depends_on.push(idx);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

impl<T: Tokenizer + Clone> Clone for TextChunker<T> {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            config: self.config,
        }
    }
}

impl<T: Tokenizer + std::fmt::Debug> std::fmt::Debug for TextChunker<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextChunker")
            .field("tokenizer", &self.tokenizer)
            .field("config", &self.config)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::tokenize::SimpleTokenizer;

    fn test_chunker() -> TextChunker<SimpleTokenizer> {
        TextChunker::new(SimpleTokenizer::new())
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.overlap_tokens, 128);
        assert_eq!(config.strategy, ChunkStrategy::Sentence);
        assert!(config.semantic_split);
    }

    #[test]
    fn test_chunk_config_presets() {
        let embed = ChunkConfig::for_embeddings();
        assert_eq!(embed.max_tokens, 512);

        let context = ChunkConfig::for_context();
        assert_eq!(context.max_tokens, 8192);

        let code = ChunkConfig::for_code();
        assert_eq!(code.overlap_tokens, 0);
    }

    #[test]
    fn test_chunk_strategy_delimiter() {
        assert_eq!(ChunkStrategy::Sentence.delimiter(), ". ");
        assert_eq!(ChunkStrategy::Paragraph.delimiter(), "\n\n");
        assert_eq!(ChunkStrategy::Section.delimiter(), "\n## ");
        assert!(ChunkStrategy::Sentence.is_semantic());
        assert!(!ChunkStrategy::Fixed.is_semantic());
    }

    #[test]
    fn test_section_type_detection() {
        assert_eq!(
            SectionType::from_header("## Introduction"),
            Some(SectionType::Introduction)
        );
        assert_eq!(
            SectionType::from_header("### Tests"),
            Some(SectionType::Tests)
        );
        assert_eq!(
            SectionType::from_header("## Code Implementation"),
            Some(SectionType::CodeBlock)
        );
        assert_eq!(SectionType::from_header("## Random Heading"), None);
    }

    #[test]
    fn test_section_type_classification() {
        assert!(SectionType::CodeBlock.is_code_section());
        assert!(SectionType::Tests.is_code_section());
        assert!(SectionType::Imports.is_code_section());

        assert!(SectionType::Introduction.is_doc_section());
        assert!(SectionType::Summary.is_doc_section());
        assert!(SectionType::Documentation.is_doc_section());

        let custom = SectionType::custom("MySection");
        assert!(custom.is_custom());
        assert_eq!(custom.name(), "MySection");
    }

    #[test]
    fn test_chunker_empty_text() {
        let chunker = test_chunker();
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunker_single_chunk() {
        let chunker = TextChunker::with_config(
            SimpleTokenizer::new(),
            ChunkConfig::default().max_tokens(100),
        );
        let text = "Hello world this is a test";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].total, 1);
    }

    #[test]
    fn test_chunker_estimate() {
        let chunker = TextChunker::with_config(
            SimpleTokenizer::new(),
            ChunkConfig::default().max_tokens(10).overlap(2),
        );
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let estimate = chunker.estimate_chunks(text);
        assert!(estimate > 0);
    }

    #[test]
    fn test_text_chunk_methods() {
        let chunk = TextChunk::new("test content", 2, 0, 3)
            .with_offset(100)
            .with_section(SectionType::CodeBlock)
            .depends_on_chunk(1);

        assert!(chunk.is_first());
        assert!(!chunk.is_last());
        assert!(chunk.has_dependencies());
        assert_eq!(chunk.as_str(), "test content");
        assert_eq!(chunk.byte_offset, 100);
        assert_eq!(chunk.section_type, Some(SectionType::CodeBlock));
        assert!(chunk.depends_on.contains(&1));
    }

    #[test]
    fn test_text_chunk_into_owned() {
        let text = "borrowed text";
        let chunk = TextChunk::new(Cow::Borrowed(text), 2, 0, 1);
        let owned = chunk.into_owned();
        assert!(matches!(owned.content, Cow::Owned(_)));
    }

    #[test]
    fn test_chunker_with_sections() {
        let chunker = test_chunker();
        let text = r#"## Introduction
This is the intro.

## Code
```rust
fn main() {}
```

## Tests
```rust
#[test]
fn test_main() {}
```
"#;
        let chunks = chunker.chunk_with_sections(text);
        assert!(!chunks.is_empty());

        // Check that sections were detected
        let section_types: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.section_type.clone())
            .collect();
        assert!(!section_types.is_empty());
    }

    #[cfg(feature = "tiktoken")]
    mod tiktoken_tests {
        use super::*;
        use crate::recursive::tokenize::TiktokenTokenizer;

        #[test]
        fn test_tiktoken_chunker() {
            let chunker = TextChunker::with_config(
                TiktokenTokenizer::claude(),
                ChunkConfig::default().max_tokens(100),
            );
            let text = "This is a test. This is another sentence. And one more.";
            let chunks = chunker.chunk(text);
            assert!(!chunks.is_empty());
            for chunk in &chunks {
                assert!(chunk.token_count <= 100);
            }
        }

        #[test]
        fn test_tiktoken_fixed_chunking() {
            let chunker = TextChunker::with_config(
                TiktokenTokenizer::claude(),
                ChunkConfig::default()
                    .max_tokens(50)
                    .overlap(10)
                    .strategy(ChunkStrategy::Fixed),
            );
            let text = "word ".repeat(200);
            let chunks = chunker.chunk(&text);
            assert!(chunks.len() > 1);
        }
    }
}
