// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Chained template execution for converged prompts.
//!
//! When a template has converged (score >= threshold), this module enables
//! chaining the output into typed sections for structured processing.
//!
//! # Features
//!
//! - Section-based output chunking
//! - Dependency tracking between chunks
//! - Context propagation for continuity
//! - Parallel chunk generation (with dependency resolution)
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::{chain, Template, TiktokenTokenizer, SectionType, ChunkConfig};
//!
//! let tokenizer = TiktokenTokenizer::claude();
//! let output = chain(&template, "input text", &tokenizer)
//!     .sections(&[SectionType::Introduction, SectionType::CodeBlock, SectionType::Tests])
//!     .chunk_config(ChunkConfig { max_tokens: 2048, ..Default::default() })
//!     .execute(&lm)
//!     .await?;
//!
//! for chunk in output.chunks {
//!     println!("Section {:?}: {} tokens", chunk.section_type, chunk.token_count);
//! }
//! ```

use std::borrow::Cow;

use smallvec::SmallVec;

use crate::error::Result;
use crate::predict::LMClient;
use crate::str_view::StrView;

use super::chunk::{ChunkConfig, SectionType, TextChunk, TextChunker};
use super::template::Template;
use super::tokenize::Tokenizer;

// ============================================================================
// Chained Output
// ============================================================================

/// Output from a chained template execution.
#[derive(Debug, Clone)]
pub struct ChainedOutput<'a> {
    /// All chunks in order.
    pub chunks: SmallVec<[OutputChunk<'a>; 8]>,
    /// Combined full text.
    pub full_text: String,
    /// Total token count.
    pub total_tokens: usize,
    /// Convergence score of the template.
    pub convergence_score: f64,
    /// Number of iterations to converge.
    pub iterations: u32,
}

impl<'a> ChainedOutput<'a> {
    /// Create a new chained output.
    pub fn new(full_text: String, total_tokens: usize) -> Self {
        Self {
            chunks: SmallVec::new(),
            full_text,
            total_tokens,
            convergence_score: 0.0,
            iterations: 0,
        }
    }

    /// Set convergence info.
    #[inline]
    pub fn with_convergence(mut self, score: f64, iterations: u32) -> Self {
        self.convergence_score = score;
        self.iterations = iterations;
        self
    }

    /// Get chunks by section type.
    pub fn chunks_by_section(&self, section: &SectionType) -> Vec<&OutputChunk<'a>> {
        self.chunks
            .iter()
            .filter(|c| c.section_type == *section)
            .collect()
    }

    /// Get the first chunk of a given section type.
    pub fn first_section(&self, section: &SectionType) -> Option<&OutputChunk<'a>> {
        self.chunks.iter().find(|c| c.section_type == *section)
    }

    /// Get all code sections.
    pub fn code_chunks(&self) -> Vec<&OutputChunk<'a>> {
        self.chunks
            .iter()
            .filter(|c| c.section_type.is_code_section())
            .collect()
    }

    /// Get all documentation sections.
    pub fn doc_chunks(&self) -> Vec<&OutputChunk<'a>> {
        self.chunks
            .iter()
            .filter(|c| c.section_type.is_doc_section())
            .collect()
    }

    /// Get total count of chunks.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the output converged successfully.
    #[inline]
    pub fn converged(&self) -> bool {
        self.convergence_score >= 0.8
    }

    /// Convert to owned (static lifetime).
    pub fn into_owned(self) -> ChainedOutput<'static> {
        ChainedOutput {
            chunks: self.chunks.into_iter().map(|c| c.into_owned()).collect(),
            full_text: self.full_text,
            total_tokens: self.total_tokens,
            convergence_score: self.convergence_score,
            iterations: self.iterations,
        }
    }
}

/// A single output chunk with metadata.
#[derive(Debug, Clone)]
pub struct OutputChunk<'a> {
    /// Chunk content.
    pub content: Cow<'a, str>,
    /// Section type.
    pub section_type: SectionType,
    /// Token count.
    pub token_count: usize,
    /// Chunk index.
    pub index: u16,
    /// Metadata.
    pub metadata: ChunkMetadata,
}

impl<'a> OutputChunk<'a> {
    /// Create a new output chunk.
    pub fn new(
        content: impl Into<Cow<'a, str>>,
        section_type: SectionType,
        token_count: usize,
        index: u16,
    ) -> Self {
        Self {
            content: content.into(),
            section_type,
            token_count,
            index,
            metadata: ChunkMetadata::default(),
        }
    }

    /// Set metadata.
    #[inline]
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get content as string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.content
    }

    /// Convert to owned (static lifetime).
    pub fn into_owned(self) -> OutputChunk<'static> {
        OutputChunk {
            content: Cow::Owned(self.content.into_owned()),
            section_type: self.section_type,
            token_count: self.token_count,
            index: self.index,
            metadata: self.metadata,
        }
    }
}

impl<'a> From<TextChunk<'a>> for OutputChunk<'a> {
    fn from(chunk: TextChunk<'a>) -> Self {
        let mut metadata = ChunkMetadata::default();
        for dep in chunk.depends_on {
            metadata.add_dependency(dep);
        }

        Self {
            content: chunk.content,
            section_type: chunk.section_type.unwrap_or_default(),
            token_count: chunk.token_count,
            index: chunk.index,
            metadata,
        }
    }
}

// ============================================================================
// Chunk Metadata
// ============================================================================

/// Metadata for an output chunk.
#[derive(Debug, Clone, Default)]
pub struct ChunkMetadata {
    /// Optional title/header for section.
    pub title: Option<String>,
    /// Quality score for this chunk (0.0 - 1.0).
    pub score: f64,
    /// Dependencies on other chunks (indices).
    pub depends_on: SmallVec<[u16; 4]>,
    /// Chunks that depend on this chunk (reverse deps).
    pub dependents: SmallVec<[u16; 4]>,
    /// Context from dependencies (for continuity).
    pub inherited_context: Option<String>,
}

impl ChunkMetadata {
    /// Add a dependency on another chunk.
    #[inline]
    pub fn add_dependency(&mut self, chunk_index: u16) {
        if !self.depends_on.contains(&chunk_index) {
            self.depends_on.push(chunk_index);
        }
    }

    /// Add a dependent chunk.
    #[inline]
    pub fn add_dependent(&mut self, chunk_index: u16) {
        if !self.dependents.contains(&chunk_index) {
            self.dependents.push(chunk_index);
        }
    }

    /// Check if this chunk has dependencies.
    #[inline]
    pub fn has_dependencies(&self) -> bool {
        !self.depends_on.is_empty()
    }

    /// Check if this chunk is a dependency for others.
    #[inline]
    pub fn has_dependents(&self) -> bool {
        !self.dependents.is_empty()
    }

    /// Set the title.
    #[inline]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the score.
    #[inline]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Set inherited context.
    #[inline]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.inherited_context = Some(context.into());
        self
    }
}

// ============================================================================
// Dependency Rules
// ============================================================================

/// Dependency rules for automatic dependency detection.
#[derive(Debug, Clone)]
pub struct DependencyRules {
    /// CodeBlock depends on Imports.
    pub code_depends_on_imports: bool,
    /// Tests depend on CodeBlock.
    pub tests_depend_on_code: bool,
    /// Example depends on CodeBlock + Documentation.
    pub example_depends_on_code_and_docs: bool,
    /// Custom rules: (dependent_type, dependency_type).
    pub custom_rules: Vec<(SectionType, SectionType)>,
}

impl Default for DependencyRules {
    fn default() -> Self {
        Self {
            code_depends_on_imports: true,
            tests_depend_on_code: true,
            example_depends_on_code_and_docs: true,
            custom_rules: Vec::new(),
        }
    }
}

impl DependencyRules {
    /// Create rules with no automatic dependencies.
    pub fn none() -> Self {
        Self {
            code_depends_on_imports: false,
            tests_depend_on_code: false,
            example_depends_on_code_and_docs: false,
            custom_rules: Vec::new(),
        }
    }

    /// Add a custom dependency rule.
    pub fn add_rule(mut self, dependent: SectionType, dependency: SectionType) -> Self {
        self.custom_rules.push((dependent, dependency));
        self
    }

    /// Disable code depends on imports.
    #[inline]
    pub fn no_code_imports_dep(mut self) -> Self {
        self.code_depends_on_imports = false;
        self
    }

    /// Disable tests depends on code.
    #[inline]
    pub fn no_tests_code_dep(mut self) -> Self {
        self.tests_depend_on_code = false;
        self
    }
}

// ============================================================================
// Chain Builder
// ============================================================================

/// Builder for chained template execution.
pub struct ChainBuilder<'a, T: Tokenizer> {
    template: &'a Template<'a>,
    chunker: TextChunker<T>,
    section_types: Vec<SectionType>,
    dependency_rules: DependencyRules,
    parallel: bool,
    propagate_context: bool,
    convergence_score: f64,
    iterations: u32,
}

impl<'a, T: Tokenizer> ChainBuilder<'a, T> {
    /// Create a new chain builder.
    pub fn new(template: &'a Template<'a>, tokenizer: T) -> Self {
        Self {
            template,
            chunker: TextChunker::new(tokenizer),
            section_types: Vec::new(),
            dependency_rules: DependencyRules::default(),
            parallel: false,
            propagate_context: false,
            convergence_score: 0.0,
            iterations: 0,
        }
    }

    /// Define expected section types in order.
    pub fn sections(mut self, types: &[SectionType]) -> Self {
        self.section_types = types.to_vec();
        self
    }

    /// Set chunk configuration.
    pub fn chunk_config(mut self, config: ChunkConfig) -> Self
    where
        T: Clone,
    {
        self.chunker = TextChunker::with_config(self.chunker.tokenizer().clone(), config);
        self
    }

    /// Set dependency rules for automatic detection.
    pub fn dependency_rules(mut self, rules: DependencyRules) -> Self {
        self.dependency_rules = rules;
        self
    }

    /// Enable context propagation from dependencies.
    pub fn propagate_context(mut self) -> Self {
        self.propagate_context = true;
        self
    }

    /// Enable parallel chunk generation.
    /// Note: Dependencies are resolved before parallel execution.
    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Set convergence info from a previous run.
    pub fn with_convergence(mut self, score: f64, iterations: u32) -> Self {
        self.convergence_score = score;
        self.iterations = iterations;
        self
    }

    /// Execute chain and collect chunks.
    ///
    /// This takes the input text, generates output using the template,
    /// then chunks the output into sections.
    pub async fn execute<L: LMClient>(self, input: &str, lm: &L) -> Result<ChainedOutput<'a>>
    where
        T: Clone,
    {
        // Generate output using the template
        let prompt = self.template.render(input);
        let output = lm.generate(StrView::new(&prompt)).await?;
        let output_text = output.text()?.to_string();

        // Chunk the output
        let mut chunks = self.chunker.chunk_with_sections(&output_text);

        // Apply section type overrides if specified
        if !self.section_types.is_empty() {
            for (i, chunk) in chunks.iter_mut().enumerate() {
                if i < self.section_types.len() {
                    chunk.section_type = Some(self.section_types[i].clone());
                }
            }
        }

        // Build output chunks with dependency resolution
        let mut output_chunks: SmallVec<[OutputChunk<'_>; 8]> = SmallVec::new();
        for chunk in chunks {
            output_chunks.push(OutputChunk::from(chunk.into_owned()));
        }

        // Resolve dependencies
        self.resolve_dependencies(&mut output_chunks);

        // Propagate context if enabled
        if self.propagate_context {
            self.propagate_context_to_chunks(&mut output_chunks);
        }

        let total_tokens = self.chunker.count_tokens(&output_text);
        let mut result = ChainedOutput::new(output_text, total_tokens);
        result.chunks = output_chunks;
        result.convergence_score = self.convergence_score;
        result.iterations = self.iterations;

        Ok(result)
    }

    /// Execute with streaming callback.
    pub async fn execute_streaming<L, F>(
        self,
        input: &str,
        lm: &L,
        mut on_chunk: F,
    ) -> Result<ChainedOutput<'a>>
    where
        L: LMClient,
        T: Clone,
        F: FnMut(&OutputChunk<'_>),
    {
        let result = self.execute(input, lm).await?;

        // Call callback for each chunk
        for chunk in &result.chunks {
            on_chunk(chunk);
        }

        Ok(result)
    }

    /// Resolve dependencies and update chunk metadata.
    fn resolve_dependencies(&self, chunks: &mut SmallVec<[OutputChunk<'_>; 8]>) {
        // Build index of chunks by section type
        let mut section_indices: std::collections::HashMap<SectionType, Vec<u16>> =
            std::collections::HashMap::new();

        for chunk in chunks.iter() {
            section_indices
                .entry(chunk.section_type.clone())
                .or_default()
                .push(chunk.index);
        }

        // Apply dependency rules
        for chunk in chunks.iter_mut() {
            // CodeBlock depends on Imports
            if self.dependency_rules.code_depends_on_imports
                && chunk.section_type == SectionType::CodeBlock
            {
                if let Some(imports_indices) = section_indices.get(&SectionType::Imports) {
                    for &idx in imports_indices {
                        if idx != chunk.index {
                            chunk.metadata.add_dependency(idx);
                        }
                    }
                }
            }

            // Tests depend on CodeBlock
            if self.dependency_rules.tests_depend_on_code
                && chunk.section_type == SectionType::Tests
            {
                if let Some(code_indices) = section_indices.get(&SectionType::CodeBlock) {
                    for &idx in code_indices {
                        if idx != chunk.index {
                            chunk.metadata.add_dependency(idx);
                        }
                    }
                }
            }

            // Example depends on CodeBlock and Documentation
            if self.dependency_rules.example_depends_on_code_and_docs
                && chunk.section_type == SectionType::Example
            {
                if let Some(code_indices) = section_indices.get(&SectionType::CodeBlock) {
                    for &idx in code_indices {
                        if idx != chunk.index {
                            chunk.metadata.add_dependency(idx);
                        }
                    }
                }
                if let Some(doc_indices) = section_indices.get(&SectionType::Documentation) {
                    for &idx in doc_indices {
                        if idx != chunk.index {
                            chunk.metadata.add_dependency(idx);
                        }
                    }
                }
            }

            // Apply custom rules
            for (dependent, dependency) in &self.dependency_rules.custom_rules {
                if chunk.section_type == *dependent {
                    if let Some(dep_indices) = section_indices.get(dependency) {
                        for &idx in dep_indices {
                            if idx != chunk.index {
                                chunk.metadata.add_dependency(idx);
                            }
                        }
                    }
                }
            }
        }

        // Build reverse dependencies (dependents)
        let deps: Vec<(u16, SmallVec<[u16; 4]>)> = chunks
            .iter()
            .map(|c| (c.index, c.metadata.depends_on.clone()))
            .collect();

        for (chunk_idx, deps_list) in deps {
            for dep_idx in deps_list {
                if let Some(dep_chunk) = chunks.iter_mut().find(|c| c.index == dep_idx) {
                    dep_chunk.metadata.add_dependent(chunk_idx);
                }
            }
        }
    }

    /// Propagate context from dependencies to dependents.
    fn propagate_context_to_chunks(&self, chunks: &mut SmallVec<[OutputChunk<'_>; 8]>) {
        // Build a map of chunk content by index
        let content_map: std::collections::HashMap<u16, String> = chunks
            .iter()
            .map(|c| (c.index, c.content.to_string()))
            .collect();

        // For each chunk with dependencies, build inherited context
        for chunk in chunks.iter_mut() {
            if chunk.metadata.has_dependencies() {
                let mut context_parts = Vec::new();
                for &dep_idx in &chunk.metadata.depends_on {
                    if let Some(content) = content_map.get(&dep_idx) {
                        // Take first 500 chars as context summary
                        let summary: String = content.chars().take(500).collect();
                        context_parts.push(summary);
                    }
                }
                if !context_parts.is_empty() {
                    chunk.metadata.inherited_context = Some(context_parts.join("\n---\n"));
                }
            }
        }
    }

    /// Get execution order respecting dependencies (topological sort).
    pub fn resolve_dependency_order(&self, chunks: &[OutputChunk<'_>]) -> Vec<u16> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();

        fn visit(
            idx: u16,
            chunks: &[OutputChunk<'_>],
            visited: &mut std::collections::HashSet<u16>,
            temp_visited: &mut std::collections::HashSet<u16>,
            order: &mut Vec<u16>,
        ) {
            if visited.contains(&idx) {
                return;
            }
            if temp_visited.contains(&idx) {
                // Cycle detected - skip to avoid infinite loop
                return;
            }

            temp_visited.insert(idx);

            if let Some(chunk) = chunks.iter().find(|c| c.index == idx) {
                for &dep in &chunk.metadata.depends_on {
                    visit(dep, chunks, visited, temp_visited, order);
                }
            }

            temp_visited.remove(&idx);
            visited.insert(idx);
            order.push(idx);
        }

        for chunk in chunks {
            visit(
                chunk.index,
                chunks,
                &mut visited,
                &mut temp_visited,
                &mut order,
            );
        }

        order
    }
}

impl<'a, T: Tokenizer + Clone> Clone for ChainBuilder<'a, T> {
    fn clone(&self) -> Self {
        Self {
            template: self.template,
            chunker: self.chunker.clone(),
            section_types: self.section_types.clone(),
            dependency_rules: self.dependency_rules.clone(),
            parallel: self.parallel,
            propagate_context: self.propagate_context,
            convergence_score: self.convergence_score,
            iterations: self.iterations,
        }
    }
}

// ============================================================================
// Chain Function (Entry Point)
// ============================================================================

/// Chain a converged template into sectioned output.
///
/// When a template has converged (score >= threshold), this function
/// chains the output into typed sections for structured processing.
///
/// # Example
///
/// ```ignore
/// use kkachi::recursive::{chain, Template, TiktokenTokenizer, SectionType, ChunkConfig};
///
/// let template = Template::from_file("my_template.md")?;
/// let tokenizer = TiktokenTokenizer::claude();
///
/// let output = chain(&template, "input text", tokenizer)
///     .sections(&[SectionType::Introduction, SectionType::CodeBlock])
///     .chunk_config(ChunkConfig { max_tokens: 2048, ..Default::default() })
///     .execute(&lm)
///     .await?;
/// ```
pub fn chain<'a, T: Tokenizer>(
    template: &'a Template<'a>,
    _input: &str, // Reserved for future use in prompt building
    tokenizer: T,
) -> ChainBuilder<'a, T> {
    ChainBuilder::new(template, tokenizer)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::tokenize::SimpleTokenizer;

    fn test_tokenizer() -> SimpleTokenizer {
        SimpleTokenizer::new()
    }

    #[test]
    fn test_chained_output_new() {
        let output = ChainedOutput::new("test text".to_string(), 2);
        assert_eq!(output.full_text, "test text");
        assert_eq!(output.total_tokens, 2);
        assert_eq!(output.chunk_count(), 0);
        assert!(!output.converged());
    }

    #[test]
    fn test_chained_output_with_convergence() {
        let output = ChainedOutput::new("text".to_string(), 1).with_convergence(0.9, 5);
        assert!((output.convergence_score - 0.9).abs() < 0.001);
        assert_eq!(output.iterations, 5);
        assert!(output.converged());
    }

    #[test]
    fn test_output_chunk_creation() {
        let chunk = OutputChunk::new("content", SectionType::CodeBlock, 10, 0);
        assert_eq!(chunk.as_str(), "content");
        assert_eq!(chunk.section_type, SectionType::CodeBlock);
        assert_eq!(chunk.token_count, 10);
        assert_eq!(chunk.index, 0);
    }

    #[test]
    fn test_output_chunk_with_metadata() {
        let metadata = ChunkMetadata::default()
            .with_title("Test Title")
            .with_score(0.95)
            .with_context("Some context");

        assert_eq!(metadata.title, Some("Test Title".to_string()));
        assert!((metadata.score - 0.95).abs() < 0.001);
        assert!(metadata.inherited_context.is_some());
    }

    #[test]
    fn test_chunk_metadata_dependencies() {
        let mut metadata = ChunkMetadata::default();
        metadata.add_dependency(1);
        metadata.add_dependency(2);
        metadata.add_dependent(3);

        assert!(metadata.has_dependencies());
        assert!(metadata.has_dependents());
        assert_eq!(metadata.depends_on.len(), 2);
        assert_eq!(metadata.dependents.len(), 1);

        // Adding same dependency should not duplicate
        metadata.add_dependency(1);
        assert_eq!(metadata.depends_on.len(), 2);
    }

    #[test]
    fn test_dependency_rules_default() {
        let rules = DependencyRules::default();
        assert!(rules.code_depends_on_imports);
        assert!(rules.tests_depend_on_code);
        assert!(rules.example_depends_on_code_and_docs);
    }

    #[test]
    fn test_dependency_rules_none() {
        let rules = DependencyRules::none();
        assert!(!rules.code_depends_on_imports);
        assert!(!rules.tests_depend_on_code);
        assert!(!rules.example_depends_on_code_and_docs);
    }

    #[test]
    fn test_dependency_rules_custom() {
        let rules = DependencyRules::none().add_rule(SectionType::Summary, SectionType::Body);
        assert_eq!(rules.custom_rules.len(), 1);
    }

    #[test]
    fn test_chained_output_chunks_by_section() {
        let mut output = ChainedOutput::new("text".to_string(), 10);
        output
            .chunks
            .push(OutputChunk::new("code1", SectionType::CodeBlock, 5, 0));
        output
            .chunks
            .push(OutputChunk::new("intro", SectionType::Introduction, 3, 1));
        output
            .chunks
            .push(OutputChunk::new("code2", SectionType::CodeBlock, 2, 2));

        let code_chunks = output.chunks_by_section(&SectionType::CodeBlock);
        assert_eq!(code_chunks.len(), 2);

        let intro_chunk = output.first_section(&SectionType::Introduction);
        assert!(intro_chunk.is_some());
        assert_eq!(intro_chunk.unwrap().as_str(), "intro");
    }

    #[test]
    fn test_chained_output_code_and_doc_chunks() {
        let mut output = ChainedOutput::new("text".to_string(), 10);
        output
            .chunks
            .push(OutputChunk::new("code", SectionType::CodeBlock, 5, 0));
        output
            .chunks
            .push(OutputChunk::new("tests", SectionType::Tests, 3, 1));
        output
            .chunks
            .push(OutputChunk::new("intro", SectionType::Introduction, 2, 2));
        output
            .chunks
            .push(OutputChunk::new("docs", SectionType::Documentation, 2, 3));

        let code_chunks = output.code_chunks();
        assert_eq!(code_chunks.len(), 2); // CodeBlock and Tests

        let doc_chunks = output.doc_chunks();
        assert_eq!(doc_chunks.len(), 2); // Introduction and Documentation
    }

    #[test]
    fn test_chained_output_into_owned() {
        let output = ChainedOutput::new("test".to_string(), 1);
        let owned = output.into_owned();
        assert_eq!(owned.full_text, "test");
    }

    #[test]
    fn test_output_chunk_into_owned() {
        let chunk = OutputChunk::new("content", SectionType::Body, 2, 0);
        let owned = chunk.into_owned();
        assert!(matches!(owned.content, Cow::Owned(_)));
    }

    #[test]
    fn test_output_chunk_from_text_chunk() {
        use super::super::chunk::TextChunk;

        let text_chunk = TextChunk::new("test", 2, 0, 1)
            .with_section(SectionType::CodeBlock)
            .depends_on_chunk(1);

        let output_chunk = OutputChunk::from(text_chunk);
        assert_eq!(output_chunk.section_type, SectionType::CodeBlock);
        assert!(output_chunk.metadata.has_dependencies());
    }

    #[test]
    fn test_dependency_order_simple() {
        // Create chunks with dependencies: 0 -> 1 -> 2
        let chunks: SmallVec<[OutputChunk<'_>; 8]> = smallvec::smallvec![
            {
                let mut c = OutputChunk::new("c0", SectionType::Imports, 1, 0);
                c
            },
            {
                let mut c = OutputChunk::new("c1", SectionType::CodeBlock, 1, 1);
                c.metadata.add_dependency(0);
                c
            },
            {
                let mut c = OutputChunk::new("c2", SectionType::Tests, 1, 2);
                c.metadata.add_dependency(1);
                c
            },
        ];

        let template = Template::simple("test");
        let builder = ChainBuilder::new(&template, test_tokenizer());
        let order = builder.resolve_dependency_order(&chunks);

        // Order should be [0, 1, 2] (dependencies first)
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_dependency_order_with_cycle() {
        // Create chunks with a cycle: 0 -> 1 -> 0
        let chunks: SmallVec<[OutputChunk<'_>; 8]> = smallvec::smallvec![
            {
                let mut c = OutputChunk::new("c0", SectionType::Body, 1, 0);
                c.metadata.add_dependency(1);
                c
            },
            {
                let mut c = OutputChunk::new("c1", SectionType::Body, 1, 1);
                c.metadata.add_dependency(0);
                c
            },
        ];

        let template = Template::simple("test");
        let builder = ChainBuilder::new(&template, test_tokenizer());
        let order = builder.resolve_dependency_order(&chunks);

        // Should handle cycle gracefully
        assert_eq!(order.len(), 2);
    }
}
