// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Zero-copy RAG document templates.
//!
//! This module provides generic building blocks for RAG document formatting.
//! Users compose their own templates - the library provides NO vendor-specific presets.
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::{RagDocumentTemplate, TemplateSection};
//!
//! // User builds their own template for any language/format
//! let template = RagDocumentTemplate::new("my_code")
//!     .header("Task", 2)
//!     .text("question")
//!     .header("Solution", 2)
//!     .code("code", "python")  // User specifies language
//!     .header("Common Mistakes", 2)
//!     .list("errors");
//! ```

use std::fmt::Write;

use smallvec::SmallVec;

use crate::str_view::StrView;

// =============================================================================
// Template Section
// =============================================================================

/// A section in a RAG document template.
#[derive(Clone, Debug)]
pub enum TemplateSection<'a> {
    /// Markdown header (e.g., `## Task`).
    Header {
        /// Header label text.
        label: StrView<'a>,
        /// Header level (1-6).
        level: u8,
    },
    /// Plain text section.
    Text {
        /// Field name to retrieve content from result.
        label: StrView<'a>,
    },
    /// Fenced code block with language.
    Code {
        /// Field name to retrieve code from result.
        label: StrView<'a>,
        /// Language for syntax highlighting (e.g., "yaml", "python", "rust").
        language: StrView<'a>,
    },
    /// Bulleted list section.
    List {
        /// Field name to retrieve list items from result.
        label: StrView<'a>,
    },
}

// =============================================================================
// RAG Document Template
// =============================================================================

/// Zero-copy template for RAG documents.
///
/// Templates define the structure of documents stored in and retrieved from
/// the RAG vector store. The library provides NO vendor-specific presets -
/// users build templates for their specific use case.
///
/// # Zero-Copy Design
///
/// - Uses `StrView<'a>` for all string references
/// - Uses `SmallVec` for inline storage (no heap allocation for ≤8 sections)
/// - `format()` is the only method that allocates (produces owned output)
///
/// # Example
///
/// ```rust,ignore
/// // For Pulumi YAML (user-defined, not in library!)
/// let pulumi_template = RagDocumentTemplate::new("pulumi_yaml")
///     .header("Task", 2)
///     .text("question")
///     .header("Solution", 2)
///     .code("code", "yaml")
///     .header("Explanation", 2)
///     .text("explanation")
///     .header("Common Mistakes", 2)
///     .list("errors");
///
/// // For Python (user-defined)
/// let python_template = RagDocumentTemplate::new("python_code")
///     .header("Task", 2)
///     .text("question")
///     .header("Solution", 2)
///     .code("code", "python")
///     .header("Common Mistakes", 2)
///     .list("errors");
///
/// // For SQL (user-defined)
/// let sql_template = RagDocumentTemplate::new("sql_query")
///     .header("Task", 2)
///     .text("question")
///     .header("Query", 2)
///     .code("code", "sql");
/// ```
#[derive(Clone, Debug)]
pub struct RagDocumentTemplate<'a> {
    /// Template name/identifier.
    pub name: StrView<'a>,
    /// Sections in the document.
    pub sections: SmallVec<[TemplateSection<'a>; 8]>,
}

impl<'a> Default for RagDocumentTemplate<'a> {
    fn default() -> Self {
        Self {
            name: StrView::new("default"),
            sections: SmallVec::new(),
        }
    }
}

impl<'a> RagDocumentTemplate<'a> {
    /// Create a new template with the given name.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let template = RagDocumentTemplate::new("kubernetes");
    /// ```
    pub fn new(name: &'a str) -> Self {
        Self {
            name: StrView::new(name),
            sections: SmallVec::new(),
        }
    }

    /// Add a section to the template.
    pub fn section(mut self, section: TemplateSection<'a>) -> Self {
        self.sections.push(section);
        self
    }

    /// Add a header section.
    ///
    /// # Arguments
    ///
    /// * `label` - The header text
    /// * `level` - Header level (1-6), e.g., 2 produces `## Header`
    pub fn header(self, label: &'a str, level: u8) -> Self {
        self.section(TemplateSection::Header {
            label: StrView::new(label),
            level,
        })
    }

    /// Add a text section.
    ///
    /// # Arguments
    ///
    /// * `label` - Field name to retrieve content from the result
    pub fn text(self, label: &'a str) -> Self {
        self.section(TemplateSection::Text {
            label: StrView::new(label),
        })
    }

    /// Add a code section with language for syntax highlighting.
    ///
    /// # Arguments
    ///
    /// * `label` - Field name to retrieve code from the result
    /// * `language` - Language for syntax highlighting (e.g., "yaml", "python", "rust", "sql")
    pub fn code(self, label: &'a str, language: &'a str) -> Self {
        self.section(TemplateSection::Code {
            label: StrView::new(label),
            language: StrView::new(language),
        })
    }

    /// Add a bulleted list section.
    ///
    /// # Arguments
    ///
    /// * `label` - Field name to retrieve list items from the result
    pub fn list(self, label: &'a str) -> Self {
        self.section(TemplateSection::List {
            label: StrView::new(label),
        })
    }

    /// Get the template name.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Get the number of sections.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Format a result into a document string.
    ///
    /// This is the only method that allocates - it produces an owned `String`
    /// containing the formatted document in Markdown format.
    ///
    /// # Arguments
    ///
    /// * `fields` - A function that maps field labels to their string values
    /// * `lists` - A function that maps list labels to their items
    /// * `metadata` - Optional metadata to append at the end
    ///
    /// # Note
    ///
    /// The closures must return `'static` references. For string literals this
    /// is automatic. For dynamic data, use `Box::leak()` or owned alternatives.
    pub fn format<F, L>(&self, fields: F, lists: L, metadata: Option<&DocumentMetadata>) -> String
    where
        F: Fn(&str) -> &'static str,
        L: Fn(&str) -> &'static [String],
    {
        let mut doc = String::with_capacity(4096);

        for section in &self.sections {
            match section {
                TemplateSection::Header { label, level } => {
                    let prefix: String = "#".repeat(*level as usize);
                    writeln!(doc, "{} {}\n", prefix, label.as_str()).unwrap();
                }
                TemplateSection::Text { label } => {
                    let content = fields(label.as_str());
                    writeln!(doc, "{}\n", content).unwrap();
                }
                TemplateSection::Code { label, language } => {
                    let code = fields(label.as_str());
                    writeln!(doc, "```{}\n{}\n```\n", language.as_str(), code).unwrap();
                }
                TemplateSection::List { label } => {
                    let items = lists(label.as_str());
                    for item in items {
                        writeln!(doc, "- {}", item).unwrap();
                    }
                    doc.push('\n');
                }
            }
        }

        // Add metadata footer if provided
        if let Some(meta) = metadata {
            writeln!(
                doc,
                "---\n_Score: {:.2} | Iterations: {} | One-shot validated: {}_",
                meta.score, meta.iterations, meta.oneshot_validated
            )
            .unwrap();
        }

        doc
    }
}

// =============================================================================
// Document Metadata
// =============================================================================

/// Metadata appended to formatted documents.
#[derive(Clone, Debug, Default)]
pub struct DocumentMetadata {
    /// Final optimization score.
    pub score: f64,
    /// Number of iterations to converge.
    pub iterations: u32,
    /// Number of optimization attempts (one-shot retries).
    pub optimization_attempts: u32,
    /// Whether one-shot validation passed.
    pub oneshot_validated: bool,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_creation() {
        let template = RagDocumentTemplate::new("test")
            .header("Task", 2)
            .text("question")
            .code("code", "python")
            .list("errors");

        assert_eq!(template.name(), "test");
        assert_eq!(template.section_count(), 4);
    }

    #[test]
    fn test_template_format() {
        let template = RagDocumentTemplate::new("test")
            .header("Task", 2)
            .text("question")
            .header("Solution", 2)
            .code("code", "python")
            .header("Errors", 2)
            .list("errors");

        let fields = |label: &str| -> &str {
            match label {
                "question" => "How do I parse JSON?",
                "code" => "import json\ndata = json.loads(text)",
                _ => "",
            }
        };

        // Use Box::leak to create 'static reference for test (acceptable in tests)
        let error_list: &'static [String] = Box::leak(Box::new(vec![
            "Missing import".to_string(),
            "Invalid syntax".to_string(),
        ]));
        let lists = |label: &str| -> &[String] {
            if label == "errors" {
                error_list
            } else {
                &[]
            }
        };

        let meta = DocumentMetadata {
            score: 0.95,
            iterations: 3,
            optimization_attempts: 1,
            oneshot_validated: true,
        };

        let doc = template.format(fields, lists, Some(&meta));

        assert!(doc.contains("## Task"));
        assert!(doc.contains("How do I parse JSON?"));
        assert!(doc.contains("```python"));
        assert!(doc.contains("import json"));
        assert!(doc.contains("```"));
        assert!(doc.contains("- Missing import"));
        assert!(doc.contains("Score: 0.95"));
        assert!(doc.contains("One-shot validated: true"));
    }

    #[test]
    fn test_template_no_metadata() {
        let template = RagDocumentTemplate::new("simple")
            .header("Question", 2)
            .text("question");

        let doc = template.format(|_| "What is 2+2?", |_| &[], None);

        assert!(doc.contains("## Question"));
        assert!(doc.contains("What is 2+2?"));
        assert!(!doc.contains("Score:"));
    }
}
