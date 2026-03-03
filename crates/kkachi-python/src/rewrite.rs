// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for markdown rewriting utilities.

use pyo3::prelude::*;

use kkachi::recursive::rewrite::{extract_all_code, extract_code, rewrite};

/// Markdown rewriter with fluent API.
///
/// Example:
/// ```python
/// result = Rewrite(markdown) \
///     .replace_code("yaml", new_yaml_content) \
///     .build()
/// ```
#[pyclass(name = "Rewrite")]
pub struct PyRewrite {
    markdown: String,
    replacements: Vec<(String, String)>,
}

#[pymethods]
impl PyRewrite {
    /// Create a new rewriter for the given markdown.
    #[new]
    fn new(markdown: String) -> Self {
        Self {
            markdown,
            replacements: Vec::new(),
        }
    }

    /// Replace the first code block of a given language.
    fn replace_code(&mut self, lang: String, content: String) -> Self {
        self.replacements.push((lang, content));
        Self {
            markdown: self.markdown.clone(),
            replacements: self.replacements.clone(),
        }
    }

    /// Build the final markdown with all replacements applied.
    fn build(&self) -> String {
        let mut rw = rewrite(&self.markdown);
        for (lang, content) in &self.replacements {
            rw = rw.replace_code(lang, content);
        }
        rw.build()
    }

    fn __repr__(&self) -> String {
        format!("Rewrite(len={})", self.markdown.len())
    }
}

/// Extract the first code block of a given language from markdown.
///
/// Args:
///     markdown: The markdown text.
///     lang: The language identifier (e.g., "yaml", "rust", "python").
///
/// Returns:
///     The code block content, or None if not found.
#[pyfunction]
#[pyo3(name = "extract_code")]
pub fn py_extract_code(markdown: &str, lang: &str) -> Option<String> {
    extract_code(markdown, lang).map(|s| s.to_string())
}

/// Extract all code blocks of a given language from markdown.
///
/// Args:
///     markdown: The markdown text.
///     lang: The language identifier (e.g., "yaml", "rust", "python").
///
/// Returns:
///     A list of code block contents.
#[pyfunction]
#[pyo3(name = "extract_all_code")]
pub fn py_extract_all_code(markdown: &str, lang: &str) -> Vec<String> {
    extract_all_code(markdown, lang)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}
