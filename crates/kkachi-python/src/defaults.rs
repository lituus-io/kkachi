// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the Defaults runtime substitution system.

use pyo3::prelude::*;

use kkachi::recursive::Defaults;

/// Runtime defaults for regex substitution on LLM output.
///
/// Stores regex→replacement pairs with metadata. Integrates at two points:
/// 1. Prompt injection via `.context()` — renders a summary for the LLM
/// 2. Output transform via `.apply(text)` — regex replacements before validation
///
/// Example:
/// ```python
/// defaults = (
///     Defaults()
///     .set("iam_user", r"user:\S+@example\.com", "user:mark@company.com",
///          note="Replace with actual IAM user")
///     .from_env("project", r"my-gcp-project",
///               "GOOGLE_CLOUD_PROJECT", fallback="default-proj")
/// )
///
/// # Inject into prompt
/// prompt = f"...{defaults.context()}..."
///
/// # Apply substitutions
/// fixed = defaults.apply("user:admin@example.com in my-gcp-project")
/// ```
#[pyclass(name = "Defaults")]
#[derive(Clone)]
pub struct PyDefaults {
    pub(crate) inner: Defaults,
}

#[pymethods]
impl PyDefaults {
    /// Create an empty defaults collection.
    #[new]
    fn new() -> Self {
        Self {
            inner: Defaults::new(),
        }
    }

    /// Add a literal regex substitution.
    ///
    /// Args:
    ///     key: A human-readable name for this default.
    ///     pattern: Regex pattern to match in LLM output.
    ///     replacement: The replacement string.
    ///     note: Optional annotation note for documentation.
    #[pyo3(signature = (key, pattern, replacement, note=None))]
    fn set(&self, key: String, pattern: String, replacement: String, note: Option<String>) -> Self {
        let inner = if let Some(note) = note {
            self.inner
                .clone()
                .set_with_note(&key, &pattern, &replacement, &note)
        } else {
            self.inner.clone().set(&key, &pattern, &replacement)
        };
        Self { inner }
    }

    /// Read replacement value from an environment variable; use fallback if unset.
    ///
    /// Args:
    ///     key: A human-readable name for this default.
    ///     pattern: Regex pattern to match in LLM output.
    ///     env_var: Environment variable name to read the replacement from.
    ///     fallback: Fallback value if the env var is not set.
    ///     note: Optional annotation note for documentation.
    #[pyo3(signature = (key, pattern, env_var, fallback, note=None))]
    fn from_env(
        &self,
        key: String,
        pattern: String,
        env_var: String,
        fallback: String,
        note: Option<String>,
    ) -> Self {
        let inner = if let Some(note) = note {
            self.inner
                .clone()
                .from_env_with_note(&key, &pattern, &env_var, &fallback, &note)
        } else {
            self.inner
                .clone()
                .from_env(&key, &pattern, &env_var, &fallback)
        };
        Self { inner }
    }

    /// Apply all regex substitutions to text. Returns transformed text.
    fn apply(&self, text: &str) -> String {
        self.inner.apply(text)
    }

    /// Generate a context string for prompt injection.
    fn context(&self) -> String {
        self.inner.context()
    }

    /// Get annotation metadata for all entries.
    fn annotations(&self) -> Vec<PyDefaultAnnotation> {
        self.inner
            .annotations()
            .into_iter()
            .map(|a| PyDefaultAnnotation {
                key: a.key,
                original_pattern: a.original_pattern,
                replacement: a.replacement,
                note: a.note,
                source: a.source,
            })
            .collect()
    }

    /// Check if there are no entries.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Defaults(entries={})", self.inner.len())
    }
}

/// Annotation metadata for a single default entry.
#[pyclass(name = "DefaultAnnotation")]
#[derive(Clone)]
pub struct PyDefaultAnnotation {
    /// Key name for this default.
    #[pyo3(get)]
    pub key: String,
    /// The original regex pattern string.
    #[pyo3(get)]
    pub original_pattern: String,
    /// The replacement value.
    #[pyo3(get)]
    pub replacement: String,
    /// Optional human-readable note.
    #[pyo3(get)]
    pub note: Option<String>,
    /// Source description: "literal" or "env:VAR_NAME".
    #[pyo3(get)]
    pub source: String,
}

#[pymethods]
impl PyDefaultAnnotation {
    fn __repr__(&self) -> String {
        format!(
            "DefaultAnnotation(key='{}', replacement='{}', source='{}')",
            self.key, self.replacement, self.source
        )
    }
}
