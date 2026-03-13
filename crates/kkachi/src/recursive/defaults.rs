// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Runtime defaults for regex substitution on LLM output.
//!
//! Provides a [`Defaults`] type that stores regex→replacement pairs with metadata.
//! It integrates at two points:
//!
//! 1. **Prompt injection** — [`Defaults::context()`] renders a human-readable summary
//! 2. **Output transform** — [`Defaults::apply()`] does regex replacements before validation
//!
//! # Example
//!
//! ```
//! use kkachi::recursive::Defaults;
//!
//! let defaults = Defaults::new()
//!     .set("email", r"admin@example\.com", "real@company.com")
//!     .set_with_note("project", r"my-project", "prod-project-123",
//!         "Replace with actual GCP project ID");
//!
//! let output = defaults.apply("user:admin@example.com in my-project");
//! assert!(output.contains("real@company.com"));
//! assert!(output.contains("prod-project-123"));
//!
//! let ctx = defaults.context();
//! assert!(ctx.contains("email"));
//! ```

use regex::Regex;
use smallvec::SmallVec;

/// A collection of runtime default values applied via regex substitution.
///
/// Used to replace known-bad placeholders in LLM output before validation,
/// and to generate context strings for prompt injection.
#[derive(Clone)]
pub struct Defaults {
    entries: SmallVec<[DefaultEntry; 8]>,
}

#[derive(Clone)]
struct DefaultEntry {
    key: String,
    pattern: Regex,
    pattern_str: String,
    replacement: String,
    source: ValueSource,
    note: Option<String>,
}

#[derive(Clone)]
enum ValueSource {
    Literal,
    EnvVar(String),
}

/// Annotation metadata for a single default entry.
#[derive(Debug, Clone)]
pub struct DefaultAnnotation {
    /// Key name for this default.
    pub key: String,
    /// The original regex pattern string.
    pub original_pattern: String,
    /// The replacement value.
    pub replacement: String,
    /// Optional human-readable note.
    pub note: Option<String>,
    /// Source description: "literal" or "env:VAR_NAME".
    pub source: String,
}

impl Defaults {
    /// Create an empty defaults collection.
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    /// Add a literal regex substitution.
    pub fn set(mut self, key: &str, pattern: &str, replacement: &str) -> Self {
        if let Ok(re) = Regex::new(pattern) {
            self.entries.push(DefaultEntry {
                key: key.to_string(),
                pattern: re,
                pattern_str: pattern.to_string(),
                replacement: replacement.to_string(),
                source: ValueSource::Literal,
                note: None,
            });
        }
        self
    }

    /// Add a literal regex substitution with a human-readable annotation note.
    pub fn set_with_note(
        mut self,
        key: &str,
        pattern: &str,
        replacement: &str,
        note: &str,
    ) -> Self {
        if let Ok(re) = Regex::new(pattern) {
            self.entries.push(DefaultEntry {
                key: key.to_string(),
                pattern: re,
                pattern_str: pattern.to_string(),
                replacement: replacement.to_string(),
                source: ValueSource::Literal,
                note: Some(note.to_string()),
            });
        }
        self
    }

    /// Read replacement value from an environment variable; use fallback if unset.
    pub fn from_env(mut self, key: &str, pattern: &str, env_var: &str, fallback: &str) -> Self {
        let value = std::env::var(env_var).unwrap_or_else(|_| fallback.to_string());
        if let Ok(re) = Regex::new(pattern) {
            self.entries.push(DefaultEntry {
                key: key.to_string(),
                pattern: re,
                pattern_str: pattern.to_string(),
                replacement: value,
                source: ValueSource::EnvVar(env_var.to_string()),
                note: None,
            });
        }
        self
    }

    /// Read replacement value from an environment variable with a note.
    pub fn from_env_with_note(
        mut self,
        key: &str,
        pattern: &str,
        env_var: &str,
        fallback: &str,
        note: &str,
    ) -> Self {
        let value = std::env::var(env_var).unwrap_or_else(|_| fallback.to_string());
        if let Ok(re) = Regex::new(pattern) {
            self.entries.push(DefaultEntry {
                key: key.to_string(),
                pattern: re,
                pattern_str: pattern.to_string(),
                replacement: value,
                source: ValueSource::EnvVar(env_var.to_string()),
                note: Some(note.to_string()),
            });
        }
        self
    }

    /// Apply all regex substitutions to text. Returns transformed text.
    pub fn apply(&self, text: &str) -> String {
        let mut result = text.to_string();
        for entry in &self.entries {
            result = entry
                .pattern
                .replace_all(&result, entry.replacement.as_str())
                .into_owned();
        }
        result
    }

    /// Generate a context string for prompt injection.
    ///
    /// Renders as:
    /// ```text
    /// ## Runtime Defaults
    /// - email: real@company.com (literal)
    /// - project: prod-123 (from env: GCP_PROJECT)
    /// ```
    pub fn context(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        let mut lines = vec!["## Runtime Defaults".to_string()];
        for entry in &self.entries {
            let source = match &entry.source {
                ValueSource::Literal => "literal".to_string(),
                ValueSource::EnvVar(var) => format!("from env: {}", var),
            };
            let mut line = format!("- {}: {} ({})", entry.key, entry.replacement, source);
            if let Some(ref note) = entry.note {
                line.push_str(&format!(" — {}", note));
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    /// Get annotation metadata for all entries.
    pub fn annotations(&self) -> Vec<DefaultAnnotation> {
        self.entries
            .iter()
            .map(|entry| DefaultAnnotation {
                key: entry.key.clone(),
                original_pattern: entry.pattern_str.clone(),
                replacement: entry.replacement.clone(),
                note: entry.note.clone(),
                source: match &entry.source {
                    ValueSource::Literal => "literal".to_string(),
                    ValueSource::EnvVar(var) => format!("env:{}", var),
                },
            })
            .collect()
    }

    /// Merge another Defaults collection into this one.
    ///
    /// Entries from `other` are appended. If `other` has an entry with the
    /// same key as one already in `self`, the one from `other` replaces it.
    pub fn merge(mut self, other: &Defaults) -> Self {
        for other_entry in &other.entries {
            self.entries.retain(|e| e.key != other_entry.key);
            self.entries.push(other_entry.clone());
        }
        self
    }

    /// Render all defaults as a markdown table.
    ///
    /// Returns a table with columns: Key, Pattern, Replacement, Source, Note.
    pub fn to_markdown_table(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        let mut out = String::with_capacity(256);
        out.push_str("| Key | Pattern | Replacement | Source | Note |\n");
        out.push_str("|-----|---------|-------------|--------|------|\n");

        for entry in &self.entries {
            let source = match &entry.source {
                ValueSource::Literal => "literal",
                ValueSource::EnvVar(var) => var.as_str(),
            };
            let note = entry.note.as_deref().unwrap_or("");
            out.push_str(&format!(
                "| {} | `{}` | `{}` | {} | {} |\n",
                entry.key, entry.pattern_str, entry.replacement, source, note
            ));
        }

        out
    }

    /// Check if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for Defaults {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_apply_single() {
        let defaults = Defaults::new().set("email", r"admin@example\.com", "real@company.com");

        let result = defaults.apply("user:admin@example.com");
        assert_eq!(result, "user:real@company.com");
    }

    #[test]
    fn test_defaults_apply_multiple() {
        let defaults = Defaults::new()
            .set("email", r"user:\S+@example\.com", "user:spicyzhug@test.com")
            .set("project", r"my-gcp-project", "prod-project-123");

        let text = "IAM: user:admin@example.com in my-gcp-project";
        let result = defaults.apply(text);
        assert!(result.contains("user:spicyzhug@test.com"));
        assert!(result.contains("prod-project-123"));
        assert!(!result.contains("example.com"));
        assert!(!result.contains("my-gcp-project"));
    }

    #[test]
    fn test_defaults_from_env() {
        // Test with a non-existent env var (should use fallback)
        let defaults = Defaults::new().from_env(
            "project",
            r"my-gcp-project",
            "KKACHI_TEST_NONEXISTENT_VAR_12345",
            "fallback-project",
        );

        let result = defaults.apply("deploy to my-gcp-project");
        assert_eq!(result, "deploy to fallback-project");
    }

    #[test]
    fn test_defaults_from_env_with_var_set() {
        std::env::set_var("KKACHI_TEST_PROJECT_ID", "env-project-value");
        let defaults = Defaults::new().from_env(
            "project",
            r"my-gcp-project",
            "KKACHI_TEST_PROJECT_ID",
            "fallback-project",
        );

        let result = defaults.apply("deploy to my-gcp-project");
        assert_eq!(result, "deploy to env-project-value");
        std::env::remove_var("KKACHI_TEST_PROJECT_ID");
    }

    #[test]
    fn test_defaults_context() {
        let defaults = Defaults::new()
            .set("email", r"admin@example\.com", "real@company.com")
            .from_env(
                "project",
                r"my-project",
                "KKACHI_TEST_NONEXISTENT_12345",
                "fallback-proj",
            );

        let ctx = defaults.context();
        assert!(ctx.contains("## Runtime Defaults"));
        assert!(ctx.contains("email: real@company.com (literal)"));
        assert!(ctx.contains("project: fallback-proj (from env: KKACHI_TEST_NONEXISTENT_12345)"));
    }

    #[test]
    fn test_defaults_context_with_notes() {
        let defaults = Defaults::new().set_with_note(
            "email",
            r"admin@example\.com",
            "real@company.com",
            "Replace with actual IAM user",
        );

        let ctx = defaults.context();
        assert!(ctx.contains("Replace with actual IAM user"));
    }

    #[test]
    fn test_defaults_annotations() {
        let defaults = Defaults::new()
            .set_with_note(
                "email",
                r"admin@example\.com",
                "real@company.com",
                "IAM user",
            )
            .from_env(
                "project",
                r"my-project",
                "KKACHI_TEST_NONEXISTENT_12345",
                "fallback",
            );

        let annotations = defaults.annotations();
        assert_eq!(annotations.len(), 2);

        assert_eq!(annotations[0].key, "email");
        assert_eq!(annotations[0].replacement, "real@company.com");
        assert_eq!(annotations[0].note.as_deref(), Some("IAM user"));
        assert_eq!(annotations[0].source, "literal");

        assert_eq!(annotations[1].key, "project");
        assert!(annotations[1].source.starts_with("env:"));
    }

    #[test]
    fn test_defaults_no_match() {
        let defaults = Defaults::new().set("email", r"admin@example\.com", "real@company.com");

        let text = "no matches here";
        let result = defaults.apply(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_defaults_empty() {
        let defaults = Defaults::new();
        assert!(defaults.is_empty());
        assert_eq!(defaults.len(), 0);
        assert_eq!(defaults.context(), "");
        assert!(defaults.annotations().is_empty());
        assert_eq!(defaults.apply("unchanged"), "unchanged");
    }

    #[test]
    fn test_defaults_multiple_occurrences() {
        let defaults = Defaults::new().set("email", r"admin@example\.com", "real@company.com");

        let text = "user:admin@example.com and group:admin@example.com";
        let result = defaults.apply(text);
        assert_eq!(result, "user:real@company.com and group:real@company.com");
    }

    #[test]
    fn test_defaults_invalid_regex_skipped() {
        // Invalid regex should be silently skipped
        let defaults = Defaults::new()
            .set("bad", r"[invalid", "replacement")
            .set("good", r"hello", "world");

        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults.apply("hello"), "world");
    }
}
