// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Diff visualization system for prompt optimization.
//!
//! This module provides GitHub-style colorized diffs for viewing changes
//! between optimization iterations, including:
//! - Text-level diffs (line and word granularity)
//! - Structured diffs for module changes (instructions, demos, outputs)
//! - ANSI terminal rendering with configurable styles

mod render;
mod structured;

pub use render::{DiffColors, DiffRenderer, DiffStyle};
pub use structured::{DemoSnapshot, DemosDiff, FieldsDiff, IterationDiffBuilder, ModuleDiff};

use similar::{ChangeTag, TextDiff as SimilarTextDiff};

/// Type of change in a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// Text is unchanged.
    Equal,
    /// Text was added.
    Insert,
    /// Text was removed.
    Delete,
}

impl From<ChangeTag> for ChangeKind {
    fn from(tag: ChangeTag) -> Self {
        match tag {
            ChangeTag::Equal => ChangeKind::Equal,
            ChangeTag::Insert => ChangeKind::Insert,
            ChangeTag::Delete => ChangeKind::Delete,
        }
    }
}

/// A single change span in a diff.
#[derive(Debug, Clone)]
pub struct Change<'a> {
    /// Type of change.
    pub kind: ChangeKind,
    /// The text content of this change.
    pub value: &'a str,
    /// Line number in old text (for deletions/equals).
    pub old_line: Option<usize>,
    /// Line number in new text (for insertions/equals).
    pub new_line: Option<usize>,
}

/// Diff algorithm to use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DiffAlgorithm {
    /// Myers diff algorithm - fast and good for most cases.
    #[default]
    Myers,
    /// Patience diff algorithm - better for code, respects structure.
    Patience,
    /// LCS (Longest Common Subsequence) - simple but effective.
    Lcs,
}

/// Text diff result between two strings.
///
/// Supports both line-based and word-based diffing.
#[derive(Debug)]
pub struct TextDiff<'a> {
    /// The original (old) text.
    old: &'a str,
    /// The new text.
    new: &'a str,
    /// List of changes.
    changes: Vec<Change<'a>>,
    /// Algorithm used.
    algorithm: DiffAlgorithm,
    /// Whether this is a word-level diff.
    word_level: bool,
}

impl<'a> TextDiff<'a> {
    /// Create a line-based diff between two strings.
    pub fn new(old: &'a str, new: &'a str) -> Self {
        Self::with_algorithm(old, new, DiffAlgorithm::default())
    }

    /// Create a diff with a specific algorithm.
    pub fn with_algorithm(old: &'a str, new: &'a str, algorithm: DiffAlgorithm) -> Self {
        let similar_diff = match algorithm {
            DiffAlgorithm::Myers => SimilarTextDiff::from_lines(old, new),
            DiffAlgorithm::Patience => SimilarTextDiff::configure()
                .algorithm(similar::Algorithm::Patience)
                .diff_lines(old, new),
            DiffAlgorithm::Lcs => SimilarTextDiff::configure()
                .algorithm(similar::Algorithm::Lcs)
                .diff_lines(old, new),
        };

        let changes = Self::extract_changes(&similar_diff);

        Self {
            old,
            new,
            changes,
            algorithm,
            word_level: false,
        }
    }

    /// Create a word-level diff (more granular than line-level).
    pub fn word_diff(old: &'a str, new: &'a str) -> Self {
        let similar_diff = SimilarTextDiff::from_words(old, new);
        let changes = Self::extract_word_changes(&similar_diff);

        Self {
            old,
            new,
            changes,
            algorithm: DiffAlgorithm::Myers,
            word_level: true,
        }
    }

    /// Create a character-level diff (most granular).
    pub fn char_diff(old: &'a str, new: &'a str) -> Self {
        let similar_diff = SimilarTextDiff::from_chars(old, new);
        let changes = Self::extract_char_changes(&similar_diff);

        Self {
            old,
            new,
            changes,
            algorithm: DiffAlgorithm::Myers,
            word_level: true, // Treat as inline diff
        }
    }

    /// Extract changes from a similar TextDiff (line-level).
    fn extract_changes(diff: &SimilarTextDiff<'a, 'a, 'a, str>) -> Vec<Change<'a>> {
        let mut changes = Vec::new();
        let mut old_line = 1usize;
        let mut new_line = 1usize;

        for change in diff.iter_all_changes() {
            let kind = ChangeKind::from(change.tag());
            let value = change.value();

            let (old_ln, new_ln) = match kind {
                ChangeKind::Equal => {
                    let result = (Some(old_line), Some(new_line));
                    old_line += 1;
                    new_line += 1;
                    result
                }
                ChangeKind::Delete => {
                    let result = (Some(old_line), None);
                    old_line += 1;
                    result
                }
                ChangeKind::Insert => {
                    let result = (None, Some(new_line));
                    new_line += 1;
                    result
                }
            };

            changes.push(Change {
                kind,
                value,
                old_line: old_ln,
                new_line: new_ln,
            });
        }

        changes
    }

    /// Extract changes from word-level diff.
    fn extract_word_changes(diff: &SimilarTextDiff<'a, 'a, 'a, str>) -> Vec<Change<'a>> {
        diff.iter_all_changes()
            .map(|change| Change {
                kind: ChangeKind::from(change.tag()),
                value: change.value(),
                old_line: None,
                new_line: None,
            })
            .collect()
    }

    /// Extract changes from char-level diff.
    fn extract_char_changes(diff: &SimilarTextDiff<'a, 'a, 'a, str>) -> Vec<Change<'a>> {
        diff.iter_all_changes()
            .map(|change| Change {
                kind: ChangeKind::from(change.tag()),
                value: change.value(),
                old_line: None,
                new_line: None,
            })
            .collect()
    }

    /// Check if there are any changes between old and new.
    pub fn has_changes(&self) -> bool {
        self.changes.iter().any(|c| c.kind != ChangeKind::Equal)
    }

    /// Get the list of changes.
    pub fn changes(&self) -> &[Change<'a>] {
        &self.changes
    }

    /// Get the old text.
    pub fn old_text(&self) -> &'a str {
        self.old
    }

    /// Get the new text.
    pub fn new_text(&self) -> &'a str {
        self.new
    }

    /// Get the algorithm used.
    pub fn algorithm(&self) -> DiffAlgorithm {
        self.algorithm
    }

    /// Whether this is a word-level diff.
    pub fn is_word_level(&self) -> bool {
        self.word_level
    }

    /// Get statistics about the diff.
    pub fn stats(&self) -> DiffStats {
        let mut lines_added = 0usize;
        let mut lines_removed = 0usize;

        for change in &self.changes {
            match change.kind {
                ChangeKind::Insert => lines_added += 1,
                ChangeKind::Delete => lines_removed += 1,
                ChangeKind::Equal => {}
            }
        }

        DiffStats {
            lines_added,
            lines_removed,
            lines_changed: lines_added.min(lines_removed),
            total_changes: lines_added + lines_removed,
        }
    }

    /// Get a unified diff string (plain text).
    pub fn unified(&self, context_lines: usize) -> String {
        let diff = SimilarTextDiff::from_lines(self.old, self.new);
        diff.unified_diff()
            .context_radius(context_lines)
            .to_string()
    }

    /// Iterate over hunks (groups of changes with context).
    pub fn hunks(&self, context_lines: usize) -> Vec<DiffHunk<'a>> {
        let mut hunks = Vec::new();
        let mut current_hunk: Option<DiffHunk<'a>> = None;
        let mut context_buffer: Vec<Change<'a>> = Vec::new();

        for change in &self.changes {
            match change.kind {
                ChangeKind::Equal => {
                    if let Some(ref mut hunk) = current_hunk {
                        // Add trailing context
                        if context_buffer.len() < context_lines {
                            hunk.changes.push(change.clone());
                            context_buffer.push(change.clone());
                        } else {
                            // End current hunk
                            hunks.push(current_hunk.take().unwrap());
                            context_buffer.clear();
                        }
                    }
                    // Buffer for potential leading context
                    context_buffer.push(change.clone());
                    if context_buffer.len() > context_lines {
                        context_buffer.remove(0);
                    }
                }
                _ => {
                    if current_hunk.is_none() {
                        // Start new hunk with leading context
                        let start_old =
                            context_buffer.first().and_then(|c| c.old_line).unwrap_or(1);
                        let start_new =
                            context_buffer.first().and_then(|c| c.new_line).unwrap_or(1);

                        current_hunk = Some(DiffHunk {
                            old_start: start_old,
                            new_start: start_new,
                            changes: context_buffer.clone(),
                        });
                        context_buffer.clear();
                    }

                    if let Some(ref mut hunk) = current_hunk {
                        hunk.changes.push(change.clone());
                    }
                }
            }
        }

        // Don't forget the last hunk
        if let Some(hunk) = current_hunk {
            hunks.push(hunk);
        }

        hunks
    }
}

/// Statistics about a diff.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiffStats {
    /// Number of lines added.
    pub lines_added: usize,
    /// Number of lines removed.
    pub lines_removed: usize,
    /// Number of lines changed (min of added/removed).
    pub lines_changed: usize,
    /// Total number of changes.
    pub total_changes: usize,
}

impl DiffStats {
    /// Check if there are any changes.
    pub fn has_changes(&self) -> bool {
        self.total_changes > 0
    }

    /// Format as a compact string like "+10 -5".
    pub fn compact(&self) -> String {
        format!("+{} -{}", self.lines_added, self.lines_removed)
    }
}

/// A hunk (group of changes with context).
#[derive(Debug, Clone)]
pub struct DiffHunk<'a> {
    /// Starting line in old text.
    pub old_start: usize,
    /// Starting line in new text.
    pub new_start: usize,
    /// Changes in this hunk (including context).
    pub changes: Vec<Change<'a>>,
}

impl<'a> DiffHunk<'a> {
    /// Get the range in the old text.
    pub fn old_range(&self) -> (usize, usize) {
        let count = self
            .changes
            .iter()
            .filter(|c| c.kind != ChangeKind::Insert)
            .count();
        (self.old_start, count)
    }

    /// Get the range in the new text.
    pub fn new_range(&self) -> (usize, usize) {
        let count = self
            .changes
            .iter()
            .filter(|c| c.kind != ChangeKind::Delete)
            .count();
        (self.new_start, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_diff_no_changes() {
        let text = "hello\nworld\n";
        let diff = TextDiff::new(text, text);

        assert!(!diff.has_changes());
        assert_eq!(diff.stats().total_changes, 0);
    }

    #[test]
    fn test_text_diff_simple_addition() {
        let old = "hello\n";
        let new = "hello\nworld\n";
        let diff = TextDiff::new(old, new);

        assert!(diff.has_changes());
        let stats = diff.stats();
        assert_eq!(stats.lines_added, 1);
        assert_eq!(stats.lines_removed, 0);
    }

    #[test]
    fn test_text_diff_simple_removal() {
        let old = "hello\nworld\n";
        let new = "hello\n";
        let diff = TextDiff::new(old, new);

        assert!(diff.has_changes());
        let stats = diff.stats();
        assert_eq!(stats.lines_added, 0);
        assert_eq!(stats.lines_removed, 1);
    }

    #[test]
    fn test_text_diff_modification() {
        let old = "hello\nworld\n";
        let new = "hello\nearth\n";
        let diff = TextDiff::new(old, new);

        assert!(diff.has_changes());
        let stats = diff.stats();
        assert_eq!(stats.lines_added, 1);
        assert_eq!(stats.lines_removed, 1);
    }

    #[test]
    fn test_word_diff() {
        let old = "hello world";
        let new = "hello earth";
        let diff = TextDiff::word_diff(old, new);

        assert!(diff.has_changes());
        assert!(diff.is_word_level());
    }

    #[test]
    fn test_diff_stats_compact() {
        let stats = DiffStats {
            lines_added: 10,
            lines_removed: 5,
            lines_changed: 5,
            total_changes: 15,
        };

        assert_eq!(stats.compact(), "+10 -5");
    }

    #[test]
    fn test_diff_unified() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nmodified\nline3\n";
        let diff = TextDiff::new(old, new);

        let unified = diff.unified(1);
        assert!(unified.contains("-line2"));
        assert!(unified.contains("+modified"));
    }

    #[test]
    fn test_diff_hunks() {
        let old = "a\nb\nc\nd\ne\n";
        let new = "a\nx\nc\nd\ne\n";
        let diff = TextDiff::new(old, new);

        let hunks = diff.hunks(1);
        assert!(!hunks.is_empty());
    }

    #[test]
    fn test_change_kind_from_tag() {
        assert_eq!(ChangeKind::from(ChangeTag::Equal), ChangeKind::Equal);
        assert_eq!(ChangeKind::from(ChangeTag::Insert), ChangeKind::Insert);
        assert_eq!(ChangeKind::from(ChangeTag::Delete), ChangeKind::Delete);
    }
}
