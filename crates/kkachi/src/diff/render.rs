// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! ANSI terminal rendering for diffs.
//!
//! Provides colorized diff output with configurable styles.

use super::{ChangeKind, DiffStats, TextDiff};
use console::{style, Color, Style, Term};
use std::fmt::Write;

/// Style for rendering diffs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DiffStyle {
    /// Classic unified diff format with context.
    #[default]
    Unified,
    /// Side-by-side two-column comparison.
    SideBySide,
    /// Word-level inline highlights.
    Inline,
    /// Minimal output showing only changes.
    Compact,
}

/// Color configuration for diff output.
#[derive(Debug, Clone)]
pub struct DiffColors {
    /// Color for added lines/text.
    pub added: Color,
    /// Color for removed lines/text.
    pub removed: Color,
    /// Color for context lines.
    pub context: Color,
    /// Color for headers and separators.
    pub header: Color,
    /// Color for line numbers.
    pub line_number: Color,
    /// Color for score improvements.
    pub score_up: Color,
    /// Color for score decreases.
    pub score_down: Color,
}

impl Default for DiffColors {
    fn default() -> Self {
        Self {
            added: Color::Green,
            removed: Color::Red,
            context: Color::Color256(245), // Dim gray
            header: Color::Cyan,
            line_number: Color::Color256(240), // Dark gray
            score_up: Color::Green,
            score_down: Color::Red,
        }
    }
}

/// Configuration for diff rendering.
#[derive(Debug, Clone)]
pub struct DiffRenderer {
    /// Rendering style.
    pub style: DiffStyle,
    /// Color configuration.
    pub colors: DiffColors,
    /// Number of context lines around changes.
    pub context_lines: usize,
    /// Whether to show line numbers.
    pub show_line_numbers: bool,
    /// Terminal width (auto-detected if None).
    pub terminal_width: Option<usize>,
    /// Whether to use box drawing characters.
    pub use_box_drawing: bool,
}

impl Default for DiffRenderer {
    fn default() -> Self {
        Self {
            style: DiffStyle::default(),
            colors: DiffColors::default(),
            context_lines: 3,
            show_line_numbers: true,
            terminal_width: None,
            use_box_drawing: true,
        }
    }
}

impl DiffRenderer {
    /// Create a new renderer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the rendering style.
    pub fn with_style(mut self, style: DiffStyle) -> Self {
        self.style = style;
        self
    }

    /// Set custom colors.
    pub fn with_colors(mut self, colors: DiffColors) -> Self {
        self.colors = colors;
        self
    }

    /// Set context lines.
    pub fn with_context(mut self, lines: usize) -> Self {
        self.context_lines = lines;
        self
    }

    /// Set whether to show line numbers.
    pub fn with_line_numbers(mut self, show: bool) -> Self {
        self.show_line_numbers = show;
        self
    }

    /// Set terminal width.
    pub fn with_width(mut self, width: usize) -> Self {
        self.terminal_width = Some(width);
        self
    }

    /// Get terminal width, auto-detecting if not set.
    fn width(&self) -> usize {
        self.terminal_width.unwrap_or_else(|| {
            Term::stdout()
                .size_checked()
                .map(|(_, w)| w as usize)
                .unwrap_or(80)
        })
    }

    /// Render a text diff to a colored string.
    pub fn render_text(&self, diff: &TextDiff<'_>) -> String {
        match self.style {
            DiffStyle::Unified => self.render_unified(diff),
            DiffStyle::SideBySide => self.render_side_by_side(diff),
            DiffStyle::Inline => self.render_inline(diff),
            DiffStyle::Compact => self.render_compact(diff),
        }
    }

    /// Render a text diff to plain text (no colors).
    pub fn render_plain(&self, diff: &TextDiff<'_>) -> String {
        diff.unified(self.context_lines)
    }

    /// Render a module diff with all sections.
    pub fn render_module(&self, diff: &super::ModuleDiff<'_>) -> String {
        diff.render(self)
    }

    /// Render unified diff format.
    fn render_unified(&self, diff: &TextDiff<'_>) -> String {
        let mut output = String::new();
        let hunks = diff.hunks(self.context_lines);

        for hunk in hunks {
            // Hunk header
            let (old_start, old_count) = hunk.old_range();
            let (new_start, new_count) = hunk.new_range();
            let header = format!(
                "@@ -{},{} +{},{} @@",
                old_start, old_count, new_start, new_count
            );
            writeln!(output, "{}", style(header).fg(self.colors.header)).ok();

            // Changes
            for change in &hunk.changes {
                let (prefix, color) = match change.kind {
                    ChangeKind::Equal => (" ", self.colors.context),
                    ChangeKind::Insert => ("+", self.colors.added),
                    ChangeKind::Delete => ("-", self.colors.removed),
                };

                let line = change.value.trim_end_matches('\n');

                if self.show_line_numbers {
                    let line_num = match change.kind {
                        ChangeKind::Equal | ChangeKind::Delete => change
                            .old_line
                            .map(|n| format!("{:4}", n))
                            .unwrap_or_else(|| "    ".to_string()),
                        ChangeKind::Insert => change
                            .new_line
                            .map(|n| format!("{:4}", n))
                            .unwrap_or_else(|| "    ".to_string()),
                    };
                    write!(output, "{} ", style(line_num).fg(self.colors.line_number)).ok();
                }

                writeln!(
                    output,
                    "{}{}",
                    style(prefix).fg(color),
                    style(line).fg(color)
                )
                .ok();
            }
        }

        output
    }

    /// Render side-by-side diff.
    fn render_side_by_side(&self, diff: &TextDiff<'_>) -> String {
        let mut output = String::new();
        let width = self.width();
        let col_width = (width - 3) / 2; // 3 for separator " │ "

        let old_lines: Vec<&str> = diff.old_text().lines().collect();
        let new_lines: Vec<&str> = diff.new_text().lines().collect();

        // Header
        let header = format!(
            "{:^col_width$} │ {:^col_width$}",
            "Old",
            "New",
            col_width = col_width
        );
        writeln!(output, "{}", style(header).fg(self.colors.header)).ok();
        writeln!(output, "{}", "─".repeat(width)).ok();

        // Simple side-by-side (not perfectly aligned for complex diffs)
        let max_lines = old_lines.len().max(new_lines.len());
        for i in 0..max_lines {
            let old_line = old_lines.get(i).unwrap_or(&"");
            let new_line = new_lines.get(i).unwrap_or(&"");

            let old_truncated = Self::truncate(old_line, col_width);
            let new_truncated = Self::truncate(new_line, col_width);

            let old_styled = if old_lines.get(i).is_some() && !new_lines.contains(old_line) {
                style(format!(
                    "{:<col_width$}",
                    old_truncated,
                    col_width = col_width
                ))
                .fg(self.colors.removed)
            } else {
                style(format!(
                    "{:<col_width$}",
                    old_truncated,
                    col_width = col_width
                ))
                .fg(self.colors.context)
            };

            let new_styled = if new_lines.get(i).is_some() && !old_lines.contains(new_line) {
                style(format!(
                    "{:<col_width$}",
                    new_truncated,
                    col_width = col_width
                ))
                .fg(self.colors.added)
            } else {
                style(format!(
                    "{:<col_width$}",
                    new_truncated,
                    col_width = col_width
                ))
                .fg(self.colors.context)
            };

            writeln!(output, "{} │ {}", old_styled, new_styled).ok();
        }

        output
    }

    /// Render inline diff with word-level highlights.
    fn render_inline(&self, diff: &TextDiff<'_>) -> String {
        let mut output = String::new();

        for change in diff.changes() {
            match change.kind {
                ChangeKind::Equal => {
                    write!(output, "{}", change.value).ok();
                }
                ChangeKind::Insert => {
                    write!(
                        output,
                        "{}",
                        style(change.value).fg(self.colors.added).bold()
                    )
                    .ok();
                }
                ChangeKind::Delete => {
                    write!(
                        output,
                        "{}",
                        style(change.value).fg(self.colors.removed).strikethrough()
                    )
                    .ok();
                }
            }
        }

        output
    }

    /// Render compact diff (only changes).
    fn render_compact(&self, diff: &TextDiff<'_>) -> String {
        let mut output = String::new();

        for change in diff.changes() {
            match change.kind {
                ChangeKind::Equal => continue,
                ChangeKind::Insert => {
                    writeln!(
                        output,
                        "{} {}",
                        style("+").fg(self.colors.added),
                        style(change.value.trim_end()).fg(self.colors.added)
                    )
                    .ok();
                }
                ChangeKind::Delete => {
                    writeln!(
                        output,
                        "{} {}",
                        style("-").fg(self.colors.removed),
                        style(change.value.trim_end()).fg(self.colors.removed)
                    )
                    .ok();
                }
            }
        }

        output
    }

    /// Truncate a string to fit within a given width.
    fn truncate(s: &str, max_width: usize) -> String {
        if s.len() <= max_width {
            s.to_string()
        } else if max_width > 3 {
            format!("{}...", &s[..max_width - 3])
        } else {
            s[..max_width].to_string()
        }
    }

    /// Render diff statistics.
    pub fn render_stats(&self, stats: &DiffStats) -> String {
        let added = style(format!("+{}", stats.lines_added)).fg(self.colors.added);
        let removed = style(format!("-{}", stats.lines_removed)).fg(self.colors.removed);
        format!("{} {}", added, removed)
    }

    /// Render a score change.
    pub fn render_score_change(&self, old_score: f64, new_score: f64) -> String {
        let diff = new_score - old_score;
        let color = if diff > 0.0 {
            self.colors.score_up
        } else if diff < 0.0 {
            self.colors.score_down
        } else {
            self.colors.context
        };

        let sign = if diff > 0.0 { "+" } else { "" };
        format!(
            "{:.2} → {} ({})",
            old_score,
            style(format!("{:.2}", new_score)).fg(color),
            style(format!("{}{:.2}", sign, diff)).fg(color)
        )
    }

    /// Render an iteration header with score.
    pub fn render_iteration_header(
        &self,
        from_iter: u32,
        to_iter: u32,
        old_score: f64,
        new_score: f64,
    ) -> String {
        let width = self.width();

        let iter_label = format!("Iteration {} → {}", from_iter, to_iter);
        let score_label = self.render_score_change(old_score, new_score);

        if self.use_box_drawing {
            let mut output = String::new();
            writeln!(output, "┌{}┐", "─".repeat(width - 2)).ok();
            writeln!(
                output,
                "│ {} │ {} │",
                style(&iter_label).fg(self.colors.header),
                score_label
            )
            .ok();
            writeln!(output, "├{}┤", "─".repeat(width - 2)).ok();
            output
        } else {
            format!(
                "=== {} | Score: {} ===\n",
                style(&iter_label).fg(self.colors.header),
                score_label
            )
        }
    }

    /// Render a section header.
    pub fn render_section_header(&self, title: &str) -> String {
        let width = self.width();

        if self.use_box_drawing {
            format!(
                "│ {} │\n├{}┤\n",
                style(title).fg(self.colors.header).bold(),
                "─".repeat(width - 2)
            )
        } else {
            format!("--- {} ---\n", style(title).fg(self.colors.header).bold())
        }
    }

    /// Render a box around content.
    pub fn render_box(&self, content: &str) -> String {
        let width = self.width();
        let mut output = String::new();

        writeln!(output, "┌{}┐", "─".repeat(width - 2)).ok();
        for line in content.lines() {
            let truncated = Self::truncate(line, width - 4);
            writeln!(output, "│ {:<width$} │", truncated, width = width - 4).ok();
        }
        writeln!(output, "└{}┘", "─".repeat(width - 2)).ok();

        output
    }

    /// Create a style for added content.
    pub fn added_style(&self) -> Style {
        Style::new().fg(self.colors.added)
    }

    /// Create a style for removed content.
    pub fn removed_style(&self) -> Style {
        Style::new().fg(self.colors.removed)
    }

    /// Create a style for context content.
    pub fn context_style(&self) -> Style {
        Style::new().fg(self.colors.context)
    }

    /// Create a style for headers.
    pub fn header_style(&self) -> Style {
        Style::new().fg(self.colors.header).bold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_unified() {
        let old = "hello\nworld\n";
        let new = "hello\nearth\n";
        let diff = TextDiff::new(old, new);
        let renderer = DiffRenderer::new();

        let output = renderer.render_text(&diff);
        assert!(output.contains("-"));
        assert!(output.contains("+"));
    }

    #[test]
    fn test_render_inline() {
        let old = "hello world";
        let new = "hello earth";
        let diff = TextDiff::word_diff(old, new);
        let renderer = DiffRenderer::new().with_style(DiffStyle::Inline);

        let output = renderer.render_text(&diff);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_render_compact() {
        let old = "line1\nline2\n";
        let new = "line1\nmodified\n";
        let diff = TextDiff::new(old, new);
        let renderer = DiffRenderer::new().with_style(DiffStyle::Compact);

        let output = renderer.render_text(&diff);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_render_stats() {
        let stats = DiffStats {
            lines_added: 5,
            lines_removed: 3,
            lines_changed: 3,
            total_changes: 8,
        };
        let renderer = DiffRenderer::new();

        let output = renderer.render_stats(&stats);
        assert!(output.contains("+5"));
        assert!(output.contains("-3"));
    }

    #[test]
    fn test_render_score_change_positive() {
        let renderer = DiffRenderer::new();
        let output = renderer.render_score_change(0.5, 0.8);
        assert!(output.contains("0.50"));
        assert!(output.contains("0.80"));
        assert!(output.contains("+0.30"));
    }

    #[test]
    fn test_render_score_change_negative() {
        let renderer = DiffRenderer::new();
        let output = renderer.render_score_change(0.8, 0.5);
        assert!(output.contains("-0.30"));
    }

    #[test]
    fn test_render_iteration_header() {
        let renderer = DiffRenderer::new().with_width(60);
        let output = renderer.render_iteration_header(1, 2, 0.5, 0.8);
        assert!(output.contains("Iteration 1"));
        assert!(output.contains("2"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(DiffRenderer::truncate("hello", 10), "hello");
        assert_eq!(DiffRenderer::truncate("hello world", 8), "hello...");
        assert_eq!(DiffRenderer::truncate("hi", 2), "hi");
    }

    #[test]
    fn test_default_colors() {
        let colors = DiffColors::default();
        assert_eq!(colors.added, Color::Green);
        assert_eq!(colors.removed, Color::Red);
    }
}
