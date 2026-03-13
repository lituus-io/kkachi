// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Structured diffs for module changes.
//!
//! Provides diffs for:
//! - Module configuration changes (instructions, demos, fields)
//! - Few-shot demonstration changes
//! - Optimization iteration changes

use super::{DiffRenderer, TextDiff};
use std::fmt::Write;

/// Snapshot of a demonstration example.
#[derive(Debug, Clone)]
pub struct DemoSnapshot<'a> {
    /// Index in the demo list.
    pub index: usize,
    /// Input text.
    pub input: &'a str,
    /// Output text.
    pub output: &'a str,
}

impl<'a> DemoSnapshot<'a> {
    /// Create a new demo snapshot.
    pub fn new(index: usize, input: &'a str, output: &'a str) -> Self {
        Self {
            index,
            input,
            output,
        }
    }

    /// Format as a display string.
    pub fn display(&self) -> String {
        let input_preview: String = self.input.chars().take(50).collect();
        let output_preview: String = self.output.chars().take(50).collect();
        format!(
            "[{}] Q: {}{}  A: {}{}",
            self.index,
            input_preview,
            if self.input.len() > 50 { "..." } else { "" },
            output_preview,
            if self.output.len() > 50 { "..." } else { "" }
        )
    }
}

/// Diff for few-shot demonstrations.
#[derive(Debug, Clone, Default)]
pub struct DemosDiff<'a> {
    /// Newly added demos.
    pub added: Vec<DemoSnapshot<'a>>,
    /// Removed demos.
    pub removed: Vec<DemoSnapshot<'a>>,
    /// Modified demos (old, new).
    pub modified: Vec<(DemoSnapshot<'a>, DemoSnapshot<'a>)>,
    /// Whether demos were reordered.
    pub reordered: bool,
}

impl<'a> DemosDiff<'a> {
    /// Create a new empty demos diff.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are any changes.
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty()
            || !self.removed.is_empty()
            || !self.modified.is_empty()
            || self.reordered
    }

    /// Get total number of changes.
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }

    /// Add an added demo.
    pub fn add_added(&mut self, demo: DemoSnapshot<'a>) {
        self.added.push(demo);
    }

    /// Add a removed demo.
    pub fn add_removed(&mut self, demo: DemoSnapshot<'a>) {
        self.removed.push(demo);
    }

    /// Add a modified demo pair.
    pub fn add_modified(&mut self, old: DemoSnapshot<'a>, new: DemoSnapshot<'a>) {
        self.modified.push((old, new));
    }

    /// Compute diff between two lists of demos.
    pub fn compute(old_demos: &'a [(String, String)], new_demos: &'a [(String, String)]) -> Self {
        let mut diff = Self::new();

        // Build maps for quick lookup
        let old_map: std::collections::HashMap<&str, (usize, &str)> = old_demos
            .iter()
            .enumerate()
            .map(|(i, (input, output))| (input.as_str(), (i, output.as_str())))
            .collect();

        let new_map: std::collections::HashMap<&str, (usize, &str)> = new_demos
            .iter()
            .enumerate()
            .map(|(i, (input, output))| (input.as_str(), (i, output.as_str())))
            .collect();

        // Find removed demos
        for (i, (input, output)) in old_demos.iter().enumerate() {
            if !new_map.contains_key(input.as_str()) {
                diff.add_removed(DemoSnapshot::new(i, input, output));
            }
        }

        // Find added and modified demos
        for (i, (input, output)) in new_demos.iter().enumerate() {
            if let Some((old_idx, old_output)) = old_map.get(input.as_str()) {
                if *old_output != output.as_str() {
                    diff.add_modified(
                        DemoSnapshot::new(*old_idx, input, old_output),
                        DemoSnapshot::new(i, input, output),
                    );
                } else if *old_idx != i {
                    diff.reordered = true;
                }
            } else {
                diff.add_added(DemoSnapshot::new(i, input, output));
            }
        }

        diff
    }

    /// Render the demos diff.
    pub fn render(&self, renderer: &DiffRenderer) -> String {
        let mut output = String::new();

        if !self.has_changes() {
            return output;
        }

        writeln!(output, "{}", renderer.render_section_header("DEMOS")).ok();

        // Summary
        let mut summary_parts = Vec::new();
        if !self.added.is_empty() {
            summary_parts.push(format!("+{} added", self.added.len()));
        }
        if !self.removed.is_empty() {
            summary_parts.push(format!("-{} removed", self.removed.len()));
        }
        if !self.modified.is_empty() {
            summary_parts.push(format!("~{} modified", self.modified.len()));
        }
        if self.reordered {
            summary_parts.push("reordered".to_string());
        }
        writeln!(output, "  ({})", summary_parts.join(", ")).ok();
        writeln!(output).ok();

        // Added demos
        for demo in &self.added {
            writeln!(
                output,
                "  {} {}",
                console::style("+").green(),
                console::style(demo.display()).green()
            )
            .ok();
        }

        // Removed demos
        for demo in &self.removed {
            writeln!(
                output,
                "  {} {}",
                console::style("-").red(),
                console::style(demo.display()).red()
            )
            .ok();
        }

        // Modified demos
        for (old, new) in &self.modified {
            writeln!(
                output,
                "  {} {} → {}",
                console::style("~").yellow(),
                console::style(old.display()).red(),
                console::style(new.display()).green()
            )
            .ok();
        }

        output
    }
}

/// Diff for field definitions.
#[derive(Debug, Clone, Default)]
pub struct FieldsDiff<'a> {
    /// Added fields.
    pub added: Vec<&'a str>,
    /// Removed fields.
    pub removed: Vec<&'a str>,
    /// Modified field descriptions.
    pub modified: Vec<(&'a str, &'a str, &'a str)>, // (name, old_desc, new_desc)
}

impl<'a> FieldsDiff<'a> {
    /// Create a new empty fields diff.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are any changes.
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.modified.is_empty()
    }

    /// Render the fields diff.
    pub fn render(&self, renderer: &DiffRenderer) -> String {
        let mut output = String::new();

        if !self.has_changes() {
            return output;
        }

        writeln!(output, "{}", renderer.render_section_header("FIELDS")).ok();

        for field in &self.added {
            writeln!(
                output,
                "  {} {}",
                console::style("+").green(),
                console::style(field).green()
            )
            .ok();
        }

        for field in &self.removed {
            writeln!(
                output,
                "  {} {}",
                console::style("-").red(),
                console::style(field).red()
            )
            .ok();
        }

        for (name, old_desc, new_desc) in &self.modified {
            writeln!(
                output,
                "  {} {}: {} → {}",
                console::style("~").yellow(),
                name,
                console::style(old_desc).red(),
                console::style(new_desc).green()
            )
            .ok();
        }

        output
    }
}

/// Complete diff for a module/iteration change.
#[derive(Debug)]
pub struct ModuleDiff<'a> {
    /// Instruction/system prompt changes.
    pub instruction: Option<TextDiff<'a>>,
    /// Few-shot demo changes.
    pub demos: DemosDiff<'a>,
    /// Output/answer changes (for refinement iterations).
    pub output: Option<TextDiff<'a>>,
    /// Field definition changes.
    pub fields: Option<FieldsDiff<'a>>,
    /// Iteration numbers (from, to).
    pub iterations: Option<(u32, u32)>,
    /// Score change (from, to).
    pub scores: Option<(f64, f64)>,
    /// Feedback message.
    pub feedback: Option<&'a str>,
}

impl<'a> ModuleDiff<'a> {
    /// Create a new empty module diff.
    pub fn new() -> Self {
        Self {
            instruction: None,
            demos: DemosDiff::new(),
            output: None,
            fields: None,
            iterations: None,
            scores: None,
            feedback: None,
        }
    }

    /// Set instruction diff.
    pub fn with_instruction(mut self, old: &'a str, new: &'a str) -> Self {
        if old != new {
            self.instruction = Some(TextDiff::new(old, new));
        }
        self
    }

    /// Set demos diff.
    pub fn with_demos(mut self, demos: DemosDiff<'a>) -> Self {
        self.demos = demos;
        self
    }

    /// Set output diff.
    pub fn with_output(mut self, old: &'a str, new: &'a str) -> Self {
        if old != new {
            self.output = Some(TextDiff::new(old, new));
        }
        self
    }

    /// Set fields diff.
    pub fn with_fields(mut self, fields: FieldsDiff<'a>) -> Self {
        self.fields = Some(fields);
        self
    }

    /// Set iteration info.
    pub fn with_iterations(mut self, from: u32, to: u32) -> Self {
        self.iterations = Some((from, to));
        self
    }

    /// Set score info.
    pub fn with_scores(mut self, from: f64, to: f64) -> Self {
        self.scores = Some((from, to));
        self
    }

    /// Set feedback message.
    pub fn with_feedback(mut self, feedback: &'a str) -> Self {
        self.feedback = Some(feedback);
        self
    }

    /// Check if there are any changes.
    pub fn has_changes(&self) -> bool {
        self.instruction
            .as_ref()
            .map(|d| d.has_changes())
            .unwrap_or(false)
            || self.demos.has_changes()
            || self
                .output
                .as_ref()
                .map(|d| d.has_changes())
                .unwrap_or(false)
            || self
                .fields
                .as_ref()
                .map(|f| f.has_changes())
                .unwrap_or(false)
    }

    /// Render the full module diff.
    pub fn render(&self, renderer: &DiffRenderer) -> String {
        let mut output = String::new();

        // Header with iteration and score
        if let (Some((from_iter, to_iter)), Some((from_score, to_score))) =
            (self.iterations, self.scores)
        {
            output.push_str(
                &renderer.render_iteration_header(from_iter, to_iter, from_score, to_score),
            );
        }

        // Instruction changes
        if let Some(ref diff) = self.instruction {
            if diff.has_changes() {
                output.push_str(&renderer.render_section_header("INSTRUCTION"));
                output.push_str(&renderer.render_text(diff));
                output.push('\n');
            }
        }

        // Demo changes
        if self.demos.has_changes() {
            output.push_str(&self.demos.render(renderer));
            output.push('\n');
        }

        // Output changes
        if let Some(ref diff) = self.output {
            if diff.has_changes() {
                output.push_str(&renderer.render_section_header("OUTPUT"));
                output.push_str(&renderer.render_text(diff));
                output.push('\n');
            }
        }

        // Field changes
        if let Some(ref fields) = self.fields {
            if fields.has_changes() {
                output.push_str(&fields.render(renderer));
                output.push('\n');
            }
        }

        // Feedback
        if let Some(feedback) = self.feedback {
            output.push_str(&renderer.render_section_header("FEEDBACK"));
            writeln!(output, "  {}", console::style(feedback).yellow()).ok();
            output.push('\n');
        }

        // Footer
        if renderer.use_box_drawing {
            let width = renderer.terminal_width.unwrap_or(80);
            writeln!(output, "└{}┘", "─".repeat(width - 2)).ok();
        }

        output
    }
}

impl Default for ModuleDiff<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating iteration diffs during optimization.
pub struct IterationDiffBuilder<'a> {
    diff: ModuleDiff<'a>,
}

impl<'a> IterationDiffBuilder<'a> {
    /// Start building a new iteration diff.
    pub fn new(from_iter: u32, to_iter: u32) -> Self {
        Self {
            diff: ModuleDiff::new().with_iterations(from_iter, to_iter),
        }
    }

    /// Set the scores.
    pub fn scores(mut self, from: f64, to: f64) -> Self {
        self.diff = self.diff.with_scores(from, to);
        self
    }

    /// Set the instruction change.
    pub fn instruction(mut self, old: &'a str, new: &'a str) -> Self {
        self.diff = self.diff.with_instruction(old, new);
        self
    }

    /// Set the output change.
    pub fn output(mut self, old: &'a str, new: &'a str) -> Self {
        self.diff = self.diff.with_output(old, new);
        self
    }

    /// Set the demos change.
    pub fn demos(mut self, old: &'a [(String, String)], new: &'a [(String, String)]) -> Self {
        self.diff = self.diff.with_demos(DemosDiff::compute(old, new));
        self
    }

    /// Set the feedback.
    pub fn feedback(mut self, feedback: &'a str) -> Self {
        self.diff = self.diff.with_feedback(feedback);
        self
    }

    /// Build the final diff.
    pub fn build(self) -> ModuleDiff<'a> {
        self.diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_snapshot_display() {
        let demo = DemoSnapshot::new(0, "What is 2+2?", "4");
        let display = demo.display();
        assert!(display.contains("[0]"));
        assert!(display.contains("What is 2+2?"));
        assert!(display.contains("4"));
    }

    #[test]
    fn test_demos_diff_compute_added() {
        let old: Vec<(String, String)> = vec![];
        let new = vec![("Q1".to_string(), "A1".to_string())];

        let diff = DemosDiff::compute(&old, &new);
        assert!(diff.has_changes());
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 0);
    }

    #[test]
    fn test_demos_diff_compute_removed() {
        let old = vec![("Q1".to_string(), "A1".to_string())];
        let new: Vec<(String, String)> = vec![];

        let diff = DemosDiff::compute(&old, &new);
        assert!(diff.has_changes());
        assert_eq!(diff.added.len(), 0);
        assert_eq!(diff.removed.len(), 1);
    }

    #[test]
    fn test_demos_diff_compute_modified() {
        let old = vec![("Q1".to_string(), "A1".to_string())];
        let new = vec![("Q1".to_string(), "A1 modified".to_string())];

        let diff = DemosDiff::compute(&old, &new);
        assert!(diff.has_changes());
        assert_eq!(diff.modified.len(), 1);
    }

    #[test]
    fn test_module_diff_no_changes() {
        let diff = ModuleDiff::new();
        assert!(!diff.has_changes());
    }

    #[test]
    fn test_module_diff_with_instruction() {
        let diff = ModuleDiff::new().with_instruction("old instruction", "new instruction");
        assert!(diff.has_changes());
    }

    #[test]
    fn test_module_diff_render() {
        let diff = ModuleDiff::new()
            .with_iterations(1, 2)
            .with_scores(0.5, 0.8)
            .with_output("old output", "new output");

        let renderer = DiffRenderer::new().with_width(60);
        let output = diff.render(&renderer);

        assert!(!output.is_empty());
        assert!(output.contains("Iteration 1"));
    }

    #[test]
    fn test_iteration_diff_builder() {
        let old_demos = vec![("Q1".to_string(), "A1".to_string())];
        let new_demos = vec![
            ("Q1".to_string(), "A1".to_string()),
            ("Q2".to_string(), "A2".to_string()),
        ];

        let diff = IterationDiffBuilder::new(0, 1)
            .scores(0.5, 0.8)
            .instruction("You are helpful", "You are a helpful assistant")
            .demos(&old_demos, &new_demos)
            .feedback("Added more examples")
            .build();

        assert!(diff.has_changes());
        assert!(diff.demos.has_changes());
    }
}
