// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Skills — Persistent Prompt Context.
//!
//! A [`Skill`] is a reusable set of instructions that gets injected into LLM prompts.
//! Instead of hardcoding domain rules in every prompt string, you build a `Skill`
//! once and attach it to any `reason()` call.
//!
//! # Example
//!
//! ```
//! use kkachi::recursive::skill::Skill;
//!
//! let skill = Skill::new()
//!     .instruct("deletionProtection",
//!         "Always set deletionProtection: false on all resources.")
//!     .instruct("naming",
//!         "Use snake_case for all resource names.");
//!
//! assert_eq!(skill.len(), 2);
//! let rendered = skill.render();
//! assert!(rendered.contains("deletionProtection"));
//! assert!(rendered.contains("snake_case"));
//! ```

use smallvec::SmallVec;
use std::borrow::Cow;

/// A reusable instruction that gets injected into LLM prompts.
///
/// Skills provide persistent context — domain rules, coding conventions,
/// or any instruction the LLM should follow across all prompts.
pub struct Skill<'a> {
    entries: SmallVec<[SkillEntry<'a>; 4]>,
}

struct SkillEntry<'a> {
    label: Cow<'a, str>,
    instruction: Cow<'a, str>,
    priority: u8,
}

impl<'a> Skill<'a> {
    /// Create a new empty skill.
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    /// Add an instruction with default priority (128).
    pub fn instruct(self, label: &'a str, instruction: &'a str) -> Self {
        self.instruct_at(label, instruction, 128)
    }

    /// Add an instruction with explicit priority (lower = earlier in prompt).
    pub fn instruct_at(mut self, label: &'a str, instruction: &'a str, priority: u8) -> Self {
        self.entries.push(SkillEntry {
            label: Cow::Borrowed(label),
            instruction: Cow::Borrowed(instruction),
            priority,
        });
        self
    }

    /// Add an instruction from owned strings with default priority (128).
    pub fn instruct_owned(self, label: String, instruction: String) -> Self {
        self.instruct_owned_at(label, instruction, 128)
    }

    /// Add an instruction from owned strings with explicit priority.
    pub fn instruct_owned_at(mut self, label: String, instruction: String, priority: u8) -> Self {
        self.entries.push(SkillEntry {
            label: Cow::Owned(label),
            instruction: Cow::Owned(instruction),
            priority,
        });
        self
    }

    /// Render all instructions into a prompt section.
    ///
    /// Instructions are sorted by priority (lowest first).
    /// Returns an empty string if there are no instructions.
    pub fn render(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        // Sort indices by priority
        let mut indices: SmallVec<[usize; 4]> = (0..self.entries.len()).collect();
        indices.sort_by_key(|&i| self.entries[i].priority);

        let mut out = String::with_capacity(128);
        out.push_str("## Instructions\n");

        for i in indices {
            let entry = &self.entries[i];
            out.push_str("- **");
            out.push_str(&entry.label);
            out.push_str("**: ");
            out.push_str(&entry.instruction);
            out.push('\n');
        }

        out
    }

    /// Number of instructions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if there are no instructions.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<'a> Default for Skill<'a> {
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
    fn test_skill_empty() {
        let skill = Skill::new();
        assert!(skill.is_empty());
        assert_eq!(skill.len(), 0);
        assert_eq!(skill.render(), "");
    }

    #[test]
    fn test_skill_single_instruction() {
        let skill = Skill::new().instruct("naming", "Use snake_case for all names.");

        assert_eq!(skill.len(), 1);
        assert!(!skill.is_empty());

        let rendered = skill.render();
        assert!(rendered.starts_with("## Instructions\n"));
        assert!(rendered.contains("**naming**"));
        assert!(rendered.contains("Use snake_case"));
    }

    #[test]
    fn test_skill_multiple_sorted() {
        let skill = Skill::new()
            .instruct_at("low_priority", "This comes last.", 200)
            .instruct_at("high_priority", "This comes first.", 10)
            .instruct_at("medium_priority", "This comes second.", 100);

        assert_eq!(skill.len(), 3);

        let rendered = skill.render();
        let high_pos = rendered.find("high_priority").unwrap();
        let medium_pos = rendered.find("medium_priority").unwrap();
        let low_pos = rendered.find("low_priority").unwrap();

        assert!(high_pos < medium_pos, "high should come before medium");
        assert!(medium_pos < low_pos, "medium should come before low");
    }

    #[test]
    fn test_skill_cow_borrowed() {
        let label = "test_label";
        let instruction = "test_instruction";
        let skill = Skill::new().instruct(label, instruction);

        // Verify the entry uses borrowed Cow (no heap allocation for the string itself)
        assert_eq!(skill.entries[0].label, Cow::Borrowed("test_label"));
        assert_eq!(
            skill.entries[0].instruction,
            Cow::Borrowed("test_instruction")
        );
    }

    #[test]
    fn test_skill_len() {
        let skill = Skill::new()
            .instruct("a", "instruction a")
            .instruct("b", "instruction b")
            .instruct("c", "instruction c");

        assert_eq!(skill.len(), 3);
    }

    #[test]
    fn test_skill_owned() {
        let skill = Skill::new()
            .instruct_owned("owned_label".to_string(), "owned_instruction".to_string());

        assert_eq!(skill.len(), 1);
        let rendered = skill.render();
        assert!(rendered.contains("owned_label"));
        assert!(rendered.contains("owned_instruction"));
    }

    #[test]
    fn test_skill_default_priority() {
        // All default priority (128) — should preserve insertion order
        let skill = Skill::new()
            .instruct("first", "A")
            .instruct("second", "B")
            .instruct("third", "C");

        let rendered = skill.render();
        let first_pos = rendered.find("first").unwrap();
        let second_pos = rendered.find("second").unwrap();
        let third_pos = rendered.find("third").unwrap();

        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }
}
