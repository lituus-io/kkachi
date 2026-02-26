// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the Skill type.

use pyo3::prelude::*;

use kkachi::recursive::skill::Skill;

/// A reusable set of instructions injected into LLM prompts.
///
/// Example:
///     skill = Skill() \
///         .instruct("naming", "Use snake_case for all names.") \
///         .instruct("policy", "Always set deletionProtection: false.")
///
///     result = reason(llm, "Generate config").skill(skill).go()
#[pyclass(name = "Skill")]
#[derive(Clone)]
pub struct PySkill {
    entries: Vec<(String, String, u8)>,
}

#[pymethods]
impl PySkill {
    /// Create a new empty skill.
    #[new]
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an instruction with default priority (128).
    fn instruct(&self, label: String, instruction: String) -> Self {
        self.instruct_at(label, instruction, 128)
    }

    /// Add an instruction with explicit priority (lower = earlier in prompt).
    #[pyo3(signature = (label, instruction, priority=128))]
    fn instruct_at(&self, label: String, instruction: String, priority: u8) -> Self {
        let mut new = self.clone();
        new.entries.push((label, instruction, priority));
        new
    }

    /// Render all instructions into a prompt section string.
    fn render(&self) -> String {
        self.to_skill().render()
    }

    /// Number of instructions.
    fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// Check if there are no instructions.
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("Skill(instructions={})", self.entries.len())
    }
}

impl PySkill {
    /// Convert to a Rust Skill (owned).
    pub fn to_skill(&self) -> Skill<'static> {
        let mut skill = Skill::new();
        for (label, instruction, priority) in &self.entries {
            skill = skill.instruct_owned_at(label.clone(), instruction.clone(), *priority);
        }
        skill
    }
}
