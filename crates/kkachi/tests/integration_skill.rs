// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for the Skill system.

use kkachi::recursive::{checks, reason, MockLlm, Skill};

#[test]
fn test_skill_with_reason() {
    // Verify skill instructions appear in the prompt sent to the LLM.
    let llm = MockLlm::new(|prompt, _| {
        if prompt.contains("## Instructions") && prompt.contains("deletionProtection") {
            "Answer: skill present".to_string()
        } else {
            "Answer: no skill".to_string()
        }
    });

    let skill = Skill::new().instruct(
        "deletionProtection",
        "Always set deletionProtection: false.",
    );

    let result = reason(&llm, "Generate config").skill(&skill).go();

    assert_eq!(result.output, "skill present");
    assert!(result.success());
}

#[test]
fn test_skill_with_defaults_combined() {
    use kkachi::recursive::Defaults;

    let llm = MockLlm::new(|prompt, _| {
        if prompt.contains("snake_case") {
            "user:admin@example.com".to_string()
        } else {
            "no skill".to_string()
        }
    });

    let skill = Skill::new().instruct("naming", "Use snake_case for all names.");

    let defaults = Defaults::new().set("email", r"admin@example\.com", "real@company.com");

    let result = reason(&llm, "Generate config")
        .skill(&skill)
        .defaults(defaults)
        .validate(checks().require("real@company.com"))
        .go();

    assert!(result.success());
    assert!(result.output.contains("real@company.com"));
}

#[test]
fn test_skill_empty_noop() {
    let llm = MockLlm::new(|prompt, _| {
        // Empty skill should not add "## Instructions" to the prompt
        if prompt.contains("## Instructions") {
            "Answer: instructions found (bad)".to_string()
        } else {
            "Answer: clean prompt".to_string()
        }
    });

    let skill = Skill::new();

    let result = reason(&llm, "Generate something").skill(&skill).go();

    assert_eq!(result.output, "clean prompt");
}

#[test]
fn test_skill_priority_ordering_in_prompt() {
    let llm = MockLlm::new(|prompt, _| {
        let high_pos = prompt.find("high_priority").unwrap_or(usize::MAX);
        let low_pos = prompt.find("low_priority").unwrap_or(usize::MAX);
        if high_pos < low_pos {
            "Answer: correct order".to_string()
        } else {
            "Answer: wrong order".to_string()
        }
    });

    let skill = Skill::new()
        .instruct_at("low_priority", "Low priority instruction.", 200)
        .instruct_at("high_priority", "High priority instruction.", 10);

    let result = reason(&llm, "Test ordering").skill(&skill).go();
    assert_eq!(result.output, "correct order");
}
