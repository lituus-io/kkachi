// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Property-based fuzz tests for validators and core operations.
//!
//! Uses proptest to verify invariants hold across arbitrary inputs.

use proptest::prelude::*;

use kkachi::recursive::validate::Validate;
use kkachi::recursive::{
    all, any, checks, extract_all_code, extract_code, extract_section, rewrite, MockLlm, Template,
};

// ============================================================================
// Checks Validator Properties
// ============================================================================

proptest! {
    /// Checks validators never panic on arbitrary input.
    #[test]
    fn checks_require_never_panic(text in "\\PC{0,500}") {
        let v = checks().require("x").require("fn ");
        let _ = v.validate(&text);
    }

    /// Checks validators with forbid never panic.
    #[test]
    fn checks_forbid_never_panic(text in "\\PC{0,500}") {
        let v = checks().forbid("unsafe").forbid("panic!");
        let _ = v.validate(&text);
    }

    /// Checks validators with require_all never panic.
    #[test]
    fn checks_require_all_never_panic(text in "\\PC{0,500}") {
        let v = checks().require_all(["fn ", "->", "{"]);
        let _ = v.validate(&text);
    }

    /// Score is always in range [0.0, 1.0].
    #[test]
    fn checks_score_bounded(text in "\\PC{0,500}") {
        let v = checks()
            .require("x")
            .forbid("y")
            .min_len(10);
        let score = v.validate(&text);
        prop_assert!(score.value >= 0.0, "Score {} < 0.0", score.value);
        prop_assert!(score.value <= 1.0, "Score {} > 1.0", score.value);
    }

    /// Checks with min_len never panic.
    #[test]
    fn checks_min_len_never_panic(text in "\\PC{0,500}", min_len in 0usize..1000) {
        let v = checks().min_len(min_len);
        let score = v.validate(&text);
        prop_assert!(score.value >= 0.0 && score.value <= 1.0);
    }

    /// Empty checks always return score 1.0.
    #[test]
    fn empty_checks_always_pass(text in "\\PC{0,500}") {
        let v = checks();
        let score = v.validate(&text);
        prop_assert_eq!(score.value, 1.0);
    }
}

// ============================================================================
// Compose Validator Properties
// ============================================================================

proptest! {
    /// Composed all() validators never panic.
    #[test]
    fn compose_all_never_panic(text in "\\PC{0,500}") {
        let v1 = checks().require("a");
        let v2 = checks().forbid("b");
        let composed = all([v1, v2]);
        let score = composed.validate(&text);
        prop_assert!(score.value >= 0.0 && score.value <= 1.0);
    }

    /// Composed any() validators never panic.
    #[test]
    fn compose_any_never_panic(text in "\\PC{0,500}") {
        let v1 = checks().require("a");
        let v2 = checks().require("b");
        let composed = any([v1, v2]);
        let score = composed.validate(&text);
        prop_assert!(score.value >= 0.0 && score.value <= 1.0);
    }

    /// all([v1, v2]) score <= min(v1.score, v2.score).
    #[test]
    fn compose_all_is_min(text in "\\PC{0,500}") {
        let v1 = checks().require("a");
        let v2 = checks().require("b");
        let s1 = v1.validate(&text).value;
        let s2 = v2.validate(&text).value;
        let composed = all([
            checks().require("a"),
            checks().require("b"),
        ]);
        let s_all = composed.validate(&text).value;
        prop_assert!(s_all <= s1 + f64::EPSILON, "all score {} > v1 score {}", s_all, s1);
        prop_assert!(s_all <= s2 + f64::EPSILON, "all score {} > v2 score {}", s_all, s2);
    }

    /// any([v1, v2]) score >= max(v1.score, v2.score).
    #[test]
    fn compose_any_is_max(text in "\\PC{0,500}") {
        let v1 = checks().require("a");
        let v2 = checks().require("b");
        let s1 = v1.validate(&text).value;
        let s2 = v2.validate(&text).value;
        let composed = any([
            checks().require("a"),
            checks().require("b"),
        ]);
        let s_any = composed.validate(&text).value;
        prop_assert!(s_any >= s1 - f64::EPSILON, "any score {} < v1 score {}", s_any, s1);
        prop_assert!(s_any >= s2 - f64::EPSILON, "any score {} < v2 score {}", s_any, s2);
    }
}

// ============================================================================
// Extract Code Properties
// ============================================================================

proptest! {
    /// extract_code never panics on arbitrary input.
    #[test]
    fn extract_code_never_panic(text in "\\PC{0,500}", lang in "[a-z]{0,10}") {
        let _ = extract_code(&text, &lang);
    }

    /// extract_all_code never panics on arbitrary input.
    #[test]
    fn extract_all_code_never_panic(text in "\\PC{0,500}", lang in "[a-z]{0,10}") {
        let _ = extract_all_code(&text, &lang);
    }

    /// extract_code returns None for text without code fences.
    #[test]
    fn extract_code_no_fences(text in "[^`]{0,200}") {
        let result = extract_code(&text, "rust");
        // If the text contains no backticks, there can't be code fences
        if !text.contains("```") {
            prop_assert!(result.is_none());
        }
    }

    /// extract_code result is a substring of the input (when Some).
    #[test]
    fn extract_code_is_substring(text in "\\PC{0,500}", lang in "[a-z]{1,5}") {
        if let Some(code) = extract_code(&text, &lang) {
            prop_assert!(
                text.contains(code),
                "Extracted code is not a substring of input"
            );
        }
    }
}

// ============================================================================
// Rewrite Properties
// ============================================================================

proptest! {
    /// rewrite with section never panics on arbitrary input.
    #[test]
    fn rewrite_section_never_panic(
        text in "\\PC{0,300}",
        title in "[a-zA-Z ]{1,20}",
        replacement in "\\PC{0,100}"
    ) {
        let _ = rewrite(&text).section(&title, &replacement).build();
    }

    /// extract_section never panics on arbitrary input.
    #[test]
    fn extract_section_never_panic(
        text in "\\PC{0,300}",
        title in "[a-zA-Z ]{1,20}"
    ) {
        let _ = extract_section(&text, &title);
    }
}

// ============================================================================
// Template Properties
// ============================================================================

proptest! {
    /// Template::new never panics on arbitrary strings.
    #[test]
    fn template_new_never_panic(name in "\\PC{0,100}") {
        let _ = Template::new(&name);
    }

    /// Template rendering never panics with arbitrary input.
    #[test]
    fn template_render_never_panic(
        input in "\\PC{0,200}"
    ) {
        let t = Template::simple(&input);
        let _ = t.render(&input);
    }
}

// ============================================================================
// MockLlm + Refine Properties
// ============================================================================

proptest! {
    /// Refine with any validator never panics (up to 2 iterations for speed).
    #[test]
    fn refine_never_panic(prompt in "\\PC{1,100}") {
        let llm = MockLlm::new(|p, _| format!("response to: {}", p));
        let v = checks().require("response");
        let result = kkachi::recursive::refine(&llm, &prompt)
            .validate(v)
            .max_iter(2)
            .go();
        prop_assert!(result.is_ok());
    }
}
