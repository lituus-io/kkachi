// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Fuzz-style tests for parsers.
//!
//! These tests use property-based testing patterns to find edge cases
//! in signature parsing and diff algorithms.

use kkachi::diff::TextDiff;
use kkachi::signature::Signature;
use std::collections::HashSet;

// ============================================================================
// Signature Parser Fuzz Tests
// ============================================================================

mod signature_fuzz {
    use super::*;

    /// Generate random valid field names
    fn random_field_names() -> Vec<&'static str> {
        vec![
            "question",
            "answer",
            "context",
            "input",
            "output",
            "query",
            "response",
            "text",
            "result",
            "data",
            "content",
            "summary",
            "reasoning",
            "explanation",
            "code",
            "error",
            "status",
            "value",
            "name",
            "type",
        ]
    }

    #[test]
    fn test_signature_single_field_combinations() {
        let fields = random_field_names();

        for input in &fields {
            for output in &fields {
                let sig_str = format!("{} -> {}", input, output);
                let result = Signature::parse(&sig_str);

                assert!(result.is_ok(), "Should parse: {} -> {}", input, output);

                let sig = result.unwrap();
                assert_eq!(sig.input_fields.len(), 1);
                assert_eq!(sig.output_fields.len(), 1);
            }
        }
    }

    #[test]
    fn test_signature_multi_field_combinations() {
        let fields = random_field_names();

        // Test 2 inputs, 1 output
        for i in 0..fields.len().saturating_sub(2) {
            let sig_str = format!("{}, {} -> {}", fields[i], fields[i + 1], fields[i + 2]);
            let result = Signature::parse(&sig_str);

            if fields[i] != fields[i + 1] {
                // Different field names should parse
                assert!(result.is_ok(), "Should parse: {}", sig_str);
            }
        }

        // Test 1 input, 2 outputs
        for i in 0..fields.len().saturating_sub(2) {
            let sig_str = format!("{} -> {}, {}", fields[i], fields[i + 1], fields[i + 2]);
            let result = Signature::parse(&sig_str);

            if fields[i + 1] != fields[i + 2] {
                assert!(result.is_ok(), "Should parse: {}", sig_str);
            }
        }
    }

    #[test]
    fn test_signature_whitespace_variations() {
        let variations = vec![
            "question->answer",
            "question -> answer",
            "question  ->  answer",
            " question -> answer ",
            "question\t->\tanswer",
            "  question  ,  context  ->  answer  ",
        ];

        for sig_str in variations {
            let result = Signature::parse(sig_str);
            assert!(result.is_ok(), "Should handle whitespace: '{}'", sig_str);
        }
    }

    #[test]
    fn test_signature_invalid_inputs() {
        // Note: The parser is lenient and splits on "->". Many edge cases
        // technically parse but may produce unexpected field names.
        // Only strings that don't contain "->" fail to parse.
        let invalid = vec![
            "",         // Empty - no arrow
            "question", // No arrow
            "abc def",  // No arrow
        ];

        for sig_str in invalid {
            let result = Signature::parse(sig_str);
            assert!(result.is_err(), "Should reject invalid: '{}'", sig_str);
        }

        // These parse but may have unusual field names due to lenient parsing
        let lenient_parses = vec![
            "->",                  // Empty fields (both sides)
            "question ->",         // Empty output
            "-> answer",           // Empty input
            "question --> answer", // Parses as "question -" -> "> answer"
        ];

        for sig_str in lenient_parses {
            let result = Signature::parse(sig_str);
            assert!(
                result.is_ok(),
                "Lenient parsing should accept: '{}'",
                sig_str
            );
        }

        // Multiple arrows create more than 2 parts and fail
        let multi_arrow = "question -> -> answer";
        assert!(
            Signature::parse(multi_arrow).is_err(),
            "Multiple arrows should fail"
        );
    }

    #[test]
    fn test_signature_special_characters() {
        // Field names with underscores should work
        let with_underscores = "user_question -> model_answer";
        assert!(Signature::parse(with_underscores).is_ok());

        // Field names with numbers
        let with_numbers = "input1, input2 -> output1";
        assert!(Signature::parse(with_numbers).is_ok());

        // Mixed case
        let mixed_case = "Question -> Answer";
        assert!(Signature::parse(mixed_case).is_ok());
    }

    #[test]
    fn test_signature_duplicate_fields() {
        // Same field in input and output - should this be allowed?
        let same_field = "question -> question";
        let result = Signature::parse(same_field);
        // May or may not be valid depending on design
        // Just ensure no panic
        let _ = result;

        // Duplicate input fields
        let dup_input = "a, a -> b";
        let result = Signature::parse(dup_input);
        // Should ideally reject, but ensure no panic
        let _ = result;
    }

    #[test]
    fn test_signature_long_field_names() {
        let long_name = "a".repeat(100);
        let sig_str = format!("{} -> {}", long_name, long_name);
        let result = Signature::parse(&sig_str);
        // Should handle long names
        assert!(result.is_ok());
    }

    #[test]
    fn test_signature_many_fields() {
        // Test with many fields
        let inputs: Vec<String> = (0..10).map(|i| format!("input{}", i)).collect();
        let outputs: Vec<String> = (0..5).map(|i| format!("output{}", i)).collect();

        let sig_str = format!("{} -> {}", inputs.join(", "), outputs.join(", "));
        let result = Signature::parse(&sig_str);

        assert!(result.is_ok());
        let sig = result.unwrap();
        assert_eq!(sig.input_fields.len(), 10);
        assert_eq!(sig.output_fields.len(), 5);
    }
}

// ============================================================================
// Diff Algorithm Fuzz Tests
// ============================================================================

mod diff_fuzz {
    use super::*;

    #[test]
    fn test_diff_identical_strings() {
        let texts: Vec<String> = vec![
            "".to_string(),
            "a".to_string(),
            "hello".to_string(),
            "hello world".to_string(),
            "line1\nline2\nline3".to_string(),
            "a".repeat(1000),
        ];

        for text in &texts {
            let diff = TextDiff::new(text, text);
            assert!(!diff.has_changes(), "Identical text should have no diff");
        }
    }

    #[test]
    fn test_diff_completely_different() {
        let pairs = vec![
            ("aaa", "bbb"),
            ("hello", "world"),
            ("12345", "abcde"),
            ("line1\nline2", "other1\nother2"),
        ];

        for (old, new) in pairs {
            let diff = TextDiff::new(old, new);
            assert!(diff.has_changes(), "Different text should have diff");
        }
    }

    #[test]
    fn test_diff_prefix_suffix() {
        // Common prefix
        let diff = TextDiff::new("hello world", "hello rust");
        assert!(diff.has_changes());

        // Common suffix
        let diff = TextDiff::new("hello world", "goodbye world");
        assert!(diff.has_changes());

        // Both
        let diff = TextDiff::new("hello world today", "hello rust today");
        assert!(diff.has_changes());
    }

    #[test]
    fn test_diff_insertions() {
        let old = "line1\nline3";
        let new = "line1\nline2\nline3";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_added >= 1);
    }

    #[test]
    fn test_diff_deletions() {
        let old = "line1\nline2\nline3";
        let new = "line1\nline3";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_removed >= 1);
    }

    #[test]
    fn test_diff_unicode() {
        let old = "Hello 世界";
        let new = "Hello 世界!";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());
    }

    #[test]
    fn test_diff_whitespace_only() {
        let old = "hello world";
        let new = "hello  world";

        let diff = TextDiff::new(old, new);
        // Whitespace changes should be detected
        assert!(diff.has_changes());
    }

    #[test]
    fn test_diff_empty_old() {
        let diff = TextDiff::new("", "new content");
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_added >= 1);
    }

    #[test]
    fn test_diff_empty_new() {
        let diff = TextDiff::new("old content", "");
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_removed >= 1);
    }

    #[test]
    fn test_diff_symmetry() {
        let old = "hello";
        let new = "world";

        let diff1 = TextDiff::new(old, new);
        let diff2 = TextDiff::new(new, old);

        // Both should show changes
        assert!(diff1.has_changes());
        assert!(diff2.has_changes());
    }

    #[test]
    fn test_diff_repeated_patterns() {
        let old = "aaa\nbbb\naaa\nbbb";
        let new = "aaa\nccc\naaa\nbbb";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());
    }

    #[test]
    fn test_diff_large_similar_texts() {
        let base: String = (0..100).map(|i| format!("line{}\n", i)).collect();
        let modified: String = (0..100)
            .map(|i| {
                if i == 50 {
                    "modified50\n".to_string()
                } else {
                    format!("line{}\n", i)
                }
            })
            .collect();

        let diff = TextDiff::new(&base, &modified);
        assert!(diff.has_changes());

        let stats = diff.stats();
        // Should only show minimal changes
        assert!(stats.lines_changed <= 5);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_null_byte_handling() {
        // Text with null bytes
        let with_null = "hello\0world";
        let without_null = "hello world";

        let diff = TextDiff::new(with_null, without_null);
        // Should handle without panic
        assert!(diff.has_changes());
    }

    #[test]
    fn test_very_long_lines() {
        let long_line = "a".repeat(10000);
        let modified = "b".repeat(10000);

        let diff = TextDiff::new(&long_line, &modified);
        assert!(diff.has_changes());
    }

    #[test]
    fn test_many_small_changes() {
        let old: String = (0..1000).map(|i| format!("line{}\n", i)).collect();
        let new: String = (0..1000)
            .map(|i| {
                if i % 10 == 0 {
                    format!("modified{}\n", i)
                } else {
                    format!("line{}\n", i)
                }
            })
            .collect();

        let diff = TextDiff::new(&old, &new);
        assert!(diff.has_changes());
    }

    #[test]
    fn test_crlf_vs_lf() {
        let unix = "line1\nline2\nline3";
        let windows = "line1\r\nline2\r\nline3";

        let diff = TextDiff::new(unix, windows);
        // Line endings should be detected as changes
        assert!(diff.has_changes());
    }

    #[test]
    fn test_trailing_newline() {
        let with_newline = "content\n";
        let without = "content";

        let diff = TextDiff::new(with_newline, without);
        assert!(diff.has_changes());
    }

    #[test]
    fn test_only_newlines() {
        let one = "\n";
        let many = "\n\n\n";

        let diff = TextDiff::new(one, many);
        assert!(diff.has_changes());
    }
}

// ============================================================================
// Consistency Tests
// ============================================================================

mod consistency {
    use super::*;

    #[test]
    fn test_signature_roundtrip() {
        let original = "question, context -> answer, reasoning";
        let parsed = Signature::parse(original).unwrap();

        // Verify field preservation
        assert_eq!(parsed.input_fields.len(), 2);
        assert_eq!(parsed.output_fields.len(), 2);

        // Field names should be preserved
        let input_names: HashSet<&str> = parsed
            .input_fields
            .iter()
            .map(|f| f.name.as_ref())
            .collect();
        assert!(input_names.contains("question"));
        assert!(input_names.contains("context"));

        let output_names: HashSet<&str> = parsed
            .output_fields
            .iter()
            .map(|f| f.name.as_ref())
            .collect();
        assert!(output_names.contains("answer"));
        assert!(output_names.contains("reasoning"));
    }

    #[test]
    fn test_diff_transitivity() {
        let a = "version1";
        let b = "version2";
        let c = "version1"; // Same as a

        let diff_ab = TextDiff::new(a, b);
        let diff_bc = TextDiff::new(b, c);
        let diff_ac = TextDiff::new(a, c);

        // a != b, so diff
        assert!(diff_ab.has_changes());
        // b != c, so diff
        assert!(diff_bc.has_changes());
        // a == c, so no diff
        assert!(!diff_ac.has_changes());
    }
}
