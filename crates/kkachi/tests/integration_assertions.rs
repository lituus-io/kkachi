// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for assertions system
//!
//! Tests validation of module outputs using assertions.

use kkachi::*;

#[test]
fn test_assertion_runner_all_pass() {
    let field = sym("answer");

    let not_empty = NotEmpty::assert();
    let min_len = LengthBounds::min(3, AssertionLevel::Assert);

    let assertions: &[&dyn Assertion] = &[&not_empty, &min_len];
    let runner = AssertionRunner::new(assertions);

    // Valid value
    let value = StrView::new("hello world");
    let results = runner.run(field, value).unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.passed));
}

#[test]
fn test_assertion_runner_hard_failure() {
    let field = sym("answer");

    let not_empty = NotEmpty::assert();
    let assertions: &[&dyn Assertion] = &[&not_empty];
    let runner = AssertionRunner::new(assertions);

    // Empty value should fail
    let value = StrView::new("");
    let result = runner.run(field, value);

    assert!(result.is_err());
}

#[test]
fn test_assertion_runner_soft_suggestions() {
    let field = sym("answer");

    let not_empty = NotEmpty::suggest();
    let min_len = LengthBounds::min(100, AssertionLevel::Suggest);

    let assertions: &[&dyn Assertion] = &[&not_empty, &min_len];
    let runner = AssertionRunner::new(assertions);

    // Short value - should collect warnings but not fail
    let value = StrView::new("short");
    let results = runner.run_soft(field, value);

    assert_eq!(results.len(), 2);
    assert!(results[0].passed); // not empty
    assert!(!results[1].passed); // too short (suggestion, not failure)
}

#[test]
fn test_length_bounds_validation() {
    let field = sym("code");

    // Between 10 and 1000 characters
    let bounds = LengthBounds::between(10, 1000, AssertionLevel::Assert);

    // Too short
    assert!(!bounds.check(field, StrView::new("short")).passed);

    // Just right
    assert!(
        bounds
            .check(field, StrView::new("this is a valid length"))
            .passed
    );

    // Just at minimum
    assert!(bounds.check(field, StrView::new("0123456789")).passed);
}

#[test]
fn test_contains_validation() {
    let field = sym("citation");

    let must_cite = Contains::assert("Source:");

    assert!(
        must_cite
            .check(field, StrView::new("Source: Wikipedia"))
            .passed
    );
    assert!(
        !must_cite
            .check(field, StrView::new("No citation here"))
            .passed
    );
}

#[test]
fn test_regex_validation() {
    let field = sym("email");

    // Simple email pattern
    let email_regex = RegexMatch::assert(r"^[\w.-]+@[\w.-]+\.\w+$");

    assert!(
        email_regex
            .check(field, StrView::new("test@example.com"))
            .passed
    );
    assert!(
        email_regex
            .check(field, StrView::new("user.name@domain.org"))
            .passed
    );
    assert!(
        !email_regex
            .check(field, StrView::new("invalid-email"))
            .passed
    );
    assert!(
        !email_regex
            .check(field, StrView::new("@no-user.com"))
            .passed
    );
}

#[test]
fn test_json_valid_assertion() {
    let field = sym("data");

    let json_check = JsonValid::assert();

    // Valid JSON
    assert!(
        json_check
            .check(field, StrView::new(r#"{"key": "value"}"#))
            .passed
    );
    assert!(json_check.check(field, StrView::new(r#"[1, 2, 3]"#)).passed);
    assert!(
        json_check
            .check(field, StrView::new(r#"{"nested": {"a": 1}}"#))
            .passed
    );

    // Invalid JSON
    assert!(!json_check.check(field, StrView::new("not json")).passed);
    assert!(!json_check.check(field, StrView::new("{unclosed")).passed);
    assert!(
        !json_check
            .check(field, StrView::new(r#"{"missing": }"#))
            .passed
    );
}

#[test]
fn test_starts_ends_with_validation() {
    let field = sym("response");

    let starts = StartsWith::assert("BEGIN:");
    let ends = EndsWith::assert(":END");

    assert!(
        starts
            .check(field, StrView::new("BEGIN: content here"))
            .passed
    );
    assert!(!starts.check(field, StrView::new("No prefix")).passed);

    assert!(ends.check(field, StrView::new("content here:END")).passed);
    assert!(!ends.check(field, StrView::new("No suffix")).passed);
}

#[test]
fn test_one_of_validation() {
    let field = sym("status");

    let valid_statuses = OneOf::assert(&["pending", "approved", "rejected"]);

    assert!(valid_statuses.check(field, StrView::new("pending")).passed);
    assert!(valid_statuses.check(field, StrView::new("approved")).passed);
    assert!(
        valid_statuses
            .check(field, StrView::new("  rejected  "))
            .passed
    ); // trimmed
    assert!(!valid_statuses.check(field, StrView::new("unknown")).passed);
}

#[test]
fn test_custom_assertion() {
    let field = sym("number");

    // Custom: must be a positive integer
    let positive_int = Custom::new(
        |s| s.trim().parse::<i32>().is_ok_and(|n| n > 0),
        "must be a positive integer",
        AssertionLevel::Assert,
    );

    assert!(positive_int.check(field, StrView::new("42")).passed);
    assert!(positive_int.check(field, StrView::new("  100  ")).passed);
    assert!(!positive_int.check(field, StrView::new("-5")).passed);
    assert!(
        !positive_int
            .check(field, StrView::new("not a number"))
            .passed
    );
}

#[test]
fn test_chained_assertions() {
    let field = sym("code");

    // Code must be non-empty, JSON, and contain a specific key
    let not_empty = NotEmpty::assert();
    let json_valid = JsonValid::assert();
    let has_key = Contains::assert(r#""result""#);

    let assertions: &[&dyn Assertion] = &[&not_empty, &json_valid, &has_key];
    let runner = AssertionRunner::new(assertions);

    // Valid
    let valid = StrView::new(r#"{"result": 42}"#);
    assert!(runner.all_pass(field, valid));

    // Missing key
    let missing_key = StrView::new(r#"{"other": 42}"#);
    assert!(!runner.all_pass(field, missing_key));

    // Invalid JSON
    let invalid_json = StrView::new("not json");
    assert!(!runner.all_pass(field, invalid_json));
}

#[test]
fn test_assertion_count_passing() {
    let field = sym("answer");

    let check1 = LengthBounds::min(5, AssertionLevel::Suggest);
    let check2 = LengthBounds::min(10, AssertionLevel::Suggest);
    let check3 = LengthBounds::min(20, AssertionLevel::Suggest);

    let assertions: &[&dyn Assertion] = &[&check1, &check2, &check3];
    let runner = AssertionRunner::new(assertions);

    // 5 chars - passes 1
    assert_eq!(runner.count_passing(field, StrView::new("hello")), 1);

    // 12 chars - passes 2
    assert_eq!(runner.count_passing(field, StrView::new("hello world!")), 2);

    // 25 chars - passes all 3
    assert_eq!(
        runner.count_passing(field, StrView::new("this is a longer sentence")),
        3
    );
}

#[test]
fn test_assertion_levels() {
    // Assert level
    let hard = NotEmpty::assert();
    assert_eq!(hard.level(), AssertionLevel::Assert);

    // Suggest level
    let soft = NotEmpty::suggest();
    assert_eq!(soft.level(), AssertionLevel::Suggest);
}

#[test]
fn test_assertion_result_hard_failure_detection() {
    let field = sym("test");

    // Hard failure
    let hard_fail = AssertionResult::fail(field, AssertionLevel::Assert);
    assert!(hard_fail.is_hard_failure());

    // Soft failure
    let soft_fail = AssertionResult::fail(field, AssertionLevel::Suggest);
    assert!(!soft_fail.is_hard_failure());

    // Pass (never hard failure)
    let pass = AssertionResult::pass(field, AssertionLevel::Assert);
    assert!(!pass.is_hard_failure());
}
