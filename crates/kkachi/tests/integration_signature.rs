// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for signature system

use kkachi::*;

#[test]
fn test_signature_creation_from_string() {
    let sig = Signature::parse("question, context -> answer").unwrap();

    assert_eq!(sig.input_fields.len(), 2);
    assert_eq!(sig.output_fields.len(), 1);
    assert_eq!(sig.input_fields[0].name, "question");
    assert_eq!(sig.input_fields[1].name, "context");
    assert_eq!(sig.output_fields[0].name, "answer");
}

#[test]
fn test_signature_builder() {
    let sig = SignatureBuilder::new("Classify the text")
        .input(field::InputField::create("text", "Input text"))
        .unwrap()
        .input(field::InputField::create("labels", "Possible labels"))
        .unwrap()
        .output(field::OutputField::create(
            "classification",
            "The classification",
        ))
        .unwrap()
        .build();

    assert_eq!(sig.instructions, "Classify the text");
    assert_eq!(sig.input_fields.len(), 2);
    assert_eq!(sig.output_fields.len(), 1);
}

#[test]
fn test_signature_to_string() {
    let sig = Signature::parse("q, ctx -> a, confidence").unwrap();
    let sig_str = sig.to_string_format();

    assert_eq!(sig_str, "q, ctx -> a, confidence");
}

#[test]
fn test_signature_field_inference() {
    let sig = Signature::parse("userQuery -> systemResponse").unwrap();

    // Check prefix inference (camelCase -> Title Case)
    assert_eq!(sig.input_fields[0].prefix.as_ref(), "User Query");
    assert_eq!(sig.output_fields[0].prefix.as_ref(), "System Response");
}

#[test]
fn test_signature_clone_and_owned() {
    let sig = Signature::parse("input -> output").unwrap();
    let owned = sig.clone().into_owned();

    assert_eq!(owned.input_fields[0].name, "input");
    assert_eq!(owned.output_fields[0].name, "output");
}

#[test]
fn test_signature_with_custom_fields() {
    let input_field = field::InputField::create("query", "User query")
        .with_prefix("Q:")
        .with_format("lowercase");

    let sig = SignatureBuilder::new("Process query")
        .input(input_field)
        .unwrap()
        .output(field::OutputField::create("result", "Result"))
        .unwrap()
        .build();

    assert_eq!(sig.input_fields[0].prefix, "Q:");
    assert_eq!(sig.input_fields[0].format.as_ref().unwrap(), "lowercase");
}

#[test]
fn test_signature_error_handling() {
    // Missing arrow
    let result = Signature::parse("input output");
    assert!(result.is_err());

    // Multiple arrows
    let result = Signature::parse("input -> output -> extra");
    assert!(result.is_err());
}
