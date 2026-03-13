// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for adapter system
//!
//! Tests the complete adapter pipeline: formatting prompts and parsing responses.

use kkachi::adapter::DemoData;
use kkachi::*;

#[test]
fn test_chat_adapter_roundtrip() {
    let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(false));
    let sig = Signature::parse("question -> answer").unwrap();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("question", "What is 2+2?");

    // Format the prompt
    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Verify prompt structure
    assert!(prompt_str.contains("Question: What is 2+2?"));
    assert!(prompt_str.contains("Answer:"));

    // Parse a simulated response
    let response = StrView::new("Answer: 4");
    let fields = adapter.parse(response, &sig).unwrap();

    assert_eq!(fields.len(), 1);
    let range = fields[0].1.as_range();
    assert_eq!(&response.as_str()[range], "4");
}

#[test]
fn test_chat_adapter_with_demos() {
    let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(false));
    let sig = Signature::parse("question -> answer").unwrap();

    let q_sym = sym("question");
    let a_sym = sym("answer");

    // Create demo
    let demo_inputs = [(q_sym, StrView::new("What is 1+1?"))];
    let demo_outputs = [(a_sym, StrView::new("2"))];
    let demo = DemoData::new(&demo_inputs, &demo_outputs);

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("question", "What is 3+3?");

    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[demo]);
    let prompt_str = prompt.as_str();

    // Should contain demo
    assert!(prompt_str.contains("What is 1+1?"));
    assert!(prompt_str.contains("2"));
    // Should contain separator
    assert!(prompt_str.contains("---"));
    // Should contain input
    assert!(prompt_str.contains("What is 3+3?"));
}

#[test]
fn test_json_adapter_roundtrip() {
    let adapter = JSONAdapter::new(JSONConfig::new().with_schema(true));
    let sig = Signature::parse("question -> answer").unwrap();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("question", "What is the capital of France?");

    // Format the prompt
    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Verify JSON structure in prompt
    assert!(prompt_str.contains("Respond with JSON"));
    assert!(prompt_str.contains("\"question\""));
    assert!(prompt_str.contains("What is the capital of France?"));

    // Parse a simulated response
    let response = StrView::new(r#"{"answer": "Paris"}"#);
    let fields = adapter.parse(response, &sig).unwrap();

    assert_eq!(fields.len(), 1);
    let range = fields[0].1.as_range();
    assert_eq!(&response.as_str()[range], "Paris");
}

#[test]
fn test_xml_adapter_roundtrip() {
    let adapter = XMLAdapter::new(XMLConfig::new().with_pretty(true));
    let sig = Signature::parse("question -> answer").unwrap();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("question", "What is 5*5?");

    // Format the prompt
    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Verify XML structure in prompt
    assert!(prompt_str.contains("<question>What is 5*5?</question>"));
    assert!(prompt_str.contains("<answer>...</answer>"));

    // Parse a simulated response
    let response = StrView::new("<response><answer>25</answer></response>");
    let fields = adapter.parse(response, &sig).unwrap();

    assert_eq!(fields.len(), 1);
    let range = fields[0].1.as_range();
    assert_eq!(&response.as_str()[range], "25");
}

#[test]
fn test_adapter_with_multiple_fields() {
    let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(false));
    let sig = Signature::parse("context, question -> reasoning, answer").unwrap();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("context", "Paris is the capital of France.");
    inputs.insert("question", "What is the capital of France?");

    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Both inputs should be present
    assert!(prompt_str.contains("Context: Paris is the capital of France."));
    assert!(prompt_str.contains("Question: What is the capital of France?"));

    // Parse response with multiple outputs
    let response =
        StrView::new("Reasoning: The context states Paris is the capital.\nAnswer: Paris");
    let fields = adapter.parse(response, &sig).unwrap();

    assert_eq!(fields.len(), 2);
}

#[test]
fn test_json_adapter_nested_response() {
    let adapter = JSONAdapter::default();
    let sig = Signature::parse("input -> data").unwrap();

    let response = StrView::new(r#"{"data": {"nested": "value", "count": 42}}"#);
    let fields = adapter.parse(response, &sig).unwrap();

    assert_eq!(fields.len(), 1);
    let range = fields[0].1.as_range();
    assert_eq!(
        &response.as_str()[range],
        r#"{"nested": "value", "count": 42}"#
    );
}

#[test]
fn test_xml_adapter_with_attributes() {
    let adapter = XMLAdapter::default();
    let sig = Signature::parse("input -> result").unwrap();

    // XML with attributes
    let response = StrView::new(r#"<response><result type="number">42</result></response>"#);
    let fields = adapter.parse(response, &sig).unwrap();

    assert_eq!(fields.len(), 1);
    let range = fields[0].1.as_range();
    assert_eq!(&response.as_str()[range], "42");
}

#[test]
fn test_chat_adapter_with_descriptions() {
    let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(true));

    let sig = SignatureBuilder::new("Answer questions")
        .input(InputField::create("question", "The user's question"))
        .unwrap()
        .output(OutputField::create("answer", "Your answer"))
        .unwrap()
        .build();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("question", "Hello");

    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Should include field descriptions
    assert!(prompt_str.contains("Given the following fields:"));
    assert!(prompt_str.contains("Produce the following fields:"));
}

#[test]
fn test_adapter_special_characters() {
    let adapter = JSONAdapter::default();
    let sig = Signature::parse("input -> output").unwrap();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("input", "Text with \"quotes\" and\nnewlines");

    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Should escape special characters
    assert!(prompt_str.contains("\\\"quotes\\\""));
    assert!(prompt_str.contains("\\n"));
}

#[test]
fn test_xml_adapter_escaping() {
    let adapter = XMLAdapter::default();
    let sig = Signature::parse("input -> output").unwrap();

    let mut buffer = Vec::new();
    let mut inputs = Inputs::new();
    inputs.insert("input", "Text with <tags> & entities");

    let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);
    let prompt_str = prompt.as_str();

    // Should escape XML entities
    assert!(prompt_str.contains("&lt;tags&gt;"));
    assert!(prompt_str.contains("&amp;"));
}
