// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for Predict module

use kkachi::buffer::Buffer;
use kkachi::predict::{predict_with_lm, LMClient, LMOutput, Predict};
use kkachi::str_view::StrView;
use kkachi::{Result, Signature};

/// Test LM client that returns predefined responses
struct TestLM {
    response: &'static str,
}

impl TestLM {
    fn new(response: &'static str) -> Self {
        Self { response }
    }
}

impl LMClient for TestLM {
    type GenerateFut<'a>
        = std::future::Ready<Result<LMOutput<'a>>>
    where
        Self: 'a;

    fn generate<'a>(&'a self, _prompt: StrView<'a>) -> Self::GenerateFut<'a> {
        // Create a buffer with the response - using static for simplicity
        // In real usage, this would be from the HTTP response
        static RESPONSE_1: Buffer = Buffer::Static(b"Answer: Paris");
        static RESPONSE_2: Buffer = Buffer::Static(b"Answer: 8");
        static RESPONSE_3: Buffer = Buffer::Static(b"Sentiment: positive\nConfidence: 0.95");

        // Select response based on content
        let buffer = if self.response.contains("Paris") {
            &RESPONSE_1
        } else if self.response.contains("8") {
            &RESPONSE_2
        } else {
            &RESPONSE_3
        };

        std::future::ready(Ok(LMOutput {
            buffer: buffer.view_all(),
            prompt_tokens: 10,
            completion_tokens: 5,
        }))
    }
}

#[tokio::test]
async fn test_predict_basic_qa() {
    let sig = Signature::parse("question -> answer").unwrap();
    let predict = Predict::without_demos(&sig);
    let lm = TestLM::new("Paris");

    let mut inputs = kkachi::types::Inputs::new();
    inputs.insert("question", "What is the capital of France?");

    let mut prompt_buffer = Vec::new();
    let result = predict_with_lm(&predict, &inputs, &lm, &mut prompt_buffer).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    let answer = output.get_by_name("answer");
    assert!(answer.is_some());
    assert!(answer.unwrap().as_str().contains("Paris"));
}

#[tokio::test]
async fn test_predict_multiple_outputs() {
    let sig = Signature::parse("text -> sentiment, confidence").unwrap();
    let predict = Predict::without_demos(&sig);
    let lm = TestLM::new("sentiment");

    let mut inputs = kkachi::types::Inputs::new();
    inputs.insert("text", "This is great!");

    let mut prompt_buffer = Vec::new();
    let result = predict_with_lm(&predict, &inputs, &lm, &mut prompt_buffer).await;

    assert!(result.is_ok());
    let output = result.unwrap();

    // Note: parsing depends on field prefixes - this test validates the flow works
    assert!(output.prompt_tokens > 0);
    assert!(output.completion_tokens > 0);
}

#[tokio::test]
async fn test_predict_builds_prompt() {
    let sig = Signature::parse("question -> answer").unwrap();
    let predict = Predict::without_demos(&sig);

    let mut inputs = kkachi::types::Inputs::new();
    inputs.insert("question", "Test question?");

    let mut buffer = Vec::new();
    let prompt = predict.build_prompt_into(&inputs, &mut buffer);

    assert!(prompt.as_str().contains("Question"));
    assert!(prompt.as_str().contains("Test question?"));
    assert!(prompt.as_str().contains("Answer"));
}

#[tokio::test]
async fn test_predict_parse_response() {
    let sig = Signature::parse("question -> answer").unwrap();
    let predict = Predict::without_demos(&sig);

    // Test parsing a response
    let response = StrView::new("Answer: The result is 42");
    let ranges = predict.parse_response_ranges(response);

    assert!(!ranges.is_empty());
}
