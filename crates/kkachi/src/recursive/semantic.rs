// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Semantic validation using LLM-as-judge.
//!
//! This module provides semantic validation capabilities that use an LLM
//! to evaluate outputs against human-defined criteria. Unlike pattern-based
//! validators, semantic validators can assess qualities like code quality,
//! correctness, and adherence to best practices.
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::{semantic, checks, refine, MockLlm};
//!
//! let judge_llm = MockLlm::new(|_, _| r#"{"overall": 0.9, "confidence": 0.85}"#.to_string());
//! let generator_llm = MockLlm::new(|_, _| "fn parse(s: &str) -> i32 { s.parse().unwrap() }".to_string());
//!
//! let semantic = semantic(&judge_llm)
//!     .criterion("Code is idiomatic Rust")
//!     .criterion("Error handling is complete")
//!     .threshold(0.8);
//!
//! let result = refine(&generator_llm, "Write a parser")
//!     .validate(checks().require("fn ").and(semantic))
//!     .go_full();
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::validate::{Score, Validate};
use smallvec::SmallVec;

/// Create a semantic validator builder.
///
/// This is the main entry point for creating LLM-as-judge validators.
///
/// # Arguments
///
/// * `llm` - The LLM to use as the judge
///
/// # Example
///
/// ```ignore
/// use kkachi::recursive::{semantic, MockLlm};
///
/// let judge = MockLlm::new(|_, _| r#"{"overall": 0.9, "confidence": 0.85}"#.to_string());
/// let validator = semantic(&judge)
///     .criterion("Code follows best practices")
///     .threshold(0.8)
///     .build();
/// ```
pub fn semantic<L: Llm>(llm: &L) -> SemanticBuilder<'_, L> {
    SemanticBuilder::new(llm)
}

/// Builder for semantic validators.
pub struct SemanticBuilder<'a, L: Llm> {
    llm: &'a L,
    criteria: SmallVec<[&'a str; 8]>,
    threshold: f64,
    system_prompt: Option<&'a str>,
}

impl<'a, L: Llm> SemanticBuilder<'a, L> {
    /// Create a new semantic validator builder.
    pub fn new(llm: &'a L) -> Self {
        Self {
            llm,
            criteria: SmallVec::new(),
            threshold: 0.7,
            system_prompt: None,
        }
    }

    /// Add a criterion to evaluate.
    ///
    /// Criteria are natural language descriptions of qualities to assess.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let validator = semantic(&llm)
    ///     .criterion("Code is idiomatic Rust")
    ///     .criterion("Error handling is complete")
    ///     .criterion("No security vulnerabilities");
    /// ```
    pub fn criterion(mut self, criterion: &'a str) -> Self {
        self.criteria.push(criterion);
        self
    }

    /// Set the minimum passing score (default: 0.7).
    ///
    /// Outputs scoring below this threshold will be considered failures.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set a custom system prompt for the judge.
    ///
    /// By default, a structured prompt is used that instructs the LLM
    /// to evaluate each criterion and return JSON scores.
    pub fn system_prompt(mut self, prompt: &'a str) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    /// Build the semantic validator.
    ///
    /// This creates a validator that can be used with the refinement loop.
    pub fn build(self) -> SemanticValidator<'a, L> {
        SemanticValidator {
            llm: self.llm,
            criteria: self.criteria,
            threshold: self.threshold,
            system_prompt: self.system_prompt,
        }
    }
}

/// Semantic validator using LLM-as-judge.
///
/// This validator calls an LLM to evaluate text against specified criteria,
/// parsing the response to extract scores and confidence levels.
pub struct SemanticValidator<'a, L: Llm> {
    llm: &'a L,
    criteria: SmallVec<[&'a str; 8]>,
    threshold: f64,
    #[allow(dead_code)]
    system_prompt: Option<&'a str>,
}

impl<'a, L: Llm> SemanticValidator<'a, L> {
    /// Build the judge prompt for evaluating text.
    fn build_judge_prompt(&self, text: &str) -> String {
        let criteria_list = self
            .criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are evaluating code/text against specific criteria.
Rate each criterion from 0.0 to 1.0.

TEXT TO EVALUATE:
```
{text}
```

CRITERIA:
{criteria}

Respond ONLY with a JSON object in this exact format:
{{"scores": [{{"criterion": "criterion text", "score": 0.0-1.0, "reason": "brief explanation"}}], "overall": 0.0-1.0, "confidence": 0.0-1.0}}

Important:
- "overall" is the weighted average of all criterion scores
- "confidence" indicates how certain you are about your assessment (1.0 = very certain, 0.5 = uncertain)
- Be strict but fair in your evaluation"#,
            text = text,
            criteria = criteria_list
        )
    }

    /// Parse the judge's response to extract scores.
    fn parse_judgment(&self, response: &str) -> Score<'static> {
        // Try to extract JSON from the response
        let json_str = self.extract_json(response);

        // Parse overall score and confidence
        let (overall, confidence) = self.parse_scores(json_str);

        // Generate feedback
        let feedback = if overall >= self.threshold {
            format!("Semantic validation passed with score {:.2}", overall)
        } else {
            format!(
                "Semantic validation failed: score {:.2} < threshold {:.2}",
                overall, self.threshold
            )
        };

        Score::with_feedback(overall, feedback).with_confidence(confidence)
    }

    /// Extract JSON from the response (handles markdown code blocks).
    fn extract_json<'b>(&self, response: &'b str) -> &'b str {
        // Try to find JSON in code blocks
        if let Some(start) = response.find("```json") {
            let after_marker = &response[start + 7..];
            if let Some(end) = after_marker.find("```") {
                return after_marker[..end].trim();
            }
        }

        // Try to find raw JSON
        if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                return &response[start..=end];
            }
        }

        response
    }

    /// Parse overall and confidence scores from JSON.
    fn parse_scores(&self, json_str: &str) -> (f64, f64) {
        // Simple JSON parsing without serde dependency
        let overall = self.extract_number(json_str, "overall").unwrap_or(0.5);
        let confidence = self.extract_number(json_str, "confidence").unwrap_or(0.5);

        (overall.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0))
    }

    /// Extract a number value from a JSON string.
    fn extract_number(&self, json: &str, key: &str) -> Option<f64> {
        let pattern = format!("\"{}\"", key);
        let start = json.find(&pattern)?;
        let after_key = &json[start + pattern.len()..];

        // Find the colon and then the number
        let colon_pos = after_key.find(':')?;
        let after_colon = &after_key[colon_pos + 1..];

        // Skip whitespace and find the number
        let trimmed = after_colon.trim_start();

        // Extract the number (digits and decimal point)
        let end = trimmed
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(trimmed.len());

        trimmed[..end].parse().ok()
    }
}

impl<'a, L: Llm> Validate for SemanticValidator<'a, L> {
    fn validate(&self, text: &str) -> Score<'static> {
        let prompt = self.build_judge_prompt(text);

        // Call LLM synchronously using block_on
        let response = futures::executor::block_on(self.llm.generate(&prompt, "", None));

        match response {
            Ok(output) => self.parse_judgment(&output.text),
            Err(e) => Score::with_feedback(
                0.5,
                format!("Semantic validation error: {}", e),
            )
            .with_confidence(0.0),
        }
    }

    fn name(&self) -> &'static str {
        "semantic"
    }
}

// Note: Send + Sync are implemented automatically if L: Llm (which requires Send + Sync)
// The SemanticValidator stores a reference to the LLM, so it's Send + Sync if the reference is.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;

    #[test]
    fn test_semantic_builder() {
        let llm = MockLlm::new(|_, _| "test".to_string());
        let builder = semantic(&llm)
            .criterion("Is idiomatic")
            .criterion("Has error handling")
            .threshold(0.8);

        assert_eq!(builder.criteria.len(), 2);
        assert!((builder.threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_judge_prompt() {
        let llm = MockLlm::new(|_, _| "test".to_string());
        let validator = semantic(&llm)
            .criterion("Code quality")
            .criterion("Readability")
            .build();

        let prompt = validator.build_judge_prompt("fn main() {}");
        assert!(prompt.contains("Code quality"));
        assert!(prompt.contains("Readability"));
        assert!(prompt.contains("fn main() {}"));
    }

    #[test]
    fn test_parse_judgment_success() {
        let llm = MockLlm::new(|_, _| "test".to_string());
        let validator = semantic(&llm).criterion("Test").build();

        let response = r#"{"scores": [], "overall": 0.85, "confidence": 0.9}"#;
        let score = validator.parse_judgment(response);

        assert!((score.value - 0.85).abs() < 0.01);
        assert!((score.confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_parse_judgment_with_code_block() {
        let llm = MockLlm::new(|_, _| "test".to_string());
        let validator = semantic(&llm).criterion("Test").build();

        let response = r#"Here's my evaluation:
```json
{"scores": [], "overall": 0.75, "confidence": 0.8}
```"#;
        let score = validator.parse_judgment(response);

        assert!((score.value - 0.75).abs() < 0.01);
        assert!((score.confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_semantic_validator_validate() {
        let llm = MockLlm::new(|_, _| {
            r#"{"scores": [{"criterion": "Test", "score": 0.9, "reason": "Good"}], "overall": 0.9, "confidence": 0.95}"#.to_string()
        });

        let validator = semantic(&llm)
            .criterion("Test criterion")
            .threshold(0.8)
            .build();

        let score = validator.validate("fn main() {}");

        assert!((score.value - 0.9).abs() < 0.01);
        assert!((score.confidence - 0.95).abs() < 0.01);
        assert!(score.passes(0.8));
    }

    #[test]
    fn test_semantic_validator_below_threshold() {
        let llm = MockLlm::new(|_, _| {
            r#"{"overall": 0.5, "confidence": 0.8}"#.to_string()
        });

        let validator = semantic(&llm)
            .criterion("Quality")
            .threshold(0.7)
            .build();

        let score = validator.validate("bad code");

        assert!((score.value - 0.5).abs() < 0.01);
        assert!(!score.passes(0.7));
        assert!(score.feedback_str().unwrap().contains("failed"));
    }

    #[test]
    fn test_extract_number() {
        let llm = MockLlm::new(|_, _| "test".to_string());
        let validator = semantic(&llm).criterion("Test").build();

        let json = r#"{"overall": 0.85, "confidence": 0.9}"#;
        assert!((validator.extract_number(json, "overall").unwrap() - 0.85).abs() < 0.01);
        assert!((validator.extract_number(json, "confidence").unwrap() - 0.9).abs() < 0.01);
        assert!(validator.extract_number(json, "missing").is_none());
    }
}
