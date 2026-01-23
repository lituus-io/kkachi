// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Chain of Thought reasoning.
//!
//! This module provides the [`reason`] entry point for Chain of Thought prompting,
//! where the LLM is guided to think step by step before providing a final answer.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, reason};
//!
//! let llm = MockLlm::new(|prompt, _| {
//!     "Let me think step by step.\n\n1. First, 25 * 30 = 750\n2. Then, 25 * 7 = 175\n3. So 25 * 37 = 750 + 175 = 925\n\nTherefore: 925".to_string()
//! });
//!
//! let result = reason(&llm, "What is 25 * 37?").go();
//! assert!(result.reasoning().contains("step"));
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::validate::{NoValidation, Score, Validate};

/// Entry point for Chain of Thought reasoning.
///
/// This creates a builder that guides the LLM to think step by step.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, reason, checks};
///
/// let llm = MockLlm::new(|_, _| "Step 1: ... Therefore: 42".to_string());
///
/// let result = reason(&llm, "Solve this problem")
///     .validate(checks().regex(r"\d+"))
///     .go();
/// ```
pub fn reason<'a, L: Llm>(llm: &'a L, prompt: &'a str) -> Reason<'a, L, NoValidation> {
    Reason::new(llm, prompt)
}

/// Configuration for Chain of Thought reasoning.
#[derive(Clone)]
pub struct ReasonConfig {
    /// Name of the reasoning field in the prompt.
    pub reasoning_field: &'static str,
    /// Whether to include the reasoning in the result.
    pub include_reasoning: bool,
    /// Maximum refinement iterations.
    pub max_iter: u32,
    /// Target score for validation.
    pub target: f64,
    /// Custom CoT instruction (replaces default "Let's think step by step").
    pub instruction: Option<&'static str>,
}

impl Default for ReasonConfig {
    fn default() -> Self {
        Self {
            reasoning_field: "reasoning",
            include_reasoning: true,
            max_iter: 5,
            target: 1.0,
            instruction: None,
        }
    }
}

/// Chain of Thought reasoning builder.
///
/// This builder constructs prompts that guide the LLM to think step by step
/// before providing a final answer.
pub struct Reason<'a, L: Llm, V: Validate> {
    llm: &'a L,
    prompt: &'a str,
    validator: V,
    config: ReasonConfig,
}

impl<'a, L: Llm> Reason<'a, L, NoValidation> {
    /// Create a new Chain of Thought builder.
    pub fn new(llm: &'a L, prompt: &'a str) -> Self {
        Self {
            llm,
            prompt,
            validator: NoValidation,
            config: ReasonConfig::default(),
        }
    }
}

impl<'a, L: Llm, V: Validate> Reason<'a, L, V> {
    /// Set a validator for the final answer.
    ///
    /// The validator is applied to the extracted answer, not the full reasoning.
    pub fn validate<V2: Validate>(self, validator: V2) -> Reason<'a, L, V2> {
        Reason {
            llm: self.llm,
            prompt: self.prompt,
            validator,
            config: self.config,
        }
    }

    /// Set a custom name for the reasoning field.
    ///
    /// Default is "reasoning".
    pub fn reasoning_field(mut self, name: &'static str) -> Self {
        self.config.reasoning_field = name;
        self
    }

    /// Set maximum refinement iterations.
    ///
    /// If the answer doesn't pass validation, the LLM will be asked to try again.
    pub fn max_iter(mut self, n: u32) -> Self {
        self.config.max_iter = n;
        self
    }

    /// Set target validation score.
    ///
    /// Refinement stops when this score is reached.
    pub fn target(mut self, score: f64) -> Self {
        self.config.target = score;
        self
    }

    /// Set a custom Chain of Thought instruction.
    ///
    /// Default is "Let's think step by step."
    pub fn instruction(mut self, inst: &'static str) -> Self {
        self.config.instruction = Some(inst);
        self
    }

    /// Disable reasoning inclusion in result.
    pub fn no_reasoning(mut self) -> Self {
        self.config.include_reasoning = false;
        self
    }

    /// Execute synchronously and return the result.
    pub fn go(self) -> ReasonResult {
        futures::executor::block_on(self.run())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> ReasonResult {
        let mut iterations = 0u32;
        let mut total_tokens = 0u32;
        let mut last_score = Score::pass();
        let mut last_reasoning: Option<String> = None;
        let mut last_answer = String::new();

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // Build the prompt with CoT instruction
            let cot_prompt = self.build_prompt(if iter == 0 { None } else { last_score.feedback_str() });

            // Call the LLM
            let output = match self.llm.generate(&cot_prompt, "", None).await {
                Ok(out) => out,
                Err(e) => {
                    return ReasonResult {
                        output: String::new(),
                        reasoning: None,
                        score: 0.0,
                        iterations,
                        tokens: total_tokens,
                        error: Some(e.to_string()),
                    };
                }
            };

            total_tokens += output.prompt_tokens + output.completion_tokens;

            // Parse the response to extract reasoning and answer
            let (reasoning, answer) = self.parse_response(&output.text);

            last_reasoning = if self.config.include_reasoning {
                reasoning.map(|s| s.to_string())
            } else {
                None
            };
            last_answer = answer.clone();

            // Validate the answer (not the full response)
            last_score = self.validator.validate(&answer);

            // Check if we've reached the target
            if last_score.value >= self.config.target {
                break;
            }
        }

        ReasonResult {
            output: last_answer,
            reasoning: last_reasoning,
            score: last_score.value,
            iterations,
            tokens: total_tokens,
            error: None,
        }
    }

    /// Build the Chain of Thought prompt.
    fn build_prompt(&self, feedback: Option<&str>) -> String {
        let instruction = self.config.instruction.unwrap_or("Let's think step by step.");

        let mut prompt = format!("{}\n\n{}", self.prompt, instruction);

        if let Some(fb) = feedback {
            prompt.push_str(&format!(
                "\n\nPrevious attempt was incorrect. Feedback: {}\n\nPlease try again, thinking more carefully.",
                fb
            ));
        }

        prompt.push_str("\n\nAfter your reasoning, provide the final answer after \"Therefore:\" or \"Answer:\".");
        prompt
    }

    /// Parse the response to extract reasoning and final answer.
    fn parse_response<'b>(&self, response: &'b str) -> (Option<&'b str>, String) {
        // Look for common answer markers
        let answer_markers = ["Therefore:", "Answer:", "Final Answer:", "So the answer is:", "Result:"];

        for marker in &answer_markers {
            if let Some(idx) = response.find(marker) {
                let reasoning = response[..idx].trim();
                let answer_start = idx + marker.len();
                let answer = response[answer_start..].trim();

                // Find end of answer (next newline or end of string)
                let answer_end = answer.find('\n').unwrap_or(answer.len());
                let answer = answer[..answer_end].trim().to_string();

                return (
                    if reasoning.is_empty() { None } else { Some(reasoning) },
                    answer,
                );
            }
        }

        // No marker found - use the whole response as the answer
        // and try to extract reasoning from structure
        if let Some(last_line_start) = response.rfind('\n') {
            let reasoning = response[..last_line_start].trim();
            let answer = response[last_line_start..].trim().to_string();
            (
                if reasoning.is_empty() { None } else { Some(reasoning) },
                answer,
            )
        } else {
            (None, response.trim().to_string())
        }
    }
}

/// Result of Chain of Thought reasoning.
#[derive(Debug, Clone)]
pub struct ReasonResult {
    /// The final answer extracted from the response.
    pub output: String,
    /// The reasoning trace (if included).
    pub reasoning: Option<String>,
    /// Validation score for the answer.
    pub score: f64,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Total tokens used.
    pub tokens: u32,
    /// Error message if reasoning failed.
    pub error: Option<String>,
}

impl ReasonResult {
    /// Get the reasoning trace.
    pub fn reasoning(&self) -> &str {
        self.reasoning.as_deref().unwrap_or("")
    }

    /// Check if the reasoning succeeded.
    pub fn success(&self) -> bool {
        self.error.is_none() && self.score >= 1.0
    }

    /// Check if there was an error.
    pub fn is_err(&self) -> bool {
        self.error.is_some()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;
    use crate::recursive::checks::checks;

    #[test]
    fn test_reason_basic() {
        let llm = MockLlm::new(|_, _| {
            "Step 1: Calculate 25 * 30 = 750\n\
             Step 2: Calculate 25 * 7 = 175\n\
             Step 3: Add them: 750 + 175 = 925\n\n\
             Therefore: 925".to_string()
        });

        let result = reason(&llm, "What is 25 * 37?").go();

        assert!(result.output.contains("925"));
        assert!(result.reasoning().contains("Step 1"));
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_reason_with_validation() {
        let llm = MockLlm::new(|_, _| {
            "Let me think...\n\nAnswer: 42".to_string()
        });

        let result = reason(&llm, "Calculate something")
            .validate(checks().regex(r"\d+"))
            .go();

        assert!(result.score >= 1.0);
        assert_eq!(result.output, "42");
    }

    #[test]
    fn test_reason_parse_response() {
        let llm = MockLlm::new(|_, _| String::new());
        let builder = reason(&llm, "test");

        // Test "Therefore:" marker
        let (reasoning, answer) = builder.parse_response(
            "First, I'll analyze the problem.\nThen, I'll solve it.\nTherefore: 42"
        );
        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("analyze"));
        assert_eq!(answer, "42");

        // Test "Answer:" marker
        let (reasoning, answer) = builder.parse_response(
            "Step 1: Do X\nStep 2: Do Y\nAnswer: The result is Z"
        );
        assert!(reasoning.is_some());
        assert_eq!(answer, "The result is Z");
    }

    #[test]
    fn test_reason_custom_instruction() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("Break this down") {
                "Breakdown: ... Answer: correct".to_string()
            } else {
                "Wrong instruction".to_string()
            }
        });

        let result = reason(&llm, "Solve X")
            .instruction("Break this down into parts.")
            .go();

        assert!(result.output.contains("correct"));
    }

    #[test]
    fn test_reason_no_reasoning() {
        let llm = MockLlm::new(|_, _| {
            "Reasoning here\nAnswer: 42".to_string()
        });

        let result = reason(&llm, "Test")
            .no_reasoning()
            .go();

        assert!(result.reasoning.is_none());
        assert_eq!(result.output, "42");
    }

    #[test]
    fn test_reason_config() {
        let llm = MockLlm::new(|_, _| String::new());

        let builder = reason(&llm, "test")
            .reasoning_field("thought")
            .max_iter(10)
            .target(0.8);

        assert_eq!(builder.config.reasoning_field, "thought");
        assert_eq!(builder.config.max_iter, 10);
        assert!((builder.config.target - 0.8).abs() < f64::EPSILON);
    }
}
