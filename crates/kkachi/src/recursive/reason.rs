// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Chain of Thought reasoning.
//!
//! This module provides the [`reason`] entry point for Chain of Thought prompting,
//! where the LLM is guided to think step by step before providing a final answer.
//!
//! # How Answer Extraction Works
//!
//! The `reason()` function automatically detects whether to extract an answer:
//!
//! **With answer marker** ("Therefore:", "Answer:", etc.) - extracts the answer:
//! ```
//! use kkachi::recursive::{MockLlm, reason};
//!
//! let llm = MockLlm::new(|_, _| {
//!     "Step 1: Calculate 25 * 30 = 750\n\
//!      Step 2: Calculate 25 * 7 = 175\n\
//!      Therefore: 925".to_string()
//! });
//!
//! let result = reason(&llm, "What is 25 * 37?").go();
//! assert_eq!(result.output, "925");  // Extracted answer
//! assert!(result.reasoning().contains("Step 1"));  // Preserved reasoning
//! ```
//!
//! **Without marker** (multi-line content) - preserves full response:
//! ```
//! use kkachi::recursive::{MockLlm, reason, checks};
//!
//! let llm = MockLlm::new(|_, _| {
//!     "name: template\ntype: yaml\nconfig:\n  key: value".to_string()
//! });
//!
//! let result = reason(&llm, "Generate YAML")
//!     .validate(checks().require("config:"))
//!     .go();
//!
//! // Full response preserved automatically!
//! assert!(result.output.contains("name: template"));
//! assert!(result.output.contains("config:"));
//! ```
//!
//! # Use Cases
//!
//! Works seamlessly for:
//! - Math problems with "Therefore: X"
//! - Questions with "Answer: Y"
//! - YAML/JSON generation
//! - Code generation
//! - Multi-line structured content
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
    /// Extract code blocks in this language before validation.
    pub extract_lang: Option<String>,
}

impl Default for ReasonConfig {
    fn default() -> Self {
        Self {
            reasoning_field: "reasoning",
            include_reasoning: true,
            max_iter: 5,
            target: 1.0,
            instruction: None,
            extract_lang: None,
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

    /// Extract code blocks in the given language before validation.
    ///
    /// When set, the validator receives extracted code instead of the raw answer.
    pub fn extract(mut self, lang: impl Into<String>) -> Self {
        self.config.extract_lang = Some(lang.into());
        self
    }

    /// Disable reasoning inclusion in result.
    pub fn no_reasoning(mut self) -> Self {
        self.config.include_reasoning = false;
        self
    }

    /// Execute synchronously and return the result.
    ///
    /// If called inside a tokio runtime, uses `block_in_place`. Otherwise,
    /// creates a new single-threaded runtime.
    #[cfg(feature = "native")]
    pub fn go(self) -> ReasonResult {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.run()))
        } else {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime")
                .block_on(self.run())
        }
    }

    /// Execute synchronously (fallback without tokio).
    #[cfg(not(feature = "native"))]
    pub fn go(self) -> ReasonResult {
        futures::executor::block_on(self.run())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> ReasonResult {
        let mut iterations = 0u32;
        let mut total_tokens = 0u32;
        let mut last_score = Score::pass();
        let mut last_reasoning: Option<String> = None;
        let mut last_output = String::new();

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // Build the prompt with CoT instruction
            let cot_prompt = self.build_prompt(if iter == 0 {
                None
            } else {
                last_score.feedback_str()
            });

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

            // Extract code if configured — use extracted code as the output.
            // Try the answer first, then the full response (code may be in reasoning).
            last_output = if let Some(ref lang) = self.config.extract_lang {
                use crate::recursive::rewrite::extract_code;
                extract_code(&answer, lang)
                    .or_else(|| extract_code(&output.text, lang))
                    .map(|s| s.to_string())
                    .unwrap_or(answer.clone())
            } else {
                answer.clone()
            };
            last_score = self.validator.validate(&last_output);

            // Check if we've reached the target
            if last_score.value >= self.config.target {
                break;
            }
        }

        ReasonResult {
            output: last_output,
            reasoning: last_reasoning,
            score: last_score.value,
            iterations,
            tokens: total_tokens,
            error: None,
        }
    }

    /// Build the Chain of Thought prompt.
    fn build_prompt(&self, feedback: Option<&str>) -> String {
        let instruction = self
            .config
            .instruction
            .unwrap_or("Let's think step by step.");

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
        let answer_markers = [
            "Therefore:",
            "Answer:",
            "Final Answer:",
            "So the answer is:",
            "Result:",
        ];

        for marker in &answer_markers {
            if let Some(idx) = response.find(marker) {
                let reasoning = response[..idx].trim();
                let answer_start = idx + marker.len();
                let answer = response[answer_start..].trim();

                // Find end of answer (next newline or end of string)
                let answer_end = answer.find('\n').unwrap_or(answer.len());
                let answer = answer[..answer_end].trim().to_string();

                return (
                    if reasoning.is_empty() {
                        None
                    } else {
                        Some(reasoning)
                    },
                    answer,
                );
            }
        }

        // No marker found - use full response as answer (no reasoning extraction)
        // This fixes multi-line content (YAML, code) while preserving single-line behavior
        (None, response.trim().to_string())
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
    use crate::recursive::checks::checks;
    use crate::recursive::llm::MockLlm;

    #[test]
    fn test_reason_basic() {
        let llm = MockLlm::new(|_, _| {
            "Step 1: Calculate 25 * 30 = 750\n\
             Step 2: Calculate 25 * 7 = 175\n\
             Step 3: Add them: 750 + 175 = 925\n\n\
             Therefore: 925"
                .to_string()
        });

        let result = reason(&llm, "What is 25 * 37?").go();

        assert!(result.output.contains("925"));
        assert!(result.reasoning().contains("Step 1"));
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_reason_with_validation() {
        let llm = MockLlm::new(|_, _| "Let me think...\n\nAnswer: 42".to_string());

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
            "First, I'll analyze the problem.\nThen, I'll solve it.\nTherefore: 42",
        );
        assert!(reasoning.is_some());
        assert!(reasoning.unwrap().contains("analyze"));
        assert_eq!(answer, "42");

        // Test "Answer:" marker
        let (reasoning, answer) =
            builder.parse_response("Step 1: Do X\nStep 2: Do Y\nAnswer: The result is Z");
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
        let llm = MockLlm::new(|_, _| "Reasoning here\nAnswer: 42".to_string());

        let result = reason(&llm, "Test").no_reasoning().go();

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

    #[test]
    fn test_reason_extract_applies_to_output() {
        // Simulates real LLM behavior: code is in the reasoning section,
        // answer marker just says "see above"
        let llm = MockLlm::new(|_, _| {
            "I need to write hello world.\n\
             Here is the code:\n\
             ```python\n\
             print(\"hello\")\n\
             ```\n\n\
             Therefore: The code above prints hello"
                .to_string()
        });

        let result = reason(&llm, "Write hello world in Python")
            .extract("python")
            .go();

        // The output should be the extracted code from the full response,
        // not the raw answer text ("The code above prints hello")
        assert_eq!(result.output.trim(), "print(\"hello\")");
        assert!(!result.output.contains("```"));
        assert!(!result.output.contains("The code above"));
    }

    #[test]
    fn test_reason_multiline_no_marker_preserves_full_output() {
        // Bug fix test: multi-line without marker should preserve full content
        let llm = MockLlm::new(|_, _| "Line 1\nLine 2\nLine 3\nLine 4".to_string());

        let result = reason(&llm, "Generate multi-line content").go();

        // Should preserve ALL lines (not just last line)
        assert_eq!(result.output, "Line 1\nLine 2\nLine 3\nLine 4");
        assert_eq!(result.output.lines().count(), 4);
        assert!(result.output.contains("Line 1"));
        assert!(result.output.contains("Line 4"));
        // No reasoning when no marker found
        assert!(result.reasoning.is_none());
    }

    #[test]
    fn test_reason_with_marker_still_extracts() {
        // Marker-based extraction still works
        let llm = MockLlm::new(|_, _| "Step 1\nStep 2\nStep 3\nTherefore: 42".to_string());

        let result = reason(&llm, "Solve problem").go();

        // Should extract only the answer after marker
        assert_eq!(result.output, "42");
        // Reasoning should contain steps
        assert!(result.reasoning().contains("Step 1"));
        assert!(result.reasoning().contains("Step 3"));
    }

    #[test]
    fn test_reason_multiline_yaml_no_marker() {
        // Real-world test: YAML generation
        let llm =
            MockLlm::new(|_, _| "name: template\ntype: yaml\nconfig:\n  key: value".to_string());

        let result = reason(&llm, "Generate YAML")
            .validate(checks().require("name:").min_len(20))
            .go();

        // Full YAML should be preserved
        assert!(result.output.contains("type: yaml"));
        assert!(result.output.contains("config:"));
        assert!(result.output.contains("key: value"));
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_reason_yaml_template_validation() {
        // Integration test: complex YAML with validation
        let llm = MockLlm::new(|_, _| {
            "name: test\nruntime: yaml\n\nresources:\n  bucket:\n    type: storage.v1.bucket\n    properties:\n      name: test-bucket".to_string()
        });

        let result = reason(&llm, "Generate deployment YAML")
            .validate(checks().min_len(50))
            .go();

        assert!(result.output.contains("resources:"));
        assert!(result.output.contains("bucket:"));
        assert!(result.output.lines().count() > 5);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_reason_single_line_no_marker_unchanged() {
        // Single line without marker
        let llm = MockLlm::new(|_, _| "Simple answer".to_string());

        let result = reason(&llm, "Question").go();

        assert_eq!(result.output, "Simple answer");
        assert!(result.reasoning.is_none());
    }
}
