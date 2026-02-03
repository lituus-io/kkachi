// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! True prompt optimization via dataset evaluation.
//!
//! This module implements DSPy-style prompt optimization where an LLM
//! is used to find the best prompt template for a given task. Unlike
//! simple refinement (which retries outputs), this module optimizes
//! the *prompt itself* by evaluating different strategies against a
//! training dataset.
//!
//! # Key Concepts
//!
//! - [`TrainingExample`]: An input/output pair for evaluation
//! - [`Dataset`]: A collection of training examples
//! - [`Metric`]: Evaluates output quality for a given input
//! - [`Optimizer`]: Finds the best prompt strategy via [`Strategy`]
//! - [`CompiledPrompt`]: The optimized prompt ready for inference
//!
//! # Examples
//!
//! ```no_run
//! use kkachi::recursive::{optimize::*, CliLlm, checks};
//!
//! let llm = CliLlm::new().unwrap();
//!
//! let dataset = Dataset::new()
//!     .example("What is 2+2?", "4")
//!     .example("What is 3*5?", "15")
//!     .example("What is 10-7?", "3");
//!
//! let result = Optimizer::new(&llm, "Answer math questions concisely")
//!     .dataset(&dataset)
//!     .metric(|output, expected| {
//!         if output.contains(expected) { 1.0 } else { 0.0 }
//!     })
//!     .strategy(Strategy::BootstrapFewShot { max_examples: 3 })
//!     .go();
//!
//! println!("Best prompt: {}", result.prompt);
//! println!("Score: {:.2}", result.score);
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::validate::Validate;

/// A single training example with input and expected output.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// The input prompt/question.
    pub input: String,
    /// The expected/reference output (for metric evaluation).
    pub expected: String,
    /// Optional label/category for the example.
    pub label: Option<String>,
}

impl TrainingExample {
    /// Create a new training example.
    pub fn new(input: impl Into<String>, expected: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            expected: expected.into(),
            label: None,
        }
    }

    /// Add a label to this example.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A collection of training examples for prompt optimization.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// The training examples.
    pub examples: Vec<TrainingExample>,
}

impl Dataset {
    /// Create a new empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an example to the dataset.
    pub fn example(mut self, input: impl Into<String>, expected: impl Into<String>) -> Self {
        self.examples.push(TrainingExample::new(input, expected));
        self
    }

    /// Add a labeled example.
    pub fn labeled_example(
        mut self,
        input: impl Into<String>,
        expected: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        self.examples
            .push(TrainingExample::new(input, expected).with_label(label));
        self
    }

    /// Number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// Metric function type for evaluating outputs.
///
/// Takes the generated output and the expected output, returns a score 0.0-1.0.
pub type MetricFn = Box<dyn Fn(&str, &str) -> f64 + Send + Sync>;

/// Optimization strategy.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Select the best few-shot examples from the training set.
    ///
    /// Generates outputs for each training example, scores them,
    /// and includes the top-scoring examples as few-shot demonstrations.
    BootstrapFewShot {
        /// Maximum number of examples to include in the prompt.
        max_examples: usize,
    },
    /// Try different instruction phrasings and pick the best.
    ///
    /// Uses the LLM to generate candidate instructions, evaluates
    /// each against the dataset, and selects the highest-scoring one.
    InstructionSearch {
        /// Number of candidate instructions to generate.
        num_candidates: usize,
    },
    /// Combine bootstrap few-shot with instruction search.
    Combined {
        /// Maximum few-shot examples.
        max_examples: usize,
        /// Number of instruction candidates.
        num_candidates: usize,
    },
}

/// Result of prompt optimization.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// The optimized prompt template.
    pub prompt: String,
    /// Few-shot examples selected (if any).
    pub examples: Vec<TrainingExample>,
    /// The instruction used.
    pub instruction: String,
    /// Average score across the dataset.
    pub score: f64,
    /// Number of evaluations performed.
    pub evaluations: u32,
    /// Scores for each candidate tried.
    pub candidate_scores: Vec<f64>,
}

/// Prompt optimizer that evaluates strategies against a dataset.
pub struct Optimizer<'a, L: Llm> {
    llm: &'a L,
    base_prompt: String,
    dataset: Option<&'a Dataset>,
    metric: Option<MetricFn>,
    validator: Option<Box<dyn Validate>>,
    strategy: Strategy,
}

impl<'a, L: Llm> Optimizer<'a, L> {
    /// Create a new optimizer with the given LLM and base prompt.
    pub fn new(llm: &'a L, base_prompt: impl Into<String>) -> Self {
        Self {
            llm,
            base_prompt: base_prompt.into(),
            dataset: None,
            metric: None,
            validator: None,
            strategy: Strategy::BootstrapFewShot { max_examples: 3 },
        }
    }

    /// Set the training dataset.
    pub fn dataset(mut self, dataset: &'a Dataset) -> Self {
        self.dataset = Some(dataset);
        self
    }

    /// Set the metric function for evaluation.
    ///
    /// The metric takes (generated_output, expected_output) and returns 0.0-1.0.
    pub fn metric<F: Fn(&str, &str) -> f64 + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.metric = Some(Box::new(f));
        self
    }

    /// Set a validator for output quality (used alongside or instead of metric).
    pub fn validate(mut self, v: impl Validate + 'static) -> Self {
        self.validator = Some(Box::new(v));
        self
    }

    /// Set the optimization strategy.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Run the optimization and return the best prompt configuration.
    ///
    /// If called inside a tokio runtime, uses `block_in_place`. Otherwise,
    /// creates a new single-threaded runtime.
    #[cfg(feature = "native")]
    pub fn go(self) -> OptimizeResult {
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

    /// Run the optimization (fallback without tokio).
    #[cfg(not(feature = "native"))]
    pub fn go(self) -> OptimizeResult {
        futures::executor::block_on(self.run())
    }

    /// Run the optimization asynchronously.
    pub async fn run(self) -> OptimizeResult {
        let dataset = match self.dataset {
            Some(d) if !d.is_empty() => d,
            _ => {
                return OptimizeResult {
                    prompt: self.base_prompt.clone(),
                    examples: Vec::new(),
                    instruction: self.base_prompt,
                    score: 0.0,
                    evaluations: 0,
                    candidate_scores: Vec::new(),
                };
            }
        };

        match self.strategy.clone() {
            Strategy::BootstrapFewShot { max_examples } => {
                self.run_bootstrap(dataset, max_examples).await
            }
            Strategy::InstructionSearch { num_candidates } => {
                self.run_instruction_search(dataset, num_candidates).await
            }
            Strategy::Combined {
                max_examples,
                num_candidates,
            } => {
                self.run_combined(dataset, max_examples, num_candidates)
                    .await
            }
        }
    }

    /// Evaluate a prompt + examples configuration against the dataset.
    async fn evaluate(
        &self,
        instruction: &str,
        examples: &[TrainingExample],
        dataset: &Dataset,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut count = 0;

        // Build context from few-shot examples
        let mut context = String::new();
        for ex in examples {
            context.push_str(&format!("Input: {}\nOutput: {}\n\n", ex.input, ex.expected));
        }

        for example in &dataset.examples {
            let prompt = format!("{}\n\nInput: {}", instruction, example.input);
            let output = match self.llm.generate(&prompt, &context, None).await {
                Ok(out) => out.text,
                Err(_) => continue,
            };

            let score = self.score_output(&output, &example.expected);
            total_score += score;
            count += 1;
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }

    /// Score a single output against the expected.
    fn score_output(&self, output: &str, expected: &str) -> f64 {
        let mut score = 0.0;
        let mut components = 0;

        // Apply metric if set
        if let Some(ref metric) = self.metric {
            score += metric(output, expected);
            components += 1;
        }

        // Apply validator if set
        if let Some(ref validator) = self.validator {
            score += validator.validate(output).value;
            components += 1;
        }

        // If neither set, use simple contains check
        if components == 0 {
            return if output.contains(expected) { 1.0 } else { 0.0 };
        }

        score / components as f64
    }

    /// Bootstrap few-shot: try each example as a demonstration, keep the best.
    async fn run_bootstrap(&self, dataset: &Dataset, max_examples: usize) -> OptimizeResult {
        let mut evaluations = 0u32;
        let mut candidate_scores = Vec::new();

        // First, generate outputs for each training example and score them
        let mut scored_examples: Vec<(f64, &TrainingExample)> = Vec::new();

        for example in &dataset.examples {
            let prompt = format!("{}\n\nInput: {}", self.base_prompt, example.input);
            if let Ok(output) = self.llm.generate(&prompt, "", None).await {
                let score = self.score_output(&output.text, &example.expected);
                scored_examples.push((score, example));
                evaluations += 1;
            }
        }

        // Sort by score (highest first) — these are the best demonstrations
        scored_examples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K as few-shot examples
        let selected: Vec<TrainingExample> = scored_examples
            .iter()
            .take(max_examples)
            .map(|(_, ex)| (*ex).clone())
            .collect();

        // Evaluate the base prompt without examples
        let base_score = self.evaluate(&self.base_prompt, &[], dataset).await;
        evaluations += dataset.len() as u32;
        candidate_scores.push(base_score);

        // Evaluate with the selected few-shot examples
        let few_shot_score = self.evaluate(&self.base_prompt, &selected, dataset).await;
        evaluations += dataset.len() as u32;
        candidate_scores.push(few_shot_score);

        // Pick the better approach
        let (final_examples, final_score) = if few_shot_score > base_score {
            (selected, few_shot_score)
        } else {
            (Vec::new(), base_score)
        };

        // Build the optimized prompt
        let mut optimized_prompt = self.base_prompt.clone();
        if !final_examples.is_empty() {
            optimized_prompt.push_str("\n\nExamples:");
            for ex in &final_examples {
                optimized_prompt
                    .push_str(&format!("\nInput: {}\nOutput: {}", ex.input, ex.expected));
            }
        }

        OptimizeResult {
            prompt: optimized_prompt,
            examples: final_examples,
            instruction: self.base_prompt.clone(),
            score: final_score,
            evaluations,
            candidate_scores,
        }
    }

    /// Instruction search: generate candidate instructions, evaluate each.
    async fn run_instruction_search(
        &self,
        dataset: &Dataset,
        num_candidates: usize,
    ) -> OptimizeResult {
        let mut evaluations = 0u32;
        let mut candidate_scores = Vec::new();
        let mut best_instruction = self.base_prompt.clone();

        // First evaluate the base prompt
        let base_score = self.evaluate(&self.base_prompt, &[], dataset).await;
        evaluations += dataset.len() as u32;
        candidate_scores.push(base_score);
        let mut best_score = base_score;

        // Generate candidate instructions using the LLM
        let meta_prompt = format!(
            "Generate {} different instruction phrasings for this task. \
             Each instruction should be a complete, self-contained prompt \
             that guides an AI to perform the task well.\n\n\
             Original instruction: {}\n\n\
             Format each candidate on its own line, prefixed with a number:\n\
             1. [instruction]\n2. [instruction]\netc.",
            num_candidates, self.base_prompt
        );

        let candidates = match self.llm.generate(&meta_prompt, "", None).await {
            Ok(output) => parse_numbered_list(&output.text),
            Err(_) => Vec::new(),
        };
        evaluations += 1;

        // Evaluate each candidate instruction
        for candidate in &candidates {
            let score = self.evaluate(candidate, &[], dataset).await;
            evaluations += dataset.len() as u32;
            candidate_scores.push(score);

            if score > best_score {
                best_score = score;
                best_instruction = candidate.clone();
            }
        }

        OptimizeResult {
            prompt: best_instruction.clone(),
            examples: Vec::new(),
            instruction: best_instruction,
            score: best_score,
            evaluations,
            candidate_scores,
        }
    }

    /// Combined: bootstrap few-shot + instruction search.
    async fn run_combined(
        &self,
        dataset: &Dataset,
        max_examples: usize,
        num_candidates: usize,
    ) -> OptimizeResult {
        // First, find the best examples via bootstrap
        let bootstrap_result = self.run_bootstrap(dataset, max_examples).await;

        // Then, search for the best instruction
        let mut evaluations = bootstrap_result.evaluations;
        let mut candidate_scores = bootstrap_result.candidate_scores;
        let mut best_instruction = bootstrap_result.instruction.clone();
        let mut best_score = bootstrap_result.score;
        let best_examples = bootstrap_result.examples;

        // Generate candidate instructions
        let meta_prompt = format!(
            "Generate {} different instruction phrasings for this task. \
             Each should be a complete prompt.\n\n\
             Original: {}\n\n\
             Format: 1. [instruction]",
            num_candidates, self.base_prompt
        );

        let candidates = match self.llm.generate(&meta_prompt, "", None).await {
            Ok(output) => parse_numbered_list(&output.text),
            Err(_) => Vec::new(),
        };
        evaluations += 1;

        // Evaluate each candidate with the best examples
        for candidate in &candidates {
            let score = self.evaluate(candidate, &best_examples, dataset).await;
            evaluations += dataset.len() as u32;
            candidate_scores.push(score);

            if score > best_score {
                best_score = score;
                best_instruction = candidate.clone();
            }
        }

        // Build optimized prompt
        let mut optimized_prompt = best_instruction.clone();
        if !best_examples.is_empty() {
            optimized_prompt.push_str("\n\nExamples:");
            for ex in &best_examples {
                optimized_prompt
                    .push_str(&format!("\nInput: {}\nOutput: {}", ex.input, ex.expected));
            }
        }

        OptimizeResult {
            prompt: optimized_prompt,
            examples: best_examples,
            instruction: best_instruction,
            score: best_score,
            evaluations,
            candidate_scores,
        }
    }
}

/// Parse a numbered list from LLM output (e.g., "1. foo\n2. bar").
fn parse_numbered_list(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Match patterns like "1. ", "2) ", etc.
        if let Some(rest) = trimmed
            .strip_prefix(|c: char| c.is_ascii_digit())
            .and_then(|s| s.strip_prefix(|c: char| c == '.' || c == ')'))
            .map(|s| s.trim())
        {
            if !rest.is_empty() {
                results.push(rest.to_string());
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_builder() {
        let ds = Dataset::new()
            .example("What is 2+2?", "4")
            .example("What is 3+3?", "6")
            .labeled_example("Capital?", "Paris", "geography");

        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());
        assert_eq!(ds.examples[2].label.as_deref(), Some("geography"));
    }

    #[test]
    fn test_training_example() {
        let ex = TrainingExample::new("input", "output").with_label("test");
        assert_eq!(ex.input, "input");
        assert_eq!(ex.expected, "output");
        assert_eq!(ex.label.as_deref(), Some("test"));
    }

    #[test]
    fn test_parse_numbered_list() {
        let text = "1. First instruction\n2. Second instruction\n3. Third one";
        let results = parse_numbered_list(text);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], "First instruction");
        assert_eq!(results[1], "Second instruction");
        assert_eq!(results[2], "Third one");
    }

    #[test]
    fn test_parse_numbered_list_with_parentheses() {
        let text = "1) First\n2) Second";
        let results = parse_numbered_list(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "First");
    }

    #[test]
    fn test_empty_dataset() {
        let ds = Dataset::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_score_output_contains() {
        use crate::recursive::llm::MockLlm;
        let llm = MockLlm::new(|_, _| String::new());
        let opt = Optimizer::new(&llm, "test");

        // No metric or validator set — uses contains check
        assert!((opt.score_output("The answer is 42", "42") - 1.0).abs() < f64::EPSILON);
        assert!((opt.score_output("The answer is 41", "42") - 0.0).abs() < f64::EPSILON);
    }
}
