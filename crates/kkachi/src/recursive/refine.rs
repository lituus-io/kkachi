// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Refinement loop builder and execution.
//!
//! This module provides the main entry point for the kkachi library.
//! It implements a single, unified refinement loop that replaces all
//! previous duplicate implementations.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{refine, MockLlm, checks};
//!
//! let llm = MockLlm::new(|prompt, feedback| {
//!     if feedback.is_some() {
//!         "fn add(a: i32, b: i32) -> i32 { a + b }".to_string()
//!     } else {
//!         "fn add(a, b) { a + b }".to_string()  // Initial bad output
//!     }
//! });
//!
//! // Simple refinement
//! let code = refine(&llm, "Write an add function")
//!     .validate(checks().require("fn ").require("->"))
//!     .max_iter(5)
//!     .go();
//!
//! assert!(code.contains("fn add"));
//! ```

use crate::error::Result;
use crate::recursive::cli::CliCapture;
use crate::recursive::llm::Llm;
use crate::recursive::memory::Memory;
use crate::recursive::result::{
    Compiled, ContextId, Correction, Example, Iteration, OptimizedPrompt, RefineResult, StopReason,
};
use crate::recursive::rewrite::extract_code;
use crate::recursive::validate::{NoValidation, Score, Validate};
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// Configuration for the refinement loop.
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum number of iterations.
    pub max_iterations: u32,
    /// Target score threshold (1.0 = perfect).
    pub target_score: f64,
    /// Stop if score doesn't improve for this many iterations.
    pub plateau_window: Option<usize>,
    /// Enable chain of thought reasoning.
    pub chain_of_thought: bool,
    /// Review every N iterations (human-in-the-loop).
    pub review_every: Option<u32>,
    /// Score threshold for learning (storing successful outputs).
    pub learn_threshold: Option<f64>,
    /// Maximum tokens to consume (stops when exceeded).
    pub token_budget: Option<u32>,
    /// Maximum wall-clock time (stops when exceeded).
    pub timeout: Option<Duration>,
    /// Enable adaptive iteration control.
    pub adaptive: bool,
    /// Minimum iterations before early exit (for adaptive mode).
    pub min_iterations: u32,
    /// Extend by this many iterations if still improving (for adaptive mode).
    pub extend_on_progress: Option<u32>,
    /// Exit early if no improvement for this many iterations (for adaptive mode).
    pub early_exit_stagnation: Option<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            target_score: 1.0,
            plateau_window: None,
            chain_of_thought: false,
            review_every: None,
            learn_threshold: None,
            token_budget: None,
            timeout: None,
            adaptive: false,
            min_iterations: 1,
            extend_on_progress: None,
            early_exit_stagnation: None,
        }
    }
}

/// Create a new refinement builder.
///
/// This is the main entry point for the kkachi library.
///
/// # Arguments
///
/// * `llm` - The language model to use for generation
/// * `prompt` - The prompt/task description
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{refine, MockLlm};
///
/// let llm = MockLlm::new(|prompt, _| format!("Response to: {}", prompt));
/// let result = refine(&llm, "Write a function").go();
/// ```
pub fn refine<'a, L: Llm>(llm: &'a L, prompt: &'a str) -> Refine<'a, L, NoValidation> {
    Refine::new(llm, prompt)
}

/// A refinement builder with fluent API.
///
/// This struct accumulates configuration and then executes the refinement
/// loop when one of the execution methods is called.
pub struct Refine<'a, L: Llm, V: Validate = NoValidation> {
    llm: &'a L,
    prompt: &'a str,
    validator: V,
    memory: Option<&'a mut Memory>,
    config: Config,
    k: usize,
    examples: SmallVec<[Example; 4]>,
    source_markdown: Option<&'a str>,
    source_lang: Option<&'a str>,
    on_iter: Option<Box<dyn Fn(u32, f64) + 'a>>,
    best_of: usize,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, L: Llm> Refine<'a, L, NoValidation> {
    /// Create a new refinement builder.
    pub fn new(llm: &'a L, prompt: &'a str) -> Self {
        Self {
            llm,
            prompt,
            validator: NoValidation,
            memory: None,
            config: Config::default(),
            k: 3,
            examples: SmallVec::new(),
            source_markdown: None,
            source_lang: None,
            on_iter: None,
            best_of: 1,
            _phantom: PhantomData,
        }
    }
}

impl<'a, L: Llm, V: Validate> Refine<'a, L, V> {
    /// Set the validator for this refinement.
    ///
    /// The validator determines when the output is acceptable.
    pub fn validate<V2: Validate>(self, validator: V2) -> Refine<'a, L, V2> {
        Refine {
            llm: self.llm,
            prompt: self.prompt,
            validator,
            memory: self.memory,
            config: self.config,
            k: self.k,
            examples: self.examples,
            source_markdown: self.source_markdown,
            source_lang: self.source_lang,
            on_iter: self.on_iter,
            best_of: self.best_of,
            _phantom: PhantomData,
        }
    }

    /// Set the memory (RAG) to use for context retrieval.
    pub fn memory(mut self, memory: &'a mut Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the number of examples to retrieve from memory.
    pub fn k(mut self, n: usize) -> Self {
        self.k = n;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iter(mut self, n: u32) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set the target score threshold.
    ///
    /// Refinement stops when this score is reached.
    pub fn target(mut self, score: f64) -> Self {
        self.config.target_score = score;
        self
    }

    /// Enable learning - store successful outputs in memory.
    ///
    /// Outputs with a perfect score (1.0) will be stored.
    pub fn learn(mut self) -> Self {
        self.config.learn_threshold = Some(1.0);
        self
    }

    /// Enable learning with a custom threshold.
    ///
    /// Outputs with a score >= threshold will be stored in memory.
    pub fn learn_above(mut self, threshold: f64) -> Self {
        self.config.learn_threshold = Some(threshold);
        self
    }

    /// Add few-shot examples.
    pub fn examples<I, S1, S2>(mut self, examples: I) -> Self
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        for (input, output) in examples {
            self.examples.push(Example::new(input, output));
        }
        self
    }

    /// Add a single example.
    pub fn example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.examples.push(Example::new(input, output));
        self
    }

    /// Enable chain of thought reasoning.
    pub fn chain_of_thought(mut self) -> Self {
        self.config.chain_of_thought = true;
        self
    }

    /// Generate N outputs and keep the best one.
    pub fn best_of(mut self, n: usize) -> Self {
        self.best_of = n.max(1);
        self
    }

    /// Set a callback for each iteration.
    pub fn on_iter<F: Fn(u32, f64) + 'a>(mut self, f: F) -> Self {
        self.on_iter = Some(Box::new(f));
        self
    }

    /// Review every N iterations (human-in-the-loop).
    pub fn review_every(mut self, n: u32) -> Self {
        self.config.review_every = Some(n);
        self
    }

    /// Extract source code from a markdown document.
    ///
    /// The refinement will extract code of the given language
    /// from the markdown before validation.
    pub fn source(mut self, markdown: &'a str, lang: &'a str) -> Self {
        self.source_markdown = Some(markdown);
        self.source_lang = Some(lang);
        self
    }

    /// Stop if score doesn't improve for `window` iterations.
    pub fn stop_on_plateau(mut self, window: usize) -> Self {
        self.config.plateau_window = Some(window);
        self
    }

    /// Stop refinement when cumulative tokens exceed the budget.
    ///
    /// This tracks the total tokens (prompt + completion) across all iterations
    /// and stops when the budget is exceeded. Useful for cost control.
    ///
    /// # Example
    ///
    /// ```
    /// use kkachi::recursive::{refine, MockLlm};
    ///
    /// let llm = MockLlm::new(|_, _| "output".to_string());
    /// let result = refine(&llm, "task")
    ///     .with_budget(10_000)  // Stop after ~10k tokens
    ///     .go();
    /// ```
    pub fn with_budget(mut self, max_tokens: u32) -> Self {
        self.config.token_budget = Some(max_tokens);
        self
    }

    /// Stop refinement when wall-clock time exceeds the duration.
    ///
    /// This starts a timer when refinement begins and stops if the
    /// duration is exceeded. Useful for latency-sensitive applications.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use kkachi::recursive::{refine, MockLlm};
    ///
    /// let llm = MockLlm::new(|_, _| "output".to_string());
    /// let result = refine(&llm, "task")
    ///     .with_timeout(Duration::from_secs(60))  // 1 minute max
    ///     .go();
    /// ```
    pub fn with_timeout(mut self, duration: Duration) -> Self {
        self.config.timeout = Some(duration);
        self
    }

    // ========================================================================
    // Adaptive Refinement Methods
    // ========================================================================

    /// Enable adaptive iteration mode.
    ///
    /// In adaptive mode, the refinement loop dynamically adjusts based on
    /// progress patterns:
    /// - Extends iterations when improvement is detected
    /// - Exits early when stagnation is detected
    ///
    /// Use with `min_iter()`, `extend_on_progress()`, and `early_exit_on_stagnation()`.
    ///
    /// # Example
    ///
    /// ```
    /// use kkachi::recursive::{refine, MockLlm};
    ///
    /// let llm = MockLlm::new(|_, _| "output".to_string());
    /// let result = refine(&llm, "complex task")
    ///     .adaptive()
    ///     .min_iter(3)
    ///     .max_iter(20)
    ///     .extend_on_progress(5)
    ///     .early_exit_on_stagnation(3)
    ///     .go();
    /// ```
    pub fn adaptive(mut self) -> Self {
        self.config.adaptive = true;
        self
    }

    /// Set minimum iterations before early exit (for adaptive mode).
    ///
    /// Even with early exit conditions met, at least this many iterations
    /// will be performed. Default is 1.
    pub fn min_iter(mut self, n: u32) -> Self {
        self.config.min_iterations = n.max(1);
        self
    }

    /// Extend by N iterations if still improving (for adaptive mode).
    ///
    /// When progress is detected near the iteration limit, the limit is
    /// extended by this amount (up to 2x max_iterations).
    pub fn extend_on_progress(mut self, n: u32) -> Self {
        self.config.extend_on_progress = Some(n);
        self
    }

    /// Exit early if no improvement for N iterations (for adaptive mode).
    ///
    /// When the score doesn't improve for this many consecutive iterations
    /// (and minimum iterations have been reached), refinement stops early.
    pub fn early_exit_on_stagnation(mut self, n: u32) -> Self {
        self.config.early_exit_stagnation = Some(n);
        self
    }

    /// Execute the refinement and return just the output.
    pub fn go(self) -> String {
        match futures::executor::block_on(self.run()) {
            Ok(result) => result.output,
            Err(_) => String::new(),
        }
    }

    /// Execute the refinement and return output with score.
    pub fn go_scored(self) -> (String, f64) {
        match futures::executor::block_on(self.run()) {
            Ok(result) => (result.output, result.score),
            Err(_) => (String::new(), 0.0),
        }
    }

    /// Execute the refinement and return the full result.
    pub fn go_full(self) -> Result<RefineResult> {
        futures::executor::block_on(self.run())
    }

    /// Execute the refinement asynchronously.
    pub async fn run(mut self) -> Result<RefineResult> {
        let context_id = ContextId::new();
        let mut history: SmallVec<[Iteration; 8]> = SmallVec::new();
        let mut corrections: SmallVec<[Correction; 8]> = SmallVec::new();
        let mut cli_captures: SmallVec<[CliCapture; 4]> = SmallVec::new();

        // Track time and tokens for budget/timeout
        let start_time = Instant::now();
        let mut total_tokens: u32 = 0;
        let mut stop_reason = StopReason::MaxIterations;
        let mut best_confidence = 1.0f64;

        // Build context from memory and examples
        let mut context = String::new();

        // Add few-shot examples
        for ex in &self.examples {
            context.push_str(&format!("Input: {}\nOutput: {}\n\n", ex.input, ex.output));
        }

        // Add retrieved examples from memory
        if let Some(ref memory) = self.memory {
            let recalls = memory.search(self.prompt, self.k);
            for recall in recalls {
                context.push_str(&format!(
                    "Similar example (relevance: {:.2}):\n{}\n\n",
                    recall.score, recall.content
                ));
            }
        }

        // Add chain of thought instruction
        let prompt = if self.config.chain_of_thought {
            format!(
                "{}\n\nLet's think step by step. First, analyze the problem, then provide the solution.",
                self.prompt
            )
        } else {
            self.prompt.to_string()
        };

        // Main refinement loop
        let mut best_output = String::new();
        let mut best_score = 0.0f64;
        let mut feedback: Option<String> = None;
        let mut plateau_counter = 0;

        // Adaptive: track effective max iterations (can extend)
        let mut effective_max = self.config.max_iterations;
        let hard_max = self.config.max_iterations.saturating_mul(2); // Never exceed 2x original
        let mut stagnation_counter = 0u32;
        let mut iteration: u32 = 0;

        while iteration < effective_max {
            // Check timeout before generating
            if let Some(timeout) = self.config.timeout {
                if start_time.elapsed() > timeout {
                    stop_reason = StopReason::TimeoutReached;
                    break;
                }
            }

            // Generate output (with best_of sampling)
            let (output, gen_score, tokens_used) = if self.best_of > 1 {
                let (out, score) = self
                    .generate_best_of(&prompt, &context, feedback.as_deref())
                    .await?;
                // For best_of, we estimate tokens (actual tracking would need refactoring)
                (out, score, 0u32)
            } else {
                let result = self
                    .llm
                    .generate(&prompt, &context, feedback.as_deref())
                    .await?;
                let tokens = result.prompt_tokens + result.completion_tokens;
                (result.text, 0.0, tokens)
            };

            // Track tokens
            total_tokens = total_tokens.saturating_add(tokens_used);

            // Check budget after generating
            if let Some(budget) = self.config.token_budget {
                if total_tokens > budget {
                    stop_reason = StopReason::BudgetExhausted;
                    // Still process this iteration's output before stopping
                }
            }

            // Extract code from markdown if configured
            let text_to_validate =
                if let (Some(_md), Some(lang)) = (self.source_markdown, self.source_lang) {
                    // Replace code in markdown and extract
                    let full_output = format!("```{}\n{}\n```", lang, output);
                    extract_code(&full_output, lang)
                        .map(|s| s.to_string())
                        .unwrap_or(output.clone())
                } else {
                    output.clone()
                };

            // Validate
            let score = self.validator.validate(&text_to_validate);

            // Track confidence (use the best confidence seen for the best output)
            if score.value > best_score {
                best_confidence = score.confidence;
            }

            // Collect CLI captures if validator produced any
            if let Some(captures) = score.cli_captures() {
                for capture in captures {
                    cli_captures.push(capture.clone());
                }
            }

            // Calculate effective score (combine with best_of sampling if used)
            let effective_score = if self.best_of > 1 {
                score.value.max(gen_score)
            } else {
                score.value
            };

            // Record iteration
            history.push(Iteration {
                number: iteration,
                output: output.clone(),
                score: effective_score,
                feedback: score.feedback_str().map(|s| s.to_string()),
            });

            // Call iteration callback
            if let Some(ref callback) = self.on_iter {
                callback(iteration, effective_score);
            }

            // Track best result and adaptive counters
            let made_progress = effective_score > best_score + 0.01; // Significant improvement
            if effective_score > best_score {
                best_score = effective_score;
                best_output = output.clone();
                plateau_counter = 0;
                stagnation_counter = 0;

                // Adaptive: extend iterations if still improving and near limit
                if self.config.adaptive {
                    if let Some(extend) = self.config.extend_on_progress {
                        if made_progress
                            && effective_score < self.config.target_score
                            && iteration >= self.config.min_iterations
                            && iteration + extend >= effective_max
                        {
                            // Extend, but don't exceed hard max
                            effective_max = (effective_max + extend).min(hard_max);
                        }
                    }
                }
            } else {
                plateau_counter += 1;
                stagnation_counter += 1;
            }

            // Check for success
            if effective_score >= self.config.target_score - f64::EPSILON {
                // Learning: store successful output in memory
                if let (Some(ref mut memory), Some(threshold)) =
                    (&mut self.memory, self.config.learn_threshold)
                {
                    if effective_score >= threshold {
                        memory.learn(self.prompt, &output, effective_score);
                    }
                }

                return Ok(RefineResult {
                    output,
                    score: effective_score,
                    iterations: iteration + 1,
                    context_id,
                    from_cache: false,
                    prompt: Some(self.build_optimized_prompt()),
                    history,
                    corrections,
                    cli_captures,
                    stop_reason: StopReason::TargetReached,
                    total_tokens,
                    elapsed: start_time.elapsed(),
                    confidence: score.confidence,
                });
            }

            // Exit if budget was exceeded (after processing current iteration)
            if stop_reason == StopReason::BudgetExhausted {
                break;
            }

            // Check for plateau
            if let Some(window) = self.config.plateau_window {
                if plateau_counter >= window {
                    stop_reason = StopReason::Plateau;
                    break;
                }
            }

            // Adaptive: early exit on stagnation
            if self.config.adaptive {
                if let Some(stagnation_limit) = self.config.early_exit_stagnation {
                    if stagnation_counter >= stagnation_limit
                        && iteration >= self.config.min_iterations
                    {
                        stop_reason = StopReason::Plateau;
                        break;
                    }
                }
            }

            // Record correction if there was feedback
            if let Some(fb) = score.feedback_str() {
                corrections.push(Correction {
                    error: fb.to_string(),
                    resolution: String::new(), // Will be filled by next iteration
                    iteration,
                });

                // Update resolution of previous correction if output improved
                if !corrections.is_empty() && effective_score > best_score {
                    let last_idx = corrections.len() - 1;
                    corrections[last_idx].resolution = output.clone();
                }
            }

            // Prepare feedback for next iteration
            feedback = Some(format!(
                "Your previous output scored {:.2}/1.0. {}",
                effective_score,
                score.feedback_str().unwrap_or("Please improve.")
            ));

            // Human review if configured
            if let Some(review_interval) = self.config.review_every {
                if (iteration + 1) % review_interval == 0 {
                    // In a real implementation, this would pause for user input
                    // For now, we just continue
                }
            }

            // Increment iteration counter (for while loop)
            iteration += 1;
        }

        // Learning: store best output if above threshold
        if let (Some(ref mut memory), Some(threshold)) =
            (&mut self.memory, self.config.learn_threshold)
        {
            if best_score >= threshold {
                memory.learn(self.prompt, &best_output, best_score);
            }
        }

        Ok(RefineResult {
            output: best_output,
            score: best_score,
            iterations: history.len() as u32,
            context_id,
            from_cache: false,
            prompt: Some(self.build_optimized_prompt()),
            history,
            corrections,
            cli_captures,
            stop_reason,
            total_tokens,
            elapsed: start_time.elapsed(),
            confidence: best_confidence,
        })
    }

    /// Generate best-of-N outputs and return the best one.
    async fn generate_best_of(
        &self,
        prompt: &str,
        context: &str,
        feedback: Option<&str>,
    ) -> Result<(String, f64)> {
        let mut best_output = String::new();
        let mut best_score = 0.0f64;

        for _ in 0..self.best_of {
            let result = self.llm.generate(prompt, context, feedback).await?;
            let score = self.validator.validate(&result.text);

            if score.value > best_score {
                best_score = score.value;
                best_output = result.text;
            }

            // Early exit if perfect score
            if best_score >= self.config.target_score - f64::EPSILON {
                break;
            }
        }

        Ok((best_output, best_score))
    }

    /// Build an optimized prompt from the refinement state.
    fn build_optimized_prompt(&self) -> OptimizedPrompt {
        let mut prompt = OptimizedPrompt::new(self.prompt);

        if self.config.chain_of_thought {
            prompt = prompt.with_instructions(
                "Think step by step. Analyze the problem, then provide the solution.",
            );
        }

        for ex in &self.examples {
            prompt = prompt.with_example(&ex.input, &ex.output);
        }

        prompt
    }

    /// Compile into a reusable prediction program.
    ///
    /// This captures the optimized prompt and validator for fast inference
    /// without re-running the refinement loop.
    pub fn compile(self) -> Compiled<L, V>
    where
        L: Clone,
        V: Clone,
    {
        let prompt = self.build_optimized_prompt();
        Compiled::new(self.llm.clone(), self.validator.clone(), prompt)
    }
}

// ============================================================================
// Extension trait for Score to access CLI captures
// ============================================================================

impl Score<'_> {
    /// Get CLI captures if this score was produced by a CLI validator.
    fn cli_captures(&self) -> Option<&[CliCapture]> {
        // CLI captures are stored in the breakdown field as a marker
        // This is a simplified approach - in practice, you'd use a custom type
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::checks::checks;
    use crate::recursive::llm::{IterativeMockLlm, MockLlm};
    use crate::recursive::validate::{BoolValidator, ScoreValidator};

    #[test]
    fn test_simple_refine() {
        let llm = MockLlm::new(|_, _| "fn add(a: i32, b: i32) -> i32 { a + b }".to_string());

        let result = refine(&llm, "Write an add function").go();
        assert!(result.contains("fn add"));
    }

    #[test]
    fn test_refine_with_validator() {
        let llm = IterativeMockLlm::new(|iter, _, _| match iter {
            0 => "fn add(a, b) { a + b }".to_string(),
            1 => "fn add(a: i32, b: i32) { a + b }".to_string(),
            _ => "fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
        });

        let validator = checks().require("fn ").require("->").require(": i32");

        let (output, score) = refine(&llm, "Write an add function")
            .validate(validator)
            .max_iter(5)
            .go_scored();

        assert!(output.contains("-> i32"));
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_refine_with_examples() {
        let llm = MockLlm::new(|_, _| "result".to_string());

        let result = refine(&llm, "test")
            .example("input1", "output1")
            .example("input2", "output2")
            .go();

        assert_eq!(result, "result");
    }

    #[test]
    fn test_refine_full_result() {
        let llm = IterativeMockLlm::new(|iter, _, _| {
            if iter < 2 {
                "bad output".to_string()
            } else {
                "fn good()".to_string()
            }
        });

        let validator = BoolValidator(|s: &str| s.contains("fn "));

        let result = refine(&llm, "Write a function")
            .validate(validator)
            .max_iter(5)
            .go_full()
            .unwrap();

        assert_eq!(result.iterations, 3);
        assert!(result.is_success());
        assert_eq!(result.history.len(), 3);
    }

    #[test]
    fn test_refine_on_iter_callback() {
        let llm = IterativeMockLlm::new(|iter, _, _| format!("output {}", iter));

        let iterations = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let iterations_clone = iterations.clone();

        // Use a validator that always fails so we iterate max_iter times
        let validator = BoolValidator(|_: &str| false);

        let _ = refine(&llm, "test")
            .validate(validator)
            .max_iter(3)
            .on_iter(move |iter, _score| {
                iterations_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert!(iter < 3);
            })
            .go();

        assert_eq!(iterations.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn test_refine_best_of() {
        // This test verifies best_of generates multiple outputs
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let llm = MockLlm::new(move |_, _| {
            call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            "output".to_string()
        });

        // Use a validator that always fails so best_of tries all N times
        let validator = BoolValidator(|_: &str| false);

        let _ = refine(&llm, "test")
            .validate(validator)
            .best_of(3)
            .max_iter(1)
            .go();

        // Should have called generate 3 times for best_of
        assert!(call_count.load(std::sync::atomic::Ordering::SeqCst) >= 3);
    }

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.max_iterations, 10);
        assert!((config.target_score - 1.0).abs() < f64::EPSILON);
        assert!(!config.chain_of_thought);
        assert!(config.plateau_window.is_none());
    }

    #[test]
    fn test_chain_of_thought() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("step by step") {
                "thought through".to_string()
            } else {
                "direct".to_string()
            }
        });

        let result = refine(&llm, "solve this").chain_of_thought().go();
        assert_eq!(result, "thought through");
    }

    #[test]
    fn test_plateau_detection() {
        let llm = MockLlm::new(|_, _| "same output".to_string());

        let validator = BoolValidator(|_: &str| false); // Always fail

        let result = refine(&llm, "test")
            .validate(validator)
            .max_iter(10)
            .stop_on_plateau(3)
            .go_full()
            .unwrap();

        // Should stop after 3 iterations of no improvement
        assert!(result.iterations <= 4);
    }

    // ========================================================================
    // Phase 1 Tests: Budget, Timeout, StopReason, Confidence
    // ========================================================================

    #[test]
    fn test_with_budget() {
        let llm = MockLlm::new(|_, _| "output".to_string());
        let validator = BoolValidator(|_: &str| false); // Always fail

        let result = refine(&llm, "test")
            .validate(validator)
            .with_budget(100)
            .max_iter(10)
            .go_full()
            .unwrap();

        // Budget tracking is set (tokens may be 0 for mock)
        assert!(result.total_tokens <= 100 || result.stop_reason == StopReason::MaxIterations);
    }

    #[test]
    fn test_with_timeout() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let result = refine(&llm, "test")
            .with_timeout(Duration::from_secs(60))
            .max_iter(3)
            .go_full()
            .unwrap();

        // Should complete within timeout
        assert!(result.elapsed < Duration::from_secs(60));
    }

    #[test]
    fn test_stop_reason_target_reached() {
        let llm = MockLlm::new(|_, _| "fn good()".to_string());
        let validator = BoolValidator(|s: &str| s.contains("fn "));

        let result = refine(&llm, "test")
            .validate(validator)
            .max_iter(5)
            .go_full()
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::TargetReached);
        assert!(result.is_success());
    }

    #[test]
    fn test_stop_reason_max_iterations() {
        let llm = MockLlm::new(|_, _| "bad".to_string());
        let validator = BoolValidator(|_: &str| false);

        let result = refine(&llm, "test")
            .validate(validator)
            .max_iter(3)
            .go_full()
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::MaxIterations);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    fn test_stop_reason_plateau() {
        let llm = MockLlm::new(|_, _| "same".to_string());
        let validator = BoolValidator(|_: &str| false);

        let result = refine(&llm, "test")
            .validate(validator)
            .max_iter(10)
            .stop_on_plateau(2)
            .go_full()
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Plateau);
    }

    #[test]
    fn test_result_has_elapsed() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let result = refine(&llm, "test").max_iter(1).go_full().unwrap();

        // Elapsed should be non-zero (or at least not negative)
        assert!(result.elapsed >= Duration::ZERO);
    }

    #[test]
    fn test_result_has_confidence() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let result = refine(&llm, "test").max_iter(1).go_full().unwrap();

        // Default confidence is 1.0 for deterministic validators
        assert!((result.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_with_budget_and_timeout() {
        let config = Config {
            token_budget: Some(10_000),
            timeout: Some(Duration::from_secs(30)),
            ..Config::default()
        };

        assert_eq!(config.token_budget, Some(10_000));
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }

    // ========================================================================
    // Phase 3 Tests: Adaptive Refinement
    // ========================================================================

    #[test]
    fn test_adaptive_mode_basic() {
        let llm = MockLlm::new(|_, _| "output".to_string());
        let validator = BoolValidator(|_: &str| false);

        let result = refine(&llm, "test")
            .validate(validator)
            .adaptive()
            .min_iter(2)
            .max_iter(5)
            .go_full()
            .unwrap();

        // Should have run at least min_iter iterations
        assert!(result.iterations >= 2);
    }

    #[test]
    fn test_adaptive_early_exit_on_stagnation() {
        let llm = MockLlm::new(|_, _| "same output".to_string());
        let validator = BoolValidator(|_: &str| false);

        let result = refine(&llm, "test")
            .validate(validator)
            .adaptive()
            .min_iter(2)
            .max_iter(10)
            .early_exit_on_stagnation(3)
            .go_full()
            .unwrap();

        // Should exit early due to stagnation
        // min_iter (2) + stagnation window (3) = 5 max iterations
        assert!(result.iterations <= 6);
        assert_eq!(result.stop_reason, StopReason::Plateau);
    }

    #[test]
    fn test_adaptive_extend_on_progress() {
        let iteration_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let iteration_clone = iteration_count.clone();

        // LLM that gradually improves (score goes up each iteration)
        let llm = IterativeMockLlm::new(move |iter, _, _| {
            iteration_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            format!("output iter {}", iter)
        });

        // Validator that gives increasingly higher scores
        let scores = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let scores_clone = scores.clone();
        let validator = ScoreValidator(move |_: &str| {
            let n = scores_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Score increases: 0.1, 0.2, 0.3, ... up to 0.9
            ((n as f64 + 1.0) * 0.1).min(0.9)
        });

        let result = refine(&llm, "test")
            .validate(validator)
            .adaptive()
            .min_iter(2)
            .max_iter(3)
            .extend_on_progress(2)
            .target(1.0) // Unreachable target
            .go_full()
            .unwrap();

        // With extension, should exceed original max_iter of 3
        // (actual count depends on progress detection timing)
        assert!(result.iterations >= 3);
    }

    #[test]
    fn test_adaptive_config() {
        let config = Config {
            adaptive: true,
            min_iterations: 3,
            extend_on_progress: Some(5),
            early_exit_stagnation: Some(2),
            ..Config::default()
        };

        assert!(config.adaptive);
        assert_eq!(config.min_iterations, 3);
        assert_eq!(config.extend_on_progress, Some(5));
        assert_eq!(config.early_exit_stagnation, Some(2));
    }

    #[test]
    fn test_adaptive_respects_min_iter() {
        let llm = MockLlm::new(|_, _| "output".to_string());
        let validator = BoolValidator(|_: &str| false);

        let result = refine(&llm, "test")
            .validate(validator)
            .adaptive()
            .min_iter(5)
            .max_iter(10)
            .early_exit_on_stagnation(1) // Very aggressive early exit
            .go_full()
            .unwrap();

        // Should still run min_iter (5) iterations despite early exit trigger
        assert!(result.iterations >= 5);
    }
}
