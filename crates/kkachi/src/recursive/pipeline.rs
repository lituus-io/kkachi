// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Pipeline composition for chaining operations.
//!
//! This module provides the [`pipeline`] entry point for composing multiple
//! operations (refine, extract, best_of, ensemble, reason, program) into a
//! single pipeline where each step feeds its output to the next.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, pipeline, checks};
//!
//! let llm = MockLlm::new(|prompt, _| {
//!     if prompt.contains("```") {
//!         "fn add(a: i32, b: i32) -> i32 { a + b }".to_string()
//!     } else {
//!         "```rust\nfn add(a: i32, b: i32) -> i32 { a + b }\n```".to_string()
//!     }
//! });
//!
//! let result = pipeline(&llm, "Write an add function in Rust")
//!     .refine(checks().require("```"))
//!     .extract("rust")
//!     .go();
//!
//! assert!(result.output.contains("fn add"));
//! assert_eq!(result.steps.len(), 2);
//! ```

use crate::recursive::best_of::{self, FnScorer, Scorer};
use crate::recursive::ensemble::{self, Aggregate};
use crate::recursive::executor::{CodeExecutor, DynCodeExecutor};
use crate::recursive::llm::Llm;
use crate::recursive::program;
use crate::recursive::reason;
use crate::recursive::refine;
use crate::recursive::rewrite::extract_code;
use crate::recursive::shared;
use crate::recursive::validate::Validate;
use futures::stream::{FuturesUnordered, StreamExt};
use std::borrow::Cow;
use std::future::Future;
use std::time::{Duration, Instant};

/// Entry point for pipeline composition.
///
/// Creates a builder that chains multiple operations together, feeding
/// each step's output into the next.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, pipeline, checks};
///
/// let llm = MockLlm::new(|prompt, _| {
///     "def is_prime(n): return n > 1 and all(n % i for i in range(2, n))".to_string()
/// });
///
/// let result = pipeline(&llm, "Write an is_prime function")
///     .refine(checks().require("def "))
///     .go();
///
/// assert!(result.output.contains("def is_prime"));
/// ```
pub fn pipeline<'a, L: Llm>(llm: &'a L, prompt: &'a str) -> Pipeline<'a, L> {
    Pipeline::new(llm, prompt)
}

/// Internal representation of a fan-out branch.
struct FanOutBranch {
    name: String,
    steps: Vec<PipelineStep>,
}

/// A single step in the pipeline.
enum PipelineStep {
    /// Refine the current output with a validator.
    Refine {
        validator: Box<dyn Validate>,
        max_iter: u32,
        target: f64,
    },
    /// Extract code from markdown fences.
    Extract { lang: String },
    /// Generate N candidates and pick the best.
    BestOf {
        n: usize,
        scorer: Option<Box<dyn Scorer>>,
    },
    /// Generate N responses and aggregate.
    Ensemble { n: usize, aggregate: Aggregate },
    /// Chain of thought reasoning.
    Reason,
    /// Generate and execute code.
    Program { executor: Box<dyn DynCodeExecutor> },
    /// Apply a transformation function.
    Map {
        f: Box<dyn Fn(String) -> String + Send + Sync>,
    },
    /// Fan-out: run multiple branches in parallel, then merge results.
    FanOut {
        branches: Vec<FanOutBranch>,
        merge: MergeStrategy,
    },
    /// A nested pipeline whose steps run inline.
    Nested { steps: Vec<PipelineStep> },
}

/// Result of a single pipeline step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Name of the step (e.g., "refine", "extract", "best_of").
    pub name: String,
    /// Input to this step.
    pub input: String,
    /// Output from this step.
    pub output: String,
    /// Score from this step (if applicable).
    pub score: Option<f64>,
    /// Tokens used by this step.
    pub tokens: u32,
    /// Duration of this step.
    pub elapsed: Duration,
}

/// Result of a complete pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The final output from the last step.
    pub output: String,
    /// Results from each step.
    pub steps: Vec<StepResult>,
    /// Total tokens used across all steps.
    pub total_tokens: u32,
    /// Total elapsed time for the pipeline.
    pub elapsed: Duration,
}

/// Events emitted during pipeline streaming via `run_stream()`.
///
/// Use these events to observe pipeline progress in real-time.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// A pipeline step has started.
    StepStart {
        /// The step index (0-indexed).
        index: usize,
        /// The step name (e.g., "refine", "extract").
        name: String,
    },
    /// A pipeline step has completed.
    StepComplete {
        /// The step index (0-indexed).
        index: usize,
        /// The result of this step.
        result: StepResult,
    },
    /// A fan-out branch has started.
    FanOutBranchStart {
        /// Parent step index.
        step_index: usize,
        /// Branch index within the fan-out.
        branch_index: usize,
        /// Branch name.
        branch_name: String,
    },
    /// A fan-out branch has completed.
    FanOutBranchComplete {
        /// Parent step index.
        step_index: usize,
        /// Branch result.
        branch_result: FanOutBranchResult,
    },
    /// The pipeline completed.
    Complete(PipelineResult),
}

/// Strategy for merging parallel fan-out branch results into a single output.
pub enum MergeStrategy {
    /// Use the first branch that produces non-empty output.
    FirstSuccess,
    /// Use the branch with the highest score.
    BestScore,
    /// Concatenate all outputs with a separator.
    Concat {
        /// Separator between branch outputs.
        separator: String,
    },
    /// Custom merge function applied to all branch results.
    Custom(Box<dyn Fn(&[FanOutBranchResult]) -> String + Send + Sync>),
}

impl Clone for MergeStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::FirstSuccess => Self::FirstSuccess,
            Self::BestScore => Self::BestScore,
            Self::Concat { separator } => Self::Concat {
                separator: separator.clone(),
            },
            Self::Custom(_) => {
                panic!("MergeStrategy::Custom cannot be cloned; use a named variant or build a new Custom")
            }
        }
    }
}

impl std::fmt::Debug for MergeStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstSuccess => write!(f, "FirstSuccess"),
            Self::BestScore => write!(f, "BestScore"),
            Self::Concat { separator } => write!(f, "Concat({:?})", separator),
            Self::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}

/// Result from a single fan-out branch.
#[derive(Debug, Clone)]
pub struct FanOutBranchResult {
    /// Branch index (0-based).
    pub index: usize,
    /// Name/label for this branch.
    pub name: String,
    /// Final output of the branch sub-pipeline.
    pub output: String,
    /// Best score from any step in this branch.
    pub score: Option<f64>,
    /// Total tokens used by this branch.
    pub tokens: u32,
    /// Duration for this branch.
    pub elapsed: Duration,
}

/// Builder for defining the steps within a single fan-out branch.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{BranchBuilder, checks};
///
/// let branch = BranchBuilder::new("rust")
///     .refine(checks().require("fn ").require("->"))
///     .extract("rust");
/// ```
pub struct BranchBuilder {
    name: String,
    steps: Vec<PipelineStep>,
}

impl BranchBuilder {
    /// Create a new branch with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            steps: Vec::new(),
        }
    }

    /// Add a refinement step to this branch.
    pub fn refine(mut self, validator: impl Validate + 'static) -> Self {
        self.steps.push(PipelineStep::Refine {
            validator: Box::new(validator),
            max_iter: 5,
            target: 1.0,
        });
        self
    }

    /// Add a refinement step with custom settings.
    pub fn refine_with(
        mut self,
        validator: impl Validate + 'static,
        max_iter: u32,
        target: f64,
    ) -> Self {
        self.steps.push(PipelineStep::Refine {
            validator: Box::new(validator),
            max_iter,
            target,
        });
        self
    }

    /// Add a code extraction step.
    pub fn extract(mut self, lang: &str) -> Self {
        self.steps.push(PipelineStep::Extract {
            lang: lang.to_string(),
        });
        self
    }

    /// Add a best-of-N step.
    pub fn best_of(mut self, n: usize) -> Self {
        self.steps.push(PipelineStep::BestOf { n, scorer: None });
        self
    }

    /// Add an ensemble step.
    pub fn ensemble(mut self, n: usize) -> Self {
        self.steps.push(PipelineStep::Ensemble {
            n,
            aggregate: Aggregate::MajorityVote,
        });
        self
    }

    /// Add a chain-of-thought reasoning step.
    pub fn reason(mut self) -> Self {
        self.steps.push(PipelineStep::Reason);
        self
    }

    /// Add a map/transform step.
    pub fn map(mut self, f: impl Fn(String) -> String + Send + Sync + 'static) -> Self {
        self.steps.push(PipelineStep::Map { f: Box::new(f) });
        self
    }

    fn build(self) -> FanOutBranch {
        FanOutBranch {
            name: self.name,
            steps: self.steps,
        }
    }
}

/// Collector for the closure-based `fan_out_with` API.
pub struct FanOutCollector {
    branches: Vec<FanOutBranch>,
}

impl FanOutCollector {
    fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    /// Add a branch with a builder closure.
    pub fn branch<F>(mut self, name: &str, f: F) -> Self
    where
        F: FnOnce(BranchBuilder) -> BranchBuilder,
    {
        let builder = f(BranchBuilder::new(name));
        self.branches.push(builder.build());
        self
    }
}

/// Pipeline composition builder.
///
/// Chains multiple operations together, where each step's output
/// becomes the next step's input prompt.
pub struct Pipeline<'a, L: Llm> {
    llm: &'a L,
    prompt: Cow<'a, str>,
    steps: Vec<PipelineStep>,
}

impl<'a, L: Llm> Pipeline<'a, L> {
    /// Create a new pipeline with the given LLM and initial prompt.
    pub fn new(llm: &'a L, prompt: &'a str) -> Self {
        Self {
            llm,
            prompt: Cow::Borrowed(prompt),
            steps: Vec::new(),
        }
    }

    /// Create a new pipeline with an owned prompt string.
    ///
    /// Use this when the prompt is not borrowed from a longer-lived scope
    /// (e.g., when building pipelines inside closures).
    pub fn new_owned(llm: &'a L, prompt: String) -> Self {
        Self {
            llm,
            prompt: Cow::Owned(prompt),
            steps: Vec::new(),
        }
    }

    /// Add a refinement step with the given validator.
    ///
    /// The current output is refined until it satisfies the validator
    /// or max iterations are reached.
    ///
    /// # Examples
    ///
    /// ```
    /// use kkachi::recursive::{MockLlm, pipeline, checks};
    ///
    /// let llm = MockLlm::new(|_, _| "fn main() {}".to_string());
    /// let result = pipeline(&llm, "Write code")
    ///     .refine(checks().require("fn "))
    ///     .go();
    /// ```
    pub fn refine(mut self, validator: impl Validate + 'static) -> Self {
        self.steps.push(PipelineStep::Refine {
            validator: Box::new(validator),
            max_iter: 5,
            target: 1.0,
        });
        self
    }

    /// Add a refinement step with custom iteration limit and target score.
    pub fn refine_with(
        mut self,
        validator: impl Validate + 'static,
        max_iter: u32,
        target: f64,
    ) -> Self {
        self.steps.push(PipelineStep::Refine {
            validator: Box::new(validator),
            max_iter,
            target,
        });
        self
    }

    /// Add a code extraction step.
    ///
    /// Extracts code from markdown fences matching the given language.
    /// If no fenced block is found, the input passes through unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use kkachi::recursive::{MockLlm, pipeline, checks};
    ///
    /// let llm = MockLlm::new(|_, _| "```rust\nfn main() {}\n```".to_string());
    /// let result = pipeline(&llm, "Write code")
    ///     .refine(checks().require("```"))
    ///     .extract("rust")
    ///     .go();
    /// assert_eq!(result.output.trim(), "fn main() {}");
    /// ```
    pub fn extract(mut self, lang: &str) -> Self {
        self.steps.push(PipelineStep::Extract {
            lang: lang.to_string(),
        });
        self
    }

    /// Add a best-of-N step.
    ///
    /// Generates N candidates from the current prompt and selects the best.
    pub fn best_of(mut self, n: usize) -> Self {
        self.steps.push(PipelineStep::BestOf { n, scorer: None });
        self
    }

    /// Add a best-of-N step with a custom scorer.
    pub fn best_of_scored(
        mut self,
        n: usize,
        scorer: impl Fn(&str) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.steps.push(PipelineStep::BestOf {
            n,
            scorer: Some(Box::new(FnScorer(scorer))),
        });
        self
    }

    /// Add an ensemble step.
    ///
    /// Generates N responses and aggregates them using majority vote.
    pub fn ensemble(mut self, n: usize) -> Self {
        self.steps.push(PipelineStep::Ensemble {
            n,
            aggregate: Aggregate::MajorityVote,
        });
        self
    }

    /// Add an ensemble step with a custom aggregation strategy.
    pub fn ensemble_with(mut self, n: usize, aggregate: Aggregate) -> Self {
        self.steps.push(PipelineStep::Ensemble { n, aggregate });
        self
    }

    /// Add a chain-of-thought reasoning step.
    ///
    /// Guides the LLM to think step by step before answering.
    pub fn reason(mut self) -> Self {
        self.steps.push(PipelineStep::Reason);
        self
    }

    /// Add a program-of-thought step.
    ///
    /// Generates code to solve the problem, executes it, and returns the output.
    pub fn program(mut self, executor: impl CodeExecutor + 'static) -> Self {
        self.steps.push(PipelineStep::Program {
            executor: Box::new(executor),
        });
        self
    }

    /// Add a map/transform step.
    ///
    /// Applies a function to the current output.
    ///
    /// # Examples
    ///
    /// ```
    /// use kkachi::recursive::{MockLlm, pipeline, checks};
    ///
    /// let llm = MockLlm::new(|_, _| "  hello world  ".to_string());
    /// let result = pipeline(&llm, "greet")
    ///     .refine(checks().min_len(1))
    ///     .map(|s| s.trim().to_uppercase())
    ///     .go();
    /// assert_eq!(result.output, "HELLO WORLD");
    /// ```
    pub fn map(mut self, f: impl Fn(String) -> String + Send + Sync + 'static) -> Self {
        self.steps.push(PipelineStep::Map { f: Box::new(f) });
        self
    }

    /// Add a fan-out step that runs multiple branches in parallel.
    ///
    /// Each branch receives the previous step's output as its input.
    /// The merge strategy determines how to combine the branch outputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use kkachi::recursive::{MockLlm, pipeline, checks, BranchBuilder, MergeStrategy};
    ///
    /// let llm = MockLlm::new(|prompt, _| {
    ///     if prompt.contains("short") { "brief".to_string() }
    ///     else { "a longer response here".to_string() }
    /// });
    ///
    /// let result = pipeline(&llm, "Write something")
    ///     .fan_out(vec![
    ///         BranchBuilder::new("short").map(|s| format!("short: {}", s)),
    ///         BranchBuilder::new("long").map(|s| format!("long: {}", s)),
    ///     ], MergeStrategy::FirstSuccess)
    ///     .go();
    /// ```
    pub fn fan_out(mut self, branches: Vec<BranchBuilder>, merge: MergeStrategy) -> Self {
        self.steps.push(PipelineStep::FanOut {
            branches: branches.into_iter().map(|b| b.build()).collect(),
            merge,
        });
        self
    }

    /// Add a fan-out step with a closure-based branch builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use kkachi::recursive::{MockLlm, pipeline, checks, MergeStrategy};
    ///
    /// let llm = MockLlm::new(|_, _| "output".to_string());
    ///
    /// let result = pipeline(&llm, "prompt")
    ///     .fan_out_with(MergeStrategy::Concat { separator: "\n".into() }, |fan| fan
    ///         .branch("a", |b| b.map(|s| format!("A: {}", s)))
    ///         .branch("b", |b| b.map(|s| format!("B: {}", s)))
    ///     )
    ///     .go();
    /// ```
    pub fn fan_out_with<F>(mut self, merge: MergeStrategy, f: F) -> Self
    where
        F: FnOnce(FanOutCollector) -> FanOutCollector,
    {
        let collector = f(FanOutCollector::new());
        self.steps.push(PipelineStep::FanOut {
            branches: collector.branches,
            merge,
        });
        self
    }

    /// Execute the pipeline synchronously.
    pub fn go(self) -> PipelineResult {
        shared::block_on(self.run())
    }

    /// Execute the pipeline as a stream of events.
    ///
    /// Returns an async stream that yields [`PipelineEvent`] items as each
    /// step starts and completes, enabling real-time progress monitoring.
    pub fn run_stream(self) -> impl futures::stream::Stream<Item = PipelineEvent> + 'a {
        async_stream::stream! {
            let start = Instant::now();
            let mut current_output = String::new();
            let mut step_results: Vec<StepResult> = Vec::with_capacity(self.steps.len());
            let mut total_tokens: u32 = 0;

            let llm = self.llm;
            let prompt = self.prompt;
            let pipeline_steps = self.steps;

            if pipeline_steps.is_empty() {
                let step_start = Instant::now();
                if let Ok(output) = llm.generate(&prompt, "", None).await {
                    let tokens = output.total_tokens();
                    total_tokens += tokens;
                    current_output = output.text.to_string();
                    let sr = StepResult {
                        name: "generate".to_string(),
                        input: prompt.to_string(),
                        output: current_output.clone(),
                        score: None,
                        tokens,
                        elapsed: step_start.elapsed(),
                    };
                    step_results.push(sr.clone());
                    yield PipelineEvent::StepComplete { index: 0, result: sr };
                }
            }

            for (i, step) in pipeline_steps.into_iter().enumerate() {
                let step_name = match &step {
                    PipelineStep::Refine { .. } => "refine",
                    PipelineStep::Extract { .. } => "extract",
                    PipelineStep::BestOf { .. } => "best_of",
                    PipelineStep::Ensemble { .. } => "ensemble",
                    PipelineStep::Reason => "reason",
                    PipelineStep::Program { .. } => "program",
                    PipelineStep::Map { .. } => "map",
                    PipelineStep::FanOut { .. } => "fan_out",
                    PipelineStep::Nested { .. } => "nested",
                };
                yield PipelineEvent::StepStart { index: i, name: step_name.to_string() };

                let step_start = Instant::now();
                let input = if i == 0 {
                    prompt.to_string()
                } else {
                    current_output.clone()
                };

                let (name, step_output) = match step {
                    PipelineStep::Refine { validator, max_iter, target } => (
                        "refine",
                        run_refine(llm, &input, validator, max_iter, target).await,
                    ),
                    PipelineStep::Extract { lang } => {
                        ("extract", run_extract(&input, &lang))
                    }
                    PipelineStep::BestOf { n, scorer } => {
                        ("best_of", run_best_of(llm, &input, n, scorer).await)
                    }
                    PipelineStep::Ensemble { n, aggregate } => {
                        ("ensemble", run_ensemble(llm, &input, n, aggregate).await)
                    }
                    PipelineStep::Reason => ("reason", run_reason(llm, &input).await),
                    PipelineStep::Program { executor } => {
                        ("program", run_program(llm, &input, executor).await)
                    }
                    PipelineStep::Map { f } => {
                        let output = f(input.clone());
                        ("map", StepOutput { output, score: None, tokens: 0 })
                    }
                    PipelineStep::FanOut { branches, merge } => {
                        ("fan_out", run_fan_out(llm, &input, branches, &merge).await)
                    }
                    PipelineStep::Nested { steps } => {
                        ("nested", {
                        let br = run_branch_steps(llm, input.clone(), steps).await;
                        StepOutput { output: br.output, score: br.best_score, tokens: br.tokens }
                    })
                    }
                };

                total_tokens += step_output.tokens;
                current_output = step_output.output.clone();

                let sr = StepResult {
                    name: name.to_string(),
                    input,
                    output: step_output.output,
                    score: step_output.score,
                    tokens: step_output.tokens,
                    elapsed: step_start.elapsed(),
                };
                step_results.push(sr.clone());
                yield PipelineEvent::StepComplete { index: i, result: sr };
            }

            yield PipelineEvent::Complete(PipelineResult {
                output: current_output,
                steps: step_results,
                total_tokens,
                elapsed: start.elapsed(),
            });
        }
    }

    /// Nest another pipeline as a step.
    ///
    /// The inner pipeline runs with the current output as its prompt
    /// and its final output becomes the next step's input.
    ///
    /// # Examples
    ///
    /// ```
    /// use kkachi::recursive::{MockLlm, pipeline, checks};
    ///
    /// let llm = MockLlm::new(|_, _| "nested output".to_string());
    ///
    /// let inner = pipeline(&llm, "ignored — overridden by outer");
    /// let result = pipeline(&llm, "outer prompt")
    ///     .refine(checks().min_len(1))
    ///     .nest(inner)
    ///     .go();
    /// ```
    pub fn nest(mut self, _inner: Pipeline<'a, L>) -> Self {
        self.steps.push(PipelineStep::Nested {
            steps: _inner.steps,
        });
        self
    }

    /// Execute the pipeline asynchronously.
    pub async fn run(self) -> PipelineResult {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("pipeline", steps = self.steps.len()).entered();

        let start = Instant::now();
        let mut current_output = String::new();
        let mut step_results: Vec<StepResult> = Vec::with_capacity(self.steps.len());
        let mut total_tokens: u32 = 0;

        let llm = self.llm;
        let prompt = self.prompt;
        let pipeline_steps = self.steps;

        // If no steps, just generate once
        if pipeline_steps.is_empty() {
            let step_start = Instant::now();
            if let Ok(output) = llm.generate(&prompt, "", None).await {
                let tokens = output.total_tokens();
                total_tokens += tokens;
                current_output = output.text.to_string();
                step_results.push(StepResult {
                    name: "generate".to_string(),
                    input: prompt.to_string(),
                    output: current_output.clone(),
                    score: None,
                    tokens,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        for (i, step) in pipeline_steps.into_iter().enumerate() {
            let step_start = Instant::now();
            // For the first step, use the initial prompt; thereafter use previous output
            let input = if i == 0 {
                prompt.to_string()
            } else {
                current_output.clone()
            };

            let (name, step_output) = match step {
                PipelineStep::Refine {
                    validator,
                    max_iter,
                    target,
                } => (
                    "refine",
                    run_refine(llm, &input, validator, max_iter, target).await,
                ),
                PipelineStep::Extract { lang } => ("extract", run_extract(&input, &lang)),
                PipelineStep::BestOf { n, scorer } => {
                    ("best_of", run_best_of(llm, &input, n, scorer).await)
                }
                PipelineStep::Ensemble { n, aggregate } => {
                    ("ensemble", run_ensemble(llm, &input, n, aggregate).await)
                }
                PipelineStep::Reason => ("reason", run_reason(llm, &input).await),
                PipelineStep::Program { executor } => {
                    ("program", run_program(llm, &input, executor).await)
                }
                PipelineStep::Map { f } => {
                    let output = f(input.clone());
                    (
                        "map",
                        StepOutput {
                            output,
                            score: None,
                            tokens: 0,
                        },
                    )
                }
                PipelineStep::FanOut { branches, merge } => {
                    ("fan_out", run_fan_out(llm, &input, branches, &merge).await)
                }
                PipelineStep::Nested { steps } => ("nested", {
                    let br = run_branch_steps(llm, input.clone(), steps).await;
                    StepOutput {
                        output: br.output,
                        score: br.best_score,
                        tokens: br.tokens,
                    }
                }),
            };

            total_tokens += step_output.tokens;
            current_output = step_output.output.clone();

            let step_elapsed = step_start.elapsed();
            #[cfg(feature = "tracing")]
            tracing::debug!(
                step = name,
                step_index = i,
                score = ?step_output.score,
                tokens = step_output.tokens,
                elapsed_ms = step_elapsed.as_millis() as u64,
                "pipeline step complete"
            );

            step_results.push(StepResult {
                name: name.to_string(),
                input,
                output: step_output.output,
                score: step_output.score,
                tokens: step_output.tokens,
                elapsed: step_elapsed,
            });
        }

        PipelineResult {
            output: current_output,
            steps: step_results,
            total_tokens,
            elapsed: start.elapsed(),
        }
    }
}

/// Wraps a [`Pipeline`] as a [`DynStep`] so it can be nested in step composition.
///
/// Uses `DynStep` (boxed future) at the boundary since Pipeline internals
/// already use dynamic dispatch. Internally uses `Mutex<Option<Vec<...>>>` so
/// the steps can be consumed on first `run_dyn` call — subsequent calls
/// return the cached result.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, pipeline, checks};
/// use kkachi::recursive::step::DynStep;
///
/// let llm = MockLlm::new(|_, _| "nested".to_string());
/// let step = pipeline(&llm, "ignored").as_step();
/// // step implements DynStep and can be used in run_all, etc.
/// ```
pub struct PipelineAsStep<'a, L: Llm> {
    llm: &'a L,
    /// Steps wrapped in Mutex<Option<...>> so we can take() them for consumption.
    steps: std::sync::Mutex<Option<Vec<PipelineStep>>>,
}

/// Wrapper that asserts `Send` on a future.
///
/// # Safety
///
/// The caller must guarantee that the wrapped future is safe to send
/// across threads. This is used for `PipelineAsStep` because the pipeline's
/// internal `Refine` step never uses the `memory` field (it's always `None`),
/// but the compiler still sees `Memory`'s `!Sync` `duckdb::Connection` in the
/// type signature when the `storage` feature is enabled.
struct AssertSend<F>(F);

// SAFETY: Pipeline steps never hold &mut Memory across await points — the memory
// field is always None in pipeline context. The !Sync duckdb::Connection is never
// actually referenced.
unsafe impl<F: Future> Send for AssertSend<F> {}

impl<F: Future> Future for AssertSend<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We're not moving the inner future, just projecting through the newtype.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll(cx)
    }
}

impl<L: Llm + 'static> crate::recursive::step::DynStep for PipelineAsStep<'_, L> {
    fn run_dyn<'b>(
        &'b self,
        input: &'b str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = crate::error::Result<crate::recursive::step::StepOutput>,
                > + Send
                + 'b,
        >,
    > {
        // Take the steps out so we can consume them (owned).
        let steps = self.steps.lock().expect("pipeline steps lock poisoned").take().unwrap_or_default();
        let llm = self.llm;
        let input_owned = input.to_string();

        // SAFETY: Pipeline steps created via the Pipeline builder never hold &mut Memory.
        // The Refine step's memory field is always None in pipeline context.
        Box::pin(AssertSend(async move {
            let start = std::time::Instant::now();

            let result = run_branch_steps(llm, input_owned, steps).await;

            Ok(crate::recursive::step::StepOutput::new(
                result.output,
                result.best_score.unwrap_or(1.0),
                result.tokens,
            )
            .with_meta("elapsed_ms", start.elapsed().as_millis().to_string()))
        }))
    }

    fn dyn_name(&self) -> &'static str {
        "pipeline"
    }
}

impl<'a, L: Llm> Pipeline<'a, L> {
    /// Convert this pipeline into a [`DynStep`] for use in step composition.
    ///
    /// Consumes the pipeline and returns a `PipelineAsStep` that implements
    /// `DynStep`, allowing it to be used with `run_all`, `then`, etc.
    ///
    /// Note: The resulting step can only be run once. Subsequent calls
    /// will produce empty output.
    pub fn as_step(self) -> PipelineAsStep<'a, L> {
        PipelineAsStep {
            llm: self.llm,
            steps: std::sync::Mutex::new(Some(self.steps)),
        }
    }
}

async fn run_refine<L: Llm>(
    llm: &L,
    input: &str,
    validator: Box<dyn Validate>,
    max_iter: u32,
    target: f64,
) -> StepOutput {
    let result = refine::refine(llm, input)
        .validate(validator)
        .max_iter(max_iter)
        .target(target)
        .run()
        .await;
    match result {
        Ok(r) => StepOutput {
            output: r.output,
            score: Some(r.score),
            tokens: 0,
        },
        Err(_) => StepOutput {
            output: input.to_string(),
            score: Some(0.0),
            tokens: 0,
        },
    }
}

fn run_extract(input: &str, lang: &str) -> StepOutput {
    let extracted = extract_code(input, lang)
        .map(|s| s.to_string())
        .unwrap_or_else(|| input.to_string());
    StepOutput {
        output: extracted,
        score: None,
        tokens: 0,
    }
}

async fn run_best_of<L: Llm>(
    llm: &L,
    input: &str,
    n: usize,
    scorer: Option<Box<dyn Scorer>>,
) -> StepOutput {
    if let Some(s) = scorer {
        let result = best_of::best_of(llm, input)
            .n(n)
            .metric(move |text| s.score(text))
            .run()
            .await;
        StepOutput {
            output: result.output,
            score: Some(result.score),
            tokens: result.tokens,
        }
    } else {
        let result = best_of::best_of(llm, input).n(n).run().await;
        StepOutput {
            output: result.output,
            score: Some(result.score),
            tokens: result.tokens,
        }
    }
}

async fn run_ensemble<L: Llm>(llm: &L, input: &str, n: usize, aggregate: Aggregate) -> StepOutput {
    let result = ensemble::ensemble(llm, input)
        .n(n)
        .aggregate(aggregate)
        .run()
        .await;
    StepOutput {
        output: result.output,
        score: Some(result.agreement_ratio),
        tokens: result.tokens,
    }
}

async fn run_reason<L: Llm>(llm: &L, input: &str) -> StepOutput {
    let result = reason::reason(llm, input).run().await;
    StepOutput {
        output: result.output,
        score: Some(result.score),
        tokens: result.tokens,
    }
}

async fn run_program<L: Llm>(
    llm: &L,
    input: &str,
    executor: Box<dyn DynCodeExecutor>,
) -> StepOutput {
    let result = program::program(llm, input)
        .executor_dyn(executor)
        .run()
        .await;
    StepOutput {
        output: result.output,
        score: if result.success { Some(1.0) } else { Some(0.0) },
        tokens: result.tokens,
    }
}

async fn run_fan_out<L: Llm>(
    llm: &L,
    input: &str,
    branches: Vec<FanOutBranch>,
    merge: &MergeStrategy,
) -> StepOutput {
    let mut futs = FuturesUnordered::new();

    for (idx, branch) in branches.into_iter().enumerate() {
        let input_owned = input.to_string();
        let name = branch.name;
        let steps = branch.steps;
        futs.push(async move {
            let result = run_branch_steps(llm, input_owned, steps).await;
            (idx, name, result)
        });
    }

    let mut branch_results: Vec<FanOutBranchResult> = Vec::new();

    while let Some((idx, name, sub_output)) = futs.next().await {
        branch_results.push(FanOutBranchResult {
            index: idx,
            name,
            output: sub_output.output,
            score: sub_output.best_score,
            tokens: sub_output.tokens,
            elapsed: sub_output.elapsed,
        });
    }

    // Sort by index for deterministic merge ordering
    branch_results.sort_by_key(|b| b.index);

    let total_tokens: u32 = branch_results.iter().map(|b| b.tokens).sum();
    let best_score = branch_results
        .iter()
        .filter_map(|b| b.score)
        .fold(None, |acc: Option<f64>, s| {
            Some(acc.map_or(s, |a| a.max(s)))
        });
    let merged_output = apply_merge(&branch_results, merge);

    StepOutput {
        output: merged_output,
        score: best_score,
        tokens: total_tokens,
    }
}

struct BranchOutput {
    output: String,
    best_score: Option<f64>,
    tokens: u32,
    elapsed: Duration,
}

async fn run_branch_steps<L: Llm>(
    llm: &L,
    input: String,
    steps: Vec<PipelineStep>,
) -> BranchOutput {
    let start = Instant::now();
    let mut current_output = input.clone();
    let mut total_tokens: u32 = 0;
    let mut best_score: Option<f64> = None;

    if steps.is_empty() {
        // No steps — just do a single LLM generation
        if let Ok(output) = llm.generate(&input, "", None).await {
            total_tokens += output.total_tokens();
            current_output = output.text.to_string();
        }
    }

    for step in steps {
        let step_input = current_output.clone();

        let step_output = match step {
            PipelineStep::Refine {
                validator,
                max_iter,
                target,
            } => run_refine(llm, &step_input, validator, max_iter, target).await,
            PipelineStep::Extract { lang } => run_extract(&step_input, &lang),
            PipelineStep::BestOf { n, scorer } => run_best_of(llm, &step_input, n, scorer).await,
            PipelineStep::Ensemble { n, aggregate } => {
                run_ensemble(llm, &step_input, n, aggregate).await
            }
            PipelineStep::Reason => run_reason(llm, &step_input).await,
            PipelineStep::Program { executor } => run_program(llm, &step_input, executor).await,
            PipelineStep::Map { f } => StepOutput {
                output: f(step_input.clone()),
                score: None,
                tokens: 0,
            },
            PipelineStep::FanOut { branches, merge } => {
                run_fan_out(llm, &step_input, branches, &merge).await
            }
            PipelineStep::Nested { steps } => {
                // Recursive: run nested steps inline
                let br = Box::pin(run_branch_steps(llm, step_input.clone(), steps)).await;
                StepOutput {
                    output: br.output,
                    score: br.best_score,
                    tokens: br.tokens,
                }
            }
        };

        total_tokens += step_output.tokens;
        current_output = step_output.output;
        if let Some(s) = step_output.score {
            best_score = Some(best_score.map_or(s, |prev: f64| prev.max(s)));
        }
    }

    BranchOutput {
        output: current_output,
        best_score,
        tokens: total_tokens,
        elapsed: start.elapsed(),
    }
}

fn apply_merge(results: &[FanOutBranchResult], strategy: &MergeStrategy) -> String {
    match strategy {
        MergeStrategy::FirstSuccess => results
            .iter()
            .find(|r| !r.output.is_empty())
            .map(|r| r.output.clone())
            .unwrap_or_default(),
        MergeStrategy::BestScore => results
            .iter()
            .filter(|r| r.score.is_some())
            .max_by(|a, b| {
                a.score
                    .unwrap()
                    .partial_cmp(&b.score.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| results.first())
            .map(|r| r.output.clone())
            .unwrap_or_default(),
        MergeStrategy::Concat { separator } => results
            .iter()
            .map(|r| r.output.as_str())
            .collect::<Vec<_>>()
            .join(separator),
        MergeStrategy::Custom(f) => f(results),
    }
}

/// Internal step output before assembling into StepResult.
struct StepOutput {
    output: String,
    score: Option<f64>,
    tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::checks::checks;
    use crate::recursive::llm::MockLlm;

    #[test]
    fn test_pipeline_single_refine() {
        let llm = MockLlm::new(|_, feedback| {
            if feedback.is_some() {
                "fn add(a: i32, b: i32) -> i32 { a + b }".to_string()
            } else {
                "fn add(a, b) { a + b }".to_string()
            }
        });

        let result = pipeline(&llm, "Write an add function")
            .refine(checks().require("->"))
            .go();

        assert!(result.output.contains("->"));
        assert_eq!(result.steps.len(), 1);
        assert!(result.steps[0].score.unwrap() > 0.0);
    }

    #[test]
    fn test_pipeline_refine_then_extract() {
        let llm = MockLlm::new(|_, _| {
            "Here's the code:\n```rust\nfn main() { println!(\"hello\"); }\n```".to_string()
        });

        let result = pipeline(&llm, "Write a hello world")
            .refine(checks().require("```"))
            .extract("rust")
            .go();

        assert!(result.output.contains("fn main"));
        assert!(!result.output.contains("```"));
        assert_eq!(result.steps.len(), 2);
    }

    #[test]
    fn test_pipeline_map() {
        let llm = MockLlm::new(|_, _| "  hello world  ".to_string());

        // Map as second step: refine produces LLM output, map transforms it
        let result = pipeline(&llm, "greet")
            .refine(checks().min_len(1))
            .map(|s| s.trim().to_uppercase())
            .go();

        assert_eq!(result.output, "HELLO WORLD");
    }

    #[test]
    fn test_pipeline_best_of() {
        let llm = MockLlm::new(|_, _| "candidate output".to_string());

        let result = pipeline(&llm, "generate something").best_of(3).go();

        assert_eq!(result.output, "candidate output");
        assert_eq!(result.steps.len(), 1);
    }

    #[test]
    fn test_pipeline_ensemble() {
        let llm = MockLlm::new(|_, _| "Paris".to_string());

        let result = pipeline(&llm, "Capital of France?").ensemble(3).go();

        assert!(result.output.contains("Paris"));
    }

    #[test]
    fn test_pipeline_reason() {
        let llm = MockLlm::new(|_, _| {
            "Let me think step by step.\n1. 2^10 = 1024\n\nTherefore: 1024".to_string()
        });

        let result = pipeline(&llm, "What is 2^10?").reason().go();

        assert!(result.output.contains("1024"));
    }

    #[test]
    fn test_pipeline_empty() {
        let llm = MockLlm::new(|_, _| "direct output".to_string());

        let result = pipeline(&llm, "just generate").go();

        assert_eq!(result.output, "direct output");
        assert_eq!(result.steps.len(), 1);
    }

    #[test]
    fn test_pipeline_multi_step() {
        let call_count = std::sync::atomic::AtomicU32::new(0);
        let llm = MockLlm::new(|_prompt, _| {
            let n = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                "```python\ndef solve(): return 42\n```".to_string()
            } else {
                // Second call uses extracted code as prompt
                "42".to_string()
            }
        });

        let result = pipeline(&llm, "Write a solver")
            .refine(checks().require("```"))
            .extract("python")
            .go();

        assert!(result.output.contains("def solve"));
        assert!(!result.output.contains("```"));
        assert_eq!(result.steps.len(), 2);
    }

    #[test]
    fn test_pipeline_result_tokens() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let result = pipeline(&llm, "prompt")
            .refine(checks().require("output"))
            .go();

        // MockLlm returns 0 tokens, but structure should be correct
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_pipeline_elapsed() {
        let llm = MockLlm::new(|_, _| "fast".to_string());

        let result = pipeline(&llm, "prompt")
            .refine(checks().require("fast"))
            .go();

        assert!(result.elapsed.as_nanos() > 0);
    }

    #[test]
    fn test_pipeline_best_of_scored() {
        let llm = MockLlm::new(|_, _| "a]b long enough text here".to_string());

        let result = pipeline(&llm, "generate")
            .best_of_scored(3, |text: &str| text.len() as f64 / 100.0)
            .go();

        assert!(!result.output.is_empty());
        assert!(result.steps[0].score.unwrap() > 0.0);
    }

    #[test]
    fn test_pipeline_fan_out_first_success() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("A:") {
                "result_a".to_string()
            } else {
                "result_b".to_string()
            }
        });

        let result = pipeline(&llm, "base prompt")
            .fan_out(
                vec![
                    BranchBuilder::new("branch_a").map(|s| format!("A: {}", s)),
                    BranchBuilder::new("branch_b").map(|s| format!("B: {}", s)),
                ],
                MergeStrategy::FirstSuccess,
            )
            .go();

        // FirstSuccess returns the first non-empty result (by index order)
        assert_eq!(result.output, "A: base prompt");
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].name, "fan_out");
    }

    #[test]
    fn test_pipeline_fan_out_concat() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let result = pipeline(&llm, "input")
            .fan_out(
                vec![
                    BranchBuilder::new("x").map(|s| format!("X:{}", s)),
                    BranchBuilder::new("y").map(|s| format!("Y:{}", s)),
                ],
                MergeStrategy::Concat {
                    separator: "|".to_string(),
                },
            )
            .go();

        assert_eq!(result.output, "X:input|Y:input");
    }

    #[test]
    fn test_pipeline_fan_out_best_score() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("good") {
                "fn add(a: i32, b: i32) -> i32 { a + b }".to_string()
            } else {
                "bad output".to_string()
            }
        });

        let result = pipeline(&llm, "Write code")
            .fan_out(
                vec![
                    BranchBuilder::new("good_branch")
                        .map(|s| format!("good {}", s))
                        .refine(checks().require("fn ").require("->")),
                    BranchBuilder::new("bad_branch").refine(checks().require("impossible_string")),
                ],
                MergeStrategy::BestScore,
            )
            .go();

        assert!(result.output.contains("fn add"));
    }

    #[test]
    fn test_pipeline_fan_out_with_closure() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let result = pipeline(&llm, "input")
            .fan_out_with(
                MergeStrategy::Concat {
                    separator: " + ".to_string(),
                },
                |fan| {
                    fan.branch("first", |b| b.map(|s| format!("1:{}", s)))
                        .branch("second", |b| b.map(|s| format!("2:{}", s)))
                },
            )
            .go();

        assert_eq!(result.output, "1:input + 2:input");
    }

    #[test]
    fn test_pipeline_fan_out_custom_merge() {
        let llm = MockLlm::new(|_, _| "out".to_string());

        let result = pipeline(&llm, "in")
            .fan_out(
                vec![
                    BranchBuilder::new("a").map(|_| "alpha".to_string()),
                    BranchBuilder::new("b").map(|_| "beta".to_string()),
                ],
                MergeStrategy::Custom(Box::new(|results| {
                    results
                        .iter()
                        .map(|r| format!("[{}]", r.output))
                        .collect::<Vec<_>>()
                        .join(",")
                })),
            )
            .go();

        assert_eq!(result.output, "[alpha],[beta]");
    }

    #[test]
    fn test_pipeline_fan_out_with_refine_branches() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("rust") {
                "fn solve() -> i32 { 42 }".to_string()
            } else {
                "def solve(): return 42".to_string()
            }
        });

        let result = pipeline(&llm, "Write a solve function")
            .fan_out(
                vec![
                    BranchBuilder::new("rust")
                        .map(|s| format!("{} in rust", s))
                        .refine(checks().require("fn ")),
                    BranchBuilder::new("python")
                        .map(|s| format!("{} in python", s))
                        .refine(checks().require("def ")),
                ],
                MergeStrategy::Concat {
                    separator: "\n---\n".to_string(),
                },
            )
            .go();

        assert!(result.output.contains("fn solve"));
        assert!(result.output.contains("def solve"));
        assert!(result.output.contains("---"));
    }

    #[test]
    fn test_pipeline_fan_out_empty_branches() {
        let llm = MockLlm::new(|_, _| "generated".to_string());

        let result = pipeline(&llm, "input")
            .fan_out(vec![], MergeStrategy::FirstSuccess)
            .go();

        // Empty fan-out produces empty output
        assert_eq!(result.output, "");
    }

    #[test]
    fn test_pipeline_fan_out_single_branch() {
        let llm = MockLlm::new(|_, _| "single".to_string());

        let result = pipeline(&llm, "input")
            .fan_out(
                vec![BranchBuilder::new("only").map(|s| format!("mapped:{}", s))],
                MergeStrategy::FirstSuccess,
            )
            .go();

        assert_eq!(result.output, "mapped:input");
    }

    #[test]
    fn test_pipeline_nest() {
        let llm = MockLlm::new(|_, _| "inner output".to_string());

        let inner = pipeline(&llm, "ignored inner prompt");
        let result = pipeline(&llm, "outer prompt")
            .refine(checks().min_len(1))
            .nest(inner)
            .go();

        // The nested pipeline runs its steps (empty = just generate) on
        // the outer step's output.
        assert!(!result.output.is_empty());
        // 3 steps: refine + nested (which has an implicit generate)
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.steps[1].name, "nested");
    }

    #[test]
    fn test_pipeline_as_step() {
        use crate::recursive::step::DynStep;

        let llm = MockLlm::new(|_, _| "step output".to_string());

        let step = pipeline(&llm, "ignored")
            .map(|s| format!("mapped: {}", s))
            .as_step();

        let result = shared::block_on(step.run_dyn("my input"));
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.text, "mapped: my input");
    }
}
