// Copyright 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Zero-copy fluent pipeline builder with compile-time critic and strategy types.
//!
//! This module provides the core `pipeline()` entry point and `FluentPipeline` builder
//! for declarative prompt refinement with CLI validation.
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::{pipeline, Cli, CliPipeline};
//!
//! // Define your own validator
//! let rust_validator = CliPipeline::new()
//!     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
//!     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
//!     .file_ext("rs");
//!
//! // Use it in a pipeline
//! let result = pipeline("question -> code")
//!     .validate(rust_validator)
//!     .refine(5, 0.9)
//!     .run(input, &llm)
//!     .await?;
//! ```

use std::future::Future;
use std::marker::PhantomData;

use crate::error::Result;
use crate::recursive::{
    Critic, CriticResult, HeuristicCritic, RecursiveState, Validator, ValidatorCritic,
};
use crate::str_view::StrView;

// Re-export chunk types only when feature is enabled
#[cfg(any(feature = "tiktoken", feature = "huggingface", feature = "chunking"))]
use crate::recursive::{ChunkConfig, ChunkStrategy};

// Fallback chunk config for when chunking is not enabled
#[cfg(not(any(feature = "tiktoken", feature = "huggingface", feature = "chunking")))]
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub strategy: ChunkStrategy,
}

#[cfg(not(any(feature = "tiktoken", feature = "huggingface", feature = "chunking")))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChunkStrategy {
    #[default]
    Token,
    Section,
    Paragraph,
}

// =============================================================================
// Entry Points
// =============================================================================

/// Create a new fluent pipeline builder.
///
/// # Example
///
/// ```rust,ignore
/// use kkachi::declarative::{pipeline, Cli, CliPipeline};
///
/// // Define your validator
/// let validator = Cli::new("python").args(["-m", "py_compile"]).file_ext("py");
///
/// let result = pipeline("question -> code")
///     .validate(validator)
///     .refine(5, 0.9)
///     .run(input, &llm)
///     .await?;
/// ```
pub fn pipeline(signature: &str) -> FluentPipeline<'_, NoCritic, NoStrategy> {
    FluentPipeline::new(signature)
}

/// Short alias for `pipeline()`.
pub use pipeline as pipe;

// =============================================================================
// Strategy Markers (Zero-Size Types)
// =============================================================================

/// No DSPy strategy - direct generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoStrategy;

/// Chain of Thought strategy marker.
#[derive(Debug, Clone, Copy)]
pub struct WithCoT;

/// Best of N strategy marker.
#[derive(Debug, Clone, Copy)]
pub struct WithBestOfN<const N: u8>;

/// Multi-chain reasoning strategy marker.
#[derive(Debug, Clone, Copy)]
pub struct WithMultiChain<const N: u8>;

// =============================================================================
// Critic Types
// =============================================================================

/// Default no-op critic that always passes.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCritic;

impl Critic for NoCritic {
    fn evaluate<'a>(&self, _output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        CriticResult::new(1.0)
    }
}

// =============================================================================
// Check Types (Zero-Copy)
// =============================================================================

/// Stage correction mode for multi-stage CLI critics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StageCorrection {
    /// Run all stages, combine feedback.
    #[default]
    AllStages,
    /// Fix each stage before proceeding to next.
    PerStage,
}

/// A single check with function pointer (no dynamic dispatch).
#[derive(Clone)]
pub struct Check<'a> {
    /// Check name for identification.
    pub name: &'static str,
    /// Pattern to match (borrowed).
    pub pattern: StrView<'a>,
    /// Weight for scoring (0.0 - 1.0).
    pub weight: f64,
    /// Feedback message on failure.
    pub feedback: &'static str,
    /// Check type.
    pub kind: CheckKind,
}

/// Type of check to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckKind {
    /// Output must contain the pattern.
    MustContain,
    /// Output should contain the pattern (soft check).
    ShouldContain,
    /// Output must not contain the pattern.
    Forbid,
}

impl<'a> Check<'a> {
    /// Create a must-contain check.
    pub const fn must_contain(pattern: &'a str, feedback: &'static str) -> Self {
        Self {
            name: "must_contain",
            pattern: StrView::new(pattern),
            weight: 1.0,
            feedback,
            kind: CheckKind::MustContain,
        }
    }

    /// Create a should-contain check.
    pub const fn should_contain(pattern: &'a str, feedback: &'static str) -> Self {
        Self {
            name: "should_contain",
            pattern: StrView::new(pattern),
            weight: 0.5,
            feedback,
            kind: CheckKind::ShouldContain,
        }
    }

    /// Create a forbid check.
    pub const fn forbid(pattern: &'a str, feedback: &'static str) -> Self {
        Self {
            name: "forbid",
            pattern: StrView::new(pattern),
            weight: 1.0,
            feedback,
            kind: CheckKind::Forbid,
        }
    }

    /// Evaluate this check against output.
    pub fn evaluate(&self, output: &str) -> bool {
        let pattern = self.pattern.as_str();
        match self.kind {
            CheckKind::MustContain => output.contains(pattern),
            CheckKind::ShouldContain => output.contains(pattern),
            CheckKind::Forbid => !output.contains(pattern),
        }
    }
}

// =============================================================================
// Check Builder (Zero-Copy, Const-Capacity)
// =============================================================================

/// Builder for inline checks using const generics.
pub struct CheckBuilder<'a, const N: usize = 16> {
    checks: [Option<Check<'a>>; N],
    len: usize,
}

impl<'a, const N: usize> Default for CheckBuilder<'a, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, const N: usize> CheckBuilder<'a, N> {
    /// Create a new check builder with capacity N.
    pub const fn new() -> Self {
        Self {
            checks: [const { None }; N],
            len: 0,
        }
    }

    /// Add a must-contain check.
    pub fn must_contain(mut self, pattern: &'a str) -> Self {
        if self.len < N {
            self.checks[self.len] = Some(Check {
                name: "must_contain",
                pattern: StrView::new(pattern),
                weight: 1.0,
                feedback: "Missing required content",
                kind: CheckKind::MustContain,
            });
            self.len += 1;
        }
        self
    }

    /// Add a should-contain check.
    pub fn should_contain(mut self, pattern: &'a str) -> Self {
        if self.len < N {
            self.checks[self.len] = Some(Check {
                name: "should_contain",
                pattern: StrView::new(pattern),
                weight: 0.5,
                feedback: "Recommended content missing",
                kind: CheckKind::ShouldContain,
            });
            self.len += 1;
        }
        self
    }

    /// Add a forbid check.
    pub fn forbid(mut self, pattern: &'a str) -> Self {
        if self.len < N {
            self.checks[self.len] = Some(Check {
                name: "forbid",
                pattern: StrView::new(pattern),
                weight: 1.0,
                feedback: "Forbidden content found",
                kind: CheckKind::Forbid,
            });
            self.len += 1;
        }
        self
    }

    /// Build into a vec of checks.
    pub fn build(self) -> Vec<Check<'a>> {
        self.checks.into_iter().flatten().collect()
    }

    /// Get number of checks.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// =============================================================================
// FluentPipeline (Zero-Copy, Generic)
// =============================================================================

/// Zero-copy fluent pipeline builder with compile-time critic and strategy types.
///
/// Uses generics (not `Box<dyn>`) for zero-cost abstraction.
/// Uses lifetimes (not `Arc`) for zero-copy string handling.
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed data (signature, checks, examples)
/// - `C`: Critic type (compile-time, monomorphized)
/// - `S`: Strategy type (NoStrategy, WithCoT, WithBestOfN, etc.)
///
/// # Example
///
/// ```rust,ignore
/// use kkachi::declarative::{pipeline, Cli, CliPipeline};
///
/// // User defines their own validator
/// let my_validator = CliPipeline::new()
///     .stage("syntax", Cli::new("python").args(["-m", "py_compile"]).required())
///     .stage("lint", Cli::new("ruff").args(["check"]))
///     .file_ext("py");
///
/// let result = pipeline("question -> code")
///     .validate(my_validator)
///     .checks(|c| c.must_contain("def ").forbid("TODO"))
///     .refine(5, 0.9)
///     .run(input, &llm)
///     .await?;
/// ```
pub struct FluentPipeline<'a, C: Critic = NoCritic, S = NoStrategy> {
    /// Signature string (zero-copy borrowed).
    pub(crate) signature: StrView<'a>,
    /// Domain namespace.
    pub(crate) domain: Option<&'static str>,
    /// Maximum iterations for refinement.
    pub(crate) max_iterations: u32,
    /// Score threshold for convergence.
    pub(crate) score_threshold: f64,
    /// Inline checks (owned vec, but Check borrows patterns).
    pub(crate) checks: Vec<Check<'a>>,
    /// Critic instance (generic, not boxed).
    pub(crate) critic: C,
    /// Chunk configuration.
    pub(crate) chunk_config: Option<ChunkConfig>,
    /// Static examples for few-shot.
    pub(crate) examples: Vec<(&'a str, &'a str)>,
    /// Number of few-shot examples to use.
    pub(crate) few_shot_k: Option<usize>,
    /// Whether to show verbose output.
    pub(crate) verbose: bool,
    /// Stage correction mode.
    pub(crate) stage_correction: StageCorrection,
    /// Chain of thought field name.
    pub(crate) cot_field: Option<&'static str>,
    /// Strategy marker.
    pub(crate) _strategy: PhantomData<S>,
}

impl<'a> FluentPipeline<'a, NoCritic, NoStrategy> {
    /// Create a new pipeline with the given signature.
    pub fn new(signature: &'a str) -> Self {
        Self {
            signature: StrView::new(signature),
            domain: None,
            max_iterations: 5,
            score_threshold: 0.9,
            checks: Vec::new(),
            critic: NoCritic,
            chunk_config: None,
            examples: Vec::new(),
            few_shot_k: None,
            verbose: false,
            stage_correction: StageCorrection::AllStages,
            cot_field: None,
            _strategy: PhantomData,
        }
    }
}

// -----------------------------------------------------------------------------
// Domain Configuration (Generic - User Provides)
// -----------------------------------------------------------------------------

impl<'a, C: Critic, S> FluentPipeline<'a, C, S> {
    /// Set the domain namespace.
    ///
    /// This is a generic domain label, not tied to any specific language.
    /// Use it to organize your pipelines by task type.
    pub fn domain(mut self, domain: &'static str) -> Self {
        self.domain = Some(domain);
        self
    }
}

// -----------------------------------------------------------------------------
// Refinement Configuration
// -----------------------------------------------------------------------------

impl<'a, C: Critic, S> FluentPipeline<'a, C, S> {
    /// Set max iterations and score threshold.
    pub fn refine(mut self, max_iter: u32, threshold: f64) -> Self {
        self.max_iterations = max_iter;
        self.score_threshold = threshold;
        self
    }

    /// Quick refinement: 3 iterations, 0.8 threshold.
    pub fn quick(self) -> Self {
        self.refine(3, 0.8)
    }

    /// Standard refinement: 5 iterations, 0.9 threshold.
    pub fn standard(self) -> Self {
        self.refine(5, 0.9)
    }

    /// Strict refinement: 10 iterations, 0.95 threshold.
    pub fn strict(self) -> Self {
        self.refine(10, 0.95)
    }

    /// Perfect refinement: until 1.0 score.
    pub fn perfect(self) -> Self {
        self.refine(20, 1.0)
    }
}

// -----------------------------------------------------------------------------
// Inline Checks
// -----------------------------------------------------------------------------

impl<'a, C: Critic, S> FluentPipeline<'a, C, S> {
    /// Add checks via inline builder.
    pub fn checks<F>(mut self, f: F) -> Self
    where
        F: FnOnce(CheckBuilder<'a, 16>) -> CheckBuilder<'a, 16>,
    {
        let builder = f(CheckBuilder::new());
        self.checks.extend(builder.build());
        self
    }

    /// Add checks from a static slice (no allocation).
    pub fn checks_static(mut self, checks: &'a [Check<'a>]) -> Self {
        self.checks.extend(checks.iter().cloned());
        self
    }
}

// -----------------------------------------------------------------------------
// Chunking Configuration
// -----------------------------------------------------------------------------

impl<'a, C: Critic, S> FluentPipeline<'a, C, S> {
    /// Enable chunking with max tokens.
    pub fn chunk(mut self, max_tokens: usize) -> Self {
        self.chunk_config = Some(ChunkConfig {
            max_tokens,
            ..Default::default()
        });
        self
    }

    /// Set chunk overlap (call after .chunk()).
    pub fn overlap(mut self, tokens: usize) -> Self {
        if let Some(ref mut config) = self.chunk_config {
            config.overlap_tokens = tokens;
        }
        self
    }

    /// Set chunk strategy to sections.
    pub fn by_sections(mut self) -> Self {
        if let Some(ref mut config) = self.chunk_config {
            config.strategy = ChunkStrategy::Section;
        }
        self
    }

    /// Set chunk strategy to paragraphs.
    pub fn by_paragraphs(mut self) -> Self {
        if let Some(ref mut config) = self.chunk_config {
            config.strategy = ChunkStrategy::Paragraph;
        }
        self
    }
}

// -----------------------------------------------------------------------------
// Few-Shot / Examples
// -----------------------------------------------------------------------------

impl<'a, C: Critic, S> FluentPipeline<'a, C, S> {
    /// Add static examples for few-shot.
    pub fn examples(mut self, examples: &[(&'a str, &'a str)]) -> Self {
        self.examples.extend(examples.iter().copied());
        self
    }

    /// Set number of few-shot examples to use.
    pub fn few_shot(mut self, k: usize) -> Self {
        self.few_shot_k = Some(k);
        self
    }
}

// -----------------------------------------------------------------------------
// Verbose / Stage Correction
// -----------------------------------------------------------------------------

impl<'a, C: Critic, S> FluentPipeline<'a, C, S> {
    /// Enable verbose output.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Fix each stage before moving to next.
    pub fn fix_per_stage(mut self) -> Self {
        self.stage_correction = StageCorrection::PerStage;
        self
    }
}

// -----------------------------------------------------------------------------
// Generic CLI Validation (User Provides Validator)
// -----------------------------------------------------------------------------

impl<'a, S> FluentPipeline<'a, NoCritic, S> {
    /// Use any validator for CLI validation.
    ///
    /// This is the generic validation method. Users provide their own
    /// `Cli`, `CliPipeline`, or custom `Validator` implementation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kkachi::declarative::{pipeline, Cli, CliPipeline};
    ///
    /// // Single command
    /// let result = pipeline("q -> code")
    ///     .validate(Cli::new("rustfmt").args(["--check"]).file_ext("rs"))
    ///     .run(input, &llm).await?;
    ///
    /// // Multi-stage pipeline
    /// let validator = CliPipeline::new()
    ///     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
    ///     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
    ///     .file_ext("rs");
    ///
    /// let result = pipeline("q -> code")
    ///     .validate(validator)
    ///     .run(input, &llm).await?;
    /// ```
    pub fn validate<V: Validator + 'static>(
        self,
        validator: V,
    ) -> FluentPipeline<'a, ValidatorCritic<V>, S> {
        self.with_critic(ValidatorCritic::new(validator))
    }

    /// Use a heuristic critic with length bounds.
    ///
    /// This is useful for simple validation without external tools.
    pub fn validate_heuristic(
        self,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> FluentPipeline<'a, HeuristicCritic, S> {
        let mut critic = HeuristicCritic::new();
        if let Some(min) = min_length {
            critic = critic.min_length(min);
        }
        if let Some(max) = max_length {
            critic = critic.max_length(max);
        }
        self.with_critic(critic)
    }

    /// Transform to use a new critic type.
    fn with_critic<C2: Critic>(self, critic: C2) -> FluentPipeline<'a, C2, S> {
        FluentPipeline {
            signature: self.signature,
            domain: self.domain,
            max_iterations: self.max_iterations,
            score_threshold: self.score_threshold,
            checks: self.checks,
            critic,
            chunk_config: self.chunk_config,
            examples: self.examples,
            few_shot_k: self.few_shot_k,
            verbose: self.verbose,
            stage_correction: self.stage_correction,
            cot_field: self.cot_field,
            _strategy: PhantomData,
        }
    }
}

// -----------------------------------------------------------------------------
// DSPy Strategy Methods (Changes Strategy Type)
// -----------------------------------------------------------------------------

impl<'a, C: Critic> FluentPipeline<'a, C, NoStrategy> {
    /// Enable chain of thought reasoning.
    pub fn chain_of_thought(self) -> FluentPipeline<'a, C, WithCoT> {
        FluentPipeline {
            signature: self.signature,
            domain: self.domain,
            max_iterations: self.max_iterations,
            score_threshold: self.score_threshold,
            checks: self.checks,
            critic: self.critic,
            chunk_config: self.chunk_config,
            examples: self.examples,
            few_shot_k: self.few_shot_k,
            verbose: self.verbose,
            stage_correction: self.stage_correction,
            cot_field: Some("reasoning"),
            _strategy: PhantomData,
        }
    }

    /// Enable best-of-N sampling.
    pub fn best_of<const N: u8>(self) -> FluentPipeline<'a, C, WithBestOfN<N>> {
        FluentPipeline {
            signature: self.signature,
            domain: self.domain,
            max_iterations: self.max_iterations,
            score_threshold: self.score_threshold,
            checks: self.checks,
            critic: self.critic,
            chunk_config: self.chunk_config,
            examples: self.examples,
            few_shot_k: self.few_shot_k,
            verbose: self.verbose,
            stage_correction: self.stage_correction,
            cot_field: None,
            _strategy: PhantomData,
        }
    }

    /// Enable multi-chain reasoning.
    pub fn multi_chain<const N: u8>(self) -> FluentPipeline<'a, C, WithMultiChain<N>> {
        FluentPipeline {
            signature: self.signature,
            domain: self.domain,
            max_iterations: self.max_iterations,
            score_threshold: self.score_threshold,
            checks: self.checks,
            critic: self.critic,
            chunk_config: self.chunk_config,
            examples: self.examples,
            few_shot_k: self.few_shot_k,
            verbose: self.verbose,
            stage_correction: self.stage_correction,
            cot_field: None,
            _strategy: PhantomData,
        }
    }
}

// CoT-specific methods
impl<'a, C: Critic> FluentPipeline<'a, C, WithCoT> {
    /// Set the chain of thought field name.
    pub fn cot_field(mut self, field: &'static str) -> Self {
        self.cot_field = Some(field);
        self
    }
}

// =============================================================================
// Pipeline Output
// =============================================================================

/// Output from a pipeline execution (zero-copy where possible).
pub struct PipelineOutput<'a> {
    /// The final answer.
    pub answer: StrView<'a>,
    /// Final score.
    pub score: f64,
    /// Number of iterations.
    pub iterations: u32,
    /// Final feedback (if any).
    pub feedback: Option<StrView<'a>>,
}

impl<'a> PipelineOutput<'a> {
    /// Get answer as borrowed str.
    pub fn answer(&self) -> &str {
        self.answer.as_str()
    }

    /// Check if converged (score >= threshold).
    pub fn converged(&self, threshold: f64) -> bool {
        self.score >= threshold
    }

    /// Convert to owned output.
    pub fn into_owned(self) -> PipelineOutputOwned {
        PipelineOutputOwned {
            answer: self.answer.to_string(),
            score: self.score,
            iterations: self.iterations,
            feedback: self.feedback.map(|f| f.to_string()),
        }
    }
}

/// Owned variant for when output needs to outlive input.
#[derive(Debug, Clone)]
pub struct PipelineOutputOwned {
    /// The final answer.
    pub answer: String,
    /// Final score.
    pub score: f64,
    /// Number of iterations.
    pub iterations: u32,
    /// Final feedback (if any).
    pub feedback: Option<String>,
}

impl PipelineOutputOwned {
    /// Get answer as borrowed str.
    pub fn answer(&self) -> &str {
        &self.answer
    }

    /// Check if converged (score >= 0.9).
    pub fn converged(&self) -> bool {
        self.score >= 0.9
    }
}

// =============================================================================
// Execution (GAT-based where possible)
// =============================================================================

/// LLM trait for pipeline execution.
///
/// Uses GATs for zero-cost async when stable.
pub trait LLM {
    /// Generate a response.
    fn generate(&self, prompt: &str) -> impl Future<Output = Result<String>> + Send;

    /// Generate in a fresh context with no history (for one-shot testing).
    ///
    /// Default implementation just calls `generate()`, but implementations
    /// should reset/ignore conversation history for true fresh-context testing.
    fn generate_fresh(&self, prompt: &str) -> impl Future<Output = Result<String>> + Send {
        self.generate(prompt)
    }
}

impl<'a, C: Critic + Send + Sync, S: Send + Sync> FluentPipeline<'a, C, S> {
    /// Execute the pipeline.
    pub async fn run<L: LLM + Send + Sync>(
        self,
        input: StrView<'a>,
        llm: &L,
    ) -> Result<PipelineOutputOwned> {
        use crate::recursive::RecursiveState;

        let mut current = input.to_string();
        let mut iteration = 0u32;
        let mut last_feedback: Option<String> = None;

        loop {
            // Evaluate with critic - extract values to avoid borrow conflict
            let (score, feedback) = {
                let state = RecursiveState::new();
                let result = self.critic.evaluate(StrView::new(&current), &state);
                (result.score, result.feedback)
            };

            if self.verbose {
                let passed = score >= self.score_threshold;
                eprintln!("[iter {}] score={:.2}, passed={}", iteration, score, passed);
            }

            // Check convergence
            if score >= self.score_threshold || iteration >= self.max_iterations {
                return Ok(PipelineOutputOwned {
                    answer: current,
                    score,
                    iterations: iteration,
                    feedback: last_feedback,
                });
            }

            // Build prompt with feedback
            let prompt = self.build_prompt(&input, iteration, feedback.as_deref());

            // Generate
            current = llm.generate(&prompt).await?;
            last_feedback = feedback;
            iteration += 1;
        }
    }

    /// Build prompt for generation.
    fn build_prompt(&self, input: &StrView<'_>, iteration: u32, feedback: Option<&str>) -> String {
        let mut prompt = String::new();

        // Add examples if configured
        if let Some(k) = self.few_shot_k {
            let examples_to_use = self.examples.iter().take(k);
            for (i, (input_ex, output_ex)) in examples_to_use.enumerate() {
                prompt.push_str(&format!(
                    "Example {}:\nInput: {}\nOutput: {}\n\n",
                    i + 1,
                    input_ex,
                    output_ex
                ));
            }
        }

        // Add signature
        prompt.push_str(&format!("Task: {}\n\n", self.signature.as_str()));

        // Add input
        prompt.push_str(&format!("Input: {}\n\n", input.as_str()));

        // Add feedback if not first iteration
        if iteration > 0 {
            if let Some(fb) = feedback {
                prompt.push_str(&format!("Previous attempt feedback:\n{}\n\n", fb));
            }
            prompt.push_str("Please improve your response based on the feedback.\n\n");
        }

        prompt.push_str("Output:");
        prompt
    }
}

// =============================================================================
// RAG-Optimized Pipeline Integration
// =============================================================================

use super::doc_template::RagDocumentTemplate;
use super::oneshot::{ErrorCorrection, OneShotFailure, OneShotPrompt, OneShotTestResult};
use crate::recursive::{Embedder, MutableVectorStore, VectorStore};
use smallvec::SmallVec;

/// Configuration for RAG-optimized pipeline.
#[derive(Debug, Clone)]
pub struct RagPipelineConfig {
    /// Minimum similarity score to use RAG context.
    pub rag_threshold: f64,
    /// Maximum one-shot optimization attempts (retry loop).
    pub max_optimization_attempts: u32,
    /// Score threshold for one-shot test to pass.
    pub oneshot_threshold: f64,
    /// Whether to write back to RAG on success.
    pub write_back: bool,
}

impl Default for RagPipelineConfig {
    fn default() -> Self {
        Self {
            rag_threshold: 0.7,
            max_optimization_attempts: 3,
            oneshot_threshold: 0.9,
            write_back: true,
        }
    }
}

/// Output from RAG-optimized pipeline execution.
#[derive(Debug, Clone)]
pub struct RagPipelineOutput {
    /// The final answer.
    pub answer: String,
    /// Final validation score.
    pub score: f64,
    /// Total refinement iterations.
    pub iterations: u32,
    /// Number of one-shot optimization attempts.
    pub optimization_attempts: u32,
    /// Whether one-shot testing passed.
    pub oneshot_passed: bool,
    /// Whether RAG was updated.
    pub rag_updated: bool,
    /// Error corrections learned during refinement.
    pub error_corrections: Vec<(String, String, u32)>,
}

/// RAG-optimized pipeline that wraps FluentPipeline.
///
/// Combines DSPy3 strategies (CoT, BestOfN, MultiChain) with RAG optimization:
/// 1. RAG lookup for context
/// 2. FluentPipeline execution with DSPy3 strategy
/// 3. One-shot prompt formation
/// 4. Fresh context testing with retry loop
/// 5. RAG write-back on success
///
/// # Type Parameters
///
/// - `'a`: Lifetime for borrowed data
/// - `C`: Critic type from FluentPipeline
/// - `S`: Strategy type (NoStrategy, WithCoT, etc.)
/// - `St`: Vector store type
/// - `E`: Embedder type
pub struct RagOptimizedPipeline<'a, C: Critic, S, St, E>
where
    St: VectorStore + MutableVectorStore,
    E: Embedder,
{
    /// Inner FluentPipeline.
    pipeline: FluentPipeline<'a, C, S>,
    /// Vector store for RAG lookup and write-back.
    store: &'a mut St,
    /// Embedder for similarity search.
    embedder: &'a E,
    /// Document template for RAG write-back.
    template: Option<RagDocumentTemplate<'a>>,
    /// Configuration.
    config: RagPipelineConfig,
}

impl<'a, C: Critic + Clone + Send + Sync, S: Send + Sync, St, E>
    RagOptimizedPipeline<'a, C, S, St, E>
where
    St: VectorStore + MutableVectorStore,
    E: Embedder,
{
    /// Set the RAG similarity threshold.
    pub fn rag_threshold(mut self, threshold: f64) -> Self {
        self.config.rag_threshold = threshold;
        self
    }

    /// Set the one-shot test threshold.
    pub fn oneshot_threshold(mut self, threshold: f64) -> Self {
        self.config.oneshot_threshold = threshold;
        self
    }

    /// Set maximum optimization attempts.
    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.config.max_optimization_attempts = attempts;
        self
    }

    /// Enable or disable RAG write-back.
    pub fn write_back(mut self, enabled: bool) -> Self {
        self.config.write_back = enabled;
        self
    }

    /// Set document template for RAG write-back.
    pub fn template(mut self, template: RagDocumentTemplate<'a>) -> Self {
        self.template = Some(template);
        self
    }

    /// Execute the RAG-optimized pipeline.
    ///
    /// # Workflow
    ///
    /// 1. RAG lookup - search for similar examples
    /// 2. Pipeline execution - run with DSPy3 strategy and validation
    /// 3. One-shot prompt formation - create optimized prompt
    /// 4. Fresh context test - validate in new session
    /// 5. Retry loop - if failed, feed failure context back
    /// 6. RAG write-back - store optimized example on success
    pub async fn run<L: LLM + Send + Sync>(
        mut self,
        question: &'a str,
        llm: &'a L,
    ) -> Result<RagPipelineOutput> {
        // Phase 1: RAG lookup
        let rag_context = self.lookup_rag(question);

        // Accumulated failures for retry context
        let mut failures: Vec<OneShotFailure> = Vec::new();
        let mut total_iterations = 0u32;

        for attempt in 0..self.config.max_optimization_attempts {
            // Phase 2: Run FluentPipeline with RAG context and failure feedback
            let input = self.build_input(question, rag_context.as_deref(), &failures);
            let pipeline_output = self.run_pipeline_once(&input, llm).await?;

            total_iterations += pipeline_output.iterations;

            // Phase 3: Form one-shot prompt
            let corrections: SmallVec<[ErrorCorrection; 8]> = pipeline_output
                .feedback
                .iter()
                .enumerate()
                .map(|(i, fb)| ErrorCorrection::new(fb.clone(), String::new(), i as u32))
                .collect();

            let oneshot = OneShotPrompt {
                question: StrView::new(question),
                answer: pipeline_output.answer.clone(),
                error_corrections: corrections,
                score: pipeline_output.score,
                iterations: pipeline_output.iterations,
            };

            // Phase 4: Test in fresh context
            let test_result = self.test_oneshot(&oneshot, llm).await?;

            if test_result.passed {
                // Phase 5: RAG write-back
                let rag_updated = if self.config.write_back {
                    self.write_back_rag(question, &oneshot);
                    true
                } else {
                    false
                };

                return Ok(RagPipelineOutput {
                    answer: pipeline_output.answer,
                    score: pipeline_output.score,
                    iterations: total_iterations,
                    optimization_attempts: attempt + 1,
                    oneshot_passed: true,
                    rag_updated,
                    error_corrections: oneshot
                        .error_corrections
                        .iter()
                        .map(|ec| (ec.error.clone(), ec.fix.clone(), ec.iteration))
                        .collect(),
                });
            }

            // Capture failure for retry
            failures.push(OneShotFailure::new(
                attempt,
                oneshot.render(),
                test_result.output.clone(),
                test_result.errors.clone(),
            ));
        }

        // Max attempts exhausted
        Ok(RagPipelineOutput {
            answer: String::new(),
            score: 0.0,
            iterations: total_iterations,
            optimization_attempts: self.config.max_optimization_attempts,
            oneshot_passed: false,
            rag_updated: false,
            error_corrections: Vec::new(),
        })
    }

    /// Lookup RAG context for similar examples.
    fn lookup_rag(&self, question: &str) -> Option<String> {
        let embedding = self.embedder.embed(question);
        let results = self.store.search(&embedding, 1);

        if let Some(top) = results.first() {
            if top.score as f64 >= self.config.rag_threshold {
                return Some(top.content.clone());
            }
        }
        None
    }

    /// Build input with RAG context and failure feedback.
    fn build_input(
        &self,
        question: &str,
        rag_context: Option<&str>,
        failures: &[OneShotFailure],
    ) -> String {
        let mut input = String::with_capacity(2048);

        input.push_str(question);

        if let Some(context) = rag_context {
            input.push_str("\n\nRelevant example:\n");
            input.push_str(context);
        }

        if !failures.is_empty() {
            input.push_str("\n\nPrevious one-shot attempts failed:\n");
            for failure in failures {
                input.push_str(&failure.summary());
            }
        }

        input
    }

    /// Run the inner pipeline once.
    async fn run_pipeline_once<L: LLM + Send + Sync>(
        &self,
        input: &str,
        llm: &L,
    ) -> Result<PipelineOutputOwned> {
        // Create a new pipeline with same config
        let pipeline = FluentPipeline {
            signature: self.pipeline.signature,
            domain: self.pipeline.domain,
            max_iterations: self.pipeline.max_iterations,
            score_threshold: self.pipeline.score_threshold,
            checks: self.pipeline.checks.clone(),
            critic: self.pipeline.critic.clone(),
            chunk_config: self.pipeline.chunk_config.clone(),
            examples: self.pipeline.examples.clone(),
            few_shot_k: self.pipeline.few_shot_k,
            verbose: self.pipeline.verbose,
            stage_correction: self.pipeline.stage_correction,
            cot_field: self.pipeline.cot_field,
            _strategy: PhantomData::<S>,
        };

        pipeline.run(StrView::new(input), llm).await
    }

    /// Test one-shot prompt in fresh context.
    async fn test_oneshot<L: LLM + Send + Sync>(
        &self,
        oneshot: &OneShotPrompt<'_>,
        llm: &L,
    ) -> Result<OneShotTestResult> {
        // Generate in fresh context
        let prompt = oneshot.render();
        let output = llm.generate_fresh(&prompt).await?;

        // Evaluate with critic - extract values before using output
        let (score, errors) = {
            let state = crate::recursive::RecursiveState::new();
            let result = self.pipeline.critic.evaluate(StrView::new(&output), &state);
            let errors = result
                .feedback
                .map(|f| vec![f.to_string()])
                .unwrap_or_default();
            (result.score, errors)
        };

        if score >= self.config.oneshot_threshold {
            Ok(OneShotTestResult::pass(output, score))
        } else {
            Ok(OneShotTestResult::fail(output, score, errors))
        }
    }

    /// Write back to RAG store.
    fn write_back_rag(&mut self, question: &str, oneshot: &OneShotPrompt<'_>) {
        use super::doc_template::TemplateSection;
        use std::fmt::Write;

        let doc = if let Some(ref template) = self.template {
            let mut doc = String::with_capacity(4096);

            for section in &template.sections {
                match section {
                    TemplateSection::Header { label, level } => {
                        let prefix: String = "#".repeat(*level as usize);
                        writeln!(doc, "{} {}\n", prefix, label.as_str()).unwrap();
                    }
                    TemplateSection::Text { label } => {
                        let content = match label.as_str() {
                            "question" => question,
                            "explanation" => "Generated and validated by kkachi optimizer",
                            _ => "",
                        };
                        writeln!(doc, "{}\n", content).unwrap();
                    }
                    TemplateSection::Code { label, language } => {
                        let code = match label.as_str() {
                            "code" => &oneshot.answer,
                            _ => "",
                        };
                        writeln!(doc, "```{}\n{}\n```\n", language.as_str(), code).unwrap();
                    }
                    TemplateSection::List { label: _ } => {
                        doc.push('\n');
                    }
                }
            }

            writeln!(
                doc,
                "---\n_Score: {:.2} | Iterations: {} | One-shot validated: true_",
                oneshot.score, oneshot.iterations
            )
            .unwrap();

            doc
        } else {
            format!(
                "Question: {}\n\nAnswer:\n{}\n\nScore: {:.2}",
                question, oneshot.answer, oneshot.score
            )
        };

        // Generate ID from question hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        question.hash(&mut hasher);
        let id = format!("rag:{:x}", hasher.finish());

        self.store.add(id, doc);
    }
}

// -----------------------------------------------------------------------------
// FluentPipeline -> RagOptimizedPipeline Bridge
// -----------------------------------------------------------------------------

impl<'a, C: Critic + Clone + Send + Sync, S: Send + Sync> FluentPipeline<'a, C, S> {
    /// Enable RAG optimization for this pipeline.
    ///
    /// Creates a RAG-optimized pipeline that:
    /// 1. Looks up similar examples from the vector store
    /// 2. Runs the pipeline with DSPy3 strategy
    /// 3. Tests output in fresh context (one-shot)
    /// 4. Retries with failure context if one-shot fails
    /// 5. Writes successful examples back to RAG
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kkachi::declarative::{pipeline, Cli, RagDocumentTemplate};
    ///
    /// let template = RagDocumentTemplate::new("pulumi")
    ///     .header("Task", 2)
    ///     .text("question")
    ///     .code("code", "yaml");
    ///
    /// let result = pipeline("question -> pulumi_yaml")
    ///     .chain_of_thought()
    ///     .validate(Cli::new("pulumi").args(["preview"]).file_ext("yaml"))
    ///     .strict()
    ///     .optimize_with_rag(&mut store, &embedder)
    ///     .template(template)
    ///     .oneshot_threshold(0.9)
    ///     .max_attempts(3)
    ///     .run("How do I create a GCS bucket?", &llm)
    ///     .await?;
    /// ```
    pub fn optimize_with_rag<St, E>(
        self,
        store: &'a mut St,
        embedder: &'a E,
    ) -> RagOptimizedPipeline<'a, C, S, St, E>
    where
        St: VectorStore + MutableVectorStore,
        E: Embedder,
    {
        RagOptimizedPipeline {
            pipeline: self,
            store,
            embedder,
            template: None,
            config: RagPipelineConfig::default(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let p = pipeline("question -> answer");
        assert_eq!(p.signature.as_str(), "question -> answer");
        assert_eq!(p.max_iterations, 5);
        assert_eq!(p.score_threshold, 0.9);
    }

    #[test]
    fn test_pipeline_domain() {
        let p = pipeline("q -> code").domain("rust");
        assert_eq!(p.domain, Some("rust"));
    }

    #[test]
    fn test_pipeline_refine_shortcuts() {
        let p = pipeline("q -> a").quick();
        assert_eq!(p.max_iterations, 3);
        assert_eq!(p.score_threshold, 0.8);

        let p = pipeline("q -> a").strict();
        assert_eq!(p.max_iterations, 10);
        assert_eq!(p.score_threshold, 0.95);
    }

    #[test]
    fn test_check_builder() {
        let checks = CheckBuilder::<16>::new()
            .must_contain("fn ")
            .should_contain("///")
            .forbid(".unwrap()")
            .build();

        assert_eq!(checks.len(), 3);
        assert_eq!(checks[0].kind, CheckKind::MustContain);
        assert_eq!(checks[1].kind, CheckKind::ShouldContain);
        assert_eq!(checks[2].kind, CheckKind::Forbid);
    }

    #[test]
    fn test_check_evaluation() {
        let check = Check::must_contain("fn ", "Missing function");
        assert!(check.evaluate("fn main() {}"));
        assert!(!check.evaluate("def main():"));

        let forbid = Check::forbid(".unwrap()", "No unwrap");
        assert!(forbid.evaluate("let x = result?;"));
        assert!(!forbid.evaluate("let x = result.unwrap();"));
    }

    #[test]
    fn test_pipeline_type_state() {
        // These should compile - type changes with each method
        let _p1: FluentPipeline<'_, NoCritic, NoStrategy> = pipeline("q -> a");
        let _p2: FluentPipeline<'_, NoCritic, WithCoT> = pipeline("q -> a").chain_of_thought();
        let _p3: FluentPipeline<'_, NoCritic, WithBestOfN<3>> = pipeline("q -> a").best_of::<3>();
    }

    // =========================================================================
    // RAG-Optimized Pipeline Tests
    // =========================================================================

    #[test]
    fn test_rag_pipeline_config_default() {
        let config = RagPipelineConfig::default();
        assert_eq!(config.rag_threshold, 0.7);
        assert_eq!(config.max_optimization_attempts, 3);
        assert_eq!(config.oneshot_threshold, 0.9);
        assert!(config.write_back);
    }

    #[test]
    fn test_rag_optimized_pipeline_creation() {
        use crate::recursive::retrieve::{HashEmbedder, InMemoryVectorStore};

        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);
        let embedder2 = HashEmbedder::new(64);

        // Test that we can create a RAG-optimized pipeline
        let _rag_pipeline = pipeline("question -> answer")
            .refine(5, 0.9)
            .optimize_with_rag(&mut store, &embedder2)
            .rag_threshold(0.8)
            .oneshot_threshold(0.95)
            .max_attempts(5)
            .write_back(false);

        // Verify config was set
        assert_eq!(_rag_pipeline.config.rag_threshold, 0.8);
        assert_eq!(_rag_pipeline.config.oneshot_threshold, 0.95);
        assert_eq!(_rag_pipeline.config.max_optimization_attempts, 5);
        assert!(!_rag_pipeline.config.write_back);
    }

    #[test]
    fn test_rag_optimized_pipeline_with_cot() {
        use crate::recursive::retrieve::{HashEmbedder, InMemoryVectorStore};

        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);
        let embedder2 = HashEmbedder::new(64);

        // Test that CoT strategy works with RAG optimization
        let _rag_cot_pipeline = pipeline("question -> answer")
            .chain_of_thought()
            .refine(5, 0.9)
            .optimize_with_rag(&mut store, &embedder2);

        // Should compile - CoT + RAG integration
    }

    #[test]
    fn test_rag_optimized_pipeline_with_best_of_n() {
        use crate::recursive::retrieve::{HashEmbedder, InMemoryVectorStore};

        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);
        let embedder2 = HashEmbedder::new(64);

        // Test that BestOfN strategy works with RAG optimization
        let _rag_bon_pipeline = pipeline("question -> answer")
            .best_of::<5>()
            .refine(5, 0.9)
            .optimize_with_rag(&mut store, &embedder2);

        // Should compile - BestOfN + RAG integration
    }

    #[test]
    fn test_rag_optimized_pipeline_with_multi_chain() {
        use crate::recursive::retrieve::{HashEmbedder, InMemoryVectorStore};

        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);
        let embedder2 = HashEmbedder::new(64);

        // Test that MultiChain strategy works with RAG optimization
        let _rag_mc_pipeline = pipeline("question -> answer")
            .multi_chain::<3>()
            .refine(5, 0.9)
            .optimize_with_rag(&mut store, &embedder2);

        // Should compile - MultiChain + RAG integration
    }

    #[test]
    fn test_rag_optimized_pipeline_with_template() {
        use crate::declarative::RagDocumentTemplate;
        use crate::recursive::retrieve::{HashEmbedder, InMemoryVectorStore};

        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);
        let embedder2 = HashEmbedder::new(64);

        let template = RagDocumentTemplate::new("test")
            .header("Question", 2)
            .text("question")
            .header("Answer", 2)
            .code("code", "rust");

        let _rag_pipeline = pipeline("question -> code")
            .chain_of_thought()
            .refine(10, 0.95)
            .optimize_with_rag(&mut store, &embedder2)
            .template(template)
            .oneshot_threshold(0.9);

        // Should compile - template integration
    }
}
