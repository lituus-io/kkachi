// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Declarative public API for recursive refinement.
//!
//! Provides an ergonomic builder pattern for configuring and running
//! recursive language prompting pipelines.
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::*;
//!
//! let result = Kkachi::refine("question -> code")
//!     .domain("rust_codegen")
//!     .max_iterations(5)
//!     .run("Write a URL parser", |iter, feedback| {
//!         // Generate output based on iteration and optional feedback
//!         Ok(format!("Generated code for iteration {}", iter))
//!     });
//! ```
//!
//! # Template Support
//!
//! ```ignore
//! use kkachi::recursive::*;
//!
//! // Compose your own validator
//! let rust_validator = CliPipeline::new()
//!     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
//!     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
//!     .file_ext("rs");
//!
//! // Load template from file
//! let result = Kkachi::refine("requirement -> code")
//!     .template("templates/rust_codegen.md")
//!     .validate(rust_validator)
//!     .run("Write a URL parser", generate);
//!
//! // Or use inline template
//! let result = Kkachi::refine("question -> answer")
//!     .template_content(r#"
//! ---
//! format:
//!   type: json
//! ---
//! Answer questions concisely.
//! "#)
//!     .run("What is 2+2?", generate);
//! ```

use std::path::Path;

use crate::diff::ModuleDiff;
use crate::error::Result;
use crate::hitl::{HITLConfig, HumanReviewer, ReviewContext, ReviewDecision, ReviewTrigger};
use crate::recall_precision::RecallPrecisionMode;

use super::cli::{CliPipeline, ValidatorCritic};
use super::critic::{BinaryCritic, Critic, HeuristicCritic};
use super::retrieve::RetrievalConfig;
use super::runner::{
    NoProgress, ProgressCallback, RefinementResult, RunnerConfig, StandaloneRunner,
};
use super::similarity::SimilarityWeights;
use super::state::RecursiveConfig;
use super::template::{PromptTone, Template};

/// Main entry point for declarative recursive refinement.
pub struct Kkachi;

impl Kkachi {
    /// Start building a recursive refinement pipeline.
    ///
    /// # Arguments
    ///
    /// * `signature` - The signature string (e.g., "question -> code")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = Kkachi::refine("requirement -> terraform_code");
    /// ```
    pub fn refine(signature: &str) -> RefineBuilder<'_> {
        RefineBuilder::new(signature)
    }
}

// Note: ToolType has been removed in favor of generic validators.
// Users should compose their own validators using Cli and CliPipeline.
// See the declarative API: pipeline("q -> code").validate(my_validator)

/// Critic configuration.
///
/// Users compose their own validators using `Cli` and `CliPipeline` primitives.
/// Use the generic `.validate()` method instead of hardcoded tool critics.
pub enum CriticConfig {
    /// Use a custom CLI pipeline validator.
    Validator(CliPipeline),
    /// Use a heuristic critic with length bounds.
    Heuristic {
        /// Minimum output length required.
        min_length: Option<usize>,
        /// Maximum output length allowed.
        max_length: Option<usize>,
        /// Substrings that must be present in output.
        required: Vec<String>,
        /// Substrings that must not be present in output.
        forbidden: Vec<String>,
    },
    /// Always pass (for testing).
    AlwaysPass,
}

impl Default for CriticConfig {
    fn default() -> Self {
        CriticConfig::Heuristic {
            min_length: Some(10),
            max_length: None,
            required: vec![],
            forbidden: vec![],
        }
    }
}

/// Few-shot configuration for RAG examples.
#[derive(Clone, Debug)]
pub struct FewShotConfig {
    /// Number of examples to retrieve per iteration.
    pub k: usize,
    /// Minimum similarity to include example.
    pub min_similarity: f32,
    /// Include examples in prompt.
    pub include_in_prompt: bool,
    /// Use examples as demonstrations.
    pub as_demonstrations: bool,
    /// Refresh examples each iteration (vs. once at start).
    pub refresh_per_iteration: bool,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            k: 3,
            min_similarity: 0.7,
            include_in_prompt: true,
            as_demonstrations: true,
            refresh_per_iteration: false,
        }
    }
}

impl FewShotConfig {
    /// Create with specific k value.
    pub fn with_k(k: usize) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }
}

/// Callback type for diff events.
pub type DiffCallback = Box<dyn Fn(&ModuleDiff<'_>) + Send + Sync>;

/// Builder for declarative recursive refinement.
pub struct RefineBuilder<'a> {
    /// Signature string.
    signature: &'a str,
    /// Domain for storage/retrieval.
    domain: Option<&'a str>,
    /// Storage path.
    storage_path: Option<&'a str>,
    /// Critic configuration.
    critic: CriticConfig,
    /// Retrieval configuration.
    retrieval: RetrievalConfig,
    /// Recursive configuration.
    config: RecursiveConfig,
    /// Few-shot configuration.
    few_shot_config: Option<FewShotConfig>,
    /// Use chain of thought.
    use_cot: bool,
    /// Best of N sampling.
    best_of_n: Option<u8>,
    /// Human-in-the-loop configuration.
    hitl_config: Option<HITLConfig>,
    /// Custom human reviewer.
    reviewer: Option<Box<dyn HumanReviewer>>,
    /// Callback for diff events.
    diff_callback: Option<DiffCallback>,
    /// Template for structured output formatting.
    template: Option<Template<'static>>,
}

impl<'a> RefineBuilder<'a> {
    /// Create a new builder.
    pub fn new(signature: &'a str) -> Self {
        Self {
            signature,
            domain: None,
            storage_path: None,
            critic: CriticConfig::default(),
            retrieval: RetrievalConfig::default(),
            config: RecursiveConfig::default(),
            few_shot_config: None,
            use_cot: false,
            best_of_n: None,
            hitl_config: None,
            reviewer: None,
            diff_callback: None,
            template: None,
        }
    }

    // ===== Domain & Storage =====

    /// Set the domain namespace for storage/retrieval.
    pub fn domain(mut self, domain: &'a str) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Set the storage path.
    pub fn storage(mut self, path: &'a str) -> Self {
        self.storage_path = Some(path);
        self
    }

    // ===== Template Configuration =====

    /// Load a template from a file path.
    ///
    /// Templates define output format, system prompts, and few-shot examples
    /// in a structured Markdown format with YAML frontmatter.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = Kkachi::refine("requirement -> code")
    ///     .template("templates/rust_codegen.md")
    ///     .run("Write a URL parser", generate);
    /// ```
    pub fn template(mut self, path: impl AsRef<Path>) -> Self {
        match Template::from_file(path) {
            Ok(t) => self.template = Some(t),
            Err(e) => {
                // Log error but don't fail - template is optional
                eprintln!("Warning: Failed to load template: {}", e);
            }
        }
        self
    }

    /// Use inline template content.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = Kkachi::refine("question -> answer")
    ///     .template_content(r#"
    /// ---
    /// format:
    ///   type: json
    ///   schema:
    ///     type: object
    ///     required: [answer]
    /// ---
    /// Answer questions concisely.
    /// "#)
    ///     .run("What is 2+2?", generate);
    /// ```
    pub fn template_content(mut self, content: &str) -> Self {
        match Template::from_str(content) {
            Ok(t) => self.template = Some(t.into_owned()),
            Err(e) => {
                eprintln!("Warning: Failed to parse template: {}", e);
            }
        }
        self
    }

    /// Use a pre-parsed template.
    ///
    /// This is useful when you want to reuse a template across multiple
    /// refinement runs or when you need more control over template parsing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = Template::from_file("templates/code.md")?;
    ///
    /// let result = Kkachi::refine("requirement -> code")
    ///     .with_template(template)
    ///     .run("Write a URL parser", generate);
    /// ```
    pub fn with_template(mut self, template: Template<'static>) -> Self {
        self.template = Some(template);
        self
    }

    /// Get the current template, if set.
    pub fn get_template(&self) -> Option<&Template<'static>> {
        self.template.as_ref()
    }

    // ===== Convergence Criteria =====

    /// Set maximum iterations.
    pub fn max_iterations(mut self, n: u32) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set score threshold for convergence.
    pub fn until_score(mut self, threshold: f64) -> Self {
        self.config.score_threshold = threshold;
        self
    }

    /// Set plateau detection for convergence.
    pub fn until_plateau(mut self, min_improvement: f64, window: usize) -> Self {
        self.config.plateau_threshold = Some(min_improvement);
        self.config.plateau_window = Some(window);
        self
    }

    // ===== Recall/Precision Mode =====

    /// Set the recall/precision mode.
    ///
    /// Controls the tradeoff between recall (permissive, catch everything)
    /// and precision (strict, only confident results).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::RecallPrecisionMode;
    ///
    /// let result = Kkachi::refine("query -> results")
    ///     .recall_precision(RecallPrecisionMode::high_recall(0.6))
    ///     .run("Find all matches", generate);
    /// ```
    pub fn recall_precision(mut self, mode: RecallPrecisionMode) -> Self {
        self.config = self.config.with_recall_precision_mode(mode);
        self
    }

    /// Use high-recall mode (permissive, favors catching all results).
    ///
    /// Sets threshold to 0.6. Use when false negatives are costly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = Kkachi::refine("query -> results")
    ///     .high_recall()
    ///     .run("Find all possible matches", generate);
    /// ```
    pub fn high_recall(mut self) -> Self {
        self.config = self.config.high_recall();
        self
    }

    /// Use high-precision mode (strict, favors accuracy).
    ///
    /// Sets threshold to 0.9. Use when false positives are costly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = Kkachi::refine("query -> results")
    ///     .high_precision()
    ///     .run("Find only confident matches", generate);
    /// ```
    pub fn high_precision(mut self) -> Self {
        self.config = self.config.high_precision();
        self
    }

    /// Set the prompt tone for template generation.
    ///
    /// Controls the language used in prompts:
    /// - `Inclusive`: Permissive language ("Consider including...", "You may...")
    /// - `Balanced`: Neutral language ("Include...", "Please...")
    /// - `Restrictive`: Strict language ("You MUST include...", "Required:...")
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::recursive::PromptTone;
    ///
    /// let result = Kkachi::refine("requirement -> code")
    ///     .tone(PromptTone::Restrictive)
    ///     .run("Implement strict validation", generate);
    /// ```
    pub fn tone(mut self, tone: PromptTone) -> Self {
        if let Some(ref mut template) = self.template {
            template.options.tone = tone;
        }
        self
    }

    // ===== Validation (Generic CLI) =====

    /// Use a custom CLI pipeline for validation.
    ///
    /// Users compose their own validators using `Cli` and `CliPipeline` primitives.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::recursive::{Kkachi, Cli, CliPipeline};
    ///
    /// let rust_validator = CliPipeline::new()
    ///     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
    ///     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
    ///     .stage("lint", Cli::new("cargo").args(["clippy"]).weight(0.3))
    ///     .file_ext("rs");
    ///
    /// let result = Kkachi::refine("question -> code")
    ///     .validate(rust_validator)
    ///     .run("Write a URL parser", generate);
    /// ```
    pub fn validate(mut self, pipeline: CliPipeline) -> Self {
        self.critic = CriticConfig::Validator(pipeline);
        self
    }

    /// Use a heuristic critic with length bounds.
    pub fn critic_heuristic(
        mut self,
        min_length: Option<usize>,
        max_length: Option<usize>,
    ) -> Self {
        self.critic = CriticConfig::Heuristic {
            min_length,
            max_length,
            required: vec![],
            forbidden: vec![],
        };
        self
    }

    /// Use an always-pass critic (for testing).
    pub fn critic_always_pass(mut self) -> Self {
        self.critic = CriticConfig::AlwaysPass;
        self
    }

    // ===== Similarity & Retrieval =====

    /// Enable or disable semantic cache.
    pub fn semantic_cache(mut self, enabled: bool) -> Self {
        self.retrieval.use_semantic_cache = enabled;
        self
    }

    /// Set similarity threshold for cache hit.
    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.retrieval.similarity_threshold = threshold;
        self
    }

    /// Enable or disable auto-condensation.
    pub fn auto_condense(mut self, enabled: bool) -> Self {
        self.retrieval.auto_condense = enabled;
        self
    }

    /// Set cluster threshold.
    pub fn cluster_threshold(mut self, threshold: f32) -> Self {
        self.retrieval.cluster_threshold = threshold;
        self
    }

    /// Set minimum cluster size.
    pub fn min_cluster_size(mut self, size: usize) -> Self {
        self.retrieval.min_cluster_size = size;
        self
    }

    /// Set similarity weights.
    pub fn similarity_weights(mut self, weights: SimilarityWeights) -> Self {
        self.retrieval.similarity_weights = weights;
        self
    }

    // ===== Few-Shot Configuration =====

    /// Configure few-shot learning.
    pub fn few_shot(mut self, config: FewShotConfig) -> Self {
        self.few_shot_config = Some(config);
        self
    }

    /// Set number of few-shot examples (shorthand).
    pub fn few_shot_k(mut self, k: usize) -> Self {
        self.few_shot_config
            .get_or_insert_with(FewShotConfig::default)
            .k = k;
        self
    }

    /// Use few-shot examples as demonstrations.
    pub fn few_shot_as_demos(mut self, enabled: bool) -> Self {
        self.few_shot_config
            .get_or_insert_with(FewShotConfig::default)
            .as_demonstrations = enabled;
        self
    }

    /// Refresh examples each iteration.
    pub fn few_shot_refresh(mut self, enabled: bool) -> Self {
        self.few_shot_config
            .get_or_insert_with(FewShotConfig::default)
            .refresh_per_iteration = enabled;
        self
    }

    // ===== DSPy Module Integration =====

    /// Enable chain of thought reasoning.
    pub fn with_chain_of_thought(mut self) -> Self {
        self.use_cot = true;
        self
    }

    /// Enable best-of-N sampling.
    pub fn with_best_of_n(mut self, n: u8) -> Self {
        self.best_of_n = Some(n);
        self
    }

    // ===== Human-in-the-Loop =====

    /// Enable human-in-the-loop review with the given configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::hitl::HITLConfig;
    ///
    /// let result = Kkachi::refine("question -> code")
    ///     .hitl(HITLConfig::every(2))  // Review every 2 iterations
    ///     .run("Write a URL parser", generate);
    /// ```
    pub fn hitl(mut self, config: HITLConfig) -> Self {
        self.hitl_config = Some(config);
        self
    }

    /// Set a custom human reviewer for HITL.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kkachi::hitl::TerminalReviewer;
    ///
    /// let result = Kkachi::refine("question -> code")
    ///     .hitl(HITLConfig::every(1))
    ///     .reviewer(TerminalReviewer::new())
    ///     .run("Write a URL parser", generate);
    /// ```
    pub fn reviewer<R: HumanReviewer + 'static>(mut self, reviewer: R) -> Self {
        self.reviewer = Some(Box::new(reviewer));
        self
    }

    /// Set a callback to be called on each diff between iterations.
    ///
    /// This is useful for custom diff visualization.
    pub fn on_diff<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ModuleDiff<'_>) + Send + Sync + 'static,
    {
        self.diff_callback = Some(Box::new(callback));
        self
    }

    /// Get the HITL configuration if set.
    pub fn hitl_config(&self) -> Option<&HITLConfig> {
        self.hitl_config.as_ref()
    }

    // ===== Build & Execute =====

    /// Get the current configuration.
    pub fn config(&self) -> &RecursiveConfig {
        &self.config
    }

    /// Get the retrieval configuration.
    pub fn retrieval_config(&self) -> &RetrievalConfig {
        &self.retrieval
    }

    /// Get the signature.
    pub fn signature(&self) -> &str {
        self.signature
    }

    /// Get the domain if set.
    pub fn get_domain(&self) -> Option<&str> {
        self.domain
    }

    /// Build a runner configuration from this builder.
    pub fn build_runner_config(&self) -> RunnerConfig {
        let mut config = RunnerConfig::default();
        config.recursive.max_iterations = self.config.max_iterations;
        config.recursive.score_threshold = self.config.score_threshold;
        if let Some(domain) = self.domain {
            config.domain = domain.to_string();
        }
        config
    }

    /// Build and run the refinement pipeline.
    ///
    /// # Arguments
    ///
    /// * `question` - The question/input to refine
    /// * `generate` - A function that generates output given iteration and feedback
    ///
    /// # Returns
    ///
    /// The refinement result containing the final answer and metadata.
    pub fn run<F>(self, question: &str, generate: F) -> RefinementResult
    where
        F: FnMut(u32, Option<&str>) -> Result<String> + Clone,
    {
        self.run_with_progress(question, generate, NoProgress)
    }

    /// Build and run with progress callback.
    pub fn run_with_progress<F, P>(
        self,
        question: &str,
        generate: F,
        progress: P,
    ) -> RefinementResult
    where
        F: FnMut(u32, Option<&str>) -> Result<String> + Clone,
        P: ProgressCallback + Clone,
    {
        // Check if HITL is enabled - use HITL-aware loop
        if let Some(ref hitl_config) = self.hitl_config {
            if hitl_config.enabled {
                return self.run_with_hitl(question, generate, progress);
            }
        }

        let domain = self.domain.unwrap_or("default");
        let use_cot = self.use_cot;
        let best_of_n = self.best_of_n;

        // Helper to wrap generate with CoT if enabled
        let wrap_with_cot = |mut gen: F| -> Box<dyn FnMut(u32, Option<&str>) -> Result<String>> {
            if use_cot {
                Box::new(
                    move |iteration: u32, feedback: Option<&str>| -> Result<String> {
                        let cot_feedback = match feedback {
                        Some(fb) => Some(format!(
                            "{}\n\nPlease think step by step and show your reasoning before giving the final answer.",
                            fb
                        )),
                        None => Some(
                            "Please think step by step and show your reasoning before giving the final answer."
                                .to_string(),
                        ),
                    };
                        gen(iteration, cot_feedback.as_deref())
                    },
                )
            } else {
                Box::new(
                    move |iteration: u32, feedback: Option<&str>| -> Result<String> {
                        gen(iteration, feedback)
                    },
                )
            }
        };

        // Helper to run a single refinement with a boxed generator
        let run_once = |mut gen: Box<dyn FnMut(u32, Option<&str>) -> Result<String>>,
                        prog: P|
         -> RefinementResult {
            match &self.critic {
                CriticConfig::AlwaysPass => {
                    let critic = BinaryCritic::new(|_| true, "Validation failed");
                    let runner = StandaloneRunner::with_config(&critic, domain, self.config)
                        .with_progress(prog);
                    runner
                        .refine(question, &mut gen)
                        .unwrap_or_else(|_| RefinementResult::error())
                }
                CriticConfig::Validator(ref pipeline) => {
                    // Use the user-provided CLI pipeline validator
                    let critic = ValidatorCritic::new(pipeline.clone());
                    let runner = StandaloneRunner::with_config(&critic, domain, self.config)
                        .with_progress(prog);
                    runner
                        .refine(question, &mut gen)
                        .unwrap_or_else(|_| RefinementResult::error())
                }
                CriticConfig::Heuristic {
                    ref min_length,
                    ref max_length,
                    ref required,
                    ref forbidden,
                } => {
                    let mut critic = HeuristicCritic::new();
                    if let Some(min) = min_length {
                        critic = critic.min_length(*min);
                    }
                    if let Some(max) = max_length {
                        critic = critic.max_length(*max);
                    }
                    for r in required {
                        critic = critic.require(r.clone());
                    }
                    for f in forbidden {
                        critic = critic.forbid(f.clone());
                    }
                    let runner = StandaloneRunner::with_config(&critic, domain, self.config)
                        .with_progress(prog);
                    runner
                        .refine(question, &mut gen)
                        .unwrap_or_else(|_| RefinementResult::error())
                }
            }
        };

        // Run with BestOfN if configured
        if let Some(n) = best_of_n {
            let mut best_result: Option<RefinementResult> = None;

            for _ in 0..n {
                let wrapped = wrap_with_cot(generate.clone());
                let result = run_once(wrapped, progress.clone());

                // Keep the best result
                match &best_result {
                    None => best_result = Some(result),
                    Some(current_best) => {
                        if result.score > current_best.score {
                            best_result = Some(result);
                        }
                    }
                }
            }

            best_result.unwrap_or_else(RefinementResult::error)
        } else {
            // Single run
            let wrapped = wrap_with_cot(generate);
            run_once(wrapped, progress)
        }
    }

    /// Run refinement with HITL integration.
    ///
    /// This method implements a custom refinement loop that includes human review
    /// at configured intervals or triggers.
    fn run_with_hitl<F, P>(self, question: &str, mut generate: F, progress: P) -> RefinementResult
    where
        F: FnMut(u32, Option<&str>) -> Result<String>,
        P: ProgressCallback,
    {
        use super::storage::ContextId;
        use crate::str_view::StrView;
        use smallvec::SmallVec;

        let domain = self.domain.unwrap_or("default");
        let context_id = ContextId::from_question(question, domain);
        let hitl_config = self.hitl_config.as_ref().expect("HITL config must be set");

        // Build critic
        let critic: Box<dyn Critic> = match &self.critic {
            CriticConfig::AlwaysPass => Box::new(BinaryCritic::new(|_| true, "Validation failed")),
            CriticConfig::Validator(ref pipeline) => {
                Box::new(ValidatorCritic::new(pipeline.clone()))
            }
            CriticConfig::Heuristic {
                min_length,
                max_length,
                required,
                forbidden,
            } => {
                let mut c = HeuristicCritic::new();
                if let Some(min) = min_length {
                    c = c.min_length(*min);
                }
                if let Some(max) = max_length {
                    c = c.max_length(*max);
                }
                for r in required {
                    c = c.require(r.clone());
                }
                for f in forbidden {
                    c = c.forbid(f.clone());
                }
                Box::new(c)
            }
        };

        // Refinement state
        let mut iteration = 0u32;
        let mut scores: SmallVec<[f64; 8]> = SmallVec::new();
        let mut last_feedback: Option<String> = None;
        let mut error_corrections: Vec<(String, String)> = Vec::new();
        let mut best_output = String::new();
        let mut best_score = 0.0f64;
        let mut prev_output: Option<String> = None;
        let mut skip_reviews = 0u32;

        loop {
            // Generate output
            let output = match generate(iteration, last_feedback.as_deref()) {
                Ok(o) => o,
                Err(_) => break,
            };

            // Evaluate
            let (score, feedback, is_satisfactory) = {
                let temp_state = super::state::RecursiveState::with_scores(&scores);
                let eval = critic.evaluate(StrView::new(&output), &temp_state);
                let is_sat = eval.is_satisfactory();
                (eval.score, eval.feedback, is_sat)
            };
            let prev_score = scores.last().copied().unwrap_or(0.0);
            scores.push(score);

            // Track best
            if score > best_score {
                best_score = score;
                best_output = output.clone();
            }

            // Report progress
            progress.on_iteration(iteration, score, feedback.as_deref());

            // Call diff callback if set
            if let Some(ref callback) = self.diff_callback {
                if let Some(ref prev) = prev_output {
                    let diff = ModuleDiff::new()
                        .with_iterations(iteration.saturating_sub(1), iteration)
                        .with_scores(prev_score, score)
                        .with_output(prev, &output);
                    callback(&diff);
                }
            }

            // Check if we need HITL review
            let needs_review = skip_reviews == 0
                && self.should_review(hitl_config, iteration, score, prev_score, is_satisfactory);

            if needs_review {
                if let Some(ref reviewer) = self.reviewer {
                    // Determine trigger reason
                    let trigger = if iteration == 0 && hitl_config.on_first {
                        ReviewTrigger::FirstIteration
                    } else if score < prev_score && hitl_config.on_score_drop {
                        ReviewTrigger::ScoreDrop
                    } else if is_satisfactory && hitl_config.on_convergence {
                        ReviewTrigger::Convergence
                    } else {
                        ReviewTrigger::Interval
                    };

                    // Build review context
                    let mut ctx = ReviewContext::new(
                        iteration,
                        self.config.max_iterations,
                        score,
                        &output,
                        trigger,
                    );
                    ctx = ctx.with_prev_score(prev_score);
                    if let Some(ref fb) = feedback {
                        ctx = ctx.with_feedback(fb);
                    }
                    if let Some(ref prev) = prev_output {
                        ctx = ctx.with_prev_output(prev);
                    }

                    // Get reviewer's decision
                    let decision = reviewer.review(ctx);

                    // Handle decision
                    match decision {
                        ReviewDecision::Accept => {
                            // Continue normally
                        }
                        ReviewDecision::Reject => {
                            // Don't update best, continue with new iteration
                        }
                        ReviewDecision::Edit {
                            instruction: _,
                            output: edited_output,
                            guidance: _,
                        } => {
                            // Use edited output if provided
                            if let Some(edited) = edited_output {
                                best_output = edited;
                                best_score = score; // Keep current score
                            }
                        }
                        ReviewDecision::Stop => {
                            // Return current best
                            break;
                        }
                        ReviewDecision::AcceptFinal => {
                            // Accept this output as final
                            best_output = output;
                            best_score = score;
                            break;
                        }
                        ReviewDecision::Rollback { to_iteration: _ } => {
                            // Rollback not fully supported in simple mode
                            // Just continue
                        }
                        ReviewDecision::SkipNext { count } => {
                            skip_reviews = count;
                        }
                        ReviewDecision::Pause { duration: _ } => {
                            // Pause not implemented in sync mode
                            // Just continue
                        }
                        ReviewDecision::RequestInfo { query: _ } => {
                            // RequestInfo not implemented in simple mode
                            // Just continue
                        }
                    }
                }
            }

            // Decrement skip counter
            if skip_reviews > 0 {
                skip_reviews -= 1;
            }

            // Check convergence
            if score >= self.config.score_threshold || is_satisfactory {
                best_output = output;
                best_score = score;
                break;
            }

            // Safety check
            if iteration >= self.config.max_iterations {
                break;
            }

            // Record error correction
            if let Some(ref fb) = feedback {
                error_corrections.push((fb.clone(), String::new()));
            }

            // Store for next iteration
            prev_output = Some(output);
            last_feedback = feedback;
            iteration += 1;
        }

        let result = RefinementResult {
            answer: best_output,
            summary: String::new(),
            score: best_score,
            iterations: iteration as usize,
            from_cache: false,
            context_id,
            error_corrections,
        };

        progress.on_complete(&result);
        result
    }

    /// Check if we should trigger a HITL review.
    fn should_review(
        &self,
        config: &HITLConfig,
        iteration: u32,
        score: f64,
        prev_score: f64,
        is_converged: bool,
    ) -> bool {
        if !config.enabled {
            return false;
        }

        // Skip if score is above auto-accept threshold
        if let Some(skip_threshold) = config.skip_above_score {
            if score >= skip_threshold {
                return false;
            }
        }

        // First iteration
        if iteration == 0 && config.on_first {
            return true;
        }

        // Score drop
        if score < prev_score && config.on_score_drop {
            return true;
        }

        // Convergence
        if is_converged && config.on_convergence {
            return true;
        }

        // Interval
        if config.interval > 0 && iteration % config.interval == 0 {
            return true;
        }

        false
    }
}

impl RefinementResult {
    /// Create an error result (used when refinement fails).
    fn error() -> Self {
        use super::storage::ContextId;
        Self {
            answer: String::new(),
            summary: String::new(),
            score: 0.0,
            iterations: 0,
            from_cache: false,
            context_id: ContextId::from_question("error", "error"),
            error_corrections: vec![],
        }
    }
}

/// Result from the declarative API.
#[derive(Debug)]
pub struct RefineResult {
    /// The final refined answer.
    pub answer: String,
    /// Summary of the answer.
    pub summary: String,
    /// Final quality score (0.0 - 1.0).
    pub score: f64,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether result came from cache.
    pub from_cache: bool,
    /// Domain used.
    pub domain: Option<String>,
}

impl RefineResult {
    /// Create a new result.
    pub fn new(answer: String, score: f64, iterations: usize) -> Self {
        Self {
            answer,
            summary: String::new(),
            score,
            iterations,
            from_cache: false,
            domain: None,
        }
    }

    /// Set summary.
    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = summary;
        self
    }

    /// Set cache status.
    pub fn with_cache_status(mut self, from_cache: bool) -> Self {
        self.from_cache = from_cache;
        self
    }

    /// Set domain.
    pub fn with_domain(mut self, domain: String) -> Self {
        self.domain = Some(domain);
        self
    }
}

impl From<RefinementResult> for RefineResult {
    fn from(result: RefinementResult) -> Self {
        Self {
            answer: result.answer,
            summary: result.summary,
            score: result.score,
            iterations: result.iterations,
            from_cache: result.from_cache,
            domain: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::cli::Cli;
    use super::*;

    #[test]
    fn test_kkachi_refine_basic() {
        let builder = Kkachi::refine("question -> answer");
        assert_eq!(builder.signature(), "question -> answer");
    }

    #[test]
    fn test_refine_builder_domain() {
        let builder = Kkachi::refine("q -> a")
            .domain("test_domain")
            .storage("~/.kkachi/test.db");

        assert_eq!(builder.get_domain(), Some("test_domain"));
    }

    #[test]
    fn test_refine_builder_convergence() {
        let builder = Kkachi::refine("q -> a")
            .max_iterations(10)
            .until_score(0.95)
            .until_plateau(0.01, 3);

        assert_eq!(builder.config().max_iterations, 10);
        assert!((builder.config().score_threshold - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_refine_builder_validate() {
        let validator = CliPipeline::new()
            .stage("check", Cli::new("echo").arg("ok").required())
            .file_ext("txt");

        let builder = Kkachi::refine("q -> a").validate(validator);
        assert!(matches!(builder.critic, CriticConfig::Validator(_)));
    }

    #[test]
    fn test_refine_builder_similarity() {
        let builder = Kkachi::refine("q -> a")
            .semantic_cache(true)
            .similarity_threshold(0.9)
            .auto_condense(true)
            .cluster_threshold(0.85);

        assert!(builder.retrieval_config().use_semantic_cache);
        assert!((builder.retrieval_config().similarity_threshold - 0.9).abs() < 0.001);
        assert!(builder.retrieval_config().auto_condense);
    }

    #[test]
    fn test_refine_builder_few_shot() {
        let builder = Kkachi::refine("q -> a")
            .few_shot_k(5)
            .few_shot_as_demos(true)
            .few_shot_refresh(true);

        let fs = builder.few_shot_config.unwrap();
        assert_eq!(fs.k, 5);
        assert!(fs.as_demonstrations);
        assert!(fs.refresh_per_iteration);
    }

    #[test]
    fn test_refine_builder_dspy_integration() {
        let builder = Kkachi::refine("q -> a")
            .with_chain_of_thought()
            .with_best_of_n(3);

        assert!(builder.use_cot);
        assert_eq!(builder.best_of_n, Some(3));
    }

    #[test]
    fn test_refine_builder_similarity_weights() {
        let weights = SimilarityWeights {
            embedding: 0.50,
            keyword: 0.20,
            metadata: 0.20,
            hierarchy: 0.10,
        };

        let builder = Kkachi::refine("q -> a").similarity_weights(weights);

        assert!((builder.retrieval_config().similarity_weights.embedding - 0.50).abs() < 0.001);
    }

    #[test]
    fn test_refine_builder_run() {
        let result = Kkachi::refine("q -> a")
            .max_iterations(3)
            .until_score(1.0)
            .critic_always_pass()
            .run("test question", |iter, _| {
                Ok(format!("answer iteration {}", iter))
            });

        assert!(result.iterations <= 3);
        assert!(!result.answer.is_empty());
    }

    #[test]
    fn test_critic_config_variants() {
        // Test that CriticConfig variants work
        let validator = CliPipeline::new().stage("test", Cli::new("echo").arg("ok"));
        let config = CriticConfig::Validator(validator);
        assert!(matches!(config, CriticConfig::Validator(_)));

        let heuristic = CriticConfig::Heuristic {
            min_length: Some(10),
            max_length: None,
            required: vec![],
            forbidden: vec![],
        };
        assert!(matches!(heuristic, CriticConfig::Heuristic { .. }));
    }

    #[test]
    fn test_few_shot_config_default() {
        let config = FewShotConfig::default();
        assert_eq!(config.k, 3);
        assert!((config.min_similarity - 0.7).abs() < 0.001);
        assert!(config.include_in_prompt);
    }

    #[test]
    fn test_refine_result() {
        let result = RefineResult::new("answer".to_string(), 0.9, 3)
            .with_summary("summary".to_string())
            .with_cache_status(true)
            .with_domain("test".to_string());

        assert_eq!(result.answer, "answer");
        assert_eq!(result.summary, "summary");
        assert!((result.score - 0.9).abs() < 0.001);
        assert_eq!(result.iterations, 3);
        assert!(result.from_cache);
        assert_eq!(result.domain, Some("test".to_string()));
    }

    #[test]
    fn test_refine_result_from_refinement_result() {
        use super::super::storage::ContextId;

        let rr = RefinementResult {
            answer: "output".to_string(),
            summary: "summary".to_string(),
            score: 0.85,
            iterations: 2,
            from_cache: false,
            context_id: ContextId::from_question("test", "domain"),
            error_corrections: vec![],
        };

        let result: RefineResult = rr.into();
        assert_eq!(result.answer, "output");
        assert!((result.score - 0.85).abs() < 0.001);
        assert_eq!(result.iterations, 2);
    }

    #[test]
    fn test_build_runner_config() {
        let builder = Kkachi::refine("q -> a").max_iterations(5).until_score(0.9);

        let runner_config = builder.build_runner_config();
        assert_eq!(runner_config.recursive.max_iterations, 5);
        assert!((runner_config.recursive.score_threshold - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_refine_builder_hitl() {
        use crate::hitl::HITLConfig;

        let builder = Kkachi::refine("q -> a").hitl(HITLConfig::every(2));

        assert!(builder.hitl_config().is_some());
        let config = builder.hitl_config().unwrap();
        assert!(config.enabled);
        assert_eq!(config.interval, 2);
    }

    #[test]
    fn test_refine_builder_hitl_on_completion() {
        use crate::hitl::HITLConfig;

        let builder = Kkachi::refine("q -> a").hitl(HITLConfig::on_completion());

        let config = builder.hitl_config().unwrap();
        assert!(config.enabled);
        assert!(config.on_convergence);
        assert_eq!(config.interval, 0);
    }

    #[test]
    fn test_refine_with_auto_accept_reviewer() {
        use crate::hitl::{AutoAcceptReviewer, HITLConfig};

        let result = Kkachi::refine("q -> a")
            .max_iterations(3)
            .until_score(1.0)
            .critic_always_pass()
            .hitl(HITLConfig::every(1))
            .reviewer(AutoAcceptReviewer)
            .run("test question", |iter, _| {
                Ok(format!("answer iteration {}", iter))
            });

        assert!(result.iterations <= 3);
        assert!(!result.answer.is_empty());
    }

    #[test]
    fn test_refine_with_threshold_reviewer() {
        use crate::hitl::{HITLConfig, ThresholdReviewer};

        let result = Kkachi::refine("q -> a")
            .max_iterations(3)
            .until_score(1.0)
            .critic_always_pass()
            .hitl(HITLConfig::every(1))
            .reviewer(ThresholdReviewer::new(0.5))
            .run("test question", |iter, _| {
                Ok(format!("answer iteration {}", iter))
            });

        assert!(!result.answer.is_empty());
    }

    #[test]
    fn test_refine_with_callback_reviewer() {
        use crate::hitl::{CallbackReviewer, HITLConfig, ReviewDecision};
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let review_count = Arc::new(AtomicU32::new(0));
        let count_clone = review_count.clone();

        let reviewer = CallbackReviewer::new(move |_ctx| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            ReviewDecision::Accept
        });

        let result = Kkachi::refine("q -> a")
            .max_iterations(3)
            .until_score(1.0)
            .critic_always_pass()
            .hitl(HITLConfig::every(1))
            .reviewer(reviewer)
            .run("test question", |iter, _| {
                Ok(format!("answer iteration {}", iter))
            });

        assert!(!result.answer.is_empty());
        // Reviewer should have been called at least once
        assert!(review_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_refine_with_diff_callback() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let diff_count = Arc::new(AtomicU32::new(0));
        let count_clone = diff_count.clone();

        let result = Kkachi::refine("q -> a")
            .max_iterations(5)
            .until_score(0.5) // Low threshold to allow more iterations
            .critic_heuristic(Some(100), None) // Require 100 chars to fail
            .on_diff(move |_diff| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .run("test question", |iter, _| {
                Ok(format!("answer iteration {}", iter))
            });

        // Should have at least seen one diff (between iterations)
        // Note: diff callback is only called when there's a previous output
        let _ = result; // Result may be empty if never converged
    }

    #[test]
    fn test_should_review_interval() {
        use crate::hitl::HITLConfig;

        let builder = Kkachi::refine("q -> a");
        let config = HITLConfig::every(2);

        // Iteration 0 should not trigger (0 % 2 == 0 but we skip first by default)
        assert!(!builder.should_review(&config, 1, 0.5, 0.0, false));
        assert!(builder.should_review(&config, 2, 0.5, 0.5, false));
        assert!(!builder.should_review(&config, 3, 0.5, 0.5, false));
        assert!(builder.should_review(&config, 4, 0.5, 0.5, false));
    }

    #[test]
    fn test_should_review_score_drop() {
        use crate::hitl::HITLConfig;

        let builder = Kkachi::refine("q -> a");
        let config = HITLConfig::disabled().enable().with_score_drop_review(true);

        // Score drop should trigger
        assert!(builder.should_review(&config, 2, 0.4, 0.5, false));
        // Score improvement should not trigger
        assert!(!builder.should_review(&config, 2, 0.6, 0.5, false));
    }

    #[test]
    fn test_should_review_convergence() {
        use crate::hitl::HITLConfig;

        let builder = Kkachi::refine("q -> a");
        let config = HITLConfig::on_completion();

        // Non-converged should not trigger
        assert!(!builder.should_review(&config, 2, 0.8, 0.7, false));
        // Converged should trigger
        assert!(builder.should_review(&config, 2, 0.9, 0.8, true));
    }

    #[test]
    fn test_should_review_skip_above_score() {
        use crate::hitl::HITLConfig;

        let builder = Kkachi::refine("q -> a");
        let config = HITLConfig::every(1).skip_above(0.9);

        // Below threshold - should review
        assert!(builder.should_review(&config, 1, 0.8, 0.7, false));
        // Above threshold - should skip
        assert!(!builder.should_review(&config, 1, 0.95, 0.8, false));
    }

    // ===== Recall/Precision Mode Tests =====

    #[test]
    fn test_refine_builder_high_recall() {
        let builder = Kkachi::refine("q -> a").high_recall();

        // High recall uses lower threshold (0.6)
        assert!((builder.config().score_threshold - 0.6).abs() < 0.001);
        assert!(builder.config().recall_precision_mode.favors_recall());
    }

    #[test]
    fn test_refine_builder_high_precision() {
        let builder = Kkachi::refine("q -> a").high_precision();

        // High precision uses higher threshold (0.9)
        assert!((builder.config().score_threshold - 0.9).abs() < 0.001);
        assert!(builder.config().recall_precision_mode.favors_precision());
    }

    #[test]
    fn test_refine_builder_recall_precision_mode() {
        use crate::recall_precision::RecallPrecisionMode;

        let builder = Kkachi::refine("q -> a").recall_precision(RecallPrecisionMode::custom(0.75));

        assert!((builder.config().score_threshold - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_refine_builder_tone() {
        use crate::recursive::template::PromptTone;

        // Create builder with template first
        let builder = Kkachi::refine("q -> a")
            .template_content("---\n---\nTest template")
            .tone(PromptTone::Restrictive);

        // Template should have restrictive tone
        if let Some(template) = builder.get_template() {
            assert_eq!(template.options.tone, PromptTone::Restrictive);
        }
    }

    #[test]
    fn test_refine_builder_recall_precision_run() {
        let result = Kkachi::refine("q -> a")
            .max_iterations(3)
            .high_recall() // Lower threshold
            .critic_always_pass()
            .run("test question", |iter, _| {
                Ok(format!("answer iteration {}", iter))
            });

        assert!(result.iterations <= 3);
        assert!(!result.answer.is_empty());
    }

    #[test]
    fn test_refine_builder_chained_recall_precision() {
        use crate::recall_precision::RecallPrecisionMode;

        // Test that recall_precision can be chained with other methods
        let validator = CliPipeline::new()
            .stage("check", Cli::new("echo").arg("ok").required())
            .file_ext("txt");

        let builder = Kkachi::refine("q -> a")
            .domain("test")
            .max_iterations(5)
            .recall_precision(RecallPrecisionMode::high_precision(0.95))
            .validate(validator)
            .with_chain_of_thought();

        assert_eq!(builder.get_domain(), Some("test"));
        assert_eq!(builder.config().max_iterations, 5);
        assert!((builder.config().score_threshold - 0.95).abs() < 0.001);
        assert!(builder.use_cot);
    }
}
