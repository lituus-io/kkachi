// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! RAG-optimized recursive workflow.
//!
//! This module provides the core RAG optimization workflow:
//! 1. RAG lookup for context
//! 2. Multi-stage recursive refinement with CLI validation
//! 3. One-shot prompt formation
//! 4. Fresh context testing
//! 5. Conditional RAG write-back
//!
//! # Zero-Copy Design
//!
//! - Generic over store, embedder, and stages (no `Box<dyn>`)
//! - Type-level stage composition using tuples
//! - `StrView<'a>` and `SmallVec` for zero-allocation hot paths
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::{RagOptimizer, Cli, RagDocumentTemplate};
//!
//! let preview = Cli::new("pulumi")
//!     .args(["preview", "--non-interactive"])
//!     .env("GOOGLE_PROJECT", "my-project")
//!     .required();
//!
//! let template = RagDocumentTemplate::new("infra")
//!     .header("Task", 2)
//!     .text("question")
//!     .code("code", "yaml");
//!
//! let result = RagOptimizer::new(&mut store, &embedder)
//!     .stage("preview", preview)
//!     .template(template)
//!     .run("Create a GCS bucket", &llm)
//!     .await?;
//! ```

use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::error::Result;
use crate::recursive::{Embedder, MutableVectorStore, Validator, VectorStore};
use crate::str_view::StrView;

use super::doc_template::RagDocumentTemplate;
use super::oneshot::{ErrorCorrection, OneShotFailure, OneShotPrompt, OneShotTestResult};
use super::pipeline::LLM;

// =============================================================================
// Stage Configuration
// =============================================================================

/// Configuration for a single optimization stage.
#[derive(Clone, Debug)]
pub struct StageConfig {
    /// Maximum refinement iterations for this stage.
    pub max_iterations: u32,
    /// Score threshold to pass this stage.
    pub threshold: f64,
    /// Whether this stage must pass to continue.
    pub required: bool,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            threshold: 0.9,
            required: true,
        }
    }
}

// =============================================================================
// Stage
// =============================================================================

/// A single optimization stage with a validator.
///
/// # Type Parameters
///
/// * `V` - The validator type (e.g., `Cli`, `CliPipeline`)
pub struct Stage<V: Validator> {
    /// Stage name.
    name: String,
    /// Validator for this stage.
    validator: V,
    /// Stage configuration.
    config: StageConfig,
}

impl<V: Validator> Stage<V> {
    /// Create a new stage.
    pub fn new(name: impl Into<String>, validator: V) -> Self {
        Self {
            name: name.into(),
            validator,
            config: StageConfig::default(),
        }
    }

    /// Set stage configuration.
    pub fn with_config(mut self, config: StageConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum iterations.
    pub fn max_iterations(mut self, max: u32) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Set threshold.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set whether this stage is required.
    pub fn required(mut self, required: bool) -> Self {
        self.config.required = required;
        self
    }

    /// Get the stage name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// =============================================================================
// Stage Result
// =============================================================================

/// Result from running a single stage.
#[derive(Clone, Debug)]
pub struct StageResult {
    /// Stage name.
    pub name: String,
    /// Final output.
    pub output: String,
    /// Final score.
    pub score: f64,
    /// Number of iterations.
    pub iterations: u32,
    /// Whether the stage passed.
    pub passed: bool,
    /// Errors encountered.
    pub errors: Vec<String>,
}

// =============================================================================
// Optimizer Configuration
// =============================================================================

/// Configuration for the RAG optimizer.
#[derive(Clone, Debug)]
pub struct RagOptimizerConfig {
    /// Minimum similarity to use RAG context.
    pub rag_threshold: f64,
    /// Maximum refinement iterations (inner loop).
    pub max_iterations: u32,
    /// Maximum one-shot optimization attempts (outer loop).
    pub max_optimization_attempts: u32,
    /// Score threshold for convergence.
    pub convergence_threshold: f64,
    /// Score threshold for one-shot test pass.
    pub oneshot_threshold: f64,
    /// Whether to write back to RAG on success.
    pub write_back: bool,
}

impl Default for RagOptimizerConfig {
    fn default() -> Self {
        Self {
            rag_threshold: 0.7,
            max_iterations: 10,
            max_optimization_attempts: 3,
            convergence_threshold: 0.95,
            oneshot_threshold: 0.9,
            write_back: true,
        }
    }
}

// =============================================================================
// Optimization Result
// =============================================================================

/// Result from the RAG optimization workflow.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Final answer.
    pub answer: String,
    /// Final score.
    pub score: f64,
    /// Total refinement iterations.
    pub iterations: u32,
    /// Number of optimization attempts (one-shot retries).
    pub optimization_attempts: u32,
    /// Whether one-shot testing passed.
    pub oneshot_passed: bool,
    /// Whether RAG was updated.
    pub rag_updated: bool,
    /// Error corrections learned.
    pub error_corrections: Vec<(String, String, u32)>, // (error, fix, iteration)
    /// Per-stage results.
    pub stage_results: Vec<(String, f64)>,
}

// =============================================================================
// RAG Optimizer
// =============================================================================

/// RAG-optimized recursive workflow orchestrator.
///
/// # Type Parameters
///
/// * `'a` - Lifetime for borrowed references
/// * `S` - Vector store type
/// * `E` - Embedder type
/// * `Stages` - Stage list type (type-level composition)
///
/// # Zero-Copy Design
///
/// - Generic over store, embedder, and stages (no `Box<dyn>`)
/// - Stages are composed at the type level using tuples
/// - References are borrowed, not cloned
///
/// # Example
///
/// ```rust,ignore
/// let result = RagOptimizer::new(&mut store, &embedder)
///     .stage("preview", preview_validator)
///     .stage("deploy", deploy_validator)
///     .template(template)
///     .config(config)
///     .run("How do I create a bucket?", &llm)
///     .await?;
/// ```
pub struct RagOptimizer<'a, S, E, Stages = ()>
where
    S: VectorStore,
    E: Embedder,
{
    /// Vector store for RAG lookup and write-back.
    store: &'a mut S,
    /// Embedder for similarity search.
    embedder: &'a E,
    /// Stages (type-level composition).
    stages: Stages,
    /// Document template.
    template: Option<RagDocumentTemplate<'a>>,
    /// Configuration.
    config: RagOptimizerConfig,
    /// Marker for lifetime.
    _marker: PhantomData<&'a ()>,
}

impl<'a, S, E> RagOptimizer<'a, S, E, ()>
where
    S: VectorStore,
    E: Embedder,
{
    /// Create a new RAG optimizer with no stages.
    pub fn new(store: &'a mut S, embedder: &'a E) -> Self {
        Self {
            store,
            embedder,
            stages: (),
            template: None,
            config: RagOptimizerConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<'a, S, E, Stages> RagOptimizer<'a, S, E, Stages>
where
    S: VectorStore,
    E: Embedder,
{
    /// Add a stage (type accumulates).
    pub fn stage<V: Validator + Send + Sync + 'static>(
        self,
        name: &str,
        validator: V,
    ) -> RagOptimizer<'a, S, E, (Stages, Stage<V>)> {
        RagOptimizer {
            store: self.store,
            embedder: self.embedder,
            stages: (self.stages, Stage::new(name, validator)),
            template: self.template,
            config: self.config,
            _marker: PhantomData,
        }
    }

    /// Set the document template.
    pub fn template(mut self, template: RagDocumentTemplate<'a>) -> Self {
        self.template = Some(template);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: RagOptimizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the RAG similarity threshold.
    pub fn rag_threshold(mut self, threshold: f64) -> Self {
        self.config.rag_threshold = threshold;
        self
    }

    /// Set the one-shot threshold.
    pub fn oneshot_threshold(mut self, threshold: f64) -> Self {
        self.config.oneshot_threshold = threshold;
        self
    }

    /// Set the maximum optimization attempts.
    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.config.max_optimization_attempts = attempts;
        self
    }

    /// Enable or disable RAG write-back.
    pub fn write_back(mut self, enabled: bool) -> Self {
        self.config.write_back = enabled;
        self
    }
}

// =============================================================================
// Single Stage Implementation
// =============================================================================

impl<'a, S, E, V> RagOptimizer<'a, S, E, ((), Stage<V>)>
where
    S: VectorStore + MutableVectorStore,
    E: Embedder,
    V: Validator + Send + Sync + 'a,
{
    /// Run the optimization with a single stage.
    pub async fn run<L: LLM + Send + Sync>(
        mut self,
        question: &'a str,
        llm: &'a L,
    ) -> Result<OptimizationResult> {
        // Phase 1: RAG lookup
        let rag_context = self.lookup_rag(question);

        // Accumulated failures for retry context
        let mut failures: Vec<OneShotFailure> = Vec::new();

        for attempt in 0..self.config.max_optimization_attempts {
            // Phase 2: Refine with the single stage
            let stage_result = self
                .refine_stage(
                    question,
                    rag_context.as_deref(),
                    &failures,
                    &self.stages.1,
                    llm,
                )
                .await?;

            // Phase 3: Form one-shot prompt
            let corrections: SmallVec<[ErrorCorrection; 8]> = stage_result
                .errors
                .iter()
                .enumerate()
                .map(|(i, e)| ErrorCorrection::new(e.clone(), String::new(), i as u32))
                .collect();

            let oneshot = OneShotPrompt {
                question: StrView::new(question),
                answer: stage_result.output.clone(),
                error_corrections: corrections,
                score: stage_result.score,
                iterations: stage_result.iterations,
            };

            // Phase 4: Test in fresh context
            let test_result = self.test_oneshot(&oneshot, &self.stages.1, llm).await?;

            if test_result.passed {
                // Phase 5: Write back to RAG
                let rag_updated = if self.config.write_back {
                    self.write_back_internal(question, &oneshot);
                    true
                } else {
                    false
                };

                return Ok(OptimizationResult {
                    answer: stage_result.output,
                    score: stage_result.score,
                    iterations: stage_result.iterations,
                    optimization_attempts: attempt + 1,
                    oneshot_passed: true,
                    rag_updated,
                    error_corrections: oneshot
                        .error_corrections
                        .iter()
                        .map(|ec| (ec.error.clone(), ec.fix.clone(), ec.iteration))
                        .collect(),
                    stage_results: vec![(stage_result.name.clone(), stage_result.score)],
                });
            }

            // Capture failure for next attempt
            failures.push(OneShotFailure::new(
                attempt,
                oneshot.render(),
                test_result.output.clone(),
                test_result.errors.clone(),
            ));
        }

        // Max attempts exhausted
        Ok(OptimizationResult {
            answer: String::new(),
            score: 0.0,
            iterations: 0,
            optimization_attempts: self.config.max_optimization_attempts,
            oneshot_passed: false,
            rag_updated: false,
            error_corrections: Vec::new(),
            stage_results: Vec::new(),
        })
    }

    /// Lookup RAG context.
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

    /// Refine a single stage.
    async fn refine_stage<L: LLM + Send + Sync>(
        &self,
        question: &str,
        rag_context: Option<&str>,
        failures: &[OneShotFailure],
        stage: &Stage<V>,
        llm: &L,
    ) -> Result<StageResult> {
        let mut current_output = String::new();
        let mut iteration = 0u32;
        let mut last_errors: Vec<String> = Vec::new();

        loop {
            // Build prompt
            let prompt = self.build_prompt(
                question,
                rag_context,
                failures,
                &current_output,
                &last_errors,
            );

            // Generate
            let output = llm.generate(&prompt).await?;

            // Validate
            let validation = stage.validator.validate(&output)?;

            if validation.passed && validation.score >= stage.config.threshold {
                return Ok(StageResult {
                    name: stage.name.clone(),
                    output,
                    score: validation.score,
                    iterations: iteration + 1,
                    passed: true,
                    errors: validation.errors,
                });
            }

            // Check max iterations
            if iteration >= stage.config.max_iterations {
                return Ok(StageResult {
                    name: stage.name.clone(),
                    output,
                    score: validation.score,
                    iterations: iteration + 1,
                    passed: false,
                    errors: validation.errors,
                });
            }

            // Update for next iteration
            current_output = output;
            last_errors = validation.errors;
            iteration += 1;
        }
    }

    /// Build prompt for generation.
    fn build_prompt(
        &self,
        question: &str,
        rag_context: Option<&str>,
        failures: &[OneShotFailure],
        current: &str,
        errors: &[String],
    ) -> String {
        let mut prompt = String::with_capacity(2048);

        prompt.push_str("Task: ");
        prompt.push_str(question);
        prompt.push_str("\n\n");

        if let Some(context) = rag_context {
            prompt.push_str("Relevant example:\n");
            prompt.push_str(context);
            prompt.push_str("\n\n");
        }

        if !failures.is_empty() {
            prompt.push_str("Previous one-shot attempts failed:\n");
            for failure in failures {
                prompt.push_str(&failure.summary());
            }
            prompt.push('\n');
        }

        if !current.is_empty() {
            prompt.push_str("Previous output:\n");
            prompt.push_str(current);
            prompt.push_str("\n\n");
        }

        if !errors.is_empty() {
            prompt.push_str("Errors to fix:\n");
            for error in errors {
                prompt.push_str("- ");
                prompt.push_str(error);
                prompt.push('\n');
            }
            prompt.push_str("\nPlease fix these errors.\n\n");
        }

        prompt.push_str("Output:");
        prompt
    }

    /// Test one-shot in fresh context.
    async fn test_oneshot<L: LLM + Send + Sync>(
        &self,
        oneshot: &OneShotPrompt<'_>,
        stage: &Stage<V>,
        llm: &L,
    ) -> Result<OneShotTestResult> {
        // Generate in fresh context
        let prompt = oneshot.render();
        let output = llm.generate_fresh(&prompt).await?;

        // Validate
        let validation = stage.validator.validate(&output)?;

        if validation.passed && validation.score >= self.config.oneshot_threshold {
            Ok(OneShotTestResult::pass(output, validation.score))
        } else {
            Ok(OneShotTestResult::fail(
                output,
                validation.score,
                validation.errors,
            ))
        }
    }

    /// Write back to RAG store.
    fn write_back_internal(&mut self, question: &str, oneshot: &OneShotPrompt<'_>) {
        use super::doc_template::TemplateSection;
        use std::fmt::Write;

        // Format document
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
                        // Error corrections list - currently empty
                        doc.push('\n');
                    }
                }
            }

            // Add metadata footer
            writeln!(
                doc,
                "---\n_Score: {:.2} | Iterations: {} | One-shot validated: true_",
                oneshot.score, oneshot.iterations
            )
            .unwrap();

            doc
        } else {
            // Simple format without template
            format!(
                "Question: {}\n\nAnswer:\n{}\n\nScore: {:.2}",
                question, oneshot.answer, oneshot.score
            )
        };

        // Generate ID from question hash
        let id = format!("rag:{:x}", hash_string(question));

        // Upsert to store
        self.store.add(id, doc);
    }
}

// =============================================================================
// Two Stage Implementation
// =============================================================================

impl<'a, S, E, V1, V2> RagOptimizer<'a, S, E, (((), Stage<V1>), Stage<V2>)>
where
    S: VectorStore + MutableVectorStore,
    E: Embedder,
    V1: Validator + Send + Sync + 'a,
    V2: Validator + Send + Sync + 'a,
{
    /// Run the optimization with two stages.
    ///
    /// Stage 1 runs first. If it passes, Stage 2 runs with Stage 1's output.
    pub async fn run<L: LLM + Send + Sync>(
        mut self,
        question: &'a str,
        llm: &'a L,
    ) -> Result<OptimizationResult> {
        // For two stages, we run stage 1 (e.g., preview) then stage 2 (e.g., deploy)
        // Phase 1: RAG lookup
        let rag_context = self.lookup_rag(question);

        let mut failures: Vec<OneShotFailure> = Vec::new();
        let stage1 = &self.stages.0 .1;
        let stage2 = &self.stages.1;

        for attempt in 0..self.config.max_optimization_attempts {
            // Phase 2a: Refine stage 1
            let result1 = self
                .refine_stage(question, rag_context.as_deref(), &failures, stage1, llm)
                .await?;

            if !result1.passed {
                // Stage 1 failed, retry
                failures.push(OneShotFailure::new(
                    attempt,
                    question.to_string(),
                    result1.output.clone(),
                    result1.errors.clone(),
                ));
                continue;
            }

            // Phase 2b: Refine stage 2 with stage 1's output
            let result2 = self
                .refine_stage_with_input(question, &result1.output, &failures, stage2, llm)
                .await?;

            if !result2.passed {
                // Stage 2 failed, retry
                failures.push(OneShotFailure::new(
                    attempt,
                    result1.output.clone(),
                    result2.output.clone(),
                    result2.errors.clone(),
                ));
                continue;
            }

            // Phase 3: Form one-shot prompt from stage 1 result
            // (Stage 1 is what we test in one-shot since it's the generation step)
            let corrections: SmallVec<[ErrorCorrection; 8]> = result1
                .errors
                .iter()
                .enumerate()
                .map(|(i, e)| ErrorCorrection::new(e.clone(), String::new(), i as u32))
                .collect();

            let oneshot = OneShotPrompt {
                question: StrView::new(question),
                answer: result1.output.clone(),
                error_corrections: corrections,
                score: result1.score,
                iterations: result1.iterations + result2.iterations,
            };

            // Phase 4: Test in fresh context (stage 1 validation)
            let test_result = self.test_oneshot(&oneshot, stage1, llm).await?;

            if test_result.passed {
                // Phase 5: Write back to RAG
                let rag_updated = if self.config.write_back {
                    self.write_back_internal(question, &oneshot);
                    true
                } else {
                    false
                };

                return Ok(OptimizationResult {
                    answer: result1.output,
                    score: (result1.score + result2.score) / 2.0,
                    iterations: result1.iterations + result2.iterations,
                    optimization_attempts: attempt + 1,
                    oneshot_passed: true,
                    rag_updated,
                    error_corrections: oneshot
                        .error_corrections
                        .iter()
                        .map(|ec| (ec.error.clone(), ec.fix.clone(), ec.iteration))
                        .collect(),
                    stage_results: vec![
                        (result1.name.clone(), result1.score),
                        (result2.name.clone(), result2.score),
                    ],
                });
            }

            // One-shot failed, retry
            failures.push(OneShotFailure::new(
                attempt,
                oneshot.render(),
                test_result.output.clone(),
                test_result.errors.clone(),
            ));
        }

        // Max attempts exhausted
        Ok(OptimizationResult {
            answer: String::new(),
            score: 0.0,
            iterations: 0,
            optimization_attempts: self.config.max_optimization_attempts,
            oneshot_passed: false,
            rag_updated: false,
            error_corrections: Vec::new(),
            stage_results: Vec::new(),
        })
    }

    /// Lookup RAG context.
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

    /// Refine a stage.
    async fn refine_stage<L: LLM + Send + Sync, V: Validator>(
        &self,
        question: &str,
        rag_context: Option<&str>,
        failures: &[OneShotFailure],
        stage: &Stage<V>,
        llm: &L,
    ) -> Result<StageResult> {
        let mut current_output = String::new();
        let mut iteration = 0u32;
        let mut last_errors: Vec<String> = Vec::new();

        loop {
            let prompt = self.build_prompt(
                question,
                rag_context,
                failures,
                &current_output,
                &last_errors,
            );
            let output = llm.generate(&prompt).await?;
            let validation = stage.validator.validate(&output)?;

            if validation.passed && validation.score >= stage.config.threshold {
                return Ok(StageResult {
                    name: stage.name.clone(),
                    output,
                    score: validation.score,
                    iterations: iteration + 1,
                    passed: true,
                    errors: validation.errors,
                });
            }

            if iteration >= stage.config.max_iterations {
                return Ok(StageResult {
                    name: stage.name.clone(),
                    output,
                    score: validation.score,
                    iterations: iteration + 1,
                    passed: false,
                    errors: validation.errors,
                });
            }

            current_output = output;
            last_errors = validation.errors;
            iteration += 1;
        }
    }

    /// Refine a stage with specific input (for stage 2).
    async fn refine_stage_with_input<L: LLM + Send + Sync, V: Validator>(
        &self,
        _question: &str,
        input: &str,
        _failures: &[OneShotFailure],
        stage: &Stage<V>,
        llm: &L,
    ) -> Result<StageResult> {
        // For stage 2, we validate the output from stage 1
        // The prompt is simpler - just validate/fix the input
        let mut current_output = input.to_string();
        let mut iteration = 0u32;
        let mut last_errors: Vec<String> = Vec::new();

        loop {
            let validation = stage.validator.validate(&current_output)?;

            if validation.passed && validation.score >= stage.config.threshold {
                return Ok(StageResult {
                    name: stage.name.clone(),
                    output: current_output,
                    score: validation.score,
                    iterations: iteration + 1,
                    passed: true,
                    errors: validation.errors,
                });
            }

            if iteration >= stage.config.max_iterations {
                return Ok(StageResult {
                    name: stage.name.clone(),
                    output: current_output,
                    score: validation.score,
                    iterations: iteration + 1,
                    passed: false,
                    errors: validation.errors,
                });
            }

            // Build prompt to fix errors
            let mut prompt = String::with_capacity(1024);
            prompt.push_str("Fix the following errors in this code:\n\n");
            prompt.push_str(&current_output);
            prompt.push_str("\n\nErrors:\n");
            for error in &last_errors {
                prompt.push_str("- ");
                prompt.push_str(error);
                prompt.push('\n');
            }
            prompt.push_str("\nFixed code:");

            current_output = llm.generate(&prompt).await?;
            last_errors = validation.errors;
            iteration += 1;
        }
    }

    /// Build prompt for generation.
    fn build_prompt(
        &self,
        question: &str,
        rag_context: Option<&str>,
        failures: &[OneShotFailure],
        current: &str,
        errors: &[String],
    ) -> String {
        let mut prompt = String::with_capacity(2048);

        prompt.push_str("Task: ");
        prompt.push_str(question);
        prompt.push_str("\n\n");

        if let Some(context) = rag_context {
            prompt.push_str("Relevant example:\n");
            prompt.push_str(context);
            prompt.push_str("\n\n");
        }

        if !failures.is_empty() {
            prompt.push_str("Previous attempts failed:\n");
            for failure in failures {
                prompt.push_str(&failure.summary());
            }
            prompt.push('\n');
        }

        if !current.is_empty() {
            prompt.push_str("Previous output:\n");
            prompt.push_str(current);
            prompt.push_str("\n\n");
        }

        if !errors.is_empty() {
            prompt.push_str("Errors to fix:\n");
            for error in errors {
                prompt.push_str("- ");
                prompt.push_str(error);
                prompt.push('\n');
            }
            prompt.push_str("\nPlease fix these errors.\n\n");
        }

        prompt.push_str("Output:");
        prompt
    }

    /// Test one-shot in fresh context.
    async fn test_oneshot<L: LLM + Send + Sync, V: Validator>(
        &self,
        oneshot: &OneShotPrompt<'_>,
        stage: &Stage<V>,
        llm: &L,
    ) -> Result<OneShotTestResult> {
        let prompt = oneshot.render();
        let output = llm.generate_fresh(&prompt).await?;
        let validation = stage.validator.validate(&output)?;

        if validation.passed && validation.score >= self.config.oneshot_threshold {
            Ok(OneShotTestResult::pass(output, validation.score))
        } else {
            Ok(OneShotTestResult::fail(
                output,
                validation.score,
                validation.errors,
            ))
        }
    }

    /// Write back to RAG store.
    fn write_back_internal(&mut self, question: &str, oneshot: &OneShotPrompt<'_>) {
        use super::doc_template::TemplateSection;
        use std::fmt::Write;

        // Format document
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
                        // Error corrections list - currently empty
                        doc.push('\n');
                    }
                }
            }

            // Add metadata footer
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

        let id = format!("rag:{:x}", hash_string(question));
        self.store.add(id, doc);
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Simple string hash for document IDs.
fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config_default() {
        let config = StageConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.threshold, 0.9);
        assert!(config.required);
    }

    #[test]
    fn test_optimizer_config_default() {
        let config = RagOptimizerConfig::default();
        assert_eq!(config.rag_threshold, 0.7);
        assert_eq!(config.max_optimization_attempts, 3);
        assert_eq!(config.oneshot_threshold, 0.9);
        assert!(config.write_back);
    }

    #[test]
    fn test_hash_string() {
        let h1 = hash_string("test");
        let h2 = hash_string("test");
        let h3 = hash_string("different");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
