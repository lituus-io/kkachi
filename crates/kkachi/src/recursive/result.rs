// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Result types for the refinement loop.
//!
//! This module provides the output types for refinement operations,
//! including iteration history, corrections, and optimized prompts.

use crate::error::Result;
use crate::recursive::cli::CliCapture;
use crate::recursive::llm::Llm;
use crate::recursive::validate::Validate;
use smallvec::SmallVec;
use std::fmt;
use std::time::Duration;

/// Reason why the refinement loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopReason {
    /// Target score was reached.
    TargetReached,
    /// Maximum iterations exceeded.
    #[default]
    MaxIterations,
    /// Token budget was exhausted.
    BudgetExhausted,
    /// Timeout was reached.
    TimeoutReached,
    /// Score plateaued (no improvement).
    Plateau,
    /// Human reviewer accepted the output.
    HumanAccepted,
    /// Human reviewer rejected and requested stop.
    HumanRejected,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetReached => write!(f, "target reached"),
            Self::MaxIterations => write!(f, "max iterations"),
            Self::BudgetExhausted => write!(f, "budget exhausted"),
            Self::TimeoutReached => write!(f, "timeout reached"),
            Self::Plateau => write!(f, "plateau"),
            Self::HumanAccepted => write!(f, "human accepted"),
            Self::HumanRejected => write!(f, "human rejected"),
        }
    }
}

/// Unique identifier for a refinement context.
///
/// This can be used to link related refinements together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ContextId(pub u64);

impl ContextId {
    /// Generate a new unique context ID.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // Combine timestamp with a counter to ensure uniqueness
        let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(nanos.wrapping_add(counter))
    }
}

impl fmt::Display for ContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// A single iteration in the refinement loop.
#[derive(Debug, Clone)]
pub struct Iteration {
    /// The iteration number (0-based).
    pub number: u32,
    /// The output generated in this iteration.
    pub output: String,
    /// The validation score.
    pub score: f64,
    /// Feedback from the validator (if any).
    pub feedback: Option<String>,
}

/// A correction made during refinement.
///
/// Records an error and its resolution for learning.
#[derive(Debug, Clone)]
pub struct Correction {
    /// Description of the error.
    pub error: String,
    /// How the error was resolved.
    pub resolution: String,
    /// The iteration in which this correction was made.
    pub iteration: u32,
}

/// An example used in few-shot prompting.
#[derive(Debug, Clone)]
pub struct Example {
    /// The input/question.
    pub input: String,
    /// The expected output/answer.
    pub output: String,
}

impl Example {
    /// Create a new example.
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }
}

/// An optimized prompt produced by refinement.
///
/// Contains the refined signature, instructions, and examples
/// that can be used for future generations.
#[derive(Debug, Clone)]
pub struct OptimizedPrompt {
    /// The signature (e.g., "question -> answer").
    pub signature: String,
    /// Refined instructions.
    pub instructions: String,
    /// Few-shot examples.
    pub examples: SmallVec<[Example; 4]>,
    /// The final rendered template.
    pub template: String,
}

impl OptimizedPrompt {
    /// Create a new optimized prompt.
    pub fn new(signature: impl Into<String>) -> Self {
        Self {
            signature: signature.into(),
            instructions: String::new(),
            examples: SmallVec::new(),
            template: String::new(),
        }
    }

    /// Add instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Add an example.
    pub fn with_example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.examples.push(Example::new(input, output));
        self
    }

    /// Set the template.
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = template.into();
        self
    }

    /// Render the prompt for display.
    pub fn render(&self) -> String {
        let mut result = format!("Signature: {}\n", self.signature);
        if !self.instructions.is_empty() {
            result.push_str("\nInstructions:\n");
            result.push_str(&self.instructions);
            result.push('\n');
        }
        if !self.examples.is_empty() {
            result.push_str("\nExamples:\n");
            for (i, ex) in self.examples.iter().enumerate() {
                result.push_str(&format!("  {}. Input: {}\n", i + 1, ex.input));
                result.push_str(&format!("     Output: {}\n", ex.output));
            }
        }
        result
    }
}

/// The result of a refinement operation.
///
/// Contains the final output along with metadata about the refinement
/// process, including iteration history and any corrections made.
#[derive(Debug, Clone)]
pub struct RefineResult {
    /// The final refined output.
    pub output: String,
    /// The final validation score.
    pub score: f64,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Unique identifier for this refinement context.
    pub context_id: ContextId,
    /// Whether the result was retrieved from cache.
    pub from_cache: bool,
    /// The optimized prompt (if generated).
    pub prompt: Option<OptimizedPrompt>,
    /// History of all iterations.
    pub history: SmallVec<[Iteration; 8]>,
    /// Corrections made during refinement.
    pub corrections: SmallVec<[Correction; 8]>,
    /// Captured CLI outputs (if any).
    pub cli_captures: SmallVec<[CliCapture; 4]>,
    /// Why the refinement loop stopped.
    pub stop_reason: StopReason,
    /// Total tokens consumed (prompt + completion).
    pub total_tokens: u32,
    /// Total wall-clock time elapsed.
    pub elapsed: Duration,
    /// Aggregate confidence from validators.
    ///
    /// For deterministic validators this is 1.0.
    /// For semantic validators, reflects the judge's certainty.
    pub confidence: f64,
}

impl RefineResult {
    /// Create a new refinement result.
    pub fn new(output: impl Into<String>, score: f64, iterations: u32) -> Self {
        Self {
            output: output.into(),
            score,
            iterations,
            context_id: ContextId::new(),
            from_cache: false,
            prompt: None,
            history: SmallVec::new(),
            corrections: SmallVec::new(),
            cli_captures: SmallVec::new(),
            stop_reason: StopReason::MaxIterations,
            total_tokens: 0,
            elapsed: Duration::ZERO,
            confidence: 1.0,
        }
    }

    /// Create a cached result.
    pub fn cached(output: impl Into<String>, score: f64) -> Self {
        Self {
            output: output.into(),
            score,
            iterations: 0,
            context_id: ContextId::new(),
            from_cache: true,
            prompt: None,
            history: SmallVec::new(),
            corrections: SmallVec::new(),
            cli_captures: SmallVec::new(),
            stop_reason: StopReason::TargetReached,
            total_tokens: 0,
            elapsed: Duration::ZERO,
            confidence: 1.0,
        }
    }

    /// Add an iteration to the history.
    pub fn add_iteration(&mut self, iteration: Iteration) {
        self.history.push(iteration);
    }

    /// Add a correction.
    pub fn add_correction(&mut self, error: impl Into<String>, resolution: impl Into<String>) {
        self.corrections.push(Correction {
            error: error.into(),
            resolution: resolution.into(),
            iteration: self.iterations,
        });
    }

    /// Add a CLI capture.
    pub fn add_cli_capture(&mut self, capture: CliCapture) {
        self.cli_captures.push(capture);
    }

    /// Get CLI output for a specific stage by name.
    pub fn cli_output(&self, stage: &str) -> Option<&str> {
        self.cli_captures
            .iter()
            .find(|c| c.stage == stage)
            .map(|c| c.stdout.as_str())
    }

    /// Generate a summary of CLI outputs.
    pub fn cli_summary(&self) -> String {
        if self.cli_captures.is_empty() {
            return String::new();
        }

        let mut summary = String::from("## CLI Outputs\n\n");
        for capture in &self.cli_captures {
            summary.push_str(&format!("### {}\n", capture.stage));
            if capture.success {
                summary.push_str("**Status:** Success\n");
            } else {
                summary.push_str(&format!(
                    "**Status:** Failed (exit code: {})\n",
                    capture.exit_code.unwrap_or(-1)
                ));
            }
            if !capture.stdout.is_empty() {
                summary.push_str("\n**stdout:**\n```\n");
                summary.push_str(&capture.stdout);
                summary.push_str("\n```\n");
            }
            if !capture.stderr.is_empty() {
                summary.push_str("\n**stderr:**\n```\n");
                summary.push_str(&capture.stderr);
                summary.push_str("\n```\n");
            }
            summary.push('\n');
        }
        summary
    }

    /// Generate a markdown table of corrections.
    pub fn corrections_table(&self) -> String {
        if self.corrections.is_empty() {
            return String::new();
        }

        let mut table = String::from("| Iteration | Error | Resolution |\n");
        table.push_str("|-----------|-------|------------|\n");

        for correction in &self.corrections {
            table.push_str(&format!(
                "| {} | {} | {} |\n",
                correction.iteration,
                correction.error.replace('|', "\\|"),
                correction.resolution.replace('|', "\\|")
            ));
        }
        table
    }

    /// Generate a markdown summary of corrections.
    pub fn corrections_markdown(&self) -> String {
        if self.corrections.is_empty() {
            return String::from("No corrections were needed.\n");
        }

        let mut md = String::from("## Corrections Made\n\n");
        for (i, correction) in self.corrections.iter().enumerate() {
            md.push_str(&format!(
                "### Correction {} (Iteration {})\n\n",
                i + 1,
                correction.iteration
            ));
            md.push_str(&format!("**Error:** {}\n\n", correction.error));
            md.push_str(&format!("**Resolution:** {}\n\n", correction.resolution));
        }
        md
    }

    /// Check if refinement was successful.
    pub fn is_success(&self) -> bool {
        self.score >= 1.0 - f64::EPSILON
    }

    /// Get the improvement over iterations.
    pub fn improvement(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let first = self.history.first().map(|h| h.score).unwrap_or(0.0);
        self.score - first
    }
}

impl fmt::Display for RefineResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RefineResult {{")?;
        writeln!(f, "  score: {:.2}", self.score)?;
        writeln!(f, "  confidence: {:.2}", self.confidence)?;
        writeln!(f, "  iterations: {}", self.iterations)?;
        writeln!(f, "  stop_reason: {}", self.stop_reason)?;
        writeln!(f, "  total_tokens: {}", self.total_tokens)?;
        writeln!(f, "  elapsed: {:?}", self.elapsed)?;
        writeln!(f, "  context_id: {}", self.context_id)?;
        writeln!(f, "  from_cache: {}", self.from_cache)?;
        writeln!(f, "  corrections: {}", self.corrections.len())?;
        writeln!(
            f,
            "  output: {}...",
            &self.output[..self.output.len().min(50)]
        )?;
        write!(f, "}}")
    }
}

/// A compiled program ready for prediction.
///
/// This holds the optimized prompt and can be used for fast inference
/// without re-running the refinement loop.
pub struct Compiled<L: Llm, V: Validate> {
    llm: L,
    validator: V,
    prompt: OptimizedPrompt,
    target_score: f64,
}

impl<L: Llm, V: Validate> Compiled<L, V> {
    /// Create a new compiled program.
    pub fn new(llm: L, validator: V, prompt: OptimizedPrompt) -> Self {
        Self {
            llm,
            validator,
            prompt,
            target_score: 1.0,
        }
    }

    /// Set the target score threshold.
    pub fn with_target(mut self, score: f64) -> Self {
        self.target_score = score;
        self
    }

    /// Get the optimized prompt.
    pub fn prompt(&self) -> &OptimizedPrompt {
        &self.prompt
    }

    /// Get a reference to the validator.
    pub fn validator(&self) -> &V {
        &self.validator
    }

    /// Make a prediction using the compiled program.
    pub async fn predict(&self, input: &str) -> String {
        // Build context from examples
        let mut context = String::new();
        for ex in &self.prompt.examples {
            context.push_str(&format!("Input: {}\nOutput: {}\n\n", ex.input, ex.output));
        }

        // Generate
        match self.llm.generate(input, &context, None).await {
            Ok(output) => output.text,
            Err(_) => String::new(),
        }
    }

    /// Make a prediction and return the score.
    pub async fn predict_scored(&self, input: &str) -> (String, f64) {
        let output = self.predict(input).await;
        let score = self.validator.validate(&output);
        (output, score.value)
    }

    /// Save the compiled program to a file.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let content = self.prompt.render();
        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_id() {
        let id1 = ContextId::new();
        let id2 = ContextId::new();
        // IDs should be different (high probability)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_refine_result_new() {
        let result = RefineResult::new("output", 0.8, 3);
        assert_eq!(result.output, "output");
        assert!((result.score - 0.8).abs() < f64::EPSILON);
        assert_eq!(result.iterations, 3);
        assert!(!result.from_cache);
    }

    #[test]
    fn test_refine_result_cached() {
        let result = RefineResult::cached("cached output", 1.0);
        assert!(result.from_cache);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_add_iteration() {
        let mut result = RefineResult::new("output", 0.8, 1);
        result.add_iteration(Iteration {
            number: 0,
            output: "first".to_string(),
            score: 0.5,
            feedback: Some("improve".to_string()),
        });
        assert_eq!(result.history.len(), 1);
        assert_eq!(result.history[0].score, 0.5);
    }

    #[test]
    fn test_add_correction() {
        let mut result = RefineResult::new("output", 0.8, 1);
        result.add_correction("missing return type", "added -> i32");
        assert_eq!(result.corrections.len(), 1);
        assert_eq!(result.corrections[0].error, "missing return type");
    }

    #[test]
    fn test_corrections_table() {
        let mut result = RefineResult::new("output", 1.0, 2);
        result.add_correction("error 1", "fix 1");
        result.add_correction("error 2", "fix 2");

        let table = result.corrections_table();
        assert!(table.contains("Iteration"));
        assert!(table.contains("error 1"));
        assert!(table.contains("fix 2"));
    }

    #[test]
    fn test_cli_summary() {
        let mut result = RefineResult::new("output", 1.0, 1);
        result.add_cli_capture(CliCapture {
            stage: "compile".to_string(),
            command: "rustc".to_string(),
            stdout: "compiled".to_string(),
            stderr: String::new(),
            success: true,
            exit_code: Some(0),
            duration_ms: 100,
        });

        let summary = result.cli_summary();
        assert!(summary.contains("compile"));
        assert!(summary.contains("Success"));
    }

    #[test]
    fn test_optimized_prompt() {
        let prompt = OptimizedPrompt::new("question -> answer")
            .with_instructions("Be concise")
            .with_example("What is 2+2?", "4")
            .with_example("What is the capital of France?", "Paris");

        assert_eq!(prompt.signature, "question -> answer");
        assert_eq!(prompt.examples.len(), 2);

        let rendered = prompt.render();
        assert!(rendered.contains("question -> answer"));
        assert!(rendered.contains("Be concise"));
        assert!(rendered.contains("2+2"));
    }

    #[test]
    fn test_is_success() {
        let success = RefineResult::new("output", 1.0, 1);
        assert!(success.is_success());

        let partial = RefineResult::new("output", 0.8, 1);
        assert!(!partial.is_success());
    }

    #[test]
    fn test_improvement() {
        let mut result = RefineResult::new("output", 0.9, 2);
        result.add_iteration(Iteration {
            number: 0,
            output: "first".to_string(),
            score: 0.3,
            feedback: None,
        });
        result.add_iteration(Iteration {
            number: 1,
            output: "second".to_string(),
            score: 0.9,
            feedback: None,
        });

        assert!((result.improvement() - 0.6).abs() < 0.01);
    }
}
