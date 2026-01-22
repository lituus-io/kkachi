// Copyright 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multi-step pipeline orchestration.
//!
//! Provides:
//! - `Step<'a, C>` - Single step with own critic and refinement config
//! - `Steps<'a>` - Multi-step pipeline orchestrator
//! - `StepBuilder<'a, C>` - Fluent builder for step chains
//! - `steps!` macro - One-liner pipeline definition

use crate::error::Result;
use crate::recursive::{Critic, RecursiveState, Validator, ValidatorCritic};
use crate::str_view::StrView;

use super::pipeline::{NoCritic, LLM};

// =============================================================================
// Step
// =============================================================================

/// A single step in a pipeline with its own refinement configuration.
pub struct Step<'a, C: Critic = NoCritic> {
    /// Step name for identification.
    pub name: &'static str,
    /// Signature string.
    pub signature: StrView<'a>,
    /// Critic for this step.
    pub critic: C,
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Score threshold.
    pub threshold: f64,
}

impl<'a> Step<'a, NoCritic> {
    /// Create a new step with the given name and signature.
    pub fn new(name: &'static str, signature: &'a str) -> Self {
        Self {
            name,
            signature: StrView::new(signature),
            critic: NoCritic,
            max_iterations: 5,
            threshold: 0.9,
        }
    }
}

impl<'a, C: Critic> Step<'a, C> {
    /// Set critic for this step.
    pub fn with_critic<C2: Critic>(self, critic: C2) -> Step<'a, C2> {
        Step {
            name: self.name,
            signature: self.signature,
            critic,
            max_iterations: self.max_iterations,
            threshold: self.threshold,
        }
    }

    /// Set validator for this step.
    ///
    /// This wraps the validator in `ValidatorCritic` to satisfy the `Critic` trait.
    pub fn with_validator<V: Validator + 'static>(
        self,
        validator: V,
    ) -> Step<'a, ValidatorCritic<V>> {
        Step {
            name: self.name,
            signature: self.signature,
            critic: ValidatorCritic::new(validator),
            max_iterations: self.max_iterations,
            threshold: self.threshold,
        }
    }

    /// Set refinement parameters.
    pub fn refine(mut self, max_iter: u32, threshold: f64) -> Self {
        self.max_iterations = max_iter;
        self.threshold = threshold;
        self
    }

    /// Quick preset: 3 iterations, 0.8 threshold.
    pub fn quick(self) -> Self {
        self.refine(3, 0.8)
    }

    /// Standard preset: 5 iterations, 0.9 threshold.
    pub fn standard(self) -> Self {
        self.refine(5, 0.9)
    }

    /// Strict preset: 10 iterations, 0.95 threshold.
    pub fn strict(self) -> Self {
        self.refine(10, 0.95)
    }
}

// =============================================================================
// StepResult
// =============================================================================

/// Result from a single step execution.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step name.
    pub name: &'static str,
    /// Step output.
    pub output: String,
    /// Final score.
    pub score: f64,
    /// Number of iterations.
    pub iterations: u32,
    /// Final feedback (if any).
    pub feedback: Option<String>,
}

// =============================================================================
// StepsOutput
// =============================================================================

/// Output from multi-step pipeline execution.
#[derive(Debug, Clone)]
pub struct StepsOutput {
    /// Final output from last step.
    pub final_output: String,
    /// Results from each step.
    pub step_results: Vec<StepResult>,
}

impl StepsOutput {
    /// Get final output.
    pub fn output(&self) -> &str {
        &self.final_output
    }

    /// Total iterations across all steps.
    pub fn total_iterations(&self) -> u32 {
        self.step_results.iter().map(|r| r.iterations).sum()
    }

    /// Lowest score across all steps.
    pub fn min_score(&self) -> f64 {
        self.step_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, f64::min)
    }

    /// Get result for specific step by name.
    pub fn step(&self, name: &str) -> Option<&StepResult> {
        self.step_results.iter().find(|r| r.name == name)
    }
}

// =============================================================================
// AnyStep (Type-Erased Step)
// =============================================================================

/// Type-erased step for heterogeneous collections.
pub struct AnyStep<'a> {
    name: &'static str,
    signature: StrView<'a>,
    max_iterations: u32,
    threshold: f64,
    // Boxed evaluate function for type erasure
    evaluate_fn: Box<dyn Fn(&str) -> (f64, bool, Option<String>) + Send + Sync + 'a>,
}

impl<'a> AnyStep<'a> {
    /// Create from a typed step.
    pub fn new<C: Critic + Send + Sync + 'a>(step: Step<'a, C>) -> Self {
        let critic = step.critic;
        let threshold = step.threshold;
        Self {
            name: step.name,
            signature: step.signature,
            max_iterations: step.max_iterations,
            threshold: step.threshold,
            evaluate_fn: Box::new(move |output| {
                let state = RecursiveState::new();
                let result = critic.evaluate(StrView::new(output), &state);
                let passed = result.score >= threshold;
                (result.score, passed, result.feedback)
            }),
        }
    }

    /// Get step name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Run this step recursively until convergence.
    pub async fn run_recursive<L: LLM + Send + Sync>(
        &self,
        input: &str,
        llm: &L,
    ) -> Result<StepResult> {
        let mut current = input.to_string();
        let mut iteration = 0u32;
        let mut last_feedback: Option<String> = None;

        loop {
            // Evaluate
            let (score, _passed, feedback) = (self.evaluate_fn)(&current);

            // Check convergence
            if score >= self.threshold || iteration >= self.max_iterations {
                return Ok(StepResult {
                    name: self.name,
                    output: current,
                    score,
                    iterations: iteration,
                    feedback: last_feedback,
                });
            }

            // Build prompt
            let prompt = format!(
                "Task: {}\n\nInput: {}\n\n{}Output:",
                self.signature.as_str(),
                current,
                if let Some(ref fb) = feedback {
                    format!("Feedback: {}\n\nPlease improve your response.\n\n", fb)
                } else {
                    String::new()
                }
            );

            // Generate
            current = llm.generate(&prompt).await?;
            last_feedback = feedback;
            iteration += 1;
        }
    }
}

// =============================================================================
// Steps (Multi-Step Pipeline)
// =============================================================================

/// Multi-step pipeline with automatic orchestration.
pub struct Steps<'a> {
    /// Steps to execute.
    steps: Vec<AnyStep<'a>>,
    /// Callback after each step completes.
    on_step_complete: Option<fn(&str, &StepResult)>,
}

impl<'a> Default for Steps<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Steps<'a> {
    /// Create empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            on_step_complete: None,
        }
    }

    /// Add a step.
    pub fn then<C: Critic + Send + Sync + 'a>(mut self, step: Step<'a, C>) -> Self {
        self.steps.push(AnyStep::new(step));
        self
    }

    /// Set callback for step completion.
    pub fn on_complete(mut self, f: fn(&str, &StepResult)) -> Self {
        self.on_step_complete = Some(f);
        self
    }

    /// Execute all steps in sequence.
    pub async fn run<L: LLM + Send + Sync>(self, input: &str, llm: &L) -> Result<StepsOutput> {
        let mut current = input.to_string();
        let mut step_results = Vec::new();

        for step in &self.steps {
            // Each step recursively refines until threshold or max_iter
            let result = step.run_recursive(&current, llm).await?;

            // Call completion callback
            if let Some(callback) = self.on_step_complete {
                callback(step.name(), &result);
            }

            // Update current for next step
            current = result.output.clone();
            step_results.push(result);
        }

        Ok(StepsOutput {
            final_output: current,
            step_results,
        })
    }
}

// =============================================================================
// StepBuilder (Fluent Builder)
// =============================================================================

/// Builder for configuring steps in a fluent chain.
pub struct StepBuilder<'a, C: Critic = NoCritic> {
    /// Collected steps.
    steps: Vec<AnyStep<'a>>,
    /// Current step being configured.
    current_step: Step<'a, C>,
    /// Completion callback.
    on_step_complete: Option<fn(&str, &StepResult)>,
}

impl<'a> StepBuilder<'a, NoCritic> {
    /// Create a new step builder starting with the given step.
    pub fn new(name: &'static str, signature: &'a str) -> Self {
        Self {
            steps: Vec::new(),
            current_step: Step::new(name, signature),
            on_step_complete: None,
        }
    }
}

impl<'a, C: Critic + Send + Sync + 'a> StepBuilder<'a, C> {
    /// Set refinement for current step.
    pub fn refine(mut self, max_iter: u32, threshold: f64) -> Self {
        self.current_step = self.current_step.refine(max_iter, threshold);
        self
    }

    /// Quick preset for current step.
    pub fn quick(mut self) -> Self {
        self.current_step = self.current_step.quick();
        self
    }

    /// Standard preset for current step.
    pub fn standard(mut self) -> Self {
        self.current_step = self.current_step.standard();
        self
    }

    /// Strict preset for current step.
    pub fn strict(mut self) -> Self {
        self.current_step = self.current_step.strict();
        self
    }

    /// Finish current step and start a new one.
    pub fn then(mut self, name: &'static str, signature: &'a str) -> StepBuilder<'a, NoCritic> {
        // Push current step
        self.steps.push(AnyStep::new(self.current_step));

        StepBuilder {
            steps: self.steps,
            current_step: Step::new(name, signature),
            on_step_complete: self.on_step_complete,
        }
    }

    /// Set completion callback.
    pub fn on_complete(mut self, f: fn(&str, &StepResult)) -> Self {
        self.on_step_complete = Some(f);
        self
    }

    /// Build into Steps and execute.
    pub async fn run<L: LLM + Send + Sync>(mut self, input: &str, llm: &L) -> Result<StepsOutput> {
        // Push final step
        self.steps.push(AnyStep::new(self.current_step));

        // Create Steps and run
        let steps = Steps {
            steps: self.steps,
            on_step_complete: self.on_step_complete,
        };

        steps.run(input, llm).await
    }
}

// =============================================================================
// steps! Macro
// =============================================================================

/// Create a multi-step pipeline in one line.
///
/// # Example
///
/// ```rust,ignore
/// let result = steps![
///     "outline": "topic -> outline" => quick,
///     "draft": "outline -> draft" => standard,
///     "edit": "draft -> final" => strict,
/// ].run("AI safety", &llm).await?;
/// ```
#[macro_export]
macro_rules! steps {
    // Base case: single step without config
    ($name:literal : $sig:literal) => {
        $crate::declarative::Steps::new()
            .then($crate::declarative::Step::new($name, $sig))
    };

    // Base case: single step with config
    ($name:literal : $sig:literal => $config:ident) => {
        $crate::declarative::Steps::new()
            .then($crate::declarative::Step::new($name, $sig).$config())
    };

    // Multiple steps without trailing comma
    ($($name:literal : $sig:literal $(=> $config:ident)?),+) => {
        $crate::declarative::Steps::new()
        $(
            .then($crate::declarative::Step::new($name, $sig) $(.$config())?)
        )+
    };

    // Multiple steps with trailing comma
    ($($name:literal : $sig:literal $(=> $config:ident)?),+ ,) => {
        steps![$($name : $sig $(=> $config)?),+]
    };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_creation() {
        let step = Step::new("test", "input -> output");
        assert_eq!(step.name, "test");
        assert_eq!(step.signature.as_str(), "input -> output");
        assert_eq!(step.max_iterations, 5);
        assert_eq!(step.threshold, 0.9);
    }

    #[test]
    fn test_step_presets() {
        let quick = Step::new("q", "a -> b").quick();
        assert_eq!(quick.max_iterations, 3);
        assert_eq!(quick.threshold, 0.8);

        let strict = Step::new("s", "a -> b").strict();
        assert_eq!(strict.max_iterations, 10);
        assert_eq!(strict.threshold, 0.95);
    }

    #[test]
    fn test_steps_output() {
        let output = StepsOutput {
            final_output: "final".to_string(),
            step_results: vec![
                StepResult {
                    name: "step1",
                    output: "out1".to_string(),
                    score: 0.9,
                    iterations: 2,
                    feedback: None,
                },
                StepResult {
                    name: "step2",
                    output: "out2".to_string(),
                    score: 0.95,
                    iterations: 3,
                    feedback: None,
                },
            ],
        };

        assert_eq!(output.output(), "final");
        assert_eq!(output.total_iterations(), 5);
        assert!((output.min_score() - 0.9).abs() < 0.001);
        assert!(output.step("step1").is_some());
        assert!(output.step("step3").is_none());
    }
}
