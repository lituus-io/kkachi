// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Core Recursive module implementation
//!
//! Wraps any module with iterative refinement capabilities.

use std::marker::PhantomData;

use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::module::Module;
use crate::prediction::Prediction;
use crate::types::Inputs;

use super::criterion::ConvergenceCriterion;
use super::critic::Critic;
use super::state::{RecursiveConfig, RecursiveState};

/// Core recursive prompting module.
///
/// Wraps an inner module with iterative refinement based on critic feedback
/// and convergence criteria.
///
/// ## Type Parameters
///
/// - `M`: Inner module (Predict, ChainOfThought, etc.)
/// - `C`: Critic for quality evaluation
/// - `V`: Convergence criterion for stopping
///
/// ## Example
///
/// ```ignore
/// use kkachi::recursive::*;
///
/// let recursive = Recursive::new(predict, critic, convergence)
///     .with_max_iterations(5)
///     .with_history(true);
///
/// let result = recursive.refine(inputs, &lm, &mut buffer).await?;
/// ```
pub struct Recursive<'a, M, C, V>
where
    M: Module,
    C: Critic,
    V: ConvergenceCriterion,
{
    /// Inner module to wrap
    inner: M,
    /// Critic for evaluation
    critic: C,
    /// Convergence criterion (reserved for future use)
    #[allow(dead_code)]
    convergence: V,
    /// Configuration
    config: RecursiveConfig,
    /// Lifetime marker
    _marker: PhantomData<&'a ()>,
}

impl<'a, M, C, V> Recursive<'a, M, C, V>
where
    M: Module,
    C: Critic,
    V: ConvergenceCriterion,
{
    /// Create a new recursive module.
    pub fn new(inner: M, critic: C, convergence: V) -> Self {
        Self {
            inner,
            critic,
            convergence,
            config: RecursiveConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set score threshold.
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.config.score_threshold = threshold;
        self
    }

    /// Enable/disable history inclusion in prompts.
    pub fn with_history(mut self, include: bool) -> Self {
        self.config.include_history = include;
        self
    }

    /// Set history window size.
    pub fn with_history_window(mut self, window: u8) -> Self {
        self.config.history_window = window;
        self
    }

    /// Set temperature decay.
    pub fn with_temperature_decay(mut self, decay: f32) -> Self {
        self.config.temperature_decay = decay;
        self
    }

    /// Get inner module reference.
    #[inline]
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Get critic reference.
    #[inline]
    pub fn critic(&self) -> &C {
        &self.critic
    }

    /// Get configuration.
    #[inline]
    pub fn config(&self) -> &RecursiveConfig {
        &self.config
    }
}

impl<M: Module + Send + Sync, C: Critic, V: ConvergenceCriterion> Module
    for Recursive<'_, M, C, V>
{
    type ForwardFut<'b>
        = std::future::Ready<Result<Prediction<'b>>>
    where
        Self: 'b;

    fn forward<'b>(&'b self, _inputs: Inputs<'b>) -> Self::ForwardFut<'b> {
        // For Module trait compliance, return an error directing to use refine()
        std::future::ready(Err(crate::error::Error::module(
            "Use Recursive::refine() for iterative refinement instead of forward()",
        )))
    }

    fn name(&self) -> &str {
        "Recursive"
    }

    fn id(&self) -> Sym {
        sym("recursive")
    }
}

/// Builder for creating recursive modules with ergonomic API.
pub struct RecursiveBuilder<'a, M> {
    inner: M,
    config: RecursiveConfig,
    _marker: PhantomData<&'a ()>,
}

impl<'a, M: Module> RecursiveBuilder<'a, M> {
    /// Create a new builder with an inner module.
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            config: RecursiveConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Set maximum iterations.
    pub fn max_iterations(mut self, n: u32) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set score threshold.
    pub fn until_score(mut self, threshold: f64) -> Self {
        self.config.score_threshold = threshold;
        self
    }

    /// Set plateau detection.
    pub fn until_plateau(mut self, min_improvement: f64, window: usize) -> Self {
        self.config.plateau_threshold = Some(min_improvement);
        self.config.plateau_window = Some(window);
        self
    }

    /// Enable chain of thought.
    pub fn with_chain_of_thought(mut self) -> Self {
        self.config.use_cot = true;
        self
    }

    /// Enable best-of-n sampling.
    pub fn with_best_of_n(mut self, n: u8) -> Self {
        self.config.best_of_n = Some(n);
        self
    }

    /// Include history in prompts.
    pub fn with_history(mut self, include: bool) -> Self {
        self.config.include_history = include;
        self
    }

    /// Build with a critic and convergence criterion.
    pub fn build<C: Critic, V: ConvergenceCriterion>(
        self,
        critic: C,
        convergence: V,
    ) -> Recursive<'a, M, C, V> {
        let mut recursive = Recursive::new(self.inner, critic, convergence);
        recursive.config = self.config;
        recursive
    }
}

/// Result from evaluating output in refine_loop.
pub struct EvalResult {
    /// Quality score (0.0 - 1.0)
    pub score: f64,
    /// Feedback for improvement (None if satisfactory)
    pub feedback: Option<String>,
}

impl EvalResult {
    /// Create a new evaluation result.
    #[inline]
    pub fn new(score: f64) -> Self {
        Self {
            score,
            feedback: None,
        }
    }

    /// Add feedback.
    #[inline]
    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    /// Check if satisfactory (no feedback needed).
    #[inline]
    pub fn is_satisfactory(&self) -> bool {
        self.feedback.is_none()
    }
}

/// Simple refinement loop that doesn't require full module infrastructure.
///
/// This function provides a standalone refinement loop for quick prototyping.
/// Uses simple closures instead of the Critic trait to avoid lifetime complexity.
///
/// # Arguments
/// * `generate` - Closure that generates output given state and optional feedback
/// * `evaluate` - Closure that evaluates output and returns score + feedback
/// * `config` - Recursive configuration
///
/// # Returns
/// The final output string and state with score history.
pub fn refine_loop<G, E>(
    mut generate: G,
    mut evaluate: E,
    config: &RecursiveConfig,
) -> Result<(String, RecursiveState<'static>)>
where
    G: FnMut(u32, Option<&str>) -> Result<String>,
    E: FnMut(&str) -> EvalResult,
{
    let mut state: RecursiveState<'static> = RecursiveState::new();
    let mut last_feedback: Option<String> = None;

    loop {
        // Generate output
        let output = generate(state.iteration, last_feedback.as_deref())?;

        // Evaluate
        let eval = evaluate(&output);
        state.record_score(eval.score);

        // Check convergence: score threshold or no feedback
        if eval.score >= config.score_threshold || eval.is_satisfactory() {
            return Ok((output, state));
        }

        // Safety check
        if state.iteration >= config.max_iterations {
            return Ok((output, state));
        }

        // Store feedback for next iteration
        last_feedback = eval.feedback;
        state.iteration += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction::Prediction;
    use crate::recursive::criterion::MaxIterations;
    use crate::recursive::critic::BinaryCritic;

    struct DummyModule;

    impl Module for DummyModule {
        type ForwardFut<'a> = std::future::Ready<Result<Prediction<'a>>>;

        fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
            std::future::ready(Ok(Prediction::new()))
        }
    }

    #[test]
    fn test_recursive_creation() {
        let inner = DummyModule;
        let critic = BinaryCritic::new(|s| s.len() > 10, "Too short");
        let convergence = MaxIterations(5);

        let recursive = Recursive::new(inner, critic, convergence)
            .with_max_iterations(10)
            .with_history(true);

        assert_eq!(recursive.config().max_iterations, 10);
        assert!(recursive.config().include_history);
    }

    #[test]
    fn test_recursive_builder() {
        let inner = DummyModule;
        let critic = BinaryCritic::new(|s| s.len() > 10, "Too short");
        let convergence = MaxIterations(5);

        let recursive = RecursiveBuilder::new(inner)
            .max_iterations(7)
            .until_score(0.95)
            .with_chain_of_thought()
            .build(critic, convergence);

        assert_eq!(recursive.config().max_iterations, 7);
        assert_eq!(recursive.config().score_threshold, 0.95);
        assert!(recursive.config().use_cot);
    }

    #[test]
    fn test_refine_loop() {
        let generate = |_iteration: u32, feedback: Option<&str>| {
            if feedback.is_some() {
                Ok("this is a longer output that should pass".to_string())
            } else {
                Ok("short".to_string())
            }
        };

        let evaluate = |s: &str| {
            if s.len() > 10 {
                EvalResult::new(1.0)
            } else {
                EvalResult::new(0.0).with_feedback("Too short")
            }
        };

        let config = RecursiveConfig::default();

        let (output, state) = refine_loop(generate, evaluate, &config).unwrap();

        assert!(output.len() > 10);
        assert!(state.iteration > 0);
    }
}
