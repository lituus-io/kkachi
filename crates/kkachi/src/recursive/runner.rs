// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Recursive runner with storage integration.
//!
//! Provides a high-level API for running recursive refinement with:
//! - Storage integration for caching successful refinements
//! - Configurable critics for output validation
//! - Progress tracking and reporting

use smallvec::SmallVec;

use crate::error::Result;
use crate::str_view::StrView;

use super::critic::Critic;
use super::state::{RecursiveConfig, RecursiveState};
use super::storage::ContextId;
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use super::storage::{ContextStore, ContextUpdate};

/// Result from a refinement run.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Final answer.
    pub answer: String,
    /// Concise summary (if generated).
    pub summary: String,
    /// Final quality score.
    pub score: f64,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether the result came from cache.
    pub from_cache: bool,
    /// Context ID for this question.
    pub context_id: ContextId,
    /// Error corrections made during refinement.
    pub error_corrections: Vec<(String, String)>,
}

/// Progress callback for monitoring refinement.
pub trait ProgressCallback: Send + Sync {
    /// Called when an iteration completes.
    fn on_iteration(&self, iteration: u32, score: f64, feedback: Option<&str>);

    /// Called when refinement completes.
    fn on_complete(&self, result: &RefinementResult);

    /// Called when using cached result.
    fn on_cache_hit(&self, context_id: &ContextId);
}

/// No-op progress callback.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl ProgressCallback for NoProgress {
    fn on_iteration(&self, _: u32, _: f64, _: Option<&str>) {}
    fn on_complete(&self, _: &RefinementResult) {}
    fn on_cache_hit(&self, _: &ContextId) {}
}

/// Simple stdout progress callback.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrintProgress;

impl ProgressCallback for PrintProgress {
    fn on_iteration(&self, iteration: u32, score: f64, feedback: Option<&str>) {
        println!(
            "[Iteration {}] Score: {:.2}{}",
            iteration,
            score,
            feedback.map(|f| format!(" - {}", f)).unwrap_or_default()
        );
    }

    fn on_complete(&self, result: &RefinementResult) {
        println!(
            "Completed in {} iterations with score {:.2}{}",
            result.iterations,
            result.score,
            if result.from_cache { " (cached)" } else { "" }
        );
    }

    fn on_cache_hit(&self, context_id: &ContextId) {
        println!("Cache hit: {}", context_id);
    }
}

/// Runner configuration.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Recursive refinement configuration.
    pub recursive: RecursiveConfig,
    /// Domain for storage namespacing.
    pub domain: String,
    /// Minimum score to store in cache.
    pub min_store_score: f64,
    /// Whether to check cache before running.
    pub use_cache: bool,
    /// Minimum cache score to return without refinement.
    pub cache_threshold: f64,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            recursive: RecursiveConfig::default(),
            domain: "default".to_string(),
            min_store_score: 0.7,
            use_cache: true,
            cache_threshold: 0.9,
        }
    }
}

impl RunnerConfig {
    /// Create with a specific domain.
    pub fn with_domain(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            ..Default::default()
        }
    }

    /// Set the score threshold.
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.recursive.score_threshold = threshold;
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.recursive.max_iterations = max;
        self
    }

    /// Set cache usage.
    pub fn with_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = use_cache;
        self
    }

    /// Set minimum store score.
    pub fn with_min_store_score(mut self, score: f64) -> Self {
        self.min_store_score = score;
        self
    }
}

/// Recursive runner that integrates refinement with storage.
///
/// ## Type Parameters
///
/// - `C`: Critic for evaluating outputs
/// - `G`: Generator function for producing outputs
/// - `P`: Progress callback
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub struct RecursiveRunner<'a, C, P = NoProgress>
where
    C: Critic,
    P: ProgressCallback,
{
    /// Critic for evaluation.
    critic: &'a C,
    /// Storage for caching.
    storage: &'a ContextStore,
    /// Configuration.
    config: RunnerConfig,
    /// Progress callback.
    progress: P,
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<'a, C: Critic> RecursiveRunner<'a, C, NoProgress> {
    /// Create a new runner.
    pub fn new(critic: &'a C, storage: &'a ContextStore, config: RunnerConfig) -> Self {
        Self {
            critic,
            storage,
            config,
            progress: NoProgress,
        }
    }
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<'a, C, P> RecursiveRunner<'a, C, P>
where
    C: Critic,
    P: ProgressCallback,
{
    /// Add progress callback.
    pub fn with_progress<P2: ProgressCallback>(self, progress: P2) -> RecursiveRunner<'a, C, P2> {
        RecursiveRunner {
            critic: self.critic,
            storage: self.storage,
            config: self.config,
            progress,
        }
    }

    /// Run refinement with a generator function.
    ///
    /// The generator function takes the current iteration and optional feedback,
    /// and returns the generated output.
    pub fn refine<G>(&self, question: &str, mut generate: G) -> Result<RefinementResult>
    where
        G: FnMut(u32, Option<&str>) -> Result<String>,
    {
        let context_id = ContextId::from_question(question, &self.config.domain);

        // Check cache
        if self.config.use_cache {
            if let Some(cached) = self.storage.lookup(question, &self.config.domain) {
                if cached.score as f64 >= self.config.cache_threshold {
                    self.progress.on_cache_hit(&context_id);
                    return Ok(RefinementResult {
                        answer: cached.answer,
                        summary: cached.summary,
                        score: cached.score as f64,
                        iterations: 0,
                        from_cache: true,
                        context_id,
                        error_corrections: vec![],
                    });
                }
            }
        }

        // Run refinement loop
        let mut iteration = 0u32;
        let mut scores: SmallVec<[f64; 8]> = SmallVec::new();
        let mut last_feedback: Option<String> = None;
        let mut error_corrections: Vec<(String, String)> = Vec::new();
        let mut best_output = String::new();
        let mut best_score = 0.0f64;

        loop {
            // Generate output
            let output = generate(iteration, last_feedback.as_deref())?;

            // Evaluate with a temporary state
            let (score, feedback, is_satisfactory) = {
                let temp_state = RecursiveState::with_scores(&scores);
                let eval = self.critic.evaluate(StrView::new(&output), &temp_state);
                let is_sat = eval.is_satisfactory();
                (eval.score, eval.feedback, is_sat)
            };
            scores.push(score);

            // Track best
            if score > best_score {
                best_score = score;
                best_output = output.clone();
            }

            // Report progress
            self.progress
                .on_iteration(iteration, score, feedback.as_deref());

            // Check convergence
            if score >= self.config.recursive.score_threshold || is_satisfactory {
                best_output = output;
                best_score = score;
                break;
            }

            // Safety check
            if iteration >= self.config.recursive.max_iterations {
                break;
            }

            // Record error correction if there was feedback
            if let Some(ref fb) = feedback {
                error_corrections.push((fb.clone(), String::new()));
            }

            // Store feedback for next iteration
            last_feedback = feedback;
            iteration += 1;
        }

        // Store if score is high enough
        if best_score >= self.config.min_store_score {
            let _ = self.storage.upsert(&ContextUpdate {
                question,
                domain: &self.config.domain,
                answer: &best_output,
                summary: "",
                score: best_score as f32,
                iterations: iteration as u16,
                error_corrections: &error_corrections,
            });
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

        self.progress.on_complete(&result);

        Ok(result)
    }
}

/// Standalone runner without storage.
pub struct StandaloneRunner<'a, C, P = NoProgress>
where
    C: Critic,
    P: ProgressCallback,
{
    /// Critic for evaluation.
    critic: &'a C,
    /// Configuration.
    config: RecursiveConfig,
    /// Domain for context ID generation.
    domain: String,
    /// Progress callback.
    progress: P,
}

impl<'a, C: Critic> StandaloneRunner<'a, C, NoProgress> {
    /// Create a new standalone runner.
    pub fn new(critic: &'a C, domain: impl Into<String>) -> Self {
        Self {
            critic,
            config: RecursiveConfig::default(),
            domain: domain.into(),
            progress: NoProgress,
        }
    }

    /// Create with configuration.
    pub fn with_config(critic: &'a C, domain: impl Into<String>, config: RecursiveConfig) -> Self {
        Self {
            critic,
            config,
            domain: domain.into(),
            progress: NoProgress,
        }
    }
}

impl<'a, C, P> StandaloneRunner<'a, C, P>
where
    C: Critic,
    P: ProgressCallback,
{
    /// Add progress callback.
    pub fn with_progress<P2: ProgressCallback>(self, progress: P2) -> StandaloneRunner<'a, C, P2> {
        StandaloneRunner {
            critic: self.critic,
            config: self.config,
            domain: self.domain,
            progress,
        }
    }

    /// Run refinement with a generator function.
    pub fn refine<G>(&self, question: &str, mut generate: G) -> Result<RefinementResult>
    where
        G: FnMut(u32, Option<&str>) -> Result<String>,
    {
        let context_id = ContextId::from_question(question, &self.domain);
        let mut iteration = 0u32;
        let mut scores: SmallVec<[f64; 8]> = SmallVec::new();
        let mut last_feedback: Option<String> = None;
        let mut error_corrections: Vec<(String, String)> = Vec::new();
        let mut best_output = String::new();
        let mut best_score = 0.0f64;

        loop {
            // Generate output
            let output = generate(iteration, last_feedback.as_deref())?;

            // Evaluate with a temporary state for this call
            // This avoids lifetime issues with the Critic trait
            let (score, feedback, is_satisfactory) = {
                let temp_state = RecursiveState::with_scores(&scores);
                let eval = self.critic.evaluate(StrView::new(&output), &temp_state);
                let is_sat = eval.is_satisfactory();
                (eval.score, eval.feedback, is_sat)
            };
            scores.push(score);

            // Track best
            if score > best_score {
                best_score = score;
                best_output = output.clone();
            }

            // Report progress
            self.progress
                .on_iteration(iteration, score, feedback.as_deref());

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

            // Store feedback for next iteration
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

        self.progress.on_complete(&result);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::critic::BinaryCritic;

    #[test]
    fn test_runner_config_defaults() {
        let config = RunnerConfig::default();
        assert_eq!(config.domain, "default");
        assert!(config.use_cache);
        assert_eq!(config.cache_threshold, 0.9);
    }

    #[test]
    fn test_runner_config_builder() {
        let config = RunnerConfig::with_domain("test")
            .with_score_threshold(0.95)
            .with_max_iterations(5)
            .with_cache(false);

        assert_eq!(config.domain, "test");
        assert_eq!(config.recursive.score_threshold, 0.95);
        assert_eq!(config.recursive.max_iterations, 5);
        assert!(!config.use_cache);
    }

    #[test]
    fn test_standalone_runner() {
        let critic = BinaryCritic::new(|s| s.len() > 10, "Too short");
        let runner = StandaloneRunner::new(&critic, "test");

        let result = runner
            .refine("test question", |iteration, _feedback| {
                if iteration == 0 {
                    Ok("short".to_string())
                } else {
                    Ok("this is a longer output that should pass".to_string())
                }
            })
            .unwrap();

        assert!(result.score > 0.5);
        assert!(result.answer.len() > 10);
        assert!(!result.from_cache);
    }

    #[test]
    fn test_standalone_runner_with_progress() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let critic = BinaryCritic::new(|s| s.contains("correct"), "Missing 'correct'");

        struct CountingProgress {
            count: Arc<AtomicU32>,
        }

        impl ProgressCallback for CountingProgress {
            fn on_iteration(&self, _: u32, _: f64, _: Option<&str>) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            fn on_complete(&self, _: &RefinementResult) {}
            fn on_cache_hit(&self, _: &ContextId) {}
        }

        let count = Arc::new(AtomicU32::new(0));
        let progress = CountingProgress {
            count: count.clone(),
        };

        let runner = StandaloneRunner::new(&critic, "test").with_progress(progress);

        let _ = runner
            .refine("test", |iteration, _| {
                if iteration < 2 {
                    Ok("wrong".to_string())
                } else {
                    Ok("correct answer".to_string())
                }
            })
            .unwrap();

        assert!(count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_refinement_result() {
        let result = RefinementResult {
            answer: "test".to_string(),
            summary: String::new(),
            score: 0.9,
            iterations: 3,
            from_cache: false,
            context_id: ContextId::from_question("test", "domain"),
            error_corrections: vec![("error".to_string(), "fix".to_string())],
        };

        assert_eq!(result.answer, "test");
        assert_eq!(result.score, 0.9);
        assert_eq!(result.iterations, 3);
        assert!(!result.from_cache);
    }
}
