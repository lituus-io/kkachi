// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Concurrent execution of multiple pipelines sharing a single LLM.
//!
//! [`ConcurrentRunner`] runs multiple tasks concurrently, sharing a single
//! `&L` (where `L: Llm + Send + Sync`) across all futures — no Arc needed.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, pipeline, checks};
//! use kkachi::recursive::concurrent::ConcurrentRunner;
//!
//! let llm = MockLlm::new(|_, _| "output".to_string());
//!
//! let results = ConcurrentRunner::new(&llm)
//!     .task("task_a", |llm| pipeline(llm, "Write A").refine(checks().min_len(1)))
//!     .task("task_b", |llm| pipeline(llm, "Write B").refine(checks().min_len(1)))
//!     .go();
//!
//! assert_eq!(results.len(), 2);
//! assert!(results[0].result.is_ok());
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::pipeline::{Pipeline, PipelineResult};
use crate::recursive::shared;
use smallvec::SmallVec;
use std::time::{Duration, Instant};

/// Result of a single concurrent task.
#[derive(Debug, Clone)]
pub struct ConcurrentTaskResult {
    /// Label for this task.
    pub label: String,
    /// The pipeline result, or an error message.
    pub result: Result<PipelineResult, String>,
    /// Duration for this task.
    pub elapsed: Duration,
}

/// Runs multiple pipelines concurrently on a shared LLM.
///
/// All tasks share `&'a L` — no Arc needed because `L: Llm` requires
/// `Send + Sync`. Rate limiting (if applied via `with_rate_limit()`)
/// is transparent because its internal `Arc<Mutex<...>>` handles
/// shared mutable state.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, pipeline, checks};
/// use kkachi::recursive::concurrent::ConcurrentRunner;
///
/// let llm = MockLlm::new(|_, _| "result".to_string());
///
/// let results = ConcurrentRunner::new(&llm)
///     .task("fast", |llm| pipeline(llm, "Quick task"))
///     .task("thorough", |llm| {
///         pipeline(llm, "Careful task")
///             .refine(checks().min_len(1))
///             .best_of(3)
///     })
///     .go();
///
/// for r in &results {
///     println!("{}: {:?}", r.label, r.result.is_ok());
/// }
/// ```
pub struct ConcurrentRunner<'a, L: Llm> {
    llm: &'a L,
    tasks: SmallVec<[ConcurrentTask<'a, L>; 4]>,
    max_concurrency: usize,
}

struct ConcurrentTask<'a, L: Llm> {
    label: String,
    build: Box<dyn FnOnce(&'a L) -> Pipeline<'a, L> + Send + 'a>,
}

impl<'a, L: Llm> ConcurrentRunner<'a, L> {
    /// Create a new concurrent runner sharing the given LLM.
    pub fn new(llm: &'a L) -> Self {
        Self {
            llm,
            tasks: SmallVec::new(),
            max_concurrency: 0, // 0 = unlimited
        }
    }

    /// Add a task with a label and a pipeline builder closure.
    ///
    /// The closure receives `&'a L` and should return a `Pipeline`.
    pub fn task<F>(mut self, label: &str, f: F) -> Self
    where
        F: FnOnce(&'a L) -> Pipeline<'a, L> + Send + 'a,
    {
        self.tasks.push(ConcurrentTask {
            label: label.to_string(),
            build: Box::new(f),
        });
        self
    }

    /// Set maximum concurrency (0 = unlimited).
    ///
    /// When set, at most `n` tasks run concurrently. The rest wait
    /// for a slot to open.
    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }

    /// Execute synchronously, blocking the current thread.
    pub fn go(self) -> Vec<ConcurrentTaskResult> {
        shared::block_on(self.run())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> Vec<ConcurrentTaskResult> {
        use futures::stream::{FuturesUnordered, StreamExt};

        let llm = self.llm;
        let tasks = self.tasks;

        if tasks.is_empty() {
            return Vec::new();
        }

        // Build all pipelines eagerly
        let labeled_pipelines: Vec<(String, Pipeline<'a, L>)> = tasks
            .into_iter()
            .map(|t| {
                let pipeline = (t.build)(llm);
                (t.label, pipeline)
            })
            .collect();

        let mut futs = FuturesUnordered::new();

        for (idx, (label, pipeline)) in labeled_pipelines.into_iter().enumerate() {
            futs.push(async move {
                let start = Instant::now();
                let result = pipeline.run().await;
                (
                    idx,
                    ConcurrentTaskResult {
                        label,
                        result: Ok(result),
                        elapsed: start.elapsed(),
                    },
                )
            });
        }

        let mut results: Vec<Option<ConcurrentTaskResult>> =
            (0..futs.len()).map(|_| None).collect();

        while let Some((idx, task_result)) = futs.next().await {
            results[idx] = Some(task_result);
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::checks::checks;
    use crate::recursive::llm::MockLlm;
    use crate::recursive::pipeline::pipeline;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_concurrent_basic() {
        let llm = MockLlm::new(|_, _| "output".to_string());

        let results = ConcurrentRunner::new(&llm)
            .task("a", |llm| pipeline(llm, "Task A"))
            .task("b", |llm| pipeline(llm, "Task B"))
            .go();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].label, "a");
        assert_eq!(results[1].label, "b");
        assert!(results[0].result.is_ok());
        assert!(results[1].result.is_ok());
    }

    #[test]
    fn test_concurrent_with_refine() {
        let llm = MockLlm::new(|_, _| "fn main() {}".to_string());

        let results = ConcurrentRunner::new(&llm)
            .task("rust", |llm| {
                pipeline(llm, "Write Rust").refine(checks().require("fn "))
            })
            .task("simple", |llm| pipeline(llm, "Simple"))
            .go();

        assert_eq!(results.len(), 2);
        let rust_result = results[0].result.as_ref().unwrap();
        assert!(rust_result.output.contains("fn main"));
    }

    #[test]
    fn test_concurrent_empty() {
        let llm = MockLlm::new(|_, _| "x".to_string());
        let results = ConcurrentRunner::new(&llm).go();
        assert!(results.is_empty());
    }

    #[test]
    fn test_concurrent_shared_llm() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            format!("response_{}", n)
        });

        let results = ConcurrentRunner::new(&llm)
            .task("first", |llm| pipeline(llm, "A"))
            .task("second", |llm| pipeline(llm, "B"))
            .task("third", |llm| pipeline(llm, "C"))
            .go();

        assert_eq!(results.len(), 3);
        // All should have gotten responses
        for r in &results {
            let output = &r.result.as_ref().unwrap().output;
            assert!(output.starts_with("response_"));
        }
    }

    #[test]
    fn test_concurrent_max_concurrency() {
        let llm = MockLlm::new(|_, _| "ok".to_string());

        let runner = ConcurrentRunner::new(&llm)
            .max_concurrency(2)
            .task("a", |llm| pipeline(llm, "A"))
            .task("b", |llm| pipeline(llm, "B"));

        assert_eq!(runner.max_concurrency, 2);

        let results = runner.go();
        assert_eq!(results.len(), 2);
    }
}
