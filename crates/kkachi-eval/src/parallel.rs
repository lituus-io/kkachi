// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Parallel evaluation using Rayon

use crate::evaluator::EvaluationResult;
use crate::metric::Metric;
use kkachi::{Example, Prediction};
use rayon::prelude::*;
use std::sync::Arc;

/// Parallel evaluator using Rayon for CPU-bound tasks
pub struct ParallelEvaluator {
    metric: Arc<dyn Metric>,
    num_threads: Option<usize>,
}

impl ParallelEvaluator {
    /// Create a new parallel evaluator
    pub fn new(metric: Arc<dyn Metric>) -> Self {
        Self {
            metric,
            num_threads: None,
        }
    }

    /// Set number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Evaluate predictions in parallel (predictions already computed)
    pub fn evaluate_predictions<'a>(
        &self,
        examples: &[Example<'a>],
        predictions: &[Prediction<'a>],
    ) -> anyhow::Result<EvaluationResult> {
        if examples.len() != predictions.len() {
            anyhow::bail!("Examples and predictions length mismatch");
        }

        let pool = if let Some(threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?
        } else {
            rayon::ThreadPoolBuilder::new().build()?
        };

        let (results, total_score, passed): (Vec<_>, f64, usize) = pool.install(|| {
            examples
                .par_iter()
                .zip(predictions.par_iter())
                .enumerate()
                .map(|(idx, (example, prediction))| {
                    let metric_result = self.metric.evaluate(example, prediction);
                    let score = metric_result.score;
                    let is_passed = metric_result.passed;
                    let result = (metric_result, format!("example_{}", idx));
                    (result, score, if is_passed { 1 } else { 0 })
                })
                .fold(
                    || (Vec::new(), 0.0, 0),
                    |(mut results, score_sum, pass_sum), (result, score, passed)| {
                        results.push(result);
                        (results, score_sum + score, pass_sum + passed)
                    },
                )
                .reduce(
                    || (Vec::new(), 0.0, 0),
                    |(mut r1, s1, p1), (r2, s2, p2)| {
                        r1.extend(r2);
                        (r1, s1 + s2, p1 + p2)
                    },
                )
        });

        let score = if examples.is_empty() {
            0.0
        } else {
            total_score / examples.len() as f64
        };

        Ok(EvaluationResult {
            score,
            results,
            passed,
            total: examples.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::ExactMatch;
    use kkachi::{Example, Prediction};
    use std::borrow::Cow;

    #[test]
    fn test_parallel_evaluation() {
        let examples: Vec<Example> = (0..10)
            .map(|i| {
                let mut ex = Example::new();
                ex.insert_output("answer", i.to_string());
                ex
            })
            .collect();

        let predictions: Vec<Prediction> = (0..10)
            .map(|i| {
                let mut pred = Prediction::new();
                pred.insert("answer", Cow::Owned(i.to_string()));
                pred
            })
            .collect();

        let evaluator = ParallelEvaluator::new(Arc::new(ExactMatch));
        let result = evaluator
            .evaluate_predictions(&examples, &predictions)
            .unwrap();

        assert_eq!(result.passed, 10);
        assert_eq!(result.score, 1.0);
    }
}
