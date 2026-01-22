// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Evaluator for running metrics on datasets

use crate::metric::{Metric, MetricResult};
use kkachi::{Example, Module};
use std::sync::Arc;

/// Evaluation results
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Average score across all examples
    pub score: f64,

    /// Individual results
    pub results: Vec<(MetricResult, String)>, // (result, example_id)

    /// Number of passed examples
    pub passed: usize,

    /// Total examples
    pub total: usize,
}

impl EvaluationResult {
    /// Calculate accuracy (passed / total)
    pub fn accuracy(&self) -> f64 {
        self.passed as f64 / self.total as f64
    }
}

/// Evaluator for running metrics
pub struct Evaluator {
    metric: Arc<dyn Metric>,
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new(metric: Arc<dyn Metric>) -> Self {
        Self { metric }
    }

    /// Evaluate a module on a dataset
    pub async fn evaluate<'a, M: Module>(
        &self,
        module: &M,
        examples: &[Example<'a>],
    ) -> anyhow::Result<EvaluationResult> {
        let mut results = Vec::new();
        let mut total_score = 0.0;
        let mut passed = 0;

        for (idx, example) in examples.iter().enumerate() {
            let inputs = kkachi::types::Inputs::from_iter(
                example.inputs.iter().map(|(k, v)| (k.clone(), v.clone())),
            );

            let prediction = module.forward(inputs).await?;
            let metric_result = self.metric.evaluate(example, &prediction);

            total_score += metric_result.score;
            if metric_result.passed {
                passed += 1;
            }

            results.push((metric_result, format!("example_{}", idx)));
        }

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
