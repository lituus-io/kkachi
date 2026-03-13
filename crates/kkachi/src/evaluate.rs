// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Evaluation harness for testing LLM performance on datasets.
//!
//! This module provides the [`Evaluate`] builder for running structured evaluation
//! of an LLM against an [`ExampleSet`] using a [`Metric`]. It produces an
//! [`EvalResult`] containing per-example scores and aggregate statistics.
//!
//! # Architecture
//!
//! The evaluation harness uses kkachi's zero-copy principles:
//! - Input/output data is read from the shared [`Buffer`] via [`ExampleSet`]
//! - Field lookup uses interned [`Sym`] symbols (4 bytes each)
//! - Results are collected into a flat `Vec<ExampleResult>` for cache locality
//!
//! # Examples
//!
//! ```
//! use kkachi::evaluate::{Evaluate, EvalResult};
//! use kkachi::metric::ExactMatch;
//! use kkachi::recursive::{MockLlm, Llm};
//! use kkachi::optimizer::{ExampleSet, ExampleMeta};
//! use kkachi::predict::FieldRange;
//! use kkachi::buffer::Buffer;
//! use kkachi::intern::{sym, Sym};
//!
//! // Create a mock LLM that echoes the input
//! let llm = MockLlm::new(|prompt, _| "4".to_string());
//!
//! // Build a simple dataset
//! let data = b"What is 2+2?4";
//! let buffer = Buffer::from_bytes(data.to_vec());
//!
//! let input_sym = sym("question");
//! let output_sym = sym("answer");
//!
//! let meta = ExampleMeta {
//!     input_ranges: [
//!         (input_sym, FieldRange::new(0, 12)),
//!         (Sym::EMPTY, FieldRange::new(0, 0)),
//!         (Sym::EMPTY, FieldRange::new(0, 0)),
//!         (Sym::EMPTY, FieldRange::new(0, 0)),
//!     ],
//!     input_count: 1,
//!     output_ranges: [
//!         (output_sym, FieldRange::new(12, 13)),
//!         (Sym::EMPTY, FieldRange::new(0, 0)),
//!     ],
//!     output_count: 1,
//! };
//!
//! let examples = [meta];
//! let dataset = ExampleSet::new(&buffer, &examples);
//!
//! let result = Evaluate::new(&llm, ExactMatch)
//!     .output_field(output_sym)
//!     .run(&dataset);
//!
//! assert_eq!(result.total, 1);
//! assert_eq!(result.mean, 1.0);
//! ```

use crate::intern::Sym;
use crate::metric::Metric;
use crate::optimizer::ExampleSet;
use crate::recursive::llm::Llm;

/// Builder for running evaluations against a dataset.
///
/// # Type Parameters
///
/// - `'a`: Lifetime of the LLM reference and demo indices slice
/// - `L`: The LLM implementation
/// - `M`: The metric implementation
pub struct Evaluate<'a, L: Llm, M: Metric> {
    /// The LLM to evaluate.
    llm: &'a L,
    /// The metric for scoring predictions.
    metric: M,
    /// Instruction prepended to each prompt.
    instruction: Option<&'a str>,
    /// Which output field to use as the expected answer.
    output_field: Option<Sym>,
    /// Score threshold for pass/fail determination.
    threshold: f64,
    /// Demo indices to include as few-shot examples in the prompt.
    demo_indices: &'a [u32],
}

impl<'a, L: Llm, M: Metric> Evaluate<'a, L, M> {
    /// Create a new evaluation builder with the given LLM and metric.
    ///
    /// Default settings:
    /// - No instruction prefix
    /// - First output field is used
    /// - Threshold: 0.5
    /// - No demo examples
    pub fn new(llm: &'a L, metric: M) -> Self {
        Self {
            llm,
            metric,
            instruction: None,
            output_field: None,
            threshold: 0.5,
            demo_indices: &[],
        }
    }

    /// Set the instruction to prepend to each prompt.
    ///
    /// This is typically a task description, e.g., "Answer the following question concisely."
    pub fn instruction(mut self, instruction: &'a str) -> Self {
        self.instruction = Some(instruction);
        self
    }

    /// Set which output field symbol to use as the expected answer.
    ///
    /// If not set, the first output field of each example is used.
    pub fn output_field(mut self, sym: Sym) -> Self {
        self.output_field = Some(sym);
        self
    }

    /// Set the score threshold for pass/fail determination.
    ///
    /// Examples scoring at or above this threshold are considered passing.
    /// Default is 0.5.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set demo indices for few-shot examples in the prompt.
    ///
    /// These indices refer to examples in the dataset that will be formatted
    /// as demonstrations in the prompt context.
    pub fn demos(mut self, indices: &'a [u32]) -> Self {
        self.demo_indices = indices;
        self
    }

    /// Run the evaluation synchronously.
    pub fn run(self, dataset: &ExampleSet<'_>) -> EvalResult {
        crate::recursive::shared::block_on(self.run_async(dataset))
    }

    /// Run the evaluation asynchronously.
    pub async fn run_async(self, dataset: &ExampleSet<'_>) -> EvalResult {
        let mut per_example = Vec::with_capacity(dataset.len());
        let mut total_tokens = 0u64;

        // Build few-shot context from demo indices
        let context = self.build_demo_context(dataset);

        for (idx, view) in dataset.iter().enumerate() {
            // Skip examples that are used as demos
            if self.demo_indices.contains(&(idx as u32)) {
                continue;
            }

            // Get the first input field's text for the prompt
            let input_text: String = view
                .inputs()
                .map(|(sym, val)| format!("{}: {}", sym.as_str(), val))
                .collect::<Vec<_>>()
                .join("\n");

            // Get expected output
            let expected = if let Some(out_sym) = self.output_field {
                view.get_output(out_sym).unwrap_or("").to_string()
            } else {
                // Use first output field
                view.outputs()
                    .next()
                    .map(|(_, val)| val.to_string())
                    .unwrap_or_default()
            };

            // Build prompt
            let prompt = self.build_prompt(&input_text);

            // Call the LLM
            let prediction = match self.llm.generate(&prompt, &context, None).await {
                Ok(output) => {
                    total_tokens += output.total_tokens() as u64;
                    output.text.to_string()
                }
                Err(_) => String::new(),
            };

            // Score with the metric
            let score = self.metric.evaluate(&prediction, &expected);
            let passed = score >= self.threshold;

            per_example.push(ExampleResult {
                index: idx,
                prediction,
                expected,
                score,
                passed,
            });
        }

        // Compute aggregate statistics
        EvalResult::from_examples(per_example, total_tokens)
    }

    /// Build the prompt for a single example.
    fn build_prompt(&self, input_text: &str) -> String {
        match self.instruction {
            Some(inst) => format!("{}\n\n{}", inst, input_text),
            None => input_text.to_string(),
        }
    }

    /// Build few-shot context from demo indices.
    fn build_demo_context(&self, dataset: &ExampleSet<'_>) -> String {
        if self.demo_indices.is_empty() {
            return String::new();
        }

        let mut context = String::new();
        for &idx in self.demo_indices {
            let idx = idx as usize;
            if let Some(view) = dataset.iter().nth(idx) {
                // Format input fields
                for (sym, val) in view.inputs() {
                    context.push_str(&format!("{}: {}\n", sym.as_str(), val));
                }
                // Format output fields
                for (sym, val) in view.outputs() {
                    context.push_str(&format!("{}: {}\n", sym.as_str(), val));
                }
                context.push('\n');
            }
        }
        context
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Aggregate result of an evaluation run.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Mean score across all evaluated examples.
    pub mean: f64,
    /// Median score.
    pub median: f64,
    /// Standard deviation of scores.
    pub std_dev: f64,
    /// Number of examples that passed the threshold.
    pub pass_count: usize,
    /// Total number of examples evaluated.
    pub total: usize,
    /// Per-example results.
    pub per_example: Vec<ExampleResult>,
    /// Total tokens consumed during evaluation.
    pub total_tokens: u64,
}

impl EvalResult {
    /// Construct an `EvalResult` from a list of per-example results.
    fn from_examples(per_example: Vec<ExampleResult>, total_tokens: u64) -> Self {
        let total = per_example.len();

        if total == 0 {
            return Self {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                pass_count: 0,
                total: 0,
                per_example,
                total_tokens,
            };
        }

        let scores: Vec<f64> = per_example.iter().map(|e| e.score).collect();
        let mean = scores.iter().sum::<f64>() / total as f64;
        let median = Self::compute_median(&scores);
        let std_dev = Self::compute_std_dev(&scores, mean);
        let pass_count = per_example.iter().filter(|e| e.passed).count();

        Self {
            mean,
            median,
            std_dev,
            pass_count,
            total,
            per_example,
            total_tokens,
        }
    }

    /// Re-score all examples with a different metric.
    ///
    /// This is useful for comparing multiple metrics on the same set of
    /// predictions without re-running the LLM. The predictions and expected
    /// values are preserved; only scores and pass/fail flags are recomputed.
    ///
    /// The threshold from the original evaluation is not preserved; a default
    /// of 0.5 is used. To use a different threshold, modify the returned
    /// result's `per_example` entries directly.
    pub fn rescore<M2: Metric>(&self, metric: &M2) -> EvalResult {
        self.rescore_with_threshold(metric, 0.5)
    }

    /// Re-score all examples with a different metric and custom threshold.
    pub fn rescore_with_threshold<M2: Metric>(&self, metric: &M2, threshold: f64) -> EvalResult {
        let per_example: Vec<ExampleResult> = self
            .per_example
            .iter()
            .map(|ex| {
                let score = metric.evaluate(&ex.prediction, &ex.expected);
                let passed = score >= threshold;
                ExampleResult {
                    index: ex.index,
                    prediction: ex.prediction.clone(),
                    expected: ex.expected.clone(),
                    score,
                    passed,
                }
            })
            .collect();

        EvalResult::from_examples(per_example, self.total_tokens)
    }

    /// Get the pass rate as a fraction in [0.0, 1.0].
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.pass_count as f64 / self.total as f64
        }
    }

    /// Compute median of a slice of f64 values.
    fn compute_median(scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Compute standard deviation given a mean.
    fn compute_std_dev(scores: &[f64], mean: f64) -> f64 {
        if scores.len() <= 1 {
            return 0.0;
        }
        let variance =
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Result for a single evaluated example.
#[derive(Debug, Clone)]
pub struct ExampleResult {
    /// Index of this example in the dataset.
    pub index: usize,
    /// The LLM's prediction.
    pub prediction: String,
    /// The expected output.
    pub expected: String,
    /// Metric score for this example.
    pub score: f64,
    /// Whether this example passed the threshold.
    pub passed: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::intern::sym;
    use crate::metric::{Contains, ExactMatch, F1Token, FnMetric};
    use crate::optimizer::ExampleMeta;
    use crate::predict::FieldRange;
    use crate::recursive::llm::MockLlm;

    /// Helper to build a simple dataset from (input, expected) pairs.
    fn build_dataset(pairs: &[(&str, &str)]) -> (Buffer, Vec<ExampleMeta>, Sym, Sym) {
        let input_sym = sym("question");
        let output_sym = sym("answer");

        let mut buf = Vec::new();
        let mut metas = Vec::new();

        for (input, expected) in pairs {
            let input_start = buf.len() as u32;
            buf.extend_from_slice(input.as_bytes());
            let input_end = buf.len() as u32;

            let output_start = buf.len() as u32;
            buf.extend_from_slice(expected.as_bytes());
            let output_end = buf.len() as u32;

            let meta = ExampleMeta {
                input_ranges: [
                    (input_sym, FieldRange::new(input_start, input_end)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                ],
                input_count: 1,
                output_ranges: [
                    (output_sym, FieldRange::new(output_start, output_end)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                ],
                output_count: 1,
            };
            metas.push(meta);
        }

        let buffer = Buffer::from_bytes(buf);
        (buffer, metas, input_sym, output_sym)
    }

    #[test]
    fn test_evaluate_exact_match_all_correct() {
        let llm = MockLlm::new(|prompt, _| {
            // Extract the expected answer from the prompt
            if prompt.contains("2+2") {
                "4".to_string()
            } else if prompt.contains("3+3") {
                "6".to_string()
            } else {
                "unknown".to_string()
            }
        });

        let (buffer, metas, _input_sym, output_sym) =
            build_dataset(&[("What is 2+2?", "4"), ("What is 3+3?", "6")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .run(&dataset);

        assert_eq!(result.total, 2);
        assert_eq!(result.pass_count, 2);
        assert_eq!(result.mean, 1.0);
        assert_eq!(result.median, 1.0);
        assert_eq!(result.std_dev, 0.0);
    }

    #[test]
    fn test_evaluate_partial_match() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("2+2") {
                "4".to_string()
            } else {
                "wrong".to_string()
            }
        });

        let (buffer, metas, _input_sym, output_sym) =
            build_dataset(&[("What is 2+2?", "4"), ("What is 3+3?", "6")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .run(&dataset);

        assert_eq!(result.total, 2);
        assert_eq!(result.pass_count, 1);
        assert_eq!(result.mean, 0.5);
    }

    #[test]
    fn test_evaluate_with_instruction() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.starts_with("Be concise.") {
                "4".to_string()
            } else {
                "The answer is four".to_string()
            }
        });

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[("What is 2+2?", "4")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, ExactMatch)
            .instruction("Be concise.")
            .output_field(output_sym)
            .run(&dataset);

        assert_eq!(result.total, 1);
        assert_eq!(result.mean, 1.0);
    }

    #[test]
    fn test_evaluate_with_threshold() {
        let llm = MockLlm::new(|_, _| "the quick brown fox".to_string());

        let (buffer, metas, _input_sym, output_sym) =
            build_dataset(&[("Repeat", "the quick brown fox jumps")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        // F1 will give partial credit; set a low threshold to pass
        let result = Evaluate::new(&llm, F1Token)
            .output_field(output_sym)
            .threshold(0.5)
            .run(&dataset);

        assert_eq!(result.total, 1);
        assert!(result.mean > 0.5);
        assert_eq!(result.pass_count, 1);
    }

    #[test]
    fn test_evaluate_with_high_threshold() {
        let llm = MockLlm::new(|_, _| "the quick brown fox".to_string());

        let (buffer, metas, _input_sym, output_sym) =
            build_dataset(&[("Repeat", "the quick brown fox jumps over")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        // High threshold that partial match won't reach
        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .threshold(1.0)
            .run(&dataset);

        assert_eq!(result.total, 1);
        assert_eq!(result.pass_count, 0);
    }

    #[test]
    fn test_evaluate_empty_dataset() {
        let llm = MockLlm::new(|_, _| String::new());

        static BUFFER: Buffer = Buffer::Static(b"");
        let dataset = ExampleSet::new(&BUFFER, &[]);

        let result = Evaluate::new(&llm, ExactMatch).run(&dataset);

        assert_eq!(result.total, 0);
        assert_eq!(result.mean, 0.0);
        assert_eq!(result.median, 0.0);
        assert_eq!(result.std_dev, 0.0);
        assert_eq!(result.pass_count, 0);
    }

    #[test]
    fn test_evaluate_contains_metric() {
        let llm = MockLlm::new(|_, _| "The answer is 42, obviously.".to_string());

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[("What is it?", "42")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, Contains)
            .output_field(output_sym)
            .run(&dataset);

        assert_eq!(result.total, 1);
        assert_eq!(result.mean, 1.0);
        assert_eq!(result.pass_count, 1);
    }

    #[test]
    fn test_evaluate_rescore() {
        let llm = MockLlm::new(|_, _| "The answer is 42".to_string());

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[("Question", "42")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        // First run with ExactMatch (should fail since "The answer is 42" != "42")
        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .run(&dataset);

        assert_eq!(result.mean, 0.0);
        assert_eq!(result.pass_count, 0);

        // Re-score with Contains (should pass since "The answer is 42" contains "42")
        let rescored = result.rescore(&Contains);
        assert_eq!(rescored.mean, 1.0);
        assert_eq!(rescored.pass_count, 1);
        assert_eq!(rescored.total, 1);

        // Predictions should be preserved
        assert_eq!(rescored.per_example[0].prediction, "The answer is 42");
        assert_eq!(rescored.per_example[0].expected, "42");
    }

    #[test]
    fn test_evaluate_rescore_with_threshold() {
        let llm = MockLlm::new(|_, _| "partial match here".to_string());

        let (buffer, metas, _input_sym, output_sym) =
            build_dataset(&[("Q", "partial match here and more")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, F1Token)
            .output_field(output_sym)
            .run(&dataset);

        // Re-score with a strict threshold
        let strict = result.rescore_with_threshold(&F1Token, 0.99);
        assert_eq!(strict.pass_count, 0);

        // Re-score with a lenient threshold
        let lenient = result.rescore_with_threshold(&F1Token, 0.1);
        assert_eq!(lenient.pass_count, 1);
    }

    #[test]
    fn test_eval_result_pass_rate() {
        let per_example = vec![
            ExampleResult {
                index: 0,
                prediction: "a".to_string(),
                expected: "a".to_string(),
                score: 1.0,
                passed: true,
            },
            ExampleResult {
                index: 1,
                prediction: "b".to_string(),
                expected: "c".to_string(),
                score: 0.0,
                passed: false,
            },
            ExampleResult {
                index: 2,
                prediction: "d".to_string(),
                expected: "d".to_string(),
                score: 1.0,
                passed: true,
            },
        ];

        let result = EvalResult::from_examples(per_example, 0);
        assert!((result.pass_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_result_pass_rate_empty() {
        let result = EvalResult::from_examples(Vec::new(), 0);
        assert_eq!(result.pass_rate(), 0.0);
    }

    #[test]
    fn test_eval_result_statistics() {
        // Scores: 0.0, 0.5, 1.0
        // Mean = 0.5, Median = 0.5
        // Variance = ((0.25 + 0.0 + 0.25) / 2) = 0.25, StdDev = 0.5
        let per_example = vec![
            ExampleResult {
                index: 0,
                prediction: String::new(),
                expected: String::new(),
                score: 0.0,
                passed: false,
            },
            ExampleResult {
                index: 1,
                prediction: String::new(),
                expected: String::new(),
                score: 0.5,
                passed: true,
            },
            ExampleResult {
                index: 2,
                prediction: String::new(),
                expected: String::new(),
                score: 1.0,
                passed: true,
            },
        ];

        let result = EvalResult::from_examples(per_example, 100);
        assert!((result.mean - 0.5).abs() < 1e-9);
        assert!((result.median - 0.5).abs() < 1e-9);
        assert!((result.std_dev - 0.5).abs() < 1e-9);
        assert_eq!(result.pass_count, 2);
        assert_eq!(result.total, 3);
        assert_eq!(result.total_tokens, 100);
    }

    #[test]
    fn test_eval_result_median_even_count() {
        // Scores: 0.2, 0.4, 0.6, 0.8
        // Median = (0.4 + 0.6) / 2 = 0.5
        let per_example: Vec<ExampleResult> = [0.2, 0.4, 0.6, 0.8]
            .iter()
            .enumerate()
            .map(|(i, &score)| ExampleResult {
                index: i,
                prediction: String::new(),
                expected: String::new(),
                score,
                passed: score >= 0.5,
            })
            .collect();

        let result = EvalResult::from_examples(per_example, 0);
        assert!((result.median - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_with_fn_metric() {
        let llm = MockLlm::new(|_, _| "abc".to_string());

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[("Q", "abcd")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let length_metric = FnMetric::new("length_ratio", |pred, expected| {
            pred.len() as f64 / expected.len().max(1) as f64
        });

        let result = Evaluate::new(&llm, length_metric)
            .output_field(output_sym)
            .run(&dataset);

        assert_eq!(result.total, 1);
        assert!((result.mean - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_with_demos() {
        let llm = MockLlm::new(|_prompt, _feedback| {
            // MockLlm receives (prompt, feedback), not context directly.
            // The demo context is passed via the LLM's `context` parameter,
            // not the `feedback` parameter. Since MockLlm ignores context,
            // we just always return the correct answer for this test.
            "4".to_string()
        });

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[
            ("What is 1+1?", "2"), // index 0 - will be demo
            ("What is 2+2?", "4"), // index 1 - will be evaluated
        ]);

        let dataset = ExampleSet::new(&buffer, &metas);
        let demo_indices = [0u32];

        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .demos(&demo_indices)
            .run(&dataset);

        // Only index 1 should be evaluated (index 0 is a demo)
        assert_eq!(result.total, 1);
        assert_eq!(result.mean, 1.0);
        assert_eq!(result.per_example[0].index, 1);
    }

    #[test]
    fn test_evaluate_preserves_tokens() {
        // MockLlm returns LmOutput with 0 tokens by default
        let llm = MockLlm::new(|_, _| "answer".to_string());

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[("Q1", "A1"), ("Q2", "A2")]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .run(&dataset);

        // MockLlm reports 0 tokens per call
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_evaluate_multiple_examples_statistics() {
        // LLM always returns "hello"
        let llm = MockLlm::new(|_, _| "hello".to_string());

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[
            ("Q1", "hello"),   // exact match -> 1.0
            ("Q2", "world"),   // no match -> 0.0
            ("Q3", "hello"),   // exact match -> 1.0
            ("Q4", "goodbye"), // no match -> 0.0
            ("Q5", "hello"),   // exact match -> 1.0
        ]);

        let dataset = ExampleSet::new(&buffer, &metas);

        let result = Evaluate::new(&llm, ExactMatch)
            .output_field(output_sym)
            .threshold(0.5)
            .run(&dataset);

        assert_eq!(result.total, 5);
        assert_eq!(result.pass_count, 3);
        assert!((result.mean - 0.6).abs() < 1e-9);
        assert!((result.median - 1.0).abs() < 1e-9);
        assert!((result.pass_rate() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_example_result_fields() {
        let ex = ExampleResult {
            index: 42,
            prediction: "predicted".to_string(),
            expected: "expected".to_string(),
            score: 0.75,
            passed: true,
        };

        assert_eq!(ex.index, 42);
        assert_eq!(ex.prediction, "predicted");
        assert_eq!(ex.expected, "expected");
        assert!((ex.score - 0.75).abs() < f64::EPSILON);
        assert!(ex.passed);
    }

    #[test]
    fn test_eval_result_single_example() {
        let per_example = vec![ExampleResult {
            index: 0,
            prediction: "a".to_string(),
            expected: "a".to_string(),
            score: 1.0,
            passed: true,
        }];

        let result = EvalResult::from_examples(per_example, 50);
        assert_eq!(result.mean, 1.0);
        assert_eq!(result.median, 1.0);
        assert_eq!(result.std_dev, 0.0); // single element -> 0 std_dev
        assert_eq!(result.pass_count, 1);
        assert_eq!(result.total, 1);
        assert_eq!(result.total_tokens, 50);
    }
}
