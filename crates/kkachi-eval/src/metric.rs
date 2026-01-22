// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Metric trait and implementations
//!
//! Includes high-performance metrics with Rayon parallelism and stop word filtering.

use kkachi::{Example, Prediction};
use rayon::prelude::*;
use std::collections::HashSet;

/// Result of a metric evaluation
#[derive(Debug, Clone)]
pub struct MetricResult {
    /// Score (typically 0.0 to 1.0 or boolean 0/1)
    pub score: f64,

    /// Whether the prediction passed
    pub passed: bool,

    /// Optional details
    pub details: Option<String>,
}

impl MetricResult {
    /// Create a new metric result
    pub fn new(score: f64) -> Self {
        Self {
            score,
            passed: score > 0.0,
            details: None,
        }
    }

    /// Create a passed result
    pub fn pass() -> Self {
        Self::new(1.0)
    }

    /// Create a failed result
    pub fn fail() -> Self {
        Self::new(0.0)
    }

    /// Add details
    pub fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }
}

/// Metric trait for evaluating predictions
pub trait Metric: Send + Sync {
    /// Evaluate a prediction against an example
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult;

    /// Get metric name
    fn name(&self) -> &str;
}

/// Exact match metric
pub struct ExactMatch;

impl Metric for ExactMatch {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        if let Some(expected_outputs) = &example.outputs {
            let all_match = expected_outputs.iter().all(|(key, expected_value)| {
                prediction
                    .get(key.as_ref())
                    .map(|pred_value| pred_value == expected_value.as_ref())
                    .unwrap_or(false)
            });

            if all_match {
                MetricResult::pass()
            } else {
                MetricResult::fail()
            }
        } else {
            MetricResult::fail().with_details("No expected outputs in example".to_string())
        }
    }

    fn name(&self) -> &str {
        "exact_match"
    }
}

/// F1 score metric (for token-level comparison)
pub struct F1Score;

impl F1Score {
    fn tokenize(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    fn calculate_f1(prediction_tokens: &[&str], gold_tokens: &[&str]) -> f64 {
        if prediction_tokens.is_empty() || gold_tokens.is_empty() {
            return 0.0;
        }

        let pred_set: std::collections::HashSet<_> = prediction_tokens.iter().collect();
        let gold_set: std::collections::HashSet<_> = gold_tokens.iter().collect();

        let intersection = pred_set.intersection(&gold_set).count();

        let precision = intersection as f64 / pred_set.len() as f64;
        let recall = intersection as f64 / gold_set.len() as f64;

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }
}

impl Metric for F1Score {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        if let Some(expected_outputs) = &example.outputs {
            let scores: Vec<f64> = expected_outputs
                .iter()
                .filter_map(|(key, expected_value)| {
                    prediction.get(key.as_ref()).map(|pred_value| {
                        let pred_tokens = Self::tokenize(pred_value);
                        let gold_tokens = Self::tokenize(expected_value.as_ref());
                        Self::calculate_f1(&pred_tokens, &gold_tokens)
                    })
                })
                .collect();

            let avg_score = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64
            };

            MetricResult::new(avg_score)
        } else {
            MetricResult::fail()
        }
    }

    fn name(&self) -> &str {
        "f1_score"
    }
}

/// Default English stop words for semantic comparison.
const DEFAULT_STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "here",
];

/// Semantic F1 metric with stop word filtering.
///
/// Computes F1 score while ignoring common stop words for more
/// meaningful semantic comparison.
pub struct SemanticF1 {
    /// Stop words to ignore (O(1) lookup via HashSet)
    stop_words: HashSet<&'static str>,
    /// Whether to normalize text (lowercase)
    normalize: bool,
}

impl Default for SemanticF1 {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticF1 {
    /// Create a new SemanticF1 metric with default stop words.
    pub fn new() -> Self {
        Self {
            stop_words: DEFAULT_STOP_WORDS.iter().copied().collect(),
            normalize: true,
        }
    }

    /// Create without stop words.
    pub fn without_stop_words() -> Self {
        Self {
            stop_words: HashSet::new(),
            normalize: true,
        }
    }

    /// Create with custom stop words.
    pub fn with_stop_words(stop_words: &'static [&'static str]) -> Self {
        Self {
            stop_words: stop_words.iter().copied().collect(),
            normalize: true,
        }
    }

    /// Set normalization.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Tokenize and filter stop words.
    fn tokenize<'a>(&self, s: &'a str) -> Vec<String> {
        s.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|t| !t.is_empty())
            .map(|t| {
                if self.normalize {
                    t.to_lowercase()
                } else {
                    t.to_string()
                }
            })
            .filter(|t| !self.stop_words.contains(t.as_str()))
            .collect()
    }

    /// Calculate F1 score between token sets.
    fn calculate_f1(&self, pred_tokens: &[String], gold_tokens: &[String]) -> f64 {
        if pred_tokens.is_empty() && gold_tokens.is_empty() {
            return 1.0; // Both empty = perfect match
        }
        if pred_tokens.is_empty() || gold_tokens.is_empty() {
            return 0.0;
        }

        let pred_set: HashSet<_> = pred_tokens.iter().collect();
        let gold_set: HashSet<_> = gold_tokens.iter().collect();

        let intersection = pred_set.intersection(&gold_set).count();

        if intersection == 0 {
            return 0.0;
        }

        let precision = intersection as f64 / pred_set.len() as f64;
        let recall = intersection as f64 / gold_set.len() as f64;

        2.0 * (precision * recall) / (precision + recall)
    }

    /// Evaluate a batch in parallel using Rayon.
    pub fn evaluate_batch<'a>(
        &self,
        examples: &[Example<'a>],
        predictions: &[Prediction<'a>],
    ) -> Vec<MetricResult> {
        examples
            .par_iter()
            .zip(predictions.par_iter())
            .map(|(example, prediction)| self.evaluate(example, prediction))
            .collect()
    }
}

impl Metric for SemanticF1 {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        if let Some(expected_outputs) = &example.outputs {
            let scores: Vec<f64> = expected_outputs
                .iter()
                .filter_map(|(key, expected_value)| {
                    prediction.get(key.as_ref()).map(|pred_value| {
                        let pred_tokens = self.tokenize(pred_value);
                        let gold_tokens = self.tokenize(expected_value.as_ref());
                        self.calculate_f1(&pred_tokens, &gold_tokens)
                    })
                })
                .collect();

            let avg_score = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64
            };

            MetricResult::new(avg_score)
        } else {
            MetricResult::fail()
        }
    }

    fn name(&self) -> &str {
        "semantic_f1"
    }
}

/// Complete and Grounded metric.
///
/// Checks that:
/// 1. All expected fields are present in the prediction (completeness)
/// 2. The prediction doesn't contain made-up information (groundedness)
///
/// Groundedness is approximated by checking that prediction tokens
/// appear in the input context.
pub struct CompleteAndGrounded {
    /// Context field name to check groundedness against
    context_field: &'static str,
    /// Weight for completeness (0.0-1.0)
    completeness_weight: f64,
    /// Weight for groundedness (0.0-1.0)
    groundedness_weight: f64,
    /// Stop words to ignore
    stop_words: HashSet<&'static str>,
}

impl Default for CompleteAndGrounded {
    fn default() -> Self {
        Self::new()
    }
}

impl CompleteAndGrounded {
    /// Create a new CompleteAndGrounded metric.
    pub fn new() -> Self {
        Self {
            context_field: "context",
            completeness_weight: 0.5,
            groundedness_weight: 0.5,
            stop_words: DEFAULT_STOP_WORDS.iter().copied().collect(),
        }
    }

    /// Set the context field name.
    pub fn context_field(mut self, field: &'static str) -> Self {
        self.context_field = field;
        self
    }

    /// Set weights for completeness and groundedness.
    pub fn weights(mut self, completeness: f64, groundedness: f64) -> Self {
        let total = completeness + groundedness;
        self.completeness_weight = completeness / total;
        self.groundedness_weight = groundedness / total;
        self
    }

    /// Tokenize text, filtering stop words.
    fn tokenize(&self, s: &str) -> HashSet<String> {
        s.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .filter(|t| !self.stop_words.contains(t.as_str()))
            .collect()
    }

    /// Calculate completeness score.
    fn completeness_score(&self, example: &Example<'_>, prediction: &Prediction<'_>) -> f64 {
        if let Some(expected_outputs) = &example.outputs {
            if expected_outputs.is_empty() {
                return 1.0;
            }

            let present = expected_outputs
                .keys()
                .filter(|k| prediction.get(k.as_ref()).is_some())
                .count();

            present as f64 / expected_outputs.len() as f64
        } else {
            1.0 // No expected outputs means completeness is satisfied
        }
    }

    /// Calculate groundedness score.
    fn groundedness_score(&self, example: &Example<'_>, prediction: &Prediction<'_>) -> f64 {
        // Get context from inputs
        let context = example
            .inputs
            .get(self.context_field)
            .map(|v| v.as_ref())
            .unwrap_or("");

        if context.is_empty() {
            return 1.0; // No context means we can't verify groundedness
        }

        let context_tokens = self.tokenize(context);

        // Get all prediction values
        let pred_tokens: HashSet<String> = prediction
            .outputs
            .iter()
            .flat_map(|(_, v)| self.tokenize(v.as_ref()))
            .collect();

        if pred_tokens.is_empty() {
            return 0.0;
        }

        // Calculate what fraction of prediction tokens are grounded in context
        let grounded = pred_tokens
            .iter()
            .filter(|t| context_tokens.contains(*t))
            .count();

        grounded as f64 / pred_tokens.len() as f64
    }

    /// Evaluate a batch in parallel using Rayon.
    pub fn evaluate_batch<'a>(
        &self,
        examples: &[Example<'a>],
        predictions: &[Prediction<'a>],
    ) -> Vec<MetricResult> {
        examples
            .par_iter()
            .zip(predictions.par_iter())
            .map(|(example, prediction)| self.evaluate(example, prediction))
            .collect()
    }
}

impl Metric for CompleteAndGrounded {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        let completeness = self.completeness_score(example, prediction);
        let groundedness = self.groundedness_score(example, prediction);

        let score =
            self.completeness_weight * completeness + self.groundedness_weight * groundedness;

        MetricResult::new(score).with_details(format!(
            "completeness={:.2}, groundedness={:.2}",
            completeness, groundedness
        ))
    }

    fn name(&self) -> &str {
        "complete_and_grounded"
    }
}

/// Answer relevancy metric.
///
/// Measures how relevant the answer is to the question by
/// checking token overlap between the question and answer.
pub struct AnswerRelevancy {
    /// Question field name
    question_field: &'static str,
    /// Answer field name
    answer_field: &'static str,
    /// Stop words
    stop_words: HashSet<&'static str>,
}

impl Default for AnswerRelevancy {
    fn default() -> Self {
        Self::new()
    }
}

impl AnswerRelevancy {
    /// Create a new AnswerRelevancy metric.
    pub fn new() -> Self {
        Self {
            question_field: "question",
            answer_field: "answer",
            stop_words: DEFAULT_STOP_WORDS.iter().copied().collect(),
        }
    }

    /// Set field names.
    pub fn fields(mut self, question: &'static str, answer: &'static str) -> Self {
        self.question_field = question;
        self.answer_field = answer;
        self
    }

    /// Tokenize text.
    fn tokenize(&self, s: &str) -> HashSet<String> {
        s.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .filter(|t| !self.stop_words.contains(t.as_str()))
            .collect()
    }

    /// Evaluate a batch in parallel.
    pub fn evaluate_batch<'a>(
        &self,
        examples: &[Example<'a>],
        predictions: &[Prediction<'a>],
    ) -> Vec<MetricResult> {
        examples
            .par_iter()
            .zip(predictions.par_iter())
            .map(|(example, prediction)| self.evaluate(example, prediction))
            .collect()
    }
}

impl Metric for AnswerRelevancy {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        let question = example
            .inputs
            .get(self.question_field)
            .map(|v| v.as_ref())
            .unwrap_or("");

        let answer = prediction.get(self.answer_field).unwrap_or("");

        if question.is_empty() || answer.is_empty() {
            return MetricResult::fail();
        }

        let question_tokens = self.tokenize(question);
        let answer_tokens = self.tokenize(answer);

        if question_tokens.is_empty() || answer_tokens.is_empty() {
            return MetricResult::fail();
        }

        // Check what fraction of question tokens appear in the answer
        let overlap = question_tokens
            .iter()
            .filter(|t| answer_tokens.contains(*t))
            .count();

        let relevancy = overlap as f64 / question_tokens.len() as f64;

        MetricResult::new(relevancy)
    }

    fn name(&self) -> &str {
        "answer_relevancy"
    }
}

/// Batch metric evaluator that runs all metrics in parallel.
pub struct BatchMetricEvaluator {
    metrics: Vec<Box<dyn Metric>>,
}

impl BatchMetricEvaluator {
    /// Create a new batch evaluator.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    /// Add a metric.
    pub fn add<M: Metric + 'static>(mut self, metric: M) -> Self {
        self.metrics.push(Box::new(metric));
        self
    }

    /// Evaluate all metrics on a batch, returning results per example.
    pub fn evaluate_batch<'a>(
        &self,
        examples: &[Example<'a>],
        predictions: &[Prediction<'a>],
    ) -> Vec<Vec<(String, MetricResult)>> {
        examples
            .par_iter()
            .zip(predictions.par_iter())
            .map(|(example, prediction)| {
                self.metrics
                    .iter()
                    .map(|m| (m.name().to_string(), m.evaluate(example, prediction)))
                    .collect()
            })
            .collect()
    }

    /// Get aggregate scores per metric.
    pub fn aggregate_scores<'a>(
        &self,
        examples: &[Example<'a>],
        predictions: &[Prediction<'a>],
    ) -> Vec<(String, f64)> {
        if examples.is_empty() {
            return self
                .metrics
                .iter()
                .map(|m| (m.name().to_string(), 0.0))
                .collect();
        }

        self.metrics
            .iter()
            .map(|metric| {
                let total: f64 = examples
                    .par_iter()
                    .zip(predictions.par_iter())
                    .map(|(ex, pred)| metric.evaluate(ex, pred).score)
                    .sum();

                (metric.name().to_string(), total / examples.len() as f64)
            })
            .collect()
    }
}

impl Default for BatchMetricEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kkachi::Prediction;
    use std::borrow::Cow;

    #[test]
    fn test_exact_match() {
        let mut example = Example::new();
        example.insert_output("answer", "42");

        let mut prediction = Prediction::new();
        prediction.insert("answer", Cow::Borrowed("42"));

        let metric = ExactMatch;
        let result = metric.evaluate(&example, &prediction);
        assert!(result.passed);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_f1_score() {
        let pred_tokens = vec!["the", "cat", "sat"];
        let gold_tokens = vec!["the", "dog", "sat"];
        let score = F1Score::calculate_f1(&pred_tokens, &gold_tokens);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_semantic_f1() {
        let mut example = Example::new();
        example.insert_output("answer", "The quick brown fox jumps");

        let mut prediction = Prediction::new();
        prediction.insert("answer", Cow::Borrowed("The fast brown fox leaps"));

        let metric = SemanticF1::new();
        let result = metric.evaluate(&example, &prediction);

        // Should have partial match (brown, fox) with stop words filtered
        assert!(result.score > 0.0);
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_semantic_f1_perfect_match() {
        let mut example = Example::new();
        example.insert_output("answer", "The quick brown fox");

        let mut prediction = Prediction::new();
        prediction.insert("answer", Cow::Borrowed("The quick brown fox"));

        let metric = SemanticF1::new();
        let result = metric.evaluate(&example, &prediction);

        // Perfect match (after stop words removed: quick, brown, fox)
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_semantic_f1_batch() {
        let examples: Vec<Example> = (0..100)
            .map(|i| {
                let mut ex = Example::new();
                ex.insert_output("answer", format!("answer {}", i));
                ex
            })
            .collect();

        let predictions: Vec<Prediction> = (0..100)
            .map(|i| {
                let mut pred = Prediction::new();
                pred.insert("answer", Cow::Owned(format!("answer {}", i)));
                pred
            })
            .collect();

        let metric = SemanticF1::new();
        let results = metric.evaluate_batch(&examples, &predictions);

        assert_eq!(results.len(), 100);
        // All should be perfect matches
        assert!(results.iter().all(|r| r.score == 1.0));
    }

    #[test]
    fn test_complete_and_grounded() {
        let mut example = Example::new();
        example.insert_input(
            "context",
            "The capital of France is Paris. It is a beautiful city.",
        );
        example.insert_output("answer", "Paris");

        let mut prediction = Prediction::new();
        prediction.insert("answer", Cow::Borrowed("Paris"));

        let metric = CompleteAndGrounded::new();
        let result = metric.evaluate(&example, &prediction);

        // Should have high score (complete and grounded)
        assert!(result.score > 0.8);
    }

    #[test]
    fn test_complete_and_grounded_ungrounded() {
        let mut example = Example::new();
        example.insert_input("context", "The capital of France is Paris.");
        example.insert_output("answer", "expected");

        let mut prediction = Prediction::new();
        prediction.insert("answer", Cow::Borrowed("Berlin Germany Munich")); // Ungrounded

        let metric = CompleteAndGrounded::new();
        let result = metric.evaluate(&example, &prediction);

        // Groundedness should be low
        assert!(result.score < 0.8);
    }

    #[test]
    fn test_answer_relevancy() {
        let mut example = Example::new();
        example.insert_input("question", "What is the capital of France?");

        let mut prediction = Prediction::new();
        prediction.insert("answer", Cow::Borrowed("The capital of France is Paris."));

        let metric = AnswerRelevancy::new();
        let result = metric.evaluate(&example, &prediction);

        // Should have good relevancy (capital, France appear in both)
        assert!(result.score > 0.3);
    }

    #[test]
    fn test_batch_metric_evaluator() {
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

        let evaluator = BatchMetricEvaluator::new()
            .add(ExactMatch)
            .add(SemanticF1::new());

        let results = evaluator.evaluate_batch(&examples, &predictions);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].len(), 2); // Two metrics

        let aggregates = evaluator.aggregate_scores(&examples, &predictions);
        assert_eq!(aggregates.len(), 2);
        assert_eq!(aggregates[0].1, 1.0); // ExactMatch should be perfect
    }
}
