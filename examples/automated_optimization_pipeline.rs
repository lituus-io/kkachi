// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Automated Prompt Optimization Pipeline
//!
//! This example demonstrates a complete feedback-driven optimization pipeline using Kkachi.
//! It shows how to:
//! 1. Collect user feedback on predictions
//! 2. Build training examples from feedback
//! 3. Optimize prompts using BootstrapFewShot
//! 4. Evaluate improvements
//! 5. Deploy optimized model
//!
//! Run with: cargo run --example automated_optimization_pipeline

use kkachi::*;
use kkachi::predict::{LMClient, LMResponse};
use kkachi::prediction::{TokenUsage, PredictionMetadata};
use kkachi_eval::{EvaluationResult, ParallelEvaluator};
use kkachi_eval::metric::{Metric, MetricResult};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;

// ============================================================================
// Mock LM Client (In production, use OpenAI/Anthropic/etc.)
// ============================================================================

struct MockLM {
    call_count: AtomicUsize,
    responses: HashMap<String, String>,
}

impl MockLM {
    fn new() -> Self {
        let mut responses = HashMap::new();

        // Simulate various responses for different prompts
        responses.insert("sentiment_positive".to_string(), "Sentiment: positive\nConfidence: 0.95".to_string());
        responses.insert("sentiment_negative".to_string(), "Sentiment: negative\nConfidence: 0.92".to_string());
        responses.insert("sentiment_neutral".to_string(), "Sentiment: neutral\nConfidence: 0.88".to_string());

        // Question answering
        responses.insert("qa_capital".to_string(), "Answer: Paris\nReasoning: Paris is the capital and largest city of France.".to_string());
        responses.insert("qa_math".to_string(), "Answer: 42\nReasoning: 6 multiplied by 7 equals 42.".to_string());

        Self {
            call_count: AtomicUsize::new(0),
            responses,
        }
    }

    fn get_response(&self, prompt: &str) -> String {
        // Simple heuristic to match prompts to responses
        if prompt.contains("happy") || prompt.contains("great") || prompt.contains("love") {
            self.responses.get("sentiment_positive").unwrap().clone()
        } else if prompt.contains("sad") || prompt.contains("terrible") || prompt.contains("hate") {
            self.responses.get("sentiment_negative").unwrap().clone()
        } else if prompt.contains("Paris") || prompt.contains("France") {
            self.responses.get("qa_capital").unwrap().clone()
        } else if prompt.contains("6 * 7") || prompt.contains("multiply") {
            self.responses.get("qa_math").unwrap().clone()
        } else {
            self.responses.get("sentiment_neutral").unwrap().clone()
        }
    }
}

#[async_trait]
impl LMClient for MockLM {
    async fn generate(&self, prompt: &str) -> kkachi::Result<LMResponse> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        println!("  [LM Call #{}] Generating response...", count + 1);

        let text = self.get_response(prompt);

        Ok(LMResponse {
            text,
            usage: Some(TokenUsage::new(50, 25)),
        })
    }
}

// ============================================================================
// Feedback Collection System
// ============================================================================

#[derive(Debug, Clone)]
struct UserFeedback {
    input: String,
    predicted_output: String,
    correct_output: String,
    is_correct: bool,
    timestamp: std::time::SystemTime,
}

struct FeedbackCollector {
    feedback: Vec<UserFeedback>,
}

impl FeedbackCollector {
    fn new() -> Self {
        Self {
            feedback: Vec::new(),
        }
    }

    fn add_feedback(&mut self, input: String, predicted: String, correct: String) {
        let is_correct = predicted.trim() == correct.trim();

        self.feedback.push(UserFeedback {
            input,
            predicted_output: predicted,
            correct_output: correct,
            is_correct,
            timestamp: std::time::SystemTime::now(),
        });
    }

    fn get_incorrect_examples(&self) -> Vec<UserFeedback> {
        self.feedback
            .iter()
            .filter(|f| !f.is_correct)
            .cloned()
            .collect()
    }

    fn accuracy(&self) -> f64 {
        if self.feedback.is_empty() {
            return 0.0;
        }

        let correct = self.feedback.iter().filter(|f| f.is_correct).count();
        correct as f64 / self.feedback.len() as f64
    }

    fn build_training_set(&self) -> Vec<Example<'static>> {
        self.feedback
            .iter()
            .map(|f| {
                let mut example = Example::new();
                example.insert_input("text", f.input.clone());
                example.insert_output("sentiment", f.correct_output.clone());
                example.into_owned()
            })
            .collect()
    }
}

// ============================================================================
// Custom Metric: Sentiment Accuracy
// ============================================================================

struct SentimentAccuracy;

impl Metric for SentimentAccuracy {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        let expected = example.get_output("sentiment");
        let actual = prediction.get("sentiment");

        let passed = match (expected, actual) {
            (Some(exp), Some(act)) => exp.trim().eq_ignore_ascii_case(act.trim()),
            _ => false,
        };

        MetricResult {
            score: if passed { 1.0 } else { 0.0 },
            passed,
            details: Some(format!(
                "Expected: {:?}, Got: {:?}",
                expected, actual
            )),
        }
    }

    fn name(&self) -> &str {
        "SentimentAccuracy"
    }
}

// ============================================================================
// Optimization Pipeline
// ============================================================================

struct OptimizationPipeline {
    signature: Signature<'static>,
    lm_client: Arc<dyn LMClient>,
    feedback_collector: FeedbackCollector,
    current_model: Option<Predict<'static>>,
    optimization_history: Vec<OptimizationRun>,
}

#[derive(Debug, Clone)]
struct OptimizationRun {
    iteration: usize,
    accuracy_before: f64,
    accuracy_after: f64,
    num_demos: usize,
    timestamp: std::time::SystemTime,
}

impl OptimizationPipeline {
    fn new(signature: Signature<'static>, lm_client: Arc<dyn LMClient>) -> Self {
        let predict = Predict::new(signature.clone()).with_lm(lm_client.clone());

        Self {
            signature,
            lm_client,
            feedback_collector: FeedbackCollector::new(),
            current_model: Some(predict),
            optimization_history: Vec::new(),
        }
    }

    async fn collect_feedback(&mut self, test_cases: Vec<(String, String)>) -> kkachi::Result<()> {
        println!("\n📝 Collecting feedback from {} test cases...", test_cases.len());

        let model = self.current_model.as_ref()
            .ok_or_else(|| kkachi::Error::module("No model initialized"))?;

        for (input_text, expected_output) in test_cases {
            let mut inputs = types::Inputs::new();
            inputs.insert("text", input_text.clone());

            let prediction = model.forward(inputs).await?;
            let predicted = prediction.get("sentiment")
                .unwrap_or("unknown")
                .to_string();

            println!("  Input: '{}' → Predicted: '{}', Expected: '{}'",
                input_text, predicted, expected_output);

            self.feedback_collector.add_feedback(
                input_text,
                predicted,
                expected_output,
            );
        }

        let accuracy = self.feedback_collector.accuracy();
        println!("\n  Current Accuracy: {:.1}%", accuracy * 100.0);

        Ok(())
    }

    async fn optimize_from_feedback(&mut self, iteration: usize) -> kkachi::Result<()> {
        println!("\n🔄 Optimization Iteration #{}", iteration);

        let accuracy_before = self.feedback_collector.accuracy();
        println!("  Accuracy before optimization: {:.1}%", accuracy_before * 100.0);

        // Get incorrect examples for training
        let incorrect = self.feedback_collector.get_incorrect_examples();
        println!("  Found {} incorrect predictions to learn from", incorrect.len());

        if incorrect.is_empty() {
            println!("  ✅ No incorrect predictions - model is perfect!");
            return Ok(());
        }

        // Build training set from all feedback
        let training_set = self.feedback_collector.build_training_set();
        println!("  Training set size: {} examples", training_set.len());

        // Create optimizer
        let config = OptimizerConfig::new()
            .with_max_iterations(3)
            .with_batch_size(training_set.len() as u16)
            .with_seed(42)
            .with_metric_threshold(0.8);

        let optimizer = BootstrapFewShot::new(config)
            .with_max_demos(std::cmp::min(5, training_set.len()));

        println!("  Running BootstrapFewShot optimizer...");

        // Optimize the model
        let current = self.current_model.take()
            .ok_or_else(|| kkachi::Error::module("No model to optimize"))?;

        let optimized = optimizer.optimize(current, &training_set).await?;

        // Evaluate on training set
        let mut correct = 0;
        let total = training_set.len();

        for example in &training_set {
            let inputs = types::Inputs::from_iter(
                example.inputs.iter().map(|(k, v)| (k.clone(), v.clone()))
            );

            let prediction = optimized.forward(inputs).await?;

            if let (Some(expected), Some(actual)) =
                (example.get_output("sentiment"), prediction.get("sentiment")) {
                if expected.trim().eq_ignore_ascii_case(actual.trim()) {
                    correct += 1;
                }
            }
        }

        let accuracy_after = correct as f64 / total as f64;
        println!("  Accuracy after optimization: {:.1}%", accuracy_after * 100.0);

        // Record optimization run
        self.optimization_history.push(OptimizationRun {
            iteration,
            accuracy_before,
            accuracy_after,
            num_demos: training_set.len(),
            timestamp: std::time::SystemTime::now(),
        });

        // Update current model
        self.current_model = Some(optimized);

        if accuracy_after > accuracy_before {
            println!("  ✅ Improvement: +{:.1}%", (accuracy_after - accuracy_before) * 100.0);
        } else if accuracy_after == accuracy_before {
            println!("  ⚖️  No change in accuracy");
        } else {
            println!("  ⚠️  Accuracy decreased: {:.1}%", (accuracy_before - accuracy_after) * 100.0);
        }

        Ok(())
    }

    fn print_optimization_summary(&self) {
        println!("\n📊 OPTIMIZATION SUMMARY");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Total feedback collected: {}", self.feedback_collector.feedback.len());
        println!("Optimization runs: {}", self.optimization_history.len());

        if !self.optimization_history.is_empty() {
            let first = &self.optimization_history[0];
            let last = &self.optimization_history[self.optimization_history.len() - 1];

            println!("\nInitial accuracy: {:.1}%", first.accuracy_before * 100.0);
            println!("Final accuracy:   {:.1}%", last.accuracy_after * 100.0);
            println!("Total improvement: +{:.1}%",
                (last.accuracy_after - first.accuracy_before) * 100.0);
        }

        println!("\nOptimization History:");
        for run in &self.optimization_history {
            println!("  Iteration {}: {:.1}% → {:.1}% ({} demos)",
                run.iteration,
                run.accuracy_before * 100.0,
                run.accuracy_after * 100.0,
                run.num_demos,
            );
        }
    }
}

// ============================================================================
// Main Pipeline Execution
// ============================================================================

#[tokio::main]
async fn main() -> kkachi::Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║    Automated Prompt Optimization Pipeline Demo            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize the pipeline
    println!("🚀 Step 1: Initialize Pipeline");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let signature = Signature::parse("text -> sentiment")?
        .into_owned();

    let lm_client = Arc::new(MockLM::new());
    let mut pipeline = OptimizationPipeline::new(signature, lm_client);

    println!("✅ Pipeline initialized with signature: text -> sentiment\n");

    // Step 2: Collect initial feedback (simulating user interactions)
    println!("🚀 Step 2: Initial Deployment - Collect User Feedback");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let initial_test_cases = vec![
        ("I love this product!".to_string(), "positive".to_string()),
        ("This is terrible".to_string(), "negative".to_string()),
        ("It's okay I guess".to_string(), "neutral".to_string()),
        ("Best purchase ever!".to_string(), "positive".to_string()),
        ("Worst experience".to_string(), "negative".to_string()),
    ];

    pipeline.collect_feedback(initial_test_cases).await?;

    // Step 3: First optimization based on feedback
    println!("\n🚀 Step 3: First Optimization Cycle");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    pipeline.optimize_from_feedback(1).await?;

    // Step 4: Deploy optimized model and collect more feedback
    println!("\n🚀 Step 4: Deploy Optimized Model - Collect More Feedback");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let second_test_cases = vec![
        ("Absolutely wonderful!".to_string(), "positive".to_string()),
        ("Not good at all".to_string(), "negative".to_string()),
        ("Pretty average".to_string(), "neutral".to_string()),
        ("Amazing quality".to_string(), "positive".to_string()),
        ("Very disappointing".to_string(), "negative".to_string()),
        ("Exceeded expectations".to_string(), "positive".to_string()),
    ];

    pipeline.collect_feedback(second_test_cases).await?;

    // Step 5: Second optimization
    println!("\n🚀 Step 5: Second Optimization Cycle");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    pipeline.optimize_from_feedback(2).await?;

    // Step 6: Evaluate final model
    println!("\n🚀 Step 6: Final Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let eval_test_cases = vec![
        ("This is fantastic!".to_string(), "positive".to_string()),
        ("Really bad quality".to_string(), "negative".to_string()),
        ("It works fine".to_string(), "neutral".to_string()),
        ("Love it so much!".to_string(), "positive".to_string()),
        ("Hate this".to_string(), "negative".to_string()),
    ];

    println!("Testing optimized model on {} new cases...", eval_test_cases.len());

    let model = pipeline.current_model.as_ref().unwrap();
    let mut correct = 0;

    for (input_text, expected) in eval_test_cases {
        let mut inputs = types::Inputs::new();
        inputs.insert("text", input_text.clone());

        let prediction = model.forward(inputs).await?;
        let predicted = prediction.get("sentiment").unwrap_or("unknown");

        let is_correct = predicted.trim().eq_ignore_ascii_case(expected.trim());
        if is_correct {
            correct += 1;
        }

        let status = if is_correct { "✅" } else { "❌" };
        println!("  {} '{}' → {} (expected: {})",
            status, input_text, predicted, expected);
    }

    println!("\nFinal Test Accuracy: {}/5 ({:.1}%)", correct, (correct as f64 / 5.0) * 100.0);

    // Step 7: Print summary
    pipeline.print_optimization_summary();

    // Step 8: Demonstrate production deployment
    println!("\n🚀 Step 7: Production Deployment Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("The optimized model is now ready for production!");
    println!("\nKey Benefits:");
    println!("  ✅ Learned from {} user interactions",
        pipeline.feedback_collector.feedback.len());
    println!("  ✅ Automated improvement without manual prompt engineering");
    println!("  ✅ Continuous learning from production feedback");
    println!("  ✅ Zero-copy Rust performance");
    println!("  ✅ Type-safe at compile time");

    println!("\n💡 Next Steps:");
    println!("  1. Deploy optimized model to production");
    println!("  2. Continue collecting feedback");
    println!("  3. Schedule periodic re-optimization (e.g., nightly)");
    println!("  4. Monitor accuracy metrics");
    println!("  5. A/B test new optimizations");

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              Pipeline Execution Complete! ✅               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
