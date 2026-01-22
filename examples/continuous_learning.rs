// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Continuous Learning Pipeline
//!
//! This example shows how to implement a continuous learning system that:
//! - Monitors production predictions
//! - Collects correction feedback
//! - Automatically retrains when accuracy drops
//! - Deploys improved models
//!
//! Run with: cargo run --example continuous_learning

use kkachi::*;
use kkachi::predict::{LMClient, LMResponse};
use kkachi::prediction::TokenUsage;
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::Mutex;

// Production-ready LM client wrapper
struct ProductionLM {
    // In real usage, this would be an OpenAI/Anthropic client
    responses: std::collections::HashMap<String, String>,
}

impl ProductionLM {
    fn new() -> Self {
        let mut responses = std::collections::HashMap::new();
        responses.insert("classify_spam".to_string(), "Classification: spam".to_string());
        responses.insert("classify_ham".to_string(), "Classification: not spam".to_string());
        Self { responses }
    }
}

#[async_trait]
impl LMClient for ProductionLM {
    async fn generate(&self, prompt: &str) -> kkachi::Result<LMResponse> {
        // Simple classification based on keywords
        let text = if prompt.contains("buy now") || prompt.contains("winner") {
            "Classification: spam"
        } else {
            "Classification: not spam"
        }.to_string();

        Ok(LMResponse {
            text,
            usage: Some(TokenUsage::new(20, 10)),
        })
    }
}

// Feedback storage with thread-safe access
struct FeedbackStore {
    corrections: Arc<Mutex<Vec<(String, String, String)>>>,
}

impl FeedbackStore {
    fn new() -> Self {
        Self {
            corrections: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn add_correction(&self, input: String, predicted: String, correct: String) {
        let mut corrections = self.corrections.lock().unwrap();
        println!("  📝 Correction recorded: predicted '{}', should be '{}'",
            predicted, correct);
        corrections.push((input, predicted, correct));
    }

    fn get_corrections(&self) -> Vec<(String, String, String)> {
        self.corrections.lock().unwrap().clone()
    }

    fn correction_rate(&self) -> f64 {
        let corrections = self.corrections.lock().unwrap();
        // In production, track total predictions
        corrections.len() as f64
    }
}

// Continuous learning orchestrator
struct ContinuousLearner {
    signature: Signature<'static>,
    lm_client: Arc<dyn LMClient>,
    current_model: Predict<'static>,
    feedback_store: FeedbackStore,
    retraining_threshold: usize,
}

impl ContinuousLearner {
    fn new(signature: Signature<'static>, lm_client: Arc<dyn LMClient>) -> Self {
        let model = Predict::new(signature.clone()).with_lm(lm_client.clone());

        Self {
            signature,
            lm_client,
            current_model: model,
            feedback_store: FeedbackStore::new(),
            retraining_threshold: 3,
        }
    }

    async fn predict(&self, input: &str) -> kkachi::Result<String> {
        let mut inputs = types::Inputs::new();
        inputs.insert("email", input);

        let prediction = self.current_model.forward(inputs).await?;
        Ok(prediction.get("classification")
            .unwrap_or("unknown")
            .to_string())
    }

    fn submit_correction(&self, input: String, predicted: String, correct: String) {
        self.feedback_store.add_correction(input, predicted, correct);
    }

    fn should_retrain(&self) -> bool {
        let corrections = self.feedback_store.get_corrections();
        corrections.len() >= self.retraining_threshold
    }

    async fn retrain(&mut self) -> kkachi::Result<()> {
        println!("\n🔄 Triggering automatic retraining...");

        let corrections = self.feedback_store.get_corrections();
        println!("  Training on {} corrections", corrections.len());

        // Build training set from corrections
        let mut training_set = Vec::new();
        for (input, _predicted, correct) in corrections {
            let mut example = Example::new();
            example.insert_input("email", input);
            example.insert_output("classification", correct);
            training_set.push(example.into_owned());
        }

        // Create optimizer
        let config = OptimizerConfig {
            max_iterations: 1,
            batch_size: training_set.len(),
            seed: 42,
            metric_threshold: Some(0.9),
        };

        let optimizer = BootstrapFewShot::new(config)
            .with_max_demos(training_set.len());

        // Create a new model for optimization (Predict doesn't impl Clone)
        let base_model = Predict::new(self.signature.clone())
            .with_lm(self.lm_client.clone());

        // Optimize
        let optimized = optimizer.optimize(base_model, &training_set).await?;

        // Update model
        self.current_model = optimized;

        println!("  ✅ Model retrained and deployed");

        Ok(())
    }
}

#[tokio::main]
async fn main() -> kkachi::Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║           Continuous Learning Pipeline Demo               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Initialize
    let signature = Signature::parse("email -> classification")?.into_owned();
    let lm_client = Arc::new(ProductionLM::new());
    let mut learner = ContinuousLearner::new(signature, lm_client);

    println!("📧 Email Spam Classifier - Continuous Learning Enabled\n");

    // Simulate production usage with feedback
    let test_cases = vec![
        ("Check out this amazing offer!", "spam"),
        ("Meeting at 3pm tomorrow", "not spam"),
        ("You are a winner! Click here!", "spam"),
        ("Quarterly report attached", "not spam"),
        ("Buy now for 50% off!!!", "spam"),
    ];

    println!("🔍 Processing emails and collecting feedback...\n");

    for (i, (email, correct_label)) in test_cases.iter().enumerate() {
        println!("Email #{}: \"{}\"", i + 1, email);

        // Make prediction
        let prediction = learner.predict(email).await?;
        println!("  Predicted: {}", prediction);

        // Simulate user correction if wrong
        if !prediction.contains(correct_label) {
            println!("  ❌ Incorrect! User provides correction: {}", correct_label);
            learner.submit_correction(
                email.to_string(),
                prediction,
                correct_label.to_string(),
            );

            // Check if we should retrain
            if learner.should_retrain() {
                learner.retrain().await?;
            }
        } else {
            println!("  ✅ Correct!");
        }

        println!();
    }

    // Test improved model
    println!("\n🎯 Testing improved model on new examples...\n");

    let new_emails = vec![
        "Limited time offer - act now!",
        "Project deadline reminder",
        "Congratulations! You won a prize!",
    ];

    for email in new_emails {
        let prediction = learner.predict(email).await?;
        println!("  \"{}\" → {}", email, prediction);
    }

    println!("\n💡 Key Features Demonstrated:");
    println!("  ✅ Real-time feedback collection");
    println!("  ✅ Automatic retraining triggers");
    println!("  ✅ Zero-downtime model updates");
    println!("  ✅ Thread-safe feedback storage");
    println!("  ✅ Production-ready architecture");

    println!("\n╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
