// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Continuous Learning Pipeline
//!
//! This example demonstrates a continuous learning system that:
//! - Monitors predictions and collects feedback
//! - Stores successful refinements for future retrieval (RAG)
//! - Uses feedback to improve code generation
//!
//! Run with: cargo run --example continuous_learning

use std::sync::{Arc, Mutex};

use kkachi::error::Result;
use kkachi::recursive::{
    checks, memory, refine, Memory, IterativeMockLlm,
};

// ============================================================================
// Feedback Storage
// ============================================================================

/// Thread-safe storage for user corrections
struct FeedbackStore {
    corrections: Arc<Mutex<Vec<Correction>>>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Correction {
    question: String,
    original_answer: String,
    corrected_answer: String,
    timestamp: u64,
}

impl FeedbackStore {
    fn new() -> Self {
        Self {
            corrections: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn add_correction(&self, question: &str, original: &str, corrected: &str) {
        let mut corrections = self.corrections.lock().unwrap();
        corrections.push(Correction {
            question: question.to_string(),
            original_answer: original.to_string(),
            corrected_answer: corrected.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
    }

    #[allow(dead_code)]
    fn get_corrections(&self) -> Vec<Correction> {
        self.corrections.lock().unwrap().clone()
    }

    fn correction_count(&self) -> usize {
        self.corrections.lock().unwrap().len()
    }
}

// ============================================================================
// Accuracy Monitor
// ============================================================================

/// Tracks prediction accuracy over time
struct AccuracyMonitor {
    predictions: Arc<Mutex<Vec<PredictionRecord>>>,
    accuracy_threshold: f64,
}

#[derive(Clone)]
#[allow(dead_code)]
struct PredictionRecord {
    question: String,
    prediction: String,
    was_correct: Option<bool>,
    score: f64,
}

impl AccuracyMonitor {
    fn new(threshold: f64) -> Self {
        Self {
            predictions: Arc::new(Mutex::new(Vec::new())),
            accuracy_threshold: threshold,
        }
    }

    fn record_prediction(&self, question: &str, prediction: &str, score: f64) {
        let mut predictions = self.predictions.lock().unwrap();
        predictions.push(PredictionRecord {
            question: question.to_string(),
            prediction: prediction.to_string(),
            was_correct: None,
            score,
        });
    }

    fn mark_correct(&self, index: usize, correct: bool) {
        let mut predictions = self.predictions.lock().unwrap();
        if let Some(record) = predictions.get_mut(index) {
            record.was_correct = Some(correct);
        }
    }

    fn get_accuracy(&self) -> f64 {
        let predictions = self.predictions.lock().unwrap();
        let labeled: Vec<_> = predictions.iter().filter(|p| p.was_correct.is_some()).collect();
        if labeled.is_empty() {
            return 1.0;
        }
        let correct = labeled.iter().filter(|p| p.was_correct == Some(true)).count();
        correct as f64 / labeled.len() as f64
    }

    fn should_retrain(&self) -> bool {
        self.get_accuracy() < self.accuracy_threshold
    }

    fn prediction_count(&self) -> usize {
        self.predictions.lock().unwrap().len()
    }
}

// ============================================================================
// Continuous Learning System
// ============================================================================

/// Main system that coordinates learning from feedback
struct ContinuousLearningSystem {
    memory: Memory,
    feedback_store: FeedbackStore,
    accuracy_monitor: AccuracyMonitor,
    generation_count: usize,
}

impl ContinuousLearningSystem {
    fn new() -> Self {
        Self {
            memory: memory(),
            feedback_store: FeedbackStore::new(),
            accuracy_monitor: AccuracyMonitor::new(0.8),
            generation_count: 0,
        }
    }

    /// Add seed examples to bootstrap the system
    fn seed_examples(&mut self, examples: &[(&str, &str, &str)]) {
        for (id, question, answer) in examples {
            let content = format!("Q: {}\nA: {}", question, answer);
            self.memory.add_tagged(id, &content);
        }
    }

    /// Generate code with refinement
    fn generate(&mut self, question: &str) -> Result<(String, f64)> {
        self.generation_count += 1;

        // Retrieve similar examples for context
        let similar = self.memory.search(question, 3);
        println!("  Retrieved {} similar examples for context", similar.len());

        // Create a checklist validator for code quality
        let validator = checks()
            .require("fn ")
            .require("->")
            .forbid("TODO")
            .min_len(20);

        // Create a mock LLM that improves over iterations
        let q = question.to_string();
        let responses = [
            format!("fn process() {{ /* {} */ }}", q),
            "/// Process data\nfn process() -> String { \"result\".into() }".to_string(),
            "/// Processes the input.\n/// Returns the result.\nfn process(input: &str) -> String {\n    input.to_uppercase()\n}".to_string(),
        ];
        let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
            let idx = (iter as usize).min(responses.len() - 1);
            responses[idx].clone()
        });

        // Run refinement
        let result = refine(&llm, question)
            .validate(validator)
            .max_iter(5)
            .target(1.0)
            .go_full()?;

        // Record the prediction
        self.accuracy_monitor.record_prediction(question, &result.output, result.score);

        Ok((result.output, result.score))
    }

    /// Submit user feedback/correction
    fn submit_feedback(&mut self, question: &str, original: &str, corrected: &str, was_correct: bool) {
        // Store the correction
        self.feedback_store.add_correction(question, original, corrected);

        // Update accuracy tracking
        let count = self.accuracy_monitor.prediction_count();
        if count > 0 {
            self.accuracy_monitor.mark_correct(count - 1, was_correct);
        }

        // If correction provided, add to memory for future retrieval
        if !was_correct && !corrected.is_empty() {
            let id = format!("correction:{}", self.feedback_store.correction_count());
            let content = format!("Q: {}\nA: {}", question, corrected);
            self.memory.add_tagged(&id, &content);
            println!("  Added correction to knowledge base");
        }
    }

    /// Check if retraining is needed
    fn check_and_retrain(&self) -> bool {
        if self.accuracy_monitor.should_retrain() {
            println!("\n⚠️  Accuracy dropped below threshold!");
            println!("   Current accuracy: {:.1}%", self.accuracy_monitor.get_accuracy() * 100.0);
            println!("   Corrections available: {}", self.feedback_store.correction_count());
            println!("   → Retraining would incorporate {} corrections", self.feedback_store.correction_count());
            true
        } else {
            false
        }
    }

    fn get_stats(&self) -> (usize, f64, usize) {
        (
            self.generation_count,
            self.accuracy_monitor.get_accuracy(),
            self.feedback_store.correction_count(),
        )
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("            Continuous Learning Pipeline Demo");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut system = ContinuousLearningSystem::new();

    // Step 1: Seed with initial examples
    println!("Step 1: Seeding knowledge base...");
    system.seed_examples(&[
        ("rust:parse", "How to parse JSON?", "use serde_json;\nfn parse(s: &str) -> Value { serde_json::from_str(s).unwrap() }"),
        ("rust:file", "How to read a file?", "use std::fs;\nfn read(path: &str) -> String { fs::read_to_string(path).unwrap() }"),
        ("rust:http", "How to make HTTP request?", "use reqwest;\nasync fn get(url: &str) -> String { reqwest::get(url).await.unwrap().text().await.unwrap() }"),
    ]);
    println!("  Loaded 3 seed examples\n");

    // Step 2: Generate some predictions
    println!("Step 2: Running predictions...\n");

    let questions = [
        "Write a function to uppercase a string",
        "How to parse TOML config?",
        "Write error handling code",
    ];

    for (i, question) in questions.iter().enumerate() {
        println!("  Prediction {}:", i + 1);
        println!("  Question: {}", question);

        let (answer, score) = system.generate(question)?;
        println!("  Score: {:.2}", score);
        println!("  Answer preview: {}...\n", &answer[..answer.len().min(50)]);

        // Simulate user feedback
        let was_correct = score >= 0.8;
        if !was_correct {
            system.submit_feedback(
                question,
                &answer,
                "/// Corrected version\nfn corrected() -> String { \"fixed\".into() }",
                false,
            );
        } else {
            system.submit_feedback(question, &answer, "", true);
        }
    }

    // Step 3: Check if retraining is needed
    println!("Step 3: Checking system health...");
    system.check_and_retrain();

    // Step 4: Show final statistics
    let (generations, accuracy, corrections) = system.get_stats();
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                        STATISTICS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Total generations: {}", generations);
    println!("  Current accuracy:  {:.1}%", accuracy * 100.0);
    println!("  Corrections stored: {}", corrections);
    println!();

    // Step 5: Demonstrate using refine() with memory
    println!("Step 5: Using refine() with memory...\n");

    let responses = [
        "read file",
        "fn read(p: &str) -> String { std::fs::read_to_string(p).unwrap() }",
        "/// Reads file contents safely.\nfn read(path: &str) -> Result<String, std::io::Error> {\n    std::fs::read_to_string(path)\n}",
    ];
    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    let validator = checks()
        .require("fn ")
        .require("->")
        .min_len(20);

    let result = refine(&llm, "Write a safe file reader")
        .validate(validator)
        .max_iter(5)
        .target(0.9)
        .on_iter(|iter, score| {
            println!("    Iteration {}: score = {:.2}", iter, score);
        })
        .go_full()?;

    println!("\n  Final result:");
    println!("    Score: {:.2}", result.score);
    println!("    Iterations: {}", result.iterations);

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                      Demo Complete!");
    println!("═══════════════════════════════════════════════════════════════════\n");

    Ok(())
}
