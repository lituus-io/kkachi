// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Basic usage example showing core Kkachi functionality

use kkachi::*;
use std::borrow::Cow;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create a signature using string format
    let signature = Signature::from_str("question, context -> answer")?;
    println!("Signature: {}", signature.to_string_format());

    // 2. Create an example with inputs and expected outputs
    let mut example = Example::new();
    example.insert_input("question", "What is the capital of France?");
    example.insert_input("context", "France is a country in Europe.");
    example.insert_output("answer", "Paris");

    println!("\nExample inputs:");
    for (key, value) in example.inputs.iter() {
        println!("  {}: {}", key, value);
    }

    // 3. Create a prediction (simulating model output)
    let mut prediction = Prediction::new();
    prediction.insert("answer", Cow::Borrowed("Paris"));

    println!("\nPrediction:");
    if let Some(answer) = prediction.get("answer") {
        println!("  answer: {}", answer);
    }

    // 4. Use a metric to evaluate
    use kkachi_eval::metric::{ExactMatch, Metric};
    let metric = ExactMatch;
    let result = metric.evaluate(&example, &prediction);

    println!("\nEvaluation:");
    println!("  Score: {}", result.score);
    println!("  Passed: {}", result.passed);

    Ok(())
}
