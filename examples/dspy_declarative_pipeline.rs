// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! DSPy-Style Declarative Pipeline Example
//!
//! This example demonstrates kkachi's APIs for building sophisticated LLM pipelines
//! with DSPy-style patterns:
//!
//! - **Declarative Refinement**: Iterative improvement with validators
//! - **Chain of Thought**: Step-by-step reasoning (via instructions)
//! - **Best of N**: Multiple candidates, select best
//! - **Tool Validation**: CLI-based code validation (Rust, Python, Go, etc.)

use kkachi::{
    // Diff visualization
    diff::{DiffRenderer, DiffStyle, TextDiff},
    // Candidate pools for analysis
    CandidatePool,
    // Recall/Precision tuning
    RecallPrecisionMode,
    ScoredCandidate,
    // Recursive refinement API
    recursive::{
        checks, cli, refine, IterativeMockLlm, Validate,
    },
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   DSPy-Style Declarative Pipeline with Kkachi");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run examples
    example_1_cli_validator_pipeline();
    example_2_refinement_with_checks()?;
    example_3_candidate_pool_analysis()?;
    example_4_recall_precision_modes()?;
    example_5_diff_visualization()?;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   All DSPy Examples Complete!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Example 1: CLI Validator Pipeline
fn example_1_cli_validator_pipeline() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: CLI Validator Pipeline                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create a multi-stage CLI validator using the new fluent API
    // Users define their own CLI validators - the library provides only primitives
    let rust_validator = cli("rustfmt")
        .args(&["--check"])
        .weight(0.1)
        .then("rustc")
        .args(&["--emit=metadata", "-o", "/dev/null"])
        .weight(0.6)
        .required()
        .then("cargo")
        .args(&["clippy", "--", "-D", "warnings"])
        .weight(0.3)
        .ext("rs");

    println!("  Created Rust pipeline with user-defined validator:");
    println!("    - Stages: rustfmt -> rustc -> clippy");
    println!("    - Weights: 0.1 -> 0.6 -> 0.3");
    println!("    - Required stage: rustc");
    println!();

    // Simple single-command validator
    let simple_validator = cli("rustfmt")
        .args(&["--check"])
        .ext("rs");

    println!("  Created simple validator:");
    println!("    - Command: rustfmt --check");
    println!("    - File extension: .rs");
    println!();

    // Check that validators implement Validate
    fn assert_validates<V: Validate>(_v: &V) {}
    assert_validates(&rust_validator);
    assert_validates(&simple_validator);
}

/// Example 2: Refinement Loop with Checks
#[allow(clippy::useless_vec)]
fn example_2_refinement_with_checks() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Refinement Loop with Checks                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Simulated responses (in real use, these come from an LLM)
    let responses = [
        "25% of 80 = 20",
        "To find 25% of 80: 0.25 * 80 = 20. Therefore, 20.",
        "Step 1: Convert 25% to decimal: 25/100 = 0.25\nStep 2: Multiply: 0.25 × 80 = 20\nTherefore, 25% of 80 is 20.",
    ];

    // Create a mock LLM that improves over iterations
    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    // Create a validator using the checks() builder
    let validator = checks()
        .require("20")          // Must contain the answer
        .require("0.25")        // Must show the decimal conversion
        .min_len(30);           // Must be reasonably detailed

    // Run refinement with the new API
    let result = refine(&llm, "What is 25% of 80?")
        .validate(validator)
        .max_iter(3)
        .target(1.0)
        .go_full()?;

    println!("  Question: What is 25% of 80?");
    println!("  Answer: {}", result.output);
    println!("  Score: {:.0}%", result.score * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.score >= 1.0);
    println!();

    Ok(())
}

/// Example 3: Candidate Pool Analysis (BestOfN style)
#[allow(clippy::useless_vec)]
fn example_3_candidate_pool_analysis() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 3: Candidate Pool Analysis (BestOfN)                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Simulated candidates with pre-computed scores
    let candidates = vec![
        ("fn parse(s: &str) { s.parse().unwrap() }", 0.2),
        ("fn parse(s: &str) -> i32 { s.parse().unwrap() }", 0.4),
        ("fn parse(s: &str) -> Option<i32> { s.parse().ok() }", 0.6),
        (
            "/// Parse string\nfn parse(s: &str) -> Option<i32> { s.parse().ok() }",
            0.8,
        ),
        (
            "/// Parse safely.\nfn parse(s: &str) -> Result<i32, _> { s.parse() }",
            1.0,
        ),
    ];

    println!("  Generated {} candidates:\n", candidates.len());
    for (i, (code, score)) in candidates.iter().enumerate() {
        println!("    Candidate {}: score={:.1}", i + 1, score);
        println!("      {}", code.lines().next().unwrap_or(""));
    }

    // Build CandidatePool
    let mut pool = CandidatePool::new();
    for (i, (_, score)) in candidates.iter().enumerate() {
        pool.push(ScoredCandidate::new(*score, i as u8));
    }
    pool.sort_by_score();

    // Statistics
    let stats = pool.stats();
    println!("\n  Pool Statistics:");
    println!("    Count: {}", stats.count);
    println!("    Best: {:.2}", stats.best_score);
    println!("    Mean: {:.2}", stats.mean_score);
    println!("    Std Dev: {:.2}", stats.std_dev);

    // Get best candidate
    if let Some(best) = pool.best() {
        let idx = best.index as usize;
        println!("\n  Selected Best Candidate (index {}):", idx);
        println!("    {}", candidates[idx].0);
    }
    println!();

    Ok(())
}

/// Example 4: Recall/Precision Modes
fn example_4_recall_precision_modes() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: Recall/Precision Modes                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // High Recall mode - more permissive, captures more results
    let high_recall = RecallPrecisionMode::high_recall(0.6);
    println!("  High Recall Mode:");
    println!("    Threshold: {:.2}", high_recall.threshold());
    println!("    Use case: Initial exploration, brainstorming");

    // High Precision mode - more strict, higher quality
    let high_precision = RecallPrecisionMode::high_precision(0.9);
    println!("\n  High Precision Mode:");
    println!("    Threshold: {:.2}", high_precision.threshold());
    println!("    Use case: Final validation, production code");

    // Balanced mode
    let balanced = RecallPrecisionMode::Balanced;
    println!("\n  Balanced Mode:");
    println!("    Threshold: {:.2}", balanced.threshold());
    println!("    Use case: General purpose, development");

    // Example usage with refinement
    let responses = ["result = x + y", "result = x + y  # Add x and y"];

    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    // Use high precision threshold for refinement
    let result = refine(&llm, "Add x and y")
        .validate(checks().require("result").require("x + y"))
        .max_iter(2)
        .target(high_precision.threshold())
        .go_full()?;

    println!("\n  Example with High Precision:");
    println!("    Final score: {:.2}", result.score);
    println!("    Converged: {}", result.score >= 0.9);
    println!();

    Ok(())
}

/// Example 5: Diff Visualization
fn example_5_diff_visualization() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 5: Diff Visualization                                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Simulated iteration improvements
    let v1 = "fn parse(s: &str) {\n    s.parse().unwrap()\n}";
    let v2 = "fn parse(s: &str) -> i32 {\n    s.parse().unwrap()\n}";
    let v3 = "/// Parse a string to i32.\nfn parse(s: &str) -> Result<i32, _> {\n    s.parse()\n}";

    let renderer = DiffRenderer::new().with_style(DiffStyle::Unified);

    println!("  Iteration 1 → 2:");
    let diff1 = TextDiff::new(v1, v2);
    println!("{}", renderer.render_text(&diff1));

    println!("\n  Iteration 2 → 3:");
    let diff2 = TextDiff::new(v2, v3);
    println!("{}", renderer.render_text(&diff2));

    println!("\n  Changes show progressive improvement:");
    println!("    1. Added return type");
    println!("    2. Added documentation");
    println!("    3. Changed to Result for better error handling");
    println!();

    Ok(())
}
