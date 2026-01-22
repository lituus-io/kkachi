// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! DSPy-Style Declarative Pipeline Example
//!
//! This example demonstrates kkachi's two declarative APIs for building
//! sophisticated LLM pipelines with DSPy-style patterns:
//!
//! 1. **New API**: `pipeline()` - Zero-copy, type-state, compile-time checked
//! 2. **Classic API**: `Kkachi::refine()` - Runtime configuration with closures
//!
//! Both APIs support:
//! - **Declarative Refinement**: Iterative improvement with critics
//! - **Chain of Thought**: Step-by-step reasoning
//! - **Best of N**: Multiple candidates, select best
//! - **Tool Validation**: CLI-based code validation (Rust, Python, Go, etc.)

use kkachi::{
    // Generic CLI validation primitives
    declarative::{Cli, CliPipeline},
    // Diff visualization
    diff::{DiffRenderer, DiffStyle, TextDiff},
    // New declarative API
    pipeline,
    // Candidate/Chain pools for analysis
    CandidatePool,
    FluentPipeline,
    // Classic recursive API
    Kkachi,
    NoStrategy,
    // Recall/Precision tuning
    RecallPrecisionMode,
    ScoredCandidate,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   DSPy-Style Declarative Pipeline with Kkachi");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run examples
    example_1_new_pipeline_api();
    example_2_classic_api_with_cot()?;
    example_3_candidate_pool_analysis()?;
    example_4_recall_precision_modes()?;
    example_5_diff_visualization()?;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   All DSPy Examples Complete!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Example 1: New Pipeline API (Type-State, Zero-Copy)
fn example_1_new_pipeline_api() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: New Pipeline API (Type-State)                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // The new pipeline() API uses type-state pattern for compile-time safety
    // Each method call returns a new type, enabling compile-time checking

    // Create a pipeline for Rust code generation with user-defined validator
    // Users define their own CLI validators - the library provides only primitives
    let rust_validator = CliPipeline::new()
        .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
        .stage(
            "compile",
            Cli::new("rustc")
                .args(["--emit=metadata", "-o", "/dev/null"])
                .weight(0.6)
                .required(),
        )
        .stage(
            "lint",
            Cli::new("cargo")
                .args(["clippy", "--", "-D", "warnings"])
                .weight(0.3),
        )
        .file_ext("rs");

    let _rust_pipeline: FluentPipeline<'_, _, NoStrategy> = pipeline("question -> code")
        .validate(rust_validator) // Generic validate() accepts any Validator
        .refine(5, 0.9); // Max 5 iterations, 0.9 score threshold

    println!("  Created Rust pipeline with user-defined validator:");
    println!("    - Stages: rustfmt -> rustc -> clippy");
    println!("    - Max iterations: 5");
    println!("    - Score threshold: 0.9");
    println!();

    // Pipeline with Chain of Thought - type changes at compile time!
    let _cot_pipeline = pipeline("question -> reasoning, answer")
        .chain_of_thought() // Returns FluentPipeline<'_, NoCritic, WithCoT>
        .refine(3, 0.85);

    println!("  Created CoT pipeline with:");
    println!("    - Strategy: Chain of Thought");
    println!("    - Max iterations: 3");
    println!("    - Score threshold: 0.85");
    println!();

    // Pipeline with validation - critic type changes at compile time!
    // This shows the generic validate() method with a single CLI command
    let _simple_validated = pipeline("requirement -> rust_code")
        .validate(Cli::new("rustfmt").args(["--check"]).file_ext("rs"))
        .strict(); // 10 iterations, 0.95 threshold

    println!("  Created validated pipeline with:");
    println!("    - Validator: rustfmt --check (single command)");
    println!("    - Preset: strict (10 iter, 0.95 threshold)");
    println!();
}

/// Example 2: Classic API with Chain of Thought (Now Fixed!)
#[allow(clippy::useless_vec)]
fn example_2_classic_api_with_cot() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Classic API with Chain of Thought                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Simulated CoT-style responses (in real use, these come from an LLM)
    let responses = vec![
        "25% of 80 = 20",
        "To find 25% of 80: 0.25 * 80 = 20. Therefore, 20.",
        "Step 1: Convert 25% to decimal: 25/100 = 0.25\nStep 2: Multiply: 0.25 × 80 = 20\nTherefore, 25% of 80 is 20.",
    ];

    // The classic API now properly uses CoT!
    // Previously, .with_chain_of_thought() was stored but never used
    let result = Kkachi::refine("question -> answer")
        .domain("math")
        .max_iterations(3)
        .until_score(0.8)
        .with_chain_of_thought() // Now actually adds CoT instructions!
        .run(
            "What is 25% of 80?",
            |iter: u32, _feedback: Option<&str>| {
                // The feedback now includes CoT instructions
                Ok(responses[iter.min(responses.len() as u32 - 1) as usize].to_string())
            },
        );

    println!("  Question: What is 25% of 80?");
    println!("  Answer: {}", result.answer);
    println!("  Score: {:.0}%", result.score * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.score >= 0.8);
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

    // Using with classic API
    let responses = ["result = x + y", "result = x + y  # Add x and y"];

    let result = Kkachi::refine("task -> code")
        .domain("python")
        .high_precision() // Use high precision mode
        .max_iterations(2)
        .run("Add x and y", |iter: u32, _: Option<&str>| {
            Ok(responses[iter.min(responses.len() as u32 - 1) as usize].to_string())
        });

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
