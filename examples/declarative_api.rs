// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Declarative API Example
//!
//! This example demonstrates the concise, fluent API for building
//! recursive refinement pipelines with kkachi.
//!
//! Run with: cargo run --example declarative_api

use kkachi::error::Result;
use kkachi::recursive::{
    checks, cli, refine, IterativeMockLlm, ValidateExt,
};

fn main() -> Result<()> {
    println!("=== Kkachi Declarative API Examples ===\n");

    // Example 1: Simple refinement with check builder
    example_checks_builder()?;

    // Example 2: CLI pipeline validation
    example_cli_pipeline()?;

    // Example 3: Using refine() with validate()
    example_refine()?;

    // Example 4: Composing validators
    example_composition()?;

    // Summary of API features
    print_api_summary();

    Ok(())
}

/// Example 1: Simple refinement with check builder
fn example_checks_builder() -> Result<()> {
    println!("1. Simple Code Generation (Checks Builder)");
    println!("   ─────────────────────────────────────────");

    // Create a check validator with length and pattern requirements
    let validator = checks()
        .min_len(20)
        .require("fn ");

    // Create a mock LLM that improves over iterations
    let responses = [
        "add numbers",                              // Too short, missing "fn "
        "fn add(a: i32) -> i32 { a }",              // Has fn, good length
        "fn add(a: i32, b: i32) -> i32 { a + b }",  // Perfect
    ];
    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    let result = refine(&llm, "Write a function to add two numbers")
        .validate(validator)
        .max_iter(5)
        .target(1.0)
        .on_iter(|iter, score| {
            println!("   Iteration {}: score = {:.2}", iter, score);
        })
        .go_full()?;

    println!("   Result: {}", result.output);
    println!("   Score: {:.2}, Iterations: {}\n", result.score, result.iterations);

    Ok(())
}

/// Example 2: CLI pipeline validation
fn example_cli_pipeline() -> Result<()> {
    println!("2. CLI Pipeline Validation");
    println!("   ────────────────────────");

    // Compose a Rust validator using the new cli() builder with chained stages
    let rust_validator = cli("echo").arg("format-check").weight(0.1)
        .then("echo").arg("compile-check").weight(0.6).required()
        .then("echo").arg("lint-check").weight(0.3)
        .ext("rs");

    println!("   Created Rust validator pipeline with 3 stages:");
    println!("     - format (weight: 0.1)");
    println!("     - compile (weight: 0.6, required)");
    println!("     - lint (weight: 0.3)");
    println!();

    // Example of a Python validator
    let python_validator = cli("python").args(&["-m", "py_compile"]).required()
        .then("ruff").arg("check")
        .ext("py");

    println!("   Created Python validator pipeline with 2 stages:");
    println!("     - syntax (required)");
    println!("     - lint");
    println!();

    // Example of a Terraform validator
    let terraform_validator = cli("terraform").args(&["fmt", "-check"]).weight(0.2)
        .then("terraform").arg("validate").required()
        .ext("tf");

    println!("   Created Terraform validator pipeline with 2 stages:");
    println!("     - fmt (weight: 0.2)");
    println!("     - validate (required)");
    println!();

    // Prevent unused variable warnings
    let _ = (rust_validator, python_validator, terraform_validator);

    Ok(())
}

/// Example 3: Using refine() with validate()
fn example_refine() -> Result<()> {
    println!("3. refine() with Custom Validator");
    println!("   ───────────────────────────────────────");

    // Create a simple validator for demonstration
    let validator = cli("echo").arg("ok").required().ext("txt");

    // Create an iterative mock LLM
    let responses = [
        "hello",
        "fn hello() { }",
        "fn hello() { println!(\"Hello, World!\"); }",
    ];
    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    // Use the new refine() API
    let result = refine(&llm, "Write a hello world function")
        .validate(validator)
        .max_iter(5)
        .target(0.9)
        .on_iter(|iter, score| {
            println!("   Iteration {}: score = {:.2}", iter, score);
        })
        .go_full()?;

    println!("   Result: {}", result.output);
    println!("   Score: {:.2}, Iterations: {}\n", result.score, result.iterations);

    Ok(())
}

/// Example 4: Composing validators with and/or
fn example_composition() -> Result<()> {
    println!("4. Validator Composition");
    println!("   ───────────────────────");

    // Create individual validators
    let has_fn = checks().require("fn ");
    let has_return = checks().require("->");
    let no_unwrap = checks().forbid(".unwrap()");

    // Compose with AND (all must pass)
    let strict = has_fn.and(has_return).and(no_unwrap);

    // Test the composed validator
    let good_code = "fn parse(s: &str) -> Option<i32> { s.parse().ok() }";
    let bad_code = "fn parse(s: &str) -> i32 { s.parse().unwrap() }";

    use kkachi::recursive::Validate;

    println!("   Composed validator: has_fn AND has_return AND no_unwrap");
    println!("   ");
    println!("   Good code: {}", good_code);
    println!("   Score: {:.2}", strict.validate(good_code).value);
    println!("   ");
    println!("   Bad code (has unwrap): {}", bad_code);
    println!("   Score: {:.2}\n", strict.validate(bad_code).value);

    Ok(())
}

fn print_api_summary() {
    println!("=== API Summary ===");
    println!("
The new simplified API provides:

  Entry Point:
    refine(llm, prompt)     - Start a refinement pipeline

  Builder Methods:
    .validate(v)            - Use a validator
    .memory(mem)            - Use RAG memory
    .k(n)                   - Number of examples to retrieve
    .max_iter(n)            - Maximum refinement iterations
    .target(threshold)      - Convergence score threshold
    .on_iter(callback)      - Progress callback
    .learn()                - Enable learning
    .learn_above(threshold) - Learn when score >= threshold
    .chain_of_thought()     - Enable CoT reasoning
    .best_of::<N>()         - Best-of-N sampling

  Check Validators:
    checks()                - Create a check builder
      .require(pattern)     - Require substring
      .forbid(pattern)      - Forbid substring
      .min_len(n)           - Minimum length
      .max_len(n)           - Maximum length
      .regex(pattern)       - Regex pattern match
      .pred(name, fn)       - Custom predicate

  CLI Validators:
    cli(cmd)                - Create a CLI validator
      .arg(arg)             - Add single argument
      .args(&[...])         - Add multiple arguments
      .weight(w)            - Set scoring weight (0.0-1.0)
      .required()           - Mark as required stage
      .then(cmd)            - Add another stage
      .ext(extension)       - Set temp file extension
      .env(k, v)            - Set environment variable
      .env_from(key)        - Inherit env variable
      .timeout(secs)        - Set timeout
      .capture()            - Enable output capture

  Composition:
    v1.and(v2)              - Both must pass
    v1.or(v2)               - Either must pass

  Execution:
    .go()                   - Run and return output (Result<String>)
    .go_scored()            - Run and return (output, score) (Result)
    .go_full()              - Run and return full RefineResult (Result)
    .compile()              - Compile to reusable program
");
}
