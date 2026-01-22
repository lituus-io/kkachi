// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Declarative API Example
//!
//! This example demonstrates the concise, fluent API for building
//! recursive refinement pipelines with kkachi.
//!
//! Run with: cargo run --example declarative_api

use kkachi::prelude::*;
use kkachi::{FnLLM, Error};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Kkachi Declarative API Examples ===\n");

    // Example 1: Simple refinement with heuristic critic
    println!("1. Simple Code Generation (Heuristic Critic)");
    println!("   ─────────────────────────────────────────");

    let result = kkachi("question -> code")
        .max_iterations(3)
        .min_length(20)
        .require("fn ")
        .run_sync("Write a function to add two numbers", |prompt, feedback, iter| {
            // Simulate LLM responses improving over iterations
            Ok(match iter {
                0 => "add numbers".to_string(), // Too short, missing "fn "
                1 => "fn add(a: i32) -> i32 { a }".to_string(), // Has fn, but wrong
                _ => "fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
            })
        })?;

    println!("   Result: {}", result.answer);
    println!("   Score: {:.2}, Iterations: {}, Converged: {}\n",
             result.score, result.iterations, result.converged);

    // Example 2: Checklist-based critic
    println!("2. Checklist-Based Validation");
    println!("   ─────────────────────────────");

    let result = kkachi("requirement -> function")
        .checklist()
        .check("has_fn", |s| s.contains("fn "), 0.3, "Missing function keyword")
        .check("has_return", |s| s.contains("->"), 0.2, "Missing return type")
        .check("has_doc", |s| s.contains("///"), 0.3, "Missing documentation")
        .check("no_unwrap", |s| !s.contains("unwrap()"), 0.2, "Avoid unwrap()")
        .max_iterations(4)
        .run_sync("Write a safe parse function", |_, feedback, iter| {
            Ok(match iter {
                0 => "parse".to_string(),
                1 => "fn parse() -> i32 { 42 }".to_string(),
                2 => "/// Parse\nfn parse() -> i32 { s.parse().unwrap() }".to_string(),
                _ => "/// Parse a string to integer safely.\nfn parse(s: &str) -> Option<i32> { s.parse().ok() }".to_string(),
            })
        })?;

    println!("   Result: {}", result.answer.lines().collect::<Vec<_>>().join(" "));
    println!("   Score: {:.2}, Iterations: {}\n", result.score, result.iterations);

    // Example 3: Async LLM with mock
    println!("3. Async LLM (MockLLM)");
    println!("   ────────────────────");

    let mock_llm = MockLLM::new(vec![
        "Hello".to_string(),
        "Hello, World!".to_string(),
        "fn greet() -> &'static str { \"Hello, World!\" }".to_string(),
    ]);

    let result = kkachi("task -> code")
        .max_iterations(3)
        .min_length(30)
        .run("Write a greeting function", &mock_llm)
        .await?;

    println!("   Result: {}", result.answer);
    println!("   Iterations: {}\n", result.iterations);

    // Example 4: Few-shot with examples
    println!("4. Few-Shot Learning");
    println!("   ─────────────────");

    let result = kkachi("question -> answer")
        .few_shot(2)
        .seed_examples(&[
            ("parsing", "Use serde for JSON: serde_json::from_str(s)"),
            ("http", "Use reqwest: reqwest::get(url).await?"),
            ("files", "Use std::fs: fs::read_to_string(path)?"),
        ])
        .max_iterations(2)
        .run_sync("How do I make HTTP requests?", |prompt, _, _| {
            // The prompt contains relevant examples
            if prompt.contains("http") || prompt.contains("reqwest") {
                Ok("Use reqwest: let resp = reqwest::get(url).await?;".to_string())
            } else {
                Ok("Use HTTP library".to_string())
            }
        })?;

    println!("   Result: {}", result.answer);
    println!("   Score: {:.2}\n", result.score);

    // Example 5: With iteration callback
    println!("5. Progress Tracking");
    println!("   ─────────────────");

    let result = kkachi("task -> solution")
        .max_iterations(3)
        .min_length(10)
        .on_iteration(|iter, score, feedback| {
            print!("   Iter {}: score={:.2}", iter, score);
            if let Some(fb) = feedback {
                print!(" ({})", fb);
            }
            println!();
        })
        .run_sync("Solve the problem", |_, _, iter| {
            Ok(match iter {
                0 => "try".to_string(),
                1 => "solution attempt".to_string(),
                _ => "complete working solution".to_string(),
            })
        })?;

    println!("   Final: {} (score: {:.2})\n", result.answer, result.score);

    // Example 6: Diff visualization
    println!("6. Diff Visualization");
    println!("   ─────────────────────");

    let result = kkachi("task -> code")
        .max_iterations(3)
        .show_diff()
        .min_length(15)
        .run_sync("Improve the code", |_, _, iter| {
            Ok(match iter {
                0 => "fn foo() {}".to_string(),
                1 => "fn foo() { println!(\"hello\"); }".to_string(),
                _ => "fn foo() { println!(\"hello, world!\"); }".to_string(),
            })
        })?;

    println!("   Final: {}\n", result.answer);

    // Example 7: FnLLM for custom async logic
    println!("7. Custom Async LLM (FnLLM)");
    println!("   ─────────────────────────");

    let custom_llm = FnLLM(|prompt: &str| {
        let prompt_preview = prompt[..20.min(prompt.len())].to_string();
        async move {
            // Simulate API call
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok::<String, Error>(format!("Response to: {}", prompt_preview))
        }
    });

    let result = kkachi("input -> output")
        .max_iterations(1)
        .no_critic()
        .run("Generate something", &custom_llm)
        .await?;

    println!("   Result: {}\n", result.answer);

    // Summary of API features
    println!("=== API Summary ===");
    println!("
The declarative API provides:

  Builder Methods:
    .domain(name)           - Set domain namespace
    .max_iterations(n)      - Maximum refinement iterations
    .until_score(threshold) - Convergence score threshold

  Critics:
    .min_length(n)          - Require minimum length
    .max_length(n)          - Require maximum length
    .require(pattern)       - Require substring
    .forbid(pattern)        - Forbid substring
    .checklist()            - Start checklist critic
    .check(name, fn, w, fb) - Add checklist item
    .critic_rust()          - Use Rust compiler
    .critic_python()        - Use Python linter
    .critic(custom)         - Custom critic

  Few-Shot:
    .few_shot(k)            - Enable k-shot retrieval
    .seed_examples(...)     - Add example pairs

  HITL:
    .hitl_every(n)          - Review every n iterations
    .hitl_on_completion()   - Review at end
    .reviewer(r)            - Set custom reviewer

  Callbacks:
    .on_iteration(fn)       - Called each iteration
    .on_diff(fn)            - Called with diff
    .show_diff()            - Print diffs
    .verbose()              - Enable verbose output

  Execution:
    .run_sync(q, gen)       - Synchronous execution
    .run(q, llm).await      - Async with LLM trait
");

    Ok(())
}
