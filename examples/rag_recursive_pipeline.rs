// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! RAG Recursive Pipeline Example (with Optional Persistent Storage)
//!
//! This example demonstrates a complete RAG + recursive refinement pipeline:
//! 1. Persistent memory storage (DuckDB via `storage` feature)
//! 2. Semantic search for few-shot examples
//! 3. Recursive refinement with validators
//! 4. Automatic learning from successful outputs
//!
//! ## Features
//!
//! - **In-memory mode** (default): Fast, no dependencies
//! - **Persistent mode** (`--features storage`): DuckDB-backed storage
//!
//! ## Running
//!
//! ```bash
//! # In-memory mode (default)
//! cargo run --example rag_recursive_pipeline
//!
//! # With persistent storage
//! cargo run --example rag_recursive_pipeline --features storage
//! ```

use kkachi::error::Result;
use kkachi::recursive::{checks, memory, refine, IterativeMockLlm, Recall};

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   RAG Recursive Pipeline Example");
    println!("═══════════════════════════════════════════════════════════════════\n");

    run_pipeline()?;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   Pipeline Complete!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

fn run_pipeline() -> Result<()> {
    // =========================================================
    // PHASE 1: Initialize Memory Store
    // =========================================================
    println!("Phase 1: Initializing memory store...\n");

    #[cfg(any(feature = "storage", feature = "storage-bundled"))]
    let mut mem = {
        let path = "/tmp/kkachi_rag_example.db";
        println!("  Using persistent storage: {}", path);
        memory().persist(path)?
    };

    #[cfg(not(any(feature = "storage", feature = "storage-bundled")))]
    let mut mem = {
        println!("  Using in-memory storage");
        println!("  (Run with --features storage for persistent DuckDB storage)");
        memory()
    };

    // =========================================================
    // PHASE 2: Seed with Training Examples
    // =========================================================
    println!("\nPhase 2: Seeding with training examples...\n");

    let seed_examples = vec![
        ("rust:error_handling", r#"Q: How do I handle errors in Rust?

A:
use std::error::Error;

/// A function that might fail.
fn risky_operation() -> Result<String, Box<dyn Error>> {
    let data = std::fs::read_to_string("config.txt")?;
    Ok(data)
}

[Score: 1.0, Iterations: 2]"#),
        ("rust:json_parsing", r#"Q: How do I parse JSON in Rust?

A:
use serde::{Deserialize, Serialize};
use serde_json;

/// A configuration structure.
#[derive(Deserialize, Serialize)]
struct Config {
    name: String,
    port: u16,
}

/// Parse a JSON string into Config.
fn parse_config(json: &str) -> Result<Config, serde_json::Error> {
    serde_json::from_str(json)
}

[Score: 1.0, Iterations: 1]"#),
        ("rust:file_io", r#"Q: How do I read and write files in Rust?

A:
use std::fs;
use std::io::{self, Write};

/// Read entire file contents.
fn read_file(path: &str) -> io::Result<String> {
    fs::read_to_string(path)
}

/// Write string to file.
fn write_file(path: &str, content: &str) -> io::Result<()> {
    fs::write(path, content)
}

[Score: 1.0, Iterations: 1]"#),
    ];

    if mem.is_empty() {
        for (tag, content) in &seed_examples {
            mem.add_tagged(tag, content);
        }
        println!("  Seeded {} examples", seed_examples.len());
    } else {
        println!("  Memory already has {} examples", mem.len());
    }

    // =========================================================
    // PHASE 3: Set Up Validators
    // =========================================================
    println!("\nPhase 3: Setting up validators...\n");

    // Primary validator: code quality checks
    let code_validator = checks()
        .require("fn ")           // Has function definition
        .require("///")           // Has documentation
        .require("->")            // Has return type
        .forbid(".unwrap()")      // No panicking unwrap
        .forbid("panic!")         // No panic macros
        .min_len(50);             // Minimum length

    println!("  Code validator checks:");
    println!("    - Has function definition");
    println!("    - Has documentation (///)");
    println!("    - Has return type (->)");
    println!("    - No .unwrap() or panic!");
    println!("    - Minimum 50 characters");

    // =========================================================
    // PHASE 4: Query Memory for Similar Examples
    // =========================================================
    println!("\nPhase 4: Searching for similar examples...\n");

    let question = "How do I parse a YAML configuration file in Rust?";
    println!("  Question: \"{}\"", question);

    let similar = mem.search(question, 3);
    println!("\n  Found {} relevant examples:", similar.len());
    for (i, r) in similar.iter().enumerate() {
        let preview = r
            .content
            .lines()
            .find(|l| l.starts_with("Q:"))
            .unwrap_or(&r.content[..60.min(r.content.len())]);
        println!("    {}. ({:.0}%) {}", i + 1, r.score * 100.0, preview);
    }

    // =========================================================
    // PHASE 5: Run Recursive Refinement
    // =========================================================
    println!("\nPhase 5: Running recursive refinement...\n");

    // Mock responses that progressively improve
    let responses = [
        // Iteration 0: Missing imports and docs
        r#"fn parse_yaml(path: &str) -> Config {
    let content = std::fs::read_to_string(path).unwrap();
    serde_yaml::from_str(&content).unwrap()
}"#,
        // Iteration 1: Added Result but still has unwrap
        r#"use std::error::Error;

fn parse_yaml(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = std::fs::read_to_string(path).unwrap();
    Ok(serde_yaml::from_str(&content)?)
}"#,
        // Iteration 2: Proper error handling
        r#"use std::error::Error;

fn parse_yaml(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = std::fs::read_to_string(path)?;
    let config = serde_yaml::from_str(&content)?;
    Ok(config)
}"#,
        // Iteration 3: With documentation
        r#"use std::error::Error;

/// Parses a YAML configuration file from the given path.
///
/// # Arguments
/// * `path` - The path to the YAML file
///
/// # Returns
/// The parsed Config struct or an error
fn parse_yaml(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = std::fs::read_to_string(path)?;
    let config = serde_yaml::from_str(&content)?;
    Ok(config)
}"#,
    ];

    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    // Build RAG context
    let rag_context = build_rag_context(&similar);

    let result = refine(&llm, &format!("{}\n\n{}", question, rag_context))
        .validate(code_validator)
        .max_iter(5)
        .target(1.0)
        .go_full()?;

    println!("  Final score: {:.0}%", result.score * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Success: {}", result.score >= 1.0);

    // =========================================================
    // PHASE 6: Store Successful Result
    // =========================================================
    println!("\nPhase 6: Storing result...\n");

    if result.score >= 0.9 {
        let learning_example = format!(
            "Q: {}\n\nA:\n{}\n\n[Score: {:.2}, Iterations: {}]",
            question, result.output, result.score, result.iterations
        );
        let id = format!("rust:yaml_{}", result.context_id);
        mem.add_tagged(&id, &learning_example);
        println!("  Stored successful result: {}", id);
        println!("  Total examples in memory: {}", mem.len());
    } else {
        println!("  Score too low ({:.0}%), not storing", result.score * 100.0);
    }

    // =========================================================
    // Print Final Result
    // =========================================================
    println!("\n=== Final Generated Code ===\n");
    println!("{}", result.output);

    // =========================================================
    // Demonstrate Memory Statistics
    // =========================================================
    println!("\n=== Memory Statistics ===\n");
    println!("  Total documents: {}", mem.len());
    let tags = mem.tags();
    println!("  Tags: {:?}", tags);

    // Verify the new example can be found
    let verify_results = mem.search("yaml config", 3);
    println!("\n  Verification search for 'yaml config':");
    for (i, r) in verify_results.iter().enumerate() {
        let tag_str = r.tag.as_deref().unwrap_or("none");
        println!("    {}. {} ({:.0}%)", i + 1, tag_str, r.score * 100.0);
    }

    Ok(())
}

/// Build RAG context from search results.
fn build_rag_context(recalls: &[Recall]) -> String {
    if recalls.is_empty() {
        return String::new();
    }

    let mut context = String::from("=== Relevant Examples ===\n\n");
    for (i, r) in recalls.iter().enumerate() {
        context.push_str(&format!("Example {} (relevance: {:.0}%):\n", i + 1, r.score * 100.0));
        context.push_str(&r.content);
        context.push_str("\n\n---\n\n");
    }
    context.push_str("=== End Examples ===\n\n");
    context
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_context_building() {
        let recalls = vec![
            Recall {
                id: "test".to_string(),
                content: "Example content".to_string(),
                score: 0.95,
                tag: Some("rust".to_string()),
            },
        ];

        let context = build_rag_context(&recalls);
        assert!(context.contains("Example content"));
        assert!(context.contains("95%"));
    }

    #[test]
    fn test_empty_rag_context() {
        let context = build_rag_context(&[]);
        assert!(context.is_empty());
    }
}
