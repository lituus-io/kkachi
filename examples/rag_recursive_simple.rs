// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Simple RAG + Recursive Pipeline Example
//!
//! This example shows how to:
//! 1. Use `memory()` for RAG context retrieval
//! 2. Run recursive refinement with the `checks()` validator
//! 3. Store successful results for future retrieval
//!
//! Run with: cargo run --example rag_recursive_simple

use kkachi::error::Result;
use kkachi::recursive::{
    checks, memory, refine, IterativeMockLlm, Recall,
};

/// Build a prompt with RAG examples.
fn build_rag_prompt(question: &str, examples: &[Recall]) -> String {
    let mut prompt = String::new();

    if !examples.is_empty() {
        prompt.push_str("=== Relevant Examples ===\n\n");
        for (i, ex) in examples.iter().enumerate() {
            prompt.push_str(&format!(
                "Example {} (relevance: {:.0}%):\n",
                i + 1,
                ex.score * 100.0
            ));
            prompt.push_str(&ex.content);
            prompt.push_str("\n\n");
        }
        prompt.push_str("=========================\n\n");
    }

    prompt.push_str(&format!("Question: {}\n\nAnswer:", question));
    prompt
}

/// Simulates an LLM generating progressively better code.
fn mock_llm_responses() -> [&'static str; 4] {
    [
        // First attempt: basic, missing error handling
        r#"fn parse_config(path: &str) -> Config {
    let content = std::fs::read_to_string(path).unwrap();
    toml::from_str(&content).unwrap()
}"#,
        // Second attempt: added Result, still has unwrap
        r#"use std::fs;

fn parse_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path).unwrap();
    Ok(toml::from_str(&content)?)
}"#,
        // Third attempt: proper error handling
        r#"use std::fs;
use std::error::Error;

fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config = toml::from_str(&content)?;
    Ok(config)
}"#,
        // Final: with documentation
        r#"use std::fs;
use std::error::Error;

/// Parses a TOML configuration file from the given path.
fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config = toml::from_str(&content)?;
    Ok(config)
}"#,
    ]
}

fn main() -> Result<()> {
    println!("=== RAG + Recursive Refinement Pipeline ===\n");

    // =========================================================
    // PHASE 1: Initialize RAG Memory Store
    // =========================================================
    println!("Phase 1: Setting up RAG memory store...\n");

    let mut rag_store = memory();

    // Seed with training examples (from past successful refinements)
    let seed_examples = vec![
        (
            "rust:json",
            r#"Q: How to parse JSON in Rust?

A:
use serde_json::{Value, Error};

/// Parses a JSON string into a dynamic value.
fn parse_json(s: &str) -> Result<Value, Error> {
    serde_json::from_str(s)
}

[Score: 1.0, Iterations: 1]"#,
        ),
        (
            "rust:file",
            r#"Q: How to read a file in Rust?

A:
use std::fs;
use std::io;

/// Reads the entire contents of a file as a string.
fn read_file(path: &str) -> io::Result<String> {
    fs::read_to_string(path)
}

[Score: 1.0, Iterations: 1]"#,
        ),
        (
            "rust:yaml",
            r#"Q: How to parse YAML in Rust?

A:
use serde_yaml::{Value, Error};

/// Parses a YAML string into a dynamic value.
fn parse_yaml(s: &str) -> Result<Value, Error> {
    serde_yaml::from_str(s)
}

[Score: 0.95, Iterations: 2]"#,
        ),
    ];

    for (id, content) in &seed_examples {
        rag_store.add_tagged(id, content);
    }

    println!("  Loaded {} examples into RAG store", seed_examples.len());

    // =========================================================
    // PHASE 2: Retrieve Context for New Question
    // =========================================================
    let question = "How do I parse a TOML configuration file in Rust?";

    println!("\nPhase 2: Retrieving RAG context...\n");
    println!("  Question: \"{}\"", question);

    let k = 2; // Retrieve top-2 similar examples
    let similar = rag_store.search(question, k);

    println!("  Found {} relevant examples:", similar.len());
    for (i, ex) in similar.iter().enumerate() {
        println!(
            "    {}. (similarity: {:.1}%)",
            i + 1,
            ex.score * 100.0
        );
    }

    // Build the prompt with RAG context
    let _prompt = build_rag_prompt(question, &similar);
    println!("\n  Built prompt with RAG context");

    // =========================================================
    // PHASE 3: Set Up Validator using checks()
    // =========================================================
    println!("\nPhase 3: Setting up code quality validator...\n");

    // Create a validator using the checks() builder
    let validator = checks()
        .require("fn ")           // Has function definition
        .require("use ")          // Has import statement
        .require("Result")        // Has error handling
        .require("///")           // Has documentation
        .forbid(".unwrap()");     // No panicking unwrap

    println!("  Validator checks: function, imports, error handling, docs, no unwrap");

    // =========================================================
    // PHASE 4: Run Recursive Refinement Loop
    // =========================================================
    println!("\nPhase 4: Running recursive refinement...\n");

    let responses = mock_llm_responses();

    // Create an iterative mock LLM
    let llm = IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    });

    // Run the refinement loop using the new API
    let result = refine(&llm, question)
        .validate(validator)
        .max_iter(5)
        .target(1.0)
        .go_full()?;

    println!("  === Refinement Complete ===");
    println!("  Final Score: {:.0}%", result.score * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.score >= 1.0);

    // =========================================================
    // PHASE 5: Store Successful Result Back to RAG
    // =========================================================
    println!("\nPhase 5: Storing result to RAG...\n");

    if result.score >= 0.8 {
        // Format as learning example and store
        let example_content = format!(
            "Q: {}\n\nA:\n{}\n\n[Score: {:.2}, Iterations: {}]",
            question, result.output, result.score, result.iterations
        );

        let example_id = format!("rust:toml_{}", result.context_id);
        rag_store.add_tagged(&example_id, &example_content);

        println!("  Stored: {}", example_id);
    }

    // =========================================================
    // PHASE 6: Verify Future Retrieval
    // =========================================================
    println!("\nPhase 6: Testing future retrieval...\n");

    let future_question = "parse toml config rust";
    let future_results = rag_store.search(future_question, 3);

    println!("  Query: \"{}\"", future_question);
    println!("  Top results:");
    for (i, r) in future_results.iter().enumerate() {
        // Show first 60 chars of content
        let preview: String = r
            .content
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(60)
            .collect();
        println!(
            "    {}. ({:.0}%) - {}",
            i + 1,
            r.score * 100.0,
            preview
        );
    }

    // =========================================================
    // Print Final Answer
    // =========================================================
    println!("\n=== Final Generated Code ===\n");
    println!("{}", result.output);
    println!("\n=== Pipeline Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_prompt_building() {
        let examples = vec![
            Recall {
                content: "Example 1".to_string(),
                score: 0.9,
            },
            Recall {
                content: "Example 2".to_string(),
                score: 0.8,
            },
        ];

        let prompt = build_rag_prompt("Test question?", &examples);

        assert!(prompt.contains("Example 1"));
        assert!(prompt.contains("Example 2"));
        assert!(prompt.contains("Test question?"));
        assert!(prompt.contains("90%")); // relevance score
    }

    #[test]
    fn test_mock_llm_improves() {
        let responses = mock_llm_responses();

        // Later iterations should have better code
        assert!(responses[0].contains(".unwrap()"));
        assert!(!responses[3].contains(".unwrap()"));
        assert!(responses[3].contains("///"));
    }
}
