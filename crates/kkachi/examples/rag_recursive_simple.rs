// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Simple RAG + Recursive Pipeline Example
//!
//! This example shows how to:
//! 1. Use InMemoryVectorStore for RAG context retrieval
//! 2. Run recursive refinement with DSPy-style critics
//! 3. Store successful results for future retrieval
//!
//! Run with: cargo run --example rag_recursive_simple

use kkachi::error::Result;
use kkachi::recursive::{
    // Critics
    ChecklistCritic,
    Critic,
    HashEmbedder,
    // Vector store (in-memory, no feature flag needed)
    InMemoryVectorStore,
    RecursiveConfig,
    // Recursive runner
    StandaloneRunner,
    TrainingExample,
    VectorSearchResult,
    VectorStore,
};
use kkachi::str_view::StrView;

/// Build a prompt with RAG examples.
fn build_rag_prompt(question: &str, examples: &[VectorSearchResult]) -> String {
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
fn mock_llm_codegen(iteration: u32, _feedback: Option<&str>) -> String {
    match iteration {
        0 => {
            // First attempt: basic, missing error handling
            r#"fn parse_config(path: &str) -> Config {
    let content = std::fs::read_to_string(path).unwrap();
    toml::from_str(&content).unwrap()
}"#
            .to_string()
        }
        1 => {
            // Second attempt: added Result, still has unwrap
            r#"use std::fs;

fn parse_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path).unwrap();
    Ok(toml::from_str(&content)?)
}"#
            .to_string()
        }
        2 => {
            // Third attempt: proper error handling
            r#"use std::fs;
use std::error::Error;

fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config = toml::from_str(&content)?;
    Ok(config)
}"#
            .to_string()
        }
        _ => {
            // Final: with documentation
            r#"use std::fs;
use std::error::Error;

/// Parses a TOML configuration file from the given path.
fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config = toml::from_str(&content)?;
    Ok(config)
}"#
            .to_string()
        }
    }
}

fn main() -> Result<()> {
    println!("=== RAG + Recursive Refinement Pipeline ===\n");

    // =========================================================
    // PHASE 1: Initialize RAG Vector Store
    // =========================================================
    println!("Phase 1: Setting up RAG vector store...\n");

    let embedder = HashEmbedder::new(64);
    let mut rag_store = InMemoryVectorStore::new(embedder);

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

    for (id, content) in seed_examples {
        rag_store.add(id, content);
    }

    println!("  Loaded {} examples into RAG store", rag_store.len());

    // =========================================================
    // PHASE 2: Retrieve Context for New Question
    // =========================================================
    let question = "How do I parse a TOML configuration file in Rust?";

    println!("\nPhase 2: Retrieving RAG context...\n");
    println!("  Question: \"{}\"", question);

    let k = 2; // Retrieve top-2 similar examples
    let similar = rag_store.search_text(question, k);

    println!("  Found {} relevant examples:", similar.len());
    for (i, ex) in similar.iter().enumerate() {
        println!(
            "    {}. {} (similarity: {:.1}%)",
            i + 1,
            ex.id,
            ex.score * 100.0
        );
    }

    // Build the prompt with RAG context
    let prompt = build_rag_prompt(question, &similar);
    println!("\n  Built prompt with {} chars of context", prompt.len());

    // =========================================================
    // PHASE 3: Set Up DSPy-Style Critic
    // =========================================================
    println!("\nPhase 3: Setting up code quality critic...\n");

    // Checklist critic evaluates multiple quality aspects
    let critic = ChecklistCritic::new()
        .add_check(
            "has_function",
            |s| s.contains("fn "),
            0.2,
            "Missing function definition",
        )
        .add_check(
            "has_import",
            |s| s.contains("use "),
            0.2,
            "Missing use/import statement",
        )
        .add_check(
            "handles_errors",
            |s| s.contains("Result") || s.contains("?"),
            0.2,
            "No error handling (should use Result and ?)",
        )
        .add_check(
            "has_docs",
            |s| s.contains("///"),
            0.2,
            "Missing documentation (/// doc comments)",
        )
        .add_check(
            "no_unwrap",
            |s| !s.contains(".unwrap()"),
            0.2,
            "Uses .unwrap() which can panic",
        );

    println!("  Critic checks: function, imports, error handling, docs, no unwrap");

    // =========================================================
    // PHASE 4: Run Recursive Refinement Loop
    // =========================================================
    println!("\nPhase 4: Running recursive refinement...\n");

    let config = RecursiveConfig {
        max_iterations: 5,
        score_threshold: 1.0,
        ..Default::default()
    };

    let runner = StandaloneRunner::with_config(&critic, "rust", config);

    // Run the refinement loop
    let result = runner.refine(question, |iteration, feedback| {
        println!("  Iteration {}:", iteration);

        if let Some(fb) = feedback {
            println!("    Feedback: {}", fb);
        }

        // In production: Call LLM with prompt + feedback
        // Here we use a mock that improves each iteration
        let code = mock_llm_codegen(iteration, feedback);

        // Evaluate with critic for logging (clone to avoid borrow issues)
        {
            let state = kkachi::recursive::RecursiveState::new();
            let eval = critic.evaluate(StrView::new(&code), &state);
            println!("    Score: {:.0}%", eval.score * 100.0);
        }

        Ok(code)
    })?;

    println!("\n  === Refinement Complete ===");
    println!("  Final Score: {:.0}%", result.score * 100.0);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.score >= 1.0);

    // =========================================================
    // PHASE 5: Store Successful Result Back to RAG
    // =========================================================
    println!("\nPhase 5: Storing result to RAG...\n");

    if result.score >= 0.8 {
        let example = TrainingExample {
            id: format!("rust:toml_{}", result.context_id),
            question: question.to_string(),
            answer: result.answer.clone(),
            score: result.score,
            iterations: result.iterations,
            domain: "rust".to_string(),
            error_corrections: result.error_corrections.clone(),
        };

        // Format as learning example and store
        rag_store.add(example.id.clone(), example.as_learning_example());

        println!("  Stored: {}", example.id);
        println!("  RAG store now has {} examples", rag_store.len());
    }

    // =========================================================
    // PHASE 6: Verify Future Retrieval
    // =========================================================
    println!("\nPhase 6: Testing future retrieval...\n");

    let future_question = "parse toml config rust";
    let future_results = rag_store.search_text(future_question, 3);

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
            "    {}. {} ({:.0}%) - {}",
            i + 1,
            r.id,
            r.score * 100.0,
            preview
        );
    }

    // =========================================================
    // Print Final Answer
    // =========================================================
    println!("\n=== Final Generated Code ===\n");
    println!("{}", result.answer);
    println!("\n=== Pipeline Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_prompt_building() {
        let examples = vec![
            VectorSearchResult::new("id1".to_string(), "Example 1".to_string(), 0.9),
            VectorSearchResult::new("id2".to_string(), "Example 2".to_string(), 0.8),
        ];

        let prompt = build_rag_prompt("Test question?", &examples);

        assert!(prompt.contains("Example 1"));
        assert!(prompt.contains("Example 2"));
        assert!(prompt.contains("Test question?"));
        assert!(prompt.contains("90%")); // relevance score
    }

    #[test]
    fn test_mock_llm_improves() {
        let v0 = mock_llm_codegen(0, None);
        let v3 = mock_llm_codegen(3, None);

        // Later iterations should have better code
        assert!(v0.contains(".unwrap()"));
        assert!(!v3.contains(".unwrap()"));
        assert!(v3.contains("///"));
    }
}
