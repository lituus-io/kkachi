// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Example: RAG + Recursive Refinement Pipeline
//!
//! This example demonstrates a full pipeline that:
//! 1. Uses DuckDB vector store for initial context retrieval (few-shot examples)
//! 2. Applies DSPy-style modules (ChainOfThought, BestOfN)
//! 3. Runs recursive refinement with CLI-based critics
//! 4. Stores successful results back to RAG for future retrieval
//!
//! Run with:
//! ```bash
//! cargo run --example rag_recursive_pipeline --features storage
//! ```

#[cfg(feature = "storage")]
mod pipeline {
    use kkachi::error::Result;
    use kkachi::recursive::{
        // Critics
        ChecklistCritic,
        Critic,
        // RAG and vector store
        DuckDBVectorStore,
        HashEmbedder,
        HeuristicCritic,
        MutableVectorStore,
        // Recursive refinement
        RecursiveConfig,
        RefinementResult,
        SimilarityWeights,
        // Runner
        StandaloneRunner,
        TrainingExample,
        VectorStore,
    };
    // Declarative API
    use kkachi::recursive::{FewShotConfig, Kkachi};

    /// Simulated LLM response generator.
    /// In production, this would call an actual LLM API.
    struct MockLLM {
        iteration_responses: Vec<String>,
    }

    impl MockLLM {
        fn new(responses: Vec<String>) -> Self {
            Self {
                iteration_responses: responses,
            }
        }

        fn generate(&self, iteration: usize, context: &str, feedback: Option<&str>) -> String {
            // In production: Call OpenAI/Anthropic/etc with context and feedback
            println!("  [LLM] Iteration {}", iteration);
            println!("  [LLM] Context: {} chars", context.len());
            if let Some(fb) = feedback {
                println!("  [LLM] Feedback: {}", fb);
            }

            self.iteration_responses
                .get(iteration)
                .cloned()
                .unwrap_or_else(|| self.iteration_responses.last().cloned().unwrap_or_default())
        }
    }

    /// Build few-shot prompt from RAG examples.
    fn build_prompt_with_examples(
        question: &str,
        examples: &[kkachi::recursive::VectorSearchResult],
    ) -> String {
        let mut prompt = String::new();

        if !examples.is_empty() {
            prompt.push_str("Here are some relevant examples:\n\n");
            for (i, example) in examples.iter().enumerate() {
                prompt.push_str(&format!(
                    "--- Example {} (similarity: {:.2}) ---\n",
                    i + 1,
                    example.score
                ));
                prompt.push_str(&example.content);
                prompt.push_str("\n\n");
            }
            prompt.push_str("---\n\n");
        }

        prompt.push_str("Now answer the following:\n\n");
        prompt.push_str(question);

        prompt
    }

    /// Full RAG + Recursive pipeline.
    pub fn run_pipeline() -> Result<()> {
        println!("=== RAG + Recursive Refinement Pipeline ===\n");

        // =========================================================
        // Step 1: Initialize DuckDB Vector Store with seed examples
        // =========================================================
        println!("Step 1: Initializing DuckDB vector store...");

        let embedder = HashEmbedder::new(64);
        let mut store = DuckDBVectorStore::in_memory(embedder.clone())?;

        // Seed with some training examples (in production, these come from past successful refinements)
        let seed_examples = vec![
            ("rust:url_parse", "Q: How to parse a URL in Rust?\n\nA:\n```rust\nuse url::Url;\n\nfn parse_url(s: &str) -> Option<String> {\n    Url::parse(s).ok().and_then(|u| u.host_str().map(|h| h.to_string()))\n}\n```\n[Score: 0.95, Iterations: 2]"),
            ("rust:json_parse", "Q: How to parse JSON in Rust?\n\nA:\n```rust\nuse serde_json::Value;\n\nfn parse_json(s: &str) -> Result<Value, serde_json::Error> {\n    serde_json::from_str(s)\n}\n```\n[Score: 1.0, Iterations: 1]"),
            ("rust:file_read", "Q: How to read a file in Rust?\n\nA:\n```rust\nuse std::fs;\n\nfn read_file(path: &str) -> std::io::Result<String> {\n    fs::read_to_string(path)\n}\n```\n[Score: 1.0, Iterations: 1]"),
            ("rust:http_get", "Q: How to make HTTP GET request in Rust?\n\nA:\n```rust\nuse reqwest;\n\nasync fn fetch(url: &str) -> Result<String, reqwest::Error> {\n    reqwest::get(url).await?.text().await\n}\n```\n[Score: 0.9, Iterations: 3]"),
        ];

        for (id, content) in seed_examples {
            store.add(id, content)?;
        }

        println!("  Loaded {} seed examples into vector store\n", store.len());

        // =========================================================
        // Step 2: Retrieve relevant context for the question
        // =========================================================
        let question = "How do I parse TOML configuration in Rust?";
        println!("Step 2: Retrieving context for question:");
        println!("  \"{}\"", question);

        let few_shot_k = 3;
        let similar_examples = store.search_text(question, few_shot_k);

        println!("  Found {} relevant examples:", similar_examples.len());
        for (i, example) in similar_examples.iter().enumerate() {
            println!(
                "    {}. {} (score: {:.3})",
                i + 1,
                example.id,
                example.score
            );
        }
        println!();

        // =========================================================
        // Step 3: Build prompt with few-shot examples (DSPy-style)
        // =========================================================
        println!("Step 3: Building few-shot prompt...");

        let prompt = build_prompt_with_examples(question, &similar_examples);
        println!("  Prompt length: {} chars\n", prompt.len());

        // =========================================================
        // Step 4: Set up recursive refinement with critics
        // =========================================================
        println!("Step 4: Setting up recursive refinement...");

        // Create a checklist critic for Rust code quality
        let critic = ChecklistCritic::new()
            .add_check(
                "has function",
                |s| s.contains("fn "),
                0.2,
                "Missing function definition",
            )
            .add_check(
                "has use statement",
                |s| s.contains("use "),
                0.2,
                "Missing use/import statement",
            )
            .add_check(
                "handles errors",
                |s| s.contains("Result") || s.contains("Option"),
                0.2,
                "No error handling",
            )
            .add_check(
                "has docstring",
                |s| s.contains("///") || s.contains("//"),
                0.2,
                "Missing documentation",
            )
            .add_check(
                "no unwrap",
                |s| !s.contains(".unwrap()"),
                0.2,
                "Uses unsafe unwrap()",
            );

        let config = RecursiveConfig {
            max_iterations: 5,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "rust", config);

        // Simulated LLM responses that improve over iterations
        let mock_llm = MockLLM::new(vec![
            // Iteration 0: Basic attempt, missing error handling
            r#"fn parse_toml(s: &str) -> toml::Value {
    toml::from_str(s).unwrap()
}"#
            .to_string(),
            // Iteration 1: Added error handling, still missing import
            r#"fn parse_toml(s: &str) -> Result<toml::Value, toml::de::Error> {
    toml::from_str(s)
}"#
            .to_string(),
            // Iteration 2: Added import, missing docs
            r#"use toml;

fn parse_toml(s: &str) -> Result<toml::Value, toml::de::Error> {
    toml::from_str(s)
}"#
            .to_string(),
            // Iteration 3: Complete with docs
            r#"use toml;

/// Parses a TOML string into a structured value.
fn parse_toml(s: &str) -> Result<toml::Value, toml::de::Error> {
    toml::from_str(s)
}"#
            .to_string(),
        ]);

        // =========================================================
        // Step 5: Run recursive refinement loop
        // =========================================================
        println!("Step 5: Running recursive refinement...\n");

        let mut iteration_idx = 0;
        let result = runner.refine(question, |iteration, feedback| {
            // Build context with RAG examples + any feedback
            let mut context = prompt.clone();
            if let Some(fb) = feedback {
                context.push_str("\n\nPrevious attempt had issues:\n");
                context.push_str(fb);
                context.push_str("\n\nPlease fix these issues.");
            }

            let response = mock_llm.generate(iteration_idx, &context, feedback);
            iteration_idx += 1;

            Ok(response)
        })?;

        println!("\n  Refinement complete!");
        println!("  Final score: {:.2}", result.score);
        println!("  Iterations: {}", result.iterations);
        println!("  Answer:\n{}\n", result.answer);

        // =========================================================
        // Step 6: Store successful result back to RAG
        // =========================================================
        if result.score >= 0.8 {
            println!("Step 6: Storing successful result to RAG...");

            let example = TrainingExample {
                id: format!("rust:{}", result.context_id),
                question: question.to_string(),
                answer: result.answer.clone(),
                score: result.score,
                iterations: result.iterations,
                domain: "rust".to_string(),
                error_corrections: result.error_corrections.clone(),
            };

            store.add(example.id.clone(), example.as_learning_example())?;

            println!("  Added example: {}", example.id);
            println!("  Vector store now has {} examples\n", store.len());
        }

        // =========================================================
        // Step 7: Verify future retrieval finds the new example
        // =========================================================
        println!("Step 7: Verifying future retrieval...");

        let future_query = "How to parse a TOML config file?";
        let future_results = store.search_text(future_query, 3);

        println!("  Query: \"{}\"", future_query);
        println!("  Results:");
        for (i, r) in future_results.iter().enumerate() {
            let preview: String = r.content.chars().take(50).collect();
            println!(
                "    {}. {} (score: {:.3}) - {}...",
                i + 1,
                r.id,
                r.score,
                preview
            );
        }

        println!("\n=== Pipeline Complete ===");

        Ok(())
    }

    /// Alternative: Using the Kkachi declarative API with RAG integration.
    pub fn run_with_kkachi_api() -> Result<()> {
        println!("\n=== Using Kkachi Declarative API ===\n");

        // Initialize vector store
        let embedder = HashEmbedder::new(64);
        let mut store = DuckDBVectorStore::in_memory(embedder)?;

        // Seed examples
        store.add("ex1", "Q: Parse JSON\nA: use serde_json")?;
        store.add("ex2", "Q: Read file\nA: use std::fs")?;

        // Retrieve context
        let question = "How to parse YAML in Rust?";
        let examples = store.search_text(question, 2);

        println!("Retrieved {} examples for context", examples.len());

        // Build context string
        let context: String = examples
            .iter()
            .map(|e| e.content.clone())
            .collect::<Vec<_>>()
            .join("\n---\n");

        // Run with Kkachi API
        let result = Kkachi::refine("question -> code")
            .domain("rust")
            .max_iterations(5)
            .until_score(0.9)
            // Enable DSPy-style features
            .with_chain_of_thought()
            .with_best_of_n(3)
            // Configure few-shot
            .few_shot(FewShotConfig {
                k: 3,
                min_similarity: 0.5,
                include_in_prompt: true,
                as_demonstrations: true,
                refresh_per_iteration: false,
            })
            // Configure similarity
            .similarity_weights(SimilarityWeights {
                embedding: 0.50,
                keyword: 0.25,
                metadata: 0.15,
                hierarchy: 0.10,
            })
            .auto_condense(true)
            .semantic_cache(true)
            // Use always-pass critic for demo (in production: use .validate(pipeline))
            .critic_always_pass()
            .run(question, |iteration, feedback| {
                // Include RAG context in the generation
                let prompt = if iteration == 0 {
                    format!("{}\n\nNow answer: {}", context, question)
                } else {
                    format!("Fix based on feedback: {:?}\n\n{}", feedback, question)
                };

                // Mock LLM response
                Ok(format!("use serde_yaml;\n\n/// Parse YAML\nfn parse(s: &str) -> Result<serde_yaml::Value, serde_yaml::Error> {{\n    serde_yaml::from_str(s)\n}}"))
            });

        println!(
            "Result: {} iterations, score {:.2}",
            result.iterations, result.score
        );
        println!("Answer:\n{}", result.answer);

        // Store back to RAG for future use
        if result.score >= 0.8 {
            let doc = format!(
                "Q: {}\nA:\n{}\n[Score: {:.2}]",
                question, result.answer, result.score
            );
            store.add(format!("rust:{}", result.context_id), doc)?;
            println!(
                "\nStored successful result to RAG. Total examples: {}",
                store.len()
            );
        }

        Ok(())
    }

    /// Chain of Thought + RAG example
    pub fn run_chain_of_thought_with_rag() -> Result<()> {
        println!("\n=== Chain of Thought with RAG ===\n");

        // Initialize store
        let embedder = HashEmbedder::new(64);
        let mut store = DuckDBVectorStore::in_memory(embedder)?;

        // Add reasoning examples
        store.add(
            "cot:math1",
            r#"
Q: What is 25% of 80?

Reasoning:
1. 25% means 25 per 100, or 0.25 as a decimal
2. To find 25% of 80, multiply: 0.25 × 80
3. 0.25 × 80 = 20

A: 25% of 80 is 20
"#,
        )?;

        store.add(
            "cot:math2",
            r#"
Q: If a shirt costs $40 and is 30% off, what's the sale price?

Reasoning:
1. 30% discount means you pay 70% of the original price
2. 70% as decimal = 0.70
3. Sale price = $40 × 0.70 = $28

A: The sale price is $28
"#,
        )?;

        // New question
        let question = "If I have $100 and spend 15%, how much do I have left?";

        // Retrieve similar reasoning examples
        let examples = store.search_text(question, 2);
        println!("Retrieved {} CoT examples", examples.len());

        // Build prompt with reasoning examples
        let mut prompt = String::from("Use step-by-step reasoning like these examples:\n\n");
        for ex in &examples {
            prompt.push_str(&ex.content);
            prompt.push_str("\n---\n");
        }
        prompt.push_str(&format!("\nNow solve:\nQ: {}\n\nReasoning:", question));

        // Critic that checks for reasoning steps
        let critic = HeuristicCritic::new()
            .min_length(50)
            .require("1.")
            .require("A:");

        let config = RecursiveConfig {
            max_iterations: 3,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "math", config);

        let result = runner.refine(question, |iteration, feedback| {
            if iteration == 0 {
                Ok(r#"
1. 15% of $100 means I spend $15
2. Amount left = $100 - $15 = $85

A: I have $85 left
"#
                .to_string())
            } else {
                Ok(format!("Refined answer based on: {:?}", feedback))
            }
        })?;

        println!(
            "Score: {:.2}, Iterations: {}",
            result.score, result.iterations
        );
        println!("Answer:\n{}", result.answer);

        Ok(())
    }
}

#[cfg(feature = "storage")]
fn main() {
    use pipeline::*;

    println!("\n{}\n", "=".repeat(60));

    // Run the main pipeline
    if let Err(e) = run_pipeline() {
        eprintln!("Pipeline error: {}", e);
    }

    println!("\n{}\n", "=".repeat(60));

    // Run with Kkachi API
    if let Err(e) = run_with_kkachi_api() {
        eprintln!("Kkachi API error: {}", e);
    }

    println!("\n{}\n", "=".repeat(60));

    // Run Chain of Thought example
    if let Err(e) = run_chain_of_thought_with_rag() {
        eprintln!("CoT error: {}", e);
    }
}

#[cfg(not(feature = "storage"))]
fn main() {
    println!("This example requires the 'storage' feature.");
    println!("Run with: cargo run --example rag_recursive_pipeline --features storage");
}
