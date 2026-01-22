// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Complete DSPy-Style Pipeline with Vector Store, HITL, and Diff Visualization
//!
//! This example demonstrates the full capabilities of Kkachi:
//!
//! 1. **Vector Store Integration** - Semantic search for few-shot examples
//! 2. **DSPy-Style Modules** - ChainOfThought, BestOfN, Refine
//! 3. **Recursive Refinement** - Iterative improvement with critics
//! 4. **Diff Visualization** - See changes between iterations
//! 5. **Human-in-the-Loop** - Interactive review and editing
//! 6. **Prompt Optimization** - Bootstrap few-shot learning
//!
//! Run with:
//! ```bash
//! cargo run --example complete_dspy_pipeline
//! ```

use kkachi::diff::{DiffRenderer, DiffStyle, TextDiff};
use kkachi::error::Result;
use kkachi::hitl::{HITLConfig, ReviewDecision};
use kkachi::recursive::{
    ChecklistCritic, Critic, HashEmbedder, HeuristicCritic, InMemoryVectorStore,
    RecursiveConfig, RecursiveState, StandaloneRunner, VectorStore,
};
use kkachi::StrView;
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

const EMBEDDING_DIM: usize = 64;
const FEW_SHOT_K: usize = 3;
const MAX_ITERATIONS: u32 = 5;
const SCORE_THRESHOLD: f64 = 0.95;

// ============================================================================
// Mock LLM Client (replace with real API in production)
// ============================================================================

struct MockLLM {
    response_templates: HashMap<String, Vec<String>>,
}

impl MockLLM {
    fn new() -> Self {
        let mut templates = HashMap::new();

        // Rust code generation responses (improving over iterations)
        templates.insert(
            "rust_code".to_string(),
            vec![
                // Iteration 0: Basic, missing error handling
                r#"fn parse_config(path: &str) -> Config {
    let content = std::fs::read_to_string(path).unwrap();
    toml::from_str(&content).unwrap()
}"#
                .to_string(),
                // Iteration 1: Added Result, still has unwrap
                r#"fn parse_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let config = toml::from_str(&content).unwrap();
    Ok(config)
}"#
                .to_string(),
                // Iteration 2: Proper error handling, missing docs
                r#"use std::fs;
use std::error::Error;

fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}"#
                .to_string(),
                // Iteration 3: Complete with docs
                r#"use std::fs;
use std::error::Error;

/// Parses a TOML configuration file into a Config struct.
///
/// # Arguments
/// * `path` - Path to the configuration file
///
/// # Returns
/// * `Ok(Config)` - Successfully parsed configuration
/// * `Err` - If file cannot be read or parsed
fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}"#
                .to_string(),
            ],
        );

        // Math reasoning responses
        templates.insert(
            "math_reasoning".to_string(),
            vec![
                // Iteration 0: Direct answer, no reasoning
                "42".to_string(),
                // Iteration 1: Some reasoning
                "Let me calculate: 6 * 7 = 42\n\nThe answer is 42.".to_string(),
                // Iteration 2: Full chain of thought
                r#"Let me solve this step by step:

1. We need to find 6 multiplied by 7
2. I can think of this as 6 groups of 7
3. 6 * 7 = (5 * 7) + (1 * 7) = 35 + 7 = 42
4. Or simply: 6 * 7 = 42

Therefore, the answer is **42**."#
                    .to_string(),
            ],
        );

        Self {
            response_templates: templates,
        }
    }

    fn generate(&self, task_type: &str, iteration: usize, context: &str, feedback: Option<&str>) -> String {
        println!("\n  {} LLM Generation (iteration {})", "→".to_string(), iteration);
        println!("    Context length: {} chars", context.len());
        if let Some(fb) = feedback {
            println!("    Feedback: {}", fb.chars().take(50).collect::<String>());
        }

        self.response_templates
            .get(task_type)
            .and_then(|responses| responses.get(iteration).or(responses.last()))
            .cloned()
            .unwrap_or_else(|| format!("Response for {} iteration {}", task_type, iteration))
    }
}

// ============================================================================
// Vector Store Management
// ============================================================================

fn create_vector_store_with_examples() -> InMemoryVectorStore<HashEmbedder> {
    let embedder = HashEmbedder::new(EMBEDDING_DIM);
    let mut store = InMemoryVectorStore::new(embedder);

    // Seed with high-quality examples from previous successful refinements
    let examples = vec![
        (
            "rust:json_parse",
            r#"Q: How to parse JSON in Rust?

A:
```rust
use serde::{Deserialize, Serialize};
use serde_json::Result;

/// Parses a JSON string into the specified type.
fn parse_json<T: for<'de> Deserialize<'de>>(s: &str) -> Result<T> {
    serde_json::from_str(s)
}
```

[Score: 1.0, Iterations: 2]"#,
        ),
        (
            "rust:file_read",
            r#"Q: How to read a file in Rust?

A:
```rust
use std::fs;
use std::io::{self, Read};

/// Reads the entire contents of a file into a string.
fn read_file(path: &str) -> io::Result<String> {
    fs::read_to_string(path)
}
```

[Score: 1.0, Iterations: 1]"#,
        ),
        (
            "rust:http_request",
            r#"Q: How to make an HTTP GET request in Rust?

A:
```rust
use reqwest::Error;

/// Makes an async HTTP GET request and returns the response body.
async fn fetch_url(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}
```

[Score: 0.95, Iterations: 3]"#,
        ),
        (
            "rust:error_handling",
            r#"Q: How to handle errors properly in Rust?

A:
```rust
use std::error::Error;
use std::fmt;

/// Custom error type with context.
#[derive(Debug)]
struct AppError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}
```

[Score: 1.0, Iterations: 2]"#,
        ),
        (
            "math:percentage",
            r#"Q: What is 25% of 80?

Reasoning:
1. 25% means 25 per 100, or 0.25 as a decimal
2. To find 25% of 80, multiply: 0.25 × 80
3. 0.25 × 80 = 20

A: 25% of 80 is **20**

[Score: 1.0, Iterations: 1]"#,
        ),
    ];

    for (id, content) in examples {
        store.add(id, content);
    }

    println!("✓ Loaded {} seed examples into vector store", store.len());
    store
}

// ============================================================================
// Diff Visualization
// ============================================================================

fn show_iteration_diff(old: &str, new: &str, iteration: u32) {
    let diff = TextDiff::new(old, new);

    if !diff.has_changes() {
        println!("\n  [No changes from previous iteration]");
        return;
    }

    let renderer = DiffRenderer::new()
        .with_style(DiffStyle::Unified)
        .with_context(2);

    let stats = diff.stats();
    println!("\n  ┌─────────────────────────────────────────────────────────────┐");
    println!(
        "  │ Iteration {} → {} Changes                                        │",
        iteration - 1,
        iteration
    );
    println!(
        "  │ +{} lines added, -{} lines removed, ~{} lines changed           │",
        stats.lines_added, stats.lines_removed, stats.lines_changed
    );
    println!("  ├─────────────────────────────────────────────────────────────┤");

    // Show diff output
    let diff_output = renderer.render_text(&diff);
    for line in diff_output.lines().take(20) {
        println!("  │ {}", line);
    }

    if diff_output.lines().count() > 20 {
        println!("  │ ... ({} more lines)", diff_output.lines().count() - 20);
    }

    println!("  └─────────────────────────────────────────────────────────────┘");
}

// ============================================================================
// HITL Integration
// ============================================================================

fn create_hitl_config() -> HITLConfig {
    HITLConfig {
        enabled: true,
        interval: 2,          // Review every 2 iterations
        on_score_drop: true,  // Review if score decreases
        on_convergence: true, // Review before final acceptance
        on_first: false,
        on_keywords: vec![],  // No keyword triggers
        timeout: None,
        show_diff: true,
        diff_style: DiffStyle::Unified,
        auto_accept_timeout: None,
        skip_above_score: None,
    }
}

fn simulate_hitl_review(
    iteration: u32,
    output: &str,
    score: f64,
    prev_score: f64,
    feedback: Option<&str>,
) -> ReviewDecision {
    println!("\n  ╔═══════════════════════════════════════════════════════════════╗");
    println!("  ║                    HUMAN-IN-THE-LOOP REVIEW                   ║");
    println!("  ╠═══════════════════════════════════════════════════════════════╣");
    println!("  ║ Iteration: {}                                                  ", iteration);
    println!(
        "  ║ Score: {:.2} (previous: {:.2}, change: {:+.2})                   ",
        score,
        prev_score,
        score - prev_score
    );
    println!("  ╠═══════════════════════════════════════════════════════════════╣");
    println!("  ║ Current Output (preview):                                     ║");

    for line in output.lines().take(5) {
        println!("  ║   {}", line.chars().take(55).collect::<String>());
    }

    if output.lines().count() > 5 {
        println!("  ║   ... ({} more lines)", output.lines().count() - 5);
    }

    if let Some(fb) = feedback {
        println!("  ╠═══════════════════════════════════════════════════════════════╣");
        println!("  ║ Critic Feedback:                                              ║");
        println!("  ║   {}", fb.chars().take(55).collect::<String>());
    }

    println!("  ╠═══════════════════════════════════════════════════════════════╣");
    println!("  ║ Options:                                                       ║");
    println!("  ║   [A]ccept - Continue to next iteration                        ║");
    println!("  ║   [R]eject - Try alternative approach                          ║");
    println!("  ║   [E]dit   - Modify the output                                 ║");
    println!("  ║   [S]top   - Accept current result as final                    ║");
    println!("  ╚═══════════════════════════════════════════════════════════════╝");

    // Simulate automatic acceptance for demo
    // In production, this would prompt for user input
    if score >= 0.9 {
        println!("\n  → Auto-accepting (score >= 0.9)");
        ReviewDecision::Accept
    } else if score < prev_score {
        println!("\n  → Score dropped, simulating rejection");
        ReviewDecision::Reject
    } else {
        println!("\n  → Accepting to continue refinement");
        ReviewDecision::Accept
    }
}

// ============================================================================
// Critics for Quality Validation
// ============================================================================

fn create_rust_code_critic() -> impl Critic {
    ChecklistCritic::new()
        .add_check(
            "has_function",
            |s| s.contains("fn "),
            0.15,
            "Missing function definition",
        )
        .add_check(
            "has_imports",
            |s| s.contains("use "),
            0.15,
            "Missing use/import statements",
        )
        .add_check(
            "has_error_handling",
            |s| s.contains("Result") || s.contains("Option"),
            0.20,
            "No error handling (Result/Option)",
        )
        .add_check(
            "has_documentation",
            |s| s.contains("///") || s.contains("//!"),
            0.15,
            "Missing documentation comments",
        )
        .add_check(
            "no_unwrap",
            |s| !s.contains(".unwrap()"),
            0.20,
            "Uses unsafe .unwrap() - prefer ? operator",
        )
        .add_check(
            "no_panic",
            |s| !s.contains("panic!"),
            0.15,
            "Uses panic! - prefer Result types",
        )
}

fn create_reasoning_critic() -> impl Critic {
    HeuristicCritic::new()
        .min_length(50)
        .require("step")
        .require("therefore")
        .forbid("I don't know")
}

// ============================================================================
// Main Pipeline
// ============================================================================

fn run_code_generation_pipeline() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║     Rust Code Generation with Recursive Refinement            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize components
    println!("Step 1: Initializing components...");
    let mut store = create_vector_store_with_examples();
    let llm = MockLLM::new();
    let critic = create_rust_code_critic();
    let hitl_config = create_hitl_config();

    // Step 2: Define the question
    let question = "How to parse a TOML configuration file in Rust?";
    println!("\nStep 2: Question: \"{}\"", question);

    // Step 3: Retrieve similar examples
    println!("\nStep 3: Retrieving similar examples from vector store...");
    let similar = store.search_text(question, FEW_SHOT_K);
    println!("  Found {} relevant examples:", similar.len());
    for (i, ex) in similar.iter().enumerate() {
        println!("    {}. {} (score: {:.3})", i + 1, ex.id, ex.score);
    }

    // Build few-shot context
    let mut context = String::from("Learn from these examples:\n\n");
    for ex in &similar {
        context.push_str(&format!("---\n{}\n", ex.content));
    }
    context.push_str(&format!("\n---\nNow answer: {}", question));

    // Step 4: Configure refinement
    println!("\nStep 4: Configuring recursive refinement...");
    let config = RecursiveConfig {
        max_iterations: MAX_ITERATIONS,
        score_threshold: SCORE_THRESHOLD,
        ..Default::default()
    };

    let runner = StandaloneRunner::with_config(&critic, "rust", config);

    // Step 5: Run refinement with HITL and diff visualization
    println!("\nStep 5: Running refinement loop...");
    println!("  Max iterations: {}", MAX_ITERATIONS);
    println!("  Score threshold: {:.2}", SCORE_THRESHOLD);
    println!("  HITL: {}", if hitl_config.enabled { "enabled" } else { "disabled" });

    let mut iteration_outputs: Vec<String> = Vec::new();
    let mut prev_score = 0.0;

    let result = runner.refine(&context, |iteration, feedback| {
        // Generate response
        let output = llm.generate(
            "rust_code",
            iteration as usize,
            &context,
            feedback,
        );

        // Show diff from previous iteration
        if iteration > 0 && !iteration_outputs.is_empty() {
            show_iteration_diff(
                iteration_outputs.last().unwrap(),
                &output,
                iteration,
            );
        }

        // Score current output
        let state = RecursiveState::new();
        let output_ref = &output;
        let score_result = critic.evaluate(StrView::new(output_ref), &state);
        let current_score = score_result.score;
        drop(state); // Release borrow early

        // HITL review at intervals or on score drop
        if hitl_config.enabled
            && (iteration as u32 % hitl_config.interval == 0
                || (hitl_config.on_score_drop && current_score < prev_score))
        {
            let decision = simulate_hitl_review(
                iteration as u32,
                &output,
                current_score,
                prev_score,
                score_result.feedback.as_deref(),
            );

            match decision {
                ReviewDecision::Reject => {
                    println!("  [HITL] Rejected - trying alternative approach");
                    // In production, would generate alternative
                }
                ReviewDecision::Stop => {
                    println!("  [HITL] Stopped by user");
                    // Would return early
                }
                _ => {}
            }
        }

        iteration_outputs.push(output.clone());
        prev_score = current_score;

        // Print iteration summary
        println!("\n  Iteration {} Summary:", iteration);
        println!("    Score: {:.2}", current_score);
        if let Some(fb) = &score_result.feedback {
            println!("    Feedback: {}", fb);
        }

        Ok(output)
    })?;

    // Step 6: Show final result
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                        FINAL RESULT                            ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Score: {:.2}                                                    ", result.score);
    println!("║ Iterations: {}                                                  ", result.iterations);
    println!("║ Converged: {}                                                   ", result.score >= SCORE_THRESHOLD);
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Generated Code:                                                 ║");
    println!("╠════════════════════════════════════════════════════════════════╣");

    for line in result.answer.lines() {
        println!("  {}", line);
    }

    println!("╚════════════════════════════════════════════════════════════════╝");

    // Step 7: Store successful result back to vector store
    if result.score >= 0.8 {
        println!("\nStep 6: Storing successful result to vector store...");
        let example_doc = format!(
            "Q: {}\n\nA:\n```rust\n{}\n```\n\n[Score: {:.2}, Iterations: {}]",
            question, result.answer, result.score, result.iterations
        );
        store.add(format!("learned:{}", result.context_id), example_doc);
        println!("  ✓ Added to vector store (total: {} examples)", store.len());
    }

    Ok(())
}

fn run_reasoning_pipeline() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║     Chain-of-Thought Reasoning with Refinement                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let store = create_vector_store_with_examples();
    let llm = MockLLM::new();
    let critic = create_reasoning_critic();

    let question = "What is 6 multiplied by 7?";
    println!("Question: {}", question);

    // Retrieve reasoning examples
    let examples = store.search_text("math calculation percentage", 2);
    println!("\nRetrieved {} reasoning examples", examples.len());

    let config = RecursiveConfig {
        max_iterations: 3,
        score_threshold: 1.0,
        ..Default::default()
    };

    let runner = StandaloneRunner::with_config(&critic, "math", config);

    let mut prev_output = String::new();

    let result = runner.refine(question, |iteration, feedback| {
        let output = llm.generate(
            "math_reasoning",
            iteration as usize,
            question,
            feedback,
        );

        // Show diff
        if iteration > 0 && !prev_output.is_empty() {
            show_iteration_diff(&prev_output, &output, iteration);
        }
        prev_output = output.clone();

        println!("\n  Iteration {} output:", iteration);
        for line in output.lines().take(5) {
            println!("    {}", line);
        }

        Ok(output)
    })?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Final Answer (score: {:.2}, iterations: {}):", result.score, result.iterations);
    println!("{}", result.answer);

    Ok(())
}

fn demonstrate_optimization() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║     DSPy-Style Prompt Optimization Demo                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("The recursive refinement process demonstrates DSPy principles:\n");

    println!("1. **Signatures**: Define input/output contracts");
    println!("   - question -> code (Rust generation)");
    println!("   - question -> reasoning, answer (Chain of Thought)\n");

    println!("2. **Few-Shot Retrieval**: Vector store provides examples");
    println!("   - Semantic search finds relevant past solutions");
    println!("   - High-quality examples bootstrap the LLM\n");

    println!("3. **Critics as Metrics**: Automated quality scoring");
    println!("   - ChecklistCritic: Weighted feature checks");
    println!("   - HeuristicCritic: Pattern matching\n");

    println!("4. **Optimization Loop**: Iterative improvement");
    println!("   - Generate → Score → Feedback → Regenerate");
    println!("   - Converges to high-quality output\n");

    println!("5. **Human-in-the-Loop**: Optional manual review");
    println!("   - Triggered at intervals or score drops");
    println!("   - Accept/Reject/Edit decisions\n");

    println!("6. **Continuous Learning**: Store successful results");
    println!("   - High-scoring outputs added to vector store");
    println!("   - Future queries benefit from learned examples\n");

    println!("This mirrors DSPy's approach of:");
    println!("  - Declarative program definitions (signatures)");
    println!("  - Automatic prompt optimization (critics + refinement)");
    println!("  - Few-shot learning (vector store retrieval)");
    println!("  - Modular composition (ChainOfThought, BestOfN, etc.)");
}

fn main() {
    println!("\n{}", "═".repeat(68));
    println!("          OPTI: Complete DSPy-Style Pipeline Demo");
    println!("{}\n", "═".repeat(68));

    // Run code generation pipeline
    if let Err(e) = run_code_generation_pipeline() {
        eprintln!("Code generation error: {}", e);
    }

    println!("\n{}", "─".repeat(68));

    // Run reasoning pipeline
    if let Err(e) = run_reasoning_pipeline() {
        eprintln!("Reasoning error: {}", e);
    }

    println!("\n{}", "─".repeat(68));

    // Demonstrate optimization concepts
    demonstrate_optimization();

    println!("\n{}", "═".repeat(68));
    println!("                    Demo Complete!");
    println!("{}\n", "═".repeat(68));
}
