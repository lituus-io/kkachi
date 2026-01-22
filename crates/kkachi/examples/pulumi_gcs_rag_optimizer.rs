// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! GCS Bucket RAG Optimizer Example
//!
//! Demonstrates the complete RAG optimization workflow for Pulumi YAML:
//! 1. RAG lookup from DuckDB vector store
//! 2. Recursive refinement with `pulumi preview` validation
//! 3. One-shot prompt formation from successful run
//! 4. Fresh context testing with retry loop
//! 5. RAG write-back on success
//!
//! # Workflow
//!
//! ```text
//! Question: "How do I create a bucket in Pulumi YAML?"
//!      │
//!      ▼
//! ┌─────────────────────────────────────┐
//! │  1. RAG LOOKUP (DuckDB Vector)      │
//! │     Search for similar examples     │
//! └─────────────────────────────────────┘
//!      │
//!      ├─── No match ────► Start with just question
//!      │
//!      └─── Match found ─► Inject as context
//!      │
//!      ▼
//! ┌─────────────────────────────────────┐
//! │  2. RECURSIVE REFINEMENT            │
//! │     - Run `pulumi preview`          │
//! │     - Capture errors/resolutions    │
//! │     - Iterate until success         │
//! └─────────────────────────────────────┘
//!      │
//!      ▼
//! ┌─────────────────────────────────────┐
//! │  3. FORM OPTIMIZED PROMPT           │
//! │     - Template with code section    │
//! │     - Include error corrections     │
//! │     - Create one-shot prompt        │
//! └─────────────────────────────────────┘
//!      │
//!      ▼
//! ┌─────────────────────────────────────┐
//! │  4. TEST IN FRESH CONTEXT           │
//! │     - New LLM session (no history)  │
//! │     - Run one-shot prompt           │
//! │     - Validate output with CLI      │
//! └─────────────────────────────────────┘
//!      │
//!      ├─── Failed ──────► Retry (back to step 2 with failure context)
//!      │
//!      └─── Passed ──────► Write to RAG
//!      │
//!      ▼
//! ┌─────────────────────────────────────┐
//! │  5. UPDATE RAG STORE                │
//! │     - Insert or update document     │
//! │     - Follow template structure     │
//! └─────────────────────────────────────┘
//! ```
//!
//! # GCP Credentials
//!
//! This example demonstrates three authentication methods:
//! - Application Default Credentials (ADC) via `GOOGLE_APPLICATION_CREDENTIALS`
//! - Project ID via `GOOGLE_PROJECT` or `CLOUDSDK_CORE_PROJECT`
//! - Workload Identity Federation (WIF) token
//!
//! # Prerequisites
//!
//! ```bash
//! # Set GCP credentials
//! export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
//! export GOOGLE_PROJECT=my-gcp-project
//!
//! # Or use gcloud ADC
//! gcloud auth application-default login
//!
//! # Run the example
//! cargo run --example pulumi_gcs_rag_optimizer --features storage
//! ```

use kkachi::declarative::{Cli, RagDocumentTemplate, RagOptimizerConfig};

#[cfg(feature = "storage")]
use std::future::Future;

#[cfg(feature = "storage")]
use kkachi::{declarative::RagOptimizer, error::Result};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   GCS Bucket RAG Optimizer - Pulumi YAML");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run synchronous examples (demonstrating API setup)
    example_1_validator_with_gcp_credentials();
    example_2_document_template();
    example_3_optimizer_configuration();
    example_4_two_stage_preview_deploy();
    example_5_dspy3_rag_integration();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   Examples Complete - API Demonstration");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("To run the actual workflow, you need:");
    println!("  1. DuckDB with vector extension");
    println!("  2. A real LLM implementation (e.g., OpenAI, Anthropic)");
    println!("  3. GCP credentials configured");
    println!("  4. Pulumi CLI installed and initialized");
}

/// Example 1: Pulumi YAML Validator with GCP Credentials
fn example_1_validator_with_gcp_credentials() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: Pulumi YAML Validator with GCP Credentials         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Method 1: Application Default Credentials (ADC)
    // Inherits GOOGLE_APPLICATION_CREDENTIALS from environment
    let _adc_validator = Cli::new("pulumi")
        .args(["preview", "--non-interactive", "--stack", "dev"])
        .env_inherit("GOOGLE_APPLICATION_CREDENTIALS") // JSON key path
        .env_inherit("CLOUDSDK_CORE_PROJECT") // Project ID
        .env("PULUMI_CONFIG_PASSPHRASE", "") // Non-interactive mode
        .file_ext("yaml")
        .required();

    println!("  ADC Validator:");
    println!("    Command: pulumi preview --non-interactive --stack dev");
    println!("    Env inherited: GOOGLE_APPLICATION_CREDENTIALS, CLOUDSDK_CORE_PROJECT");
    println!("    Env set: PULUMI_CONFIG_PASSPHRASE=\"\"");
    println!();

    // Method 2: Explicit project configuration
    let _explicit_validator = Cli::new("pulumi")
        .args(["preview", "--non-interactive"])
        .env("GOOGLE_PROJECT", "my-gcp-project")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .file_ext("yaml")
        .required();

    println!("  Explicit Project Validator:");
    println!("    Command: pulumi preview --non-interactive");
    println!("    Env set: GOOGLE_PROJECT=my-gcp-project");
    println!();

    // Method 3: Workload Identity Federation (WIF)
    let _wif_validator = Cli::new("pulumi")
        .args(["preview", "--non-interactive"])
        .env_inherit("GOOGLE_APPLICATION_CREDENTIALS") // Points to WIF config
        .env_inherit("GOOGLE_CLOUD_PROJECT")
        .file_ext("yaml")
        .required();

    println!("  WIF Validator:");
    println!("    Env inherited: GOOGLE_APPLICATION_CREDENTIALS (WIF config)");
    println!();

    // Method 4: Multiple environment variables at once
    let _multi_env_validator = Cli::new("pulumi")
        .args(["preview", "--non-interactive"])
        .envs([
            ("GOOGLE_PROJECT", "my-gcp-project"),
            ("PULUMI_CONFIG_PASSPHRASE", ""),
            ("PULUMI_SKIP_UPDATE_CHECK", "true"),
        ])
        .file_ext("yaml")
        .required();

    println!("  Multi-env Validator:");
    println!("    Uses .envs() for batch configuration");
    println!();
}

/// Example 2: RAG Document Template for Pulumi YAML
fn example_2_document_template() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: RAG Document Template                              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Define document template for Pulumi YAML
    // This template follows GitHub-style markdown with fenced code blocks
    let template = RagDocumentTemplate::new("pulumi_gcs")
        .header("Task", 2)
        .text("question")
        .header("Solution", 2)
        .code("code", "yaml")
        .header("Explanation", 2)
        .text("explanation")
        .header("Common Mistakes", 2)
        .list("errors");

    println!("  Created RagDocumentTemplate: {}", template.name());
    println!("  Sections: {}", template.section_count());
    println!();
    println!("  Output format:");
    println!("    ## Task");
    println!("    <question>");
    println!();
    println!("    ## Solution");
    println!("    ```yaml");
    println!("    <generated code>");
    println!("    ```");
    println!();
    println!("    ## Explanation");
    println!("    <explanation>");
    println!();
    println!("    ## Common Mistakes");
    println!("    - <error 1>");
    println!("    - <error 2>");
    println!();
}

/// Example 3: RagOptimizer Configuration
fn example_3_optimizer_configuration() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 3: RagOptimizer Configuration                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Configure the optimizer
    let config = RagOptimizerConfig {
        rag_threshold: 0.7,           // Min similarity to use RAG context
        max_iterations: 10,           // Inner loop iterations
        max_optimization_attempts: 3, // Outer loop (one-shot retry)
        convergence_threshold: 0.95,  // Score to converge
        oneshot_threshold: 0.9,       // Score for one-shot pass
        write_back: true,             // Update RAG on success
    };

    println!("  RagOptimizerConfig:");
    println!(
        "    rag_threshold: {}           # Min similarity for RAG context",
        config.rag_threshold
    );
    println!(
        "    max_iterations: {}          # Refinement iterations per attempt",
        config.max_iterations
    );
    println!(
        "    max_optimization_attempts: {} # One-shot retry attempts",
        config.max_optimization_attempts
    );
    println!(
        "    convergence_threshold: {}  # Score threshold for success",
        config.convergence_threshold
    );
    println!(
        "    oneshot_threshold: {}      # Score for one-shot test pass",
        config.oneshot_threshold
    );
    println!(
        "    write_back: {}             # Update RAG on success",
        config.write_back
    );
    println!();

    // Fluent configuration
    println!("  Fluent builder pattern:");
    println!("    RagOptimizer::new(&mut store, &embedder)");
    println!("        .stage(\"preview\", preview_validator)");
    println!("        .template(template)");
    println!("        .rag_threshold(0.7)");
    println!("        .oneshot_threshold(0.9)");
    println!("        .max_attempts(3)");
    println!("        .write_back(true)");
    println!("        .run(question, &llm).await");
    println!();
}

/// Example 4: Two-Stage Workflow (Preview + Deploy)
fn example_4_two_stage_preview_deploy() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: Two-Stage Workflow (Preview + Deploy)              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Stage 1: Preview (validate without applying)
    let _preview = Cli::new("pulumi")
        .args(["preview", "--non-interactive"])
        .env_inherit("GOOGLE_APPLICATION_CREDENTIALS")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .file_ext("yaml")
        .required();

    // Stage 2: Deploy (apply changes)
    let _deploy = Cli::new("pulumi")
        .args(["up", "--yes", "--non-interactive"])
        .env_inherit("GOOGLE_APPLICATION_CREDENTIALS")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .file_ext("yaml")
        .required();

    println!("  Two-stage workflow:");
    println!("    Stage 1 (preview): pulumi preview --non-interactive");
    println!("    Stage 2 (deploy):  pulumi up --yes --non-interactive");
    println!();
    println!("  Usage:");
    println!("    RagOptimizer::new(&mut store, &embedder)");
    println!("        .stage(\"preview\", preview)");
    println!("        .stage(\"deploy\", deploy)  // Runs after preview passes");
    println!("        .template(template)");
    println!("        .run(question, &llm).await");
    println!();
    println!("  Stage execution:");
    println!("    1. Refine until preview passes");
    println!("    2. Run deploy on preview-validated code");
    println!("    3. Form one-shot prompt from preview stage");
    println!("    4. Test one-shot in fresh context");
    println!("    5. On success, write to RAG");
    println!();
}

/// Example 5: DSPy3 Strategy + RAG Integration (NEW!)
fn example_5_dspy3_rag_integration() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 5: DSPy3 Strategy + RAG Integration                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("  The declarative API now supports combining DSPy3 strategies with");
    println!("  RAG optimization via the .optimize_with_rag() method:\n");

    // Chain of Thought + RAG
    println!("  Chain of Thought + RAG Optimization:");
    println!("    pipeline(\"question -> pulumi_yaml\")");
    println!("        .chain_of_thought()              // DSPy3: CoT reasoning");
    println!("        .validate(preview_validator)");
    println!("        .strict()                        // 10 iter, 0.95 threshold");
    println!("        .optimize_with_rag(&mut store, &embedder)");
    println!("        .template(template)");
    println!("        .oneshot_threshold(0.9)");
    println!("        .run(question, &llm).await");
    println!();

    // Best of N + RAG
    println!("  Best of N + RAG Optimization:");
    println!("    pipeline(\"question -> pulumi_yaml\")");
    println!("        .best_of::<5>()                  // DSPy3: 5 candidates");
    println!("        .validate(preview_validator)");
    println!("        .optimize_with_rag(&mut store, &embedder)");
    println!("        .max_attempts(3)                 // Retry loop attempts");
    println!("        .run(question, &llm).await");
    println!();

    // Multi-Chain + RAG
    println!("  Multi-Chain + RAG Optimization:");
    println!("    pipeline(\"question -> pulumi_yaml\")");
    println!("        .multi_chain::<3>()              // DSPy3: 3 reasoning chains");
    println!("        .validate(preview_validator)");
    println!("        .optimize_with_rag(&mut store, &embedder)");
    println!("        .write_back(true)                // Update RAG on success");
    println!("        .run(question, &llm).await");
    println!();

    println!("  Benefits of DSPy3 + RAG Integration:");
    println!("    - CoT reasoning improves code generation quality");
    println!("    - BestOfN selects best candidate from multiple generations");
    println!("    - MultiChain uses consensus from multiple reasoning paths");
    println!("    - RAG provides relevant examples as context");
    println!("    - One-shot testing validates reproducibility");
    println!("    - Successful patterns are stored for future use");
    println!();
}

/// Mock LLM for demonstration (implement `kkachi::declarative::LLM` trait)
///
/// In production, implement this trait with your actual LLM provider.
#[cfg(feature = "storage")]
struct MockLLM;

/// Implementation of the LLM trait for demonstration.
///
/// The key method is `generate_fresh` which creates a new session
/// without conversation history for one-shot testing.
#[cfg(feature = "storage")]
impl MockLLM {
    fn new() -> Self {
        Self
    }

    /// Generate with conversation context
    fn generate(&self, _prompt: &str) -> impl Future<Output = Result<String>> + Send {
        async {
            // In real implementation, call your LLM API
            Ok(r#"name: gcs-bucket-project
runtime: yaml

resources:
  my-bucket:
    type: gcp:storage:Bucket
    properties:
      name: my-unique-bucket-name-12345
      location: US
      uniformBucketLevelAccess: true
      forceDestroy: true
      labels:
        environment: dev
        managed-by: pulumi
"#
            .to_string())
        }
    }

    /// Generate in fresh context (no history) - for one-shot testing
    fn generate_fresh(&self, prompt: &str) -> impl Future<Output = Result<String>> + Send {
        // Fresh context means starting a new conversation/session
        // This is crucial for one-shot testing to verify the prompt
        // works without conversation history
        self.generate(prompt)
    }
}

/// Full workflow example (async, requires storage feature and real dependencies)
///
/// This is the complete workflow as described in the module documentation.
#[cfg(feature = "storage")]
#[allow(dead_code)]
async fn full_workflow_example() -> anyhow::Result<()> {
    use kkachi::recursive::{DuckDBVectorStore, HashEmbedder, MutableVectorStore};

    // 1. Setup vector store and embedder
    let embedder = HashEmbedder::new(64);
    let mut store = DuckDBVectorStore::open("./gcs_rag.db")?;

    // 2. Define Pulumi YAML validator with GCP credentials
    let preview_validator = Cli::new("pulumi")
        .args(["preview", "--non-interactive", "--stack", "dev"])
        .env_inherit("GOOGLE_APPLICATION_CREDENTIALS")
        .env_inherit("GOOGLE_PROJECT")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .file_ext("yaml")
        .required();

    // 3. Define document template
    let template = RagDocumentTemplate::new("pulumi_gcs")
        .header("Task", 2)
        .text("question")
        .header("Solution", 2)
        .code("code", "yaml")
        .header("Common Mistakes", 2)
        .list("errors");

    // 4. Configure optimizer
    let config = RagOptimizerConfig {
        rag_threshold: 0.7,
        max_iterations: 10,
        max_optimization_attempts: 3,
        convergence_threshold: 0.95,
        oneshot_threshold: 0.9,
        write_back: true,
    };

    // 5. Create LLM (user implements the trait)
    let llm = MockLLM::new();

    // 6. Run optimization
    let result = RagOptimizer::new(&mut store, &embedder)
        .stage("preview", preview_validator)
        .template(template)
        .config(config)
        .run("How do I create a GCS bucket in Pulumi YAML?", &llm)
        .await?;

    // 7. Print results
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   Optimization Result");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Answer:\n{}", result.answer);
    println!("\nScore: {:.2}", result.score);
    println!("Iterations: {}", result.iterations);
    println!("Optimization attempts: {}", result.optimization_attempts);
    println!("One-shot passed: {}", result.oneshot_passed);
    println!("RAG updated: {}", result.rag_updated);

    if !result.error_corrections.is_empty() {
        println!("\nError corrections learned:");
        for (error, fix, iter) in &result.error_corrections {
            println!("  Iteration {}: {} -> {}", iter, error, fix);
        }
    }

    Ok(())
}
